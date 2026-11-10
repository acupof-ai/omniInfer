// SPDX-License-Identifier: Apache-2.0
#include "omniinfer/serve/openai_api.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <ctime>
#include <deque>
#include <limits>
#include <map>
#include <mutex>
#include <thread>

namespace omniinfer {
namespace {

using json = nlohmann::json;

Value json_to_value(const json& j) {
  if (j.is_null()) return Value();
  if (j.is_boolean()) return Value(j.get<bool>());
  if (j.is_number_unsigned()) {
    const uint64_t value = j.get<uint64_t>();
    if (value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      throw ProtocolError("unsigned integer exceeds signed int64");
    }
    return Value(value);
  }
  if (j.is_number_integer()) return Value(j.get<int64_t>());
  if (j.is_number()) return Value(j.get<double>());
  if (j.is_string()) return Value(j.get<std::string>());
  if (j.is_array()) {
    Value::Array out;
    out.reserve(j.size());
    for (const auto& item : j) out.push_back(json_to_value(item));
    return Value(std::move(out));
  }
  Value::Map out;
  for (auto it = j.begin(); it != j.end(); ++it) out[it.key()] = json_to_value(it.value());
  return Value(std::move(out));
}

json value_to_json(const Value& v) {
  switch (v.type()) {
    case Value::Type::Nil:
      return nullptr;
    case Value::Type::Bool:
      return v.as_bool();
    case Value::Type::Int:
      return v.as_int();
    case Value::Type::Float:
      return v.as_double();
    case Value::Type::Str:
      return v.as_str();
    case Value::Type::Bin:
      return v.as_bin();
    case Value::Type::Array: {
      json out = json::array();
      for (const auto& item : v.as_array()) out.push_back(value_to_json(item));
      return out;
    }
    case Value::Type::Map: {
      json out = json::object();
      for (const auto& [key, item] : v.as_map()) out[key] = value_to_json(item);
      return out;
    }
    case Value::Type::Tensor_:
      return {{"dtype", v.as_tensor().dtype},
              {"shape", v.as_tensor().shape},
              {"nbytes", v.as_tensor().bytes.size()}};
  }
  return nullptr;
}

// The reply text a stage returns. A stage may return a bare string or a map
// with a "text" or "content" field.
std::string result_text(const Value& result) {
  if (result.is_str()) return result.as_str();
  if (const Value* text = result.find("text"); text != nullptr && text->is_str()) {
    return text->as_str();
  }
  if (const Value* text = result.find("content"); text != nullptr && text->is_str()) {
    return text->as_str();
  }
  return value_to_json(result).dump();
}

struct SseSink {
  std::mutex mu;
  std::condition_variable cv;
  std::deque<Value> chunks;
  bool done = false;
  bool has_deltas = false;
  std::string error;

  void push(Value v, bool delta = false) {
    {
      std::lock_guard<std::mutex> lock(mu);
      has_deltas = has_deltas || delta;
      chunks.push_back(std::move(v));
    }
    cv.notify_all();
  }
  void finish(std::string error_message = "") {
    {
      std::lock_guard<std::mutex> lock(mu);
      error = std::move(error_message);
      done = true;
    }
    cv.notify_all();
  }
  bool pop(Value& out, double timeout_s) {
    std::unique_lock<std::mutex> lock(mu);
    if (!cv.wait_for(lock, std::chrono::duration<double>(timeout_s),
                     [this] { return !chunks.empty() || done; })) {
      return false;
    }
    if (chunks.empty()) return false;
    out = std::move(chunks.front());
    chunks.pop_front();
    return true;
  }
};

}  // namespace

class OpenAIApi::Impl {
 public:
  Impl(Coordinator& coordinator, ApiConfig config)
      : coordinator_(coordinator), config_(std::move(config)) {
    if (config_.served_model_name.empty()) config_.served_model_name = config_.model;
  }

  void install_routes() {
    server_.set_pre_routing_handler(
        [this](const httplib::Request& req, httplib::Response& res) {
          if (config_.api_key.empty()) return httplib::Server::HandlerResponse::Unhandled;
          const std::string expected = "Bearer " + config_.api_key;
          const auto auth = req.get_header_value("Authorization");
          if (auth == expected) return httplib::Server::HandlerResponse::Unhandled;
          res.status = 401;
          res.set_content(R"({"error":{"message":"invalid api key","type":"auth_error"}})",
                          "application/json");
          return httplib::Server::HandlerResponse::Handled;
        });

    server_.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
      json body = {{"status", "ok"}, {"in_flight", coordinator_.in_flight()}};
      res.set_content(body.dump(), "application/json");
    });

    server_.Get("/ready", [this](const httplib::Request&, httplib::Response& res) {
      res.set_content(R"({"status":"ready"})", "application/json");
    });

    server_.Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) {
      json body = {{"object", "list"},
                   {"data", {{{"id", config_.served_model_name},
                              {"object", "model"},
                              {"owned_by", "omniInfer"}}}}};
      res.set_content(body.dump(), "application/json");
    });

    server_.Post("/v1/chat/completions",
                 [this](const httplib::Request& req, httplib::Response& res) {
                   handle_chat(req, res);
                 });

    server_.Post("/abort", [this](const httplib::Request& req, httplib::Response& res) {
      const json body = json::parse(req.body, nullptr, false);
      if (body.is_discarded() || !body.is_object() ||
          !body.contains("request_id") || !body["request_id"].is_string() ||
          body["request_id"].get_ref<const std::string&>().empty()) {
        res.status = 400;
        res.set_content(R"({"error":{"message":"request_id is required"}})",
                        "application/json");
        return;
      }
      coordinator_.abort(body["request_id"].get<std::string>());
      res.set_content(R"({"success":true})", "application/json");
    });

    server_.Post("/v1/audio/speech",
                 [](const httplib::Request&, httplib::Response& res) {
                   // Speech requires a TTS stage, which this build lacks.
                   res.status = 501;
                   res.set_content(
                       R"({"error":{"message":"speech routes are served by the TTS stage"}})",
                       "application/json");
                 });
  }

  bool bind() {
    if (bound_) return true;
    // Port 0 asks the OS for a free port, which tests use.
    if (config_.port == 0) {
      bound_port_ = server_.bind_to_any_port(config_.host);
      if (bound_port_ < 0) return false;
    } else {
      if (!server_.bind_to_port(config_.host, config_.port)) return false;
      bound_port_ = config_.port;
    }
    bound_ = true;
    return true;
  }
  bool listen() {
    if (!bound_ && !bind()) return false;
    return server_.listen_after_bind();
  }
  void stop() { server_.stop(); }

  int bound_port() const { return bound_port_; }

 private:
  void handle_chat(const httplib::Request& req, httplib::Response& res) {
    const json body = json::parse(req.body, nullptr, false);
    auto reject = [&res](const std::string& message) {
      res.status = 400;
      res.set_content(json{{"error", {{"message", message}, {"type", "invalid_request"}}}}
                          .dump(),
                      "application/json");
    };
    if (body.is_discarded() || !body.is_object() || !body.contains("messages") ||
        !body["messages"].is_array() || body["messages"].empty()) {
      reject("messages must be a non-empty array");
      return;
    }
    for (const auto& message : body["messages"]) {
      if (!message.is_object() || !message.contains("role") ||
          !message["role"].is_string() || !message.contains("content")) {
        reject("each message requires a string role and content");
        return;
      }
    }
    if (body.contains("stream") && !body["stream"].is_boolean()) {
      reject("stream must be a boolean");
      return;
    }
    if (body.contains("temperature") && !body["temperature"].is_null()) {
      if (!body["temperature"].is_number()) {
        reject("temperature must be a number");
        return;
      }
      const double temperature = body["temperature"].get<double>();
      if (temperature < 0 || temperature > 2) {
        reject("temperature must be between 0 and 2");
        return;
      }
    }
    const json* token_limit = nullptr;
    for (const char* key : {"max_tokens", "max_completion_tokens"}) {
      if (!body.contains(key) || body[key].is_null()) continue;
      const json& value = body[key];
      if (!value.is_number_integer() ||
          (value.is_number_unsigned() &&
           value.get<uint64_t>() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) ||
          value.get<int64_t>() < 1) {
        reject(std::string(key) + " must be a positive integer");
        return;
      }
      token_limit = &value;
    }
    const bool stream = body.value("stream", false);

    StagePayload payload;
    Value::Map inputs;
    try {
      inputs["messages"] = json_to_value(body["messages"]);
    } catch (const std::exception& exc) {
      reject(exc.what());
      return;
    }
    payload.request.inputs = Value(std::move(inputs));
    Value::Map params;
    if (token_limit) params["max_new_tokens"] = Value(token_limit->get<int64_t>());
    if (body.contains("temperature") && !body["temperature"].is_null()) {
      params["temperature"] = Value(body["temperature"].get<double>());
    }
    params["stream"] = Value(stream);
    payload.request.params = std::move(params);

    if (!stream) {
      std::string request_id;
      try {
        request_id = coordinator_.submit(std::move(payload), config_.entry_stage);
      } catch (const std::exception& exc) {
        res.status = 503;
        res.set_content(
            json{{"error", {{"message", exc.what()}, {"type", "pipeline_error"}}}}
                .dump(),
            "application/json");
        return;
      }
      const bool settled = coordinator_.wait(request_id, config_.request_timeout_s);
      auto info = coordinator_.info(request_id);
      if (!settled || !info) {
        coordinator_.abort(request_id);
        coordinator_.release(request_id);
        res.status = 504;
        res.set_content(json{{"error", {{"message", "request timed out"}}}}.dump(),
                        "application/json");
        return;
      }
      coordinator_.release(request_id);
      if (info->state != "completed") {
        res.status = 500;
        res.set_content(
            json{{"error", {{"message", info->error}, {"type", "generation_error"}}}}
                .dump(),
            "application/json");
        return;
      }
      json out = {{"id", request_id},
                  {"object", "chat.completion"},
                  {"created", std::time(nullptr)},
                  {"model", config_.served_model_name},
                  {"choices",
                   {{{"index", 0},
                     {"message", {{"role", "assistant"},
                                  {"content", result_text(info->result)}}},
                     {"finish_reason", "stop"}}}}};
      res.set_content(out.dump(), "application/json");
      return;
    }

    // Claim the id before submitting. A stage can complete the request before
    // submit returns, and on_complete must find the sink by then.
    const std::string request_id = "http-" + std::to_string(++next_request_id_);
    payload.request_id = request_id;
    auto sink = std::make_shared<SseSink>();
    {
      std::lock_guard<std::mutex> lock(streams_mu_);
      streams_[request_id] = sink;
    }
    try {
      coordinator_.submit(std::move(payload), config_.entry_stage);
    } catch (const std::exception& exc) {
      {
        std::lock_guard<std::mutex> lock(streams_mu_);
        streams_.erase(request_id);
      }
      res.status = 503;
      res.set_content(
          json{{"error", {{"message", exc.what()}, {"type", "pipeline_error"}}}}
              .dump(),
          "application/json");
      return;
    }
    // httplib may destroy a response without invoking its content provider.
    // The last provider copy therefore owns request cleanup as well.
    auto cleanup = std::shared_ptr<void>(nullptr, [this, request_id](void*) {
      coordinator_.abort(request_id);
      {
        std::lock_guard<std::mutex> lock(streams_mu_);
        streams_.erase(request_id);
      }
      coordinator_.release(request_id);
    });
    const std::string model = config_.served_model_name;
    res.set_chunked_content_provider(
        "text/event-stream",
        [this, sink, request_id, model, cleanup](size_t, httplib::DataSink& out_sink) {
          auto send_chunk = [&](const json& payload) {
            const std::string data = "data: " + payload.dump() + "\n\n";
            return out_sink.write(data.data(), data.size());
          };
          if (!send_chunk({{"id", request_id},
                           {"object", "chat.completion.chunk"},
                           {"model", model},
                           {"choices", {{{"index", 0},
                                        {"delta", {{"role", "assistant"}}},
                                        {"finish_reason", nullptr}}}}})) {
            coordinator_.abort(request_id);
            return false;
          }
          std::string stream_error;
          bool settled = false;
          bool disconnected = false;
          const auto timeout = std::chrono::duration<double>(config_.request_timeout_s);
          auto deadline = std::chrono::steady_clock::now() + timeout;
          while (true) {
            if (!out_sink.is_writable()) {
              disconnected = true;
              break;
            }
            const double remaining =
                std::chrono::duration<double>(deadline - std::chrono::steady_clock::now())
                    .count();
            if (remaining <= 0) {
              stream_error = "stream timed out";
              break;
            }
            Value chunk;
            if (sink->pop(chunk, std::min(0.2, remaining))) {
              if (!send_chunk({{"id", request_id},
                               {"object", "chat.completion.chunk"},
                               {"model", model},
                               {"choices", {{{"index", 0},
                                             {"delta", {{"content", result_text(chunk)}}},
                                             {"finish_reason", nullptr}}}}})) {
                disconnected = true;
                break;
              }
              deadline = std::chrono::steady_clock::now() + timeout;
              continue;
            }
            {
              std::lock_guard<std::mutex> lock(sink->mu);
              settled = sink->done;
              if (settled) stream_error = sink->error;
            }
            if (settled) break;
          }
          if (!settled) coordinator_.abort(request_id);
          if (!disconnected) {
            if (stream_error.empty()) {
              send_chunk({{"id", request_id},
                          {"object", "chat.completion.chunk"},
                          {"model", model},
                          {"choices", {{{"index", 0},
                                        {"delta", json::object()},
                                        {"finish_reason", "stop"}}}}});
            } else {
              send_chunk({{"error", {{"message", stream_error},
                                      {"type", "generation_error"},
                                      {"code", 500}}}});
            }
            static const std::string kDone = "data: [DONE]\n\n";
            out_sink.write(kDone.data(), kDone.size());
            out_sink.done();
          }
          return !disconnected;
        });
  }

 public:
  // The coordinator calls this for each stream chunk of a request.
  void on_stream(const StreamMessage& msg) {
    std::shared_ptr<SseSink> sink;
    {
      std::lock_guard<std::mutex> lock(streams_mu_);
      auto it = streams_.find(msg.request_id);
      if (it != streams_.end()) sink = it->second;
    }
    if (sink) sink->push(msg.chunk, true);
  }

  void on_complete(const RequestInfo& info) {
    std::shared_ptr<SseSink> sink;
    {
      std::lock_guard<std::mutex> lock(streams_mu_);
      auto it = streams_.find(info.request_id);
      if (it != streams_.end()) {
        sink = it->second;
        streams_.erase(it);
      }
    }
    if (!sink) return;
    if (info.state == "completed") {
      // Incremental deltas already contain the text. The aggregate result is
      // needed only when the stage sent no deltas.
      if (!sink->has_deltas) sink->push(info.result);
      sink->finish();
      return;
    }
    sink->finish(info.error.empty() ? "request " + info.state : info.error);
  }

 private:
  Coordinator& coordinator_;
  ApiConfig config_;
  httplib::Server server_;
  int bound_port_ = 0;
  bool bound_ = false;
  std::mutex streams_mu_;
  std::map<std::string, std::shared_ptr<SseSink>> streams_;
  std::atomic<uint64_t> next_request_id_{0};
};

OpenAIApi::OpenAIApi(Coordinator& coordinator, ApiConfig config)
    : impl_(std::make_unique<Impl>(coordinator, std::move(config))) {
  impl_->install_routes();
  coordinator.on_stream = [this](const StreamMessage& msg) { impl_->on_stream(msg); };
  coordinator.on_complete = [this](const RequestInfo& info) { impl_->on_complete(info); };
}

OpenAIApi::~OpenAIApi() = default;

bool OpenAIApi::bind() { return impl_->bind(); }
bool OpenAIApi::listen() { return impl_->listen(); }
void OpenAIApi::stop() { impl_->stop(); }
int OpenAIApi::bound_port() const { return impl_->bound_port(); }

}  // namespace omniinfer

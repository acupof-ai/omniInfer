// SPDX-License-Identifier: Apache-2.0
#include "umminfer/serve/openai_api.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "umminfer/client/client.hpp"
#include "asr_common.hpp"
#include "client/http.hpp"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <ctime>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

namespace umminfer {
namespace {

using json = nlohmann::json;

// Multipart values are raw bytes, not JSON strings. nlohmann's dump() throws on
// invalid UTF-8, which would turn an intended 400 into a 500. Replace each
// invalid byte so the error message stays serializable.
std::string utf8_sanitized(const std::string& text) {
  std::string out;
  out.reserve(text.size());
  size_t index = 0;
  while (index < text.size()) {
    const unsigned char lead = static_cast<unsigned char>(text[index]);
    size_t length = 0;
    // The lead ranges exclude overlong forms (C0, C1) and values above U+10FFFF.
    if (lead < 0x80) length = 1;
    else if (lead >= 0xC2 && lead <= 0xDF) length = 2;
    else if (lead >= 0xE0 && lead <= 0xEF) length = 3;
    else if (lead >= 0xF0 && lead <= 0xF4) length = 4;
    bool valid = length > 0 && index + length <= text.size();
    for (size_t offset = 1; valid && offset < length; ++offset) {
      valid = (static_cast<unsigned char>(text[index + offset]) & 0xC0) == 0x80;
    }
    // Continuation-byte shape alone still admits overlong forms (E0 80 80),
    // surrogates (ED A0 80) and values above U+10FFFF (F4 90 80 80). The second
    // byte range is restricted per lead byte to exclude them.
    if (valid && length > 1) {
      const unsigned char second = static_cast<unsigned char>(text[index + 1]);
      if (lead == 0xE0) valid = second >= 0xA0;
      else if (lead == 0xED) valid = second <= 0x9F;
      else if (lead == 0xF0) valid = second >= 0x90;
      else if (lead == 0xF4) valid = second <= 0x8F;
    }
    if (!valid) {
      out.push_back('?');
      ++index;
      continue;
    }
    out.append(text, index, length);
    index += length;
  }
  return out;
}

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

// The finish reason the stage reported. Empty when the stage reported none;
// callers must not substitute "stop".
std::string result_finish_reason(const Value& result) {
  const Value* reason = result.find("finish_reason");
  if (reason == nullptr || !reason->is_str()) return {};
  return reason->as_str();
}

struct SseSink {
  std::mutex mu;
  std::condition_variable cv;
  std::deque<Value> chunks;
  bool done = false;
  bool has_deltas = false;
  std::string error;
  std::string finish_reason;

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

// ---- Fixed-upstream proxy -------------------------------------------------

constexpr size_t kRelayQueueBytes = 4u << 20;
constexpr size_t kMaxJsonUpstreamBytes = 64u << 20;
constexpr size_t kMaxBinaryUpstreamBytes = 512u << 20;

// Path segments are forwarded, so only an opaque identifier is accepted. This
// blocks traversal, query injection and absolute-URL smuggling in one place.
bool valid_path_id(std::string_view id) {
  if (id.empty() || id.size() > 128) return false;
  for (const char c : id) {
    const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                    (c >= '0' && c <= '9') || c == '_' || c == '-';
    if (!ok) return false;
  }
  return true;
}

// One upstream transfer whose response is consumed incrementally. The handler
// waits for the first byte before it commits the client response, so an
// upstream error still maps to its own status code instead of a committed 200.
class UpstreamRelay {
 public:
  using HttpRequest = client::detail::HttpRequest;

  UpstreamRelay() = default;
  UpstreamRelay(const UpstreamRelay&) = delete;
  UpstreamRelay& operator=(const UpstreamRelay&) = delete;

  ~UpstreamRelay() {
    cancel();
    if (worker_.joinable()) worker_.join();
  }

  void start(HttpRequest request, std::function<void()> on_transfer_done = {}) {
    worker_ = std::thread([this, request = std::move(request),
                           done_hook = std::move(on_transfer_done)]() mutable {
      request.cancel = &cancel_flag_;
      try {
        client::detail::perform(request);
      } catch (const client::Error& error) {
        std::lock_guard<std::mutex> lock(mu_);
        error_ = error.what();
        error_code_ = error.code();
        status_ = error.status();
        retry_after_ = error.retry_after();
      } catch (const std::exception& error) {
        std::lock_guard<std::mutex> lock(mu_);
        error_ = error.what();
      }
      // The hook may push a trailing event, so it runs without the lock. A
      // throw there must not escape the thread; it becomes a transfer error.
      try {
        if (done_hook) done_hook();
      } catch (const std::exception& error) {
        std::lock_guard<std::mutex> lock(mu_);
        if (error_.empty()) error_ = error.what();
      } catch (...) {
        std::lock_guard<std::mutex> lock(mu_);
        if (error_.empty()) error_ = "upstream stream finalization failed";
      }
      {
        std::lock_guard<std::mutex> lock(mu_);
        done_ = true;
      }
      cv_.notify_all();
    });
  }

  // Called on the transfer thread. Returns false to abort the transfer. The
  // bounded queue applies backpressure instead of buffering the whole body.
  bool push(std::string bytes) {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock, [this] {
      return cancel_flag_.load() || queued_bytes_ < kRelayQueueBytes;
    });
    if (cancel_flag_.load()) return false;
    started_ = true;
    queued_bytes_ += bytes.size();
    queue_.push_back(std::move(bytes));
    cv_.notify_all();
    return true;
  }

  void cancel() {
    cancel_flag_.store(true);
    cv_.notify_all();
  }

  bool pop_wait(std::string& out, std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait_for(lock, timeout, [this] { return !queue_.empty() || done_; });
    if (queue_.empty()) return false;
    out = std::move(queue_.front());
    queue_.pop_front();
    queued_bytes_ -= out.size();
    cv_.notify_all();
    return true;
  }

  bool wait_started(double seconds) {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait_for(lock, std::chrono::duration<double>(seconds),
                 [this] { return started_ || done_; });
    return started_;
  }

  bool started() {
    std::lock_guard<std::mutex> lock(mu_);
    return started_;
  }
  // True when the transfer ended before the first body byte: the handler must
  // map the stored error (or an empty response) instead of waiting for the
  // request timeout.
  bool settled_without_start() {
    std::lock_guard<std::mutex> lock(mu_);
    return done_ && !started_;
  }
  bool finished() {
    std::lock_guard<std::mutex> lock(mu_);
    return done_ && queue_.empty();
  }
  bool has_error() {
    std::lock_guard<std::mutex> lock(mu_);
    return !error_.empty();
  }
  std::string error_text() {
    std::lock_guard<std::mutex> lock(mu_);
    return error_;
  }
  client::ErrorCode error_code() {
    std::lock_guard<std::mutex> lock(mu_);
    return error_code_;
  }
  client::Milliseconds retry_after() {
    std::lock_guard<std::mutex> lock(mu_);
    return retry_after_;
  }
  long upstream_status() {
    std::lock_guard<std::mutex> lock(mu_);
    return status_;
  }

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::string> queue_;
  size_t queued_bytes_ = 0;
  bool started_ = false;
  bool done_ = false;
  std::string error_;
  client::ErrorCode error_code_ = client::ErrorCode::kBadResponse;
  long status_ = 0;
  client::Milliseconds retry_after_{0};
  std::atomic<bool> cancel_flag_{false};
  std::thread worker_;
};

// Re-emits upstream SSE data payloads without rewriting them. Each payload line
// becomes one data line, which preserves multi-line events and the [DONE]
// terminator exactly as the upstream sent them. No event is invented.
class SseFramer {
 public:
  // Ownership invariant: the worker thread and this framer hold nothing that
  // can outlive the relay's external owners. The framer stores a raw pointer,
  // so a worker callback can never drop the last reference (which would run
  // ~UpstreamRelay on the worker thread and join it with itself). The relay
  // outlives every worker callback because its destructor runs only from the
  // handler or the response provider, and it cancels and joins the worker
  // before any member is destroyed.
  explicit SseFramer(UpstreamRelay* relay) : relay_(relay) {}

  bool feed(std::string_view bytes) {
    return parser_.feed(bytes, [this](std::string_view data) { return emit(data); });
  }
  void finish() {
    parser_.finish([this](std::string_view data) { return emit(data); });
  }

  // True once the upstream sent its [DONE] terminator. A stream that ends
  // without it is truncated and must not be reported as complete.
  bool saw_done() const { return saw_done_.load(); }

 private:
  bool emit(std::string_view data) {
    if (data == "[DONE]") saw_done_.store(true);
    std::string framed;
    size_t start = 0;
    while (true) {
      const size_t newline = data.find('\n', start);
      const std::string_view line =
          newline == std::string_view::npos ? data.substr(start)
                                            : data.substr(start, newline - start);
      framed += "data: ";
      framed.append(line);
      framed += "\n";
      if (newline == std::string_view::npos) break;
      start = newline + 1;
    }
    framed += "\n";
    return relay_->push(std::move(framed));
  }

  client::detail::SSEParser parser_;
  UpstreamRelay* relay_;
  std::atomic<bool> saw_done_{false};
};

int upstream_error_status(client::ErrorCode code, long status) {
  if (status >= 400 && status <= 599) return static_cast<int>(status);
  if (code == client::ErrorCode::kTimeout) return 504;
  if (code == client::ErrorCode::kAuthentication) return 502;
  return 502;
}

}  // namespace

std::string validate_upstream_config(const ApiConfig& config) {
  const bool any = !config.upstream_kind.empty() || !config.upstream_base_url.empty() ||
                   !config.upstream_model.empty() || !config.upstream_api_key.empty() ||
                   !config.upstream_variant.empty();
  if (!any) return {};
  if (config.upstream_kind.empty()) return "upstream_kind is required with the other upstream fields";
  if (config.upstream_base_url.empty()) return "upstream_base_url is required";
  if (config.upstream_model.empty()) return "upstream_model is required";
  if (config.upstream_variant.empty()) return "upstream_variant is required";
  if (config.upstream_kind != "qwen" && config.upstream_kind != "sglang") {
    return "upstream_kind must be qwen or sglang";
  }
  if (config.upstream_kind == "qwen") {
    if (config.upstream_variant != "omni") {
      return "upstream_variant must be omni for a qwen upstream";
    }
    if (config.upstream_api_key.empty()) {
      return "upstream_api_key is required for a qwen upstream";
    }
  } else if (config.upstream_variant != "image" && config.upstream_variant != "video") {
    return "upstream_variant must be image or video for an sglang upstream";
  }
  // The two keys are separate credentials. Reusing one value would send the
  // inbound credential to the upstream and blur the boundary the proxy keeps.
  if (!config.upstream_api_key.empty() && config.upstream_api_key == config.api_key) {
    return "upstream_api_key must differ from api_key";
  }
  try {
    client::detail::validate_base_url(config.upstream_base_url, false);
  } catch (const std::exception& error) {
    return std::string("invalid upstream_base_url: ") + error.what();
  }
  return {};
}

class OpenAIApi::Impl {
 public:
  Impl(Coordinator& coordinator, ApiConfig config)
      : coordinator_(coordinator), config_(std::move(config)) {
    if (config_.served_model_name.empty()) config_.served_model_name = config_.model;
    upstream_base_url_ = config_.upstream_base_url;
    while (upstream_base_url_.size() > 8 && upstream_base_url_.back() == '/') {
      upstream_base_url_.pop_back();
    }
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

    // Admission only: 503 until the configured external entry stage has sent
    // stage_ready. This is not a heartbeat and not a liveness probe.
    server_.Get("/ready", [this](const httplib::Request&, httplib::Response& res) {
      const bool ready = !config_.external_entry_stage ||
                         coordinator_.stage_ready(config_.entry_stage);
      res.status = ready ? 200 : 503;
      res.set_content(ready ? R"({"status":"ready"})" : R"({"status":"starting"})",
                      "application/json");
    });

    server_.Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) {
      json body = {{"object", "list"},
                   {"data", {{{"id", config_.served_model_name},
                              {"object", "model"},
                              // The server has no model creation timestamp.
                              {"created", 0},
                              {"owned_by", "ummInfer"}}}}};
      res.set_content(body.dump(), "application/json");
    });

    // Each route answers only from the one configured upstream. A route outside
    // that upstream's variant fails explicitly instead of reaching the
    // coordinator or the local stage path.
    server_.Post("/v1/chat/completions",
                 [this](const httplib::Request& req, httplib::Response& res) {
                   if (config_.upstream_kind == "qwen") {
                     proxy_qwen_chat(req, res);
                     return;
                   }
                   if (!config_.upstream_kind.empty()) {
                     reject_route(res, "chat completions");
                     return;
                   }
                   handle_chat(req, res);
                 });

    server_.Post("/v1/audio/transcriptions",
                 [this](const httplib::Request& req, httplib::Response& res) {
                   if (!config_.upstream_kind.empty()) {
                     reject_route(res, "audio transcriptions");
                     return;
                   }
                   handle_transcription(req, res);
                 });

    server_.Post("/v1/images/generations",
                 [this](const httplib::Request& req, httplib::Response& res) {
                   if (config_.upstream_variant != "image") {
                     reject_route(res, "image generation");
                     return;
                   }
                   handle_image_generation(req, res);
                 });

    server_.Post("/v1/images/edits",
                 [this](const httplib::Request& req, httplib::Response& res) {
                   if (config_.upstream_variant != "image") {
                     reject_route(res, "image editing");
                     return;
                   }
                   handle_image_edit(req, res);
                 });

    server_.Post("/v1/videos",
                 [this](const httplib::Request& req, httplib::Response& res) {
                   if (config_.upstream_variant != "video") {
                     reject_route(res, "video creation");
                     return;
                   }
                   handle_video_create(req, res);
                 });

    // The id segment is matched by a restricted character class, so no request
    // path can carry a segment separator or a query.
    server_.Get(R"(/v1/videos/([A-Za-z0-9_\-]{1,128})/content)",
                [this](const httplib::Request& req, httplib::Response& res) {
                  if (config_.upstream_variant != "video") {
                    reject_route(res, "video content");
                    return;
                  }
                  handle_video_content(req, req.matches[1].str(), res);
                });

    server_.Get(R"(/v1/videos/([A-Za-z0-9_\-]{1,128}))",
                [this](const httplib::Request& req, httplib::Response& res) {
                  if (config_.upstream_variant != "video") {
                    reject_route(res, "video status");
                    return;
                  }
                  handle_video_status(req, req.matches[1].str(), res);
                });

    server_.Delete(R"(/v1/videos/([A-Za-z0-9_\-]{1,128}))",
                   [this](const httplib::Request& req, httplib::Response& res) {
                     if (config_.upstream_variant != "video") {
                       reject_route(res, "video deletion");
                       return;
                     }
                     handle_video_delete(req, req.matches[1].str(), res);
                   });

    server_.Post("/abort", [this](const httplib::Request& req, httplib::Response& res) {
      // In fixed-upstream mode the generation is not a coordinator job, so an
      // abort cannot reach it. Answering success here would be a lie.
      if (!config_.upstream_kind.empty()) {
        reject(res, 501,
               "abort is not supported: requests are served by a fixed upstream",
               "unsupported_route");
        return;
      }
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
                   // This HTTP surface has no speech implementation.
                   res.status = 501;
                   res.set_content(
                       R"({"error":{"message":"speech is not implemented by this server"}})",
                       "application/json");
                 });
  }

  bool bind() {
    if (bound_) return true;
    // A partial or self-contradictory upstream must fail before the port is
    // reserved, never at the first request.
    const std::string config_error = validate_upstream_config(config_);
    if (!config_error.empty()) {
      std::fprintf(stderr, "invalid upstream configuration: %s\n",
                   config_error.c_str());
      return false;
    }
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
  void handle_transcription(const httplib::Request& req, httplib::Response& res) {
    auto reject = [&res](int status, const std::string& message) {
      res.status = status;
      res.set_content(json{{"error", {{"message", utf8_sanitized(message)},
                                     {"type", "transcription_error"}}}}
                          .dump(),
                      "application/json");
    };
    if (!req.is_multipart_form_data()) {
      reject(400, "one multipart WAV file is required");
      return;
    }
    // Every part is checked against an explicit allowlist. A part that carries
    // a filename lands in the file namespace, so both namespaces are walked:
    // only "file" may arrive that way, and a scalar field cannot bypass the
    // field checks by being resent with a filename.
    for (const auto& part : req.form.files) {
      if (part.first != "file") {
        reject(400, "unsupported multipart file part: " + part.first);
        return;
      }
    }
    if (req.form.get_file_count("file") != 1) {
      reject(400, "one multipart WAV file is required");
      return;
    }
    std::map<std::string, int> field_counts;
    for (const auto& part : req.form.fields) ++field_counts[part.first];
    for (const auto& [name, count] : field_counts) {
      if (count > 1) {
        reject(400, "duplicate multipart field: " + name);
        return;
      }
    }
    static const std::set<std::string> kAllowedFields = {
        "model", "stream", "response_format", "include_generation_metadata",
        "language", "prompt", "max_new_tokens", "temperature"};
    for (const auto& [name, count] : field_counts) {
      if (!kAllowedFields.count(name)) {
        reject(400, "unsupported multipart field: " + name);
        return;
      }
    }
    const std::string requested_model = req.form.get_field("model");
    if (requested_model.empty()) {
      reject(400, "model is required");
      return;
    }
    if (requested_model != config_.served_model_name) {
      reject(400, "unsupported model: " + requested_model +
                      "; this server serves " + config_.served_model_name);
      return;
    }
    // The stage never reads these, so accepting them would be a silent no-op.
    // Presence alone is rejected, including an empty value.
    if (req.form.has_field("language")) {
      reject(400, "language is not supported by this server");
      return;
    }
    if (req.form.has_field("prompt")) {
      reject(400, "prompt is not supported by this server");
      return;
    }
    if (req.form.has_field("temperature") && req.form.get_field("temperature") != "0") {
      reject(400, "temperature is fixed at 0 by this server");
      return;
    }
    bool stream = false;
    if (req.form.has_field("stream")) {
      const std::string value = req.form.get_field("stream");
      if (value != "true" && value != "false") {
        reject(400, "stream must be true or false");
        return;
      }
      stream = value == "true";
    }
    const std::string response_format = req.form.has_field("response_format")
                                            ? req.form.get_field("response_format")
                                            : std::string("json");
    if (response_format != "json") {
      reject(400, "unsupported response_format: " + response_format +
                      "; only json is supported, for both streaming and non-streaming");
      return;
    }
    bool want_metadata = false;
    if (req.form.has_field("include_generation_metadata")) {
      const std::string value = req.form.get_field("include_generation_metadata");
      if (value != "true" && value != "false") {
        reject(400, "include_generation_metadata must be true or false");
        return;
      }
      want_metadata = value == "true";
    }
    int64_t max_tokens = 128;
    if (req.form.has_field("max_new_tokens")) {
      const std::string text = req.form.get_field("max_new_tokens");
      const auto parsed = std::from_chars(text.data(), text.data() + text.size(), max_tokens);
      if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
          max_tokens < 1 || max_tokens > 4096) {
        reject(400, "max_new_tokens must be between 1 and 4096");
        return;
      }
    }
    std::string pcm;
    try {
      pcm = client::pcm16_from_wav(req.form.get_file("file").content);
    } catch (const std::exception& error) {
      reject(400, error.what());
      return;
    }
    StagePayload payload;
    try {
      payload = serve_detail::make_asr_payload(std::move(pcm), max_tokens, "", "");
    } catch (const std::exception& error) {
      reject(400, error.what());
      return;
    }
    std::string request_id;
    try {
      request_id = coordinator_.submit(std::move(payload), config_.entry_stage);
    } catch (const std::exception& error) {
      reject(503, error.what());
      return;
    }
    bool settled = false;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::duration<double>(config_.request_timeout_s);
    while (!settled && !req.is_connection_closed()) {
      const auto remaining = deadline - std::chrono::steady_clock::now();
      if (remaining <= std::chrono::steady_clock::duration::zero()) break;
      settled = coordinator_.wait(
          request_id,
          std::min(0.1, std::chrono::duration<double>(remaining).count()));
    }
    if (req.is_connection_closed()) {
      coordinator_.abort(request_id);
      coordinator_.release(request_id);
      return;
    }
    auto info = coordinator_.info(request_id);
    if (!settled || !info) {
      coordinator_.abort(request_id);
      coordinator_.release(request_id);
      reject(504, "transcription timed out");
      return;
    }
    coordinator_.release(request_id);
    if (info->state != "completed") {
      reject(500, info->error);
      return;
    }
    std::string text;
    try {
      text = serve_detail::asr_result_text(info->result);
    } catch (const std::exception& error) {
      reject(500, error.what());
      return;
    }
    // Metadata comes only from values the stage actually reported. A stage that
    // cannot report them is rejected when the caller asked for metadata, and
    // the fields stay absent otherwise. No zero or default is fabricated.
    serve_detail::AsrGenerationMetadata metadata;
    const bool has_metadata =
        serve_detail::asr_generation_metadata(info->result, metadata);
    if (want_metadata && !has_metadata) {
      reject(500, "the ASR stage did not report generation metadata");
      return;
    }
    json out = {{"text", text}};
    if (has_metadata) {
      out["generation_metadata"] = {{"generated_token_count", metadata.generated_token_count},
                                    {"finish_reason", metadata.finish_reason}};
    }
    if (!stream) {
      res.set_content(out.dump(), "application/json");
      return;
    }
    json event = {{"type", "transcript.text.done"}, {"text", text}};
    if (has_metadata) event["generation_metadata"] = out["generation_metadata"];
    res.set_content("data: " + event.dump() + "\n\ndata: [DONE]\n\n",
                    "text/event-stream");
  }

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
      // Forward text and structured content parts unchanged. The external
      // worker validates the parts required by its upstream model.
      if (!message["content"].is_string() && !message["content"].is_array()) {
        reject("message content must be a text string or content part array");
        return;
      }
    }
    if (body.contains("stream") && !body["stream"].is_boolean()) {
      reject("stream must be a boolean");
      return;
    }
    const bool stream = body.value("stream", false);
    // One creation time for the whole request: the non-stream body and every
    // SSE chunk of that request carry the same value.
    const int64_t created = static_cast<int64_t>(std::time(nullptr));
    if (body.contains("model") && !body["model"].is_null()) {
      if (!body["model"].is_string()) {
        reject("model must be a string");
        return;
      }
      if (body["model"].get<std::string>() != config_.served_model_name) {
        reject("unsupported model: " + body["model"].get<std::string>() +
               "; this server serves " + config_.served_model_name);
        return;
      }
    }
    if (body.contains("top_p") && !body["top_p"].is_null()) {
      reject("top_p is not implemented by this server");
      return;
    }
    if (body.contains("seed") && !body["seed"].is_null()) {
      reject("seed is not implemented by this server");
      return;
    }
    if (body.contains("stop") && !body["stop"].is_null()) {
      const json& stop = body["stop"];
      const bool empty = (stop.is_string() && stop.get<std::string>().empty()) ||
                         (stop.is_array() && stop.empty());
      if (!empty) {
        reject("stop is not implemented by this server");
        return;
      }
    }
    if (body.contains("response_format") && !body["response_format"].is_null()) {
      const json& format = body["response_format"];
      // Check the type before reading it: a numeric type is a request error,
      // not a JSON exception that escapes as HTTP 500.
      if (!format.is_object() || !format.contains("type") ||
          !format["type"].is_string() || format["type"].get<std::string>() != "text") {
        reject("only response_format type text is supported");
        return;
      }
    }
    // An allowlist, not a list of known-bad names: a field this server does not
    // implement must be rejected whatever it is called. Null means absent.
    static const std::set<std::string> kAllowedFields = {
        "model", "messages", "stream", "temperature", "max_tokens",
        "max_completion_tokens", "top_p", "seed", "stop", "response_format",
        "stream_options"};
    for (auto it = body.begin(); it != body.end(); ++it) {
      if (!kAllowedFields.count(it.key()) && !it.value().is_null()) {
        reject("unsupported field: " + it.key());
        return;
      }
    }
    if (body.contains("stream_options") && !body["stream_options"].is_null()) {
      const json& options = body["stream_options"];
      if (!stream) {
        reject("stream_options requires stream=true");
        return;
      }
      if (!options.is_object()) {
        reject("stream_options must be an object");
        return;
      }
      for (auto it = options.begin(); it != options.end(); ++it) {
        if (it.key() != "include_usage") {
          reject("unsupported stream_options field: " + it.key());
          return;
        }
      }
      if (options.contains("include_usage") && !options["include_usage"].is_boolean()) {
        reject("stream_options.include_usage must be a boolean");
        return;
      }
      if (options.value("include_usage", false)) {
        reject("stream_options.include_usage is not implemented by this server");
        return;
      }
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
      bool settled = false;
      const auto deadline = std::chrono::steady_clock::now() +
                            std::chrono::duration<double>(config_.request_timeout_s);
      while (!settled && !req.is_connection_closed()) {
        const auto remaining = deadline - std::chrono::steady_clock::now();
        if (remaining <= std::chrono::steady_clock::duration::zero()) break;
        const double wait_s = std::min(
            0.1, std::chrono::duration<double>(remaining).count());
        settled = coordinator_.wait(request_id, wait_s);
      }
      if (req.is_connection_closed()) {
        coordinator_.abort(request_id);
        coordinator_.release(request_id);
        return;
      }
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
      const std::string finish = result_finish_reason(info->result);
      json out = {{"id", request_id},
                  {"object", "chat.completion"},
                  {"created", created},
                  {"model", config_.served_model_name},
                  {"choices",
                   {{{"index", 0},
                     {"message", {{"role", "assistant"},
                                  {"content", result_text(info->result)}}},
                     {"finish_reason", finish.empty() ? json(nullptr) : json(finish)}}}}};
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
        [this, sink, request_id, model, created, cleanup](size_t,
                                                          httplib::DataSink& out_sink) {
          auto send_chunk = [&](const json& payload) {
            const std::string data = "data: " + payload.dump() + "\n\n";
            return out_sink.write(data.data(), data.size());
          };
          if (!send_chunk({{"id", request_id},
                           {"object", "chat.completion.chunk"},
                           {"created", created},
                           {"model", model},
                           {"choices", {{{"index", 0},
                                        {"delta", {{"role", "assistant"}}},
                                        {"finish_reason", nullptr}}}}})) {
            coordinator_.abort(request_id);
            return false;
          }
          std::string stream_error;
          std::string finish;
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
                               {"created", created},
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
              if (settled) {
                stream_error = sink->error;
                finish = sink->finish_reason;
              }
            }
            if (settled) break;
          }
          if (!settled) coordinator_.abort(request_id);
          if (!disconnected) {
            if (stream_error.empty()) {
              send_chunk({{"id", request_id},
                          {"object", "chat.completion.chunk"},
                          {"created", created},
                          {"model", model},
                          {"choices", {{{"index", 0},
                                        {"delta", json::object()},
                                        {"finish_reason", finish.empty()
                                                              ? json(nullptr)
                                                              : json(finish)}}}}});
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

  // ---- Fixed upstream proxy routes ----------------------------------------

  void reject_route(httplib::Response& res, const char* route) {
    const std::string variant =
        config_.upstream_variant.empty() ? "none" : config_.upstream_variant;
    reject(res, 501,
           std::string(route) +
               " is not served by this process: the configured upstream variant is '" +
               variant + "'",
           "unsupported_route");
  }

  void reject(httplib::Response& res, int status, const std::string& message,
              const char* type) {
    res.status = status;
    res.set_content(json{{"error", {{"message", utf8_sanitized(message)},
                                   {"type", type}}}}
                        .dump(),
                    "application/json");
  }

  std::string upstream_url(const std::string& path) const {
    return upstream_base_url_ + path;
  }

  client::detail::HttpRequest upstream_request(const std::string& path) const {
    client::detail::HttpRequest out;
    out.url = upstream_url(path);
    out.bearer = config_.upstream_api_key;
    out.timeout = client::Milliseconds{static_cast<int64_t>(
        std::max(1.0, config_.request_timeout_s) * 1000.0)};
    out.connect_timeout = client::Milliseconds{10000};
    out.body_limit = kMaxJsonUpstreamBytes;
    // A plain-HTTP upstream is a local SGLang process; it must not be routed
    // through an environment proxy. HTTPS upstreams keep the default behavior.
    out.no_proxy = out.url.rfind("http://", 0) == 0;
    return out;
  }

  // Every upstream message passes through the key scrubber before it can reach
  // a client-visible error body.
  std::string scrubbed(const std::string& text) const {
    return utf8_sanitized(client::detail::redact(text, config_.upstream_api_key));
  }

  // Maps an upstream failure and keeps the retry metadata the upstream sent.
  void reject_upstream(httplib::Response& res, const client::Error& error) {
    reject_upstream(res, error.code(), error.status(), error.what(),
                    error.retry_after());
  }

  void reject_upstream(httplib::Response& res, client::ErrorCode code, long status,
                       const std::string& text,
                       client::Milliseconds retry_after = client::Milliseconds{0}) {
    reject(res, upstream_error_status(code, status), scrubbed(text),
           "upstream_error");
    if (status == 503 && retry_after.count() > 0) {
      res.set_header("Retry-After", std::to_string((retry_after.count() + 999) / 1000));
    }
  }

  void relay_json(const httplib::Request& req, client::detail::HttpRequest out,
                  httplib::Response& res) {
    out.require_content_type = "application/json";
    // The blocking transfer has no callback that sees a client disconnect, so a
    // watcher cancels it through the existing cancel flag.
    std::atomic<bool> cancel{false};
    std::atomic<bool> done{false};
    out.cancel = &cancel;
    std::thread watcher([&req, &cancel, &done] {
      while (!done.load() && !req.is_connection_closed()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
      if (!done.load()) cancel.store(true);
    });
    try {
      auto response = client::detail::perform(out);
      done.store(true);
      watcher.join();
      // Keep the upstream status (200, 201, 202) instead of flattening it.
      res.status = static_cast<int>(response.status);
      res.set_content(response.body, "application/json");
    } catch (const client::Error& error) {
      done.store(true);
      watcher.join();
      // The client is gone; there is nobody to answer.
      if (error.code() == client::ErrorCode::kCancelled) return;
      reject_upstream(res, error);
    } catch (const std::exception& error) {
      done.store(true);
      watcher.join();
      reject(res, 502, scrubbed(error.what()), "upstream_error");
    }
  }

  // Streams an upstream response to the client without aggregating it. The
  // handler waits for the first byte, so an upstream failure still maps to its
  // own HTTP status instead of a committed 200.
  void relay_stream(const httplib::Request& req, client::detail::HttpRequest out,
                    httplib::Response& res, const char* content_type, bool sse) {
    auto relay = std::make_shared<UpstreamRelay>();
    auto framer = sse ? std::make_shared<SseFramer>(relay.get()) : nullptr;
    if (framer) {
      out.on_body = [framer](std::string_view bytes) { return framer->feed(bytes); };
      relay->start(std::move(out), [framer] { framer->finish(); });
    } else {
      // Same invariant as SseFramer: the worker holds a raw pointer and
      // cannot drop the last reference. The relay destructor runs only on the
      // handler or provider thread and joins the worker before members die.
      UpstreamRelay* raw = relay.get();
      out.on_body = [raw](std::string_view bytes) {
        return raw->push(std::string(bytes));
      };
      relay->start(std::move(out));
    }
    // Wait for the first upstream byte, but stop when the client disconnects:
    // there is no response to commit and the upstream transfer must be
    // cancelled instead of running to completion.
    const auto deadline =
        std::chrono::steady_clock::now() +
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(config_.request_timeout_s));
    while (!relay->wait_started(0.2)) {
      // A connect failure settles the transfer without a first byte; map the
      // stored error at once instead of waiting for the request timeout.
      if (relay->settled_without_start()) {
        relay->cancel();
        if (relay->has_error()) {
          reject_upstream(res, relay->error_code(), relay->upstream_status(),
                          relay->error_text(), relay->retry_after());
        } else {
          reject(res, 502, "upstream ended before sending a response body",
                 "upstream_error");
        }
        return;
      }
      if (req.is_connection_closed()) {
        relay->cancel();
        return;
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        relay->cancel();
        if (relay->has_error()) {
          reject_upstream(res, relay->error_code(), relay->upstream_status(),
                          relay->error_text(), relay->retry_after());
        } else {
          reject(res, 504, "upstream did not start a response", "upstream_error");
        }
        return;
      }
    }
    res.set_chunked_content_provider(
        content_type, [relay, framer](size_t, httplib::DataSink& sink) {
          while (true) {
            if (!sink.is_writable()) {
              relay->cancel();
              return false;
            }
            std::string piece;
            if (relay->pop_wait(piece, std::chrono::milliseconds(200))) {
              if (!sink.write(piece.data(), piece.size())) {
                relay->cancel();
                return false;
              }
              continue;
            }
            if (relay->finished()) break;
          }
          // A transport, length, content-type or missing-terminator failure is
          // reported by aborting the body: no complete response is signalled.
          if (relay->has_error()) return false;
          if (framer && !framer->saw_done()) return false;
          sink.done();
          return true;
        });
  }

  bool served_model_matches(const json& body, std::string& error) const {
    if (!body.contains("model") || body["model"].is_null()) return true;
    if (!body["model"].is_string()) {
      error = "model must be a string";
      return false;
    }
    const std::string requested = body["model"].get<std::string>();
    if (requested != config_.served_model_name) {
      error = "unsupported model: " + requested + "; this server serves " +
              config_.served_model_name;
      return false;
    }
    return true;
  }

  bool valid_image_generation(const json& body, std::string& error) const {
    if (!body.is_object()) {
      error = "body must be a JSON object";
      return false;
    }
    if (!body.contains("prompt") || !body["prompt"].is_string() ||
        body["prompt"].get_ref<const std::string&>().empty()) {
      error = "prompt must be a non-empty string";
      return false;
    }
    static const std::set<std::string> kAllowed = {
        "model", "prompt", "n", "size", "response_format", "output_format",
        "background", "seed", "generator_device", "user", "negative_prompt",
        "guidance_scale", "true_cfg_scale", "num_inference_steps", "quality",
        "output_quality", "output_compression", "enhance_prompt", "task_type",
        "style"};
    for (auto it = body.begin(); it != body.end(); ++it) {
      if (!it.value().is_null() && !kAllowed.count(it.key())) {
        error = "unsupported field: " + it.key();
        return false;
      }
    }
    if (!served_model_matches(body, error)) return false;
    // A local single-model process has no cloud storage. An upstream "url"
    // result points at the upstream's own address, which this proxy must not
    // expose, so only inline bytes are accepted.
    if (body.contains("response_format") && !body["response_format"].is_null()) {
      if (!body["response_format"].is_string() ||
          body["response_format"].get<std::string>() != "b64_json") {
        error = "response_format must be b64_json: url responses would expose "
                "upstream storage";
        return false;
      }
    }
    return true;
  }

  bool valid_video_create(const json& body, std::string& error) const {
    if (!body.is_object()) {
      error = "body must be a JSON object";
      return false;
    }
    if (!body.contains("prompt") || !body["prompt"].is_string() ||
        body["prompt"].get_ref<const std::string&>().empty()) {
      error = "prompt must be a non-empty string";
      return false;
    }
    static const std::set<std::string> kAllowed = {
        "model", "prompt", "task", "conditions", "target", "seed", "size",
        "seconds", "n", "fps", "num_frames", "negative_prompt", "enhance_prompt",
        "input_reference", "reference_url", "video_url", "video_path",
        "generator_device", "task_type", "num_outputs_per_prompt",
        // Official MiniMax-H3 sampling fields.
        "num_inference_steps", "flow_shift", "audio_flow_shift",
        "reference_resize_mode"};
    for (auto it = body.begin(); it != body.end(); ++it) {
      if (!it.value().is_null() && !kAllowed.count(it.key())) {
        error = "unsupported field: " + it.key();
        return false;
      }
    }
    if (!served_model_matches(body, error)) return false;
    if (body.contains("task") && !body["task"].is_null()) {
      if (!body["task"].is_string()) {
        error = "task must be a string";
        return false;
      }
      const std::string task = body["task"].get<std::string>();
      if (task != "t2va" && task != "fl2va" && task != "ref2va") {
        error = "task must be t2va, fl2va or ref2va";
        return false;
      }
    }
    if (body.contains("conditions") && !body["conditions"].is_null()) {
      if (!body["conditions"].is_array()) {
        error = "conditions must be an array";
        return false;
      }
      for (const auto& condition : body["conditions"]) {
        if (!condition.is_object() || !condition.contains("type") ||
            !condition["type"].is_string() || !condition.contains("uri") ||
            !condition["uri"].is_string()) {
          error = "each condition requires a string type and uri";
          return false;
        }
        const std::string type = condition["type"].get<std::string>();
        if (type != "image" && type != "video" && type != "audio" &&
            type != "video_audio") {
          error = "condition type must be image, video, audio or video_audio";
          return false;
        }
        if (condition.contains("role") && !condition["role"].is_null()) {
          if (!condition["role"].is_string()) {
            error = "condition role must be a string";
            return false;
          }
          const std::string role = condition["role"].get<std::string>();
          if (role != "keyframe" && role != "reference") {
            error = "condition role must be keyframe or reference";
            return false;
          }
        }
      }
    }
    return true;
  }

  bool valid_qwen_chat(const json& body, std::string& error) const {
    if (!body.is_object()) {
      error = "body must be a JSON object";
      return false;
    }
    if (!served_model_matches(body, error)) return false;
    if (!body.contains("messages") || !body["messages"].is_array() ||
        body["messages"].empty()) {
      error = "messages must be a non-empty array";
      return false;
    }
    for (const auto& message : body["messages"]) {
      if (!message.is_object() || !message.contains("role") ||
          !message["role"].is_string() || !message.contains("content")) {
        error = "each message requires a string role and content";
        return false;
      }
      const std::string role = message["role"].get<std::string>();
      if (role != "system" && role != "user" && role != "assistant") {
        error = "message role must be system, user or assistant";
        return false;
      }
      const json& content = message["content"];
      if (content.is_string()) continue;
      if (!content.is_array()) {
        error = "message content must be a string or a content part array";
        return false;
      }
      for (const auto& part : content) {
        if (!part.is_object() || !part.contains("type") ||
            !part["type"].is_string()) {
          error = "each content part requires a string type";
          return false;
        }
        const std::string type = part["type"].get<std::string>();
        if (type == "text") {
          if (!part.contains("text") || !part["text"].is_string()) {
            error = "text part requires a string text";
            return false;
          }
        } else if (type == "image_url") {
          if (!part.contains("image_url") || !part["image_url"].is_object() ||
              !part["image_url"].contains("url") ||
              !part["image_url"]["url"].is_string()) {
            error = "image_url part requires image_url.url";
            return false;
          }
        } else if (type == "input_audio") {
          if (!part.contains("input_audio") || !part["input_audio"].is_object() ||
              !part["input_audio"].contains("data") ||
              !part["input_audio"]["data"].is_string() ||
              !part["input_audio"].contains("format") ||
              !part["input_audio"]["format"].is_string()) {
            error = "input_audio part requires input_audio.data and format";
            return false;
          }
        } else if (type == "video") {
          // The Qwen3.5-Omni contract accepts 2 to 2048 frames.
          if (!part.contains("video") || !part["video"].is_array() ||
              part["video"].size() < 2 || part["video"].size() > 2048) {
            error = "video part requires 2 to 2048 frame urls";
            return false;
          }
          for (const auto& url : part["video"]) {
            if (!url.is_string()) {
              error = "video part urls must be strings";
              return false;
            }
          }
        } else if (type == "video_url") {
          if (!part.contains("video_url") || !part["video_url"].is_object() ||
              !part["video_url"].contains("url") ||
              !part["video_url"]["url"].is_string()) {
            error = "video_url part requires video_url.url";
            return false;
          }
          if (part.contains("fps") && !part["fps"].is_null()) {
            if (!part["fps"].is_number()) {
              error = "video fps must be a number";
              return false;
            }
            const double fps = part["fps"].get<double>();
            if (fps < 0.1 || fps > 10) {
              error = "video fps must be between 0.1 and 10";
              return false;
            }
          }
        } else {
          error = "unsupported content part type: " + type;
          return false;
        }
      }
    }
    static const std::set<std::string> kAllowed = {
        "model", "messages", "stream", "stream_options", "modalities", "audio",
        "temperature", "top_p", "max_tokens", "max_completion_tokens", "stop",
        "seed", "response_format"};
    for (auto it = body.begin(); it != body.end(); ++it) {
      if (!it.value().is_null() && !kAllowed.count(it.key())) {
        error = "unsupported field: " + it.key();
        return false;
      }
    }
    if (body.contains("stream") && !body["stream"].is_boolean()) {
      error = "stream must be a boolean";
      return false;
    }
    const bool stream = body.value("stream", false);
    if (body.contains("stream_options") && !body["stream_options"].is_null()) {
      const json& options = body["stream_options"];
      if (!options.is_object()) {
        error = "stream_options must be an object";
        return false;
      }
      for (auto it = options.begin(); it != options.end(); ++it) {
        if (it.key() != "include_usage") {
          error = "unsupported stream_options field: " + it.key();
          return false;
        }
      }
      if (options.contains("include_usage") && !options["include_usage"].is_boolean()) {
        error = "stream_options.include_usage must be a boolean";
        return false;
      }
    }
    bool audio_output = false;
    if (body.contains("modalities") && !body["modalities"].is_null()) {
      const json& modalities = body["modalities"];
      if (!modalities.is_array() || modalities.empty()) {
        error = "modalities must be a non-empty array";
        return false;
      }
      for (const auto& modality : modalities) {
        if (!modality.is_string()) {
          error = "modalities values must be strings";
          return false;
        }
        const std::string name = modality.get<std::string>();
        if (name != "text" && name != "audio") {
          error = "modalities values must be text or audio";
          return false;
        }
        if (name == "audio") audio_output = true;
      }
    }
    if (body.contains("audio") && !body["audio"].is_null()) {
      const json& audio = body["audio"];
      if (!audio_output) {
        error = "audio requires modalities to include audio";
        return false;
      }
      if (!audio.is_object() || !audio.contains("voice") ||
          !audio["voice"].is_string() ||
          audio["voice"].get_ref<const std::string&>().empty() ||
          !audio.contains("format") || !audio["format"].is_string()) {
        error = "audio requires a non-empty voice and a format";
        return false;
      }
      if (audio["format"].get<std::string>() != "wav") {
        error = "audio.format must be wav: it is the only documented format";
        return false;
      }
    } else if (audio_output) {
      error = "audio output requires the audio object with voice and format";
      return false;
    }
    if (audio_output && !stream) {
      // The official contract requires streaming for speech output.
      error = "audio output requires stream=true";
      return false;
    }
    if (body.contains("response_format") && !body["response_format"].is_null()) {
      const json& format = body["response_format"];
      if (!format.is_object() || !format.contains("type") ||
          !format["type"].is_string() ||
          format["type"].get<std::string>() != "text") {
        error = "only response_format type text is supported";
        return false;
      }
    }
    return true;
  }

  void proxy_qwen_chat(const httplib::Request& req, httplib::Response& res) {
    const json body = json::parse(req.body, nullptr, false);
    std::string error;
    if (!valid_qwen_chat(body, error)) {
      reject(res, 400, error, "invalid_request");
      return;
    }
    const bool stream = body.value("stream", false);
    json forwarded = body;
    forwarded["model"] = config_.upstream_model;
    auto out = upstream_request("/chat/completions");
    out.method = client::detail::HttpMethod::kPost;
    out.body = forwarded.dump();
    out.headers.emplace("Content-Type", "application/json");
    if (stream) {
      out.accept = "text/event-stream";
      out.require_content_type = "text/event-stream";
      out.stream_limit = kMaxJsonUpstreamBytes;
      relay_stream(req, std::move(out), res, "text/event-stream", true);
    } else {
      out.accept = "application/json";
      relay_json(req, std::move(out), res);
    }
  }

  void handle_image_generation(const httplib::Request& req,
                               httplib::Response& res) {
    const json body = json::parse(req.body, nullptr, false);
    std::string error;
    if (!valid_image_generation(body, error)) {
      reject(res, 400, error, "invalid_request");
      return;
    }
    json forwarded = body;
    forwarded["model"] = config_.upstream_model;
    auto out = upstream_request("/v1/images/generations");
    out.method = client::detail::HttpMethod::kPost;
    out.body = forwarded.dump();
    out.headers.emplace("Content-Type", "application/json");
    out.accept = "application/json";
    relay_json(req, std::move(out), res);
  }

  void handle_image_edit(const httplib::Request& req, httplib::Response& res) {
    if (!req.is_multipart_form_data()) {
      reject(res, 400, "one multipart form is required", "invalid_request");
      return;
    }
    for (const auto& part : req.form.files) {
      if (part.first != "image") {
        reject(res, 400, "unsupported multipart file part: " + part.first,
               "invalid_request");
        return;
      }
    }
    if (req.form.get_file_count("image") == 0) {
      reject(res, 400, "at least one image part is required", "invalid_request");
      return;
    }
    static const std::set<std::string> kAllowed = {
        "prompt", "model", "n", "response_format", "size", "output_format",
        "background", "seed", "generator_device", "user", "negative_prompt",
        "guidance_scale", "true_cfg_scale", "num_inference_steps", "quality",
        "output_quality", "output_compression", "enhance_prompt", "url",
        "task_type"};
    std::map<std::string, int> counts;
    for (const auto& part : req.form.fields) {
      ++counts[part.first];
      if (!kAllowed.count(part.first)) {
        reject(res, 400, "unsupported multipart field: " + part.first,
               "invalid_request");
        return;
      }
    }
    for (const auto& [name, count] : counts) {
      // url is the documented repeatable text field; everything else is scalar.
      if (count > 1 && name != "url") {
        reject(res, 400, "duplicate multipart field: " + name, "invalid_request");
        return;
      }
    }
    const std::string prompt = req.form.get_field("prompt");
    if (prompt.empty()) {
      reject(res, 400, "prompt is required", "invalid_request");
      return;
    }
    const std::string requested_model = req.form.get_field("model");
    if (!requested_model.empty() && requested_model != config_.served_model_name) {
      reject(res, 400,
             "unsupported model: " + requested_model + "; this server serves " +
                 config_.served_model_name,
             "invalid_request");
      return;
    }
    const std::string format = req.form.get_field("response_format");
    if (!format.empty() && format != "b64_json") {
      reject(res, 400,
             "response_format must be b64_json: url responses would expose "
             "upstream storage",
             "invalid_request");
      return;
    }
    auto out = upstream_request("/v1/images/edits");
    out.method = client::detail::HttpMethod::kPost;
    // Image order defines the prompt placeholders, so the parts are rebuilt in
    // request order and no other part is interleaved before them.
    for (const auto& [name, part] : req.form.files) {
      client::detail::MimePart mime;
      mime.name = "image";
      mime.data = part.content;
      mime.filename = "image";
      mime.content_type =
          part.content_type.empty() ? "application/octet-stream" : part.content_type;
      out.mime_parts.push_back(std::move(mime));
    }
    for (const auto& [name, field] : req.form.fields) {
      if (name == "model") continue;
      client::detail::MimePart mime;
      mime.name = name;
      mime.data = field.content;
      out.mime_parts.push_back(std::move(mime));
    }
    client::detail::MimePart model_part;
    model_part.name = "model";
    model_part.data = config_.upstream_model;
    out.mime_parts.push_back(std::move(model_part));
    relay_json(req, std::move(out), res);
  }

  void handle_video_create(const httplib::Request& req, httplib::Response& res) {
    const json body = json::parse(req.body, nullptr, false);
    std::string error;
    if (!valid_video_create(body, error)) {
      reject(res, 400, error, "invalid_request");
      return;
    }
    json forwarded = body;
    forwarded["model"] = config_.upstream_model;
    auto out = upstream_request("/v1/videos");
    out.method = client::detail::HttpMethod::kPost;
    out.body = forwarded.dump();
    out.headers.emplace("Content-Type", "application/json");
    out.accept = "application/json";
    // Creation is not retried: a retry would submit a second job.
    relay_json(req, std::move(out), res);
  }

  void handle_video_status(const httplib::Request& req, const std::string& video_id,
                           httplib::Response& res) {
    if (!valid_path_id(video_id)) {
      reject(res, 400, "invalid video id", "invalid_request");
      return;
    }
    auto out = upstream_request("/v1/videos/" + video_id);
    out.method = client::detail::HttpMethod::kGet;
    out.accept = "application/json";
    relay_json(req, std::move(out), res);
  }

  void handle_video_content(const httplib::Request& req, const std::string& video_id,
                            httplib::Response& res) {
    if (!valid_path_id(video_id)) {
      reject(res, 400, "invalid video id", "invalid_request");
      return;
    }
    // variant selects among multiple outputs of one job. Only that key is
    // forwarded; any other query parameter is rejected rather than dropped.
    // variant is a zero-based non-negative integer. Any other value, a repeat,
    // or an unknown query key is rejected locally instead of reaching upstream.
    bool seen_variant = false;
    std::string variant;
    for (const auto& [name, value] : req.params) {
      if (name != "variant") {
        reject(res, 400, "unsupported query parameter: " + name, "invalid_request");
        return;
      }
      if (seen_variant) {
        reject(res, 400, "duplicate query parameter: variant", "invalid_request");
        return;
      }
      seen_variant = true;
      int parsed = 0;
      const char* begin = value.data();
      const char* end = begin + value.size();
      const auto result = std::from_chars(begin, end, parsed);
      if (value.empty() || result.ec != std::errc{} || result.ptr != end ||
          parsed < 0) {
        reject(res, 400, "variant must be a non-negative integer",
               "invalid_request");
        return;
      }
      variant = value;
    }
    std::string path = "/v1/videos/" + video_id + "/content";
    if (seen_variant) path += "?variant=" + variant;
    auto out = upstream_request(path);
    out.method = client::detail::HttpMethod::kGet;
    out.accept = "video/mp4";
    out.require_content_type = "video/mp4";
    out.stream_limit = kMaxBinaryUpstreamBytes;
    relay_stream(req, std::move(out), res, "video/mp4", false);
  }

  void handle_video_delete(const httplib::Request& req, const std::string& video_id,
                           httplib::Response& res) {
    if (!valid_path_id(video_id)) {
      reject(res, 400, "invalid video id", "invalid_request");
      return;
    }
    auto out = upstream_request("/v1/videos/" + video_id);
    out.method = client::detail::HttpMethod::kDelete;
    out.accept = "application/json";
    relay_json(req, std::move(out), res);
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
      bool has_deltas;
      {
        std::lock_guard<std::mutex> lock(sink->mu);
        has_deltas = sink->has_deltas;
        // The finish reason comes from the stage result, never from a default.
        sink->finish_reason = result_finish_reason(info.result);
      }
      // Incremental deltas already contain the text. The aggregate result is
      // needed only when the stage sent no deltas.
      if (!has_deltas) sink->push(info.result);
      sink->finish();
      return;
    }
    sink->finish(info.error.empty() ? "request " + info.state : info.error);
  }

 private:
  Coordinator& coordinator_;
  ApiConfig config_;
  std::string upstream_base_url_;
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

}  // namespace umminfer

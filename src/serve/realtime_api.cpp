// SPDX-License-Identifier: Apache-2.0
#include "umminfer/serve/realtime_api.hpp"

#include <libwebsockets.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <deque>
#include <new>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "asr_common.hpp"

namespace umminfer {
namespace {

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

constexpr size_t kMaxMessageBytes = 1u << 20;
constexpr size_t kMaxAudioBytes = 64u << 20;

struct RealtimeSession {
  std::string incoming;
  std::string audio;
  std::string request_id;
  std::deque<std::string> outbound;
  Clock::time_point deadline;
  int64_t max_tokens = 128;
  bool ready = false;
  bool close_after_write = false;
};

std::string header(lws* wsi, lws_token_indexes token) {
  const int length = lws_hdr_total_length(wsi, token);
  if (length <= 0 || length > 8192) return {};
  std::string value(static_cast<size_t>(length) + 1, '\0');
  if (lws_hdr_copy(wsi, value.data(), length + 1, token) != length) return {};
  value.resize(static_cast<size_t>(length));
  return value;
}

// libwebsockets stores the request path without the query string. The query
// values are available separately, already percent-decoded.
bool query_argument(lws* wsi, const char* name, std::string& out) {
  char buffer[8192];
  const int length =
      lws_get_urlarg_by_name_safe(wsi, name, buffer, sizeof(buffer));
  if (length < 0 || length >= static_cast<int>(sizeof(buffer))) return false;
  out.assign(buffer, static_cast<size_t>(length));
  return true;
}

// The URI lws exposes is normalized: it collapses //, resolves dot segments
// and decodes percent escapes. A different path is rejected, but a normalized
// alias of /v1/realtime is indistinguishable here. That is a known limitation.
bool valid_handshake(lws* wsi, const std::string& served_model,
                     std::string& error) {
  const std::string path = header(wsi, WSI_TOKEN_GET_URI);
  if (path != "/v1/realtime") {
    error = "realtime WebSocket path must be exactly /v1/realtime";
    return false;
  }
  std::string intent;
  if (!query_argument(wsi, "intent", intent) || intent != "transcription") {
    error = "realtime WebSocket intent must be transcription";
    return false;
  }
  std::string model;
  if (!query_argument(wsi, "model", model)) {
    error = "realtime WebSocket requires a model query parameter";
    return false;
  }
  if (model != served_model) {
    error = "realtime WebSocket model does not match the served model '" +
            served_model + "'";
    return false;
  }
  return true;
}

// The session.update reply must describe the configuration the server now
// enforces, not echo fields that were ignored.
json effective_session(const RealtimeSession& session) {
  return {{"input_audio_format", "pcm16"},
          {"turn_detection", nullptr},
          {"max_new_tokens", session.max_tokens}};
}

// Copies only metadata the stage actually reported. A stage that reports a
// nested generation_metadata map takes precedence over top-level fields.
std::optional<json> live_generation_metadata(const Value& result) {
  const Value* nested = result.find("generation_metadata");
  const Value& source =
      nested != nullptr && nested->is_map() ? *nested : result;
  json out = json::object();
  if (const Value* count = source.find("generated_token_count");
      count != nullptr && count->is_int() && count->as_int() >= 0) {
    out["generated_token_count"] = count->as_int();
  }
  if (const Value* finish = source.find("finish_reason");
      finish != nullptr && finish->is_str() && !finish->as_str().empty()) {
    out["finish_reason"] = finish->as_str();
  }
  if (out.empty()) return std::nullopt;
  return out;
}

int base64_value(unsigned char c) {
  if (c >= 'A' && c <= 'Z') return c - 'A';
  if (c >= 'a' && c <= 'z') return c - 'a' + 26;
  if (c >= '0' && c <= '9') return c - '0' + 52;
  if (c == '+') return 62;
  if (c == '/') return 63;
  return -1;
}

std::string decode_base64(std::string_view input) {
  if (input.empty() || input.size() % 4 != 0) throw ProtocolError("invalid audio base64");
  std::string output;
  output.reserve(input.size() / 4 * 3);
  for (size_t offset = 0; offset < input.size(); offset += 4) {
    const bool last = offset + 4 == input.size();
    const int padding = (input[offset + 3] == '=' ? 1 : 0) +
                        (input[offset + 2] == '=' ? 1 : 0);
    if ((!last && padding) || input[offset] == '=' || input[offset + 1] == '=' ||
        (input[offset + 2] == '=' && input[offset + 3] != '=')) {
      throw ProtocolError("invalid audio base64 padding");
    }
    int values[4] = {0, 0, 0, 0};
    for (size_t index = 0; index < 4 - static_cast<size_t>(padding); ++index) {
      values[index] = base64_value(static_cast<unsigned char>(input[offset + index]));
      if (values[index] < 0) throw ProtocolError("invalid audio base64 character");
    }
    if ((padding == 2 && (values[1] & 15) != 0) ||
        (padding == 1 && (values[2] & 3) != 0)) {
      throw ProtocolError("non-canonical audio base64");
    }
    const uint32_t bits = (static_cast<uint32_t>(values[0]) << 18) |
                          (static_cast<uint32_t>(values[1]) << 12) |
                          (static_cast<uint32_t>(values[2]) << 6) |
                          static_cast<uint32_t>(values[3]);
    output.push_back(static_cast<char>((bits >> 16) & 0xff));
    if (padding < 2) output.push_back(static_cast<char>((bits >> 8) & 0xff));
    if (padding < 1) output.push_back(static_cast<char>(bits & 0xff));
  }
  return output;
}

int realtime_callback(lws* wsi, lws_callback_reasons reason, void* user,
                      void* in, size_t len);

const lws_protocols protocols[] = {
    {"umminfer-realtime", realtime_callback, sizeof(RealtimeSession), 16384, 0, nullptr, 0},
    LWS_PROTOCOL_LIST_TERM,
};

}  // namespace

class RealtimeApi::Impl {
 public:
  Impl(Coordinator& coordinator, ApiConfig config)
      : coordinator_(coordinator), config_(std::move(config)) {
    if (config_.served_model_name.empty()) {
      config_.served_model_name = config_.model;
    }
  }

  ~Impl() {
    if (context_) lws_context_destroy(context_);
  }

  bool bind() {
    if (context_) return true;
    lws_context_creation_info info{};
    info.port = config_.port;
    info.iface = config_.host.empty() ? nullptr : config_.host.c_str();
    info.protocols = protocols;
    info.user = this;
    info.vhost_name = "umminfer";
    info.options = LWS_SERVER_OPTION_DISABLE_IPV6;
    context_ = lws_create_context(&info);
    if (!context_) return false;
    lws_vhost* vhost = lws_get_vhost_by_name(context_, "umminfer");
    if (!vhost) return false;
    bound_port_ = config_.port == 0 ? lws_get_vhost_listen_port(vhost)
                                    : config_.port;
    return bound_port_ >= 0;
  }

  bool listen() {
    if (!context_ && !bind()) return false;
    running_ = true;
    schedule_poll();
    bool ok = true;
    while (running_) {
      if (lws_service(context_, 0) < 0) {
        ok = false;
        break;
      }
    }
    lws_sul_cancel(&poll_sul_);
    return ok;
  }

  void schedule_poll() {
    lws_sul_schedule(context_, 0, &poll_sul_, poll_timer, 10 * LWS_US_PER_MS);
  }

  static void poll_timer(lws_sorted_usec_list_t* sul) {
    auto* self = lws_container_of(sul, RealtimeApi::Impl, poll_sul_);
    self->poll_sessions();
    if (self->running_) self->schedule_poll();
  }

  void stop() {
    running_ = false;
    if (context_) lws_cancel_service(context_);
  }

  int bound_port() const { return bound_port_; }

  bool authorized(lws* wsi) const {
    if (config_.api_key.empty()) return true;
    return header(wsi, WSI_TOKEN_HTTP_AUTHORIZATION) == "Bearer " + config_.api_key;
  }

  void established(lws* wsi, RealtimeSession* session) {
    new (session) RealtimeSession;
    sessions_.insert(wsi);
  }

  void closed(lws* wsi, RealtimeSession* session) {
    if (!session->request_id.empty()) {
      coordinator_.abort(session->request_id);
      coordinator_.release(session->request_id);
    }
    sessions_.erase(wsi);
    session->~RealtimeSession();
  }

  void send(lws* wsi, RealtimeSession* session, const json& event) {
    session->outbound.push_back(event.dump());
    lws_callback_on_writable(wsi);
  }

  void fail(lws* wsi, RealtimeSession* session, const std::string& message) {
    if (session->close_after_write) return;
    send(wsi, session,
         {{"type", "error"}, {"error", {{"code", "request_error"}, {"message", message}}}});
    session->close_after_write = true;
  }

  void submit(lws* wsi, RealtimeSession* session) {
    if (!session->ready || session->audio.empty() || session->audio.size() % 2 != 0 ||
        !session->request_id.empty() || session->close_after_write) {
      fail(wsi, session, "transcription.done requires an accepted session and PCM16 audio");
      return;
    }
    try {
      StagePayload payload = serve_detail::make_asr_payload(
          session->audio, session->max_tokens, "", "");
      session->request_id = "realtime-" + std::to_string(++next_request_id_);
      payload.request_id = session->request_id;
      session->deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(
          std::chrono::duration<double>(config_.request_timeout_s));
      coordinator_.submit(std::move(payload), config_.entry_stage);
    } catch (const std::exception& error) {
      if (!session->request_id.empty()) coordinator_.release(session->request_id);
      session->request_id.clear();
      fail(wsi, session, error.what());
    }
  }

  // Accepts only fields this server actually applies. Validation completes
  // before any state changes, so a rejected update leaves the session intact.
  void update_session(lws* wsi, RealtimeSession* session, const json& settings) {
    if (!session->request_id.empty()) {
      throw ProtocolError("session.update is not accepted after transcription.done");
    }
    static const std::set<std::string> kAllowed = {
        "input_audio_format", "turn_detection", "max_new_tokens"};
    for (auto it = settings.begin(); it != settings.end(); ++it) {
      // The stage never reads these, so accepting them would report a
      // configuration that is not in effect. Presence alone is rejected,
      // including an empty string or null.
      if (it.key() == "language") {
        throw ProtocolError("language is not supported by this server");
      }
      if (it.key() == "prompt") {
        throw ProtocolError("prompt is not supported by this server");
      }
      if (it.key() == "audio_layout") {
        throw ProtocolError(
            "audio_layout is not supported: this backend consumes headerless "
            "PCM16 mono 16 kHz audio and reports no named layout; omit the field");
      }
      if (!kAllowed.count(it.key())) {
        throw ProtocolError("unsupported session field: " + it.key());
      }
    }
    if (settings.contains("input_audio_format")) {
      const json& format = settings["input_audio_format"];
      if (!format.is_string() || format.get<std::string>() != "pcm16") {
        throw ProtocolError("input_audio_format must be pcm16");
      }
    }
    if (settings.contains("turn_detection") &&
        !settings["turn_detection"].is_null()) {
      throw ProtocolError(
          "turn_detection must be null: server-side VAD is not implemented");
    }
    int64_t max_tokens = session->max_tokens;
    if (settings.contains("max_new_tokens")) {
      const json& limit = settings["max_new_tokens"];
      // nlohmann parses a non-negative JSON integer as number_unsigned, so both
      // signed and unsigned values must be accepted when they are in range.
      const bool valid =
          limit.is_number_integer() &&
          (limit.is_number_unsigned()
               ? limit.get<uint64_t>() >= 1 && limit.get<uint64_t>() <= 4096
               : limit.get<int64_t>() >= 1 && limit.get<int64_t>() <= 4096);
      if (!valid) {
        throw ProtocolError("max_new_tokens must be between 1 and 4096");
      }
      max_tokens = limit.is_number_unsigned()
                       ? static_cast<int64_t>(limit.get<uint64_t>())
                       : limit.get<int64_t>();
    }
    session->max_tokens = max_tokens;
    session->ready = true;
    send(wsi, session,
         {{"type", "transcription_session.updated"},
          {"session", effective_session(*session)}});
  }

  void receive(lws* wsi, RealtimeSession* session, const void* data, size_t length) {
    if (session->close_after_write) return;
    if (length > kMaxMessageBytes - std::min(session->incoming.size(), kMaxMessageBytes)) {
      fail(wsi, session, "realtime message exceeds 1 MiB");
      return;
    }
    session->incoming.append(static_cast<const char*>(data), length);
    if (!lws_is_final_fragment(wsi) || lws_remaining_packet_payload(wsi) > 0) return;
    std::string message = std::move(session->incoming);
    session->incoming.clear();
    try {
      const json event = json::parse(message);
      if (!event.is_object() || !event.contains("type") || !event["type"].is_string()) {
        throw ProtocolError("realtime event requires a type");
      }
      const std::string type = event["type"].get<std::string>();
      if (type == "session.update") {
        if (!event.contains("session") || !event["session"].is_object()) {
          throw ProtocolError("session.update requires a session object");
        }
        update_session(wsi, session, event["session"]);
      } else if (type == "input_audio_buffer.append") {
        if (!session->request_id.empty()) {
          throw ProtocolError(
              "audio append is not accepted after transcription.done");
        }
        if (!session->ready || !event.contains("audio") || !event["audio"].is_string()) {
          throw ProtocolError("audio append requires a string after session.update");
        }
        std::string pcm = decode_base64(event["audio"].get_ref<const std::string&>());
        if (pcm.empty() || pcm.size() % 2 != 0) throw ProtocolError("invalid PCM16 audio append");
        if (pcm.size() > kMaxAudioBytes - std::min(session->audio.size(), kMaxAudioBytes)) {
          throw ProtocolError("realtime audio exceeds the server limit");
        }
        session->audio.append(pcm);
      } else if (type == "transcription.done") {
        submit(wsi, session);
      } else {
        throw ProtocolError("unsupported realtime event type");
      }
    } catch (const std::exception& error) {
      fail(wsi, session, error.what());
    }
  }

  void poll_sessions() {
    const auto now = Clock::now();
    for (lws* wsi : sessions_) {
      auto* session = static_cast<RealtimeSession*>(lws_wsi_user(wsi));
      if (!session || session->request_id.empty()) continue;
      const std::string request_id = session->request_id;
      if (coordinator_.wait(request_id, 0)) {
        auto info = coordinator_.info(request_id);
        coordinator_.release(request_id);
        session->request_id.clear();
        if (!info || info->state != "completed") {
          fail(wsi, session, info ? info->error : "request result disappeared");
          continue;
        }
        try {
          const std::string text = serve_detail::asr_result_text(info->result);
          send(wsi, session, {{"type", "transcription.segment"},
                              {"event_index", 1}, {"segment_id", 0},
                              {"text", text}, {"is_final", true}});
          json completed = {{"type", "transcription.completed"},
                            {"text", text}};
          if (const auto metadata = live_generation_metadata(info->result)) {
            completed["generation_metadata"] = *metadata;
          }
          send(wsi, session, completed);
        } catch (const std::exception& error) {
          fail(wsi, session, error.what());
        }
        session->close_after_write = true;
      } else if (now >= session->deadline) {
        coordinator_.abort(request_id);
        coordinator_.release(request_id);
        session->request_id.clear();
        fail(wsi, session, "realtime transcription timed out");
      }
    }
  }

  Coordinator& coordinator_;
  ApiConfig config_;
  lws_context* context_ = nullptr;
  lws_sorted_usec_list_t poll_sul_{};
  std::atomic<bool> running_{false};
  std::set<lws*> sessions_;
  uint64_t next_request_id_ = 0;
  int bound_port_ = -1;
};

namespace {

int realtime_callback(lws* wsi, lws_callback_reasons reason, void* user,
                      void* in, size_t len) {
  auto* self = static_cast<RealtimeApi::Impl*>(lws_context_user(lws_get_context(wsi)));
  auto* session = static_cast<RealtimeSession*>(user);
  switch (reason) {
    case LWS_CALLBACK_FILTER_PROTOCOL_CONNECTION:
      if (!self->authorized(wsi)) {
        lws_return_http_status(wsi, HTTP_STATUS_UNAUTHORIZED, "invalid api key");
        return -1;
      }
      {
        std::string error;
        if (!valid_handshake(wsi, self->config_.served_model_name, error)) {
          lws_return_http_status(wsi, HTTP_STATUS_BAD_REQUEST, error.c_str());
          return -1;
        }
      }
      return 0;
    case LWS_CALLBACK_HTTP:
      lws_return_http_status(wsi, HTTP_STATUS_NOT_FOUND, "realtime WebSocket only");
      return -1;
    case LWS_CALLBACK_ESTABLISHED:
      self->established(wsi, session);
      return 0;
    case LWS_CALLBACK_CLOSED:
      self->closed(wsi, session);
      return 0;
    case LWS_CALLBACK_RECEIVE:
      self->receive(wsi, session, in, len);
      return 0;
    case LWS_CALLBACK_SERVER_WRITEABLE: {
      if (!session || session->outbound.empty()) return 0;
      std::string message = std::move(session->outbound.front());
      session->outbound.pop_front();
      std::vector<unsigned char> buffer(LWS_PRE + message.size());
      std::memcpy(buffer.data() + LWS_PRE, message.data(), message.size());
      if (lws_write(wsi, buffer.data() + LWS_PRE, message.size(), LWS_WRITE_TEXT) <
          static_cast<int>(message.size())) {
        return -1;
      }
      if (!session->outbound.empty()) lws_callback_on_writable(wsi);
      if (session->close_after_write && session->outbound.empty()) {
        lws_close_reason(wsi, LWS_CLOSE_STATUS_NORMAL, nullptr, 0);
        return -1;
      }
      return 0;
    }
    default:
      return 0;
  }
}

}  // namespace

RealtimeApi::RealtimeApi(Coordinator& coordinator, ApiConfig config)
    : impl_(std::make_unique<Impl>(coordinator, std::move(config))) {}
RealtimeApi::~RealtimeApi() = default;
bool RealtimeApi::bind() { return impl_->bind(); }
bool RealtimeApi::listen() { return impl_->listen(); }
void RealtimeApi::stop() { impl_->stop(); }
int RealtimeApi::bound_port() const { return impl_->bound_port(); }

}  // namespace umminfer

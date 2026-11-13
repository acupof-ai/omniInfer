// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cctype>
#include <limits>
#include <map>
#include <mutex>
#include <thread>

#include "http.hpp"
#include "websocket.hpp"

namespace umminfer::client {
namespace {

using Clock = std::chrono::steady_clock;

uint32_t decode_utf8(std::string_view text, bool last, size_t* length = nullptr) {
  if (text.empty()) return 0xfffd;
  size_t at = 0;
  if (last) {
    at = text.size() - 1;
    size_t steps = 0;
    while (at > 0 && steps++ < 3 &&
           (static_cast<unsigned char>(text[at]) & 0xc0) == 0x80) --at;
  }
  const auto first = static_cast<unsigned char>(text[at]);
  size_t count = first < 0x80 ? 1 : (first & 0xe0) == 0xc0 ? 2
                                : (first & 0xf0) == 0xe0 ? 3
                                : (first & 0xf8) == 0xf0 ? 4 : 0;
  if (count == 0 || at + count > text.size()) return 0xfffd;
  uint32_t rune = first & ((1u << (7 - count)) - 1);
  for (size_t i = 1; i < count; ++i) {
    const auto next = static_cast<unsigned char>(text[at + i]);
    if ((next & 0xc0) != 0x80) return 0xfffd;
    rune = (rune << 6) | (next & 0x3f);
  }
  if (length) *length = count;
  return rune;
}

bool spaced(uint32_t rune) {
  static constexpr uint32_t unspaced[][2] = {
      {0x0e00, 0x0eff}, {0x1000, 0x109f}, {0x1780, 0x17ff},
      {0x2e80, 0x303f}, {0x3040, 0x30ff}, {0x3400, 0x9fff},
      {0xf900, 0xfaff}, {0xff00, 0xffef}, {0x20000, 0x2fa1f}};
  if (rune == 0xfffd || rune == 0x20 || rune == 0xa0 || rune == 0x3000 ||
      (rune >= 0x09 && rune <= 0x0d)) return false;
  for (const auto& range : unspaced) {
    if (rune >= range[0] && rune <= range[1]) return false;
  }
  return true;
}

std::string join_parts(const std::vector<std::string>& parts) {
  std::string out;
  for (std::string part : parts) {
    const size_t first = part.find_first_not_of(" \t\n\r\v\f");
    if (first == std::string::npos) continue;
    const size_t last = part.find_last_not_of(" \t\n\r\v\f");
    part = part.substr(first, last - first + 1);
    if (!out.empty() && spaced(decode_utf8(out, true)) && spaced(decode_utf8(part, false))) {
      out.push_back(' ');
    }
    out += part;
  }
  return out;
}

class Assembler {
 public:
  std::optional<std::string> apply(int index, int segment, std::string text, bool final) {
    if (index > 0) {
      if (index <= last_index_) return std::nullopt;
      last_index_ = index;
    }
    if (final) {
      finals_[segment] = std::move(text);
      if (has_provisional_ && provisional_segment_ == segment) {
        has_provisional_ = false;
        provisional_text_.clear();
      }
    } else if (!finals_.count(segment)) {
      provisional_segment_ = segment;
      provisional_text_ = std::move(text);
      has_provisional_ = true;
    } else {
      return std::nullopt;
    }
    return value();
  }

 private:
  std::string value() const {
    std::vector<std::string> parts;
    for (const auto& [id, text] : finals_) parts.push_back(text);
    if (has_provisional_) parts.push_back(provisional_text_);
    return join_parts(parts);
  }

  int last_index_ = 0;
  std::map<int, std::string> finals_;
  int provisional_segment_ = 0;
  std::string provisional_text_;
  bool has_provisional_ = false;
};

void apply_generation_metadata(const nlohmann::json& event, TranscribeResponse& response) {
  const auto metadata = event.find("generation_metadata");
  if (metadata == event.end() || metadata->is_null()) return;
  if (!metadata->is_object()) {
    throw Error(ErrorCode::kBadResponse, "invalid realtime generation metadata");
  }
  const auto count = metadata->find("generated_token_count");
  if (count != metadata->end() && !count->is_null()) {
    if (!count->is_number_integer() ||
        (count->is_number_unsigned() &&
         count->get<uint64_t>() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) ||
        (!count->is_number_unsigned() && count->get<int64_t>() < 0)) {
      throw Error(ErrorCode::kBadResponse, "invalid realtime generated token count");
    }
    response.generated_tokens = count->get<int64_t>();
  }
  const auto finish = metadata->find("finish_reason");
  if (finish != metadata->end() && !finish->is_null()) {
    if (!finish->is_string() || finish->get_ref<const std::string&>().empty()) {
      throw Error(ErrorCode::kBadResponse, "invalid realtime finish reason");
    }
    response.finish_reason = finish->get<std::string>();
    response.length_limited = response.finish_reason == "length";
  }
}

}  // namespace

struct RealtimeSession::Impl {
  Impl(const std::string& url, const RealtimeRequest& input, FinalPass pass,
       const std::string& api_key)
      : request(input),
        external_cancel(input.cancel),
        final_pass(std::move(pass)),
        socket(url, request.connect_timeout, &cancelled, external_cancel, api_key) {}
  ~Impl() { close(); }

  void start() {
    if (request.max_new_tokens < 0) {
      throw Error(ErrorCode::kConfig, "max_new_tokens must not be negative");
    }
    if (request.connect_timeout.count() < 0 || request.final_timeout.count() < 0) {
      throw Error(ErrorCode::kConfig, "realtime timeouts must not be negative");
    }
    if (request.cancel) {
      cancel_thread = std::thread([this] {
        std::unique_lock lock(mu);
        while (!cancelled.load() && !external_cancel->load(std::memory_order_relaxed)) {
          cv.wait_for(lock, std::chrono::milliseconds(20));
        }
        if (!cancelled.exchange(true)) socket.close();
        cv.notify_all();
      });
    }
    reader = std::thread([this] {
      try {
        socket.run([this](std::string_view message) { event(message); });
      } catch (...) {
        std::lock_guard lock(mu);
        if (!reader_error) reader_error = std::current_exception();
      }
      {
        std::lock_guard lock(mu);
        reader_done = true;
      }
      cv.notify_all();
    });
    nlohmann::json session = {{"input_audio_format", "pcm16"}, {"turn_detection", nullptr}};
    if (request.max_new_tokens > 0) session["max_new_tokens"] = request.max_new_tokens;
    if (!request.audio_layout.empty()) session["audio_layout"] = request.audio_layout;
    if (!request.language.empty()) session["language"] = request.language;
    if (!request.prompt.empty()) session["prompt"] = request.prompt;
    socket.send_text(nlohmann::json{{"type", "session.update"}, {"session", session}}.dump());
    std::unique_lock lock(mu);
    const auto deadline = Clock::now() + request.connect_timeout;
    if (!cv.wait_until(lock, deadline, [this] { return ready || reader_done || cancelled.load(); })) {
      throw Error(ErrorCode::kTimeout, "realtime session setup timeout");
    }
    if (cancelled) throw Error(ErrorCode::kCancelled, "request cancelled");
    if (!ready) {
      if (reader_error) std::rethrow_exception(reader_error);
      throw Error(ErrorCode::kBadResponse, "realtime session closed before it was accepted");
    }
    start_pings();
  }

  void event(std::string_view message) {
    nlohmann::json value;
    std::string type;
    try {
      value = nlohmann::json::parse(message);
      if (!value.is_object() || !value.contains("type") || !value["type"].is_string()) return;
      type = value["type"].get<std::string>();
      if (type == "transcription_session.updated") {
        {
          std::lock_guard lock(mu);
          ready = true;
        }
        cv.notify_all();
        return;
      }
      if (type == "transcription.completed") {
        if (!value.contains("text") || !value["text"].is_string()) return;
        TranscribeResponse response;
        response.text = value["text"].get<std::string>();
        response.language = request.language;
        apply_generation_metadata(value, response);
        {
          std::lock_guard lock(mu);
          live_response = std::move(response);
          completed = true;
        }
        cv.notify_all();
        socket.close();
        return;
      }
      if (type == "transcription.segment") {
        const int index = value.value("event_index", 0);
        const int segment = value.value("segment_id", 0);
        if (!value.contains("text") || !value["text"].is_string() ||
            !value.contains("is_final") || !value["is_final"].is_boolean()) return;
        auto text = assembler.apply(index, segment, value["text"].get<std::string>(),
                                    value["is_final"].get<bool>());
        if (!text || value["is_final"].get<bool>()) return;
        {
          std::lock_guard lock(mu);
          if (finished) return;
        }
        if (request.on_partial) request.on_partial(*text);
        return;
      }
      if (type == "error") {
        std::string code = "error";
        if (value.contains("error") && value["error"].is_object() &&
            value["error"].contains("code") && value["error"]["code"].is_string() &&
            !value["error"]["code"].get_ref<const std::string&>().empty()) {
          code = value["error"]["code"].get<std::string>();
        }
        std::string detail;
        if (value.contains("error") && value["error"].is_object() &&
            value["error"].contains("message") && value["error"]["message"].is_string()) {
          detail = value["error"]["message"].get<std::string>();
        }
        std::string text = "realtime transcription failed: " + code;
        if (!detail.empty() && detail != code) text += ": " + detail;
        throw Error(ErrorCode::kServer, text, true);
      }
    } catch (const nlohmann::json::exception&) {
      return;  // Voxt ignores unreadable realtime events.
    }
  }

  void send(std::string_view pcm) {
    if (pcm.empty()) return;
    if (pcm.size() % 2 != 0) throw Error(ErrorCode::kBadRequest, "PCM16 audio has an odd byte count");
    static const size_t max_append_pcm = [] {
      constexpr size_t max_message = 1u << 20;
      const size_t envelope = nlohmann::json{
          {"type", "input_audio_buffer.append"}, {"audio", ""}}.dump().size();
      return (((max_message - envelope) / 4) * 3) & ~size_t{1};
    }();
    std::lock_guard send_lock(send_mu);
    {
      std::lock_guard lock(mu);
      if (finished || reader_done && completed) {
        throw Error(ErrorCode::kBadRequest, "the realtime stream has ended");
      }
      if (cancelled) throw Error(ErrorCode::kCancelled, "request cancelled");
      if (pcm.size() > (1u << 30) - std::min<size_t>(audio.size(), 1u << 30)) {
        throw Error(ErrorCode::kBadRequest, "realtime audio exceeds 1 GiB");
      }
      audio.append(pcm);
      if (reader_done) return;  // Preview failure does not lose the Final input.
    }
    try {
      for (size_t offset = 0; offset < pcm.size();) {
        const size_t count = std::min(max_append_pcm, pcm.size() - offset);
        const std::string message = nlohmann::json{
            {"type", "input_audio_buffer.append"},
            {"audio", detail::base64_encode(pcm.substr(offset, count))}}.dump();
        socket.send_text(message);
        offset += count;
      }
    } catch (...) {
      if (cancelled || (request.cancel && request.cancel->load())) throw;
      socket.close();  // Keep recording for the Final pass.
    }
  }

  TranscribeResponse final(const std::string& pcm) {
    TranscribeRequest next;
    next.pcm = pcm;
    next.language = request.language;
    next.prompt = request.prompt;
    next.audio_layout = request.audio_layout;
    next.max_new_tokens = request.max_new_tokens;
    next.cancel = &cancelled;
    return final_pass(next);
  }

  void stop_transport() {
    socket.close();
    if (reader.joinable() && reader.get_id() != std::this_thread::get_id()) reader.join();
  }

  TranscribeResponse finish_impl() {
    std::string pcm;
    {
      std::lock_guard send_lock(send_mu);
      std::lock_guard lock(mu);
      if (finished) throw Error(ErrorCode::kBadRequest, "the realtime stream has ended");
      finished = true;
      pcm = audio;
    }
    if (request.live_result) {
      try {
        socket.send_text(nlohmann::json{{"type", "transcription.done"}}.dump());
        std::unique_lock lock(mu);
        const auto deadline = Clock::now() + request.final_timeout;
        cv.wait_until(lock, deadline, [this] { return completed || reader_done || cancelled.load(); });
        if (cancelled) throw Error(ErrorCode::kCancelled, "request cancelled");
        if (completed) {
          TranscribeResponse response = live_response;
          lock.unlock();
          stop_transport();
          return response;
        }
      } catch (...) {
        if (cancelled || (request.cancel && request.cancel->load())) throw;
      }
    }
    stop_transport();
    if (cancelled) throw Error(ErrorCode::kCancelled, "request cancelled");
    return final(pcm);
  }

  void start_pings() {
    if (request.ping_interval.count() <= 0) return;
    ping_thread = std::thread([this] {
      constexpr auto kStep = std::chrono::milliseconds{10};
      auto next = Clock::now() + request.ping_interval;
      while (!ping_stop.load() && !cancelled.load()) {
        if (Clock::now() >= next) {
          socket.ping();
          next = Clock::now() + request.ping_interval;
          std::lock_guard lock(mu);
          if (finished || reader_done || completed) return;
        }
        std::this_thread::sleep_for(kStep);
      }
    });
  }

  void close() {
    ping_stop = true;
    if (ping_thread.joinable() &&
        ping_thread.get_id() != std::this_thread::get_id()) {
      ping_thread.join();
    }
    if (!closed.exchange(true)) {
      cancelled = true;
      socket.close();
      cv.notify_all();
    }
    if (reader.joinable() && reader.get_id() != std::this_thread::get_id()) reader.join();
    if (cancel_thread.joinable() && cancel_thread.get_id() != std::this_thread::get_id()) {
      cancel_thread.join();
    }
  }

  RealtimeRequest request;
  CancelFlag external_cancel;
  FinalPass final_pass;
  std::atomic<bool> cancelled{false};
  detail::WebSocket socket;
  std::thread reader;
  std::thread cancel_thread;
  std::thread ping_thread;
  std::atomic<bool> ping_stop{false};
  std::mutex mu;
  std::mutex send_mu;
  std::condition_variable cv;
  Assembler assembler;
  std::string audio;
  TranscribeResponse live_response;
  std::once_flag finish_once;
  TranscribeResponse result;
  std::exception_ptr finish_error;
  std::exception_ptr reader_error;
  bool ready = false;
  bool reader_done = false;
  bool completed = false;
  bool finished = false;
  std::atomic<bool> closed{false};
};

RealtimeSession::RealtimeSession(std::string url, const RealtimeRequest& request,
                                 FinalPass final_pass, std::string api_key)
    : impl_(std::make_unique<Impl>(url, request, std::move(final_pass), api_key)) {
  impl_->start();
}
RealtimeSession::~RealtimeSession() { close(); }

void RealtimeSession::send(std::string_view pcm) { impl_->send(pcm); }

TranscribeResponse RealtimeSession::finish() {
  std::call_once(impl_->finish_once, [this] {
    try {
      impl_->result = impl_->finish_impl();
    } catch (...) {
      impl_->finish_error = std::current_exception();
    }
    impl_->close();
  });
  if (impl_->finish_error) std::rethrow_exception(impl_->finish_error);
  return impl_->result;
}

void RealtimeSession::close() {
  if (impl_) impl_->close();
}

RealtimeSession::PingStats RealtimeSession::ping_stats() const {
  return {impl_->socket.pings_sent(), impl_->socket.pongs_received()};
}

std::unique_ptr<RealtimeSession> ASRClient::realtime(const RealtimeRequest& input) const {
  RealtimeRequest request = input;
  if (request.max_new_tokens < 0) {
    throw Error(ErrorCode::kConfig, "max_new_tokens must not be negative");
  }
  if (request.connect_timeout.count() < 0 || request.final_timeout.count() < 0 ||
      config_.connect_timeout.count() < 0 || config_.stall_timeout.count() < 0) {
    throw Error(ErrorCode::kConfig, "realtime timeouts must not be negative");
  }
  if (request.connect_timeout.count() == 0) request.connect_timeout = config_.connect_timeout;
  if (request.final_timeout.count() == 0) request.final_timeout = Milliseconds{20000};
  if (config_.model.empty()) throw Error(ErrorCode::kConfig, "ASR model is required");
  std::string url = base_url_;
  if (url.rfind("http://", 0) == 0) {
    url.replace(0, 4, "ws");
  } else {
    throw Error(ErrorCode::kConfig, "realtime ASR requires loopback HTTP");
  }
  url += "/v1/realtime?intent=transcription&model=" + detail::url_encode(config_.model);
  ASRConfig final_config = config_;
  if (!final_config.final_base_url.empty()) {
    final_config.base_url =
        detail::validate_base_url(final_config.final_base_url, true);
    final_config.final_base_url.clear();
  }
  return std::make_unique<RealtimeSession>(
      url, request, [final_config](const TranscribeRequest& next) {
        return ASRClient(final_config).transcribe(next);
      }, config_.api_key);
}

}  // namespace umminfer::client

// SPDX-License-Identifier: Apache-2.0
#include "http.hpp"
#include "umminfer/audio/audio.hpp"

#include <cctype>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace umminfer::client {
namespace {

[[noreturn]] void unreadable_event() {
  throw Error(ErrorCode::kBadResponse, "unreadable transcription event");
}

[[noreturn]] void incomplete_transcription() {
  throw Error(ErrorCode::kServer, "transcription ended before its result", true);
}

std::string_view trim_space(std::string_view value) {
  while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) {
    value.remove_prefix(1);
  }
  while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
    value.remove_suffix(1);
  }
  return value;
}

}  // namespace

ASRClient::ASRClient(ASRConfig config)
    : config_(std::move(config)),
      base_url_(detail::validate_base_url(config_.base_url, true)) {
  if (config_.model.empty()) throw Error(ErrorCode::kConfig, "ASR model is required");
}

TranscribeResponse ASRClient::transcribe(const TranscribeRequest& request) const {
  detail::check_cancelled(request.cancel);
  if (request.max_new_tokens < 0) {
    throw Error(ErrorCode::kConfig, "max_new_tokens must not be negative");
  }
  if (request.timeout.count() < 0 || request.stall_timeout.count() < 0) {
    throw Error(ErrorCode::kConfig, "request timeouts must not be negative");
  }
  std::string wav = wav_from_pcm16(request.pcm);

  detail::HttpRequest http;
  auto add_field = [&](std::string name, std::string value) {
    http.mime_parts.push_back({std::move(name), std::move(value), {}, {}});
  };

  add_field("model", config_.model);
  add_field("stream", "true");
  add_field("response_format", "json");
  if (!request.language.empty()) add_field("language", request.language);
  if (!request.prompt.empty()) add_field("prompt", request.prompt);
  if (request.max_new_tokens > 0) {
    add_field("max_new_tokens", std::to_string(request.max_new_tokens));
  }
  // Ask for the stage's real token count and finish reason. The server rejects
  // the request when the stage cannot report them, so the client never presents
  // a missing count as zero.
  add_field("include_generation_metadata", "true");
  if (!request.audio_layout.empty()) add_field("audio_layout", request.audio_layout);
  http.mime_parts.push_back({"file", std::move(wav), "audio.wav", "application/octet-stream"});

  detail::Progress progress;
  progress.activity = request.stall_timeout.count() > 0 ? request.stall_timeout
                                                        : config_.stall_timeout;
  detail::SSEParser parser;
  std::optional<TranscribeResponse> done;
  bool saw_done_marker = false;

  auto on_event = [&](std::string_view raw_payload) -> bool {
    const std::string_view payload = trim_space(raw_payload);
    if (payload == "[DONE]") {
      if (!done) incomplete_transcription();
      saw_done_marker = true;
      return false;
    }

    const auto event = nlohmann::json::parse(payload, nullptr, false);
    if (event.is_discarded() || !event.is_object()) unreadable_event();

    if (event.contains("type") && !event.at("type").is_null() &&
        !event.at("type").is_string()) {
      unreadable_event();
    }
    const std::string type = event.contains("type") && event.at("type").is_string()
                                 ? event.at("type").get<std::string>()
                                 : std::string{};
    if (type.empty()) unreadable_event();

    if (event.contains("text") && !event.at("text").is_null() &&
        !event.at("text").is_string()) {
      unreadable_event();
    }

    const nlohmann::json* metadata = nullptr;
    if (event.contains("generation_metadata") &&
        !event.at("generation_metadata").is_null()) {
      metadata = &event.at("generation_metadata");
      if (!metadata->is_object()) unreadable_event();
      if (metadata->contains("generated_token_count") &&
          !metadata->at("generated_token_count").is_null()) {
        const auto& count = metadata->at("generated_token_count");
        if (!count.is_number_integer()) unreadable_event();
        if (count.is_number_unsigned() &&
            count.get<uint64_t>() >
                static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
          unreadable_event();
        }
      }
      if (metadata->contains("language") && !metadata->at("language").is_null() &&
          !metadata->at("language").is_string()) {
        unreadable_event();
      }
      if (metadata->contains("finish_reason") && !metadata->at("finish_reason").is_null() &&
          !metadata->at("finish_reason").is_string()) {
        unreadable_event();
      }
    }

    const nlohmann::json* error = nullptr;
    if (event.contains("error") && !event.at("error").is_null()) {
      error = &event.at("error");
      if (!error->is_object()) unreadable_event();
      for (const char* field : {"type", "code"}) {
        if (error->contains(field) && !error->at(field).is_null() &&
            !error->at(field).is_string()) {
          unreadable_event();
        }
      }
    }

    if (type == "error") {
      std::string code = "error";
      if (error && error->contains("code") && error->at("code").is_string() &&
          !error->at("code").get_ref<const std::string&>().empty()) {
        code = error->at("code").get<std::string>();
      }
      throw Error(ErrorCode::kServer, "transcription failed: " + code);
    }
    if (type != "transcript.text.done") return true;

    TranscribeResponse result;
    if (event.contains("text") && event.at("text").is_string()) {
      result.text = event.at("text").get<std::string>();
    }
    if (metadata) {
      const auto count = metadata->find("generated_token_count");
      const auto finish = metadata->find("finish_reason");
      if (count == metadata->end() || count->is_null() || finish == metadata->end() ||
          finish->is_null() || !finish->is_string() ||
          finish->get_ref<const std::string&>().empty()) {
        throw Error(ErrorCode::kBadResponse, "incomplete generation metadata");
      }
      result.generated_tokens = count->get<int64_t>();
      result.finish_reason = finish->get<std::string>();
      result.length_limited = result.finish_reason == "length";
      const auto language = metadata->find("language");
      if (language != metadata->end() && language->is_string()) {
        result.language = language->get<std::string>();
      }
    }
    done = std::move(result);
    return true;
  };

  http.url = base_url_ + "/v1/audio/transcriptions";
  http.bearer = config_.api_key;
  http.timeout = request.timeout;
  http.connect_timeout = config_.connect_timeout;
  http.cancel = request.cancel;
  http.progress = &progress;
  http.no_proxy = true;
  http.on_body = [&](std::string_view bytes) { return parser.feed(bytes, on_event); };
  detail::perform(http);

  if (!saw_done_marker) parser.finish(on_event);
  if (!done) incomplete_transcription();
  return std::move(*done);
}

std::string wav_from_pcm16(std::string_view pcm) {
  try {
    return audio::wav_from_pcm16(pcm);
  } catch (const std::invalid_argument& error) {
    throw Error(ErrorCode::kBadRequest, error.what());
  }
}

std::string pcm16_from_wav(std::string_view wav) {
  try {
    return audio::pcm16_from_wav(wav);
  } catch (const std::invalid_argument& error) {
    throw Error(ErrorCode::kBadRequest, error.what());
  }
}

}  // namespace umminfer::client

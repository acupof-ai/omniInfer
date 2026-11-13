// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace umminfer::client {

using Milliseconds = std::chrono::milliseconds;
using CancelFlag = const std::atomic<bool>*;

enum class ErrorCode {
  kConfig,
  kAuthentication,
  kRateLimited,
  kBadRequest,
  kServer,
  kNetwork,
  kTimeout,
  kCancelled,
  kBadResponse,
  kContentFilter,
  kEmpty,
};

class Error : public std::runtime_error {
 public:
  Error(ErrorCode code, std::string detail, bool retryable = false, int status = 0,
        Milliseconds retry_after = Milliseconds{0});

  ErrorCode code() const noexcept { return code_; }
  bool retryable() const noexcept { return retryable_; }
  int status() const noexcept { return status_; }
  Milliseconds retry_after() const noexcept { return retry_after_; }

 private:
  ErrorCode code_;
  bool retryable_;
  int status_;
  Milliseconds retry_after_;
};

// One part of a multimodal message. The kind selects which fields are read.
struct ContentPart {
  enum class Kind { kText, kImageUrl, kInputAudio, kVideoUrl, kVideo };

  Kind kind = Kind::kText;
  std::string text;                  // kText
  std::string url;                   // kImageUrl, kVideoUrl
  std::string audio;                 // kInputAudio, base64
  std::string audio_format;          // kInputAudio, e.g. wav or mp3
  std::vector<std::string> video_urls;  // kVideo
  std::optional<double> fps;         // kVideoUrl, 0.1 to 10
};

struct Message {
  std::string role;
  std::string content;
  // When non-empty, sent as a content part array and content must be empty.
  std::vector<ContentPart> parts;
};

struct Usage {
  int64_t prompt_tokens = 0;
  int64_t completion_tokens = 0;
  int64_t total_tokens = 0;
};

struct OpenAIConfig {
  std::string base_url = "https://api.openai.com/v1";
  std::string api_key;
  std::string model;
  Milliseconds timeout{120000};
  Milliseconds connect_timeout{10000};
  int max_attempts = 3;
  Milliseconds retry_base{500};
  Milliseconds retry_max{20000};
  Milliseconds max_retry_after{60000};
};

// Requested speech output. The route supports wav only and requires stream.
struct AudioOutput {
  std::string voice;
  std::string format = "wav";
};

struct ChatRequest {
  std::string model;
  std::vector<Message> messages;
  // "text" and/or "audio". Empty keeps the server default.
  std::vector<std::string> modalities;
  std::optional<AudioOutput> audio;
  std::optional<double> temperature;
  std::optional<double> top_p;
  int max_tokens = 0;
  std::vector<std::string> stop;
  std::optional<int64_t> seed;
  nlohmann::json response_format;
  nlohmann::json extra_body = nlohmann::json::object();
  bool stream = false;
  std::function<void(const std::string&)> on_delta;
  Milliseconds timeout{0};
  Milliseconds first_token_timeout{0};
  Milliseconds stall_timeout{0};
  CancelFlag cancel = nullptr;
};

struct ChatResponse {
  std::string text;
  // Decoded WAV bytes when the request asked for audio output.
  std::string audio;
  std::string audio_transcript;
  std::string finish_reason;
  std::string model;
  Usage usage;
  int attempts = 0;
};

// JSON image generation. The server answers with inline b64_json only.
struct ImageRequest {
  std::string model;
  std::string prompt;
  int n = 0;
  std::string size;
  // Documented generation knobs (seed, negative_prompt, steps, ...). The
  // reserved keys model, prompt, response_format and stream are rejected.
  nlohmann::json extra_body = nlohmann::json::object();
  Milliseconds timeout{0};
  CancelFlag cancel = nullptr;
};

struct ImagePart {
  std::string data;
  std::string filename;
  std::string content_type;
};

// Ordered multipart image edit. Images are sent first in vector order, then
// `fields` in vector order, then model. Only `url` may repeat.
struct ImageEditRequest {
  std::string model;
  std::string prompt;
  std::vector<ImagePart> images;
  std::vector<std::pair<std::string, std::string>> fields;
  Milliseconds timeout{0};
  CancelFlag cancel = nullptr;
};

struct ImageResponse {
  // Decoded image bytes in response order. Each image is bounded.
  std::vector<std::string> images;
};

struct VideoRequest {
  std::string model;
  std::string prompt;
  // Documented creation fields (task, conditions, size, seconds, ...). The
  // reserved keys model and prompt are rejected.
  nlohmann::json body = nlohmann::json::object();
  Milliseconds timeout{0};
  CancelFlag cancel = nullptr;
};

struct VideoJob {
  std::string id;
  std::string status;
  // The upstream JSON object unchanged.
  nlohmann::json raw;
};

class OpenAIClient {
 public:
  explicit OpenAIClient(OpenAIConfig config);
  ChatResponse chat(const ChatRequest& request) const;

  // Creation calls are sent once. A retry could submit a second job or a
  // second billable generation, so none of these methods retries.
  ImageResponse generate_image(const ImageRequest& request) const;
  ImageResponse edit_image(const ImageEditRequest& request) const;
  VideoJob create_video(const VideoRequest& request) const;

  // Polling and deletion are also sent once. A caller that wants to poll owns
  // the retry loop and its interval.
  VideoJob video_status(std::string_view video_id, CancelFlag cancel = nullptr) const;
  nlohmann::json delete_video(std::string_view video_id, CancelFlag cancel = nullptr) const;
  // Streams the MP4 to a new file. The bytes are bounded, written to a private
  // part file and published atomically; an existing path is never replaced.
  // `variant` selects one output of a multi-output job: it is sent as the
  // zero-based `?variant=N` query, and no query is sent when it is unset.
  void download_video(std::string_view video_id, const std::string& output_path,
                      std::optional<int> variant = std::nullopt,
                      CancelFlag cancel = nullptr) const;

 private:
  OpenAIConfig config_;
  std::string base_url_;
  std::string endpoint_;
};

struct ASRConfig {
  std::string base_url;
  // Optional separate HTTP endpoint for the Final pass after realtime preview.
  std::string final_base_url;
  std::string model;
  std::string api_key;
  Milliseconds connect_timeout{10000};
  Milliseconds stall_timeout{30000};
};

struct TranscribeRequest {
  std::string pcm;
  std::string language;
  std::string prompt;
  std::string audio_layout;
  int max_new_tokens = 0;
  Milliseconds timeout{0};
  Milliseconds stall_timeout{0};
  CancelFlag cancel = nullptr;
};

struct TranscribeResponse {
  std::string text;
  std::string language;
  std::string finish_reason;
  // Unknown when the stage omitted the corresponding generation metadata.
  std::optional<int64_t> generated_tokens;
  std::optional<bool> length_limited;
};

struct RealtimeRequest {
  std::string language;
  std::string prompt;
  std::string audio_layout;
  int max_new_tokens = 0;
  bool live_result = false;
  Milliseconds connect_timeout{0};
  Milliseconds final_timeout{20000};
  // Sends WebSocket PING control frames at this interval while the session is
  // active. Zero disables them. Pongs are counted, not injected by a test.
  Milliseconds ping_interval{0};
  std::function<void(const std::string&)> on_partial;
  CancelFlag cancel = nullptr;
};

class RealtimeSession;

class ASRClient {
 public:
  explicit ASRClient(ASRConfig config);
  TranscribeResponse transcribe(const TranscribeRequest& request) const;
  std::unique_ptr<RealtimeSession> realtime(const RealtimeRequest& request) const;

 private:
  ASRConfig config_;
  std::string base_url_;
};

class RealtimeSession {
 public:
  using FinalPass = std::function<TranscribeResponse(const TranscribeRequest&)>;
  RealtimeSession(std::string url, const RealtimeRequest& request, FinalPass final_pass,
                  std::string api_key = {});
  ~RealtimeSession();
  RealtimeSession(const RealtimeSession&) = delete;
  RealtimeSession& operator=(const RealtimeSession&) = delete;

  struct PingStats {
    uint64_t sent = 0;
    uint64_t pongs = 0;
  };

  void send(std::string_view pcm);
  TranscribeResponse finish();
  void close();
  PingStats ping_stats() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

std::string wav_from_pcm16(std::string_view pcm);
std::string pcm16_from_wav(std::string_view wav);

}  // namespace umminfer::client

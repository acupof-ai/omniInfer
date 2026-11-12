// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <curl/curl.h>

#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "umminfer/client/client.hpp"

namespace umminfer::client::detail {

struct Progress {
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point last_token = start;
  std::chrono::steady_clock::time_point last_data = start;
  Milliseconds first_token{0};
  Milliseconds stall{0};
  Milliseconds activity{0};
  bool saw_token = false;
  bool saw_data = false;
  std::string expired;

  void token();
  void data();
};

enum class HttpMethod { kPost, kGet, kDelete };

// Streams one multipart part body from a caller callback. `size` is the exact
// byte count the callback must produce. pump() never asks for more than
// `max_bytes` and throws Error(kBadRequest) when the callback fails or ends
// early. The callback runs during the transfer, so no file is read up front.
class MimeSource {
 public:
  using Read = std::function<size_t(char* buffer, size_t max_bytes)>;

  MimeSource(Read read, uint64_t size);
  size_t pump(char* buffer, size_t max_bytes);
  uint64_t size() const { return size_; }
  uint64_t consumed() const { return consumed_; }

 private:
  Read read_;
  uint64_t size_ = 0;
  uint64_t consumed_ = 0;
};

struct MimePart {
  std::string name;
  std::string data;
  std::string filename;
  std::string content_type;
  // When `read` is set, the body is pumped from it instead of `data` and
  // `size` must be the exact byte count. Parts are sent in vector order.
  MimeSource::Read read;
  uint64_t size = 0;
};

struct HttpRequest {
  std::string url;
  // The default stays POST so existing chat and ASR calls are unchanged.
  HttpMethod method = HttpMethod::kPost;
  std::string body;
  std::string bearer;
  // Replaces the default Accept header. Use a binary media type for downloads.
  std::string accept;
  // When set, a 2xx response must carry this media type or the call fails.
  std::string require_content_type;
  std::multimap<std::string, std::string> headers;
  std::vector<MimePart> mime_parts;
  Milliseconds timeout{0};
  Milliseconds connect_timeout{10000};
  CancelFlag cancel = nullptr;
  Progress* progress = nullptr;
  bool no_proxy = false;
  size_t body_limit = 32u << 20;
  // Cumulative byte cap for a callback-delivered response. 0 leaves the limit
  // to the callback.
  size_t stream_limit = 0;
  std::function<bool(std::string_view)> on_body;
};

struct HttpResponse {
  long status = 0;
  std::string body;
  std::map<std::string, std::string> headers;
};

void initialize_curl();
std::string validate_base_url(std::string_view url, bool loopback_only);
std::string error_detail(std::string_view raw);
std::string redact(std::string_view text, std::string_view secret);
std::string clip(std::string_view text, size_t limit);
Milliseconds retry_after(const std::map<std::string, std::string>& headers);

// Rejects requests the transport cannot honor. Network-free; perform() calls it
// before opening a connection.
void validate_request(const HttpRequest& request);
// Case-insensitive media-type comparison that ignores parameters.
bool content_type_matches(std::string_view actual, std::string_view expected);
HttpResponse perform(const HttpRequest& request);
[[noreturn]] void throw_http(long status, std::string_view body,
                             const std::map<std::string, std::string>& headers,
                             std::string_view secret = {});
void check_cancelled(CancelFlag cancel);

// Reusable file writes that never expose partial data at the final path.
// Data goes to a private sibling ".<name>.part.<pid>.<counter>" opened with
// O_EXCL. publish() fsyncs and hard-links it to the final path; the link fails
// when the final path already exists. The destructor unlinks an unpublished
// part. Both calls throw std::runtime_error on I/O failure.
class PartWriter {
 public:
  explicit PartWriter(std::string final_path);
  ~PartWriter();
  PartWriter(const PartWriter&) = delete;
  PartWriter& operator=(const PartWriter&) = delete;

  void write(std::string_view bytes);
  void publish();
  const std::string& part_path() const { return part_; }

 private:
  std::string final_;
  std::string part_;
  int fd_ = -1;
  bool published_ = false;
};

class SSEParser {
 public:
  using Event = std::function<bool(std::string_view)>;
  bool feed(std::string_view bytes, const Event& event);
  bool finish(const Event& event);
  bool stopped() const { return stopped_; }

 private:
  bool dispatch(const Event& event);
  std::string buffer_;
  std::string data_;
  bool has_data_ = false;
  bool stopped_ = false;
};

class CurlHandle {
 public:
  CurlHandle();
  ~CurlHandle();
  CURL* get() const { return handle_; }

 private:
  CURL* handle_;
};

std::string base64_encode(std::string_view bytes);
// Strict base64: rejects non-alphabet characters, illegal padding, non-zero
// trailing bits, and output larger than `max_bytes`.
std::string base64_decode(std::string_view text, size_t max_bytes);
std::string url_encode(std::string_view text);

}  // namespace umminfer::client::detail

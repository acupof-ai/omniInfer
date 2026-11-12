// SPDX-License-Identifier: Apache-2.0
#include "http.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace umminfer::client {

Error::Error(ErrorCode code, std::string detail, bool retryable, int status,
             Milliseconds retry_after)
    : std::runtime_error(std::move(detail)),
      code_(code),
      retryable_(retryable),
      status_(status),
      retry_after_(retry_after) {}

namespace detail {
// Defined below; declared here so the streaming callback can check the header
// before the first body byte reaches on_body.
bool content_type_matches(std::string_view actual, std::string_view expected);
namespace {

using Clock = std::chrono::steady_clock;

std::string lower(std::string_view value) {
  std::string out(value);
  std::transform(out.begin(), out.end(), out.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return out;
}

std::string trim(std::string_view value) {
  while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) {
    value.remove_prefix(1);
  }
  while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
    value.remove_suffix(1);
  }
  return std::string(value);
}

struct MimeStream {
  MimeStream(MimeSource::Read read, uint64_t size) : source(std::move(read), size) {}
  MimeSource source;
  std::exception_ptr error;
};

struct Transfer {
  const HttpRequest& request;
  CURL* curl = nullptr;
  HttpResponse response;
  std::exception_ptr error;
  bool stopped = false;
  bool body_started = false;
  uint64_t received = 0;
};

int progress(void* opaque, curl_off_t, curl_off_t, curl_off_t, curl_off_t);

size_t mime_read(char* buffer, size_t size, size_t count, void* opaque) {
  auto* stream = static_cast<MimeStream*>(opaque);
  if (size != 0 && count > std::numeric_limits<size_t>::max() / size) {
    return CURL_READFUNC_ABORT;
  }
  const size_t want = size * count;
  if (want == 0) return 0;
  try {
    return stream->source.pump(buffer, want);
  } catch (...) {
    stream->error = std::current_exception();
    return CURL_READFUNC_ABORT;
  }
}

void mime_free(void* opaque) { delete static_cast<MimeStream*>(opaque); }

bool parse_u64(std::string_view text, uint64_t& out) {
  const std::string value = trim(text);
  if (value.empty()) return false;
  uint64_t result = 0;
  for (const char c : value) {
    if (c < '0' || c > '9') return false;
    const uint64_t digit = static_cast<uint64_t>(c - '0');
    if (result > (std::numeric_limits<uint64_t>::max() - digit) / 10) return false;
    result = result * 10 + digit;
  }
  out = result;
  return true;
}

// libcurl reports the body length after decompression. Comparing it with
// Content-Length is only valid when the body is neither compressed nor
// chunked, so both cases are skipped.
void check_content_length(const Transfer& transfer) {
  const auto length = transfer.response.headers.find("content-length");
  if (length == transfer.response.headers.end()) return;
  const auto encoding = transfer.response.headers.find("content-encoding");
  if (encoding != transfer.response.headers.end()) {
    const std::string value = lower(trim(encoding->second));
    if (!value.empty() && value != "identity") return;
  }
  const auto chunked = transfer.response.headers.find("transfer-encoding");
  if (chunked != transfer.response.headers.end() &&
      lower(chunked->second).find("chunked") != std::string::npos) {
    return;
  }
  uint64_t declared = 0;
  if (!parse_u64(length->second, declared)) {
    throw Error(ErrorCode::kBadResponse, "invalid Content-Length");
  }
  if (declared != transfer.received) {
    throw Error(ErrorCode::kBadResponse,
                "response length does not match Content-Length");
  }
}

std::string directory_of(const std::string& path) {
  const size_t slash = path.find_last_of('/');
  if (slash == std::string::npos) return ".";
  return slash == 0 ? std::string("/") : path.substr(0, slash);
}

std::string basename_of(const std::string& path) {
  const size_t slash = path.find_last_of('/');
  return slash == std::string::npos ? path : path.substr(slash + 1);
}

std::atomic<uint64_t> part_counter{0};

size_t write_body(char* data, size_t size, size_t count, void* opaque) {
  auto& transfer = *static_cast<Transfer*>(opaque);
  if (size != 0 && count > std::numeric_limits<size_t>::max() / size) {
    transfer.error = std::make_exception_ptr(
        Error(ErrorCode::kBadResponse, "response chunk size overflow"));
    return 0;
  }
  const size_t bytes = size * count;
  try {
    if (progress(&transfer, 0, 0, 0, 0) != 0) return 0;
    if (transfer.request.progress) transfer.request.progress->data();
    if (transfer.received > std::numeric_limits<uint64_t>::max() - bytes) {
      throw Error(ErrorCode::kBadResponse, "response size overflow");
    }
    transfer.received += bytes;
    long status = 0;
    curl_easy_getinfo(transfer.curl, CURLINFO_RESPONSE_CODE, &status);
    if (status / 100 == 2 && transfer.request.on_body) {
      // The content type must hold before the first byte reaches the callback;
      // otherwise a mis-typed stream would already be committed to the client.
      if (!transfer.body_started) {
        transfer.body_started = true;
        if (!transfer.request.require_content_type.empty()) {
          const auto type = transfer.response.headers.find("content-type");
          if (type == transfer.response.headers.end() ||
              !content_type_matches(type->second,
                                    transfer.request.require_content_type)) {
            throw Error(ErrorCode::kBadResponse,
                        "unexpected response content type");
          }
        }
      }
      if (transfer.request.stream_limit > 0 &&
          transfer.received > transfer.request.stream_limit) {
        throw Error(ErrorCode::kBadResponse,
                    "response stream exceeds its size limit");
      }
      if (!transfer.request.on_body(std::string_view(data, bytes))) {
        transfer.stopped = true;
        return 0;
      }
    } else {
      const size_t limit = status / 100 == 2
                               ? transfer.request.body_limit
                               : std::min<size_t>(transfer.request.body_limit, 64u << 10);
      if (bytes > limit - std::min(limit, transfer.response.body.size())) {
        throw Error(ErrorCode::kBadResponse, "response body exceeds its size limit");
      }
      transfer.response.body.append(data, bytes);
    }
  } catch (...) {
    transfer.error = std::current_exception();
    return 0;
  }
  return bytes;
}

size_t write_header(char* data, size_t size, size_t count, void* opaque) {
  auto& transfer = *static_cast<Transfer*>(opaque);
  const size_t bytes = size * count;
  std::string_view line(data, bytes);
  const size_t colon = line.find(':');
  if (colon == std::string_view::npos) return bytes;
  const std::string name = lower(trim(line.substr(0, colon)));
  const std::string value = trim(line.substr(colon + 1));
  // Bound what a remote endpoint can make this map hold.
  if (!name.empty() && name.size() + value.size() <= 8192 &&
      transfer.response.headers.size() < 200) {
    transfer.response.headers[name] = value;
  }
  return bytes;
}

int progress(void* opaque, curl_off_t, curl_off_t, curl_off_t, curl_off_t) {
  auto& transfer = *static_cast<Transfer*>(opaque);
  if (transfer.request.cancel && transfer.request.cancel->load(std::memory_order_relaxed)) {
    return 1;
  }
  Progress* watch = transfer.request.progress;
  if (!watch) return 0;
  const auto now = Clock::now();
  if (watch->first_token.count() > 0 && !watch->saw_token &&
      now - watch->start >= watch->first_token) {
    watch->expired = "first_token";
    return 1;
  }
  if (watch->stall.count() > 0 && watch->saw_token &&
      now - watch->last_token >= watch->stall) {
    watch->expired = "stall";
    return 1;
  }
  if (watch->activity.count() > 0 && now - watch->last_data >= watch->activity) {
    watch->expired = "activity";
    return 1;
  }
  return 0;
}

Milliseconds parse_delay(std::string_view text, bool milliseconds) {
  const std::string value = trim(text);
  if (value.empty()) return Milliseconds{0};
  char* end = nullptr;
  const long double number = std::strtold(value.c_str(), &end);
  if (end != value.c_str() + value.size() || !std::isfinite(number) || number <= 0) {
    return Milliseconds{0};
  }
  const long double units = number * (milliseconds ? 1.0L : 1000.0L);
  if (units >= static_cast<long double>(Milliseconds::max().count())) {
    return Milliseconds::max();
  }
  return Milliseconds{static_cast<int64_t>(units)};
}

std::string curl_part(CURLU* url, CURLUPart part) {
  char* raw = nullptr;
  if (curl_url_get(url, part, &raw, 0) != CURLUE_OK || !raw) return {};
  std::string out(raw);
  curl_free(raw);
  return out;
}

// A textual prefix is not a loopback address: "127.example.com" is a DNS name.
// Only a complete dotted-quad IPv4 in 127.0.0.0/8, "::1", and "localhost" count.
bool loopback_host(std::string_view host) {
  // libcurl reports an IPv6 literal with its brackets.
  if (host == "localhost" || host == "::1" || host == "[::1]") return true;
  unsigned octet = 0;
  int dots = 0;
  bool digits = false;
  for (const char ch : host) {
    if (ch == '.') {
      if (!digits) return false;
      if (dots == 0 && octet != 127) return false;
      ++dots;
      octet = 0;
      digits = false;
    } else if (ch >= '0' && ch <= '9') {
      digits = true;
      octet = octet * 10 + static_cast<unsigned>(ch - '0');
      if (octet > 255) return false;
    } else {
      return false;
    }
  }
  return digits && dots == 3;
}

}  // namespace

MimeSource::MimeSource(Read read, uint64_t size)
    : read_(std::move(read)), size_(size) {
  if (!read_) throw std::invalid_argument("multipart source needs a read callback");
}

size_t MimeSource::pump(char* buffer, size_t max_bytes) {
  if (consumed_ == size_) return 0;
  if (max_bytes == 0) return 0;
  const uint64_t remaining = size_ - consumed_;
  const size_t want = static_cast<size_t>(std::min<uint64_t>(remaining, max_bytes));
  const size_t got = read_(buffer, want);
  if (got == 0) {
    throw Error(ErrorCode::kBadRequest,
                "multipart file ended before its declared size");
  }
  if (got > want) {
    throw Error(ErrorCode::kBadRequest,
                "multipart file returned more bytes than requested");
  }
  consumed_ += got;
  return got;
}

void validate_request(const HttpRequest& request) {
  if (request.url.empty()) throw Error(ErrorCode::kConfig, "request URL is required");
  if (request.method == HttpMethod::kGet) {
    if (!request.body.empty()) throw Error(ErrorCode::kConfig, "GET must not carry a body");
    if (!request.mime_parts.empty()) {
      throw Error(ErrorCode::kConfig, "GET must not carry multipart data");
    }
  }
  if (request.method == HttpMethod::kDelete && !request.mime_parts.empty()) {
    throw Error(ErrorCode::kConfig, "DELETE must not carry multipart data");
  }
  if (request.stream_limit > 0 && !request.on_body) {
    throw Error(ErrorCode::kConfig, "stream_limit requires on_body");
  }
  for (const auto& part : request.mime_parts) {
    if (part.name.empty()) {
      throw Error(ErrorCode::kConfig, "multipart part name is required");
    }
    if (part.read) {
      if (!part.data.empty()) {
        throw Error(ErrorCode::kConfig,
                    "multipart part has both data and a read callback");
      }
      if (part.size > static_cast<uint64_t>(std::numeric_limits<curl_off_t>::max())) {
        throw Error(ErrorCode::kConfig, "multipart part is too large");
      }
    }
  }
}

bool content_type_matches(std::string_view actual, std::string_view expected) {
  auto media = [](std::string_view value) {
    const size_t semi = value.find(';');
    if (semi != std::string_view::npos) value = value.substr(0, semi);
    return lower(trim(value));
  };
  const std::string want = media(expected);
  return !want.empty() && media(actual) == want;
}

PartWriter::PartWriter(std::string final_path) : final_(std::move(final_path)) {
  const std::string name = basename_of(final_);
  if (name.empty() || name == "." || name == "..") {
    throw std::invalid_argument("part writer needs a file path");
  }
  const std::string dir = directory_of(final_);
  for (int attempt = 0; attempt < 100; ++attempt) {
    const uint64_t id = part_counter.fetch_add(1, std::memory_order_relaxed);
    part_ = dir + "/." + name + ".part." + std::to_string(::getpid()) + "." +
            std::to_string(id);
    fd_ = ::open(part_.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd_ >= 0) return;
    if (errno != EEXIST) {
      throw std::runtime_error("cannot create " + part_ + ": " + std::strerror(errno));
    }
  }
  throw std::runtime_error("cannot create a private part file for " + final_);
}

PartWriter::~PartWriter() {
  if (fd_ >= 0) ::close(fd_);
  if (!published_ && !part_.empty()) ::unlink(part_.c_str());
}

void PartWriter::write(std::string_view bytes) {
  if (published_ || fd_ < 0) throw std::runtime_error("part writer is not open");
  const char* data = bytes.data();
  size_t left = bytes.size();
  while (left > 0) {
    const ssize_t written = ::write(fd_, data, left);
    if (written < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error("write failed: " + std::string(std::strerror(errno)));
    }
    if (written == 0) throw std::runtime_error("write returned zero bytes");
    data += written;
    left -= static_cast<size_t>(written);
  }
}

void PartWriter::publish() {
  if (published_) throw std::runtime_error("part is already published");
  if (fd_ < 0) throw std::runtime_error("part writer is not open");
  if (::fsync(fd_) != 0 && errno != EINVAL) {
    throw std::runtime_error("cannot flush " + part_ + ": " + std::strerror(errno));
  }
  const int fd = fd_;
  fd_ = -1;
  if (::close(fd) != 0) {
    throw std::runtime_error("cannot close " + part_ + ": " + std::strerror(errno));
  }
  // link() is atomic and fails with EEXIST, so an existing file is never
  // replaced.
  if (::link(part_.c_str(), final_.c_str()) != 0) {
    throw std::runtime_error("cannot publish " + final_ + ": " + std::strerror(errno));
  }
  ::unlink(part_.c_str());
  published_ = true;
}

void Progress::token() {
  saw_token = true;
  last_token = Clock::now();
}

void Progress::data() {
  saw_data = true;
  last_data = Clock::now();
}

void initialize_curl() {
  static const CURLcode result = curl_global_init(CURL_GLOBAL_DEFAULT);
  if (result != CURLE_OK) throw Error(ErrorCode::kConfig, "libcurl initialization failed");
}

CurlHandle::CurlHandle() : handle_(nullptr) {
  initialize_curl();
  handle_ = curl_easy_init();
  if (!handle_) throw Error(ErrorCode::kNetwork, "curl_easy_init failed");
}

CurlHandle::~CurlHandle() {
  if (handle_) curl_easy_cleanup(handle_);
}

void check_cancelled(CancelFlag cancel) {
  if (cancel && cancel->load(std::memory_order_relaxed)) {
    throw Error(ErrorCode::kCancelled, "request cancelled");
  }
}

std::string validate_base_url(std::string_view input, bool loopback_only) {
  initialize_curl();
  std::unique_ptr<CURLU, decltype(&curl_url_cleanup)> url(curl_url(), curl_url_cleanup);
  if (!url) throw Error(ErrorCode::kConfig, "URL parser initialization failed");
  const std::string text = trim(input);
  if (text.empty() || curl_url_set(url.get(), CURLUPART_URL, text.c_str(), 0) != CURLUE_OK) {
    throw Error(ErrorCode::kConfig, "invalid base URL");
  }
  const std::string scheme = lower(curl_part(url.get(), CURLUPART_SCHEME));
  const std::string host = lower(curl_part(url.get(), CURLUPART_HOST));
  if ((scheme != "http" && scheme != "https") || host.empty() ||
      !curl_part(url.get(), CURLUPART_USER).empty() ||
      !curl_part(url.get(), CURLUPART_PASSWORD).empty() ||
      !curl_part(url.get(), CURLUPART_QUERY).empty() ||
      !curl_part(url.get(), CURLUPART_FRAGMENT).empty()) {
    throw Error(ErrorCode::kConfig, "base URL must be HTTP(S) without credentials or query");
  }
  const bool loopback = loopback_host(host);
  if ((loopback_only || scheme == "http") && !loopback) {
    throw Error(ErrorCode::kConfig,
                loopback_only ? "ASR URL must use a loopback host"
                              : "plain HTTP is allowed only on loopback");
  }
  std::string out = text;
  while (out.size() > scheme.size() + 3 && out.back() == '/') out.pop_back();
  return out;
}

std::string redact(std::string_view text, std::string_view secret) {
  std::string out(text);
  // Even a one-byte key must never reach error text verbatim.
  if (secret.empty()) return out;
  size_t at = 0;
  size_t replacements = 0;
  while ((at = out.find(secret, at)) != std::string::npos && replacements++ < 1024) {
    out.replace(at, secret.size(), "[key]");
    at += 5;
  }
  return out;
}

std::string clip(std::string_view text, size_t limit) {
  if (text.size() <= limit) return std::string(text);
  size_t cut = limit;
  while (cut > 0 && (static_cast<unsigned char>(text[cut]) & 0xc0) == 0x80) --cut;
  return std::string(text.substr(0, cut)) + "…";
}

std::string error_detail(std::string_view raw) {
  try {
    const auto value = nlohmann::json::parse(raw);
    if (value.contains("error")) {
      const auto& error = value["error"];
      if (error.is_string()) return error.get<std::string>();
      if (error.is_object()) {
        std::string out;
        for (const char* key : {"type", "code", "message"}) {
          if (error.contains(key) && error[key].is_string()) {
            if (!out.empty()) out += ": ";
            out += error[key].get<std::string>();
          }
        }
        if (!out.empty()) return out;
      }
    }
    if (value.contains("message") && value["message"].is_string()) {
      return value["message"].get<std::string>();
    }
    if (value.contains("detail")) return value["detail"].dump();
  } catch (const std::exception&) {
  }
  return trim(raw);
}

Milliseconds retry_after(const std::map<std::string, std::string>& headers) {
  if (const auto it = headers.find("retry-after-ms"); it != headers.end()) {
    if (const auto delay = parse_delay(it->second, true); delay.count() > 0) return delay;
  }
  const auto it = headers.find("retry-after");
  if (it == headers.end()) return Milliseconds{0};
  if (const auto delay = parse_delay(it->second, false); delay.count() > 0) return delay;
  const std::time_t when = curl_getdate(it->second.c_str(), nullptr);
  if (when < 0) return Milliseconds{0};
  const auto now = std::time(nullptr);
  return when > now ? Milliseconds{(when - now) * 1000} : Milliseconds{0};
}

[[noreturn]] void throw_http(long status, std::string_view body,
                             const std::map<std::string, std::string>& headers,
                             std::string_view secret) {
  const std::string detail = redact(error_detail(body), secret);
  const auto after = retry_after(headers);
  const bool rate_limited = status == 429;
  const bool authentication = status == 401 || status == 403;
  // A 504 from the first-party server means request processing timed out.
  // Other 5xx statuses stay kServer.
  const bool timeout = status == 408 || status == 504;
  const bool server = status >= 500;
  const ErrorCode code = authentication      ? ErrorCode::kAuthentication
                         : rate_limited      ? ErrorCode::kRateLimited
                         : timeout           ? ErrorCode::kTimeout
                         : server            ? ErrorCode::kServer
                                             : ErrorCode::kBadRequest;
  throw Error(code, "HTTP " + std::to_string(status) + ": " + clip(detail, 300),
              rate_limited || timeout || server, static_cast<int>(status), after);
}

HttpResponse perform(const HttpRequest& request) {
  validate_request(request);
  initialize_curl();
  check_cancelled(request.cancel);
  CurlHandle curl;
  std::vector<MimeStream*> streams;
  std::unique_ptr<curl_mime, decltype(&curl_mime_free)> mime(nullptr, &curl_mime_free);
  if (!request.mime_parts.empty()) {
    mime.reset(curl_mime_init(curl.get()));
    if (!mime) throw Error(ErrorCode::kNetwork, "failed to allocate multipart request");
    for (const auto& item : request.mime_parts) {
      curl_mimepart* part = curl_mime_addpart(mime.get());
      if (!part || curl_mime_name(part, item.name.c_str()) != CURLE_OK) {
        throw Error(ErrorCode::kNetwork, "failed to build multipart request");
      }
      if (item.read) {
        auto stream = std::make_unique<MimeStream>(item.read, item.size);
        const CURLcode added =
            curl_mime_data_cb(part, static_cast<curl_off_t>(item.size), mime_read,
                              nullptr, mime_free, stream.get());
        if (added != CURLE_OK) {
          throw Error(ErrorCode::kNetwork, "failed to build multipart file part");
        }
        streams.push_back(stream.release());
      } else if (curl_mime_data(part, item.data.data(), item.data.size()) != CURLE_OK) {
        throw Error(ErrorCode::kNetwork, "failed to build multipart request");
      }
      if ((!item.filename.empty() &&
           curl_mime_filename(part, item.filename.c_str()) != CURLE_OK) ||
          (!item.content_type.empty() &&
           curl_mime_type(part, item.content_type.c_str()) != CURLE_OK)) {
        throw Error(ErrorCode::kNetwork, "failed to build multipart request");
      }
    }
  }
  Transfer transfer{request};
  transfer.curl = curl.get();
  curl_slist* headers = nullptr;
  struct HeaderCleanup {
    curl_slist*& value;
    ~HeaderCleanup() { curl_slist_free_all(value); }
  } cleanup{headers};
  auto add_header = [&](const std::string& value) {
    curl_slist* next = curl_slist_append(headers, value.c_str());
    if (!next) throw Error(ErrorCode::kNetwork, "failed to allocate HTTP headers");
    headers = next;
  };
  add_header("User-Agent: ummInfer/0.1");
  add_header(request.accept.empty() ? "Accept: application/json, text/event-stream"
                                    : "Accept: " + request.accept);
  // A callback part cannot rewind. Suppress Expect: 100-continue so a server
  // that answers 417 does not force libcurl into a failed resend.
  for (const auto& part : request.mime_parts) {
    if (part.read) {
      add_header("Expect:");
      break;
    }
  }
  if (!request.bearer.empty()) add_header("Authorization: Bearer " + request.bearer);
  for (const auto& [name, value] : request.headers) add_header(name + ": " + value);

  curl_easy_setopt(curl.get(), CURLOPT_URL, request.url.c_str());
  switch (request.method) {
    case HttpMethod::kGet:
      curl_easy_setopt(curl.get(), CURLOPT_HTTPGET, 1L);
      break;
    case HttpMethod::kDelete:
      curl_easy_setopt(curl.get(), CURLOPT_CUSTOMREQUEST, "DELETE");
      break;
    case HttpMethod::kPost:
      curl_easy_setopt(curl.get(), CURLOPT_POST, 1L);
      break;
  }
  curl_easy_setopt(curl.get(), CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, write_body);
  curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &transfer);
  curl_easy_setopt(curl.get(), CURLOPT_HEADERFUNCTION, write_header);
  curl_easy_setopt(curl.get(), CURLOPT_HEADERDATA, &transfer);
  curl_easy_setopt(curl.get(), CURLOPT_XFERINFOFUNCTION, progress);
  curl_easy_setopt(curl.get(), CURLOPT_XFERINFODATA, &transfer);
  curl_easy_setopt(curl.get(), CURLOPT_NOPROGRESS, 0L);
  curl_easy_setopt(curl.get(), CURLOPT_NOSIGNAL, 1L);
  curl_easy_setopt(curl.get(), CURLOPT_PROTOCOLS_STR, "http,https");
  curl_easy_setopt(curl.get(), CURLOPT_REDIR_PROTOCOLS_STR, "http,https");
  curl_easy_setopt(curl.get(), CURLOPT_FOLLOWLOCATION, 0L);
  curl_easy_setopt(curl.get(), CURLOPT_SSL_VERIFYPEER, 1L);
  curl_easy_setopt(curl.get(), CURLOPT_SSL_VERIFYHOST, 2L);
  curl_easy_setopt(curl.get(), CURLOPT_ACCEPT_ENCODING, "");
  curl_easy_setopt(curl.get(), CURLOPT_CONNECTTIMEOUT_MS,
                   static_cast<long>(std::max<int64_t>(1, request.connect_timeout.count())));
  if (request.timeout.count() > 0) {
    curl_easy_setopt(curl.get(), CURLOPT_TIMEOUT_MS,
                     static_cast<long>(std::max<int64_t>(1, request.timeout.count())));
  }
  if (request.no_proxy) curl_easy_setopt(curl.get(), CURLOPT_NOPROXY, "*");
  if (mime) {
    curl_easy_setopt(curl.get(), CURLOPT_MIMEPOST, mime.get());
  } else if (request.method != HttpMethod::kGet) {
    curl_easy_setopt(curl.get(), CURLOPT_POSTFIELDS, request.body.data());
    curl_easy_setopt(curl.get(), CURLOPT_POSTFIELDSIZE_LARGE,
                     static_cast<curl_off_t>(request.body.size()));
  }

  char error_buffer[CURL_ERROR_SIZE] = {};
  curl_easy_setopt(curl.get(), CURLOPT_ERRORBUFFER, error_buffer);

  CURLM* multi = curl_multi_init();
  if (!multi) throw Error(ErrorCode::kNetwork, "curl_multi_init failed");
  struct MultiCleanup {
    CURLM* multi;
    CURL* easy;
    ~MultiCleanup() {
      curl_multi_remove_handle(multi, easy);
      curl_multi_cleanup(multi);
    }
  } multi_cleanup{multi, curl.get()};
  if (curl_multi_add_handle(multi, curl.get()) != CURLM_OK) {
    throw Error(ErrorCode::kNetwork, "curl_multi_add_handle failed");
  }

  CURLcode result = CURLE_OK;
  int running = 0;
  CURLMcode multi_result = curl_multi_perform(multi, &running);
  while (multi_result == CURLM_OK && running > 0) {
    if (progress(&transfer, 0, 0, 0, 0) != 0) {
      result = CURLE_ABORTED_BY_CALLBACK;
      break;
    }
    int descriptors = 0;
    multi_result = curl_multi_poll(multi, nullptr, 0, 10, &descriptors);
    if (multi_result == CURLM_OK) multi_result = curl_multi_perform(multi, &running);
  }
  for (MimeStream* stream : streams) {
    if (stream->error) std::rethrow_exception(stream->error);
  }
  if (multi_result != CURLM_OK) {
    throw Error(ErrorCode::kNetwork, curl_multi_strerror(multi_result));
  }
  if (result == CURLE_OK) {
    int queued = 0;
    while (CURLMsg* message = curl_multi_info_read(multi, &queued)) {
      if (message->easy_handle == curl.get() && message->msg == CURLMSG_DONE) {
        result = message->data.result;
      }
    }
  }
  curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &transfer.response.status);
  if (transfer.error) std::rethrow_exception(transfer.error);
  check_cancelled(request.cancel);
  if (result != CURLE_OK && !(transfer.stopped && result == CURLE_WRITE_ERROR)) {
    if (request.progress && !request.progress->expired.empty()) {
      const bool activity = request.progress->expired == "activity";
      throw Error(ErrorCode::kTimeout, request.progress->expired + " timeout", activity);
    }
    if (result == CURLE_OPERATION_TIMEDOUT) {
      throw Error(ErrorCode::kTimeout, "request timeout", true);
    }
    const std::string detail = error_buffer[0] ? error_buffer : curl_easy_strerror(result);
    const bool refused = detail.find("Couldn't connect") != std::string::npos ||
                         detail.find("Connection refused") != std::string::npos;
    throw Error(ErrorCode::kNetwork, detail, !refused);
  }
  if (transfer.response.status / 100 != 2) {
    throw_http(transfer.response.status, transfer.response.body, transfer.response.headers,
               request.bearer);
  }
  if (!request.require_content_type.empty()) {
    const auto type = transfer.response.headers.find("content-type");
    if (type == transfer.response.headers.end() ||
        !content_type_matches(type->second, request.require_content_type)) {
      throw Error(ErrorCode::kBadResponse, "unexpected response content type");
    }
  }
  if (!transfer.stopped) check_content_length(transfer);
  return std::move(transfer.response);
}

bool SSEParser::dispatch(const Event& event) {
  if (!has_data_) return true;
  const bool keep_going = event(data_);
  data_.clear();
  has_data_ = false;
  if (!keep_going) stopped_ = true;
  return keep_going;
}

bool SSEParser::feed(std::string_view bytes, const Event& event) {
  buffer_.append(bytes);
  size_t start = 0;
  size_t lines = 0;
  const size_t max_lines = buffer_.size() + 1;
  while (start < buffer_.size() && lines++ < max_lines) {
    const size_t newline = buffer_.find('\n', start);
    if (newline == std::string::npos) break;
    std::string_view line(buffer_.data() + start, newline - start);
    start = newline + 1;
    if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
    if (line.empty()) {
      if (!dispatch(event)) {
        buffer_.erase(0, start);
        return false;
      }
      continue;
    }
    const size_t colon = line.find(':');
    const std::string_view field = line.substr(0, colon);
    if (field != "data") continue;
    std::string_view value = colon == std::string_view::npos ? std::string_view{} : line.substr(colon + 1);
    if (!value.empty() && value.front() == ' ') value.remove_prefix(1);
    if (has_data_) data_.push_back('\n');
    if (value.size() > (4u << 20) - std::min<size_t>(data_.size(), 4u << 20)) {
      throw Error(ErrorCode::kBadResponse, "SSE event exceeds 4 MiB");
    }
    data_.append(value);
    has_data_ = true;
  }
  buffer_.erase(0, start);
  if (buffer_.size() > 4u << 20) throw Error(ErrorCode::kBadResponse, "SSE line exceeds 4 MiB");
  return true;
}

bool SSEParser::finish(const Event& event) {
  if (!buffer_.empty()) {
    feed("\n", event);
    if (stopped_) return false;
  }
  return dispatch(event);
}

std::string base64_encode(std::string_view bytes) {
  static constexpr char alphabet[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve((bytes.size() + 2) / 3 * 4);
  for (size_t i = 0; i < bytes.size(); i += 3) {
    const uint32_t a = static_cast<unsigned char>(bytes[i]);
    const uint32_t b = i + 1 < bytes.size() ? static_cast<unsigned char>(bytes[i + 1]) : 0;
    const uint32_t c = i + 2 < bytes.size() ? static_cast<unsigned char>(bytes[i + 2]) : 0;
    const uint32_t value = (a << 16) | (b << 8) | c;
    out.push_back(alphabet[(value >> 18) & 63]);
    out.push_back(alphabet[(value >> 12) & 63]);
    out.push_back(i + 1 < bytes.size() ? alphabet[(value >> 6) & 63] : '=');
    out.push_back(i + 2 < bytes.size() ? alphabet[value & 63] : '=');
  }
  return out;
}

std::string base64_decode(std::string_view text, size_t max_bytes) {
  auto value = [](char c) -> int {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
  };
  if (text.size() % 4 != 0) {
    throw Error(ErrorCode::kBadResponse, "base64 length is not a multiple of 4");
  }
  size_t padding = 0;
  if (!text.empty() && text.back() == '=') ++padding;
  if (text.size() >= 2 && text[text.size() - 2] == '=') ++padding;
  const size_t decoded_size = text.size() / 4 * 3 - padding;
  if (decoded_size > max_bytes) {
    throw Error(ErrorCode::kBadResponse, "base64 output exceeds its size limit");
  }
  std::string out;
  out.reserve(decoded_size);
  for (size_t i = 0; i < text.size(); i += 4) {
    const bool last = i + 4 == text.size();
    int v[4] = {0, 0, 0, 0};
    for (size_t j = 0; j < 4; ++j) {
      const char c = text[i + j];
      if (c == '=') {
        if (!last || j + padding < 4) {
          throw Error(ErrorCode::kBadResponse, "invalid base64 padding");
        }
        continue;
      }
      v[j] = value(c);
      if (v[j] < 0) {
        throw Error(ErrorCode::kBadResponse, "invalid base64 character");
      }
    }
    // Two padding characters carry one byte (low four bits of v[1] unused);
    // one padding character carries two bytes (low two bits of v[2] unused).
    if (last && padding == 2 && (v[1] & 0x0f) != 0) {
      throw Error(ErrorCode::kBadResponse, "non-canonical base64 padding");
    }
    if (last && padding == 1 && (v[2] & 0x03) != 0) {
      throw Error(ErrorCode::kBadResponse, "non-canonical base64 padding");
    }
    const uint32_t chunk = (static_cast<uint32_t>(v[0]) << 18) |
                           (static_cast<uint32_t>(v[1]) << 12) |
                           (static_cast<uint32_t>(v[2]) << 6) |
                           static_cast<uint32_t>(v[3]);
    const size_t bytes_here = 3 - (last ? padding : 0);
    if (bytes_here >= 1) out.push_back(static_cast<char>((chunk >> 16) & 0xff));
    if (bytes_here >= 2) out.push_back(static_cast<char>((chunk >> 8) & 0xff));
    if (bytes_here >= 3) out.push_back(static_cast<char>(chunk & 0xff));
  }
  return out;
}

std::string url_encode(std::string_view text) {
  CurlHandle curl;
  char* encoded = curl_easy_escape(curl.get(), text.data(), static_cast<int>(text.size()));
  if (!encoded) throw Error(ErrorCode::kNetwork, "URL encoding failed");
  std::string out(encoded);
  curl_free(encoded);
  return out;
}

}  // namespace detail
}  // namespace umminfer::client

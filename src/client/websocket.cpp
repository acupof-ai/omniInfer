// SPDX-License-Identifier: Apache-2.0
#include "websocket.hpp"

#include <poll.h>
#include <sys/socket.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>

#include "http.hpp"

namespace umminfer::client::detail {
namespace {

using Clock = std::chrono::steady_clock;

std::string lowercase(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

std::string url_part(CURLU* url, CURLUPart part) {
  char* value = nullptr;
  if (curl_url_get(url, part, &value, 0) != CURLUE_OK || !value) return {};
  std::string out(value);
  curl_free(value);
  return out;
}

}  // namespace

struct WebSocket::Impl {
  Impl(const std::string& url, Milliseconds connect_timeout, CancelFlag internal_cancel,
       CancelFlag outer_cancel, const std::string& bearer)
      : cancel(internal_cancel), external_cancel(outer_cancel) {
    initialize_curl();
    bool websocket_supported = false;
    const curl_version_info_data* version = curl_version_info(CURLVERSION_NOW);
    if (version && version->protocols) {
      for (const char* const* protocol = version->protocols; *protocol; ++protocol) {
        websocket_supported |= std::string_view(*protocol) == "ws";
      }
    }
    if (!websocket_supported) {
      throw Error(ErrorCode::kConfig,
                  "realtime requires libcurl built with WebSocket support");
    }
    std::unique_ptr<CURLU, decltype(&curl_url_cleanup)> parsed(curl_url(), curl_url_cleanup);
    if (!parsed ||
        curl_url_set(parsed.get(), CURLUPART_URL, url.c_str(), CURLU_NON_SUPPORT_SCHEME) !=
            CURLUE_OK) {
      throw Error(ErrorCode::kConfig, "invalid realtime WebSocket URL");
    }
    const std::string scheme = lowercase(url_part(parsed.get(), CURLUPART_SCHEME));
    const std::string host = lowercase(url_part(parsed.get(), CURLUPART_HOST));
    const bool loopback = host == "localhost" || host == "::1" || host.rfind("127.", 0) == 0;
    if (scheme != "ws" || !loopback || !url_part(parsed.get(), CURLUPART_USER).empty() ||
        !url_part(parsed.get(), CURLUPART_PASSWORD).empty() ||
        !url_part(parsed.get(), CURLUPART_FRAGMENT).empty()) {
      throw Error(ErrorCode::kConfig, "realtime WebSocket must use loopback ws://");
    }

    curl_slist* headers = curl_slist_append(nullptr, "User-Agent: ummInfer/0.1");
    if (!headers) throw Error(ErrorCode::kNetwork, "failed to allocate WebSocket headers");
    if (!bearer.empty()) {
      const std::string authorization = "Authorization: Bearer " + bearer;
      curl_slist* updated = curl_slist_append(headers, authorization.c_str());
      if (!updated) {
        curl_slist_free_all(headers);
        throw Error(ErrorCode::kNetwork, "failed to allocate WebSocket headers");
      }
      headers = updated;
    }
    curl_easy_setopt(curl.get(), CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl.get(), CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl.get(), CURLOPT_CONNECT_ONLY, 2L);
    curl_easy_setopt(curl.get(), CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl.get(), CURLOPT_NOPROXY, "*");
    curl_easy_setopt(curl.get(), CURLOPT_PROTOCOLS_STR, "ws");
    curl_easy_setopt(curl.get(), CURLOPT_CONNECTTIMEOUT_MS,
                     static_cast<long>(std::max<int64_t>(1, connect_timeout.count())));
    curl_easy_setopt(curl.get(), CURLOPT_TIMEOUT_MS,
                     static_cast<long>(std::max<int64_t>(1, connect_timeout.count())));
    curl_easy_setopt(curl.get(), CURLOPT_XFERINFOFUNCTION, handshake_progress);
    curl_easy_setopt(curl.get(), CURLOPT_XFERINFODATA, this);
    curl_easy_setopt(curl.get(), CURLOPT_NOPROGRESS, 0L);
    char error_buffer[CURL_ERROR_SIZE] = {};
    curl_easy_setopt(curl.get(), CURLOPT_ERRORBUFFER, error_buffer);
    const CURLcode result = curl_easy_perform(curl.get());
    curl_slist_free_all(headers);
    check_cancelled(cancel);
    check_cancelled(external_cancel);
    if (result != CURLE_OK) {
      const std::string detail = error_buffer[0] ? error_buffer : curl_easy_strerror(result);
      throw Error(result == CURLE_OPERATION_TIMEDOUT ? ErrorCode::kTimeout : ErrorCode::kNetwork,
                  detail, result == CURLE_OPERATION_TIMEDOUT);
    }
    long status = 0;
    curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &status);
    if (status != 101) {
      const ErrorCode code = status == 401 || status == 403
                                 ? ErrorCode::kAuthentication
                                 : ErrorCode::kBadResponse;
      throw Error(code, "realtime WebSocket upgrade returned HTTP " +
                            std::to_string(status));
    }
    curl_socket_t active = CURL_SOCKET_BAD;
    if (curl_easy_getinfo(curl.get(), CURLINFO_ACTIVESOCKET, &active) != CURLE_OK ||
        active == CURL_SOCKET_BAD) {
      throw Error(ErrorCode::kNetwork, "realtime WebSocket has no active socket");
    }
    socket = active;
  }

  ~Impl() { close(); }

  static int handshake_progress(void* opaque, curl_off_t, curl_off_t, curl_off_t, curl_off_t) {
    const auto* self = static_cast<const Impl*>(opaque);
    return self->is_cancelled() ? 1 : 0;
  }

  bool is_cancelled() const {
    return (cancel && cancel->load(std::memory_order_relaxed)) ||
           (external_cancel && external_cancel->load(std::memory_order_relaxed));
  }

  void close() {
    if (closed || closing.exchange(true)) return;
    if (is_cancelled() || !running) {
      shutdown_now();
      return;
    }
    send_close();
  }

  void shutdown_now() {
    if (!closed.exchange(true) && socket != CURL_SOCKET_BAD) {
      ::shutdown(socket, SHUT_RDWR);
    }
  }

  void send_close() {
    if (close_sent.exchange(true)) return;
    static constexpr unsigned char normal[] = {0x03, 0xe8};
    const auto deadline = Clock::now() + std::chrono::seconds{1};
    try {
      std::lock_guard lock(io_mu);
      size_t sent = 0;
      CURLcode result;
      do {
        result = curl_ws_send(curl.get(), normal, sizeof(normal), &sent, 0, CURLWS_CLOSE);
        if (result == CURLE_AGAIN) wait(POLLOUT, deadline);
      } while (result == CURLE_AGAIN);
      if (result != CURLE_OK || sent != sizeof(normal)) shutdown_now();
    } catch (...) {
      shutdown_now();
    }
  }

  bool wait(short events, Clock::time_point deadline) {
    while (!closed && !is_cancelled()) {
      const auto now = Clock::now();
      if (now >= deadline) throw Error(ErrorCode::kTimeout, "WebSocket I/O timeout");
      const auto remaining = std::chrono::duration_cast<Milliseconds>(deadline - now).count();
      pollfd fd{socket, events, 0};
      const int result = ::poll(&fd, 1, static_cast<int>(std::clamp<int64_t>(remaining, 1, 50)));
      if (result > 0) return true;
      if (result < 0 && errno != EINTR) {
        throw Error(ErrorCode::kNetwork, "WebSocket poll failed: " + std::string(std::strerror(errno)));
      }
    }
    throw Error(ErrorCode::kCancelled, "request cancelled");
  }

  void send(std::string_view text) {
    std::lock_guard lock(io_mu);
    const auto deadline = Clock::now() + std::chrono::seconds{30};
    size_t offset = 0;
    const size_t partial_size = text.size() > 1 ? text.size() - 1 : 0;
    while (offset < partial_size) {
      if (closed || closing) throw Error(ErrorCode::kNetwork, "realtime WebSocket is closed");
      size_t sent = 0;
      const CURLcode result = curl_ws_send(curl.get(), text.data() + offset,
                                           partial_size - offset, &sent,
                                           offset == 0 ? text.size() : 0,
                                           CURLWS_TEXT | CURLWS_OFFSET);
      if (result == CURLE_AGAIN) {
        wait(POLLOUT, deadline);
        continue;
      }
      if (result != CURLE_OK) {
        throw Error(ErrorCode::kNetwork,
                    "realtime WebSocket write failed: " + std::string(curl_easy_strerror(result)));
      }
      if (sent == 0) throw Error(ErrorCode::kNetwork, "realtime WebSocket made no write progress");
      offset += sent;
    }
    if (text.empty() || text.size() == 1 || partial_size > 0) {
      size_t sent = 0;
      const char* last = text.empty() ? "" : text.data() + text.size() - 1;
      const size_t length = text.empty() ? 0 : 1;
      CURLcode result;
      do {
        if (closed || closing) throw Error(ErrorCode::kNetwork, "realtime WebSocket is closed");
        result = curl_ws_send(curl.get(), last, length, &sent, 0, CURLWS_TEXT);
        if (result == CURLE_AGAIN) wait(POLLOUT, deadline);
      } while (result == CURLE_AGAIN);
      if (result != CURLE_OK || sent != length) {
        throw Error(ErrorCode::kNetwork, "realtime WebSocket final write failed");
      }
    }
  }

  void run(const std::function<void(std::string_view)>& on_message) {
    running = true;
    struct RunningGuard {
      std::atomic<bool>& running;
      ~RunningGuard() { running = false; }
    } running_guard{running};
    std::string message;
    bool message_active = false;
    bool text_message = false;
    bool close_wait = false;
    Clock::time_point close_deadline;
    size_t frame_offset = 0;
    char buffer[16384];
    while (!closed && !is_cancelled()) {
      int timeout_ms = 50;
      if (closing) {
        if (!close_wait) {
          close_wait = true;
          close_deadline = Clock::now() + std::chrono::seconds{1};
        }
        const auto remaining =
            std::chrono::duration_cast<Milliseconds>(close_deadline - Clock::now()).count();
        if (remaining <= 0) {
          shutdown_now();
          return;
        }
        timeout_ms = static_cast<int>(std::clamp<int64_t>(remaining, 1, 50));
      }
      pollfd fd{socket, POLLIN, 0};
      const int ready = ::poll(&fd, 1, timeout_ms);
      if (ready < 0) {
        if (errno == EINTR) continue;
        throw Error(ErrorCode::kNetwork, "WebSocket poll failed");
      }
      if (ready == 0) continue;
      for (size_t chunks = 0; chunks < 1024 && !closed; ++chunks) {
        size_t received = 0;
        const curl_ws_frame* meta = nullptr;
        CURLcode result;
        {
          std::lock_guard lock(io_mu);
          result = curl_ws_recv(curl.get(), buffer, sizeof(buffer), &received, &meta);
        }
        if (result == CURLE_AGAIN) break;
        if (result == CURLE_GOT_NOTHING) {
          shutdown_now();
          return;
        }
        if (closed) return;
        if (result != CURLE_OK) {
          throw Error(ErrorCode::kNetwork,
                      "realtime WebSocket read failed: " + std::string(curl_easy_strerror(result)));
        }
        if (!meta || (meta->flags & (CURLWS_PING | CURLWS_PONG))) continue;
        if (meta->flags & CURLWS_CLOSE) {
          send_close();
          shutdown_now();
          return;
        }
        if (meta->offset < 0 || static_cast<uint64_t>(meta->offset) != frame_offset) {
          throw Error(ErrorCode::kBadResponse, "realtime WebSocket fragments are out of order");
        }
        frame_offset += received;
        if (!message_active) {
          message.clear();
          text_message = (meta->flags & CURLWS_TEXT) != 0;
          message_active = true;
        }
        if (text_message) {
          if (received > (1u << 20) - std::min<size_t>(message.size(), 1u << 20)) {
            throw Error(ErrorCode::kBadResponse, "realtime WebSocket message exceeds 1 MiB");
          }
          message.append(buffer, received);
        }
        if (meta->bytesleft == 0) {
          frame_offset = 0;
          if (!(meta->flags & CURLWS_CONT)) {
            if (text_message) on_message(message);
            message.clear();
            message_active = false;
            text_message = false;
          }
        }
      }
    }
    shutdown_now();
  }

  CancelFlag cancel;
  CancelFlag external_cancel;
  std::atomic<bool> closed{false};
  std::atomic<bool> closing{false};
  std::atomic<bool> close_sent{false};
  std::atomic<bool> running{false};
  CurlHandle curl;
  curl_socket_t socket = CURL_SOCKET_BAD;
  std::mutex io_mu;
};

WebSocket::WebSocket(std::string url, Milliseconds connect_timeout, CancelFlag cancel,
                     CancelFlag external_cancel, std::string bearer)
    : impl_(std::make_unique<Impl>(url, connect_timeout, cancel, external_cancel, bearer)) {}
WebSocket::~WebSocket() = default;
void WebSocket::send_text(std::string_view text) { impl_->send(text); }
void WebSocket::run(const std::function<void(std::string_view)>& on_message) {
  impl_->run(on_message);
}
void WebSocket::close() { impl_->close(); }

}  // namespace umminfer::client::detail

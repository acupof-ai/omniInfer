// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <string_view>

#include "umminfer/client/client.hpp"

namespace umminfer::client::detail {

class WebSocket {
 public:
  WebSocket(std::string url, Milliseconds connect_timeout, CancelFlag cancel,
            CancelFlag external_cancel = nullptr, std::string bearer = {});
  ~WebSocket();
  WebSocket(const WebSocket&) = delete;
  WebSocket& operator=(const WebSocket&) = delete;

  void send_text(std::string_view text);
  // Best-effort control frame. A closed or busy socket keeps the last count.
  void ping();
  uint64_t pings_sent() const;
  uint64_t pongs_received() const;
  void run(const std::function<void(std::string_view)>& on_message);
  void close();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace umminfer::client::detail

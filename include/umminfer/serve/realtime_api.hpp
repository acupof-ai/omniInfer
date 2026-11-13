// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "umminfer/serve/openai_api.hpp"

namespace umminfer {

class RealtimeApi {
 public:
  RealtimeApi(Coordinator& coordinator, ApiConfig config);
  ~RealtimeApi();

  // Chains this listener in front of the sinks already installed by OpenAIApi
  // and the unified engine. Must be called before Coordinator::run.
  void install_stream_sink();

  bool bind();
  bool listen();
  void stop();
  int bound_port() const;

  class Impl;

 private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace umminfer

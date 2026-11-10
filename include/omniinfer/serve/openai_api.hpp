// SPDX-License-Identifier: Apache-2.0
// OpenAI-compatible HTTP surface.
//
// Text pipeline routes: health, models, chat completions, and abort.
// Speech and realtime routes belong to specialized stages.
#pragma once

#include <atomic>
#include <memory>
#include <string>

#include "omniinfer/pipeline/pipeline.hpp"

namespace omniinfer {

struct ApiConfig {
  std::string host = "0.0.0.0";
  int port = 30000;
  std::string model = "omni";
  std::string served_model_name;  // defaults to model
  std::string entry_stage = "entry";
  std::string api_key;            // empty disables auth
  double request_timeout_s = 600.0;
};

class OpenAIApi {
 public:
  OpenAIApi(Coordinator& coordinator, ApiConfig config);
  ~OpenAIApi();

  bool bind();     // reserves the configured port without accepting requests
  bool listen();   // binds if needed, then blocks
  void stop();
  int bound_port() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace omniinfer

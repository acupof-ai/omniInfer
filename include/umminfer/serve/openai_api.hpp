// SPDX-License-Identifier: Apache-2.0
// OpenAI-compatible HTTP surface.
//
// Served routes: health, ready, models, chat completions, audio
// transcriptions, and abort. /v1/audio/speech answers 501. Realtime uses a
// separate WebSocket route.
//
// The surface accepts only behavior this server implements. Unsupported
// request fields are rejected with HTTP 400 instead of being ignored.
#pragma once

#include <atomic>
#include <memory>
#include <string>

#include "umminfer/pipeline/pipeline.hpp"

namespace umminfer {

class UnifiedEngine;

struct ApiConfig {
  std::string host = "0.0.0.0";
  int port = 30000;
  std::string model = "omni";
  std::string served_model_name;  // defaults to model
  std::string entry_stage = "entry";
  std::string api_key;            // empty disables auth
  // True when the entry stage runs in another process. /ready then answers 503
  // until that stage announces stage_ready. The in-process demo keeps false.
  bool external_entry_stage = false;
  double request_timeout_s = 600.0;

  // Fixed upstream. One process serves one upstream, one model and one
  // variant; there is no routing or fallback between upstreams. All five
  // fields empty keeps every route on its local behavior.
  //   upstream_kind    "qwen" (official DashScope compatible API) or "sglang"
  //   upstream_base_url  HTTP(S) origin plus API prefix, no credentials
  //   upstream_model   the model ID sent to the upstream
  //   upstream_api_key sent to the upstream only, never logged
  //   upstream_variant "omni" for qwen; "image" or "video" for sglang
  // A route outside the configured variant answers 501.
  std::string upstream_kind;
  std::string upstream_base_url;
  std::string upstream_model;
  std::string upstream_api_key;
  std::string upstream_variant;
};

// Returns an empty string when the upstream configuration is usable, otherwise
// the reason. Network-free; bind() calls it before reserving the port.
std::string validate_upstream_config(const ApiConfig& config);

class OpenAIApi {
 public:
  OpenAIApi(Coordinator& coordinator, ApiConfig config);
  ~OpenAIApi();

  // Call after construction and before Coordinator::run. Chat then uses the
  // unified ASR -> VLM engine; transcription and realtime stay on the ASR stage.
  void set_unified_engine(UnifiedEngine& engine, std::string vlm_stage);

  bool bind();     // reserves the configured port without accepting requests
  bool listen();   // binds if needed, then blocks
  void stop();
  int bound_port() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace umminfer

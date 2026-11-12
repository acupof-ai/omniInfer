// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>

#include "umminfer/pipeline/pipeline.hpp"

namespace umminfer::serve_detail {

inline StagePayload make_asr_payload(std::string pcm, int64_t max_tokens,
                                     const std::string& language,
                                     const std::string& prompt) {
  StagePayload payload;
  Value::Map inputs;
  inputs["audio"] = Value::bin(std::move(pcm));
  payload.request.inputs = Value(std::move(inputs));
  payload.request.params["max_new_tokens"] = Value(max_tokens);
  payload.request.params["temperature"] = Value(0.0);
  payload.request.params["stream"] = Value(false);
  if (!language.empty()) payload.request.metadata["language"] = Value(language);
  if (!prompt.empty()) payload.request.metadata["prompt"] = Value(prompt);
  return payload;
}

inline std::string asr_result_text(const Value& result) {
  if (result.is_str()) return result.as_str();
  if (const Value* text = result.find("text"); text && text->is_str()) {
    return text->as_str();
  }
  throw ProtocolError("ASR stage returned no text");
}

struct AsrGenerationMetadata {
  int64_t generated_token_count = 0;
  std::string finish_reason;
};

// True only when the stage reported both values with real types. A stage that
// cannot count its tokens leaves them out; the route then omits the metadata
// or rejects an explicit request instead of inventing zeros.
inline bool asr_generation_metadata(const Value& result,
                                    AsrGenerationMetadata& out) {
  const Value* count = result.find("generated_token_count");
  const Value* finish = result.find("finish_reason");
  if (count == nullptr || finish == nullptr) return false;
  if (!count->is_int() || !finish->is_str()) return false;
  if (count->as_int() < 0 || finish->as_str().empty()) return false;
  out.generated_token_count = count->as_int();
  out.finish_reason = finish->as_str();
  return true;
}

}  // namespace umminfer::serve_detail

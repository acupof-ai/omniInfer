// SPDX-License-Identifier: Apache-2.0
#include "umminfer/pipeline/unified.hpp"

#include <string_view>

namespace umminfer {
namespace {

// Content-part "type" values the client contract uses (client.hpp ContentPart).
UnifiedModality modality_of_part_type(const std::string& type) {
  if (type == "input_audio") return UnifiedModality::kAudio;
  if (type == "image_url") return UnifiedModality::kImage;
  if (type == "video_url" || type == "video") return UnifiedModality::kVideo;
  return UnifiedModality::kText;
}

std::string_view chunk_text(const Value& chunk) {
  if (!chunk.is_map()) return {};
  const Value* text = chunk.find("text");
  if (text != nullptr && text->is_str()) return text->as_str();
  return {};
}

bool chunk_is_final(const Value& chunk) {
  if (!chunk.is_map()) return false;
  const Value* flag = chunk.find("is_final");
  return flag != nullptr && flag->is_bool() && flag->as_bool();
}

// Extracts ASR result text under the same contract as
// serve_detail::asr_result_text (src/serve/asr_common.hpp): a plain string or a
// map with a string "text". Kept local so the pipeline layer does not include a
// serve-private header. Nullopt means the stage returned no usable text; the
// caller must fail the request, not start the VLM leg with an empty prompt.
std::optional<std::string> asr_result_text(const Value& result) {
  if (result.is_str()) return result.as_str();
  if (result.is_map()) {
    const Value* text = result.find("text");
    if (text != nullptr && text->is_str()) return text->as_str();
  }
  return std::nullopt;
}

}  // namespace

const char* to_string(UnifiedModality modality) {
  switch (modality) {
    case UnifiedModality::kAudio: return "audio";
    case UnifiedModality::kText: return "text";
    case UnifiedModality::kImage: return "image";
    case UnifiedModality::kVideo: return "video";
  }
  return "unknown";
}

std::optional<UnifiedModality> modality_from_string(const std::string& token) {
  if (token == "audio") return UnifiedModality::kAudio;
  if (token == "text") return UnifiedModality::kText;
  if (token == "image") return UnifiedModality::kImage;
  if (token == "video") return UnifiedModality::kVideo;
  return std::nullopt;
}

const char* to_string(UnifiedEvent::Kind kind) {
  switch (kind) {
    case UnifiedEvent::Kind::kAsrPartial: return "asr_partial";
    case UnifiedEvent::Kind::kAsrFinal: return "asr_final";
    case UnifiedEvent::Kind::kVlmDelta: return "vlm_delta";
    case UnifiedEvent::Kind::kComplete: return "complete";
    case UnifiedEvent::Kind::kError: return "error";
    case UnifiedEvent::Kind::kCancelled: return "cancelled";
  }
  return "unknown";
}

UnifiedEngine::UnifiedEngine(Coordinator& coordinator, UnifiedConfig config)
    : coordinator_(coordinator), config_(std::move(config)) {
  if (config_.asr_stage == config_.vlm_stage) {
    throw ProtocolError("unified engine requires distinct asr and vlm stages");
  }
  // Chain sinks the embedded server installed instead of replacing them: the
  // HTTP SSE path and this engine share one Coordinator.
  previous_stream_ = std::move(coordinator_.on_stream);
  previous_complete_ = std::move(coordinator_.on_complete);
  coordinator_.on_stream = [this](const StreamMessage& msg) { handle_stream(msg); };
  coordinator_.on_complete = [this](const RequestInfo& info) { handle_complete(info); };
}

UnifiedEngine::~UnifiedEngine() {
  coordinator_.on_stream = std::move(previous_stream_);
  coordinator_.on_complete = std::move(previous_complete_);
}

std::string UnifiedEngine::submit(StagePayload payload) {
  const Value& inputs = payload.request.inputs;
  if (!inputs.is_map() || inputs.find("messages") == nullptr ||
      !inputs.find("messages")->is_array()) {
    throw ProtocolError("unified request inputs must contain a messages array");
  }

  // Image and video have no worker in v1. Reject an explicit modality claim or
  // a matching content part before any stage sees the work.
  if (const Value* declared = inputs.find("modalities"); declared != nullptr) {
    if (!declared->is_array()) {
      throw ProtocolError("unified request modalities must be an array");
    }
    for (const Value& token : declared->as_array()) {
      if (!token.is_str()) {
        throw ProtocolError("unified request modality must be a string");
      }
      const auto modality = modality_from_string(token.as_str());
      if (!modality) throw ProtocolError("unknown modality: " + token.as_str());
      if (*modality == UnifiedModality::kImage ||
          *modality == UnifiedModality::kVideo) {
        throw ProtocolError("modality " + token.as_str() +
                            " is declared but has no worker in unified engine v1");
      }
    }
  }
  for (const Value& message : inputs.find("messages")->as_array()) {
    if (!message.is_map()) continue;
    const Value* content = message.find("content");
    if (content == nullptr || !content->is_array()) continue;
    for (const Value& part : content->as_array()) {
      if (!part.is_map()) continue;
      const Value* type = part.find("type");
      if (type == nullptr || !type->is_str()) continue;
      const UnifiedModality modality = modality_of_part_type(type->as_str());
      if (modality == UnifiedModality::kImage || modality == UnifiedModality::kVideo) {
        throw ProtocolError(std::string("content part ") + type->as_str() +
                            " is declared but has no worker in unified engine v1");
      }
    }
  }

  std::string session_id;
  if (const Value* session = inputs.find("session_id");
      session != nullptr && session->is_str() && !session->as_str().empty()) {
    session_id = session->as_str();
  } else {
    session_id = "session-" + std::to_string(session_counter_++);
  }

  std::lock_guard<std::mutex> lock(mu_);
  if (admission_.size() >= config_.max_in_flight) {
    throw QueueFullError();
  }
  const std::string request_id = coordinator_.submit(payload, config_.asr_stage);
  admission_.push_back(request_id);
  Record record;
  record.session_id = std::move(session_id);
  record.original_inputs = inputs;
  records_[request_id] = std::move(record);
  return request_id;
}

bool UnifiedEngine::cancel(const std::string& request_id) {
  std::string active_id;
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = records_.find(request_id);
    if (it == records_.end() || it->second.phase == Phase::kDone) return false;
    // Record the intent under mu_ first. The ASR completion and the VLM start
    // are decided under the same lock in handle_complete, so either this flag
    // is visible there and the request settles as kCancelled, or the VLM leg
    // already exists and its child id is what we abort.
    it->second.cancel_requested = true;
    active_id = it->second.vlm_id.empty() ? request_id : it->second.vlm_id;
  }
  // No engine lock held: Coordinator::abort can synchronously invoke
  // on_complete, which needs mu_.
  coordinator_.abort(active_id);
  return true;
}

bool UnifiedEngine::wait(const std::string& request_id, double timeout_s) {
  std::unique_lock<std::mutex> lock(mu_);
  return cv_.wait_for(lock, std::chrono::duration<double>(timeout_s), [&] {
    auto it = records_.find(request_id);
    return it == records_.end() || it->second.phase == Phase::kDone;
  });
}

std::optional<RequestInfo> UnifiedEngine::info(const std::string& request_id) {
  return coordinator_.info(request_id);
}

bool UnifiedEngine::release(const std::string& request_id) {
  std::string vlm_id;
  bool removed = false;
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = records_.find(request_id);
    if (it == records_.end()) return false;
    vlm_id = it->second.vlm_id;
    records_.erase(it);
    for (auto queue_it = admission_.begin(); queue_it != admission_.end();
         ++queue_it) {
      if (*queue_it == request_id) {
        admission_.erase(queue_it);
        break;
      }
    }
    removed = true;
  }
  // Both coordinator leg records belong to this engine request.
  coordinator_.release(request_id);
  if (!vlm_id.empty()) coordinator_.release(vlm_id);
  return removed;
}

size_t UnifiedEngine::in_flight() const {
  std::lock_guard<std::mutex> lock(mu_);
  return admission_.size();
}

// Records are keyed by the ASR leg id; VLM events carry the derived child id.
// ponytail: linear fallback scan, max_in_flight bounds the record count.
std::map<std::string, UnifiedEngine::Record>::iterator
UnifiedEngine::find_record_locked(const std::string& event_request_id) {
  auto it = records_.find(event_request_id);
  if (it != records_.end()) return it;
  for (auto candidate = records_.begin(); candidate != records_.end();
       ++candidate) {
    if (!candidate->second.vlm_id.empty() &&
        candidate->second.vlm_id == event_request_id) {
      return candidate;
    }
  }
  return records_.end();
}

void UnifiedEngine::handle_stream(const StreamMessage& msg) {
  std::vector<UnifiedEvent> due;
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto record_it = find_record_locked(msg.request_id);
    if (record_it == records_.end() || record_it->second.phase == Phase::kDone) {
      // Unknown or terminal: idempotent drop, matching the Coordinator's own
      // in-flight-only stream filter.
      if (previous_stream_) previous_stream_(msg);
      return;
    }
    Record& record = record_it->second;
    const std::string& unified_id = record_it->first;
    if (msg.from_stage == config_.vlm_stage) {
      if (record.phase != Phase::kVlm) {
        // A VLM chunk before the VLM leg starts is out of order: drop it.
        if (previous_stream_) previous_stream_(msg);
        return;
      }
      record.vlm_text += chunk_text(msg.chunk);
      UnifiedEvent event;
      event.request_id = unified_id;  // contract id, never the child id
      event.session_id = record.session_id;
      event.stage_id = kVlmStageId;
      event.kind = UnifiedEvent::Kind::kVlmDelta;
      event.seq = record.next_seq++;
      event.text = chunk_text(msg.chunk);
      due.push_back(std::move(event));
    } else {
      if (record.phase == Phase::kVlm) {
        // An ASR chunk after the VLM leg started is late and out of order:
        // the ASR final already fixed the transcript, so drop it.
        if (previous_stream_) previous_stream_(msg);
        return;
      }
      const bool is_final = chunk_is_final(msg.chunk);
      if (is_final) {
        // A repeated final is idempotent: the first owns the transcript, a
        // later one changes nothing and emits no second event.
        if (record.asr_final_seen) {
          if (previous_stream_) previous_stream_(msg);
          return;
        }
        record.asr_final_seen = true;
        record.transcript = chunk_text(msg.chunk);  // final replaces partials
      } else {
        record.transcript += chunk_text(msg.chunk);  // partials are deltas
      }
      UnifiedEvent event;
      event.request_id = unified_id;
      event.session_id = record.session_id;
      event.stage_id = kAsrStageId;
      event.kind = is_final ? UnifiedEvent::Kind::kAsrFinal
                            : UnifiedEvent::Kind::kAsrPartial;
      event.seq = record.next_seq++;
      event.is_final = is_final;
      event.text = chunk_text(msg.chunk);
      due.push_back(std::move(event));
    }
  }
  if (previous_stream_) previous_stream_(msg);
  for (const auto& event : due) {
    if (on_event) on_event(event);
  }
}

Value UnifiedEngine::build_vlm_inputs(const std::string& transcript,
                                     const Value& original_inputs) {
  Value::Map out;
  if (original_inputs.is_map()) {
    for (const auto& [key, value] : original_inputs.as_map()) {
      if (key != "messages" && key != "modalities" && key != "session_id") {
        out[key] = value;
      }
    }
  }
  Value::Map user_message;
  user_message["role"] = Value("user");
  user_message["content"] = Value(transcript);
  Value::Array messages;
  messages.push_back(Value(std::move(user_message)));
  out["messages"] = Value(std::move(messages));
  Value::Array modalities;
  modalities.push_back(Value("text"));
  out["modalities"] = Value(std::move(modalities));
  return Value(std::move(out));
}

void UnifiedEngine::start_vlm_locked(const std::string& request_id,
                                     Record& record) {
  StagePayload vlm_payload;
  // The Coordinator rejects a duplicate request_id, so the VLM leg runs under a
  // derived child id. The suffix shape is not "req-N" and therefore never
  // advances the Coordinator's id counter.
  record.vlm_id = request_id + ":vlm";
  vlm_payload.request_id = record.vlm_id;
  vlm_payload.request.inputs =
      build_vlm_inputs(record.transcript, record.original_inputs);
  record.phase = Phase::kVlm;
  coordinator_.submit(vlm_payload, config_.vlm_stage);
}

UnifiedEvent UnifiedEngine::finish_locked(const std::string& request_id,
                                          Record& record,
                                          UnifiedEvent::Kind kind,
                                          std::string error, Value result) {
  record.phase = Phase::kDone;
  UnifiedEvent event;
  event.request_id = request_id;  // always the unified id, never the child id
  event.session_id = record.session_id;
  event.kind = kind;
  event.is_final = true;
  event.seq = record.next_seq++;
  event.error = std::move(error);
  event.result = std::move(result);
  if (kind == UnifiedEvent::Kind::kComplete) {
    event.stage_id = kVlmStageId;
    event.text = record.vlm_text;
  } else if (kind == UnifiedEvent::Kind::kError) {
    event.stage_id = record.vlm_id.empty() ? kAsrStageId : kVlmStageId;
  }
  for (auto it = admission_.begin(); it != admission_.end(); ++it) {
    if (*it == request_id) {
      admission_.erase(it);
      break;
    }
  }
  return event;
}

void UnifiedEngine::handle_complete(const RequestInfo& info) {
  std::vector<UnifiedEvent> due;
  std::optional<UnifiedEvent> terminal_event;
  {
    std::lock_guard<std::mutex> lock(mu_);
    // The completion carries either the ASR leg id or the VLM child id.
    auto it = find_record_locked(info.request_id);
    if (it != records_.end() && it->second.phase != Phase::kDone) {
      Record& record = it->second;
      const std::string& unified_id = it->first;
      if (info.state == "aborted") {
        terminal_event = finish_locked(unified_id, record,
                                       UnifiedEvent::Kind::kCancelled, {}, {});
      } else if (info.state == "failed") {
        terminal_event = finish_locked(
            unified_id, record, UnifiedEvent::Kind::kError,
            info.error.empty() ? "stage failure" : info.error, {});
      } else if (record.cancel_requested) {
        // A successful completion beat the abort frame through the
        // Coordinator. The intent was recorded under mu_ first, so this path is
        // deterministic: settle as kCancelled and never run (or trust) the VLM
        // leg. A record that already failed is kDone and cannot reach here.
        terminal_event = finish_locked(unified_id, record,
                                       UnifiedEvent::Kind::kCancelled, {}, {});
      } else if (record.vlm_id.empty()) {
        // ASR leg succeeded. A non-streaming stage sends no final chunk, so
        // synthesize the ASR final from the result text.
        if (!record.asr_final_seen) {
          const auto text = asr_result_text(info.result);
          if (!text) {
            // No usable text means a malformed stage result: fail here and
            // never start the VLM leg with an empty or invented prompt.
            terminal_event = finish_locked(
                unified_id, record, UnifiedEvent::Kind::kError,
                "ASR stage returned no text", {});
          } else {
            record.asr_final_seen = true;
            record.transcript = *text;
            UnifiedEvent event;
            event.request_id = unified_id;
            event.session_id = record.session_id;
            event.stage_id = kAsrStageId;
            event.kind = UnifiedEvent::Kind::kAsrFinal;
            event.seq = record.next_seq++;
            event.is_final = true;
            event.text = *text;
            due.push_back(std::move(event));
          }
        }
        if (!terminal_event) {
          try {
            start_vlm_locked(unified_id, record);
          } catch (const std::exception& error) {
            terminal_event = finish_locked(unified_id, record,
                                           UnifiedEvent::Kind::kError,
                                           error.what(), {});
          }
        }
      } else {
        if (record.vlm_text.empty() && info.result.is_map()) {
          if (const Value* value = info.result.find("text");
              value != nullptr && value->is_str()) {
            record.vlm_text = value->as_str();
          }
        }
        terminal_event = finish_locked(unified_id, record,
                                       UnifiedEvent::Kind::kComplete, {},
                                       info.result);
      }
    }
  }

  if (previous_complete_) previous_complete_(info);
  for (const auto& event : due) {
    if (on_event) on_event(event);
  }
  if (terminal_event) {
    if (on_event) on_event(*terminal_event);
    if (on_terminal) on_terminal(*terminal_event);
    cv_.notify_all();
  }
}

}  // namespace umminfer

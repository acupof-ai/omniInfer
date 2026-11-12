// SPDX-License-Identifier: Apache-2.0
// Unified multimodal engine v1: in-process ASR -> VLM composition.
//
// This is a value object plus a thin orchestrator over the existing
// Coordinator. It adds no second engine: the ASR leg is one StagePayload
// submitted to the registered ASR stage, the VLM leg reuses the same
// Coordinator against the registered VLM stage, every stage event stays a
// StreamMessage or CompleteMessage, and cancellation is the Coordinator abort
// broadcast the stages already implement. The orchestrator adds only:
//   - the unified request/event contract (modality, sequence, terminal state),
//   - deterministic aggregation of ASR output into one VLM prompt,
//   - a bounded admission queue (backpressure).
//
// Image and video are declared as optional modalities/capabilities only. No
// worker handles them in v1; a request that requires them is rejected at
// submission so the executed path stays audio -> ASR -> text -> VLM.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>

#include "umminfer/pipeline/pipeline.hpp"
#include "umminfer/proto/messages.hpp"

namespace umminfer {

// Modalities understood by the unified contract. Audio and text run today;
// image and video are capability declarations only.
enum class UnifiedModality { kAudio, kText, kImage, kVideo };

const char* to_string(UnifiedModality modality);
std::optional<UnifiedModality> modality_from_string(const std::string& token);

// Stage ids used inside unified events. They are contract values, not
// Coordinator stage names: deployed stages keep their configured names.
inline constexpr const char* kAsrStageId = "asr";
inline constexpr const char* kVlmStageId = "vlm";

// One event of a unified request. The sequence number is gap-free per request
// and starts at 0, so a consumer verifies ordering directly.
struct UnifiedEvent {
  enum class Kind {
    kAsrPartial,  // streaming ASR transcript, is_final == false
    kAsrFinal,    // terminal ASR transcript, is_final == true
    kVlmDelta,    // streaming VLM token
    kComplete,    // terminal success
    kError,       // terminal failure
    kCancelled,   // terminal cancellation
  };

  std::string request_id;
  std::string session_id;
  std::string stage_id;  // kAsrStageId / kVlmStageId; empty on kCancelled
  Kind kind = Kind::kError;
  int64_t seq = 0;
  bool is_final = false;
  // ASR transcript text or one VLM delta. Empty on terminal error/cancel.
  std::string text;
  // Present on kComplete: the aggregated VLM result value.
  Value result;
  // Present on kError.
  std::string error;
};

const char* to_string(UnifiedEvent::Kind kind);

struct UnifiedConfig {
  // Coordinator stage names the two deployed stages registered under.
  std::string asr_stage = "asr";
  std::string vlm_stage = "vlm";
  // Hard backpressure: requests past this many in-flight are rejected at
  // submission instead of being buffered without limit.
  size_t max_in_flight = 32;
};

// Thrown by UnifiedEngine::submit when the admission limit is reached.
class QueueFullError : public std::runtime_error {
 public:
  QueueFullError() : std::runtime_error("unified engine queue is full") {}
};

// Owns no threads. The caller pumps the Coordinator (Coordinator::run) exactly
// as today; the engine attaches stream/completion sinks and chains any sinks
// the embedded server installed earlier.
class UnifiedEngine {
 public:
  UnifiedEngine(Coordinator& coordinator, UnifiedConfig config);
  ~UnifiedEngine();
  UnifiedEngine(const UnifiedEngine&) = delete;
  UnifiedEngine& operator=(const UnifiedEngine&) = delete;

  // Submits one unified request. Reuses StagePayload:
  //   payload.request.inputs = {"messages": [...], "modalities": [...]}
  // Returns the Coordinator-allocated request id. Throws ProtocolError on a
  // bad contract (including image/video) and QueueFullError on backpressure.
  std::string submit(StagePayload payload);

  // Cancels the request on both stages via the Coordinator abort broadcast.
  // Returns false when the request is unknown or already terminal.
  bool cancel(const std::string& request_id);

  // Blocks until the request reaches a terminal state.
  bool wait(const std::string& request_id, double timeout_s);
  // Drops the terminal engine record. Mirrors Coordinator::release.
  bool release(const std::string& request_id);
  std::optional<RequestInfo> info(const std::string& request_id);
  size_t in_flight() const;

  // Fires for every ASR partial/final and VLM delta, in sequence order.
  std::function<void(const UnifiedEvent&)> on_event;
  // Fires once per request, including failures and cancellations.
  std::function<void(const UnifiedEvent&)> on_terminal;

 private:
  enum class Phase { kAsr, kVlm, kDone };

  struct Record {
    std::string session_id;
    Phase phase = Phase::kAsr;
    int64_t next_seq = 0;
    bool asr_final_seen = false;
    std::string transcript;  // partials concatenated; final replaces them
    std::string vlm_text;
    std::string vlm_id;
    Value original_inputs;
  };

  void handle_stream(const StreamMessage& msg);
  void handle_complete(const RequestInfo& info);
  void start_vlm_locked(const std::string& request_id, Record& record);
  UnifiedEvent finish_locked(const std::string& request_id, Record& record,
                             UnifiedEvent::Kind kind, std::string error,
                             Value result);
  // Records are keyed by the ASR leg id; VLM events carry the derived child id.
  // ponytail: linear fallback scan; max_in_flight bounds the record count.
  std::map<std::string, Record>::iterator find_record_locked(
      const std::string& event_request_id);
  static Value build_vlm_inputs(const std::string& transcript,
                                const Value& original_inputs);

  Coordinator& coordinator_;
  UnifiedConfig config_;
  std::function<void(const StreamMessage&)> previous_stream_;
  std::function<void(const RequestInfo&)> previous_complete_;

  mutable std::mutex mu_;
  std::condition_variable cv_;
  // FIFO of active unified ids, bounded by config_.max_in_flight.
  std::deque<std::string> admission_;
  std::map<std::string, Record> records_;
  std::atomic<uint64_t> session_counter_{0};
};

}  // namespace umminfer

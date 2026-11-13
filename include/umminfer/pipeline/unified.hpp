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
// One request may carry one image_url content part and one client text
// instruction next to the audio. The ASR leg still receives PCM only; the VLM
// leg receives the instruction, the image part, and the final ASR transcript as
// the last text part. Video has no worker and is rejected at submission.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>

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
  // Opt-in multi-turn history. Disabled by default: session_id stays a grouping
  // label. When enabled, a request must also set inputs.session_history=true;
  // only requests carrying an explicit session_id then share history. The store
  // is process-local, guarded by the engine mutex, and dies with the engine.
  struct SessionHistory {
    bool enabled = false;
    // Completed turns kept per session. Oldest turns are evicted first.
    size_t max_turns = 8;
    // Hard character budget per session across transcript and answer text.
    // Oldest turns leave first until the budget holds.
    size_t max_chars = 16384;
    // Hard cap on distinct sessions retained. The session whose last recorded
    // turn is oldest is evicted first.
    size_t max_sessions = 256;
  } session_history;
};

// Thrown by UnifiedEngine::submit when the admission limit is reached.
class QueueFullError : public std::runtime_error {
 public:
  QueueFullError() : std::runtime_error("unified engine queue is full") {}
};

// Owns no threads. Construction must precede the first Coordinator::run call;
// exchange_sinks_before_run enforces this. Keep other Coordinator calls quiescent
// during construction. The caller owns the pump and must stop and join it before
// destroying the Coordinator. The Coordinator must outlive the engine.
// Quiesce other engine API calls before destruction. Do not destroy the engine
// from one of its callbacks: the destructor waits for active callbacks.
// Destruction does not require stopping the pump; retired sinks keep forwarding
// to the sinks that preceded this engine.
class UnifiedEngine {
 public:
  UnifiedEngine(Coordinator& coordinator, UnifiedConfig config);
  // Destructor retirement drains forwarded callbacks but does not propagate a
  // foreign sink exception. Non-destructor copied sinks drain then rethrow.
  ~UnifiedEngine() noexcept;
  UnifiedEngine(const UnifiedEngine&) = delete;
  UnifiedEngine& operator=(const UnifiedEngine&) = delete;

  // Submits one unified request. Reuses StagePayload:
  //   payload.request.inputs = {"messages": [...], "modalities": [...]}
  // An additional binary "audio" value selects the real ASR worker contract.
  // The ASR leg then receives only that audio with stream=false/temperature=0;
  // the VLM leg keeps the original sampling parameters and receives the image
  // parts (at most one) plus the ASR transcript as text.
  // inputs.session_id is optional. It is a grouping label by default: every
  // event of every request carrying it reports the same session_id, and
  // requests in one session stay independent (own request id, own seq, own
  // terminal, no history). When config_.session_history.enabled is true, a
  // request may additionally set inputs.session_history=true; the VLM prompt of
  // later requests in the same session then carries prior completed turns.
  // Returns the Coordinator-allocated request id. Throws ProtocolError on a
  // bad contract (including video) and QueueFullError on backpressure.
  std::string submit(StagePayload payload);

  // Cancels the request on both stages via the Coordinator abort broadcast.
  // Returns false when the request is unknown or already terminal.
  bool cancel(const std::string& request_id);

  // Returns false for an unknown id or timeout, true only for a terminal record.
  bool wait(const std::string& request_id, double timeout_s);
  // Drops the terminal engine record. Mirrors Coordinator::release.
  bool release(const std::string& request_id);
  // Returns only unified state: unified id, public stage id, and unified result.
  std::optional<RequestInfo> info(const std::string& request_id);
  size_t in_flight() const;
  // Whether the engine was constructed with session history enabled.
  bool history_enabled() const { return config_.session_history.enabled; }

  // Fires for every ASR partial/final and VLM delta, in sequence order.
  std::function<void(const UnifiedEvent&)> on_event;
  // Fires once per request, including failures and cancellations.
  std::function<void(const UnifiedEvent&)> on_terminal;

 private:
  enum class Phase { kAsr, kVlm, kDone };
  struct CallbackGate;
  friend bool unified_retirement_waiting_for_test(const UnifiedEngine& engine);

  struct Record {
    std::string session_id;
    Phase phase = Phase::kAsr;
    RequestInfo public_info;
    int64_t next_seq = 0;
    bool asr_final_seen = false;
    // Set by cancel() under mu_ before any coordinator call. A completion that
    // wins the race against abort is still required to settle as kCancelled
    // and must not start the VLM leg.
    bool cancel_requested = false;
    // The request opted into multi-turn history. Its turn is only recorded on
    // a successful terminal; cancelled and failed requests leave no history.
    bool history_requested = false;
    std::string transcript;  // partials concatenated; final replaces them
    std::string vlm_text;
    std::string vlm_id;
    Value original_inputs;
    Value::Map vlm_params;
    Value::Map vlm_metadata;
  };

  // One completed user/assistant turn kept in process-local session history.
  struct SessionTurn {
    std::string user;
    std::string assistant;
  };

  void handle_stream(const StreamMessage& msg);
  void handle_complete(const RequestInfo& info);
  void drain_dispatch();
  void start_vlm_locked(const std::string& request_id, Record& record);
  // Returns the current messages with prior completed turns of
  // record.session_id prepended as user/assistant messages.
  Value::Array with_session_history_locked(
      const Record& record, const Value::Array& current_messages);
  // Stores the finished turn and enforces the turn and character bounds.
  void record_session_turn_locked(const Record& record);
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
  std::shared_ptr<CallbackGate> callback_gate_;

  mutable std::mutex mu_;
  std::condition_variable cv_;
  // FIFO of active unified ids, bounded by config_.max_in_flight.
  std::deque<std::string> admission_;
  std::map<std::string, Record> records_;
  using DispatchItem = std::variant<UnifiedEvent, StreamMessage, RequestInfo>;
  std::deque<DispatchItem> dispatch_queue_;
  bool dispatching_ = false;  // guarded by mu_; reentrant callbacks only queue
  std::atomic<uint64_t> session_counter_{0};
  // Process-local, in-memory history. No persistence: engine destruction or
  // process restart drops every turn.
  std::map<std::string, std::deque<SessionTurn>> session_history_;
  // Insertion order of sessions with retained turns; front is the oldest.
  // Entries for sessions that emptied through self-eviction are stale and are
  // skipped when the global cap is enforced.
  std::deque<std::string> session_order_;
};

}  // namespace umminfer

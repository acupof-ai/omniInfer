// SPDX-License-Identifier: Apache-2.0
// Stage execution and request coordination.
//
// A pipeline is a set of stages joined by ZMQ control edges and relay data
// edges. The coordinator submits at the entry stage and receives completions.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "umminfer/comm/comm.hpp"
#include "umminfer/proto/messages.hpp"
#include "umminfer/scheduling/simple_scheduler.hpp"

namespace umminfer {

struct RequestInfo {
  std::string request_id;
  std::string state = "pending";  // pending, running, completed, failed, aborted
  std::string current_stage;
  Value result;
  std::string error;
};

// One stage of the pipeline: receives work, runs its compute function, routes
// the result to the next stage or completes it at the coordinator.
class Stage {
 public:
  struct Config {
    std::string name;
    std::string recv_endpoint;         // stage binds here
    std::string coordinator_endpoint;  // stage pushes completions here
    std::string abort_endpoint;        // stage subscribes here
    std::map<std::string, std::string> next_stages;  // name -> endpoint
    std::map<std::string, std::string> peer_endpoints;  // every stage, for acks
    std::optional<int> gpu_id;                       // this stage's CUDA device
    std::map<std::string, std::vector<int>> stage_gpu_ids;  // peer name -> devices
    // A transfer whose ack never arrives must not hold a relay credit forever.
    // A value <= 0 disables the timeout.
    int transfer_timeout_ms = 30000;
    // Largest packed payload this stage accepts from a peer. A DataRef above
    // this bound fails before any receiver-side allocation.
    int64_t max_payload_bytes = int64_t{1} << 30;  // 1 GiB
    // Announces stage_ready after the control plane and abort subscription start.
    // Embedded users that do not gate Submit can leave this disabled.
    bool announce_ready = false;
    // Relay options forwarded to CommRouter (slot_size_mb, credits, worker_id).
    std::map<std::string, Value> comm_config;
    // Replaces every relay the router builds. Test seam only.
    std::shared_ptr<Relay> injected_relay;
  };

  Stage(Config config, ComputeFn compute_fn);
  ~Stage();

  void start();
  void run();  // blocks until shutdown
  void stop();

  const std::string& name() const { return config_.name; }
  size_t request_count();

 private:
  // Sends one payload across an edge.
  void send_payload(const std::string& request_id, const std::string& target,
                    const StagePayload& payload);
  void execute(StagePayload payload);
  void handle_message(Message msg);
  void route_result(const std::string& request_id, Value result);
  void on_submit(const SubmitMessage& msg);
  void on_data_ready(const DataReadyMessage& msg);
  void on_data_ack(const DataAckMessage& msg);
  void pump_outbox();
  void fail_request(const std::string& request_id, const std::string& error);
  void send_failure_ack(const DataReadyMessage& msg, const std::string& object_id,
                        const std::string& error);
  // Drops one pending entry and settles its operation. Never throws.
  void discard_pending(const std::string& object_id, const char* reason) noexcept;
  // Drops selected pending transfers and frees their credits. all=true drains
  // every entry; a request_id selects only that request. Draining unblocks a
  // sender waiting in CreditAllocator::acquire.
  void reap_pending(bool all, const std::string& request_id = {});

  struct PendingTransfer {
    std::shared_ptr<RelayOperation> op;
    std::string request_id;
    std::chrono::steady_clock::time_point deadline;
  };

  Config config_;
  std::unique_ptr<SimpleScheduler> scheduler_;
  std::unique_ptr<StageControlPlane> control_plane_;
  CommRouter router_;

  std::map<std::string, OmniRequest> requests_;
  std::mutex requests_mu_;

  std::map<std::string, PendingTransfer> pending_ops_;
  std::mutex pending_mu_;

  std::atomic<uint64_t> next_object_seq_{0};
  std::atomic<bool> running_{false};
  std::atomic<bool> stopped_{false};
  std::mutex lifecycle_mu_;
  std::thread scheduler_thread_;
  std::thread abort_thread_;
  std::thread outbox_thread_;
  std::thread reaper_thread_;
};


// Tracks requests, submits at the entry stage, collects completions.
class Coordinator {
 public:
  Coordinator(std::string completion_endpoint, std::string abort_endpoint);
  ~Coordinator();

  // Register every stage before start; dynamic re-registration is rejected.
  // require_ready holds Submit locally until that stage sends stage_ready.
  // The default preserves compatibility for embedded legacy external stages.
  void register_stage(const std::string& name, const std::string& endpoint,
                      bool require_ready = false);
  // True after that stage sent stage_ready. False before, and false for a
  // stage that was not registered with require_ready.
  bool stage_ready(const std::string& name) const;
  void start();
  void run();  // blocks; pumps completions and stream chunks
  void stop();

  // Returns the request id. The coordinator generates one from a counter.
  // A caller-supplied id must stay unique for this coordinator's lifetime,
  // including after release: delayed abort frames can outlive the result record.
  // An explicit id of the form req-<decimal> advances the generator past it, so
  // a released explicit id is never reissued automatically. Automatic
  // allocation throws ProtocolError once the counter reaches its maximum.
  std::string submit(StagePayload payload, const std::string& entry_stage);
  bool wait(const std::string& request_id, double timeout_s);
  std::optional<RequestInfo> info(const std::string& request_id);
  // Drops a terminal request record. The caller owns the result by then.
  bool release(const std::string& request_id);
  size_t in_flight() const;
  void abort(const std::string& request_id);

  using StreamSink = std::function<void(const StreamMessage&)>;
  using CompleteSink = std::function<void(const RequestInfo&)>;
  // Atomically installs sinks only before the first run() call. Direct sink
  // access has the same before-run/quiescent-caller contract.
  std::pair<StreamSink, CompleteSink> exchange_sinks_before_run(
      StreamSink stream, CompleteSink complete);
  // Called on the coordinator thread for every stream chunk of a request.
  StreamSink on_stream;
  // Called when a request reaches a terminal state.
  CompleteSink on_complete;

 private:
  std::string next_request_id();
  void handle_event(const Message& msg);

  std::string completion_endpoint_;
  std::string abort_endpoint_;
  std::unique_ptr<CoordinatorControlPlane> control_plane_;
  std::map<std::string, std::string> endpoints_;
  std::set<std::string> stages_requiring_ready_;
  std::set<std::string> ready_stages_;
  std::map<std::string, std::deque<SubmitMessage>> pending_submits_;

  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::map<std::string, RequestInfo> requests_;
  std::atomic<uint64_t> counter_{0};
  std::atomic<bool> running_{false};
  bool pump_started_ = false;  // guarded by mu_
};

}  // namespace umminfer

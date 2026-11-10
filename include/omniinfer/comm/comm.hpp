// SPDX-License-Identifier: Apache-2.0
// ZMQ control plane, transport routing, and relay selection.
#pragma once

#include <zmq.h>

#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "omniinfer/proto/messages.hpp"
#include "omniinfer/relay/relay.hpp"

namespace omniinfer {

// Shared ZMQ context, one per process.
class ControlPlaneContext {
 public:
  static void* get();
  static void close();

 private:
  static void* ctx_;
  static std::mutex mu_;
};

class PushSocket {
 public:
  explicit PushSocket(std::string endpoint);
  ~PushSocket();
  void connect();
  void send(const Message& msg);
  void close();

 private:
  std::string endpoint_;
  void* socket_ = nullptr;
  std::mutex mu_;
};

class PullSocket {
 public:
  explicit PullSocket(std::string endpoint, bool bind);
  ~PullSocket();
  void start();
  Message recv();
  std::optional<Message> recv_nowait();
  // Waits up to timeout_ms; nullopt on timeout.
  std::optional<Message> recv_timeout(int timeout_ms);
  void close();

 private:
  std::string endpoint_;
  bool bind_;
  void* socket_ = nullptr;
  std::mutex mu_;
};

class PubSocket {
 public:
  explicit PubSocket(std::string endpoint);
  ~PubSocket();
  void bind();
  void publish(const AbortMessage& msg);
  void close();

 private:
  std::string endpoint_;
  void* socket_ = nullptr;
  std::mutex mu_;
};

class SubSocket {
 public:
  explicit SubSocket(std::string endpoint);
  ~SubSocket();
  void connect();
  AbortMessage recv();
  bool poll(int timeout_ms = 0);
  void close();

 private:
  std::string endpoint_;
  void* socket_ = nullptr;
  std::mutex mu_;
};

// Stage side: pull work, push to the coordinator and to next stages, subscribe
// to abort broadcasts.
class StageControlPlane {
 public:
  StageControlPlane(std::string stage_name, std::string recv_endpoint,
                    std::string coordinator_endpoint, std::string abort_endpoint);
  ~StageControlPlane();

  void start();
  Message recv();
  std::optional<Message> recv_nowait();
  void send_to_stage(const std::string& next_stage, const std::string& endpoint,
                     const Message& msg);
  void send_complete(const CompleteMessage& msg);
  void send_stream(const StreamMessage& msg);
  void send_admin_result(const AdminResultMessage& msg);
  AbortMessage recv_abort();
  // Non-blocking check so the abort loop can observe shutdown.
  bool recv_abort_ready(int timeout_ms);
  void close();

 private:
  std::string stage_name_;
  std::string recv_endpoint_;
  std::string coordinator_endpoint_;
  std::string abort_endpoint_;
  std::unique_ptr<PullSocket> recv_socket_;
  std::unique_ptr<PushSocket> coordinator_socket_;
  std::unique_ptr<SubSocket> abort_socket_;
  // Send paths run on the outbox thread and on the run loop; the map must be
  // guarded, and sockets are shared because close() may run concurrently.
  std::mutex next_stage_sockets_mu_;
  std::map<std::string, std::shared_ptr<PushSocket>> next_stage_sockets_;
};

// Coordinator side: push work to stages, pull completions, publish aborts.
class CoordinatorControlPlane {
 public:
  CoordinatorControlPlane(std::string completion_endpoint,
                          std::string abort_endpoint);
  ~CoordinatorControlPlane();

  void start();
  void submit_to_stage(const std::string& stage_name, const std::string& endpoint,
                       const Message& msg);
  Message recv_event();
  // Timeout-bounded receive so stop() can close the socket safely.
  std::optional<Message> recv_event_timeout(int timeout_ms);
  void send_admin(const std::string& stage_name, const std::string& endpoint,
                  const AdminMessage& msg);
  void broadcast_abort(const AbortMessage& msg);
  void send_shutdown(const std::string& stage_name, const std::string& endpoint);
  void close();

 private:
  std::shared_ptr<PushSocket> stage_socket(const std::string& stage_name,
                                           const std::string& endpoint);
  std::string completion_endpoint_;
  std::string abort_endpoint_;
  std::unique_ptr<PullSocket> completion_socket_;
  std::unique_ptr<PubSocket> abort_socket_;
  // httplib serves requests on several threads, so submits race on this map.
  std::mutex stage_sockets_mu_;
  std::map<std::string, std::shared_ptr<PushSocket>> stage_sockets_;
};

// Device facts the router needs. A CPU-only build reports cpu everywhere, so
// every GPU edge classifies as SHM, which is also what the CUDA fallback does.
struct Platform {
  std::string device_type = "cpu";
  TransportKind intra_node_transport = TransportKind::Shm;

  bool is_cuda_alike() const {
    return device_type == "cuda" || device_type == "rocm" ||
           device_type == "musa";
  }
  static const Platform& current();
};

// Classifies each edge to a physical mover and owns its relay instances.
class CommRouter {
 public:
  CommRouter(std::string stage_name, std::optional<int> gpu_id,
             std::optional<int> placement_gpu_id,
             std::set<std::string> same_process_targets,
             std::set<std::string> gpu_stage_names,
             std::map<std::string, std::vector<int>> stage_gpu_ids,
             std::set<std::string> remote_stage_names,
             std::map<std::string, Value> comm_config = {},
             std::unique_ptr<Relay> injected_relay = nullptr);

  bool self_is_gpu() const { return gpu_id_.has_value(); }
  bool is_local_object(const std::string& target) const;
  bool can_use_direct_cuda_ipc(const std::string& target) const;
  TransportKind intra_node_transport(const std::string& target);
  TransportKind outbound(const std::string& target);
  TransportKind physical_outbound(const std::string& target);
  TransportKind outbound_stream(const std::string& target, const Tensor& data);
  TransportKind inbound(const std::string& from_stage);
  TransportKind outbound_payload(const std::string& target, const Value& payload);
  Relay& relay(TransportKind kind);
  std::pair<TransportKind, Relay*> relay_for(const std::string& target);
  std::pair<TransportKind, Relay*> relay_for_payload(const std::string& target,
                                                     const Value& payload);
  std::pair<TransportKind, Relay*> relay_for_stream(const std::string& target,
                                                    const Tensor& data);
  Relay& inbound_relay(const std::string& from_stage);
  void cleanup(const std::string& request_id);
  void close();

  const std::string& stage_name() const { return stage_name_; }

 private:
  bool cuda_ipc_peer_available(const std::string& target);
  void warn_cuda_ipc_fallback(const std::string& target, int source_gpu,
                              int target_gpu, const std::string& reason);
  TransportKind note_transport(const std::string& direction,
                               const std::string& target, TransportKind kind);
  std::unique_ptr<Relay> build_relay(TransportKind kind);
  std::vector<Relay*> active_relays();

  std::string stage_name_;
  std::optional<int> gpu_id_;
  std::optional<int> placement_gpu_id_;
  std::set<std::string> same_process_targets_;
  std::set<std::string> gpu_stage_names_;
  std::map<std::string, std::vector<int>> stage_gpu_ids_;
  std::set<std::string> remote_stage_names_;
  std::set<std::string> direct_cuda_ipc_targets_;
  std::map<std::string, Value> comm_config_;
  std::unique_ptr<Relay> injected_relay_;
  std::mutex relays_mu_;
  std::map<TransportKind, std::unique_ptr<Relay>> relays_;
  // Send and receive paths query the same peer cache on different threads.
  std::mutex cuda_mu_;
  std::map<std::string, bool> cuda_ipc_peer_cache_;
};

// Set of device types in a payload, as tensor_devices() computes it.
std::set<std::string> tensor_devices(const Value& obj);

}  // namespace omniinfer

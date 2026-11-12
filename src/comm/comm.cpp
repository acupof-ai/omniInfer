// SPDX-License-Identifier: Apache-2.0
#include "umminfer/comm/comm.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <set>
#include <thread>

#if defined(UMMINFER_WITH_CUDA)
#include <cuda_runtime_api.h>
#endif

namespace umminfer {

void* ControlPlaneContext::ctx_ = nullptr;
std::mutex ControlPlaneContext::mu_;

void* ControlPlaneContext::get() {
  std::lock_guard<std::mutex> lock(mu_);
  if (ctx_ == nullptr) ctx_ = zmq_ctx_new();
  return ctx_;
}

void ControlPlaneContext::close() {
  std::lock_guard<std::mutex> lock(mu_);
  if (ctx_ != nullptr) {
    zmq_ctx_term(ctx_);
    ctx_ = nullptr;
  }
}

namespace {

void send_bytes(void* socket, const std::string& data) {
  int rc = zmq_send(socket, data.data(), data.size(), 0);
  if (rc < 0) throw ProtocolError("zmq_send failed: " + std::string(zmq_strerror(errno)));
}

std::string recv_bytes(void* socket) {
  zmq_msg_t msg;
  zmq_msg_init(&msg);
  int rc = zmq_msg_recv(&msg, socket, 0);
  if (rc < 0) {
    zmq_msg_close(&msg);
    throw ProtocolError("zmq recv failed: " + std::string(zmq_strerror(errno)));
  }
  std::string out(static_cast<char*>(zmq_msg_data(&msg)), zmq_msg_size(&msg));
  zmq_msg_close(&msg);
  return out;
}

void* make_socket(int type) {
  void* s = zmq_socket(ControlPlaneContext::get(), type);
  if (s == nullptr) throw ProtocolError("zmq_socket failed");
  // A lost peer must fail the send instead of blocking the stage forever.
  int linger = 0;
  zmq_setsockopt(s, ZMQ_LINGER, &linger, sizeof(linger));
  if (type == ZMQ_PUSH) {
    // Full send queue means the peer is gone or far behind. Fail the send
    // rather than block a worker that stop() is waiting to join.
    int sndtimeo = 5000;
    zmq_setsockopt(s, ZMQ_SNDTIMEO, &sndtimeo, sizeof(sndtimeo));
  }
  return s;
}

// Cache key. A stage that is re-registered at a new endpoint must get a new
// socket instead of reusing the one still pointed at the old address.
std::string socket_key(const std::string& stage, const std::string& endpoint) {
  return stage + '\x1f' + endpoint;
}

void must(int rc, const char* what) {
  if (rc != 0) throw ProtocolError(std::string(what) + " failed: " + zmq_strerror(errno));
}

}  // namespace

// ---- sockets -------------------------------------------------------------

PushSocket::PushSocket(std::string endpoint) : endpoint_(std::move(endpoint)) {}
PushSocket::~PushSocket() { close(); }

void PushSocket::connect() {
  std::lock_guard<std::mutex> lock(mu_);
  if (socket_ != nullptr) return;
  socket_ = make_socket(ZMQ_PUSH);
  must(zmq_connect(socket_, endpoint_.c_str()), "zmq_connect");
}

void PushSocket::send(const Message& msg) {
  std::lock_guard<std::mutex> lock(mu_);
  if (socket_ == nullptr) throw ProtocolError("Socket not connected");
  send_bytes(socket_, serialize_message(msg));
}

void PushSocket::close() {
  std::lock_guard<std::mutex> lock(mu_);
  if (socket_ != nullptr) {
    zmq_close(socket_);
    socket_ = nullptr;
  }
}

PullSocket::PullSocket(std::string endpoint, bool bind)
    : endpoint_(std::move(endpoint)), bind_(bind) {}
PullSocket::~PullSocket() { close(); }

void PullSocket::start() {
  std::lock_guard<std::mutex> lock(mu_);
  if (socket_ != nullptr) return;
  socket_ = make_socket(ZMQ_PULL);
  if (bind_) {
    must(zmq_bind(socket_, endpoint_.c_str()), "zmq_bind");
  } else {
    must(zmq_connect(socket_, endpoint_.c_str()), "zmq_connect");
  }
}

Message PullSocket::recv() {
  void* socket;
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (socket_ == nullptr) throw ProtocolError("Socket not started");
    socket = socket_;
  }
  return deserialize_message(recv_bytes(socket));
}

std::optional<Message> PullSocket::recv_timeout(int timeout_ms) {
  void* socket;
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (socket_ == nullptr) throw ProtocolError("Socket not started");
    socket = socket_;
  }
  zmq_pollitem_t item{socket, 0, ZMQ_POLLIN, 0};
  const int rc = zmq_poll(&item, 1, timeout_ms);
  if (rc < 0) {
    if (errno == EINTR) return std::nullopt;
    throw ProtocolError("zmq_poll failed: " + std::string(zmq_strerror(errno)));
  }
  // Only a POLLIN event means a message is ready; a blind recv would block.
  if (rc == 0 || (item.revents & ZMQ_POLLIN) == 0) return std::nullopt;
  return deserialize_message(recv_bytes(socket));
}

std::optional<Message> PullSocket::recv_nowait() {
  void* socket;
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (socket_ == nullptr) throw ProtocolError("Socket not started");
    socket = socket_;
  }
  zmq_pollitem_t item{socket, 0, ZMQ_POLLIN, 0};
  const int rc = zmq_poll(&item, 1, 0);
  if (rc < 0) {
    if (errno == EINTR) return std::nullopt;
    throw ProtocolError("zmq_poll failed: " + std::string(zmq_strerror(errno)));
  }
  if (rc == 0 || (item.revents & ZMQ_POLLIN) == 0) return std::nullopt;
  return deserialize_message(recv_bytes(socket));
}

void PullSocket::close() {
  std::lock_guard<std::mutex> lock(mu_);
  if (socket_ != nullptr) {
    zmq_close(socket_);
    socket_ = nullptr;
  }
}

PubSocket::PubSocket(std::string endpoint) : endpoint_(std::move(endpoint)) {}
PubSocket::~PubSocket() { close(); }

void PubSocket::bind() {
  std::lock_guard<std::mutex> lock(mu_);
  if (socket_ != nullptr) return;
  socket_ = make_socket(ZMQ_PUB);
  must(zmq_bind(socket_, endpoint_.c_str()), "zmq_bind");
  // Slow-joiner window: a SUB that connects after the first publish misses it.
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

void PubSocket::publish(const AbortMessage& msg) {
  std::lock_guard<std::mutex> lock(mu_);
  if (socket_ == nullptr) throw ProtocolError("Socket not bound");
  send_bytes(socket_, serialize_message(Message{msg}));
}

void PubSocket::close() {
  std::lock_guard<std::mutex> lock(mu_);
  if (socket_ != nullptr) {
    zmq_close(socket_);
    socket_ = nullptr;
  }
}

SubSocket::SubSocket(std::string endpoint) : endpoint_(std::move(endpoint)) {}
SubSocket::~SubSocket() { close(); }

void SubSocket::connect() {
  std::lock_guard<std::mutex> lock(mu_);
  if (socket_ != nullptr) return;
  socket_ = make_socket(ZMQ_SUB);
  must(zmq_connect(socket_, endpoint_.c_str()), "zmq_connect");
  must(zmq_setsockopt(socket_, ZMQ_SUBSCRIBE, "", 0), "zmq_setsockopt");
}

AbortMessage SubSocket::recv() {
  void* socket;
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (socket_ == nullptr) throw ProtocolError("Socket not connected");
    socket = socket_;
  }
  Message msg = deserialize_message(recv_bytes(socket));
  if (auto* abort = std::get_if<AbortMessage>(&msg)) return *abort;
  throw ProtocolError("Expected AbortMessage");
}

bool SubSocket::poll(int timeout_ms) {
  std::lock_guard<std::mutex> lock(mu_);
  if (socket_ == nullptr) throw ProtocolError("Socket not connected");
  zmq_pollitem_t item{socket_, 0, ZMQ_POLLIN, 0};
  const int rc = zmq_poll(&item, 1, timeout_ms);
  if (rc < 0) {
    if (errno == EINTR) return false;
    throw ProtocolError("zmq_poll failed: " + std::string(zmq_strerror(errno)));
  }
  return rc > 0 && (item.revents & ZMQ_POLLIN) != 0;
}

void SubSocket::close() {
  std::lock_guard<std::mutex> lock(mu_);
  if (socket_ != nullptr) {
    zmq_close(socket_);
    socket_ = nullptr;
  }
}

// ---- stage control plane -------------------------------------------------

StageControlPlane::StageControlPlane(std::string stage_name,
                                     std::string recv_endpoint,
                                     std::string coordinator_endpoint,
                                     std::string abort_endpoint)
    : stage_name_(std::move(stage_name)),
      recv_endpoint_(std::move(recv_endpoint)),
      coordinator_endpoint_(std::move(coordinator_endpoint)),
      abort_endpoint_(std::move(abort_endpoint)) {}
StageControlPlane::~StageControlPlane() { close(); }

void StageControlPlane::start() {
  recv_socket_ = std::make_unique<PullSocket>(recv_endpoint_, true);
  recv_socket_->start();
  coordinator_socket_ = std::make_unique<PushSocket>(coordinator_endpoint_);
  coordinator_socket_->connect();
  abort_socket_ = std::make_unique<SubSocket>(abort_endpoint_);
  abort_socket_->connect();
}

Message StageControlPlane::recv() {
  if (!recv_socket_) throw ProtocolError("Control plane not started");
  Message msg = recv_socket_->recv();
  const bool accepted =
      std::holds_alternative<DataReadyMessage>(msg) ||
      std::holds_alternative<StreamChunkBatchMessage>(msg) ||
      std::holds_alternative<DataAckMessage>(msg) ||
      std::holds_alternative<SubmitMessage>(msg) ||
      std::holds_alternative<ShutdownMessage>(msg) ||
      std::holds_alternative<FreezeGCMessage>(msg) ||
      std::holds_alternative<ProfilerStartMessage>(msg) ||
      std::holds_alternative<ProfilerStopMessage>(msg) ||
      std::holds_alternative<AdminMessage>(msg) ||
      std::holds_alternative<AbortMessage>(msg);
  if (!accepted) {
    throw ProtocolError(std::string("Unexpected message type: ") +
                        message_type_name(msg));
  }
  return msg;
}

void StageControlPlane::send_to_stage(const std::string& next_stage,
                                      const std::string& endpoint,
                                      const Message& msg) {
  std::shared_ptr<PushSocket> sock;
  {
    std::lock_guard<std::mutex> lock(next_stage_sockets_mu_);
    const std::string key = socket_key(next_stage, endpoint);
    auto it = next_stage_sockets_.find(key);
    if (it == next_stage_sockets_.end()) {
      auto created = std::make_shared<PushSocket>(endpoint);
      created->connect();
      it = next_stage_sockets_.emplace(key, std::move(created)).first;
    }
    sock = it->second;
  }
  // The send runs outside the map lock; PushSocket guards its own socket.
  sock->send(msg);
}

void StageControlPlane::send_ready(const StageReadyMessage& msg) {
  if (!coordinator_socket_) throw ProtocolError("Control plane not started");
  coordinator_socket_->send(Message{msg});
}

void StageControlPlane::send_complete(const CompleteMessage& msg) {
  if (!coordinator_socket_) throw ProtocolError("Control plane not started");
  coordinator_socket_->send(Message{msg});
}

void StageControlPlane::send_stream(const StreamMessage& msg) {
  if (!coordinator_socket_) throw ProtocolError("Control plane not started");
  coordinator_socket_->send(Message{msg});
}

void StageControlPlane::send_admin_result(const AdminResultMessage& msg) {
  if (!coordinator_socket_) throw ProtocolError("Control plane not started");
  coordinator_socket_->send(Message{msg});
}

std::optional<Message> StageControlPlane::recv_nowait() {
  if (!recv_socket_) throw ProtocolError("Control plane not started");
  return recv_socket_->recv_timeout(0);
}

AbortMessage StageControlPlane::recv_abort() {
  if (!abort_socket_) throw ProtocolError("Control plane not started");
  return abort_socket_->recv();
}

bool StageControlPlane::poll_abort(int timeout_ms) {
  if (!abort_socket_) throw ProtocolError("Control plane not started");
  return abort_socket_->poll(timeout_ms);
}

void StageControlPlane::close() {
  if (recv_socket_) recv_socket_->close();
  if (coordinator_socket_) coordinator_socket_->close();
  if (abort_socket_) abort_socket_->close();
  std::map<std::string, std::shared_ptr<PushSocket>> sockets;
  {
    std::lock_guard<std::mutex> lock(next_stage_sockets_mu_);
    sockets.swap(next_stage_sockets_);
  }
  for (auto& [name, sock] : sockets) sock->close();
}

// ---- coordinator control plane -------------------------------------------

CoordinatorControlPlane::CoordinatorControlPlane(std::string completion_endpoint,
                                                 std::string abort_endpoint)
    : completion_endpoint_(std::move(completion_endpoint)),
      abort_endpoint_(std::move(abort_endpoint)) {}
CoordinatorControlPlane::~CoordinatorControlPlane() { close(); }

void CoordinatorControlPlane::start() {
  completion_socket_ = std::make_unique<PullSocket>(completion_endpoint_, true);
  completion_socket_->start();
  abort_socket_ = std::make_unique<PubSocket>(abort_endpoint_);
  abort_socket_->bind();
}

std::shared_ptr<PushSocket> CoordinatorControlPlane::stage_socket(
    const std::string& stage_name, const std::string& endpoint) {
  std::lock_guard<std::mutex> lock(stage_sockets_mu_);
  const std::string key = socket_key(stage_name, endpoint);
  auto it = stage_sockets_.find(key);
  if (it == stage_sockets_.end()) {
    auto created = std::make_shared<PushSocket>(endpoint);
    created->connect();
    it = stage_sockets_.emplace(key, std::move(created)).first;
  }
  return it->second;
}

void CoordinatorControlPlane::submit_to_stage(const std::string& stage_name,
                                              const std::string& endpoint,
                                              const Message& msg) {
  stage_socket(stage_name, endpoint)->send(msg);
}

Message CoordinatorControlPlane::recv_event() {
  if (!completion_socket_) throw ProtocolError("Control plane not started");
  Message msg = completion_socket_->recv();
  if (std::holds_alternative<StageReadyMessage>(msg) ||
      std::holds_alternative<CompleteMessage>(msg) ||
      std::holds_alternative<StreamMessage>(msg) ||
      std::holds_alternative<AdminResultMessage>(msg)) {
    return msg;
  }
  throw ProtocolError("Expected StageReadyMessage, CompleteMessage, StreamMessage, or AdminResultMessage");
}

std::optional<Message> CoordinatorControlPlane::recv_event_timeout(int timeout_ms) {
  if (!completion_socket_) throw ProtocolError("Control plane not started");
  std::optional<Message> msg = completion_socket_->recv_timeout(timeout_ms);
  if (!msg) return std::nullopt;
  if (std::holds_alternative<StageReadyMessage>(*msg) ||
      std::holds_alternative<CompleteMessage>(*msg) ||
      std::holds_alternative<StreamMessage>(*msg) ||
      std::holds_alternative<AdminResultMessage>(*msg)) {
    return msg;
  }
  throw ProtocolError("Expected StageReadyMessage, CompleteMessage, StreamMessage, or AdminResultMessage");
}

void CoordinatorControlPlane::send_admin(const std::string& stage_name,
                                         const std::string& endpoint,
                                         const AdminMessage& msg) {
  submit_to_stage(stage_name, endpoint, Message{msg});
}

void CoordinatorControlPlane::broadcast_abort(const AbortMessage& msg) {
  if (!abort_socket_) throw ProtocolError("Control plane not started");
  std::vector<std::shared_ptr<PushSocket>> sockets;
  {
    std::lock_guard<std::mutex> lock(stage_sockets_mu_);
    sockets.reserve(stage_sockets_.size());
    for (const auto& [key, socket] : stage_sockets_) {
      (void)key;
      sockets.push_back(socket);
    }
  }
  std::exception_ptr failure;
  for (const auto& socket : sockets) {
    try {
      socket->send(Message{msg});
    } catch (...) {
      failure = std::current_exception();
    }
  }
  // PUB remains a low-latency supplement. PUSH is the reliable ordered path,
  // including aborts queued before a stage binds its input socket.
  abort_socket_->publish(msg);
  if (failure) std::rethrow_exception(failure);
}

void CoordinatorControlPlane::send_shutdown(const std::string& stage_name,
                                            const std::string& endpoint) {
  stage_socket(stage_name, endpoint)->send(Message{ShutdownMessage{}});
}

void CoordinatorControlPlane::close() {
  if (completion_socket_) completion_socket_->close();
  if (abort_socket_) abort_socket_->close();
  std::map<std::string, std::shared_ptr<PushSocket>> sockets;
  {
    std::lock_guard<std::mutex> lock(stage_sockets_mu_);
    sockets.swap(stage_sockets_);
  }
  for (auto& [name, sock] : sockets) sock->close();
}

// ---- platform ------------------------------------------------------------

const Platform& Platform::current() {
  static const Platform platform = [] {
    Platform detected;
#if defined(UMMINFER_WITH_CUDA)
    int count = 0;
    if (cudaGetDeviceCount(&count) == cudaSuccess && count > 0) {
      detected.device_type = "cuda";
      detected.intra_node_transport = TransportKind::CudaIpc;
    } else {
      // A driver without devices is not an error; stay on the CPU path.
      (void)cudaGetLastError();
    }
#endif
    return detected;
  }();
  return platform;
}

namespace {

bool cuda_ipc_disabled_by_env() {
  const char* value = std::getenv("UMMINFER_DISABLE_CUDA_IPC");
  return value != nullptr && *value != '\0' && std::string(value) != "0";
}

}  // namespace

// ---- router --------------------------------------------------------------

CommRouter::CommRouter(std::string stage_name, std::optional<int> gpu_id,
                       std::optional<int> placement_gpu_id,
                       std::set<std::string> same_process_targets,
                       std::set<std::string> gpu_stage_names,
                       std::map<std::string, std::vector<int>> stage_gpu_ids,
                       std::set<std::string> remote_stage_names,
                       std::map<std::string, Value> comm_config,
                       std::shared_ptr<Relay> injected_relay)
    : stage_name_(std::move(stage_name)),
      gpu_id_(gpu_id),
      placement_gpu_id_(placement_gpu_id ? placement_gpu_id : gpu_id),
      same_process_targets_(std::move(same_process_targets)),
      gpu_stage_names_(std::move(gpu_stage_names)),
      stage_gpu_ids_(std::move(stage_gpu_ids)),
      remote_stage_names_(std::move(remote_stage_names)),
      comm_config_(std::move(comm_config)),
      injected_relay_(std::move(injected_relay)) {
  for (const auto& [name, gpus] : stage_gpu_ids_) {
    if (gpu_id_ == placement_gpu_id_ && gpus.size() == 1 &&
        placement_gpu_id_ && gpus[0] == *placement_gpu_id_ &&
        !same_process_targets_.count(name) && !remote_stage_names_.count(name)) {
      direct_cuda_ipc_targets_.insert(name);
    }
  }
}

bool CommRouter::is_local_object(const std::string& target) const {
  return same_process_targets_.count(target) != 0;
}

bool CommRouter::can_use_direct_cuda_ipc(const std::string& target) const {
  return direct_cuda_ipc_targets_.count(target) != 0 &&
         Platform::current().intra_node_transport == TransportKind::CudaIpc;
}

bool CommRouter::cuda_ipc_peer_available(const std::string& target) {
  std::lock_guard<std::mutex> lock(cuda_mu_);
  // The override wins over a warm cache, same-GPU edges and missing mappings.
  if (cuda_ipc_disabled_by_env()) {
    auto it = stage_gpu_ids_.find(target);
    const int target_gpu = (it != stage_gpu_ids_.end() && !it->second.empty())
                               ? it->second[0]
                               : -1;
    warn_cuda_ipc_fallback(target, placement_gpu_id_.value_or(-1), target_gpu,
                           "disabled by UMMINFER_DISABLE_CUDA_IPC");
    cuda_ipc_peer_cache_[target] = false;
    return false;
  }
  auto cached = cuda_ipc_peer_cache_.find(target);
  if (cached != cuda_ipc_peer_cache_.end()) return cached->second;
  auto it = stage_gpu_ids_.find(target);
  if (it == stage_gpu_ids_.end()) return true;
  if (it->second.empty() || !placement_gpu_id_) return true;

  const int source_gpu = *placement_gpu_id_;
  const int target_gpu = it->second[0];
  // The relay allocates on gpu_id_ but the peer decision would use
  // placement_gpu_id_. Reject the remap before the same-GPU shortcut, or a
  // remapped source would skip the peer query entirely.
  if (!gpu_id_ || *gpu_id_ != source_gpu) {
    warn_cuda_ipc_fallback(target, source_gpu, target_gpu,
                           "source process uses a remapped GPU");
    cuda_ipc_peer_cache_[target] = false;
    return false;
  }
  if (source_gpu == target_gpu) {
    // Cross-process IPC on one GPU needs no peer access.
    cuda_ipc_peer_cache_[target] = true;
    return true;
  }
#if defined(UMMINFER_WITH_CUDA)
  // The receiver opens the sender's handle from the target device, so both
  // directions must be available. Cache only a complete positive answer.
  auto accessible = [](int from, int to, std::string& reason) {
    int can = 0;
    const cudaError_t rc = cudaDeviceCanAccessPeer(&can, from, to);
    if (rc != cudaSuccess) {
      reason = cudaGetErrorString(rc);
      (void)cudaGetLastError();
      return false;
    }
    if (can == 0) {
      reason = "peer access not supported";
      return false;
    }
    return true;
  };
  std::string reason;
  if (!accessible(source_gpu, target_gpu, reason)) {
    warn_cuda_ipc_fallback(target, source_gpu, target_gpu, reason);
    cuda_ipc_peer_cache_[target] = false;
    return false;
  }
  if (!accessible(target_gpu, source_gpu, reason)) {
    warn_cuda_ipc_fallback(target, source_gpu, target_gpu,
                           "reverse direction unavailable: " + reason);
    cuda_ipc_peer_cache_[target] = false;
    return false;
  }
  cuda_ipc_peer_cache_[target] = true;
  return true;
#else
  warn_cuda_ipc_fallback(target, source_gpu, target_gpu,
                         "peer query unavailable: built without CUDA");
  cuda_ipc_peer_cache_[target] = false;
  return false;
#endif
}

void CommRouter::warn_cuda_ipc_fallback(const std::string& target, int source_gpu,
                                        int target_gpu, const std::string& reason) {
  static std::mutex warned_mu;
  static std::set<std::string> warned;
  const std::string key = stage_name_ + "|" + target + "|" +
                          std::to_string(source_gpu) + "|" +
                          std::to_string(target_gpu) + "|" + reason;
  // Several routers can race on the process-wide dedup set.
  std::lock_guard<std::mutex> lock(warned_mu);
  if (!warned.insert(key).second) return;
  std::cerr << "CommRouter: using SHM for CUDA edge " << stage_name_
            << "(gpu=" << source_gpu << ") -> " << target << "(gpu=" << target_gpu
            << "): " << reason << "\n";
}

TransportKind CommRouter::intra_node_transport(const std::string& target) {
  const TransportKind transport = Platform::current().intra_node_transport;
  if (transport == TransportKind::CudaIpc && !cuda_ipc_peer_available(target)) {
    return TransportKind::Shm;
  }
  return transport;
}

TransportKind CommRouter::note_transport(const std::string& direction,
                                         const std::string& target,
                                         TransportKind kind) {
  // Comm tracing is a profiler feature; the port keeps the call sites so the
  // hook has one place to land.
  (void)direction;
  (void)target;
  return kind;
}

TransportKind CommRouter::outbound(const std::string& target) {
  if (same_process_targets_.count(target)) return TransportKind::LocalObject;
  return note_transport("outbound", target, physical_outbound(target));
}

TransportKind CommRouter::physical_outbound(const std::string& target) {
  if (remote_stage_names_.count(target)) return TransportKind::Mooncake;
  if (self_is_gpu() && gpu_stage_names_.count(target)) {
    return intra_node_transport(target);
  }
  return TransportKind::Shm;
}

TransportKind CommRouter::outbound_stream(const std::string& target,
                                          const Tensor& data) {
  if (remote_stage_names_.count(target)) {
    return note_transport("stream", target, TransportKind::Mooncake);
  }
  const std::string& device_type = Platform::current().device_type;
  if (data.device.substr(0, device_type.size()) != device_type) {
    return note_transport("stream", target, TransportKind::Shm);
  }
  if (self_is_gpu() && gpu_stage_names_.count(target)) {
    return note_transport("stream", target, intra_node_transport(target));
  }
  throw ProtocolError(device_type + " stream chunk cannot be sent from " +
                      stage_name_ + " to non-GPU target " + target);
}

TransportKind CommRouter::inbound(const std::string& from_stage) {
  if (remote_stage_names_.count(from_stage)) {
    return note_transport("inbound", from_stage, TransportKind::Mooncake);
  }
  if (self_is_gpu() && gpu_stage_names_.count(from_stage)) {
    // The sender picks the relay with the same peer check; a plain platform
    // lookup here would build a relay kind the sender never used.
    return note_transport("inbound", from_stage,
                          intra_node_transport(from_stage));
  }
  return note_transport("inbound", from_stage, TransportKind::Shm);
}

TransportKind CommRouter::outbound_payload(const std::string& target,
                                           const Value& payload) {
  if (remote_stage_names_.count(target)) {
    return note_transport("payload", target, TransportKind::Mooncake);
  }
  const std::set<std::string> devices = tensor_devices(payload);
  const std::string device_type = Platform::current().device_type;
  TransportKind kind;
  if (devices.empty() || devices == std::set<std::string>{"cpu"}) {
    kind = TransportKind::Shm;
  } else if (devices.count(device_type) &&
             (devices.size() == 1 ||
              (devices.size() == 2 && devices.count("cpu")))) {
    kind = (self_is_gpu() && gpu_stage_names_.count(target))
               ? intra_node_transport(target)
               : TransportKind::Shm;
  } else {
    std::string text;
    for (const auto& d : devices) text += (text.empty() ? "" : ", ") + d;
    throw ProtocolError("mixed or unsupported tensor devices in payload: {" + text +
                        "}");
  }
  return note_transport("payload", target, kind);
}

Relay& CommRouter::relay(TransportKind kind) {
  if (kind == TransportKind::LocalObject) {
    throw ProtocolError("local_object has no relay");
  }
  std::lock_guard<std::mutex> lock(relays_mu_);
  if (closed_) throw ProtocolError("comm router is closed");
  if (injected_relay_) return *injected_relay_;
  auto it = relays_.find(kind);
  if (it == relays_.end()) {
    it = relays_.emplace(kind, build_relay(kind)).first;
  }
  return *it->second;
}

Relay& CommRouter::relay_for_target(TransportKind kind,
                                    const std::string& target) {
  if (kind != TransportKind::CudaIpc) return relay(kind);
  std::lock_guard<std::mutex> lock(relays_mu_);
  if (closed_) throw ProtocolError("comm router is closed");
  if (injected_relay_) return *injected_relay_;
  auto it = target_relays_.find(target);
  if (it == target_relays_.end()) {
    it = target_relays_.emplace(target, build_relay(kind)).first;
  }
  return *it->second;
}

std::pair<TransportKind, Relay*> CommRouter::relay_for(const std::string& target) {
  const TransportKind kind = outbound(target);
  if (kind == TransportKind::LocalObject) {
    throw ProtocolError("same-process target " + target +
                        " has no relay transport; use local-object dispatch");
  }
  return {kind, &relay_for_target(kind, target)};
}

std::pair<TransportKind, Relay*> CommRouter::relay_for_payload(
    const std::string& target, const Value& payload) {
  const TransportKind kind = outbound_payload(target, payload);
  if (kind == TransportKind::LocalObject) {
    throw ProtocolError("local_object has no relay");
  }
  return {kind, &relay_for_target(kind, target)};
}

std::pair<TransportKind, Relay*> CommRouter::relay_for_payload(
    const std::string& target, const Value& payload, int64_t packed_bytes) {
  TransportKind kind = outbound_payload(target, payload);
  if (kind == TransportKind::LocalObject) {
    throw ProtocolError("local_object has no relay");
  }
  // The CUDA relay rejects an oversized block before any device allocation.
  // Pick SHM here instead so the transfer still completes.
  if (kind == TransportKind::CudaIpc && packed_bytes > cuda_slot_bytes()) {
    warn_cuda_ipc_fallback(target, placement_gpu_id_.value_or(-1),
                           stage_gpu_ids_.count(target) &&
                                   !stage_gpu_ids_.at(target).empty()
                               ? stage_gpu_ids_.at(target)[0]
                               : -1,
                           "payload exceeds the CUDA slot limit; using SHM");
    kind = TransportKind::Shm;
  }
  return {kind, &relay_for_target(kind, target)};
}

int CommRouter::slot_size_mb() const {
  auto it = comm_config_.find("slot_size_mb");
  const int64_t mb = it == comm_config_.end() ? 512 : it->second.as_int();
  if (mb > std::numeric_limits<int>::max()) {
    throw ProtocolError("slot_size_mb " + std::to_string(mb) +
                        " is out of int range");
  }
  return mb > 0 ? static_cast<int>(mb) : 512;
}

int64_t CommRouter::cuda_slot_bytes() const {
  return static_cast<int64_t>(slot_size_mb()) * 1024 * 1024;
}

std::pair<TransportKind, Relay*> CommRouter::relay_for_stream(
    const std::string& target, const Tensor& data) {
  const TransportKind kind = outbound_stream(target, data);
  if (kind == TransportKind::LocalObject) {
    throw ProtocolError("same-process stream target " + target +
                        " has no relay transport; use local-object dispatch");
  }
  return {kind, &relay_for_target(kind, target)};
}

Relay& CommRouter::inbound_relay(const std::string& from_stage) {
  return relay(inbound(from_stage));
}

Relay& CommRouter::inbound_relay(const std::string& from_stage,
                                 TransportKind advertised_transport) {
  (void)from_stage;
  if (advertised_transport != TransportKind::Shm &&
      advertised_transport != TransportKind::CudaIpc) {
    throw ProtocolError("unsupported inbound relay transport");
  }
  // The sender already applied the P2P and fallback decision. Recomputing it on
  // the receiver can select a different relay when placement maps differ.
  return relay(advertised_transport);
}

void CommRouter::cleanup(const std::string& request_id) {
  for (Relay* relay_ptr : active_relays()) {
    try {
      relay_ptr->cleanup(request_id);
    } catch (...) {
    }
  }
}

void CommRouter::close() {
  std::vector<Relay*> owned;
  {
    std::lock_guard<std::mutex> lock(relays_mu_);
    if (closed_) return;
    closed_ = true;
    // Keep the allocations: callers may still hold raw relay pointers after
    // close. The relays are closed here and destroyed with the router.
    for (auto& [kind, relay_ptr] : relays_) owned.push_back(relay_ptr.get());
    for (auto& [target, relay_ptr] : target_relays_) {
      owned.push_back(relay_ptr.get());
    }
  }
  if (injected_relay_) injected_relay_->close();
  for (Relay* relay_ptr : owned) {
    try {
      relay_ptr->close();
    } catch (...) {
    }
  }
}

std::vector<Relay*> CommRouter::active_relays() {
  if (injected_relay_) return {injected_relay_.get()};
  std::lock_guard<std::mutex> lock(relays_mu_);
  std::vector<Relay*> out;
  for (auto& [kind, relay_ptr] : relays_) out.push_back(relay_ptr.get());
  for (auto& [target, relay_ptr] : target_relays_) out.push_back(relay_ptr.get());
  return out;
}

std::unique_ptr<Relay> CommRouter::build_relay(TransportKind kind) {
  std::map<std::string, Value> cfg;
  auto it = comm_config_.find("worker_id");
  cfg["engine_id"] = Value(it != comm_config_.end()
                               ? it->second.as_str()
                               : stage_name_ + "_relay");
  auto number = [&](const char* key, int fallback) {
    auto found = comm_config_.find(key);
    cfg[key] = Value(found != comm_config_.end()
                         ? static_cast<int>(found->second.as_int())
                         : fallback);
  };
  cfg["slot_size_mb"] = Value(slot_size_mb());
  number("credits", 2);
  switch (kind) {
    case TransportKind::Shm:
      return create_relay("shm", cfg);
    case TransportKind::CudaIpc:
      if (!gpu_id_) {
        throw ProtocolError("cuda_ipc relay requested for non-GPU stage " +
                            stage_name_);
      }
      // The relay allocates on the GPU this stage was placed on.
      cfg["device"] = Value(*gpu_id_);
      return create_relay("cuda_ipc", cfg);
    case TransportKind::Mooncake:
      if (gpu_id_ && !Platform::current().is_cuda_alike()) {
        throw ProtocolError("cross-node transfer from " + stage_name_ +
                            " needs a mooncake relay");
      }
      return create_relay("mooncake", cfg);
    case TransportKind::LocalObject:
      break;
  }
  throw ProtocolError("CommRouter cannot build a relay for this transport");
}

std::set<std::string> tensor_devices(const Value& obj) {
  std::set<std::string> out;
  switch (obj.type()) {
    case Value::Type::Tensor_: {
      const std::string& device = obj.as_tensor().device;
      const size_t colon = device.find(':');
      out.insert(colon == std::string::npos ? device : device.substr(0, colon));
      break;
    }
    case Value::Type::Array:
      for (const auto& item : obj.as_array()) {
        for (const auto& d : tensor_devices(item)) out.insert(d);
      }
      break;
    case Value::Type::Map:
      for (const auto& [key, item] : obj.as_map()) {
        for (const auto& d : tensor_devices(item)) out.insert(d);
      }
      break;
    default:
      break;
  }
  return out;
}

}  // namespace umminfer

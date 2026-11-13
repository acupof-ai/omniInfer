// SPDX-License-Identifier: Apache-2.0
#include "umminfer/pipeline/pipeline.hpp"

#include <chrono>
#include <exception>
#include <iostream>
#include <limits>

namespace umminfer {
namespace {

// One DataRef per payload. The payload is msgpack bytes, so every edge moves one
// contiguous block through the relay and the ref names it.
DataRef make_payload_ref(const Value& payload_value, TransportKind transport,
                         const Value& relay_info, int64_t nbytes,
                         const std::string& object_id) {
  DataRef ref;
  ref.version = 1;
  ref.kind = DataKind::StagePayload;
  ref.object_id = object_id;
  ref.transport = transport;
  ref.layout = DataLayout::PackedTensors;
  ref.buffer.transport = transport;
  ref.buffer.info = relay_info;
  ref.buffer.length = nbytes;
  TensorMeta meta;
  meta.path = "";
  meta.shape = {nbytes};
  meta.dtype = "uint8";
  meta.device = "cpu";
  meta.offset = 0;
  meta.size = nbytes;
  ref.tensors.push_back(std::move(meta));
  (void)payload_value;
  return ref;
}

std::string object_id_for(const std::string& request_id, uint64_t n) {
  return request_id + ":" + std::to_string(n);
}

// Parses an id in the generated namespace, "req-" followed by decimal digits.
// Leading zeros are accepted. Any other shape, including a number that does
// not fit in 64 bits, returns nullopt so the caller leaves the counter alone.
std::optional<uint64_t> generated_id_number(const std::string& id) {
  constexpr size_t kPrefixSize = 4;  // "req-"
  if (id.size() <= kPrefixSize || id.compare(0, kPrefixSize, "req-") != 0) {
    return std::nullopt;
  }
  uint64_t value = 0;
  for (size_t index = kPrefixSize; index < id.size(); ++index) {
    const char digit = id[index];
    if (digit < '0' || digit > '9') return std::nullopt;
    const uint64_t add = static_cast<uint64_t>(digit - '0');
    if (value > (std::numeric_limits<uint64_t>::max() - add) / 10) {
      return std::nullopt;
    }
    value = value * 10 + add;
  }
  return value;
}

// Rejects a DataRef before the receiver allocates or touches a relay. The
// length is untrusted: a peer supplies it in a control message.
void validate_data_ref(const DataRef& ref, int64_t max_payload_bytes) {
  // Only the two host-to-host movers exist here. Reject any other transport
  // before the untrusted length can reach an allocation.
  const bool supported_transport =
      (ref.transport == TransportKind::Shm ||
       ref.transport == TransportKind::CudaIpc) &&
      (ref.buffer.transport == TransportKind::Shm ||
       ref.buffer.transport == TransportKind::CudaIpc);
  if (!supported_transport) {
    throw ProtocolError("unsupported inbound relay transport");
  }
  const int64_t length = ref.buffer.length;
  if (length < 0) {
    throw ProtocolError("data_ref buffer length is negative");
  }
  if (max_payload_bytes > 0 && length > max_payload_bytes) {
    throw ProtocolError("data_ref length " + std::to_string(length) +
                        " exceeds the configured payload limit " +
                        std::to_string(max_payload_bytes));
  }
  if (ref.transport != ref.buffer.transport) {
    throw ProtocolError("data_ref transport differs from the buffer transport");
  }
  int64_t total = 0;
  for (const auto& tensor : ref.tensors) {
    if (tensor.size < 0 || tensor.offset < 0) {
      throw ProtocolError("tensor metadata has a negative size or offset");
    }
    if (tensor.offset > length || tensor.size > length - tensor.offset) {
      throw ProtocolError("tensor metadata exceeds the data_ref buffer length");
    }
    if (tensor.size > length - total) {
      throw ProtocolError("tensor metadata sizes exceed the data_ref buffer length");
    }
    total += tensor.size;
  }
  if (ref.tensors.size() == 1 &&
      ref.tensors[0].offset + ref.tensors[0].size != length) {
    throw ProtocolError("tensor size does not match the data_ref buffer length");
  }
  // Both host and device movers copy transfer_info.size bytes. A mismatch can
  // under-copy, add trailing zeros, or defer a negative size to a later failure.
  if (ref.buffer.transport == TransportKind::Shm ||
      ref.buffer.transport == TransportKind::CudaIpc) {
    const Value& info = require_field(ref.buffer.info, "transfer_info");
    const int64_t declared = require_non_negative_int(info, "size");
    if (declared != length) {
      throw ProtocolError("transfer size " + std::to_string(declared) +
                          " does not match the data_ref length " +
                          std::to_string(length));
    }
  }
}

}  // namespace

// ---- stage ---------------------------------------------------------------

Stage::Stage(Config config, ComputeFn compute_fn)
    : config_(std::move(config)),
      scheduler_(std::make_unique<SimpleScheduler>(std::move(compute_fn))),
      router_(config_.name, config_.gpu_id, config_.gpu_id, {},
              [&] {
                std::set<std::string> names;
                for (const auto& [name, devices] : config_.stage_gpu_ids) {
                  (void)devices;
                  names.insert(name);
                }
                return names;
              }(),
              config_.stage_gpu_ids, {}, config_.comm_config,
              config_.injected_relay) {}

Stage::~Stage() {
  stop();
  // Closing the relay map destroys in-flight operations. The caller must have
  // joined run() by now, so no thread can still hold an inbound relay.
  router_.close();
}

void Stage::start() {
  std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mu_);
  if (running_ || stopped_) return;
  control_plane_ = std::make_unique<StageControlPlane>(
      config_.name, config_.recv_endpoint, config_.coordinator_endpoint,
      config_.abort_endpoint);
  control_plane_->start();
  running_ = true;

  // Settle controls queued before start before compute can consume a Submit.
  while (true) {
    std::optional<Message> msg;
    try {
      msg = control_plane_->recv_nowait();
    } catch (const ProtocolError&) {
      // A malformed frame is consumed; it must not prevent stage readiness.
      continue;
    }
    if (!msg) break;
    handle_message(std::move(*msg));
  }
  if (!running_) return;

  scheduler_thread_ = std::thread([this] { scheduler_->start(); });
  outbox_thread_ = std::thread([this] { pump_outbox(); });
  reaper_thread_ = std::thread([this] {
    while (running_) {
      reap_pending(false);
      scheduler_->reap_aborted();
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  });
  abort_thread_ = std::thread([this] {
    while (running_) {
      try {
        if (!control_plane_ || !control_plane_->poll_abort(50)) continue;
        const std::string id = control_plane_->recv_abort().request_id;
        scheduler_->abort(id);
        reap_pending(false, id);
        // The scheduler drops an aborted request without an outbox result, so
        // the request record and the coordinator still need a terminal event.
        fail_request(id, "request aborted");
      } catch (...) {
        if (!running_) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
      }
    }
  });
  if (config_.announce_ready) {
    control_plane_->send_ready(StageReadyMessage{config_.name});
  }
}

void Stage::run() {
  if (!running_ && !stopped_) start();
  while (running_) {
    std::optional<Message> msg;
    try {
      msg = control_plane_->recv_nowait();
    } catch (const ProtocolError&) {
      if (!running_) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      continue;
    }
    if (!msg) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      continue;
    }
    handle_message(std::move(*msg));
  }
}

void Stage::handle_message(Message msg) {
  std::visit(
      [this](auto&& value) {
        using T = std::decay_t<decltype(value)>;
        std::string request_id;
        if constexpr (requires { value.request_id; }) request_id = value.request_id;
        try {
          if constexpr (std::is_same_v<T, SubmitMessage>) {
            on_submit(value);
          } else if constexpr (std::is_same_v<T, DataReadyMessage>) {
            on_data_ready(value);
          } else if constexpr (std::is_same_v<T, DataAckMessage>) {
            on_data_ack(value);
          } else if constexpr (std::is_same_v<T, AbortMessage>) {
            scheduler_->abort(value.request_id);
            reap_pending(false, value.request_id);
            fail_request(value.request_id, "request aborted");
          } else if constexpr (std::is_same_v<T, ShutdownMessage>) {
            running_ = false;
          } else {
            // Admin and profiler messages are accepted by the control plane
            // and have no behaviour in this build.
          }
        } catch (const std::exception& e) {
          // A malformed message must fail its request, not kill the thread.
          if (!request_id.empty()) fail_request(request_id, e.what());
        }
      },
      msg);
}

void Stage::stop() {
  // A shutdown message may have cleared running_; worker threads still join.
  {
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mu_);
    stopped_ = true;
    running_ = false;
  }
  // run() polls with a zero timeout and the abort loop with a bounded poll, so
  // clearing the flag ends both. Closing the sockets here would race with a
  // caller that still sits in run(); close happens in the destructor instead.
  if (scheduler_) scheduler_->stop();
  // Drain pending transfers first: each dropped operation releases its credit,
  // which wakes the outbox thread if it is blocked in CreditAllocator::acquire.
  reap_pending(true);
  for (auto& t : {&scheduler_thread_, &outbox_thread_}) {
    if (t->joinable()) t->join();
  }
  // The outbox may have inserted an entry after the first drain.
  reap_pending(true);
  for (auto& t : {&reaper_thread_, &abort_thread_}) {
    if (t->joinable()) t->join();
  }
}

void Stage::reap_pending(bool all, const std::string& request_id) {
  std::vector<std::pair<std::string, PendingTransfer>> expired;
  {
    std::lock_guard<std::mutex> lock(pending_mu_);
    const auto now = std::chrono::steady_clock::now();
    for (auto it = pending_ops_.begin(); it != pending_ops_.end();) {
      const bool selected = request_id.empty()
                                ? (all || it->second.deadline <= now)
                                : it->second.request_id == request_id;
      if (selected) {
        expired.push_back(*it);
        it = pending_ops_.erase(it);
      } else {
        ++it;
      }
    }
  }
  const std::string reason = !request_id.empty()
                                 ? "request aborted"
                                 : all ? "stage stopping before transfer ack"
                                       : "transfer ack timed out";
  for (auto& entry : expired) {
    try {
      entry.second.op->mark_receiver_abandoned(reason);
    } catch (...) {
      // An operation without an ack protocol has no credit to release.
    }
    // Request-selected cleanup is followed by the abort/outbox terminal event.
    // Timeout and stop cleanup must settle the request here instead.
    if (request_id.empty()) fail_request(entry.second.request_id, reason);
  }
}

void Stage::on_submit(const SubmitMessage& msg) {
  StagePayload payload;
  try {
    payload = StagePayload::from_value(msg.data);
  } catch (const std::exception&) {
    // A bare payload object is accepted too, as SubmitMessage.from_dict allows.
    payload.request_id = msg.request_id;
    payload.data = msg.data;
  }
  if (!payload.request_id.empty() && payload.request_id != msg.request_id) {
    throw ProtocolError("submit request_id differs from its payload");
  }
  payload.request_id = msg.request_id;
  execute(std::move(payload));
}

size_t Stage::request_count() {
  std::lock_guard<std::mutex> lock(requests_mu_);
  return requests_.size();
}

void Stage::execute(StagePayload payload) {
  {
    std::lock_guard<std::mutex> lock(requests_mu_);
    // abort() marks the scheduler before fail_request() takes requests_mu_.
    // Consuming that mark here closes the abort-before-submit race without
    // leaving a request record that no scheduler result will remove.
    if (scheduler_->consume_if_aborted(payload.request_id)) return;
    requests_[payload.request_id] = payload.request;
  }
  // The entry stage converts a request into work: when the caller sent no data,
  // the request inputs are the compute input.
  const Value work = payload.data.is_nil() ? payload.request.inputs : payload.data;
  scheduler_->enqueue(IncomingMessage{payload.request_id, "new_request", work});
}

void Stage::on_data_ready(const DataReadyMessage& msg) {
  if (msg.is_done || msg.error) return;  // stream signal: no payload to receive
  if (!msg.data_ref) {
    fail_request(msg.request_id, "data_ready message without data_ref");
    return;
  }
  std::string object_id;
  // Best-effort id for the failure NACK: a malformed ref throws before the full
  // decode assigns object_id. It only labels the NACK; the decode below still
  // decides whether the payload is accepted.
  if (const Value* id = msg.data_ref->find("object_id");
      id != nullptr && id->is_str()) {
    object_id = id->as_str();
  }
  try {
    const DataRef ref = DataRef::from_value(*msg.data_ref);
    object_id = ref.object_id;
    // A misrouted message must fail here, before the payload is read from the
    // wrong relay.
    if (msg.to_stage != config_.name) {
      throw ProtocolError("data_ready for " + msg.to_stage + " arrived at " +
                          config_.name);
    }
    validate_data_ref(ref, config_.max_payload_bytes);
    Tensor dest;
    dest.device = "cpu";
    dest.dtype = "uint8";
    dest.shape = {ref.buffer.length};
    dest.bytes.assign(static_cast<size_t>(ref.buffer.length), '\0');

    Relay& relay = router_.inbound_relay(msg.from_stage, ref.buffer.transport);
    relay.get(ref.buffer.info, dest);

    StagePayload payload = StagePayload::from_value(unpack(dest.bytes));
    if (!payload.request_id.empty() && payload.request_id != msg.request_id) {
      throw ProtocolError("data_ready request_id differs from its payload");
    }
    payload.request_id = msg.request_id;

    // Ack the sender so its credit is released. The ack goes back over the same
    // edge the payload came from.
    DataAckMessage ack;
    ack.request_id = msg.request_id;
    ack.from_stage = config_.name;
    ack.to_stage = msg.from_stage;
    ack.object_id = ref.object_id;
    ack.success = true;
    auto peer = config_.peer_endpoints.find(msg.from_stage);
    if (peer != config_.peer_endpoints.end()) {
      control_plane_->send_to_stage(msg.from_stage, peer->second, Message{ack});
    }
    execute(std::move(payload));
  } catch (const std::exception& e) {
    // A relay or decode failure must free the sender and settle the request.
    send_failure_ack(msg, object_id, e.what());
    fail_request(msg.request_id, e.what());
  }
}

void Stage::fail_request(const std::string& request_id, const std::string& error) {
  if (request_id.empty()) return;
  {
    std::lock_guard<std::mutex> lock(requests_mu_);
    requests_.erase(request_id);
  }
  if (!control_plane_) return;
  CompleteMessage done;
  done.request_id = request_id;
  done.from_stage = config_.name;
  done.success = false;
  done.error = error.empty() ? "stage failure" : error;
  try {
    control_plane_->send_complete(done);
  } catch (...) {
    // The coordinator is unreachable; there is nobody left to notify.
  }
}

void Stage::send_failure_ack(const DataReadyMessage& msg,
                             const std::string& object_id,
                             const std::string& error) {
  if (object_id.empty() || !control_plane_) return;
  auto peer = config_.peer_endpoints.find(msg.from_stage);
  if (peer == config_.peer_endpoints.end()) return;
  DataAckMessage ack;
  ack.request_id = msg.request_id;
  ack.from_stage = config_.name;
  ack.to_stage = msg.from_stage;
  ack.object_id = object_id;
  ack.success = false;
  ack.error = error.empty() ? "receive failed" : error;
  try {
    control_plane_->send_to_stage(msg.from_stage, peer->second, Message{ack});
  } catch (...) {
    // Best effort: the failed completion below still settles the request.
  }
}

void Stage::on_data_ack(const DataAckMessage& msg) {
  std::shared_ptr<RelayOperation> op;
  {
    std::lock_guard<std::mutex> lock(pending_mu_);
    auto it = pending_ops_.find(msg.object_id);
    if (it != pending_ops_.end()) {
      op = it->second.op;
      pending_ops_.erase(it);
    }
  }
  if (!op) return;
  if (msg.success) {
    op->mark_receiver_done();
  } else {
    op->mark_receiver_failed(msg.error ? *msg.error : "transfer failed");
  }
}

void Stage::send_payload(const std::string& request_id, const std::string& target,
                         const StagePayload& payload) {
  if (router_.is_local_object(target)) {
    throw ProtocolError("local-object dispatch needs a sibling stage handle");
  }
  const Value payload_value = payload.to_value();
  std::string bytes = pack(payload_value);
  const int64_t nbytes = static_cast<int64_t>(bytes.size());
  Tensor tensor;
  tensor.device = "cpu";
  tensor.dtype = "uint8";
  tensor.shape = {nbytes};
  tensor.bytes = std::move(bytes);

  auto [transport, relay] =
      router_.relay_for_payload(target, payload_value, nbytes);
  auto op = std::shared_ptr<RelayOperation>(relay->put(tensor, request_id).release());
  const std::string object_id = object_id_for(request_id, next_object_seq_++);
  {
    // A zero or negative timeout disables reaping.
    const auto deadline =
        config_.transfer_timeout_ms > 0
            ? std::chrono::steady_clock::now() +
                  std::chrono::milliseconds(config_.transfer_timeout_ms)
            : std::chrono::steady_clock::time_point::max();
    std::lock_guard<std::mutex> lock(pending_mu_);
    pending_ops_[object_id] = PendingTransfer{op, request_id, deadline};
  }

  // Build the message before the final synchronization point: metadata() may
  // block, and the pending lock must not be held across that wait.
  DataReadyMessage msg;
  try {
    msg.request_id = request_id;
    msg.from_stage = config_.name;
    msg.to_stage = target;
    msg.data_ref = make_payload_ref(payload_value, transport, op->metadata(),
                                    nbytes, object_id)
                       .to_value();
  } catch (...) {
    // ZMQ never saw a message, so no receiver can own this block.
    discard_pending(object_id, "data_ready build failed");
    throw;  // the original failure reaches the caller
  }

  // Hold pending_mu_ across the existence check, the abort check and the send
  // so send and reap are linearizable: either DataReady queues before reap, or
  // reap settles the operation first and this send is skipped. The socket send
  // has a 5-second timeout, so the hold is bounded.
  std::exception_ptr send_failure;
  {
    std::unique_lock<std::mutex> lock(pending_mu_);
    if (pending_ops_.find(object_id) == pending_ops_.end()) {
      // A reaper settled this transfer while metadata() was running.
      lock.unlock();
      throw ProtocolError("transfer was already settled");
    }
    if (scheduler_->consume_if_aborted(request_id)) {
      lock.unlock();
      // reap_pending takes pending_mu_ itself, so it must run unlocked.
      reap_pending(false, request_id);
      throw ProtocolError("request aborted");
    }
    try {
      control_plane_->send_to_stage(target, config_.next_stages.at(target),
                                    Message{msg});
    } catch (...) {
      send_failure = std::current_exception();
    }
  }
  if (send_failure) {
    // ZMQ did not queue DataReady, so no receiver can own this block.
    discard_pending(object_id, "data_ready send failed");
    std::rethrow_exception(send_failure);
  }
}

void Stage::discard_pending(const std::string& object_id,
                            const char* reason) noexcept {
  std::shared_ptr<RelayOperation> op;
  {
    std::lock_guard<std::mutex> lock(pending_mu_);
    auto it = pending_ops_.find(object_id);
    if (it == pending_ops_.end()) return;
    op = it->second.op;
    pending_ops_.erase(it);
  }
  // A relay without an ack protocol throws here; that must not replace the
  // original exception or skip the erase above.
  try {
    op->mark_receiver_failed(reason);
  } catch (...) {
  }
}

void Stage::route_result(const std::string& request_id, Value result) {
  OmniRequest request;
  {
    std::lock_guard<std::mutex> lock(requests_mu_);
    auto it = requests_.find(request_id);
    if (it != requests_.end()) {
      request = it->second;
      requests_.erase(it);
    }
  }
  if (config_.next_stages.empty()) {
    CompleteMessage done;
    done.request_id = request_id;
    done.from_stage = config_.name;
    done.success = true;
    done.result = std::move(result);
    control_plane_->send_complete(done);
    return;
  }
  StagePayload payload{request_id, request, std::move(result)};
  for (const auto& [target, endpoint] : config_.next_stages) {
    (void)endpoint;
    send_payload(request_id, target, payload);
  }
}

void Stage::pump_outbox() {
  while (running_) {
    auto msg = scheduler_->outbox().get(0.1);
    if (!msg) continue;
    try {
      if (msg->type == "error") {
        fail_request(msg->request_id,
                     msg->data.is_str() ? msg->data.as_str() : "stage compute failed");
        continue;
      }
      route_result(msg->request_id, std::move(msg->data));
    } catch (const std::exception& e) {
      // A failed route must not terminate the outbox thread.
      fail_request(msg->request_id, e.what());
    }
  }
}

// ---- coordinator ---------------------------------------------------------

Coordinator::Coordinator(std::string completion_endpoint, std::string abort_endpoint)
    : completion_endpoint_(std::move(completion_endpoint)),
      abort_endpoint_(std::move(abort_endpoint)) {}

Coordinator::~Coordinator() { stop(); }

void Coordinator::register_stage(const std::string& name, const std::string& endpoint,
                                 bool require_ready) {
  std::lock_guard<std::mutex> lock(mu_);
  if (control_plane_) {
    throw ProtocolError("stages must be registered before coordinator start");
  }
  endpoints_[name] = endpoint;
  if (require_ready) {
    stages_requiring_ready_.insert(name);
  } else {
    stages_requiring_ready_.erase(name);
  }
}

bool Coordinator::stage_ready(const std::string& name) const {
  std::lock_guard<std::mutex> lock(mu_);
  return ready_stages_.count(name) != 0;
}

void Coordinator::start() {
  if (control_plane_) return;
  control_plane_ = std::make_unique<CoordinatorControlPlane>(completion_endpoint_,
                                                             abort_endpoint_);
  control_plane_->start();
  running_ = true;
}

std::pair<Coordinator::StreamSink, Coordinator::CompleteSink>
Coordinator::exchange_sinks_before_run(StreamSink stream, CompleteSink complete) {
  std::lock_guard<std::mutex> lock(mu_);
  if (pump_started_) {
    throw ProtocolError("coordinator sink exchange must precede the first pump run");
  }
  auto previous = std::make_pair(std::move(on_stream), std::move(on_complete));
  on_stream = std::move(stream);
  on_complete = std::move(complete);
  return previous;
}

void Coordinator::run() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    pump_started_ = true;
  }
  if (!control_plane_) start();
  while (running_) {
    std::optional<Message> msg;
    try {
      msg = control_plane_->recv_event_timeout(100);
    } catch (const ProtocolError&) {
      if (!running_) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      continue;
    }
    if (!msg) continue;
    handle_event(*msg);
  }
}

void Coordinator::stop() {
  if (!running_.exchange(false)) return;
  std::vector<std::string> ids;
  {
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto& [id, info] : requests_) {
      if (info.state == "pending" || info.state == "running") ids.push_back(id);
    }
  }
  // Wake every HTTP or embedded caller before its coordinator disappears.
  for (const auto& id : ids) abort(id);
  // run() receives with a 100 ms timeout. Leave the sockets open until the
  // destructor, after the caller has joined the run thread.
}

std::string Coordinator::next_request_id() {
  // The counter never wraps: an explicit req-18446744073709551615 consumes the
  // last number, so automatic allocation must fail instead of reissuing req-0.
  // submit() holds mu_ here and has not recorded anything yet, so a throw
  // leaves no request behind.
  if (counter_.load() == std::numeric_limits<uint64_t>::max()) {
    throw ProtocolError("request id space exhausted");
  }
  return "req-" + std::to_string(++counter_);
}

std::string Coordinator::submit(StagePayload payload,
                                const std::string& entry_stage) {
  const bool explicit_id = !payload.request_id.empty();
  std::string request_id = payload.request_id;
  std::lock_guard<std::mutex> lock(mu_);
  if (!running_) throw ProtocolError("coordinator is stopping");
  auto it = endpoints_.find(entry_stage);
  if (it == endpoints_.end()) {
    throw ProtocolError("unknown entry stage: " + entry_stage);
  }
  if (explicit_id) {
    if (requests_.find(request_id) != requests_.end()) {
      throw ProtocolError("duplicate request_id: " + request_id);
    }
    // An explicit id in the generated namespace must not be reissued later by
    // the counter. The abort tombstone outlives the request record, so a reuse
    // would be dropped by the stage. The counter only moves forward.
    if (const auto number = generated_id_number(request_id)) {
      uint64_t current = counter_.load();
      while (*number > current &&
             !counter_.compare_exchange_weak(current, *number)) {
      }
    }
  } else {
    do {
      request_id = next_request_id();
    } while (requests_.find(request_id) != requests_.end());
  }
  payload.request_id = request_id;
  SubmitMessage msg;
  msg.request_id = request_id;
  msg.data = payload.to_value();
  RequestInfo info;
  info.request_id = request_id;
  info.state = "pending";
  info.current_stage = entry_stage;
  requests_[request_id] = std::move(info);

  const bool wait_for_ready = stages_requiring_ready_.count(entry_stage) != 0 &&
                              ready_stages_.count(entry_stage) == 0;
  if (wait_for_ready) {
    pending_submits_[entry_stage].push_back(std::move(msg));
    return request_id;
  }
  try {
    // Keep the request lock across send and the state transition. abort() uses
    // the same lock, so a ready flush cannot overtake an abort transition.
    control_plane_->submit_to_stage(entry_stage, it->second, Message{msg});
    requests_.at(request_id).state = "running";
  } catch (...) {
    requests_.erase(request_id);
    throw;
  }
  return request_id;
}

void Coordinator::handle_event(const Message& msg) {
  if (const auto* ready = std::get_if<StageReadyMessage>(&msg)) {
    std::vector<RequestInfo> failed;
    {
      std::lock_guard<std::mutex> lock(mu_);
      const auto endpoint = endpoints_.find(ready->from_stage);
      if (endpoint == endpoints_.end()) return;
      if (!ready_stages_.insert(ready->from_stage).second) return;
      const auto queued = pending_submits_.find(ready->from_stage);
      if (queued == pending_submits_.end()) return;
      for (const auto& submit : queued->second) {
        auto request = requests_.find(submit.request_id);
        if (request == requests_.end() || request->second.state != "pending" ||
            request->second.current_stage != ready->from_stage) {
          continue;
        }
        try {
          // Serialize every flush send with abort()'s terminal transition.
          control_plane_->submit_to_stage(ready->from_stage, endpoint->second,
                                          Message{submit});
          request->second.state = "running";
        } catch (const std::exception& error) {
          request->second.state = "failed";
          request->second.error = error.what();
          failed.push_back(request->second);
        }
      }
      pending_submits_.erase(queued);
    }
    if (!failed.empty()) cv_.notify_all();
    for (const auto& request : failed) {
      if (on_complete) on_complete(request);
    }
    return;
  }
  if (const auto* done = std::get_if<CompleteMessage>(&msg)) {
    RequestInfo snapshot;
    {
      std::lock_guard<std::mutex> lock(mu_);
      auto it = requests_.find(done->request_id);
      if (it == requests_.end()) return;
      // Only an in-flight request may settle. A late completion after abort or
      // after another completion changes nothing and fires no callback.
      if (it->second.state != "pending" && it->second.state != "running") {
        return;
      }
      it->second.state = done->success ? "completed" : "failed";
      it->second.result = done->result;
      if (done->error) it->second.error = *done->error;
      it->second.current_stage = done->from_stage;
      snapshot = it->second;
    }
    cv_.notify_all();
    if (on_complete) on_complete(snapshot);
    return;
  }
  if (const auto* chunk = std::get_if<StreamMessage>(&msg)) {
    bool active = false;
    {
      std::lock_guard<std::mutex> lock(mu_);
      const auto it = requests_.find(chunk->request_id);
      active = it != requests_.end() &&
               (it->second.state == "pending" || it->second.state == "running");
    }
    if (active && on_stream) on_stream(*chunk);
  }
}

bool Coordinator::wait(const std::string& request_id, double timeout_s) {
  std::unique_lock<std::mutex> lock(mu_);
  return cv_.wait_for(lock, std::chrono::duration<double>(timeout_s), [&] {
    auto it = requests_.find(request_id);
    return it != requests_.end() &&
           (it->second.state == "completed" || it->second.state == "failed" ||
            it->second.state == "aborted");
  });
}

std::optional<RequestInfo> Coordinator::info(const std::string& request_id) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = requests_.find(request_id);
  if (it == requests_.end()) return std::nullopt;
  return it->second;
}

size_t Coordinator::in_flight() const {
  std::lock_guard<std::mutex> lock(mu_);
  size_t n = 0;
  for (const auto& [id, info] : requests_) {
    if (info.state == "pending" || info.state == "running") ++n;
  }
  return n;
}

void Coordinator::abort(const std::string& request_id) {
  std::optional<RequestInfo> snapshot;
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = requests_.find(request_id);
    if (it == requests_.end()) return;
    if (it->second.state != "pending" && it->second.state != "running") return;
    const std::string stage = it->second.current_stage;
    it->second.state = "aborted";
    it->second.error = "request aborted";
    snapshot = it->second;
    const auto queued = pending_submits_.find(stage);
    if (queued != pending_submits_.end()) {
      std::erase_if(queued->second, [&](const SubmitMessage& submit) {
        return submit.request_id == request_id;
      });
      if (queued->second.empty()) pending_submits_.erase(queued);
    }
  }
  if (control_plane_) {
    try {
      control_plane_->broadcast_abort(AbortMessage{request_id});
    } catch (...) {
      // The local state is terminal either way.
    }
  }
  cv_.notify_all();
  // An aborted request is terminal, so stream sinks need the same finish signal
  // a completion gives them.
  if (on_complete && snapshot) on_complete(*snapshot);
}

bool Coordinator::release(const std::string& request_id) {
  std::lock_guard<std::mutex> lock(mu_);
  return requests_.erase(request_id) > 0;
}

}  // namespace umminfer

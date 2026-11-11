// SPDX-License-Identifier: Apache-2.0
#include "umminfer/relay/relay.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <random>

#if defined(UMMINFER_WITH_CUDA)
#include <cuda_runtime_api.h>
#endif

namespace umminfer {
namespace {

std::string fresh_shm_name(const std::string& engine_id) {
  // macOS caps a POSIX shm name at 31 bytes (PSHMNAMLEN). Hash the engine id and
  // keep the whole name under that cap.
  static std::mutex rng_mu;
  static std::mt19937_64 rng(std::random_device{}());
  std::lock_guard<std::mutex> lock(rng_mu);
  const uint64_t tag = std::hash<std::string>{}(engine_id);
  char buf[32];
  std::snprintf(buf, sizeof(buf), "/umm_%08x_%016llx", static_cast<uint32_t>(tag),
                static_cast<unsigned long long>(rng()));
  return buf;
}

void unlink_quiet(const std::string& name) { ::shm_unlink(name.c_str()); }

// Creates and fills one POSIX block. The block never has size 0 because the
// receiver expects a block for every tensor.
struct ShmBlock {
  std::string name;
  int64_t size = 0;

  static ShmBlock create(const std::string& name, const void* data, size_t n) {
    int fd = ::shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd < 0) {
      throw ProtocolError("shm_open failed: " + std::string(std::strerror(errno)));
    }
    const int64_t size = static_cast<int64_t>(n == 0 ? 1 : n);
    if (::ftruncate(fd, static_cast<off_t>(size)) != 0) {
      ::close(fd);
      unlink_quiet(name);
      throw ProtocolError("ftruncate failed: " + std::string(std::strerror(errno)));
    }
    if (n > 0) {
      void* dst = ::mmap(nullptr, static_cast<size_t>(size), PROT_WRITE, MAP_SHARED,
                         fd, 0);
      if (dst == MAP_FAILED) {
        ::close(fd);
        unlink_quiet(name);
        throw ProtocolError("mmap failed: " + std::string(std::strerror(errno)));
      }
      std::memcpy(dst, data, n);
      ::munmap(dst, static_cast<size_t>(size));
    }
    ::close(fd);
    return ShmBlock{name, static_cast<int64_t>(n)};
  }
};

class ShmOperation : public RelayOperation {
 public:
  explicit ShmOperation(Value metadata) : metadata_(std::move(metadata)) {}
  const Value& metadata() const override { return metadata_; }

 protected:
  Value metadata_;
};

class ShmPutOperation : public ShmOperation {
 public:
  ShmPutOperation(Value metadata, std::string shm_name, int64_t credit,
                  std::shared_ptr<CreditAllocator> credits)
      : ShmOperation(std::move(metadata)),
        shm_name_(std::move(shm_name)),
        credit_(credit),
        credits_(std::move(credits)),
        promise_(std::make_shared<std::promise<void>>()),
        future_(promise_->get_future().share()) {}

  ~ShmPutOperation() override {
    // The block must not outlive its sender, whether or not the receiver ever
    // looked at it. Unlinking twice is harmless.
    unlink_quiet(shm_name_);
    release_credit();
  }

  void wait_for_completion(double timeout_s = 30.0) override {
    if (completed_) return;
    if (future_.wait_for(std::chrono::duration<double>(timeout_s)) !=
        std::future_status::ready) {
      completed_ = true;
      release_credit();
      throw ProtocolError("SHM block " + shm_name_ + " was not consumed in time");
    }
    try {
      future_.get();
    } catch (...) {
      completed_ = true;
      throw;
    }
    completed_ = true;
  }

  // A completion is reported once and frees the credit once, no matter whether
  // it arrived through an ack, a timeout, or this operation's destruction.
  void mark_receiver_done() override {
    if (acked_.exchange(true)) return;
    promise_->set_value();
    release_credit();
  }
  void mark_receiver_failed(const std::string& error) override {
    if (acked_.exchange(true)) return;
    promise_->set_exception(std::make_exception_ptr(ProtocolError(error)));
    release_credit();
  }

 private:
  void release_credit() {
    if (released_.exchange(true)) return;
    credits_->release(credit_);
  }

  std::string shm_name_;
  int64_t credit_;
  // Held by shared_ptr: the operation may outlive ShmRelay.
  std::shared_ptr<CreditAllocator> credits_;
  std::shared_ptr<std::promise<void>> promise_;
  std::shared_future<void> future_;
  std::atomic<bool> completed_{false};
  std::atomic<bool> acked_{false};
  std::atomic<bool> released_{false};
};

#if defined(UMMINFER_WITH_CUDA)

void cuda_check(cudaError_t rc, const char* what) {
  if (rc == cudaSuccess) return;
  const std::string message =
      std::string(what) + " failed: " + cudaGetErrorString(rc);
  (void)cudaGetLastError();  // reported here; must not leak into later calls
  throw ProtocolError(message);
}

// Runs a call on one device and puts the thread back where it was.
class DeviceGuard {
 public:
  explicit DeviceGuard(int device) {
    if (cudaGetDevice(&previous_) != cudaSuccess) {
      (void)cudaGetLastError();
      return;
    }
    ok_ = cudaSetDevice(device) == cudaSuccess;
    if (!ok_) (void)cudaGetLastError();
  }
  ~DeviceGuard() {
    if (ok_) (void)cudaSetDevice(previous_);
  }
  bool ok() const { return ok_; }

 private:
  int previous_ = 0;
  bool ok_ = false;
};

// Shared by the relay and by operations that may outlive it. The failure flag
// stops new allocations once a transfer was abandoned: a sender that keeps
// allocating after its peer is gone only accumulates device memory.
struct CudaIpcShared {
  explicit CudaIpcShared(int credits) : credits(credits) {}
  ~CudaIpcShared();

  CreditAllocator credits;

  std::mutex mu;
  bool failed = false;
  std::string failure;
  // One reusable buffer per credit id. A credit is held by one operation at a
  // time, so its slot can only be reused after that operation released it.
  struct Slot {
    void* buffer = nullptr;
    int64_t capacity = 0;
    int device = 0;
  };
  std::map<int64_t, Slot> slots;
};

void free_on_device(int device, void* buffer) {
  if (buffer == nullptr) return;
  DeviceGuard guard(device);
  if (!guard.ok()) return;  // the context is gone; nothing left to free
  (void)cudaFree(buffer);
  (void)cudaGetLastError();
}

CudaIpcShared::~CudaIpcShared() {
  // Only pooled slots are freed. A dropped slot has a null buffer and its
  // allocation is left to the CUDA context.
  for (auto& [credit, slot] : slots) {
    (void)credit;
    free_on_device(slot.device, slot.buffer);
  }
}

// Returns a buffer of at least `size` bytes for a credit. Grows in place; the
// credit is exclusively held, so no other operation uses this slot.
void* slot_for_credit(const std::shared_ptr<CudaIpcShared>& shared,
                      int64_t credit, int64_t size, int device) {
  {
    std::lock_guard<std::mutex> lock(shared->mu);
    auto& slot = shared->slots[credit];
    if (slot.buffer != nullptr && slot.capacity >= size) return slot.buffer;
  }
  void* fresh = nullptr;
  cuda_check(cudaMalloc(&fresh, static_cast<size_t>(size)), "cudaMalloc");
  void* previous = nullptr;
  {
    std::lock_guard<std::mutex> lock(shared->mu);
    auto& slot = shared->slots[credit];
    previous = slot.buffer;
    slot.buffer = fresh;
    slot.capacity = size;
    slot.device = device;
  }
  free_on_device(device, previous);
  return fresh;
}

// Takes a credit's slot out of the pool. The receiver may still map the block,
// so the allocation is dropped without cudaFree and never reused.
void retire_slot(const std::shared_ptr<CudaIpcShared>& shared, int64_t credit) {
  std::lock_guard<std::mutex> lock(shared->mu);
  auto it = shared->slots.find(credit);
  if (it == shared->slots.end()) return;
  it->second.buffer = nullptr;
  it->second.capacity = 0;
}

class CudaIpcPutOperation : public RelayOperation {
 public:
  CudaIpcPutOperation(Value metadata, std::shared_ptr<CudaIpcShared> shared,
                      int64_t credit)
      : metadata_(std::move(metadata)),
        shared_(std::move(shared)),
        credit_(credit),
        promise_(std::make_shared<std::promise<void>>()),
        future_(promise_->get_future().share()) {}

  ~CudaIpcPutOperation() override {
    // An ack means the receiver finished with the block, so the slot stays in
    // the pool for the next holder of this credit. Without an ack the receiver
    // may still be copying: take the slot out of the pool and retire it.
    if (!acked_ || abandoned_) retire_slot(shared_, credit_);
    release_credit();
  }

  const Value& metadata() const override { return metadata_; }

  void wait_for_completion(double timeout_s = 30.0) override {
    if (completed_) return;
    if (future_.wait_for(std::chrono::duration<double>(timeout_s)) !=
        std::future_status::ready) {
      completed_ = true;
      // A late receiver may still read the block, so this is an abandonment,
      // not a plain failure: retire the buffer and poison the relay.
      mark_receiver_abandoned("CUDA IPC block was not consumed in time");
      throw ProtocolError("CUDA IPC block was not consumed in time");
    }
    try {
      future_.get();
    } catch (...) {
      completed_ = true;
      throw;
    }
    completed_ = true;
  }

  void mark_receiver_done() override {
    if (acked_.exchange(true)) return;
    promise_->set_value();
    release_credit();
  }

  void mark_receiver_failed(const std::string& error) override {
    if (acked_.exchange(true)) return;
    promise_->set_exception(std::make_exception_ptr(ProtocolError(error)));
    release_credit();
  }

  // No ack will arrive. Retire the block, release the credit and mark the whole
  // relay failed so the next put() fails instead of allocating again.
  void mark_receiver_abandoned(const std::string& error) override {
    if (acked_.exchange(true)) return;
    abandoned_ = true;
    const std::string reason = error.empty() ? "transfer abandoned" : error;
    {
      std::lock_guard<std::mutex> lock(shared_->mu);
      if (!shared_->failed) {
        shared_->failed = true;
        shared_->failure = reason;
      }
    }
    // The receiver may still map this block, so it must leave the pool before
    // the credit is released and a later put could reuse it.
    retire_slot(shared_, credit_);
    promise_->set_exception(std::make_exception_ptr(ProtocolError(reason)));
    release_credit();
  }

 private:
  void release_credit() {
    if (released_.exchange(true)) return;
    shared_->credits.release(credit_);
  }

  Value metadata_;
  std::shared_ptr<CudaIpcShared> shared_;
  int64_t credit_;
  std::shared_ptr<std::promise<void>> promise_;
  std::shared_future<void> future_;
  std::atomic<bool> completed_{false};
  std::atomic<bool> acked_{false};
  std::atomic<bool> abandoned_{false};
  std::atomic<bool> released_{false};
};

// Copies host bytes into the credit's pooled device buffer and publishes its
// IPC handle. The receiver maps the handle and copies device-to-host directly
// from the peer pointer; it does not allocate a local device block.
class CudaIpcRelay : public Relay {
 public:
  CudaIpcRelay(std::string engine_id, int device, int credits, int max_mb)
      : engine_id_(std::move(engine_id)),
        device_(device),
        max_bytes_(static_cast<int64_t>(max_mb) * 1024 * 1024),
        shared_(std::make_shared<CudaIpcShared>(credits)) {}

  ~CudaIpcRelay() override { close(); }

  std::unique_ptr<RelayOperation> put(const Tensor& tensor,
                                      const std::string& request_id) override {
    {
      std::lock_guard<std::mutex> lock(shared_->mu);
      if (shared_->failed) {
        throw ProtocolError("cuda_ipc relay is marked failed (" +
                            shared_->failure +
                            "); restart the sender process before sending again");
      }
    }
    if (tensor.device != "cpu") {
      throw ProtocolError("cuda_ipc relay takes host bytes, not " +
                          tensor.device);
    }
    const int64_t declared = static_cast<int64_t>(tensor.bytes.size());
    if (declared > max_bytes_) {
      throw ProtocolError("cuda_ipc transfer of " + std::to_string(declared) +
                          " bytes exceeds the slot_size_mb limit");
    }
    const int64_t credit = shared_->credits.acquire();
    {
      // A concurrent abandon or close may land while this call waits for credit.
      std::lock_guard<std::mutex> lock(shared_->mu);
      if (shared_->failed) {
        shared_->credits.release(credit);
        throw ProtocolError("cuda_ipc relay is marked failed (" +
                            shared_->failure +
                            "); restart the sender process to rebuild the edge");
      }
    }

    DeviceGuard guard(device_);
    if (!guard.ok()) {
      shared_->credits.release(credit);
      throw ProtocolError("cudaSetDevice(" + std::to_string(device_) +
                          ") failed");
    }
    // An empty tensor still gets a device block so the handle is always valid.
    // The block is bound to the credit and reused by the next holder of it.
    const int64_t size = declared == 0 ? 1 : declared;
    try {
      void* buffer = slot_for_credit(shared_, credit, size, device_);
      if (declared > 0) {
        cuda_check(cudaMemcpy(buffer, tensor.bytes.data(),
                              static_cast<size_t>(declared),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy host to device");
      }
      cudaIpcMemHandle_t handle{};
      cuda_check(cudaIpcGetMemHandle(&handle, buffer), "cudaIpcGetMemHandle");

      Value::Map transfer_info;
      transfer_info["ipc_handle"] = Value::bin(std::string(
          reinterpret_cast<const char*>(&handle), sizeof(handle)));
      transfer_info["size"] = Value(declared);
      transfer_info["device"] = Value(device_);
      transfer_info["req_id"] = Value(request_id);

      Value::Map metadata;
      metadata["engine_id"] = Value(engine_id_);
      metadata["transfer_info"] = Value(std::move(transfer_info));
      return std::make_unique<CudaIpcPutOperation>(
          Value(std::move(metadata)), shared_, credit);
    } catch (...) {
      // The slot stays in the pool: no receiver owns this block yet, and the
      // next holder of the credit may reuse it.
      shared_->credits.release(credit);
      throw;
    }
  }

  void get(const Value& metadata, Tensor& dest) override {
    if (dest.device != "cpu") {
      throw ProtocolError("cuda_ipc relay copies into host memory, not " +
                          dest.device);
    }
    const Value& info = require_field(metadata, "transfer_info");
    const std::string& raw = require_field(info, "ipc_handle").as_bin();
    if (raw.size() != sizeof(cudaIpcMemHandle_t)) {
      throw ProtocolError("cuda_ipc handle must be " +
                          std::to_string(sizeof(cudaIpcMemHandle_t)) + " bytes");
    }
    const int64_t size = require_field(info, "size").as_int();
    if (size < 0) throw ProtocolError("cuda_ipc transfer size is negative");
    // The caller sizes dest from the DataRef length. A transfer larger than
    // that would silently overwrite a smaller declared block.
    if (static_cast<int64_t>(dest.bytes.size()) < size) {
      throw ProtocolError("cuda_ipc destination is smaller than the transfer");
    }

    DeviceGuard guard(device_);
    if (!guard.ok()) {
      throw ProtocolError("cudaSetDevice(" + std::to_string(device_) +
                          ") failed");
    }

    cudaIpcMemHandle_t handle{};
    std::memcpy(&handle, raw.data(), sizeof(handle));
    void* remote = nullptr;
    cuda_check(cudaIpcOpenMemHandle(&remote, handle,
                                    cudaIpcMemLazyEnablePeerAccess),
               "cudaIpcOpenMemHandle");
    struct RemoteGuard {
      void* ptr;
      ~RemoteGuard() {
        if (ptr != nullptr) (void)cudaIpcCloseMemHandle(ptr);
      }
    } remote_guard{remote};

    if (size == 0) {
      dest.bytes.clear();
      return;
    }
    dest.bytes.assign(static_cast<size_t>(size), '\0');
    // The mapped peer pointer is directly readable from host memory. The
    // receiver's local device block and device-to-device hop were measured to
    // cost more than this direct copy; see the cuda-bench baseline.
    cuda_check(cudaMemcpy(dest.bytes.data(), remote,
                          static_cast<size_t>(size), cudaMemcpyDeviceToHost),
               "cudaMemcpy device to host");
  }

  void close() override {
    // Retired blocks may still be mapped by a receiver, so they are never freed
    // here. They stay allocated until the process exits; the CUDA context and
    // the OS reclaim them then.
    shared_->credits.close();
    std::lock_guard<std::mutex> lock(shared_->mu);
    shared_->failed = true;
    if (shared_->failure.empty()) shared_->failure = "relay closed";
  }

 private:
  std::string engine_id_;
  int device_;
  int64_t max_bytes_;
  std::shared_ptr<CudaIpcShared> shared_;
};

#endif  // UMMINFER_WITH_CUDA

}  // namespace

void shm_unlink_quiet(const std::string& name) { unlink_quiet(name); }

CreditAllocator::CreditAllocator(int credits, std::optional<int64_t> slot_size) {
  for (int i = 0; i < std::max(credits, 1); ++i) {
    free_.push_back(slot_size ? i * *slot_size : i);
  }
}

int64_t CreditAllocator::acquire() {
  std::unique_lock<std::mutex> lock(mu_);
  cv_.wait(lock, [this] { return closed_ || !free_.empty(); });
  if (closed_) throw ProtocolError("credit allocator is closed");
  int64_t id = free_.front();
  free_.pop_front();
  return id;
}

void CreditAllocator::release(int64_t credit) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    free_.push_back(credit);
  }
  cv_.notify_one();
}

void CreditAllocator::close() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    closed_ = true;
  }
  cv_.notify_all();
}

ShmRelay::ShmRelay(std::string engine_id, int slot_size_mb, int credits)
    : engine_id_(std::move(engine_id)),
      credits_(std::make_shared<CreditAllocator>(credits)) {
  // Host shared memory is allocated per transfer; slot_size_mb is unused.
  (void)slot_size_mb;
}

std::unique_ptr<RelayOperation> ShmRelay::put(const Tensor& tensor,
                                              const std::string& request_id) {
  const int64_t credit = credits_->acquire();
  const std::string name = fresh_shm_name(engine_id_);
  ShmBlock block;
  try {
    block = ShmBlock::create(name, tensor.bytes.data(), tensor.bytes.size());
  } catch (...) {
    credits_->release(credit);
    throw;
  }

  Value::Map transfer_info;
  transfer_info["shm_name"] = Value(block.name);
  transfer_info["size"] = Value(block.size);
  transfer_info["req_id"] = Value(request_id);

  Value::Map metadata;
  metadata["engine_id"] = Value(engine_id_);
  metadata["transfer_info"] = Value(std::move(transfer_info));
  // The credit stays held until the receiver acks, so a sender cannot reuse a
  // block that a receiver has not read yet.
  return std::make_unique<ShmPutOperation>(Value(std::move(metadata)), block.name,
                                           credit, credits_);
}

void ShmRelay::get(const Value& metadata, Tensor& dest) {
  const Value& info = require_field(metadata, "transfer_info");
  const std::string shm_name = require_str(info, "shm_name");
  const int64_t size = require_field(info, "size").as_int();

  const int fd = ::shm_open(shm_name.c_str(), O_RDWR, 0600);
  if (fd < 0) throw ProtocolError("SHM block " + shm_name + " not found.");
  if (static_cast<int64_t>(dest.bytes.size()) < size) {
    ::close(fd);
    throw ProtocolError("SHM destination is smaller than the transfer");
  }
  if (size > 0) {
    void* src = ::mmap(nullptr, static_cast<size_t>(size), PROT_READ, MAP_SHARED, fd, 0);
    if (src == MAP_FAILED) {
      ::close(fd);
      throw ProtocolError("mmap failed for " + shm_name);
    }
    std::memcpy(dest.bytes.data(), src, static_cast<size_t>(size));
    ::munmap(src, static_cast<size_t>(size));
  }
  ::close(fd);
  // The receiver unlinks the block after copying it.
  unlink_quiet(shm_name);
}

void ShmRelay::close() { credits_->close(); }

std::unique_ptr<Relay> create_relay(const std::string& type,
                                    const std::map<std::string, Value>& config) {
  auto int_or = [&](const std::string& key, int fallback) {
    auto it = config.find(key);
    if (it == config.end()) return fallback;
    const int64_t value = it->second.as_int();
    if (value < std::numeric_limits<int>::min() ||
        value > std::numeric_limits<int>::max()) {
      throw ProtocolError(key + " value " + std::to_string(value) +
                          " is outside the int range");
    }
    return static_cast<int>(value);
  };
  if (type == "shm") {
    auto it = config.find("engine_id");
    const std::string engine_id = it == config.end() ? "relay" : it->second.as_str();
    return std::make_unique<ShmRelay>(engine_id, int_or("slot_size_mb", 64),
                                      int_or("credits", 2));
  }
  if (type == "cuda_ipc") {
#if defined(UMMINFER_WITH_CUDA)
    auto it = config.find("engine_id");
    const std::string engine_id = it == config.end() ? "relay" : it->second.as_str();
    return std::make_unique<CudaIpcRelay>(
        engine_id, int_or("device", 0), int_or("credits", 2),
        int_or("slot_size_mb", 512));
#else
    throw ProtocolError(
        "cuda_ipc relay requires a build with UMMINFER_WITH_CUDA");
#endif
  }
  throw ProtocolError("Unknown relay type: '" + type +
                      "'. Available types: [shm, cuda_ipc]");
}

}  // namespace umminfer

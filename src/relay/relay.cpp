// SPDX-License-Identifier: Apache-2.0
#include "omniinfer/relay/relay.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>

namespace omniinfer {
namespace {

std::string fresh_shm_name(const std::string& engine_id) {
  // macOS caps a POSIX shm name at 31 bytes (PSHMNAMLEN). Hash the engine id and
  // keep the whole name under that cap.
  static std::mutex rng_mu;
  static std::mt19937_64 rng(std::random_device{}());
  std::lock_guard<std::mutex> lock(rng_mu);
  const uint64_t tag = std::hash<std::string>{}(engine_id);
  char buf[32];
  std::snprintf(buf, sizeof(buf), "/omni_%08x_%016llx", static_cast<uint32_t>(tag),
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

}  // namespace

void shm_unlink_quiet(const std::string& name) { unlink_quiet(name); }

CreditAllocator::CreditAllocator(int credits, std::optional<int64_t> slot_size) {
  for (int i = 0; i < std::max(credits, 1); ++i) {
    free_.push_back(slot_size ? i * *slot_size : i);
  }
}

int64_t CreditAllocator::acquire() {
  std::unique_lock<std::mutex> lock(mu_);
  cv_.wait(lock, [this] { return !free_.empty(); });
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

void ShmRelay::close() {}

std::unique_ptr<Relay> create_relay(const std::string& type,
                                    const std::map<std::string, Value>& config) {
  auto int_or = [&](const std::string& key, int fallback) {
    auto it = config.find(key);
    return it == config.end() ? fallback : static_cast<int>(it->second.as_int());
  };
  if (type == "shm") {
    auto it = config.find("engine_id");
    const std::string engine_id = it == config.end() ? "relay" : it->second.as_str();
    return std::make_unique<ShmRelay>(engine_id, int_or("slot_size_mb", 64),
                                      int_or("credits", 2));
  }
  // ponytail: CUDA backends stay unmapped here. Add the class and its vendor
  // link when a GPU build is needed; the interface above already fits them.
  throw ProtocolError("Unknown relay type: '" + type + "'. Available types: [shm]");
}

}  // namespace omniinfer

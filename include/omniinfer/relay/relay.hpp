// SPDX-License-Identifier: Apache-2.0
// Host shared-memory relay with bounded transfer credits.
//
// cuda_ipc, nccl, nixl and mooncake are not available in this build.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "omniinfer/proto/messages.hpp"
#include "omniinfer/proto/wire.hpp"

namespace omniinfer {

// One in-flight transfer. Completion means the receiver consumed the block.
class RelayOperation {
 public:
  virtual ~RelayOperation() = default;
  virtual const Value& metadata() const = 0;
  virtual void wait_for_completion(double timeout_s = 30.0) = 0;
  virtual void mark_receiver_done() { throw ProtocolError("no receiver ack"); }
  virtual void mark_receiver_failed(const std::string& error) {
    (void)error;
    throw ProtocolError("no receiver ack");
  }
};

class Relay {
 public:
  virtual ~Relay() = default;
  virtual std::unique_ptr<RelayOperation> put(const Tensor& tensor,
                                              const std::string& request_id) = 0;
  // Copies the block named by metadata into dest. dest.bytes is resized first.
  virtual void get(const Value& metadata, Tensor& dest) = 0;
  virtual void cleanup(const std::string& request_id) { (void)request_id; }
  virtual void close() {}
};

// Credits gate concurrent transfers. Memory mode returns slot offsets;
// simple mode returns sequential ids.
class CreditAllocator {
 public:
  CreditAllocator(int credits, std::optional<int64_t> slot_size = std::nullopt);

  int64_t acquire();
  void release(int64_t credit);

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<int64_t> free_;
};

// POSIX shared-memory relay for host tensors.
class ShmRelay : public Relay {
 public:
  ShmRelay(std::string engine_id, int slot_size_mb = 64, int credits = 2);

  std::unique_ptr<RelayOperation> put(const Tensor& tensor,
                                      const std::string& request_id) override;
  void get(const Value& metadata, Tensor& dest) override;
  void close() override;

 private:
  std::string engine_id_;
  // Shared with in-flight operations, which may outlive the relay and must
  // still release their credit.
  std::shared_ptr<CreditAllocator> credits_;
};

// Unknown or unavailable backends raise ProtocolError.
std::unique_ptr<Relay> create_relay(const std::string& type,
                                    const std::map<std::string, Value>& config);

// Removes a POSIX shm name; exposed for the receiver side and for tests.
void shm_unlink_quiet(const std::string& name);

}  // namespace omniinfer

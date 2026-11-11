// SPDX-License-Identifier: Apache-2.0
// Serial and batched request scheduling.
//
// A non-autoregressive stage runs one callable per request. This scheduler owns
// the inbox/outbox pair the stage polls, so every scheduler backend presents the
// same shape to Stage.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "umminfer/proto/wire.hpp"

namespace umminfer {

using IncomingMessageType = std::string;  // new_request, abort, stop, wakeup, ...

struct IncomingMessage {
  std::string request_id;
  std::string type;
  Value data;
};

struct OutgoingMessage {
  std::string request_id;
  std::string type;  // result, stream, error
  Value data;
  std::optional<std::string> target;
  std::optional<Value> metadata;
};

// Bounded-free thread-safe queue; the C++ stand-in for queue.Queue.
template <typename T>
class MessageQueue {
 public:
  void put(T msg) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      items_.push_back(std::move(msg));
    }
    cv_.notify_one();
  }
  std::optional<T> get(double timeout_s = -1) {
    std::unique_lock<std::mutex> lock(mu_);
    if (timeout_s < 0) {
      cv_.wait(lock, [this] { return !items_.empty(); });
    } else if (!cv_.wait_for(lock, std::chrono::duration<double>(timeout_s),
                             [this] { return !items_.empty(); })) {
      return std::nullopt;
    }
    T item = std::move(items_.front());
    items_.pop_front();
    return item;
  }
  std::optional<T> get_nowait() {
    std::lock_guard<std::mutex> lock(mu_);
    if (items_.empty()) return std::nullopt;
    T item = std::move(items_.front());
    items_.pop_front();
    return item;
  }
  size_t size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return items_.size();
  }

 private:
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::deque<T> items_;
};

using ComputeFn = std::function<Value(const Value&)>;
using BatchComputeFn = std::function<std::vector<Value>(const std::vector<Value>&)>;
using RequestCostFn = std::function<int64_t(const Value&)>;
using RequestArrivalHook = std::function<void(const Value&)>;

class SimpleScheduler {
 public:
  SimpleScheduler(ComputeFn compute_fn, BatchComputeFn batch_compute_fn = nullptr,
                  int max_batch_size = 1, int max_batch_wait_ms = 0,
                  bool batch_wait_when_idle = true,
                  RequestCostFn request_cost_fn = nullptr,
                  std::optional<int> max_batch_cost = std::nullopt,
                  int max_concurrency = 1,
                  std::function<void(const std::string&)> abort_callback = nullptr,
                  std::function<void()> shutdown_callback = nullptr,
                  RequestArrivalHook request_arrival_hook = nullptr);

  MessageQueue<IncomingMessage>& inbox() { return inbox_; }
  MessageQueue<OutgoingMessage>& outbox() { return outbox_; }

  void enqueue(IncomingMessage msg);
  void start();  // blocks the calling thread
  void stop();
  void abort(const std::string& request_id);

  bool is_aborted(const std::string& request_id);
  bool consume_if_aborted(const std::string& request_id);

 private:
  void run_serial();
  void run_concurrent();
  void run_single(const IncomingMessage& msg);
  void run_batch(const std::vector<IncomingMessage>& batch);
  void collect_batch(std::vector<IncomingMessage>& batch);
  int64_t message_cost(const IncomingMessage& msg) const;
  void emit_result(const std::string& request_id, Value result);
  void emit_error(const std::string& request_id, const std::string& error);

  ComputeFn fn_;
  BatchComputeFn batch_fn_;
  int max_batch_size_;
  double max_batch_wait_s_;
  bool batch_wait_when_idle_;
  RequestCostFn request_cost_fn_;
  std::optional<int> max_batch_cost_;
  int max_concurrency_;
  std::function<void(const std::string&)> abort_callback_;
  std::function<void()> shutdown_callback_;
  RequestArrivalHook request_arrival_hook_;

  MessageQueue<IncomingMessage> inbox_;
  MessageQueue<OutgoingMessage> outbox_;
  std::deque<IncomingMessage> pending_messages_;
  std::mutex pending_mu_;

  std::mutex abort_mu_;
  std::set<std::string> aborted_;

  std::atomic<bool> running_{false};
};

}  // namespace umminfer

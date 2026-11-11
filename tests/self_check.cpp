// SPDX-License-Identifier: Apache-2.0
// Runnable checks for the runtime core: wire format, relay, router, scheduler,
// and one end-to-end pipeline request.
#include <cassert>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <thread>
#include <unistd.h>

#include "umminfer/comm/comm.hpp"
#include "umminfer/pipeline/pipeline.hpp"
#include "umminfer/proto/messages.hpp"
#include "umminfer/relay/relay.hpp"
#include "umminfer/scheduling/simple_scheduler.hpp"

using namespace umminfer;

namespace {

int failures = 0;

#define CHECK(cond)                                                          \
  do {                                                                       \
    if (!(cond)) {                                                           \
      std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << " " << #cond    \
                << "\n";                                                     \
      ++failures;                                                            \
    }                                                                        \
  } while (0)

std::string hex_encode(const std::string& raw) {
  static const char* digits = "0123456789abcdef";
  std::string out;
  for (unsigned char c : raw) {
    out.push_back(digits[c >> 4]);
    out.push_back(digits[c & 0xf]);
  }
  return out;
}

void test_value_roundtrip() {
  Value::Map inner;
  inner["text"] = Value("hello");
  inner["n"] = Value(static_cast<int64_t>(7));
  Value::Array arr;
  arr.push_back(Value(true));
  arr.push_back(Value::bin(std::string("\x00\x01", 2)));
  inner["items"] = Value(std::move(arr));

  Tensor tensor;
  tensor.device = "cuda:1";
  tensor.dtype = "float32";
  tensor.shape = {2, 2};
  tensor.bytes = std::string(16, '\x01');
  inner["t"] = Value(tensor);

  const Value packed = unpack(pack(Value(inner)));
  CHECK(packed.find("text")->as_str() == "hello");
  CHECK(packed.find("n")->as_int() == 7);
  CHECK(packed.find("items")->as_array()[1].as_bin() == std::string("\x00\x01", 2));
  CHECK(packed.find("t")->as_tensor() == tensor);
}

// Bytes produced by Python:
//   DataReadyMessage(request_id='r1', from_stage='a', to_stage='b', is_done=True)
void test_decode_python_bytes() {
  const std::string hex =
      "85a474797065aa646174615f7265616479aa726571756573745f6964a27231aa66726f6d5f7374"
      "616765a161a8746f5f7374616765a162a769735f646f6e65c3";
  std::string raw;
  for (size_t i = 0; i + 1 < hex.size(); i += 2) {
    raw.push_back(static_cast<char>(std::stoi(hex.substr(i, 2), nullptr, 16)));
  }
  const Message msg = deserialize_message(raw);
  const auto* ready = std::get_if<DataReadyMessage>(&msg);
  CHECK(ready != nullptr);
  if (ready != nullptr) {
    CHECK(ready->request_id == "r1");
    CHECK(ready->from_stage == "a");
    CHECK(ready->to_stage == "b");
    CHECK(ready->is_done);
  }
}

void test_message_roundtrip() {
  SubmitMessage submit;
  submit.request_id = "req-1";
  StagePayload payload;
  payload.request_id = "req-1";
  payload.request.inputs = Value("ping");
  payload.data = Value("pong");
  submit.data = payload.to_value();

  const Message decoded = deserialize_message(serialize_message(Message{submit}));
  const auto* back = std::get_if<SubmitMessage>(&decoded);
  CHECK(back != nullptr);
  if (back != nullptr) {
    CHECK(back->request_id == "req-1");
    const StagePayload parsed = StagePayload::from_value(back->data);
    CHECK(parsed.data.as_str() == "pong");
    CHECK(parsed.request.inputs.as_str() == "ping");
  }

  CompleteMessage complete;
  complete.request_id = "req-1";
  complete.from_stage = "entry";
  complete.success = true;
  complete.result = Value("pong");
  const Message complete_decoded =
      deserialize_message(serialize_message(Message{complete}));
  CHECK(std::get<CompleteMessage>(complete_decoded).result.as_str() == "pong");

  DataReadyMessage signal;
  signal.request_id = "req-2";
  signal.from_stage = "a";
  signal.to_stage = "b";
  signal.is_done = true;
  const Message signal_decoded =
      deserialize_message(serialize_message(Message{signal}));
  CHECK(std::get<DataReadyMessage>(signal_decoded).is_done);
}

void test_shm_relay() {
  ShmRelay relay("test_engine", 1, 2);
  Tensor src;
  src.device = "cpu";
  src.dtype = "uint8";
  src.shape = {5};
  src.bytes = "abcde";

  auto op = relay.put(src, "req-1");
  Tensor dst;
  dst.bytes.assign(5, '\0');
  relay.get(op->metadata(), dst);
  CHECK(dst.bytes == "abcde");
  op->mark_receiver_done();
  op->wait_for_completion(1.0);

  // A second transfer after the first frees its credit.
  auto op2 = relay.put(src, "req-2");
  CHECK(op2->metadata().contains("transfer_info"));
}

void test_router_classification() {
  CommRouter router("a", std::nullopt, std::nullopt, {},
                    std::set<std::string>{"gpu_b"},
                    {{"gpu_b", {0}}}, {});
  CHECK(router.outbound("gpu_b") == TransportKind::Shm);  // CPU build falls back
  CHECK(router.outbound("unknown") == TransportKind::Shm);

  CommRouter local("a", std::nullopt, std::nullopt, {"sibling"}, {}, {}, {});
  CHECK(local.outbound("sibling") == TransportKind::LocalObject);
  CHECK(local.is_local_object("sibling"));

  CommRouter remote("a", std::nullopt, std::nullopt, {}, {}, {},
                    {"remote_stage"});
  CHECK(remote.outbound("remote_stage") == TransportKind::Mooncake);

  Value payload_map;
  Value::Map m;
  Tensor t;
  t.device = "cuda:0";
  m["x"] = Value(t);
  payload_map = Value(std::move(m));
  CHECK(tensor_devices(payload_map) == std::set<std::string>{"cuda"});
}

void test_scheduler_serial_and_batch() {
  SimpleScheduler scheduler([](const Value& in) { return Value(in.as_str() + "!"); });
  std::thread serial_thread([&scheduler] { scheduler.start(); });  // start() blocks
  scheduler.enqueue(IncomingMessage{"r1", "new_request", Value("hi")});
  auto out = scheduler.outbox().get(2.0);
  CHECK(out.has_value());
  if (out) {
    CHECK(out->request_id == "r1");
    CHECK(out->data.as_str() == "hi!");
  }
  scheduler.stop();
  serial_thread.join();

  SimpleScheduler batched(
      [](const Value&) { return Value("single"); },
      [](const std::vector<Value>& inputs) {
        std::vector<Value> results;
        for (const auto& in : inputs) results.push_back(Value(in.as_str() + "+"));
        return results;
      },
      8, 20, true);
  std::thread batch_thread([&batched] { batched.start(); });
  batched.enqueue(IncomingMessage{"b1", "new_request", Value("a")});
  batched.enqueue(IncomingMessage{"b2", "new_request", Value("b")});
  size_t got = 0;
  while (auto msg = batched.outbox().get(1.0)) ++got;
  CHECK(got == 2);
  batched.stop();
  batch_thread.join();
}

void test_pipeline_end_to_end() {
  const std::string base =
      "/tmp/omni-self-check-" + std::to_string(getpid()) + "-";
  const std::string entry_ep = "ipc://" + base + "entry";
  const std::string completion_ep = "ipc://" + base + "completion";
  const std::string abort_ep = "ipc://" + base + "abort";
  // A run killed mid-test leaves its ipc sockets behind and the next bind fails.
  for (const auto& path : {base + "entry", base + "completion", base + "abort"}) {
    std::remove(path.c_str());
  }

  Coordinator coordinator(completion_ep, abort_ep);
  coordinator.register_stage("entry", entry_ep);
  coordinator.start();
  std::thread coordinator_thread([&] { coordinator.run(); });

  Stage::Config config;
  config.name = "entry";
  config.recv_endpoint = entry_ep;
  config.coordinator_endpoint = completion_ep;
  config.abort_endpoint = abort_ep;
  config.peer_endpoints["entry"] = entry_ep;
  Stage stage(config, [](const Value& in) { return Value(in.as_str() + " -> done"); });
  stage.start();
  // run() is the blocking control-plane loop; start() only spawns the worker
  // threads, so the loop runs on its own thread here.
  std::thread stage_thread([&stage] { stage.run(); });

  StagePayload payload;
  payload.request.inputs = Value("ping");
  payload.data = Value("ping");
  const std::string request_id = coordinator.submit(payload, "entry");
  CHECK(coordinator.wait(request_id, 5.0));
  const auto info = coordinator.info(request_id);
  CHECK(info.has_value());
  if (info) {
    // Check the state first: reading result on a failed request raises and
    // would hide the failure it is meant to report.
    CHECK(info->state == "completed");
    if (info->state == "completed") {
      CHECK(info->result.as_str() == "ping -> done");
    }
  }

  stage.stop();
  coordinator.stop();
  stage_thread.join();
  coordinator_thread.join();
}

}  // namespace

int main() {
  alarm(20);
  std::cerr << "[1] value roundtrip\n";
  test_value_roundtrip();
  std::cerr << "[2] python bytes\n";
  test_decode_python_bytes();
  std::cerr << "[3] message roundtrip\n";
  test_message_roundtrip();
  std::cerr << "[4] shm relay\n";
  test_shm_relay();
  std::cerr << "[5] router\n";
  test_router_classification();
  std::cerr << "[6] scheduler\n";
  test_scheduler_serial_and_batch();
  std::cerr << "[7] pipeline\n";
  test_pipeline_end_to_end();
  std::cerr << "[done]\n";

  if (failures != 0) {
    std::cerr << failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "all checks passed\n";
  return 0;
}

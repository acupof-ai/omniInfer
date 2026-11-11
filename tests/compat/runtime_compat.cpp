// SPDX-License-Identifier: Apache-2.0
#include <atomic>
#include <chrono>
#include <cstdio>
#include <httplib.h>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "omniinfer/comm/comm.hpp"
#include "omniinfer/pipeline/pipeline.hpp"
#include "omniinfer/relay/relay.hpp"
#include "omniinfer/scheduling/simple_scheduler.hpp"
#include "omniinfer/serve/openai_api.hpp"

using namespace omniinfer;

namespace {

void check(bool condition, const char* message, int& failures) {
  if (!condition) {
    std::cerr << message << "\n";
    ++failures;
  }
}

void test_batch_strings_are_results(int& failures) {
  SimpleScheduler scheduler(
      [](const Value&) { return Value("unused"); },
      [](const std::vector<Value>& inputs) {
        std::vector<Value> results;
        for (const auto& input : inputs) results.push_back(Value(input.as_str() + "!"));
        return results;
      },
      2, 20, true);
  std::thread worker([&scheduler] { scheduler.start(); });
  scheduler.enqueue(IncomingMessage{"batch-1", "new_request", Value("a")});
  scheduler.enqueue(IncomingMessage{"batch-2", "new_request", Value("b")});

  for (int index = 0; index < 2; ++index) {
    const auto message = scheduler.outbox().get(2.0);
    check(message.has_value(), "batch result timed out", failures);
    if (message) {
      check(message->type == "result", "string batch result was emitted as an error", failures);
    }
  }
  scheduler.stop();
  worker.join();
}

void test_batch_cost_error_reaches_request(int& failures) {
  SimpleScheduler scheduler(
      [](const Value&) { return Value("unused"); },
      [](const std::vector<Value>& inputs) { return inputs; }, 2, 0, false,
      [](const Value&) -> int64_t { throw ProtocolError("cost failed"); }, 10);
  std::thread worker([&scheduler] { scheduler.start(); });
  scheduler.enqueue(IncomingMessage{"cost-1", "new_request", Value("x")});
  const auto message = scheduler.outbox().get(2.0);
  check(message.has_value(), "batch cost exception dropped the request", failures);
  if (message) {
    check(message->request_id == "cost-1" && message->type == "error",
          "batch cost exception did not emit an error", failures);
  }
  scheduler.stop();
  worker.join();
}

void test_later_batch_cost_error(int& failures) {
  SimpleScheduler scheduler(
      [](const Value& input) { return input; },
      [](const std::vector<Value>& inputs) { return inputs; }, 2, 0, false,
      [](const Value& input) -> int64_t {
        if (input.as_str() == "bad") throw ProtocolError("bad cost");
        return 1;
      },
      10);
  std::thread worker([&] { scheduler.start(); });
  scheduler.enqueue(IncomingMessage{"good", "new_request", Value("good")});
  scheduler.enqueue(IncomingMessage{"bad", "new_request", Value("bad")});
  bool good_result = false;
  bool bad_error = false;
  for (int i = 0; i < 2; ++i) {
    const auto message = scheduler.outbox().get(1.0);
    if (!message) break;
    good_result |= message->request_id == "good" && message->type == "result";
    bad_error |= message->request_id == "bad" && message->type == "error";
  }
  check(good_result && bad_error, "a later cost error corrupted the batch", failures);
  scheduler.stop();
  worker.join();
}

void test_abort_retention_and_callback(int& failures) {
  int callbacks = 0;
  SimpleScheduler scheduler(
      [](const Value& input) { return input; }, nullptr, 1, 0, true, nullptr,
      std::nullopt, 1, [&](const std::string&) { ++callbacks; });
  scheduler.abort("once");
  check(scheduler.consume_if_aborted("once"), "abort was not retained", failures);
  check(callbacks == 1, "abort callback ran more than once", failures);

  scheduler.abort("a");
  for (int i = 0; i < 10000; ++i) {
    scheduler.abort("b" + std::to_string(i));
  }
  check(scheduler.is_aborted("a"), "abort capacity cleanup forgot a request", failures);
}

void test_batch_filters_aborted_inputs(int& failures) {
  std::string seen;
  SimpleScheduler scheduler(
      [](const Value& input) { return input; },
      [&](const std::vector<Value>& inputs) {
        for (const auto& input : inputs) seen += input.as_str();
        return inputs;
      },
      3, 0, false);
  scheduler.abort("b");
  std::thread worker([&] { scheduler.start(); });
  for (const char* id : {"a", "b", "c"}) {
    scheduler.enqueue(IncomingMessage{id, "new_request", Value(id)});
  }
  int results = 0;
  for (int i = 0; i < 2; ++i) {
    if (scheduler.outbox().get(1.0)) ++results;
  }
  check(results == 2 && seen == "ac", "an aborted request entered batch compute",
        failures);
  scheduler.stop();
  worker.join();
}

void test_batch_cost_overflow(int& failures) {
  int batch_calls = 0;
  SimpleScheduler scheduler(
      [](const Value&) { return Value("single"); },
      [&](const std::vector<Value>& inputs) {
        ++batch_calls;
        return std::vector<Value>(inputs.size(), Value("batch"));
      },
      2, 0, false,
      [](const Value& input) -> int64_t {
        return input.as_str() == "first" ? 10 : std::numeric_limits<int64_t>::max();
      },
      10);
  std::thread worker([&] { scheduler.start(); });
  scheduler.enqueue(IncomingMessage{"first", "new_request", Value("first")});
  scheduler.enqueue(IncomingMessage{"second", "new_request", Value("second")});
  for (int i = 0; i < 2; ++i) {
    check(scheduler.outbox().get(1.0).has_value(), "cost overflow dropped a request",
          failures);
  }
  check(batch_calls == 0, "cost overflow bypassed max_batch_cost", failures);
  scheduler.stop();
  worker.join();
}

void test_scheduler_control_messages(int& failures) {
  {
    SimpleScheduler scheduler(
        [](const Value& input) { return input; },
        [](const std::vector<Value>& inputs) { return inputs; }, 2, 5000, true);
    std::atomic<bool> done{false};
    std::thread worker([&] {
      scheduler.start();
      done = true;
    });
    scheduler.enqueue(IncomingMessage{"wait", "new_request", Value("x")});
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    scheduler.stop();
    for (int i = 0; i < 30 && !done; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!done) scheduler.enqueue(IncomingMessage{"release", "new_request", Value("x")});
    worker.join();
    check(done, "stop waited for the batch deadline", failures);
  }
  {
    std::atomic<int> calls{0};
    SimpleScheduler scheduler([&](const Value& input) {
      ++calls;
      return input;
    });
    std::thread worker([&] { scheduler.start(); });
    scheduler.enqueue(IncomingMessage{"r", "abort", Value()});
    scheduler.enqueue(IncomingMessage{"r", "new_request", Value("x")});
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    check(calls == 0 && !scheduler.outbox().get_nowait(),
          "an inbox abort message was ignored", failures);
    scheduler.stop();
    worker.join();
  }
  {
    int shutdowns = 0;
    std::atomic<bool> done{false};
    SimpleScheduler scheduler(
        [](const Value& input) { return input; }, nullptr, 1, 0, true, nullptr,
        std::nullopt, 1, nullptr, [&] { ++shutdowns; });
    std::thread worker([&] {
      scheduler.start();
      done = true;
    });
    scheduler.enqueue(IncomingMessage{"", "stop", Value()});
    for (int i = 0; i < 50 && !done; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!done) scheduler.stop();
    worker.join();
    check(done && shutdowns == 1, "an inbox stop message was ignored", failures);
  }
}

void test_ack_releases_shm_credit(int& failures) {
  ShmRelay relay("compat_credit", 1, 1);
  Tensor source;
  source.device = "cpu";
  source.dtype = "uint8";
  source.shape = {1};
  source.bytes = "x";

  auto first = relay.put(source, "credit-1");
  Tensor destination;
  destination.bytes.assign(1, '\0');
  relay.get(first->metadata(), destination);
  first->mark_receiver_done();
  first.reset();

  auto second = relay.put(source, "credit-2");
  check(second != nullptr, "ack did not release the SHM credit", failures);
  if (second) {
    relay.get(second->metadata(), destination);
    second->mark_receiver_done();
  }
}

void test_wire_boundaries(int& failures) {
  const std::string too_large("\xcf\x80\x00\x00\x00\x00\x00\x00\x00", 9);
  bool rejected = false;
  try {
    (void)unpack(too_large);
  } catch (const ProtocolError&) {
    rejected = true;
  }
  check(rejected, "uint64 above INT64_MAX was not rejected", failures);

  bool constructor_rejected = false;
  try {
    Value invalid(uint64_t{1} << 63);
    (void)invalid;
  } catch (const ProtocolError&) {
    constructor_rejected = true;
  }
  check(constructor_rejected, "Value(uint64_t) silently wrapped an integer", failures);

  Value::Map collision;
  collision["__tensor__"] = Value(true);
  collision["value"] = Value(7);
  const Value decoded = unpack(pack(Value(collision)));
  check(decoded.is_map() && decoded.find("value") != nullptr,
        "a regular __tensor__ map lost its entries", failures);

  DataRef ref;
  ref.version = 1;
  ref.kind = DataKind::StagePayload;
  ref.object_id = "object-1";
  ref.transport = TransportKind::Shm;
  ref.layout = DataLayout::PackedTensors;
  ref.buffer.transport = TransportKind::Shm;
  ref.buffer.info = Value::map({});
  ref.buffer.length = 0;
  TensorMeta meta;
  meta.path = "";
  meta.shape = {0};
  meta.dtype = "uint8";
  meta.device = "cpu";
  meta.offset = 0;
  meta.size = 0;
  ref.tensors.push_back(meta);
  Value encoded = ref.to_value();
  encoded["header"] = Value("AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8=");
  try {
    const DataRef decoded_ref = DataRef::from_value(encoded);
    check(decoded_ref.tensors.size() == 1 && decoded_ref.tensors[0].path.empty(),
          "an empty tensor path was rejected", failures);
    check(decoded_ref.header && decoded_ref.header->size() == 32,
          "a legacy base64 header was not decoded", failures);
    if (decoded_ref.header && decoded_ref.header->size() == 32) {
      bool bytes_match = true;
      for (size_t index = 0; index < 32; ++index) {
        bytes_match = bytes_match &&
                      static_cast<unsigned char>((*decoded_ref.header)[index]) == index;
      }
      check(bytes_match, "a long base64 header was corrupted", failures);
    }
  } catch (const std::exception& exc) {
    std::cerr << "DataRef boundary decode failed: " << exc.what() << "\n";
    ++failures;
  }
  for (const char* header : {"A", "AA=A"}) {
    encoded["header"] = Value(header);
    bool malformed_rejected = false;
    try {
      (void)DataRef::from_value(encoded);
    } catch (const ProtocolError&) {
      malformed_rejected = true;
    }
    check(malformed_rejected, "malformed base64 padding was accepted", failures);
  }
}

void remove_ipc(const std::string& suffix) {
  for (const char* name : {"a", "b", "done", "abort", "entry"}) {
    std::remove(("/tmp/omni-runtime-" + suffix + "-" +
                 std::to_string(getpid()) + "-" + name)
                    .c_str());
  }
}

void test_coordinator_terminal_state(int& failures) {
  const std::string base =
      "/tmp/omni-runtime-state-" + std::to_string(getpid()) + "-";
  remove_ipc("state");
  const std::string done_ep = "ipc://" + base + "done";
  const std::string abort_ep = "ipc://" + base + "abort";
  const std::string entry_ep = "ipc://" + base + "entry";
  PullSocket entry(entry_ep, true);
  entry.start();

  Coordinator coordinator(done_ep, abort_ep);
  coordinator.register_stage("entry", entry_ep);
  coordinator.start();
  std::atomic<int> completions{0};
  coordinator.on_complete = [&](const RequestInfo&) { ++completions; };
  std::thread run_thread([&] { coordinator.run(); });

  StagePayload explicit_payload;
  explicit_payload.request_id = "req-1";
  coordinator.submit(explicit_payload, "entry");
  const std::string generated = coordinator.submit(StagePayload{}, "entry");
  check(generated != "req-1", "an automatic request id collided with an explicit id", failures);

  bool duplicate_rejected = false;
  try {
    coordinator.submit(explicit_payload, "entry");
  } catch (const ProtocolError&) {
    duplicate_rejected = true;
  }
  check(duplicate_rejected, "a duplicate request id was accepted", failures);

  coordinator.abort("req-1");
  PushSocket completion(done_ep);
  completion.connect();
  CompleteMessage late;
  late.request_id = "req-1";
  late.from_stage = "entry";
  late.success = true;
  late.result = Value("late");
  completion.send(Message{late});
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  const auto info = coordinator.info("req-1");
  check(info && info->state == "aborted", "a late completion overwrote an abort", failures);
  check(completions == 1, "terminal callbacks were not emitted exactly once", failures);

  coordinator.release("req-1");
  coordinator.release(generated);
  check(!coordinator.info("req-1"), "release retained the request", failures);
  check(coordinator.in_flight() == 0, "released requests remained in flight", failures);

  completion.close();
  coordinator.stop();
  run_thread.join();
  entry.close();
}

void test_two_stage_pipeline(int& failures) {
  const std::string base =
      "/tmp/omni-runtime-two-" + std::to_string(getpid()) + "-";
  remove_ipc("two");
  const std::string a_ep = "ipc://" + base + "a";
  const std::string b_ep = "ipc://" + base + "b";
  const std::string done_ep = "ipc://" + base + "done";
  const std::string abort_ep = "ipc://" + base + "abort";

  Coordinator coordinator(done_ep, abort_ep);
  coordinator.register_stage("a", a_ep);
  coordinator.start();
  std::thread coordinator_thread([&] { coordinator.run(); });

  Stage::Config a_config;
  a_config.name = "a";
  a_config.recv_endpoint = a_ep;
  a_config.coordinator_endpoint = done_ep;
  a_config.abort_endpoint = abort_ep;
  a_config.next_stages["b"] = b_ep;
  a_config.peer_endpoints = {{"a", a_ep}, {"b", b_ep}};
  Stage a(a_config, [](const Value& in) { return Value(in.as_str() + "A"); });

  Stage::Config b_config;
  b_config.name = "b";
  b_config.recv_endpoint = b_ep;
  b_config.coordinator_endpoint = done_ep;
  b_config.abort_endpoint = abort_ep;
  b_config.peer_endpoints = a_config.peer_endpoints;
  Stage b(b_config, [](const Value& in) { return Value(in.as_str() + "B"); });

  a.start();
  b.start();
  std::thread a_thread([&] { a.run(); });
  std::thread b_thread([&] { b.run(); });

  std::vector<std::string> ids;
  for (int index = 0; index < 4; ++index) {
    StagePayload payload;
    payload.data = Value("p" + std::to_string(index));
    ids.push_back(coordinator.submit(payload, "a"));
  }
  for (size_t index = 0; index < ids.size(); ++index) {
    check(coordinator.wait(ids[index], 5.0), "two-stage request timed out", failures);
    const auto info = coordinator.info(ids[index]);
    check(info && info->state == "completed", "two-stage request failed", failures);
    if (info && info->state == "completed") {
      const std::string expected = "p" + std::to_string(index) + "AB";
      check(info->result.as_str() == expected, "two-stage result was corrupted", failures);
    }
    coordinator.release(ids[index]);
  }

  PushSocket shutdown(a_ep);
  shutdown.connect();
  shutdown.send(Message{ShutdownMessage{}});
  a_thread.join();
  a.stop();
  b.stop();
  coordinator.stop();
  b_thread.join();
  coordinator_thread.join();
  shutdown.close();
}

void test_transfer_ack_timeout(int& failures) {
  const std::string base =
      "/tmp/omni-runtime-timeout-" + std::to_string(getpid()) + "-";
  const std::string a_ep = "ipc://" + base + "a";
  const std::string done_ep = "ipc://" + base + "done";
  const std::string abort_ep = "ipc://" + base + "abort";
  for (const auto& path : {base + "a", base + "b", base + "done", base + "abort"}) {
    std::remove(path.c_str());
  }

  Coordinator coordinator(done_ep, abort_ep);
  coordinator.register_stage("a", a_ep);
  coordinator.start();
  std::thread coordinator_thread([&] { coordinator.run(); });
  Stage::Config config;
  config.name = "a";
  config.recv_endpoint = a_ep;
  config.coordinator_endpoint = done_ep;
  config.abort_endpoint = abort_ep;
  config.next_stages["b"] = "ipc://" + base + "b";
  config.peer_endpoints["a"] = a_ep;
  config.transfer_timeout_ms = 50;
  Stage stage(config, [](const Value& input) { return input; });
  stage.start();
  std::thread stage_thread([&] { stage.run(); });

  std::vector<std::string> ids;
  for (int i = 0; i < 3; ++i) {
    StagePayload payload;
    payload.data = Value("x");
    ids.push_back(coordinator.submit(payload, "a"));
  }
  for (const auto& id : ids) {
    check(coordinator.wait(id, 3.0), "a missing ack blocked the pipeline", failures);
    const auto info = coordinator.info(id);
    check(info && info->state == "failed" &&
              info->error.find("transfer ack timed out") != std::string::npos,
          "a missing ack did not produce a timeout failure", failures);
    coordinator.release(id);
  }

  stage.stop();
  coordinator.stop();
  stage_thread.join();
  coordinator_thread.join();
}

void test_abort_releases_pending_transfer(int& failures) {
  const std::string base =
      "/tmp/omni-runtime-abort-credit-" + std::to_string(getpid()) + "-";
  const std::string a_ep = "ipc://" + base + "a";
  const std::string b_ep = "ipc://" + base + "b";
  const std::string done_ep = "ipc://" + base + "done";
  const std::string abort_ep = "ipc://" + base + "abort";
  for (const auto& path : {base + "a", base + "b", base + "done", base + "abort"}) {
    std::remove(path.c_str());
  }

  PullSocket downstream(b_ep, true);
  downstream.start();
  PullSocket done(done_ep, true);
  done.start();
  PubSocket abort(abort_ep);
  abort.bind();
  Stage::Config config;
  config.name = "a";
  config.recv_endpoint = a_ep;
  config.coordinator_endpoint = done_ep;
  config.abort_endpoint = abort_ep;
  config.next_stages["b"] = b_ep;
  config.peer_endpoints["a"] = a_ep;
  config.transfer_timeout_ms = 0;
  Stage stage(config, [](const Value& input) { return input; });
  stage.start();
  std::thread stage_thread([&] { stage.run(); });
  PushSocket entry(a_ep);
  entry.connect();

  auto submit = [&](const std::string& id) {
    StagePayload payload;
    payload.request_id = id;
    payload.data = Value("x");
    SubmitMessage message;
    message.request_id = id;
    message.data = payload.to_value();
    entry.send(Message{message});
  };
  submit("credit-1");
  submit("credit-2");
  for (int i = 0; i < 2; ++i) {
    const auto ready = downstream.recv_timeout(1000);
    check(ready && std::holds_alternative<DataReadyMessage>(*ready),
          "the initial transfers did not reach the downstream stage", failures);
  }

  abort.publish(AbortMessage{"credit-1"});
  abort.publish(AbortMessage{"credit-2"});
  bool first_settled = false;
  bool second_settled = false;
  for (int i = 0; i < 4 && !(first_settled && second_settled); ++i) {
    const auto message = done.recv_timeout(1000);
    if (!message || !std::holds_alternative<CompleteMessage>(*message)) break;
    const auto& complete = std::get<CompleteMessage>(*message);
    first_settled |= complete.request_id == "credit-1" && !complete.success;
    second_settled |= complete.request_id == "credit-2" && !complete.success;
  }
  check(first_settled && second_settled, "aborted transfers did not settle", failures);

  submit("credit-3");
  const auto third = downstream.recv_timeout(1000);
  check(third && std::holds_alternative<DataReadyMessage>(*third) &&
            std::get<DataReadyMessage>(*third).request_id == "credit-3",
        "aborted transfers did not release their relay credits", failures);

  stage.stop();
  stage_thread.join();
  entry.close();
  abort.close();
  done.close();
  downstream.close();
}

void test_sse_content_and_timeout(int& failures) {
  const std::string base =
      "/tmp/omni-runtime-http-" + std::to_string(getpid()) + "-";
  const std::string entry_ep = "ipc://" + base + "entry";
  const std::string done_ep = "ipc://" + base + "done";
  const std::string abort_ep = "ipc://" + base + "abort";
  for (const auto& path : {base + "entry", base + "done", base + "abort"}) {
    std::remove(path.c_str());
  }

  Coordinator coordinator(done_ep, abort_ep);
  coordinator.register_stage("entry", entry_ep);
  coordinator.start();
  std::thread coordinator_thread([&] { coordinator.run(); });
  Stage::Config stage_config;
  stage_config.name = "entry";
  stage_config.recv_endpoint = entry_ep;
  stage_config.coordinator_endpoint = done_ep;
  stage_config.abort_endpoint = abort_ep;
  Stage stage(stage_config, [](const Value&) {
    Value::Map result;
    result["content"] = Value("hello");
    return Value(std::move(result));
  });
  stage.start();
  std::thread stage_thread([&] { stage.run(); });

  ApiConfig api_config;
  api_config.host = "127.0.0.1";
  api_config.port = 0;
  api_config.request_timeout_s = 0.2;
  OpenAIApi api(coordinator, api_config);
  const bool bound = api.bind();
  check(bound, "HTTP test server did not bind", failures);
  if (!bound) {
    stage.stop();
    coordinator.stop();
    stage_thread.join();
    coordinator_thread.join();
    return;
  }
  std::thread server_thread([&] { (void)api.listen(); });
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  httplib::Client client("127.0.0.1", api.bound_port());
  const std::string body =
      R"({"messages":[{"role":"user","content":"x"}],"stream":true})";
  const auto response = client.Post("/v1/chat/completions", body, "application/json");
  check(response && response->status == 200 &&
            response->body.find("\"content\":\"hello\"") != std::string::npos &&
            response->body.find("\\\"content\\\"") == std::string::npos,
        "SSE double-encoded a content map", failures);

  stage.stop();
  stage_thread.join();
  const auto started = std::chrono::steady_clock::now();
  const auto stalled = client.Post("/v1/chat/completions", body, "application/json");
  const double elapsed = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - started)
                             .count();
  check(stalled && stalled->body.find("stream timed out") != std::string::npos &&
            elapsed < 3.0,
        "SSE ignored request_timeout_s", failures);

  api.stop();
  server_thread.join();
  coordinator.stop();
  coordinator_thread.join();
}

void test_abort_cleans_stage_request(int& failures) {
  const std::string base =
      "/tmp/omni-runtime-abort-" + std::to_string(getpid()) + "-";
  const std::string entry_ep = "ipc://" + base + "entry";
  const std::string done_ep = "ipc://" + base + "done";
  const std::string abort_ep = "ipc://" + base + "abort";
  for (const auto& path : {base + "entry", base + "done", base + "abort"}) {
    std::remove(path.c_str());
  }

  PullSocket done(done_ep, true);
  done.start();
  PubSocket abort(abort_ep);
  abort.bind();
  Stage::Config config;
  config.name = "entry";
  config.recv_endpoint = entry_ep;
  config.coordinator_endpoint = done_ep;
  config.abort_endpoint = abort_ep;
  Stage stage(config, [](const Value& value) {
    if (value.as_str() == "work") {
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    return value;
  });
  stage.start();
  std::thread stage_thread([&] { stage.run(); });
  PushSocket entry(entry_ep);
  entry.connect();

  StagePayload payload;
  payload.request_id = "abort-stage";
  payload.data = Value("work");
  SubmitMessage submit;
  submit.request_id = payload.request_id;
  submit.data = payload.to_value();
  entry.send(Message{submit});
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  abort.publish(AbortMessage{payload.request_id});

  const auto message = done.recv_timeout(1000);
  check(message && std::holds_alternative<CompleteMessage>(*message),
        "an aborted stage request did not settle", failures);
  if (message && std::holds_alternative<CompleteMessage>(*message)) {
    const auto& complete = std::get<CompleteMessage>(*message);
    check(complete.request_id == payload.request_id && !complete.success,
          "an aborted stage request reported the wrong completion", failures);
  }
  check(stage.request_count() == 0, "an aborted stage request was retained", failures);

  abort.publish(AbortMessage{"abort-before-submit"});
  const auto early_abort = done.recv_timeout(1000);
  check(early_abort && std::holds_alternative<CompleteMessage>(*early_abort) &&
            std::get<CompleteMessage>(*early_abort).request_id == "abort-before-submit",
        "an early abort did not settle", failures);

  StagePayload delayed;
  delayed.request_id = "abort-before-submit";
  delayed.data = Value("delayed");
  SubmitMessage delayed_submit;
  delayed_submit.request_id = delayed.request_id;
  delayed_submit.data = delayed.to_value();
  entry.send(Message{delayed_submit});

  StagePayload sentinel;
  sentinel.request_id = "abort-sentinel";
  sentinel.data = Value("sentinel");
  SubmitMessage sentinel_submit;
  sentinel_submit.request_id = sentinel.request_id;
  sentinel_submit.data = sentinel.to_value();
  entry.send(Message{sentinel_submit});
  const auto sentinel_done = done.recv_timeout(1000);
  check(sentinel_done && std::holds_alternative<CompleteMessage>(*sentinel_done) &&
            std::get<CompleteMessage>(*sentinel_done).request_id == sentinel.request_id,
        "the delayed-submit sentinel did not complete", failures);
  check(stage.request_count() == 0, "a delayed submit revived an aborted request",
        failures);

  stage.stop();
  stage_thread.join();
  entry.close();
  abort.close();
  done.close();
}

void test_data_ready_checks_target(int& failures) {
  const std::string base =
      "/tmp/omni-runtime-target-" + std::to_string(getpid()) + "-";
  const std::string b_ep = "ipc://" + base + "b";
  const std::string peer_ep = "ipc://" + base + "a";
  const std::string done_ep = "ipc://" + base + "done";
  const std::string abort_ep = "ipc://" + base + "abort";
  for (const auto& path : {base + "a", base + "b", base + "done", base + "abort"}) {
    std::remove(path.c_str());
  }

  PullSocket peer(peer_ep, true);
  peer.start();
  PullSocket done(done_ep, true);
  done.start();
  PubSocket abort(abort_ep);
  abort.bind();
  Stage::Config config;
  config.name = "b";
  config.recv_endpoint = b_ep;
  config.coordinator_endpoint = done_ep;
  config.abort_endpoint = abort_ep;
  config.peer_endpoints["a"] = peer_ep;
  std::atomic<int> calls{0};
  Stage stage(config, [&](const Value& value) {
    ++calls;
    return value;
  });
  stage.start();
  std::thread stage_thread([&] { stage.run(); });
  PushSocket sender(b_ep);
  sender.connect();

  StagePayload inbound;
  inbound.request_id = "misrouted";
  inbound.data = Value("work");
  Tensor source;
  source.device = "cpu";
  source.dtype = "uint8";
  source.bytes = pack(inbound.to_value());
  source.shape = {static_cast<int64_t>(source.bytes.size())};
  ShmRelay relay("misrouted_source", 1, 1);
  auto operation = relay.put(source, inbound.request_id);

  DataRef ref;
  ref.version = 1;
  ref.kind = DataKind::StagePayload;
  ref.object_id = "wrong-target";
  ref.transport = TransportKind::Shm;
  ref.layout = DataLayout::PackedTensors;
  ref.buffer.transport = TransportKind::Shm;
  ref.buffer.info = operation->metadata();
  ref.buffer.length = static_cast<int64_t>(source.bytes.size());
  TensorMeta meta;
  meta.path = "";
  meta.shape = source.shape;
  meta.dtype = "uint8";
  meta.device = "cpu";
  meta.offset = 0;
  meta.size = ref.buffer.length;
  ref.tensors.push_back(meta);
  DataReadyMessage ready;
  ready.request_id = "misrouted";
  ready.from_stage = "a";
  ready.to_stage = "c";
  ready.data_ref = ref.to_value();
  sender.send(Message{ready});

  const auto ack_message = peer.recv_timeout(1000);
  check(ack_message && std::holds_alternative<DataAckMessage>(*ack_message),
        "a misrouted data_ready did not return an ack", failures);
  if (ack_message && std::holds_alternative<DataAckMessage>(*ack_message)) {
    const auto& ack = std::get<DataAckMessage>(*ack_message);
    check(!ack.success && ack.object_id == ref.object_id,
          "a misrouted data_ready did not fail its ack", failures);
  }
  const auto done_message = done.recv_timeout(1000);
  check(done_message && std::holds_alternative<CompleteMessage>(*done_message) &&
            !std::get<CompleteMessage>(*done_message).success,
        "a misrouted data_ready did not fail its request", failures);
  check(calls == 0, "a misrouted data_ready executed compute", failures);

  stage.stop();
  stage_thread.join();
  sender.close();
  abort.close();
  done.close();
  peer.close();
}

// ---- item 1: untrusted DataRef validation --------------------------------

DataRef make_ref(const std::string& object_id, int64_t length,
                 TransportKind transport, int64_t tensor_size,
                 const Value& info) {
  DataRef ref;
  ref.version = 1;
  ref.kind = DataKind::StagePayload;
  ref.object_id = object_id;
  ref.transport = transport;
  ref.layout = DataLayout::PackedTensors;
  ref.buffer.transport = transport;
  ref.buffer.info = info;
  ref.buffer.length = length;
  TensorMeta meta;
  meta.path = "";
  meta.shape = {tensor_size};
  meta.dtype = "uint8";
  meta.device = "cpu";
  meta.offset = 0;
  meta.size = tensor_size;
  ref.tensors.push_back(meta);
  return ref;
}

void test_data_ready_validation(int& failures) {
  const std::string base =
      "/tmp/omni-runtime-validate-" + std::to_string(getpid()) + "-";
  const std::string b_ep = "ipc://" + base + "b";
  const std::string peer_ep = "ipc://" + base + "a";
  const std::string done_ep = "ipc://" + base + "done";
  const std::string abort_ep = "ipc://" + base + "abort";
  for (const auto& path : {base + "a", base + "b", base + "done", base + "abort"}) {
    std::remove(path.c_str());
  }

  PullSocket peer(peer_ep, true);
  peer.start();
  PullSocket done(done_ep, true);
  done.start();
  PubSocket abort(abort_ep);
  abort.bind();
  Stage::Config config;
  config.name = "b";
  config.recv_endpoint = b_ep;
  config.coordinator_endpoint = done_ep;
  config.abort_endpoint = abort_ep;
  config.peer_endpoints["a"] = peer_ep;
  // Small bound: the oversized case must fail without allocating 8 KiB.
  config.max_payload_bytes = 4096;
  std::atomic<int> calls{0};
  Stage stage(config, [&](const Value& value) {
    ++calls;
    return value;
  });
  stage.start();
  std::thread stage_thread([&] { stage.run(); });
  PushSocket sender(b_ep);
  sender.connect();

  auto expect_rejected = [&](const char* name, const DataRef& ref) {
    DataReadyMessage ready;
    ready.request_id = ref.object_id;
    ready.from_stage = "a";
    ready.to_stage = "b";
    ready.data_ref = ref.to_value();
    const auto started = std::chrono::steady_clock::now();
    sender.send(Message{ready});

    const auto ack_message = peer.recv_timeout(1000);
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
            .count();
    const bool acked = ack_message &&
                       std::holds_alternative<DataAckMessage>(*ack_message) &&
                       !std::get<DataAckMessage>(*ack_message).success &&
                       std::get<DataAckMessage>(*ack_message).object_id ==
                           ref.object_id;
    check(acked, name, failures);
    const auto done_message = done.recv_timeout(1000);
    check(done_message && std::holds_alternative<CompleteMessage>(*done_message) &&
              !std::get<CompleteMessage>(*done_message).success,
          name, failures);
    // Settlement must not wait for the 30-second transfer reaper.
    check(elapsed < 1.0, name, failures);
    check(calls == 0, name, failures);
  };

  // Matching transfer_info keeps each case isolated to the check it names.
  auto shm_info = [](int64_t size) {
    Value::Map transfer;
    transfer["size"] = Value(size);
    Value::Map info;
    info["transfer_info"] = Value(std::move(transfer));
    return Value(std::move(info));
  };
  expect_rejected("a negative buffer length was accepted",
                  make_ref("neg-length", -1, TransportKind::Shm, -1, shm_info(-1)));
  expect_rejected("an oversized buffer length was accepted",
                  make_ref("huge-length", int64_t{1} << 40, TransportKind::Shm,
                           int64_t{1} << 40, shm_info(int64_t{1} << 40)));
  expect_rejected("a conflicting tensor size was accepted",
                  make_ref("bad-tensor", 64, TransportKind::Shm, 65, shm_info(64)));
  {
    // Genuine mismatch: the DataRef says CUDA, the buffer says SHM.
    DataRef ref = make_ref("bad-transport", 64, TransportKind::Shm, 64,
                           shm_info(64));
    ref.transport = TransportKind::CudaIpc;
    expect_rejected("a transport mismatch was accepted", ref);
  }
  {
    // The ref claims SHM but the buffer advertises CUDA.
    DataRef ref = make_ref("mixed-transport", 64, TransportKind::Shm, 64,
                           shm_info(64));
    ref.buffer.transport = TransportKind::CudaIpc;
    expect_rejected("a buffer transport mismatch was accepted", ref);
  }
  expect_rejected("a payload above the configured limit was accepted",
                  make_ref("too-large", 8192, TransportKind::Shm, 8192,
                           shm_info(8192)));
  expect_rejected("a negative transfer size was accepted",
                  make_ref("neg-transfer", 64, TransportKind::Shm, 64,
                           shm_info(-1)));
  expect_rejected("a short transfer size was accepted",
                  make_ref("short-transfer", 64, TransportKind::Shm, 64,
                           shm_info(32)));
  expect_rejected("a long transfer size was accepted",
                  make_ref("long-transfer", 64, TransportKind::Shm, 64,
                           shm_info(128)));

  // A valid transfer still succeeds after the rejected ones.
  {
    StagePayload payload;
    payload.request_id = "valid-after-reject";
    payload.data = Value("ok");
    const std::string bytes = pack(payload.to_value());
    ShmRelay relay("validate_source", 1, 1);
    Tensor source;
    source.device = "cpu";
    source.dtype = "uint8";
    source.shape = {static_cast<int64_t>(bytes.size())};
    source.bytes = bytes;
    auto operation = relay.put(source, payload.request_id);
    DataRef ref = make_ref("valid-after-reject", static_cast<int64_t>(bytes.size()),
                           TransportKind::Shm,
                           static_cast<int64_t>(bytes.size()),
                           operation->metadata());
    DataReadyMessage ready;
    ready.request_id = payload.request_id;
    ready.from_stage = "a";
    ready.to_stage = "b";
    ready.data_ref = ref.to_value();
    sender.send(Message{ready});
    const auto ack_message = peer.recv_timeout(1000);
    check(ack_message && std::holds_alternative<DataAckMessage>(*ack_message) &&
              std::get<DataAckMessage>(*ack_message).success,
          "a valid transfer was rejected after malformed ones", failures);
    const auto done_message = done.recv_timeout(1000);
    check(done_message && std::holds_alternative<CompleteMessage>(*done_message) &&
              std::get<CompleteMessage>(*done_message).success,
          "a valid request did not complete after malformed ones", failures);
    check(calls == 1, "the valid request did not run compute", failures);
  }

  stage.stop();
  stage_thread.join();
  sender.close();
  abort.close();
  done.close();
  peer.close();
}

// ---- item 4: send_payload failure paths ----------------------------------

class NoAckOperation : public RelayOperation {
 public:
  const Value& metadata() const override { return metadata_; }
  void wait_for_completion(double) override {}

 private:
  Value metadata_ = Value::map({});
};

// Every relay has an operation without an ack protocol, so the base
// mark_receiver_failed throws. The cleanup path must swallow that.
class NoAckRelay : public Relay {
 public:
  std::unique_ptr<RelayOperation> put(const Tensor&, const std::string&) override {
    return std::make_unique<NoAckOperation>();
  }
  void get(const Value&, Tensor&) override {}
};

class ThrowingMetadataOperation : public NoAckOperation {
 public:
  const Value& metadata() const override {
    throw ProtocolError("metadata conversion failed");
  }
};

class ThrowingMetadataRelay : public NoAckRelay {
 public:
  std::unique_ptr<RelayOperation> put(const Tensor&, const std::string&) override {
    return std::make_unique<ThrowingMetadataOperation>();
  }
};

void test_send_payload_faults(int& failures) {
  struct Case {
    const char* name;
    std::shared_ptr<Relay> relay;
    const char* expected_error;
  };
  const Case cases[] = {
      {"a send failure left the transfer pending", nullptr, "zmq_connect"},
      // The base cleanup throws; the original send failure must still win.
      {"cleanup failure replaced the original exception",
       std::make_shared<NoAckRelay>(), "zmq_connect"},
      {"a metadata failure settled late", std::make_shared<ThrowingMetadataRelay>(),
       "metadata conversion failed"},
  };

  for (const Case& item : cases) {
    const std::string base = "/tmp/omni-runtime-fault-" +
                             std::to_string(getpid()) + "-" +
                             std::to_string(&item - cases) + "-";
    const std::string a_ep = "ipc://" + base + "a";
    const std::string done_ep = "ipc://" + base + "done";
    const std::string abort_ep = "ipc://" + base + "abort";
    for (const auto& path : {base + "a", base + "done", base + "abort"}) {
      std::remove(path.c_str());
    }

    Coordinator coordinator(done_ep, abort_ep);
    coordinator.register_stage("a", a_ep);
    coordinator.start();
    std::thread coordinator_thread([&] { coordinator.run(); });

    Stage::Config config;
    config.name = "a";
    config.recv_endpoint = a_ep;
    config.coordinator_endpoint = done_ep;
    config.abort_endpoint = abort_ep;
    // A connect to an unknown protocol fails inside send_to_stage.
    config.next_stages["b"] = "invalid://broken";
    config.peer_endpoints["a"] = a_ep;
    config.injected_relay = item.relay;
    Stage stage(config, [](const Value& input) { return input; });
    stage.start();
    std::thread stage_thread([&] { stage.run(); });

    StagePayload payload;
    payload.data = Value("x");
    const std::string request_id = coordinator.submit(payload, "a");
    const auto started = std::chrono::steady_clock::now();
    const bool settled = coordinator.wait(request_id, 3.0);
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
            .count();
    check(settled, item.name, failures);
    const auto info = coordinator.info(request_id);
    check(info && info->state == "failed", item.name, failures);
    if (info) {
      check(info->error.find(item.expected_error) != std::string::npos,
            item.name, failures);
    }
    check(elapsed < 2.0, item.name, failures);

    stage.stop();
    coordinator.stop();
    stage_thread.join();
    coordinator_thread.join();
  }
}

// ---- round 2: F9 abort/reap linearizes with the DataReady send -------------

struct LatchState {
  std::atomic<bool> entered{false};
  std::atomic<bool> release{false};
  std::atomic<int> settled{0};
};

class LatchOperation : public RelayOperation {
 public:
  explicit LatchOperation(std::shared_ptr<LatchState> state)
      : state_(std::move(state)) {}
  const Value& metadata() const override {
    state_->entered = true;
    while (!state_->release.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return metadata_;
  }
  void wait_for_completion(double) override {}
  void mark_receiver_done() override { ++state_->settled; }
  void mark_receiver_failed(const std::string&) override { ++state_->settled; }
  void mark_receiver_abandoned(const std::string&) override { ++state_->settled; }

 private:
  std::shared_ptr<LatchState> state_;
  Value metadata_ = Value::map({});
};

class LatchRelay : public Relay {
 public:
  explicit LatchRelay(std::shared_ptr<LatchState> state)
      : state_(std::move(state)) {}
  std::unique_ptr<RelayOperation> put(const Tensor&, const std::string&) override {
    return std::make_unique<LatchOperation>(state_);
  }
  void get(const Value&, Tensor&) override {}

 private:
  std::shared_ptr<LatchState> state_;
};

// metadata() blocks, the abort thread reaps the transfer, then metadata()
// returns: no DataReady may reach the next stage, and the operation settles
// exactly once.
void test_abort_before_data_ready(int& failures) {
  const std::string base =
      "/tmp/omni-runtime-latch-" + std::to_string(getpid()) + "-";
  const std::string a_ep = "ipc://" + base + "a";
  const std::string b_ep = "ipc://" + base + "b";
  const std::string done_ep = "ipc://" + base + "done";
  const std::string abort_ep = "ipc://" + base + "abort";
  for (const auto& path : {base + "a", base + "b", base + "done", base + "abort"}) {
    std::remove(path.c_str());
  }

  PullSocket downstream(b_ep, true);
  downstream.start();
  PullSocket done(done_ep, true);
  done.start();
  PubSocket abort(abort_ep);
  abort.bind();

  auto state = std::make_shared<LatchState>();
  Stage::Config config;
  config.name = "a";
  config.recv_endpoint = a_ep;
  config.coordinator_endpoint = done_ep;
  config.abort_endpoint = abort_ep;
  config.next_stages["b"] = b_ep;
  config.peer_endpoints["a"] = a_ep;
  config.transfer_timeout_ms = 0;
  config.injected_relay = std::make_shared<LatchRelay>(state);
  Stage stage(config, [](const Value& input) { return input; });
  stage.start();
  std::thread stage_thread([&] { stage.run(); });
  PushSocket entry(a_ep);
  entry.connect();

  StagePayload payload;
  payload.request_id = "latched";
  payload.data = Value("x");
  SubmitMessage submit;
  submit.request_id = "latched";
  submit.data = payload.to_value();
  entry.send(Message{submit});

  for (int i = 0; i < 200 && !state->entered.load(); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  check(state->entered.load(), "send_payload never reached metadata()", failures);

  abort.publish(AbortMessage{"latched"});
  for (int i = 0; i < 200 && state->settled.load() < 1; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  check(state->settled.load() == 1,
        "the abort did not settle the blocked transfer exactly once", failures);

  state->release = true;
  const auto ready = downstream.recv_timeout(700);
  check(!ready, "a reaped transfer still queued DataReady", failures);
  check(state->settled.load() == 1, "the transfer settled more than once",
        failures);

  const auto done_message = done.recv_timeout(1000);
  check(done_message && std::holds_alternative<CompleteMessage>(*done_message) &&
            !std::get<CompleteMessage>(*done_message).success,
        "the aborted transfer did not settle as failed", failures);

  stage.stop();  // joins the outbox thread; a hang fails the process alarm
  stage_thread.join();
  entry.close();
  abort.close();
  done.close();
  downstream.close();
}

// ---- round 2: F10 malformed DataRef still NACKs ---------------------------

void test_malformed_data_ref_nack(int& failures) {
  const std::string base =
      "/tmp/omni-runtime-nack-" + std::to_string(getpid()) + "-";
  const std::string b_ep = "ipc://" + base + "b";
  const std::string peer_ep = "ipc://" + base + "a";
  const std::string done_ep = "ipc://" + base + "done";
  const std::string abort_ep = "ipc://" + base + "abort";
  for (const auto& path : {base + "a", base + "b", base + "done", base + "abort"}) {
    std::remove(path.c_str());
  }

  PullSocket peer(peer_ep, true);
  peer.start();
  PullSocket done(done_ep, true);
  done.start();
  PubSocket abort(abort_ep);
  abort.bind();
  Stage::Config config;
  config.name = "b";
  config.recv_endpoint = b_ep;
  config.coordinator_endpoint = done_ep;
  config.abort_endpoint = abort_ep;
  config.peer_endpoints["a"] = peer_ep;
  std::atomic<int> calls{0};
  Stage stage(config, [&](const Value& value) {
    ++calls;
    return value;
  });
  stage.start();
  std::thread stage_thread([&] { stage.run(); });
  PushSocket sender(b_ep);
  sender.connect();

  // One sender credit: the next transfer can start only after the failure ack
  // settles the operation that owns it.
  ShmRelay sender_relay("nack_source", 1, 1);
  Tensor source;
  source.device = "cpu";
  source.dtype = "uint8";
  source.shape = {4};
  source.bytes = "abcd";
  auto first = sender_relay.put(source, "req-malformed");

  DataRef ref;
  ref.version = 1;
  ref.kind = DataKind::StagePayload;
  ref.object_id = "obj-malformed";
  ref.transport = TransportKind::Shm;
  ref.layout = DataLayout::PackedTensors;
  ref.buffer.transport = TransportKind::Shm;
  ref.buffer.info = first->metadata();
  ref.buffer.length = 4;
  TensorMeta meta;
  meta.path = "";
  meta.shape = {4};
  meta.dtype = "uint8";
  meta.device = "cpu";
  meta.offset = 0;
  meta.size = 4;
  ref.tensors.push_back(meta);
  Value ref_value = ref.to_value();
  // A valid object_id with a header that fails DataRef decoding.
  ref_value.as_map_mut()["header"] = Value(int64_t{7});

  DataReadyMessage ready;
  ready.request_id = "req-malformed";
  ready.from_stage = "a";
  ready.to_stage = "b";
  ready.data_ref = ref_value;
  const auto started = std::chrono::steady_clock::now();
  sender.send(Message{ready});

  const auto ack_message = peer.recv_timeout(1000);
  check(ack_message && std::holds_alternative<DataAckMessage>(*ack_message) &&
            !std::get<DataAckMessage>(*ack_message).success &&
            std::get<DataAckMessage>(*ack_message).object_id == "obj-malformed",
        "a malformed data_ref skipped its failure ack", failures);
  const auto done_message = done.recv_timeout(1000);
  const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  check(done_message && std::holds_alternative<CompleteMessage>(*done_message) &&
            !std::get<CompleteMessage>(*done_message).success,
        "a malformed data_ref did not fail its request", failures);
  check(elapsed < 1.0, "a malformed data_ref settlement was not prompt",
        failures);
  check(calls == 0, "a malformed data_ref executed compute", failures);

  std::atomic<bool> second_done{false};
  std::thread second([&] {
    auto op = sender_relay.put(source, "req-valid");
    op->mark_receiver_done();
    second_done = true;
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  check(!second_done, "a single sender credit was not held", failures);
  first->mark_receiver_failed("header must be bytes or base64 str");
  second.join();
  check(second_done, "the failure ack did not release the sender credit",
        failures);

  stage.stop();
  stage_thread.join();
  sender.close();
  abort.close();
  done.close();
  peer.close();
}

// ---- round 2: F11 unsupported transport rejected before allocation --------

void test_unsupported_transport_rejected(int& failures) {
  const std::string base =
      "/tmp/omni-runtime-transport-" + std::to_string(getpid()) + "-";
  const std::string b_ep = "ipc://" + base + "b";
  const std::string peer_ep = "ipc://" + base + "a";
  const std::string done_ep = "ipc://" + base + "done";
  const std::string abort_ep = "ipc://" + base + "abort";
  for (const auto& path : {base + "a", base + "b", base + "done", base + "abort"}) {
    std::remove(path.c_str());
  }

  PullSocket peer(peer_ep, true);
  peer.start();
  PullSocket done(done_ep, true);
  done.start();
  PubSocket abort(abort_ep);
  abort.bind();
  Stage::Config config;
  config.name = "b";
  config.recv_endpoint = b_ep;
  config.coordinator_endpoint = done_ep;
  config.abort_endpoint = abort_ep;
  config.peer_endpoints["a"] = peer_ep;
  config.max_payload_bytes = std::numeric_limits<int64_t>::max();
  std::atomic<int> calls{0};
  Stage stage(config, [&](const Value& value) {
    ++calls;
    return value;
  });
  stage.start();
  std::thread stage_thread([&] { stage.run(); });
  PushSocket sender(b_ep);
  sender.connect();

  DataRef ref;
  ref.version = 1;
  ref.kind = DataKind::StagePayload;
  ref.object_id = "local-object";
  ref.transport = TransportKind::LocalObject;
  ref.layout = DataLayout::PackedTensors;
  ref.buffer.transport = TransportKind::LocalObject;
  ref.buffer.info = Value::map({});
  ref.buffer.length = std::numeric_limits<int64_t>::max();
  TensorMeta meta;
  meta.path = "";
  meta.shape = {std::numeric_limits<int64_t>::max()};
  meta.dtype = "uint8";
  meta.device = "cpu";
  meta.offset = 0;
  meta.size = std::numeric_limits<int64_t>::max();
  ref.tensors.push_back(meta);

  DataReadyMessage ready;
  ready.request_id = "local-object";
  ready.from_stage = "a";
  ready.to_stage = "b";
  ready.data_ref = ref.to_value();
  const auto started = std::chrono::steady_clock::now();
  sender.send(Message{ready});

  const auto ack_message = peer.recv_timeout(1000);
  check(ack_message && std::holds_alternative<DataAckMessage>(*ack_message) &&
            !std::get<DataAckMessage>(*ack_message).success,
        "an unsupported transport skipped its failure ack", failures);
  const auto done_message = done.recv_timeout(1000);
  const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  check(done_message && std::holds_alternative<CompleteMessage>(*done_message) &&
            !std::get<CompleteMessage>(*done_message).success,
        "an unsupported transport did not fail its request", failures);
  if (done_message && std::holds_alternative<CompleteMessage>(*done_message)) {
    const auto& complete = std::get<CompleteMessage>(*done_message);
    check(complete.error &&
              complete.error->find("unsupported inbound relay transport") !=
                  std::string::npos,
          "an unsupported transport was rejected as a length error", failures);
  }
  check(elapsed < 1.0, "an unsupported transport settlement was not prompt",
        failures);
  check(calls == 0, "an unsupported transport executed compute", failures);

  stage.stop();
  stage_thread.join();
  sender.close();
  abort.close();
  done.close();
  peer.close();
}

// ---- round 2: F13 close keeps relays alive, rejects new work --------------

void test_router_close_keeps_relays(int& failures) {
  CommRouter router("a", std::nullopt, std::nullopt, {}, {}, {}, {});
  Relay& shm = router.relay(TransportKind::Shm);
  Tensor source;
  source.device = "cpu";
  source.dtype = "uint8";
  source.shape = {3};
  source.bytes = "abc";
  auto op = shm.put(source, "req-close");

  router.close();

  bool put_failed = false;
  try {
    (void)shm.put(source, "req-after-close");
  } catch (const ProtocolError&) {
    put_failed = true;
  }
  check(put_failed, "a relay closed by the router accepted new work", failures);

  bool access_failed = false;
  try {
    (void)router.relay(TransportKind::Shm);
  } catch (const ProtocolError&) {
    access_failed = true;
  }
  check(access_failed, "relay access after close returned a live relay", failures);

  bool new_kind_failed = false;
  try {
    (void)router.relay(TransportKind::Mooncake);
  } catch (const ProtocolError&) {
    new_kind_failed = true;
  }
  check(new_kind_failed, "close allowed a new relay kind to be built", failures);

  router.close();  // idempotent

  std::thread churn([&] {
    for (int i = 0; i < 200; ++i) router.cleanup("req-close");
  });
  router.close();
  churn.join();

  // The old pointer and its operation stay valid after close.
  op->mark_receiver_done();
  op.reset();
}

// ---- round 2: F14 slot_size_mb range -------------------------------------

void test_slot_size_range(int& failures) {
  const int64_t default_bytes = int64_t{512} * 1024 * 1024;
  CommRouter plain("a", std::nullopt, std::nullopt, {}, {}, {}, {});
  check(plain.cuda_slot_bytes() == default_bytes,
        "the default slot size changed", failures);

  std::map<std::string, Value> comm_config;
  comm_config["slot_size_mb"] = Value(int64_t{0});
  CommRouter defaulted("a", std::nullopt, std::nullopt, {}, {}, {}, {},
                       comm_config);
  check(defaulted.cuda_slot_bytes() == default_bytes,
        "a nonpositive slot size did not take the default", failures);

  comm_config["slot_size_mb"] = Value(int64_t{4294967297});
  CommRouter oversized("a", std::nullopt, std::nullopt, {}, {}, {}, {},
                       comm_config);
  bool rejected = false;
  try {
    (void)oversized.cuda_slot_bytes();
  } catch (const ProtocolError& e) {
    rejected = std::string(e.what()).find("slot_size_mb") != std::string::npos;
  }
  check(rejected, "an out-of-int-range slot_size_mb was narrowed", failures);

  bool parser_rejected = false;
  try {
    (void)create_relay("shm", {{"slot_size_mb", Value(int64_t{4294967297})}});
  } catch (const ProtocolError& e) {
    parser_rejected =
        std::string(e.what()).find("int range") != std::string::npos;
  }
  check(parser_rejected,
        "create_relay truncated an out-of-range slot_size_mb", failures);

  bool credits_rejected = false;
  try {
    (void)create_relay("shm", {{"credits", Value(int64_t{4294967297})}});
  } catch (const ProtocolError&) {
    credits_rejected = true;
  }
  check(credits_rejected, "create_relay truncated an out-of-range credits value",
        failures);
}

}  // namespace

int main() {
  alarm(20);
  int failures = 0;
  test_batch_strings_are_results(failures);
  test_batch_cost_error_reaches_request(failures);
  test_later_batch_cost_error(failures);
  test_abort_retention_and_callback(failures);
  test_batch_filters_aborted_inputs(failures);
  test_batch_cost_overflow(failures);
  test_scheduler_control_messages(failures);
  test_ack_releases_shm_credit(failures);
  test_wire_boundaries(failures);
  test_coordinator_terminal_state(failures);
  test_two_stage_pipeline(failures);
  test_transfer_ack_timeout(failures);
  test_abort_releases_pending_transfer(failures);
  test_sse_content_and_timeout(failures);
  test_abort_cleans_stage_request(failures);
  test_data_ready_checks_target(failures);
  test_data_ready_validation(failures);
  test_send_payload_faults(failures);
  test_abort_before_data_ready(failures);
  test_malformed_data_ref_nack(failures);
  test_unsupported_transport_rejected(failures);
  test_router_close_keeps_relays(failures);
  test_slot_size_range(failures);
  return failures == 0 ? 0 : 1;
}

// SPDX-License-Identifier: Apache-2.0
// Runnable checks for the runtime core: wire format, relay, router, scheduler,
// and one end-to-end pipeline request.
#include <atomic>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "umminfer/client/client.hpp"
#include "umminfer/comm/comm.hpp"
#include "umminfer/pipeline/pipeline.hpp"
#include "umminfer/proto/messages.hpp"
#include "umminfer/relay/relay.hpp"
#include "umminfer/scheduling/simple_scheduler.hpp"

using namespace umminfer;

namespace {

int failures = 0;

constexpr int kMaxIpcPaths = 128;
char ipc_root[PATH_MAX];
char ipc_paths[kMaxIpcPaths][PATH_MAX];
volatile sig_atomic_t ipc_path_active[kMaxIpcPaths];
volatile sig_atomic_t ipc_path_count = 0;
int ipc_foreign_paths = 0;

void cleanup_ipc_path(int index, int& failure_count) {
  if (!ipc_path_active[index]) return;
  if (::unlink(ipc_paths[index]) != 0 && errno != ENOENT) {
    std::cerr << "FAIL IPC cleanup unlink " << ipc_paths[index] << ": "
              << std::strerror(errno) << "\n";
    ++failure_count;
  }
  ipc_path_active[index] = 0;
}

void cleanup_ipc_paths(int& failure_count, bool include_root) {
  for (int i = 0; i < ipc_path_count; ++i) {
    cleanup_ipc_path(i, failure_count);
  }
  if (!include_root || ipc_root[0] == '\0') return;
  if (::rmdir(ipc_root) != 0) {
    // A foreign entry keeps the private root non-empty on purpose. Retaining it
    // is the required behaviour, not a cleanup failure.
    if (errno == ENOTEMPTY && ipc_foreign_paths > 0) {
      std::cerr << "IPC root retained: foreign entries in " << ipc_root << "\n";
    } else if (errno != ENOENT) {
      std::cerr << "FAIL IPC cleanup rmdir " << ipc_root << ": "
                << std::strerror(errno) << "\n";
      ++failure_count;
    }
  }
}

void ipc_signal_handler(int signal) {
  bool failed = false;
  for (int i = 0; i < ipc_path_count; ++i) {
    if (ipc_path_active[i] && ::unlink(ipc_paths[i]) != 0 && errno != ENOENT) {
      failed = true;
    }
    ipc_path_active[i] = 0;
  }
  if (ipc_root[0] != '\0' && ::rmdir(ipc_root) != 0 && errno != ENOENT &&
      !(errno == ENOTEMPTY && ipc_foreign_paths > 0)) {
    failed = true;
  }
  if (failed) {
    static const char message[] = "FAIL IPC signal cleanup\n";
    (void)::write(STDERR_FILENO, message, sizeof(message) - 1);
  }
  _exit(128 + signal);
}

void init_ipc_root() {
  char path[PATH_MAX];
  const int length = std::snprintf(path, sizeof(path),
                                   "/tmp/omni-self-check-XXXXXX");
  if (length < 0 || static_cast<size_t>(length) >= sizeof(path)) {
    throw std::runtime_error("IPC root path is too long");
  }
  if (::mkdtemp(path) == nullptr) {
    throw std::runtime_error(std::string("mkdtemp failed: ") +
                             std::strerror(errno));
  }
  if (::chmod(path, 0700) != 0) {
    const int error = errno;
    (void)::rmdir(path);
    throw std::runtime_error(std::string("chmod IPC root failed: ") +
                             std::strerror(error));
  }
  std::memcpy(ipc_root, path, static_cast<size_t>(length) + 1);
  std::cerr << "IPC_ROOT " << ipc_root << std::endl;
}

void install_ipc_signal_handlers() {
  struct sigaction action = {};
  action.sa_handler = ipc_signal_handler;
  sigemptyset(&action.sa_mask);
  for (const int signal : {SIGALRM, SIGINT, SIGTERM}) {
    if (::sigaction(signal, &action, nullptr) != 0) {
      throw std::runtime_error(std::string("sigaction failed: ") +
                               std::strerror(errno));
    }
  }
}

std::string ipc_root_path() { return std::string(ipc_root) + "/"; }

int register_ipc_path(const std::string& path) {
  if (path.size() >= PATH_MAX || ipc_path_count >= kMaxIpcPaths) {
    throw std::runtime_error("too many or too long IPC paths");
  }
  struct stat metadata {};
  errno = 0;
  const int stat_result = ::lstat(path.c_str(), &metadata);
  if (stat_result != 0 && errno != ENOENT) {
    throw std::runtime_error(std::string("lstat failed: ") + std::strerror(errno));
  }
  const int index = ipc_path_count;
  std::memcpy(ipc_paths[index], path.c_str(), path.size() + 1);
  ipc_path_active[index] = stat_result != 0;
  if (stat_result == 0) ++ipc_foreign_paths;
  ipc_path_count = static_cast<sig_atomic_t>(index + 1);
  return index;
}

struct IpcCleanup {
  IpcCleanup(int& failure_count, std::vector<std::string> owned_paths)
      : failures(&failure_count) {
    for (const auto& path : owned_paths) indices.push_back(register_ipc_path(path));
  }
  ~IpcCleanup() {
    for (const int index : indices) cleanup_ipc_path(index, *failures);
  }
  std::vector<int> indices;
  int* failures;
};

#define CHECK(cond)                                                          \
  do {                                                                       \
    if (!(cond)) {                                                           \
      std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << " " << #cond    \
                << "\n";                                                     \
      ++failures;                                                            \
    }                                                                        \
  } while (0)

// Test-only process control. No environment variable means no behavior change.
void ipc_test_hook(const char* phase, const std::string& delete_path = "") {
  const char* selected = std::getenv("OMNI_IPC_TEST_PHASE");
  if (selected == nullptr || std::string_view(selected) != phase) return;
  const char* fault = std::getenv("OMNI_IPC_TEST_FAULT");
  if (fault == nullptr) return;

  const char* gate = std::getenv("OMNI_IPC_TEST_GATE");
  if (gate != nullptr) {
    const std::string prefix = std::string(gate) + "." + phase;
    std::ofstream ready(prefix + ".ready", std::ios::trunc);
    ready << getpid() << " " << ipc_root << " " << phase << "\n";
    ready.close();
    if (!ready) throw std::runtime_error("could not write IPC test ready file");
    bool released = false;
    for (int i = 0; i < 500; ++i) {
      if (::access((prefix + ".go").c_str(), F_OK) == 0) {
        released = true;
        break;
      }
      ::usleep(20000);
    }
    if (!released) throw std::runtime_error("IPC test gate timed out");
  }

  if (std::string_view(fault) == "wait") {
    if (gate == nullptr) throw std::runtime_error("wait needs an IPC test gate");
    return;
  }
  if (std::string_view(fault) == "check") {
    CHECK(false);
    return;
  }
  if (std::string_view(fault) == "exception") {
    throw std::runtime_error("injected IPC test exception");
  }
  if (std::string_view(fault) == "delete-failure") {
    if (delete_path.empty()) throw std::runtime_error("missing delete test path");
    if (::unlink(delete_path.c_str()) != 0 && errno != ENOENT) {
      throw std::runtime_error("could not prepare delete failure");
    }
    if (::mkdir(delete_path.c_str(), 0700) != 0) {
      throw std::runtime_error("could not inject delete failure");
    }
    std::cerr << "injected IPC delete failure at " << delete_path << "\n";
    return;
  }
  throw std::runtime_error("unknown IPC test fault");
}

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

  const Message ready_decoded = deserialize_message(
      serialize_message(Message{StageReadyMessage{"entry"}}));
  CHECK(std::get<StageReadyMessage>(ready_decoded).from_stage == "entry");
}

void test_shm_relay() {
  // One credit: the second put can only return after the first ack frees it.
  ShmRelay relay("test_engine", 1, 1);
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

  std::atomic<bool> started{false};
  std::atomic<bool> second_returned{false};
  std::unique_ptr<RelayOperation> op2;
  std::thread waiter([&] {
    started = true;
    op2 = relay.put(src, "req-2");
    second_returned = true;
  });
  while (!started.load()) std::this_thread::yield();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  CHECK(!second_returned);  // the credit is still held by req-1

  op->mark_receiver_done();
  waiter.join();
  CHECK(second_returned);
  CHECK(op2 != nullptr && op2->metadata().contains("transfer_info"));
  if (op2) op2->mark_receiver_done();
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
      ipc_root_path() + "omni-self-check-" + std::to_string(getpid()) + "-";
  const std::string entry_ep = "ipc://" + base + "entry";
  const std::string completion_ep = "ipc://" + base + "completion";
  const std::string abort_ep = "ipc://" + base + "abort";
  const IpcCleanup ipc_cleanup{
      failures, {base + "entry", base + "completion", base + "abort"}};

  Coordinator coordinator(completion_ep, abort_ep);
  coordinator.register_stage("entry", entry_ep);
  coordinator.start();
  ipc_test_hook("partial");
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
  ipc_test_hook("full", base + "entry");
}

void test_chat_rejects_reserved_extra_stream() {
  client::OpenAIConfig config;
  config.base_url = "http://127.0.0.1:1";
  config.model = "test-model";
  config.max_attempts = 1;
  config.connect_timeout = client::Milliseconds{10};
  const client::OpenAIClient api(config);

  const auto expect_config_error = [&](nlohmann::json value, bool typed_stream) {
    client::ChatRequest request;
    request.messages = {{"user", "ping"}};
    request.stream = typed_stream;
    request.extra_body = nlohmann::json{{"stream", std::move(value)}};
    try {
      api.chat(request);
      CHECK(false);
    } catch (const client::Error& error) {
      CHECK(error.code() == client::ErrorCode::kConfig);
    } catch (...) {
      CHECK(false);
    }
  };
  expect_config_error(true, false);
  expect_config_error(false, false);
  expect_config_error(1, false);
  expect_config_error(true, true);
}

}  // namespace

int main() {
  try {
    init_ipc_root();
    install_ipc_signal_handlers();
    alarm(20);
    ipc_test_hook("start");
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
    std::cerr << "[8] chat reserved stream\n";
    test_chat_rejects_reserved_extra_stream();
  } catch (const std::exception& error) {
    std::cerr << "FAIL exception: " << error.what() << "\n";
    ++failures;
  } catch (...) {
    std::cerr << "FAIL unknown exception\n";
    ++failures;
  }
  cleanup_ipc_paths(failures, true);
  alarm(0);
  std::cerr << "[done]\n";

  if (failures != 0) {
    std::cerr << failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "all checks passed\n";
  return 0;
}

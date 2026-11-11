// SPDX-License-Identifier: Apache-2.0
// CUDA IPC relay checks on two real GPUs.
//
// Every receiver runs in its own exec'd process, so the CUDA runtime starts from
// a clean image. Forking after CUDA state exists in the parent is unsupported by
// the CUDA runtime and must not be used here.
//
// Layout per test: the parent process is the sender (GPU 0); a child process is
// the receiver (GPU 1 unless the test asks for the same device). Control
// metadata travels over a pipe, packed exactly as the control plane packs it.
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "omniinfer/comm/comm.hpp"
#include "omniinfer/relay/relay.hpp"
#include "omniinfer/proto/wire.hpp"

using namespace omniinfer;

namespace {

int failures = 0;

#define CHECK(cond)                                                       \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << " " << #cond \
                << "\n";                                                  \
      ++failures;                                                         \
    }                                                                     \
  } while (0)

constexpr size_t kMaxAckError = 512;

uint64_t fnv1a(const std::string& data) {
  uint64_t h = 1469598103934665603ull;
  for (unsigned char c : data) {
    h ^= c;
    h *= 1099511628211ull;
  }
  return h;
}

bool write_all(int fd, const void* data, size_t n) {
  const char* p = static_cast<const char*>(data);
  while (n > 0) {
    const ssize_t w = ::write(fd, p, n);
    if (w < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    p += w;
    n -= static_cast<size_t>(w);
  }
  return true;
}

bool read_all(int fd, void* data, size_t n) {
  char* p = static_cast<char*>(data);
  while (n > 0) {
    const ssize_t r = ::read(fd, p, n);
    if (r < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    if (r == 0) return false;
    p += r;
    n -= static_cast<size_t>(r);
  }
  return true;
}

bool send_metadata(int fd, const Value& metadata) {
  const std::string payload = pack(metadata);
  const uint32_t size = static_cast<uint32_t>(payload.size());
  return write_all(fd, &size, sizeof(size)) &&
         write_all(fd, payload.data(), payload.size());
}

bool recv_metadata(int fd, Value& out) {
  uint32_t size = 0;
  if (!read_all(fd, &size, sizeof(size))) return false;
  if (size > 64u * 1024u * 1024u) return false;
  std::string payload(size, '\0');
  if (!read_all(fd, payload.data(), payload.size())) return false;
  out = unpack(payload);
  return true;
}

// ok=0 means the receiver copied successfully, ok=1 means get() threw. The
// bounded error string proves which failure the receiver reported.
struct Ack {
  uint8_t ok = 1;
  uint32_t size = 0;
  uint64_t hash = 0;
  std::string error;
};

bool send_ack(int fd, const Ack& ack) {
  const uint16_t length =
      static_cast<uint16_t>(std::min(ack.error.size(), kMaxAckError));
  return write_all(fd, &ack.ok, sizeof(ack.ok)) &&
         write_all(fd, &ack.size, sizeof(ack.size)) &&
         write_all(fd, &ack.hash, sizeof(ack.hash)) &&
         write_all(fd, &length, sizeof(length)) &&
         write_all(fd, ack.error.data(), length);
}

bool recv_ack(int fd, Ack& ack) {
  uint16_t length = 0;
  if (!read_all(fd, &ack.ok, sizeof(ack.ok)) ||
      !read_all(fd, &ack.size, sizeof(ack.size)) ||
      !read_all(fd, &ack.hash, sizeof(ack.hash)) ||
      !read_all(fd, &length, sizeof(length))) {
    return false;
  }
  if (length > kMaxAckError) return false;
  ack.error.assign(length, '\0');
  return read_all(fd, ack.error.data(), length);
}

struct Child {
  pid_t pid = -1;
  int to_child = -1;
  int from_child = -1;
  int go = -1;
};

std::string self_exe() {
  char buf[4096];
  const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (n <= 0) return {};
  buf[n] = '\0';
  return buf;
}

std::string arg_value(int argc, char** argv, const std::string& key) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (key == argv[i]) return argv[i + 1];
  }
  return {};
}

bool wait_bounded(pid_t pid, int seconds) {
  for (int i = 0; i < seconds * 100; ++i) {
    int status = 0;
    const pid_t r = ::waitpid(pid, &status, WNOHANG);
    if (r == pid) return WIFEXITED(status) && WEXITSTATUS(status) == 0;
    if (r < 0) return false;
    ::usleep(10000);
  }
  ::kill(pid, SIGKILL);
  ::waitpid(pid, nullptr, 0);
  return false;
}

bool finish_child(Child& child) {
  for (int fd : {child.to_child, child.from_child, child.go}) {
    if (fd >= 0) ::close(fd);
  }
  child.to_child = child.from_child = child.go = -1;
  return wait_bounded(child.pid, 10);
}

// Spawns this binary with a receiver role. The child is a fresh process image,
// so no CUDA state is inherited. Pipe numbers are passed explicitly, so no
// descriptor is assumed to be free.
Child spawn_receiver(const std::string& mode, int device, int credits,
                     int count = 1) {
  Child child;
  int to_child[2] = {-1, -1};
  int from_child[2] = {-1, -1};
  int go[2] = {-1, -1};
  if (::pipe(to_child) != 0 || ::pipe(from_child) != 0 || ::pipe(go) != 0) {
    return child;
  }
  const std::string exe = self_exe();
  if (exe.empty()) return child;
  const pid_t pid = ::fork();
  if (pid == 0) {
    // Keep sanitizer and runtime descriptors intact.
    ::close(to_child[1]);
    ::close(from_child[0]);
    ::close(go[1]);
    const std::string dev = std::to_string(device);
    const std::string cred = std::to_string(credits);
    const std::string n = std::to_string(count);
    const std::string meta_fd = std::to_string(to_child[0]);
    const std::string ack_fd = std::to_string(from_child[1]);
    const std::string go_fd = std::to_string(go[0]);
    char* argv[] = {const_cast<char*>(exe.c_str()),
                    const_cast<char*>("--role"),
                    const_cast<char*>(mode.c_str()),
                    const_cast<char*>("--device"),
                    const_cast<char*>(dev.c_str()),
                    const_cast<char*>("--credits"),
                    const_cast<char*>(cred.c_str()),
                    const_cast<char*>("--count"),
                    const_cast<char*>(n.c_str()),
                    const_cast<char*>("--meta-fd"),
                    const_cast<char*>(meta_fd.c_str()),
                    const_cast<char*>("--ack-fd"),
                    const_cast<char*>(ack_fd.c_str()),
                    const_cast<char*>("--go-fd"),
                    const_cast<char*>(go_fd.c_str()),
                    nullptr};
    ::execv(exe.c_str(), argv);
    ::_exit(127);
  }
  ::close(to_child[0]);
  ::close(from_child[1]);
  ::close(go[0]);
  if (pid < 0) return child;
  child.pid = pid;
  child.to_child = to_child[1];
  child.from_child = from_child[0];
  child.go = go[1];
  return child;
}

std::map<std::string, Value> relay_config(int device, int credits,
                                          int slot_mb = 64) {
  std::map<std::string, Value> config;
  config["engine_id"] = Value("cuda-ipc-test");
  config["device"] = Value(device);
  config["credits"] = Value(credits);
  config["slot_size_mb"] = Value(slot_mb);
  return config;
}

std::unique_ptr<Relay> make_relay(int device, int credits, int slot_mb = 64) {
  return create_relay("cuda_ipc", relay_config(device, credits, slot_mb));
}

Tensor host_tensor(const std::string& bytes) {
  Tensor tensor;
  tensor.device = "cpu";
  tensor.dtype = "uint8";
  tensor.shape = {static_cast<int64_t>(bytes.size())};
  tensor.bytes = bytes;
  return tensor;
}

// ---- receiver roles ------------------------------------------------------

int receiver_main(int argc, char** argv) {
  ::alarm(60);
  const std::string mode = arg_value(argc, argv, "--role");
  const int device = std::stoi(arg_value(argc, argv, "--device"));
  const int credits = std::stoi(arg_value(argc, argv, "--credits"));
  const int count = std::stoi(arg_value(argc, argv, "--count"));
  const int meta_fd = std::stoi(arg_value(argc, argv, "--meta-fd"));
  const int ack_fd = std::stoi(arg_value(argc, argv, "--ack-fd"));
  const int go_fd = std::stoi(arg_value(argc, argv, "--go-fd"));
  auto relay = make_relay(device, credits);

  auto receive_one = [&](bool wait_for_go) {
    Value metadata;
    if (!recv_metadata(meta_fd, metadata)) return 1;
    if (wait_for_go) {
      char byte = 0;
      if (!read_all(go_fd, &byte, 1)) return 1;
    }
    Tensor dest;
    dest.device = "cpu";
    dest.dtype = "uint8";
    Ack ack;
    try {
      // The stage sizes the destination from the DataRef length; the relay
      // rejects a smaller destination instead of resizing it. Malformed
      // metadata must produce a failed ack, not kill this process.
      const Value& info = require_field(metadata, "transfer_info");
      const int64_t declared = require_field(info, "size").as_int();
      dest.bytes.assign(declared > 0 ? static_cast<size_t>(declared) : 0, '\0');
      relay->get(metadata, dest);
      ack.ok = 0;
      ack.size = static_cast<uint32_t>(dest.bytes.size());
      ack.hash = fnv1a(dest.bytes);
    } catch (const std::exception& e) {
      std::cerr << "receiver " << mode << " get failed: " << e.what() << "\n";
      ack.ok = 1;
      ack.error = e.what();
    }
    return send_ack(ack_fd, ack) ? 0 : 1;
  };

  // Reads the first `count` blocks, then acknowledges them in reverse order.
  if (mode == "reverse") {
    std::vector<std::pair<Ack, uint32_t>> results;
    for (int i = 0; i < count; ++i) {
      Value metadata;
      if (!recv_metadata(meta_fd, metadata)) return 1;
      Tensor dest;
      dest.device = "cpu";
      dest.dtype = "uint8";
      const Value& size_info = require_field(metadata, "transfer_info");
      Ack ack;
      try {
        const int64_t declared = require_field(size_info, "size").as_int();
        dest.bytes.assign(declared > 0 ? static_cast<size_t>(declared) : 0, '\0');
        relay->get(metadata, dest);
        ack.ok = 0;
        ack.size = static_cast<uint32_t>(dest.bytes.size());
        ack.hash = fnv1a(dest.bytes);
      } catch (const std::exception& e) {
        ack.ok = 1;
        ack.error = e.what();
      }
      results.emplace_back(
          ack,
          static_cast<uint32_t>(require_field(size_info, "req_index").as_int()));
    }
    std::sort(results.begin(), results.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    for (const auto& [ack, index] : results) {
      (void)index;
      if (!send_ack(ack_fd, ack)) return 1;
    }
    return 0;
  }

  for (int i = 0; i < count; ++i) {
    if (receive_one(mode == "delayed" && i == 0) != 0) return 1;
  }
  return 0;
}

// ---- parent helpers ------------------------------------------------------

std::string binary_payload(size_t n) {
  std::string bytes(n, '\0');
  for (size_t i = 0; i < n; ++i) {
    bytes[i] = static_cast<char>((i * 37u + 11u) & 0xffu);
  }
  return bytes;
}

Value metadata_with_index(const Value& metadata, int index) {
  Value copy = metadata;
  copy.as_map_mut()["transfer_info"].as_map_mut()["req_index"] = Value(index);
  return copy;
}

void test_roundtrip(bool empty, const char* label, int receiver_device) {
  std::cerr << label << "\n";
  Child child = spawn_receiver("roundtrip", receiver_device, 2);
  CHECK(child.pid > 0);
  if (child.pid <= 0) return;

  const std::string bytes = empty ? std::string() : binary_payload(4096);
  auto relay = make_relay(0, 2);
  auto op = relay->put(host_tensor(bytes), "req-roundtrip");
  CHECK(send_metadata(child.to_child, op->metadata()));

  Ack ack;
  CHECK(recv_ack(child.from_child, ack));
  CHECK(ack.ok == 0);
  CHECK(ack.size == bytes.size());
  if (!empty) CHECK(ack.hash == fnv1a(bytes));
  op->mark_receiver_done();
  CHECK(finish_child(child));
}

// Sizes around the msgpack and page boundaries. The two large cases only run
// when OMNIINFER_CUDA_TEST_STRESS=1, so the default test stays fast.
void test_sizes() {
  std::cerr << "[3] boundary sizes\n";
  std::vector<size_t> sizes = {1, 4095, 4096, 4097};
  const bool stress = std::getenv("OMNIINFER_CUDA_TEST_STRESS") != nullptr;
  if (stress) {
    sizes.push_back(1u << 20);
    sizes.push_back(64u * 1024u * 1024u - 1);  // slot limit for this relay
  }

  Child child = spawn_receiver("multi", 1, 2, static_cast<int>(sizes.size()));
  CHECK(child.pid > 0);
  if (child.pid <= 0) return;

  auto relay = make_relay(0, 2);
  for (size_t size : sizes) {
    const std::string bytes = binary_payload(size);
    auto op = relay->put(host_tensor(bytes), "req-size");
    CHECK(send_metadata(child.to_child, op->metadata()));
    Ack ack;
    CHECK(recv_ack(child.from_child, ack));
    CHECK(ack.ok == 0 && ack.size == size && ack.hash == fnv1a(bytes));
    op->mark_receiver_done();
  }
  CHECK(finish_child(child));
}

// One credit, three sequential transfers: the slot is reused, grown for the
// 1 MiB round, and reused again for the small round without shrinking.
void test_slot_grow_and_reuse() {
  std::cerr << "[3b] slot reuse and growth\n";
  const std::vector<size_t> sizes = {4096, 1u << 20, 8192};
  Child child = spawn_receiver("multi", 1, 1, static_cast<int>(sizes.size()));
  CHECK(child.pid > 0);
  if (child.pid <= 0) return;

  auto relay = make_relay(0, 1);
  for (size_t size : sizes) {
    const std::string bytes = binary_payload(size);
    auto op = relay->put(host_tensor(bytes), "req-grow");
    CHECK(send_metadata(child.to_child, op->metadata()));
    Ack ack;
    CHECK(recv_ack(child.from_child, ack));
    CHECK(ack.ok == 0 && ack.size == size && ack.hash == fnv1a(bytes));
    op->mark_receiver_done();
  }
  CHECK(finish_child(child));
}

void test_credit_unlock() {
  std::cerr << "[4] credits gate and ack unlock\n";
  Child child = spawn_receiver("credits", 1, 1, 2);
  CHECK(child.pid > 0);
  if (child.pid <= 0) return;

  auto relay = make_relay(0, 1);
  const std::string first = binary_payload(1024);
  auto op1 = relay->put(host_tensor(first), "req-credit-1");
  CHECK(send_metadata(child.to_child, op1->metadata()));
  Ack ack;
  CHECK(recv_ack(child.from_child, ack));
  CHECK(ack.ok == 0 && ack.size == first.size() && ack.hash == fnv1a(first));

  // The credit is still held: a second put must block.
  std::atomic<bool> second_done{false};
  std::unique_ptr<RelayOperation> op2;
  std::thread second([&] {
    op2 = relay->put(host_tensor("second"), "req-credit-2");
    second_done = true;
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  CHECK(!second_done);
  op1->mark_receiver_done();
  op1.reset();
  second.join();
  CHECK(second_done);

  CHECK(send_metadata(child.to_child, op2->metadata()));
  CHECK(recv_ack(child.from_child, ack));
  CHECK(ack.ok == 0 && ack.size == 6 && ack.hash == fnv1a("second"));
  op2->mark_receiver_done();
  CHECK(finish_child(child));
}

void test_bad_handle_and_failure_ack() {
  std::cerr << "[5] bad handle and receiver failure ack\n";
  Child child = spawn_receiver("roundtrip", 1, 1);
  CHECK(child.pid > 0);
  if (child.pid <= 0) return;

  auto relay = make_relay(0, 1);
  auto op1 = relay->put(host_tensor(binary_payload(64)), "req-bad");
  Value metadata = op1->metadata();
  Value::Map fields = metadata.find("transfer_info")->as_map();
  std::string handle = fields["ipc_handle"].as_bin();
  // Reserved bytes can be ignored by the driver. Invalidate the entire opaque
  // handle instead of changing only its first byte.
  std::fill(handle.begin(), handle.end(), static_cast<char>(0xff));
  fields["ipc_handle"] = Value::bin(handle);
  metadata.as_map_mut()["transfer_info"] = Value(std::move(fields));
  CHECK(send_metadata(child.to_child, metadata));

  Ack ack;
  CHECK(recv_ack(child.from_child, ack));
  CHECK(ack.ok == 1);
  // Proves the failure came from opening the handle, not from another fault.
  CHECK(ack.error.find("cudaIpcOpenMemHandle") != std::string::npos);
  op1->mark_receiver_failed("bad handle");
  op1.reset();

  // The failed ack frees the credit, so the next put succeeds.
  auto op2 = relay->put(host_tensor("after-failure"), "req-after");
  CHECK(op2 != nullptr);
  op2->mark_receiver_done();
  CHECK(finish_child(child));
}

// Malformed metadata must be rejected before any copy.
void test_malformed_metadata() {
  std::cerr << "[6] malformed metadata\n";
  struct Case {
    const char* name;
    const char* expected;
    std::function<Value(const Value&)> mutate;
  };
  const std::vector<Case> cases = {
      {"wrong handle length", "must be",
       [](const Value& base) {
         Value copy = base;
         copy.as_map_mut()["transfer_info"].as_map_mut()["ipc_handle"] =
             Value::bin(std::string(8, '\0'));
         return copy;
       }},
      {"negative size", "negative",
       [](const Value& base) {
         Value copy = base;
         copy.as_map_mut()["transfer_info"].as_map_mut()["size"] = Value(-1);
         return copy;
       }},
      {"missing handle", "missing field",
       [](const Value& base) {
         Value copy = base;
         copy.as_map_mut()["transfer_info"].as_map_mut().erase("ipc_handle");
         return copy;
       }},
      {"missing transfer info", "missing field",
       [](const Value& base) {
         Value copy = base;
         copy.as_map_mut().erase("transfer_info");
         return copy;
       }},
  };

  for (const Case& item : cases) {
    Child child = spawn_receiver("roundtrip", 1, 1);
    CHECK(child.pid > 0);
    if (child.pid <= 0) continue;
    auto relay = make_relay(0, 1);
    auto op = relay->put(host_tensor(binary_payload(32)), "req-malformed");
    CHECK(send_metadata(child.to_child, item.mutate(op->metadata())));
    Ack ack;
    CHECK(recv_ack(child.from_child, ack));
    CHECK(ack.ok == 1);
    CHECK(ack.error.find(item.expected) != std::string::npos);
    std::cerr << "  " << item.name << ": " << ack.error << "\n";
    op->mark_receiver_failed(item.name);
    op.reset();
    CHECK(finish_child(child));
  }
}

// The sender resizes the destination from the transfer size. A destination
// already sized smaller by the DataRef must be rejected, not overwritten.
void test_size_mismatch() {
  std::cerr << "[7] transfer larger than the declared destination\n";
  auto relay = make_relay(0, 1);
  auto op = relay->put(host_tensor(binary_payload(128)), "req-mismatch");
  Tensor dest;
  dest.device = "cpu";
  dest.dtype = "uint8";
  dest.bytes.assign(64, '\0');
  bool threw = false;
  try {
    relay->get(op->metadata(), dest);
  } catch (const ProtocolError&) {
    threw = true;
  }
  CHECK(threw);
  CHECK(dest.bytes.size() == 64);
  op->mark_receiver_failed("size mismatch");
}

void test_abandon_fail_fast() {
  std::cerr << "[8] abandon poisons the edge\n";
  auto relay = make_relay(0, 2);
  auto op = relay->put(host_tensor(binary_payload(32)), "req-abandon");
  op->mark_receiver_abandoned("test abandon");
  bool threw = false;
  try {
    relay->put(host_tensor("never"), "req-blocked");
  } catch (const ProtocolError&) {
    threw = true;
  }
  CHECK(threw);
  op.reset();
}

void test_close_wakes_acquire() {
  std::cerr << "[9] close wakes a blocked acquire\n";
  auto relay = make_relay(0, 1);
  auto op = relay->put(host_tensor("held"), "req-hold");

  std::atomic<bool> blocked_threw{false};
  std::thread waiter([&] {
    try {
      auto blocked = relay->put(host_tensor("blocked"), "req-blocked");
      blocked->mark_receiver_done();
    } catch (const ProtocolError&) {
      blocked_threw = true;
    }
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  CHECK(!blocked_threw);
  relay->close();
  waiter.join();
  CHECK(blocked_threw);
  op.reset();
}

// The abandoned block must stay readable: close() may not free it. The sender
// waits for the receiver to finish before exiting.
void test_delayed_receiver() {
  std::cerr << "[10] abandoned block survives close for a late receiver\n";
  Child child = spawn_receiver("delayed", 1, 2);
  CHECK(child.pid > 0);
  if (child.pid <= 0) return;

  const std::string bytes = binary_payload(2048);
  auto relay = make_relay(0, 2);
  auto op = relay->put(host_tensor(bytes), "req-delayed");
  CHECK(send_metadata(child.to_child, op->metadata()));
  // The receiver holds the metadata and waits for the go byte.
  op->mark_receiver_abandoned("sender gave up");
  relay->close();
  op.reset();

  const char go = 'g';
  CHECK(write_all(child.go, &go, 1));
  Ack ack;
  CHECK(recv_ack(child.from_child, ack));
  CHECK(ack.ok == 0 && ack.size == bytes.size() && ack.hash == fnv1a(bytes));
  CHECK(finish_child(child));
}

// Out-of-order acks and duplicate terminal calls must return each credit
// exactly once. The proof is behavioral: with one credit, a duplicate ack
// must not make a second put succeed without an ack.
void test_out_of_order_and_duplicate_acks() {
  std::cerr << "[11] out-of-order and duplicate acks\n";
  {
    Child child = spawn_receiver("reverse", 1, 3, 3);
    CHECK(child.pid > 0);
    if (child.pid > 0) {
      auto relay = make_relay(0, 3);
      std::vector<std::unique_ptr<RelayOperation>> ops;
      for (int i = 0; i < 3; ++i) {
        ops.push_back(relay->put(host_tensor(binary_payload(64 + i)), "req-o"));
        CHECK(send_metadata(child.to_child,
                            metadata_with_index(ops.back()->metadata(), i)));
      }
      // Payload sizes are 64, 65 and 66 bytes. Acks arrive in reverse index
      // order, so the reported sizes must be 66, 65, 64.
      const size_t expected[] = {66, 65, 64};
      for (size_t i = 0; i < 3; ++i) {
        Ack ack;
        CHECK(recv_ack(child.from_child, ack));
        CHECK(ack.ok == 0);
        CHECK(ack.size == expected[i]);
      }
      // Acks arrived 2,1,0; every credit is back.
      ops[2]->mark_receiver_done();
      ops[0]->mark_receiver_done();
      ops[1]->mark_receiver_done();
      ops.clear();
      auto again = relay->put(host_tensor("again"), "req-again");
      CHECK(again != nullptr);
      again->mark_receiver_done();
      CHECK(finish_child(child));
    }
  }
  {
    // One credit: a duplicate completion must not create a second credit.
    auto relay = make_relay(0, 1);
    auto op = relay->put(host_tensor("x"), "req-dup");
    op->mark_receiver_done();
    op->mark_receiver_done();              // no-op
    op->mark_receiver_abandoned("late");   // no-op after done
    op.reset();

    auto held = relay->put(host_tensor("held"), "req-held");
    std::atomic<bool> third_done{false};
    std::thread third([&] {
      auto blocked = relay->put(host_tensor("third"), "req-third");
      blocked->mark_receiver_done();
      third_done = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    CHECK(!third_done);  // a double release would have let this through
    held->mark_receiver_done();
    held.reset();
    third.join();
    CHECK(third_done);
  }
}

// Abandoning one edge must not disable another. Each CUDA target owns a relay.
void test_abandon_is_scoped_per_target() {
  std::cerr << "[12] abandon is scoped to one target\n";
  std::set<std::string> gpu_names = {"A", "B", "C"};
  std::map<std::string, std::vector<int>> gpu_ids = {
      {"A", {0}}, {"B", {1}}, {"C", {1}}};
  std::map<std::string, Value> comm_config;
  comm_config["credits"] = Value(1);
  CommRouter router("A", 0, 0, {}, gpu_names, gpu_ids, {}, comm_config);

  Value::Map payload_map;
  Tensor device_tensor;
  device_tensor.device = "cuda:0";
  device_tensor.dtype = "uint8";
  payload_map["x"] = Value(device_tensor);
  const Value payload(std::move(payload_map));

  auto [kind_b, relay_b] = router.relay_for_payload("B", payload);
  auto [kind_c, relay_c] = router.relay_for_payload("C", payload);
  CHECK(kind_b == TransportKind::CudaIpc);
  CHECK(kind_c == TransportKind::CudaIpc);
  CHECK(relay_b != relay_c);

  auto op_b = relay_b->put(host_tensor("b"), "req-b");
  op_b->mark_receiver_abandoned("edge failed");
  bool b_closed = false;
  try {
    relay_b->put(host_tensor("b2"), "req-b2");
  } catch (const ProtocolError&) {
    b_closed = true;
  }
  CHECK(b_closed);

  // C still roundtrips through the same router.
  auto op_c = relay_c->put(host_tensor("c"), "req-c");
  CHECK(op_c != nullptr);
  op_c->mark_receiver_done();
  op_c.reset();

  // close() must wake a waiter blocked on the healthy edge.
  auto held = relay_c->put(host_tensor("held"), "req-held");
  std::atomic<bool> waiter_threw{false};
  std::thread waiter([&] {
    try {
      auto blocked = relay_c->put(host_tensor("blocked"), "req-blocked");
      blocked->mark_receiver_done();
    } catch (const ProtocolError&) {
      waiter_threw = true;
    }
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  CHECK(!waiter_threw);
  router.close();
  waiter.join();
  CHECK(waiter_threw);
  held.reset();
}

}  // namespace

int main(int argc, char** argv) {
#if !defined(OMNIINFER_WITH_CUDA)
  std::cerr << "cuda_ipc_test: built without OMNIINFER_WITH_CUDA, skipped\n";
  (void)argc;
  (void)argv;
  return 0;
#else
  // A failed receiver must produce a failed check, not kill the sender with
  // SIGPIPE before it can report the failing test.
  ::signal(SIGPIPE, SIG_IGN);
  if (arg_value(argc, argv, "--role").size() > 0) {
    return receiver_main(argc, argv);
  }
  ::alarm(300);
  test_roundtrip(false, "[1] binary roundtrip GPU0->GPU1", 1);
  test_roundtrip(true, "[2] zero-byte transfer", 1);
  test_sizes();
  test_slot_grow_and_reuse();
  test_roundtrip(false, "[2b] same-GPU cross-process roundtrip", 0);
  test_credit_unlock();
  test_bad_handle_and_failure_ack();
  test_malformed_metadata();
  test_size_mismatch();
  test_abandon_fail_fast();
  test_close_wakes_acquire();
  test_delayed_receiver();
  test_out_of_order_and_duplicate_acks();
  test_abandon_is_scoped_per_target();
  if (failures != 0) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cerr << "all checks passed\n";
  return 0;
#endif
}

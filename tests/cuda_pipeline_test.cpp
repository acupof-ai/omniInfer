// SPDX-License-Identifier: Apache-2.0
// End-to-end GPU pipeline check: one coordinator, two stage processes.
//
// Stage A runs on GPU 0, stage B on GPU 1. The submitted payload carries a CUDA
// tensor value, so A->B must select the cuda_ipc relay. The same request must
// still complete over SHM when OMNIINFER_DISABLE_CUDA_IPC=1.
//
// Stages are spawned with fork+exec, so each CUDA runtime starts from a clean
// process image instead of inheriting coordinator state.
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "omniinfer/comm/comm.hpp"
#include "omniinfer/pipeline/pipeline.hpp"
#include "omniinfer/proto/messages.hpp"
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

std::string arg_value(int argc, char** argv, const std::string& key) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (key == argv[i]) return argv[i + 1];
  }
  return {};
}

std::string self_exe() {
  char buf[4096];
  const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (n <= 0) return {};
  buf[n] = '\0';
  return buf;
}

std::string ipc_path(const std::string& endpoint) {
  const std::string prefix = "ipc://";
  if (endpoint.compare(0, prefix.size(), prefix) == 0) {
    return endpoint.substr(prefix.size());
  }
  return endpoint;
}

void remove_endpoint(const std::string& endpoint) {
  std::remove(ipc_path(endpoint).c_str());
}

bool wait_for_path(const std::string& endpoint, int seconds) {
  for (int i = 0; i < seconds * 100; ++i) {
    if (::access(ipc_path(endpoint).c_str(), F_OK) == 0) return true;
    ::usleep(10000);
  }
  return false;
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

bool stop_child(pid_t pid, PushSocket& control) {
  if (pid <= 0) return false;
  try {
    control.send(Message{ShutdownMessage{}});
  } catch (const std::exception&) {
  }
  if (wait_bounded(pid, 10)) return true;
  ::kill(pid, SIGKILL);
  ::waitpid(pid, nullptr, 0);
  return false;
}

// ---- stage process -------------------------------------------------------

int stage_main(int argc, char** argv) {
  ::alarm(90);
  Stage::Config config;
  config.name = arg_value(argc, argv, "--name");
  config.recv_endpoint = arg_value(argc, argv, "--recv");
  config.coordinator_endpoint = arg_value(argc, argv, "--coord");
  config.abort_endpoint = arg_value(argc, argv, "--abort");
  config.gpu_id = std::stoi(arg_value(argc, argv, "--device"));
  const std::string slot_mb = arg_value(argc, argv, "--slot-mb");
  if (!slot_mb.empty()) {
    config.comm_config["slot_size_mb"] = Value(std::stoi(slot_mb));
  }
  // Only the sender needs a peer placement map. The receiver follows the
  // transport recorded in DataRef after the sender's P2P/fallback decision.
  if (config.name == "A") config.stage_gpu_ids["B"] = {1};
  config.peer_endpoints["A"] = arg_value(argc, argv, "--peer-a");
  config.peer_endpoints["B"] = arg_value(argc, argv, "--peer-b");
  const std::string next_name = arg_value(argc, argv, "--next-name");
  if (!next_name.empty()) {
    config.next_stages[next_name] = arg_value(argc, argv, "--next-endpoint");
  }
  if (config.name.empty() || config.recv_endpoint.empty()) return 2;

  Stage stage(config, [](const Value& in) { return in; });
  stage.run();  // binds, starts workers, and blocks until shutdown
  return 0;
}

// ---- coordinator process -------------------------------------------------

struct StagePair {
  pid_t a = -1;
  pid_t b = -1;
};

StagePair spawn_stages(const std::string& a_recv, const std::string& b_recv,
                       const std::string& coord, const std::string& abort,
                       const std::string& slot_mb) {
  StagePair pair;
  const std::string exe = self_exe();
  if (exe.empty()) return pair;

  auto spawn = [&](const std::string& name, int device, const std::string& recv,
                   const std::string& next_name,
                   const std::string& next_endpoint) {
    const pid_t pid = ::fork();
    if (pid == 0) {
      const std::string dev = std::to_string(device);
      char* argv[] = {const_cast<char*>(exe.c_str()),
                      const_cast<char*>("--role"),
                      const_cast<char*>("stage"),
                      const_cast<char*>("--name"),
                      const_cast<char*>(name.c_str()),
                      const_cast<char*>("--device"),
                      const_cast<char*>(dev.c_str()),
                      const_cast<char*>("--recv"),
                      const_cast<char*>(recv.c_str()),
                      const_cast<char*>("--coord"),
                      const_cast<char*>(coord.c_str()),
                      const_cast<char*>("--abort"),
                      const_cast<char*>(abort.c_str()),
                      const_cast<char*>("--peer-a"),
                      const_cast<char*>(a_recv.c_str()),
                      const_cast<char*>("--peer-b"),
                      const_cast<char*>(b_recv.c_str()),
                      const_cast<char*>("--next-name"),
                      const_cast<char*>(next_name.c_str()),
                      const_cast<char*>("--next-endpoint"),
                      const_cast<char*>(next_endpoint.c_str()),
                      const_cast<char*>("--slot-mb"),
                      const_cast<char*>(slot_mb.c_str()),
                      nullptr};
      ::execv(exe.c_str(), argv);
      ::_exit(127);
    }
    return pid;
  };

  pair.a = spawn("A", 0, a_recv, "B", b_recv);
  pair.b = spawn("B", 1, b_recv, "", "");
  return pair;
}

// The router classification the stage processes must reach for this payload.
// packed_bytes selects the size-aware path the stage uses; slot_mb is the
// configured CUDA slot limit.
TransportKind classify(const Value& value, int64_t packed_bytes, int slot_mb) {
  std::set<std::string> gpu_names = {"A", "B"};
  std::map<std::string, std::vector<int>> gpu_ids = {{"A", {0}}, {"B", {1}}};
  std::map<std::string, Value> comm_config;
  comm_config["slot_size_mb"] = Value(slot_mb);
  CommRouter router("A", 0, 0, {}, gpu_names, gpu_ids, {}, comm_config);
  return router.relay_for_payload("B", value, packed_bytes).first;
}

bool run_pipeline(const std::string& tag, TransportKind expected,
                  const std::string& payload_bytes, int slot_mb = 512) {
  const std::string base =
      "/tmp/omni-cuda-pipeline-" + tag + "-" + std::to_string(getpid()) + "-";
  const std::string a_recv = "ipc://" + base + "a";
  const std::string b_recv = "ipc://" + base + "b";
  const std::string coord_ep = "ipc://" + base + "completion";
  const std::string abort_ep = "ipc://" + base + "abort";
  for (const auto& ep : {a_recv, b_recv, coord_ep, abort_ep}) remove_endpoint(ep);

  Tensor tensor;
  tensor.device = "cuda:0";
  tensor.dtype = "uint8";
  tensor.shape = {static_cast<int64_t>(payload_bytes.size())};
  tensor.bytes = payload_bytes;
  const Value value(tensor);

  // The stage measures the packed StagePayload, not the raw tensor bytes.
  StagePayload probe;
  probe.request.inputs = value;
  probe.data = value;
  const int64_t packed_bytes =
      static_cast<int64_t>(pack(probe.to_value()).size());

  // Both the parent and the stage processes must agree on the transport.
  CHECK(classify(value, packed_bytes, slot_mb) == expected);

  bool completed = false;
  {
    Coordinator coordinator(coord_ep, abort_ep);
    coordinator.register_stage("A", a_recv);
    coordinator.register_stage("B", b_recv);
    coordinator.start();
    std::thread coordinator_thread([&] { coordinator.run(); });

    // Keep these sockets connected for the whole test. Closing a freshly
    // connected ZeroMQ socket with linger=0 can discard the shutdown message.
    PushSocket a_shutdown(a_recv);
    PushSocket b_shutdown(b_recv);
    a_shutdown.connect();
    b_shutdown.connect();
    const StagePair pair = spawn_stages(a_recv, b_recv, coord_ep, abort_ep,
                                        std::to_string(slot_mb));
    const bool ready = wait_for_path(a_recv, 10) && wait_for_path(b_recv, 10);
    CHECK(ready);
    if (ready) {
      StagePayload payload;
      payload.request.inputs = value;
      payload.data = value;
      const std::string request_id = coordinator.submit(payload, "A");
      CHECK(coordinator.wait(request_id, 30.0));
      const auto info = coordinator.info(request_id);
      CHECK(info.has_value());
      if (info) {
        // Check the state first: reading result on a failed request throws.
        CHECK(info->state == "completed");
        if (info->state == "completed") {
          const Tensor& got = info->result.as_tensor();
          CHECK(got.device == "cuda:0");
          CHECK(got.bytes == payload_bytes);
          completed = got.bytes == payload_bytes;
        }
      }
    }

    coordinator.stop();
    coordinator_thread.join();
    CHECK(stop_child(pair.a, a_shutdown));
    CHECK(stop_child(pair.b, b_shutdown));
  }
  for (const auto& ep : {a_recv, b_recv, coord_ep, abort_ep}) remove_endpoint(ep);
  return completed;
}

}  // namespace

int main(int argc, char** argv) {
#if !defined(OMNIINFER_WITH_CUDA)
  std::cerr << "cuda_pipeline_test: built without OMNIINFER_WITH_CUDA, skipped\n";
  (void)argc;
  (void)argv;
  return 0;
#else
  if (!arg_value(argc, argv, "--role").empty()) {
    return stage_main(argc, argv);
  }
  ::alarm(240);

  std::string bytes(8192, '\0');
  for (size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<char>((i * 131u + 7u) & 0xffu);
  }

  std::cerr << "[1] A->B over cuda_ipc\n";
  const bool cuda_ok = run_pipeline("cuda", TransportKind::CudaIpc, bytes);
  CHECK(cuda_ok);

  std::cerr << "[2] slot boundary selection\n";
  {
    // slot_size_mb=1 is 1048576 bytes. One byte under keeps CUDA IPC; one over
    // must select SHM before put() ever allocates device memory. The tensor is
    // small; only the measured packed size decides.
    Tensor probe_tensor;
    probe_tensor.device = "cuda:0";
    probe_tensor.dtype = "uint8";
    probe_tensor.shape = {1};
    probe_tensor.bytes = "x";
    const Value probe(probe_tensor);
    const int64_t slot = int64_t{1024} * 1024;
    CHECK(classify(probe, slot - 1, 1) == TransportKind::CudaIpc);
    CHECK(classify(probe, slot + 1, 1) == TransportKind::Shm);
    CHECK(classify(probe, slot, 1) == TransportKind::CudaIpc);  // limit inclusive
  }

  std::cerr << "[3] over-limit payload completes through shm\n";
  {
    std::string big(2u << 20, '\0');
    for (size_t i = 0; i < big.size(); ++i) {
      big[i] = static_cast<char>((i * 29u + 3u) & 0xffu);
    }
    // Both stages keep GPU identities; the sender's size check must still put
    // the transfer on SHM.
    const bool big_ok =
        run_pipeline("overlimit", TransportKind::Shm, big, /*slot_mb=*/1);
    CHECK(big_ok);
  }

  std::cerr << "[4] A->B over shm with OMNIINFER_DISABLE_CUDA_IPC=1\n";
  ::setenv("OMNIINFER_DISABLE_CUDA_IPC", "1", 1);
  const bool shm_ok = run_pipeline("shm", TransportKind::Shm, bytes);
  CHECK(shm_ok);
  ::unsetenv("OMNIINFER_DISABLE_CUDA_IPC");

  std::cerr << "[5] remapped GPU does not skip the peer decision\n";
  {
    // gpu_id 0 has no peer to GPU 1, but placement_gpu_id 1 matches the target.
    // The same-GPU shortcut must not hide the remap rejection.
    std::set<std::string> gpu_names = {"A", "B"};
    std::map<std::string, std::vector<int>> gpu_ids = {{"A", {0}}, {"B", {1}}};
    CommRouter router("A", /*gpu_id=*/0, /*placement_gpu_id=*/1, {}, gpu_names,
                      gpu_ids, {}, {});
    Tensor tensor;
    tensor.device = "cuda:0";
    tensor.dtype = "uint8";
    tensor.shape = {16};
    const Value payload(tensor);
    CHECK(router.relay_for_payload("B", payload, 16).first ==
          TransportKind::Shm);
  }

  std::cerr << "[6] close rejects a target relay not built before close\n";
  {
    std::set<std::string> gpu_names = {"A", "B"};
    std::map<std::string, std::vector<int>> gpu_ids = {{"A", {0}}, {"B", {1}}};
    CommRouter router("A", 0, 0, {}, gpu_names, gpu_ids, {}, {});
    router.close();
    Tensor tensor;
    tensor.device = "cuda:0";
    tensor.dtype = "uint8";
    tensor.shape = {16};
    const Value payload(tensor);
    bool rejected = false;
    try {
      (void)router.relay_for_payload("B", payload, 16);
    } catch (const ProtocolError&) {
      rejected = true;
    }
    CHECK(rejected);
  }


  if (failures != 0) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cerr << "all checks passed\n";
  return 0;
#endif
}

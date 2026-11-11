// SPDX-License-Identifier: Apache-2.0
// End-to-end relay benchmark: put -> metadata -> get -> ack, per payload size.
//
// Runs two processes on two GPUs and covers the same path the pipeline uses.
// SHM and CUDA IPC are measured with identical rounds and payloads. The
// direct_d2h column is 1 for the CUDA rows: the receiver copies straight from
// the mapped peer pointer, the path the measured baseline selected.
//
// Output is CSV: GPU, driver and P2P facts as leading # lines, one row per
// (relay, size). Run with -Doptimize=ReleaseFast; the build kind is printed.
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "omniinfer/relay/relay.hpp"
#include "omniinfer/proto/wire.hpp"

#if defined(OMNIINFER_WITH_CUDA)
#include <cuda_runtime_api.h>
#endif

#if defined(OMNIINFER_WITH_CUDA)
#include <cuda_runtime.h>
#endif

using namespace omniinfer;

namespace {

constexpr int kWarmup = 3;
constexpr int kMeasured = 10;

std::string arg_value(int argc, char** argv, const std::string& key) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (key == argv[i]) return argv[i + 1];
  }
  return {};
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

uint64_t fnv1a(const std::string& data) {
  uint64_t h = 1469598103934665603ull;
  for (unsigned char c : data) {
    h ^= c;
    h *= 1099511628211ull;
  }
  return h;
}

std::string payload_for(size_t n) {
  std::string bytes(n, '\0');
  for (size_t i = 0; i < n; ++i) {
    bytes[i] = static_cast<char>((i * 37u + 11u) & 0xffu);
  }
  return bytes;
}

std::map<std::string, Value> relay_config(const std::string& type, int device) {
  std::map<std::string, Value> config;
  config["engine_id"] = Value("cuda-bench");
  config["device"] = Value(device);
  config["credits"] = Value(4);
  config["slot_size_mb"] = Value(512);
  (void)type;
  return config;
}

struct Child {
  pid_t pid = -1;
  int to_child = -1;
  int from_child = -1;
};

std::string self_exe() {
  char buf[4096];
  const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (n <= 0) return {};
  buf[n] = '\0';
  return buf;
}

// ---- receiver process ----------------------------------------------------

int receiver_main(int argc, char** argv) {
  ::alarm(1800);
  const std::string type = arg_value(argc, argv, "--relay");
  const int device = std::stoi(arg_value(argc, argv, "--device"));
  const int count = std::stoi(arg_value(argc, argv, "--count"));
  const int meta_fd = std::stoi(arg_value(argc, argv, "--meta-fd"));
  const int ack_fd = std::stoi(arg_value(argc, argv, "--ack-fd"));
  auto relay = create_relay(type, relay_config(type, device));

  for (int i = 0; i < count; ++i) {
    Value metadata;
    if (!recv_metadata(meta_fd, metadata)) return 1;
    Tensor dest;
    dest.device = "cpu";
    dest.dtype = "uint8";
    // The receive path sizes the destination from the DataRef length, and the
    // SHM relay requires that capacity. Do the same here.
    const Value& info = require_field(metadata, "transfer_info");
    const int64_t declared = require_field(info, "size").as_int();
    dest.bytes.assign(declared > 0 ? static_cast<size_t>(declared) : 0, '\0');
    uint8_t ok = 1;
    try {
      relay->get(metadata, dest);
      ok = 0;
    } catch (const std::exception& e) {
      std::cerr << "bench receiver: " << e.what() << "\n";
    }
    // The ack marks the end of the measured path. The checksum is computed
    // afterwards so a 256 MiB CPU hash cannot hide a copy improvement.
    if (!write_all(ack_fd, &ok, sizeof(ok))) return 1;
    const uint64_t hash = ok == 0 ? fnv1a(dest.bytes) : 0;
    if (!write_all(ack_fd, &hash, sizeof(hash))) return 1;
  }
  return 0;
}

Child spawn_receiver(const std::string& exe, const std::string& type, int device,
                     int count) {
  Child child;
  int to_child[2] = {-1, -1};
  int from_child[2] = {-1, -1};
  if (::pipe(to_child) != 0 || ::pipe(from_child) != 0) return child;
  const pid_t pid = ::fork();
  if (pid == 0) {
    ::close(to_child[1]);
    ::close(from_child[0]);
    const std::string dev = std::to_string(device);
    const std::string n = std::to_string(count);
    const std::string meta_fd = std::to_string(to_child[0]);
    const std::string ack_fd = std::to_string(from_child[1]);
    char* argv[] = {const_cast<char*>(exe.c_str()),
                    const_cast<char*>("--role"),
                    const_cast<char*>("bench"),
                    const_cast<char*>("--relay"),
                    const_cast<char*>(type.c_str()),
                    const_cast<char*>("--device"),
                    const_cast<char*>(dev.c_str()),
                    const_cast<char*>("--count"),
                    const_cast<char*>(n.c_str()),
                    const_cast<char*>("--meta-fd"),
                    const_cast<char*>(meta_fd.c_str()),
                    const_cast<char*>("--ack-fd"),
                    const_cast<char*>(ack_fd.c_str()),
                    nullptr};
    ::execv(exe.c_str(), argv);
    ::_exit(127);
  }
  ::close(to_child[0]);
  ::close(from_child[1]);
  if (pid < 0) return child;
  child.pid = pid;
  child.to_child = to_child[1];
  child.from_child = from_child[0];
  return child;
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

void print_hardware(bool direct_available) {
#if defined(OMNIINFER_WITH_CUDA)
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
    std::cout << "# gpu_count=" << devices << " (two devices required)\n";
    return;
  }
  cudaDeviceProp prop0{};
  cudaDeviceProp prop1{};
  (void)cudaGetDeviceProperties(&prop0, 0);
  (void)cudaGetDeviceProperties(&prop1, 1);
  std::cout << "# gpu0=" << prop0.name << " gpu1=" << prop1.name << "\n";
  std::cout << "# pci_bus_0=" << prop0.pciBusID << " pci_bus_1=" << prop1.pciBusID
            << "\n";
  int driver = 0;
  int runtime = 0;
  (void)cudaDriverGetVersion(&driver);
  (void)cudaRuntimeGetVersion(&runtime);
  std::cout << "# driver=" << driver / 1000 << "." << (driver % 1000) / 10
            << " runtime=" << runtime / 1000 << "." << (runtime % 1000) / 10
            << "\n";
  int p2p_01 = 0;
  int p2p_10 = 0;
  (void)cudaDeviceCanAccessPeer(&p2p_01, 0, 1);
  (void)cudaDeviceCanAccessPeer(&p2p_10, 1, 0);
  std::cout << "# p2p_0_to_1=" << p2p_01 << " p2p_1_to_0=" << p2p_10 << "\n";
  (void)direct_available;
#else
  (void)direct_available;
  std::cout << "# built without OMNIINFER_WITH_CUDA\n";
#endif
}

}  // namespace

int main(int argc, char** argv) {
#if !defined(OMNIINFER_WITH_CUDA)
  (void)argc;
  (void)argv;
  std::cerr << "cuda_bench: built without OMNIINFER_WITH_CUDA\n";
  return 2;
#else
  ::signal(SIGPIPE, SIG_IGN);
  if (arg_value(argc, argv, "--role") == "bench") {
    return receiver_main(argc, argv);
  }
  ::alarm(3600);

  std::vector<size_t> sizes = {4096, 1u << 20, 16u << 20, 64u << 20,
                               256u << 20};
  std::string exe = self_exe();
  if (exe.empty()) {
    std::cerr << "cuda_bench: cannot resolve own path\n";
    return 2;
  }

#ifdef NDEBUG
  std::cout << "# build=release\n";
#else
  std::cout << "# build=debug\n";
#endif
  std::cout << "# warmup=" << kWarmup << " measured=" << kMeasured << "\n";
  const char* git = std::getenv("OMNIINFER_BENCH_GIT");
  const char* cmd = std::getenv("OMNIINFER_BENCH_CMD");
  std::cout << "# git=" << (git ? git : "unknown") << "\n";
  std::cout << "# cmd=" << (cmd ? cmd : "cuda_bench") << "\n";
  print_hardware(true);
  std::cout << "relay,direct_d2h,size_bytes,median_us,p95_us,mib_per_s,hash_ok\n";

  struct Variant {
    const char* relay;
    bool direct;  // label only; the CUDA receiver always copies directly
  };
  const Variant variants[] = {{"shm", false}, {"cuda_ipc", true}};

  bool all_ok = true;
  for (const Variant& variant : variants) {
    const int count =
        static_cast<int>(sizes.size()) * (kWarmup + kMeasured);
    Child child = spawn_receiver(exe, variant.relay, 1, count);
    if (child.pid <= 0) {
      std::cerr << "cuda_bench: spawn failed for " << variant.relay << "\n";
      return 1;
    }
    auto relay = create_relay(variant.relay, relay_config(variant.relay, 0));

    for (size_t size : sizes) {
      std::vector<double> samples;
      bool hash_ok = true;
      for (int round = 0; round < kWarmup + kMeasured; ++round) {
        const std::string bytes = payload_for(size);
        Tensor tensor;
        tensor.device = "cpu";
        tensor.dtype = "uint8";
        tensor.shape = {static_cast<int64_t>(size)};
        tensor.bytes = bytes;

        const auto start = std::chrono::steady_clock::now();
        auto op = relay->put(tensor, "bench");
        if (!send_metadata(child.to_child, op->metadata())) {
          std::cerr << "cuda_bench: metadata write failed\n";
          return 1;
        }
        uint8_t ok = 1;
        uint64_t hash = 0;
        if (!read_all(child.from_child, &ok, sizeof(ok))) {
          std::cerr << "cuda_bench: ack read failed\n";
          return 1;
        }
        op->mark_receiver_done();
        const auto stop = std::chrono::steady_clock::now();
        // The checksum arrives after the ack; it stays outside the timed path
        // but every round is still verified.
        if (!read_all(child.from_child, &hash, sizeof(hash))) {
          std::cerr << "cuda_bench: hash read failed\n";
          return 1;
        }
        if (ok != 0) {
          std::cerr << "cuda_bench: receiver rejected a block\n";
          return 1;
        }
        if (hash != fnv1a(bytes)) hash_ok = false;
        if (round >= kWarmup) {
          samples.push_back(
              std::chrono::duration<double, std::micro>(stop - start).count());
        }
      }
      std::sort(samples.begin(), samples.end());
      const double median = samples[samples.size() / 2];
      const size_t p95_index =
          std::min(samples.size() - 1,
                   static_cast<size_t>(samples.size() * 0.95));
      const double p95 = samples[p95_index];
      const double mib_per_s =
          median > 0 ? (static_cast<double>(size) / (1024.0 * 1024.0)) /
                           (median / 1e6)
                     : 0.0;
      std::cout << variant.relay << "," << (variant.direct ? 1 : 0) << ","
                << size << "," << median << "," << p95 << "," << mib_per_s
                << "," << (hash_ok ? 1 : 0) << "\n";
      std::cout.flush();
      all_ok = all_ok && hash_ok;
    }

    ::close(child.to_child);
    ::close(child.from_child);
    all_ok = wait_bounded(child.pid, 60) && all_ok;
  }
  std::cout << "# done hash_ok=" << (all_ok ? 1 : 0) << "\n";
  return all_ok ? 0 : 1;
#endif
}

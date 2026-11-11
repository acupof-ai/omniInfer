// SPDX-License-Identifier: Apache-2.0
// Launcher and stage host for the C++ pipeline runtime.
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "omniinfer/pipeline/pipeline.hpp"
#include "omniinfer/serve/openai_api.hpp"

namespace {

using namespace omniinfer;

std::atomic<bool> g_running{true};

void on_signal(int) { g_running = false; }

using Flags = std::multimap<std::string, std::string>;

Flags parse_flags(int argc, char** argv, int from) {
  Flags flags;
  for (int i = from; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind("--", 0) != 0) continue;
    std::string key = arg.substr(2);
    std::string value = "true";
    if (const size_t eq = key.find('='); eq != std::string::npos) {
      value = key.substr(eq + 1);
      key = key.substr(0, eq);
    } else if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
      value = argv[++i];
    }
    flags.emplace(std::move(key), std::move(value));
  }
  return flags;
}

std::string flag(const Flags& flags, const char* key,
                 const std::string& fallback = "") {
  auto it = flags.find(key);
  return it == flags.end() ? fallback : it->second;
}

int parse_port(const Flags& flags) {
  const std::string text = flag(flags, "port", "30000");
  size_t used = 0;
  int port;
  try {
    port = std::stoi(text, &used);
  } catch (const std::exception&) {
    throw ProtocolError("invalid --port: " + text);
  }
  if (used != text.size() || port < 0 || port > 65535) {
    throw ProtocolError("invalid --port: " + text);
  }
  return port;
}

int parse_gpu_id(const std::string& text, const std::string& option) {
  size_t used = 0;
  int id;
  try {
    id = std::stoi(text, &used);
  } catch (const std::exception&) {
    throw ProtocolError("invalid " + option + ": " + text);
  }
  if (used != text.size() || id < 0) {
    throw ProtocolError("invalid " + option + ": " + text);
  }
  return id;
}

// ZMQ binds an ipc path but never creates its parent directory.
void ensure_ipc_parent(const std::string& endpoint) {
  const std::string prefix = "ipc://";
  if (endpoint.rfind(prefix, 0) != 0) return;
  const size_t slash = endpoint.rfind('/');
  if (slash == std::string::npos || slash <= prefix.size()) return;
  std::error_code ignored;
  std::filesystem::create_directories(endpoint.substr(prefix.size(),
                                                      slash - prefix.size()),
                                      ignored);
}

// Reference compute function: echoes the last user message. Replace it with a
// model call; the stage protocol does not change.
Value echo_compute(const Value& payload) {
  // Accept the compute input directly, or a request carrying it under "inputs".
  const Value* inputs = payload.find("inputs");
  const Value& source = inputs != nullptr ? *inputs : payload;
  std::string text;
  if (const Value* messages = source.find("messages"); messages != nullptr) {
    for (const auto& message : messages->as_array()) {
      if (const Value* content = message.find("content");
          content != nullptr && content->is_str()) {
        text = content->as_str();
      }
    }
  } else if (const Value* user = source.find("user");
             user != nullptr && user->is_str()) {
    text = user->as_str();
  }
  Value::Map out;
  out["text"] = Value("echo: " + text);
  return Value(std::move(out));
}

int usage() {
  std::cerr
      << "usage:\n"
      << "  omni-infer serve --port 30000 [--host 0.0.0.0] [--model omni] "
         "[--api-key K]\n"
      << "      [--entry name --entry-endpoint ipc:///tmp/x/in] "
         "[--completion ipc:///tmp/x/done] [--abort ipc:///tmp/x/abort]\n"
      << "      [--demo]  run one in-process echo stage (development only)\n"
      << "  omni-infer stage --name N --recv EP --coordinator EP --abort EP "
         "[--next name=EP ...] [--peer name=EP ...]\n"
         "      [--gpu-id N --peer-gpu name=N ...]  staged CUDA IPC\n";
  return 2;
}

int run_demo(const Flags& flags) {
  ApiConfig api_config;
  api_config.host = flag(flags, "host", api_config.host);
  api_config.port = parse_port(flags);
  api_config.model = flag(flags, "model", api_config.model);
  api_config.api_key = flag(flags, "api-key", "");

  const std::string base = flag(flags, "ipc-dir", "/tmp/omni-demo");
  // A killed server leaves its ipc sockets behind and the next bind fails.
  std::error_code ignored;
  for (const auto* name : {"entry", "completion", "abort"}) {
    std::filesystem::remove(base + "/" + name, ignored);
  }
  const std::string entry_ep = "ipc://" + base + "/entry";
  const std::string completion_ep = "ipc://" + base + "/completion";
  const std::string abort_ep = "ipc://" + base + "/abort";
  ensure_ipc_parent(entry_ep);
  ensure_ipc_parent(completion_ep);
  ensure_ipc_parent(abort_ep);

  Coordinator coordinator(completion_ep, abort_ep);
  OpenAIApi api(coordinator, api_config);
  if (!api.bind()) {
    std::cerr << "failed to bind " << api_config.host << ":" << api_config.port << "\n";
    return 1;
  }
  coordinator.register_stage("entry", entry_ep);
  coordinator.start();
  std::thread coordinator_thread([&] { coordinator.run(); });

  Stage::Config stage_config;
  stage_config.name = "entry";
  stage_config.recv_endpoint = entry_ep;
  stage_config.coordinator_endpoint = completion_ep;
  stage_config.abort_endpoint = abort_ep;
  stage_config.peer_endpoints["entry"] = entry_ep;
  Stage stage(stage_config, echo_compute);
  stage.start();
  std::thread stage_thread([&stage] { stage.run(); });

  std::atomic<bool> server_failed{false};
  std::thread server_thread([&] {
    if (!api.listen() && g_running.exchange(false)) {
      std::cerr << "HTTP listen failed\n";
      server_failed = true;
    }
  });
  std::cout << "omni-infer demo serving on http://" << api_config.host << ":"
            << api.bound_port() << " (model " << api_config.model << ")" << std::endl;
  while (g_running) std::this_thread::sleep_for(std::chrono::milliseconds(100));

  api.stop();
  coordinator.stop();
  stage.stop();
  server_thread.join();
  stage_thread.join();
  coordinator_thread.join();
  return server_failed ? 1 : 0;
}

int run_serve(const Flags& flags) {
  if (flags.count("demo")) return run_demo(flags);

  const std::string entry_name = flag(flags, "entry", "entry");
  const std::string entry_ep = flag(flags, "entry-endpoint");
  if (entry_ep.empty()) {
    std::cerr << "--entry-endpoint is required (or pass --demo)\n";
    return usage();
  }
  const std::string completion_ep =
      flag(flags, "completion", "ipc:///tmp/omni/completion");
  const std::string abort_ep = flag(flags, "abort", "ipc:///tmp/omni/abort");
  ensure_ipc_parent(entry_ep);
  ensure_ipc_parent(completion_ep);
  ensure_ipc_parent(abort_ep);

  ApiConfig api_config;
  api_config.host = flag(flags, "host", api_config.host);
  api_config.port = parse_port(flags);
  api_config.model = flag(flags, "model", api_config.model);
  api_config.entry_stage = entry_name;
  api_config.api_key = flag(flags, "api-key", "");
  Coordinator coordinator(completion_ep, abort_ep);
  OpenAIApi api(coordinator, api_config);
  if (!api.bind()) {
    std::cerr << "failed to bind " << api_config.host << ":" << api_config.port << "\n";
    return 1;
  }
  coordinator.register_stage(entry_name, entry_ep);
  coordinator.start();
  std::thread coordinator_thread([&] { coordinator.run(); });

  std::atomic<bool> server_failed{false};
  std::thread server_thread([&] {
    if (!api.listen()) {
      server_failed = g_running.exchange(false);
    }
  });

  std::cout << "omni-infer serving on " << api_config.host << ":" << api.bound_port()
            << std::endl;
  while (g_running) std::this_thread::sleep_for(std::chrono::milliseconds(100));

  api.stop();
  coordinator.stop();
  server_thread.join();
  coordinator_thread.join();
  return server_failed ? 1 : 0;
}

int run_stage(const Flags& flags) {
  Stage::Config config;
  config.name = flag(flags, "name");
  config.recv_endpoint = flag(flags, "recv");
  config.coordinator_endpoint = flag(flags, "coordinator");
  config.abort_endpoint = flag(flags, "abort");
  if (flags.count("gpu-id")) {
    config.gpu_id = parse_gpu_id(flag(flags, "gpu-id"), "--gpu-id");
  }
  if (config.name.empty() || config.recv_endpoint.empty() ||
      config.coordinator_endpoint.empty() || config.abort_endpoint.empty()) {
    return usage();
  }
  config.peer_endpoints[config.name] = config.recv_endpoint;
  ensure_ipc_parent(config.recv_endpoint);
  ensure_ipc_parent(config.coordinator_endpoint);
  ensure_ipc_parent(config.abort_endpoint);
  for (const auto& [key, value] : flags) {
    if (key != "next" && key != "peer") continue;
    const size_t eq = value.find('=');
    if (eq == std::string::npos) continue;
    const std::string target = value.substr(0, eq);
    const std::string endpoint = value.substr(eq + 1);
    ensure_ipc_parent(endpoint);
    if (key == "next") config.next_stages[target] = endpoint;
    config.peer_endpoints[target] = endpoint;
  }
  for (const auto& [key, value] : flags) {
    if (key != "peer-gpu") continue;
    const size_t eq = value.find('=');
    if (eq == std::string::npos || eq == 0 || eq + 1 == value.size()) {
      throw ProtocolError("--peer-gpu requires name=N");
    }
    const std::string target = value.substr(0, eq);
    if (!config.peer_endpoints.count(target)) {
      throw ProtocolError("--peer-gpu target needs --next or --peer: " + target);
    }
    if (config.stage_gpu_ids.count(target)) {
      throw ProtocolError("duplicate --peer-gpu target: " + target);
    }
    config.stage_gpu_ids[target] = {
        parse_gpu_id(value.substr(eq + 1), "--peer-gpu")};
  }
  if (!config.stage_gpu_ids.empty() && !config.gpu_id) {
    throw ProtocolError("--peer-gpu requires --gpu-id");
  }
  Stage stage(std::move(config), echo_compute);
  stage.start();
  std::atomic<bool> stage_done{false};
  std::atomic<bool> stage_failed{false};
  std::thread stage_thread([&] {
    try {
      stage.run();
    } catch (const std::exception& exc) {
      std::cerr << "stage failed: " << exc.what() << "\n";
      stage_failed = true;
      g_running = false;
    }
    stage_done = true;
  });
  while (g_running && !stage_done) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  stage.stop();
  stage_thread.join();
  return stage_failed ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  if (argc < 2) return usage();
  const std::string command = argv[1];
  const auto flags = parse_flags(argc, argv, 2);
  try {
    if (command == "serve") return run_serve(flags);
    if (command == "stage") return run_stage(flags);
  } catch (const std::exception& exc) {
    std::cerr << "error: " << exc.what() << "\n";
    return 1;
  }
  return usage();
}

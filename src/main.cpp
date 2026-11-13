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
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "umminfer/pipeline/pipeline.hpp"
#include "umminfer/pipeline/unified.hpp"
#include "umminfer/serve/openai_api.hpp"
#include "umminfer/serve/realtime_api.hpp"
#include "client/cli.hpp"

namespace {

using namespace umminfer;

std::atomic<bool> g_running{true};
std::atomic<bool> g_cancelled{false};

void on_signal(int) {
  g_running = false;
  g_cancelled = true;
}

using Flags = std::multimap<std::string, std::string>;

struct FlagError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct ParsedFlags {
  Flags flags;
  bool help = false;
};

struct CommandOptions {
  std::set<std::string> values;
  std::set<std::string> booleans;
  std::set<std::string> repeated;
};

CommandOptions command_options(const std::string& command) {
  if (command == "serve") {
    return {{"port", "host", "model", "api-key", "entry", "entry-endpoint", "completion",
             "abort", "realtime-port", "ipc-dir", "upstream-kind", "upstream-base-url",
             "upstream-model", "upstream-api-key", "upstream-variant", "vlm-entry",
             "vlm-entry-endpoint"},
            {"demo"},
            {}};
  }
  // --next, --peer and --peer-gpu carry one name=value pair each and repeat.
  return {{"name", "recv", "coordinator", "abort", "next", "peer", "gpu-id", "peer-gpu"},
          {},
          {"next", "peer", "peer-gpu"}};
}

ParsedFlags parse_flags(const std::string& command, int argc, char** argv, int from) {
  const CommandOptions allowed = command_options(command);
  ParsedFlags parsed;
  std::set<std::string> seen;
  for (int i = from; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help") {
      parsed.help = true;
      continue;
    }
    if (arg.rfind("--", 0) != 0) throw FlagError("unexpected argument: " + arg);
    std::string key = arg.substr(2);
    std::string value;
    if (const size_t eq = key.find('='); eq != std::string::npos) {
      value = key.substr(eq + 1);
      key.resize(eq);
      // A boolean is a bare switch. "--demo=false" must not silently enable it.
      if (allowed.booleans.count(key)) {
        throw FlagError("--" + key + " does not take a value");
      }
      if (!allowed.values.count(key)) {
        throw FlagError("unknown option: --" + key);
      }
    } else if (allowed.booleans.count(key)) {
      value = "true";
    } else if (allowed.values.count(key)) {
      if (i + 1 >= argc || std::string(argv[i + 1]).rfind("--", 0) == 0) {
        throw FlagError("--" + key + " needs a value");
      }
      value = argv[++i];
    } else {
      throw FlagError("unknown option: --" + key);
    }
    if (!allowed.repeated.count(key) && !seen.insert(key).second) {
      throw FlagError("duplicate option: --" + key);
    }
    parsed.flags.emplace(std::move(key), std::move(value));
  }
  return parsed;
}

std::string flag(const Flags& flags, const char* key,
                 const std::string& fallback = "") {
  auto it = flags.find(key);
  return it == flags.end() ? fallback : it->second;
}

int parse_port(const Flags& flags, const char* key = "port") {
  const std::string text = flag(flags, key, "30000");
  size_t used = 0;
  int port;
  try {
    port = std::stoi(text, &used);
  } catch (const std::exception&) {
    throw ProtocolError("invalid --" + std::string(key) + ": " + text);
  }
  if (used != text.size() || port < 0 || port > 65535) {
    throw ProtocolError("invalid --" + std::string(key) + ": " + text);
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
  // The echo stage returns the whole message in one pass and never truncates it.
  out["finish_reason"] = Value("stop");
  return Value(std::move(out));
}

void print_usage(std::ostream& out) {
  out
      << "usage:\n"
      << "  umm-infer serve --port 30000 [--host 0.0.0.0] [--model omni] "
         "[--api-key K]\n"
      << "      [--entry name --entry-endpoint ipc:///tmp/x/in] "
         "[--completion ipc:///tmp/x/done] [--abort ipc:///tmp/x/abort]\n"
         "      [--vlm-entry name --vlm-entry-endpoint ipc:///tmp/x/vlm]  ASR -> VLM chat\n"
         "      [--realtime-port 30001]  enable the ASR WebSocket listener\n"
      << "      [--demo]  run one in-process echo stage (development only)\n"
      << "  umm-infer stage --name N --recv EP --coordinator EP --abort EP "
         "[--next name=EP ...] [--peer name=EP ...]\n"
         "      [--gpu-id N --peer-gpu name=N ...]  staged CUDA IPC\n"
      << "  umm-infer chat --prompt TEXT [--url URL --model M --stream]\n"
      << "      [--image-url URL ...] [--video-url URL ...] [--audio PATH]\n"
      << "      [--audio-out PATH --voice NAME]  speech output, requires --stream\n"
      << "  umm-infer transcribe --url URL --model M --file audio.wav\n"
      << "  umm-infer realtime --url URL --model M --file audio.wav [--live-result]\n"
      << "  umm-infer image generate --url URL --model M --prompt TEXT --out PATH\n"
      << "  umm-infer image edit --url URL --model M --prompt TEXT --image PATH ... --out PATH\n"
      << "  umm-infer video create|status|content|delete --url URL [--id ID --out PATH]\n"
      << "      [--variant N]  zero-based output index for video content\n"
      << "\n"
      << "serve upstream options (one upstream per process, no routing):\n"
      << "  --upstream-kind qwen|sglang --upstream-base-url URL --upstream-model M\n"
      << "  --upstream-variant omni|image|video --upstream-api-key K\n"
      << "  UMMINFER_UPSTREAM_API_KEY is used when --upstream-api-key is absent.\n";
}

int usage() {
  print_usage(std::cerr);
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
  coordinator.register_stage("entry", entry_ep, true);
  coordinator.start();
  std::thread coordinator_thread([&] { coordinator.run(); });

  Stage::Config stage_config;
  stage_config.name = "entry";
  stage_config.recv_endpoint = entry_ep;
  stage_config.coordinator_endpoint = completion_ep;
  stage_config.abort_endpoint = abort_ep;
  stage_config.announce_ready = true;
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
  std::cout << "umm-infer demo serving on http://" << api_config.host << ":"
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

  ApiConfig api_config;
  api_config.host = flag(flags, "host", api_config.host);
  api_config.port = parse_port(flags);
  api_config.model = flag(flags, "model", api_config.model);
  api_config.api_key = flag(flags, "api-key", "");
  // Fixed upstream: one process serves one upstream, one model and one
  // variant. The upstream key is read from the environment when the flag is
  // absent, so it does not have to appear in the process arguments.
  api_config.upstream_kind = flag(flags, "upstream-kind");
  api_config.upstream_base_url = flag(flags, "upstream-base-url");
  api_config.upstream_model = flag(flags, "upstream-model");
  api_config.upstream_variant = flag(flags, "upstream-variant");
  api_config.upstream_api_key = flag(flags, "upstream-api-key");
  if (api_config.upstream_api_key.empty()) {
    const char* key = std::getenv("UMMINFER_UPSTREAM_API_KEY");
    if (key != nullptr) api_config.upstream_api_key = key;
  }
  const std::string upstream_error = validate_upstream_config(api_config);
  if (!upstream_error.empty()) {
    std::cerr << upstream_error << "\n";
    return usage();
  }
  const bool fixed_upstream = !api_config.upstream_kind.empty();

  const std::string entry_name = flag(flags, "entry", "entry");
  const std::string entry_ep = flag(flags, "entry-endpoint");
  const std::string vlm_name = flag(flags, "vlm-entry", "vlm");
  const std::string vlm_ep = flag(flags, "vlm-entry-endpoint");
  const bool unified = !vlm_ep.empty();
  if (flags.count("vlm-entry") && !unified) {
    std::cerr << "--vlm-entry requires --vlm-entry-endpoint\n";
    return usage();
  }
  if (unified && fixed_upstream) {
    std::cerr << "--vlm-entry-endpoint cannot be combined with a fixed upstream\n";
    return usage();
  }
  if (unified && (vlm_name == entry_name || vlm_ep == entry_ep)) {
    std::cerr << "unified ASR and VLM stages must have distinct names and endpoints\n";
    return usage();
  }
  // A fixed-upstream process serves only the upstream. Without an entry
  // endpoint it runs no local stage, so nothing is registered and /ready can
  // answer at once.
  if (entry_ep.empty() && !fixed_upstream) {
    std::cerr << "--entry-endpoint is required (or pass --demo)\n";
    return usage();
  }
  if (entry_ep.empty() && flags.count("realtime-port")) {
    std::cerr << "--realtime-port requires --entry-endpoint: realtime needs a local stage\n";
    return usage();
  }
  const std::string completion_ep =
      flag(flags, "completion", "ipc:///tmp/omni/completion");
  const std::string abort_ep = flag(flags, "abort", "ipc:///tmp/omni/abort");
  ensure_ipc_parent(completion_ep);
  ensure_ipc_parent(abort_ep);

  api_config.entry_stage = entry_name;
  // True only when this process waits for an entry stage in another process.
  api_config.external_entry_stage = !entry_ep.empty();
  Coordinator coordinator(completion_ep, abort_ep);
  OpenAIApi api(coordinator, api_config);
  if (!api.bind()) {
    std::cerr << "failed to bind " << api_config.host << ":" << api_config.port << "\n";
    return 1;
  }
  std::unique_ptr<RealtimeApi> realtime;
  if (flags.count("realtime-port")) {
    ApiConfig realtime_config = api_config;
    realtime_config.port = parse_port(flags, "realtime-port");
    realtime = std::make_unique<RealtimeApi>(coordinator, realtime_config);
    if (!realtime->bind()) {
      std::cerr << "failed to bind realtime " << realtime_config.host << ":"
                << realtime_config.port << "\n";
      return 1;
    }
  }
  if (!entry_ep.empty()) {
    ensure_ipc_parent(entry_ep);
    coordinator.register_stage(entry_name, entry_ep, true);
  }
  if (unified) {
    ensure_ipc_parent(vlm_ep);
    coordinator.register_stage(vlm_name, vlm_ep, true);
  }
  std::unique_ptr<UnifiedEngine> unified_engine;
  if (unified) {
    UnifiedConfig unified_config;
    unified_config.asr_stage = entry_name;
    unified_config.vlm_stage = vlm_name;
    unified_engine = std::make_unique<UnifiedEngine>(coordinator, unified_config);
    api.set_unified_engine(*unified_engine, vlm_name);
  }
  coordinator.start();
  std::thread coordinator_thread([&] { coordinator.run(); });

  std::atomic<bool> server_failed{false};
  std::thread server_thread([&] {
    if (!api.listen()) {
      server_failed = g_running.exchange(false);
    }
  });
  std::thread realtime_thread;
  if (realtime) {
    realtime_thread = std::thread([&] {
      if (!realtime->listen() && g_running.exchange(false)) {
        std::cerr << "realtime listen failed\n";
        server_failed = true;
      }
    });
  }

  std::cout << "umm-infer serving on " << api_config.host << ":" << api.bound_port()
            << std::endl;
  if (realtime) {
    std::cout << "umm-infer realtime serving on " << api_config.host << ":"
              << realtime->bound_port() << std::endl;
  }
  while (g_running) std::this_thread::sleep_for(std::chrono::milliseconds(100));

  if (realtime) realtime->stop();
  api.stop();
  coordinator.stop();
  server_thread.join();
  if (realtime_thread.joinable()) realtime_thread.join();
  coordinator_thread.join();
  return server_failed ? 1 : 0;
}

int run_stage(const Flags& flags) {
  Stage::Config config;
  config.name = flag(flags, "name");
  config.recv_endpoint = flag(flags, "recv");
  config.coordinator_endpoint = flag(flags, "coordinator");
  config.abort_endpoint = flag(flags, "abort");
  config.announce_ready = true;
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
  try {
    if (command == "serve" || command == "stage") {
      const ParsedFlags options = parse_flags(command, argc, argv, 2);
      if (options.help) {
        print_usage(std::cout);
        return 0;
      }
      if (command == "serve") return run_serve(options.flags);
      return run_stage(options.flags);
    }
    if (command == "chat" || command == "transcribe" || command == "realtime" ||
        command == "image" || command == "video") {
      return umminfer::client::run_cli(command, argc, argv, &g_cancelled);
    }
  } catch (const FlagError& exc) {
    std::cerr << "error: " << exc.what() << "\n";
    return usage();
  } catch (const std::exception& exc) {
    std::cerr << "error: " << exc.what() << "\n";
    return 1;
  }
  return usage();
}

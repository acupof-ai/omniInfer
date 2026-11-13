// SPDX-License-Identifier: Apache-2.0
#include "cli.hpp"

#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "http.hpp"
#include "umminfer/client/client.hpp"

namespace umminfer::client {
namespace {

struct UsageError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

class Options {
 public:
  Options(int argc, char** argv, int start, std::set<std::string> values,
          std::set<std::string> booleans, std::set<std::string> repeated = {})
      : values_allowed_(std::move(values)),
        booleans_allowed_(std::move(booleans)),
        repeated_(std::move(repeated)) {
    for (int i = start; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--help") {
        help = true;
        continue;
      }
      if (arg.rfind("--", 0) != 0) throw UsageError("unexpected argument: " + arg);
      std::string key = arg.substr(2);
      std::string value;
      if (const size_t eq = key.find('='); eq != std::string::npos) {
        value = key.substr(eq + 1);
        key.resize(eq);
        if (!values_allowed_.count(key) && !repeated_.count(key)) {
          throw UsageError("unknown option: --" + key);
        }
      } else if (booleans_allowed_.count(key)) {
        if (!repeated_.count(key) && flags_.count(key)) {
          throw UsageError("duplicate option: --" + key);
        }
        flags_.emplace(key, "true");
        continue;
      } else if (values_allowed_.count(key) || repeated_.count(key)) {
        if (++i >= argc) throw UsageError("--" + key + " needs a value");
        value = argv[i];
      } else {
        throw UsageError("unknown option: --" + key);
      }
      if (!repeated_.count(key) && flags_.count(key)) {
        throw UsageError("duplicate option: --" + key);
      }
      flags_.emplace(key, value);
    }
  }

  bool has(const std::string& key) const { return flags_.count(key) != 0; }
  std::string get(const std::string& key, const std::string& fallback = {}) const {
    const auto it = flags_.find(key);
    return it == flags_.end() ? fallback : it->second;
  }
  std::vector<std::string> all(const std::string& key) const {
    std::vector<std::string> out;
    const auto range = flags_.equal_range(key);
    for (auto it = range.first; it != range.second; ++it) out.push_back(it->second);
    return out;
  }

  bool help = false;

 private:
  std::set<std::string> values_allowed_;
  std::set<std::string> booleans_allowed_;
  std::set<std::string> repeated_;
  std::multimap<std::string, std::string> flags_;
};

void usage(const std::string& command, std::ostream& out) {
  if (command == "chat") {
    out << "usage: umm-infer chat --prompt TEXT [--url https://api.openai.com/v1] "
           "[--model M] [--system TEXT] [--stream] [--max-tokens N] [--temperature X] "
           "[--top-p X] [--timeout-ms N] [--first-token-ms N] [--stall-ms N] "
           "[--max-attempts N]\n"
           "       [--image-url URL ...] [--video-url URL ...] [--audio PATH]\n"
           "       [--audio-out PATH --voice NAME]  speech output, requires --stream\n";
  } else if (command == "transcribe") {
    out << "usage: umm-infer transcribe --url http://127.0.0.1:8000 --model M --file PATH "
           "[--format wav|pcm] [--language L] [--prompt TEXT] [--max-new-tokens N] "
           "[--timeout-ms N] [--stall-ms N]\n";
  } else if (command == "realtime") {
    out << "usage: umm-infer realtime --url http://127.0.0.1:8000 --model M --file PATH "
           "[--format wav|pcm] [--language L] [--prompt TEXT] [--live-result] "
           "[--chunk-ms N] [--final-timeout-ms N] [--final-url URL] [--stall-ms N] "
           "[--ping-interval-ms N]\n";
  } else if (command == "image") {
    out << "usage: umm-infer image generate --url URL --model M --prompt TEXT --out PATH\n"
           "           [--n N] [--size WxH] [--field NAME=VALUE ...] [--timeout-ms N]\n"
           "       umm-infer image edit --url URL --model M --prompt TEXT --out PATH\n"
           "           --image PATH [--image PATH ...] [--field NAME=VALUE ...]\n"
           "           [--timeout-ms N]\n"
           "       image parts are sent in command-line order. A field value that is\n"
           "       valid JSON is sent as JSON, otherwise as a string.\n";
  } else if (command == "video") {
    out << "usage: umm-infer video create --url URL --model M --prompt TEXT\n"
           "           [--field NAME=VALUE ...] [--timeout-ms N]\n"
           "       umm-infer video status --url URL --id ID [--timeout-ms N]\n"
           "       umm-infer video content --url URL --id ID --out PATH [--variant N]\n"
           "           [--timeout-ms N]  --variant selects a zero-based output index\n"
           "       umm-infer video delete --url URL --id ID [--timeout-ms N]\n";
  }
}

int64_t integer(const Options& options, const std::string& key, int64_t fallback,
                int64_t minimum, int64_t maximum) {
  if (!options.has(key)) return fallback;
  const std::string text = options.get(key);
  int64_t value = 0;
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || value < minimum ||
      value > maximum) {
    throw UsageError("invalid --" + key + ": " + text);
  }
  return value;
}

double number(const Options& options, const std::string& key, double fallback,
              double minimum, double maximum) {
  if (!options.has(key)) return fallback;
  const std::string text = options.get(key);
  // "--temperature=" must not fall through as 0.0.
  if (text.empty()) throw UsageError("--" + key + " needs a value");
  char* end = nullptr;
  const double value = std::strtod(text.c_str(), &end);
  if (end != text.c_str() + text.size() || !std::isfinite(value) || value < minimum ||
      value > maximum) {
    throw UsageError("invalid --" + key + ": " + text);
  }
  return value;
}

Milliseconds milliseconds(const Options& options, const std::string& key, int64_t fallback) {
  return Milliseconds{integer(options, key, fallback, 0, 86400000)};
}

std::string environment(const char* name) {
  const char* value = std::getenv(name);
  return value ? value : "";
}

std::string api_key_from_environment() {
  std::string key = environment("UMMINFER_API_KEY");
  if (key.empty()) key = environment("OPENAI_API_KEY");
  return key;
}

std::string read_stream(std::istream& input) {
  std::string out;
  char buffer[16384];
  while (input) {
    input.read(buffer, sizeof(buffer));
    const auto count = input.gcount();
    if (count > 0 && static_cast<size_t>(count) > (1u << 30) -
          std::min<size_t>(out.size(), 1u << 30)) {
      throw Error(ErrorCode::kBadRequest, "input exceeds 1 GiB");
    }
    out.append(buffer, static_cast<size_t>(count));
  }
  return out;
}

std::string read_file(const std::string& path) {
  if (path.empty()) throw UsageError("--file is required");
  if (path == "-") return read_stream(std::cin);
  std::error_code error;
  const auto size = std::filesystem::file_size(path, error);
  if (error || size > 1u << 30) throw Error(ErrorCode::kBadRequest, "audio file is missing or too large");
  std::ifstream input(path, std::ios::binary);
  if (!input) throw Error(ErrorCode::kBadRequest, "cannot open audio file");
  std::string out(size, '\0');
  if (size > 0) input.read(out.data(), static_cast<std::streamsize>(size));
  if (!input) throw Error(ErrorCode::kBadRequest, "cannot read complete audio file");
  return out;
}

std::string read_limited(const std::string& path, size_t limit, const char* label) {
  if (path.empty()) throw UsageError(std::string("--") + label + " path is required");
  std::error_code error;
  const auto size = std::filesystem::file_size(path, error);
  if (error || size > limit) {
    throw Error(ErrorCode::kBadRequest, std::string(label) + " is missing or too large");
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) throw Error(ErrorCode::kBadRequest, std::string("cannot open ") + label);
  std::string out(size, '\0');
  if (size > 0) input.read(out.data(), static_cast<std::streamsize>(size));
  if (!input) throw Error(ErrorCode::kBadRequest, std::string("cannot read complete ") + label);
  return out;
}

// Every output target is checked before the request is sent. A billable
// generation must not be discarded because the local path is unusable.
void check_output(const std::string& path) {
  std::error_code status;
  if (std::filesystem::exists(path, status)) {
    throw Error(ErrorCode::kBadRequest, "output file already exists: " + path);
  }
  if (status) {
    throw Error(ErrorCode::kBadRequest, "cannot inspect output path: " + status.message());
  }
}

// The first artifact uses --out unchanged. Each later one inserts -<index>
// before the extension, so the set is deterministic for a given command line.
std::vector<std::string> artifact_paths(const std::string& out, size_t count) {
  const size_t slash = out.find_last_of('/');
  const size_t dot = out.find_last_of('.');
  const bool has_extension = dot != std::string::npos &&
                             (slash == std::string::npos || dot > slash) && dot + 1 < out.size();
  const std::string stem = has_extension ? out.substr(0, dot) : out;
  const std::string extension = has_extension ? out.substr(dot) : std::string{};
  std::vector<std::string> paths;
  paths.reserve(count);
  for (size_t index = 0; index < count; ++index) {
    if (index == 0) {
      paths.push_back(out);
    } else {
      paths.push_back(stem + "-" + std::to_string(index + 1) + extension);
    }
  }
  return paths;
}

// Outputs go through PartWriter: the final path appears atomically and an
// existing file is never replaced. All parts are written first and published
// after, so a write failure leaves no artifact behind. A publish failure
// removes the ones this call already published.
void write_outputs(const std::vector<std::string>& paths,
                   const std::vector<std::string>& blobs) {
  if (paths.size() != blobs.size()) {
    throw Error(ErrorCode::kBadRequest, "output count does not match the result");
  }
  std::set<std::string> unique(paths.begin(), paths.end());
  if (unique.size() != paths.size()) {
    throw Error(ErrorCode::kBadRequest, "output paths collide");
  }
  for (const auto& path : paths) check_output(path);
  std::vector<std::unique_ptr<detail::PartWriter>> writers;
  try {
    for (const auto& path : paths) {
      writers.push_back(std::make_unique<detail::PartWriter>(path));
    }
    for (size_t index = 0; index < paths.size(); ++index) {
      writers[index]->write(blobs[index]);
    }
  } catch (const std::exception& error) {
    throw Error(ErrorCode::kBadRequest, std::string("cannot write output: ") + error.what());
  }
  std::vector<std::string> published;
  try {
    for (size_t index = 0; index < paths.size(); ++index) {
      writers[index]->publish();
      published.push_back(paths[index]);
    }
  } catch (const std::exception& error) {
    std::error_code ignored;
    for (const auto& path : published) std::filesystem::remove(path, ignored);
    throw Error(ErrorCode::kBadRequest, std::string("cannot publish output: ") + error.what());
  }
  for (const auto& path : paths) std::cout << path << '\n';
}

void write_output(const std::string& path, std::string_view bytes) {
  if (path.empty()) throw UsageError("--out is required");
  check_output(path);
  try {
    detail::PartWriter writer(path);
    writer.write(bytes);
    writer.publish();
  } catch (const std::exception& error) {
    throw Error(ErrorCode::kBadRequest, std::string("cannot write output: ") + error.what());
  }
}

std::string audio_pcm(const Options& options) {
  std::string bytes = read_file(options.get("file"));
  std::string format = options.get("format");
  if (format.empty()) {
    format = options.get("file").size() >= 4 &&
                     options.get("file").substr(options.get("file").size() - 4) == ".wav"
                 ? "wav"
                 : "pcm";
  }
  if (format == "wav") return pcm16_from_wav(bytes);
  if (format != "pcm") throw UsageError("--format must be wav or pcm");
  if (bytes.size() % 2 != 0) throw Error(ErrorCode::kBadRequest, "PCM16 audio has an odd byte count");
  return bytes;
}

std::string extension_of(const std::string& path) {
  const size_t dot = path.find_last_of('.');
  if (dot == std::string::npos) return {};
  std::string out = path.substr(dot + 1);
  for (char& c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return out;
}

std::string basename_of(const std::string& path) {
  const size_t slash = path.find_last_of('/');
  return slash == std::string::npos ? path : path.substr(slash + 1);
}

std::string image_content_type(const std::string& path) {
  const std::string extension = extension_of(path);
  if (extension == "png") return "image/png";
  if (extension == "jpg" || extension == "jpeg") return "image/jpeg";
  if (extension == "webp") return "image/webp";
  return "application/octet-stream";
}

std::string input_audio_format(const std::string& path) {
  const std::string extension = extension_of(path);
  if (extension == "wav" || extension == "mp3") return extension;
  throw UsageError("--audio must be a .wav or .mp3 file");
}

nlohmann::json field_value(const std::string& text) {
  const nlohmann::json parsed = nlohmann::json::parse(text, nullptr, false);
  // A valid JSON value is sent with its decoded meaning. That includes a quoted
  // string: --field task="t2va" is the string t2va, not the source text.
  if (!parsed.is_discarded()) return parsed;
  return nlohmann::json(text);
}

void add_fields(const Options& options, nlohmann::json& body) {
  for (const auto& field : options.all("field")) {
    const size_t eq = field.find('=');
    if (eq == std::string::npos || eq == 0) {
      throw UsageError("--field requires NAME=VALUE: " + field);
    }
    body[field.substr(0, eq)] = field_value(field.substr(eq + 1));
  }
}

OpenAIConfig media_config(const Options& options) {
  OpenAIConfig config;
  const std::string url = options.get("url", environment("OPENAI_BASE_URL"));
  if (url.empty()) {
    throw UsageError("--url is required: pass the ummInfer serve address");
  }
  config.base_url = url;
  config.model = options.get("model", environment("OPENAI_MODEL"));
  config.api_key = api_key_from_environment();
  config.timeout = milliseconds(options, "timeout-ms", 600000);
  return config;
}

void require_model(const OpenAIConfig& config) {
  if (config.model.empty()) throw UsageError("--model is required");
}

int chat(int argc, char** argv, std::atomic<bool>* cancel) {
  Options options(argc, argv, 2,
                  {"url", "model", "prompt", "system", "max-tokens", "temperature",
                   "top-p", "timeout-ms", "first-token-ms", "stall-ms", "max-attempts",
                   "audio", "audio-out", "voice"},
                  {"stream"}, {"image-url", "video-url"});
  if (options.help) {
    usage("chat", std::cout);
    return 0;
  }
  OpenAIConfig config;
  config.base_url = options.get("url", environment("OPENAI_BASE_URL").empty()
                                               ? config.base_url
                                               : environment("OPENAI_BASE_URL"));
  config.model = options.get("model", environment("OPENAI_MODEL").empty()
                                            ? "gpt-4.1-mini"
                                            : environment("OPENAI_MODEL"));
  config.api_key = api_key_from_environment();
  config.timeout = milliseconds(options, "timeout-ms", config.timeout.count());
  config.max_attempts = static_cast<int>(integer(options, "max-attempts", 3, 1, 10));
  OpenAIClient client(config);
  ChatRequest request;
  std::string prompt = options.get("prompt");
  if (!options.has("prompt")) prompt = read_stream(std::cin);
  if (options.has("system")) request.messages.push_back({"system", options.get("system")});

  const std::vector<std::string> image_urls = options.all("image-url");
  const std::vector<std::string> video_urls = options.all("video-url");
  const bool input_audio = options.has("audio");
  if (image_urls.empty() && video_urls.empty() && !input_audio) {
    request.messages.push_back({"user", std::move(prompt)});
  } else {
    Message message;
    message.role = "user";
    if (!prompt.empty()) {
      ContentPart part;
      part.kind = ContentPart::Kind::kText;
      part.text = std::move(prompt);
      message.parts.push_back(std::move(part));
    }
    for (const auto& url : image_urls) {
      ContentPart part;
      part.kind = ContentPart::Kind::kImageUrl;
      part.url = url;
      message.parts.push_back(std::move(part));
    }
    for (const auto& url : video_urls) {
      ContentPart part;
      part.kind = ContentPart::Kind::kVideoUrl;
      part.url = url;
      message.parts.push_back(std::move(part));
    }
    if (input_audio) {
      const std::string path = options.get("audio");
      ContentPart part;
      part.kind = ContentPart::Kind::kInputAudio;
      part.audio_format = input_audio_format(path);
      part.audio = detail::base64_encode(read_limited(path, 32u << 20, "audio"));
      message.parts.push_back(std::move(part));
    }
    request.messages.push_back(std::move(message));
  }

  request.max_tokens = static_cast<int>(integer(options, "max-tokens", 0, 0, 1000000));
  if (options.has("temperature")) request.temperature = number(options, "temperature", 0, 0, 2);
  if (options.has("top-p")) request.top_p = number(options, "top-p", 0, 0, 1);
  request.first_token_timeout = milliseconds(options, "first-token-ms", 0);
  request.stall_timeout = milliseconds(options, "stall-ms", 0);
  request.stream = options.has("stream");
  request.cancel = cancel;

  const std::string audio_out = options.get("audio-out");
  if (options.has("audio-out") && audio_out.empty()) {
    throw UsageError("--audio-out requires a path");
  }
  if (!audio_out.empty()) {
    AudioOutput output;
    output.voice = options.get("voice");
    if (output.voice.empty()) throw UsageError("--audio-out requires --voice");
    if (!request.stream) throw UsageError("--audio-out requires --stream");
    // Checked before the request: the speech is billable and local failure must
    // not discard it.
    check_output(audio_out);
    request.modalities = {"text", "audio"};
    request.audio = std::move(output);
  } else if (options.has("voice")) {
    throw UsageError("--voice requires --audio-out");
  }

  if (request.stream) {
    request.on_delta = [](const std::string& delta) {
      std::cout << delta << std::flush;
    };
  }
  const ChatResponse response = client.chat(request);
  if (!request.stream) std::cout << response.text;
  std::cout << '\n';
  if (!audio_out.empty()) {
    if (response.audio.empty()) {
      throw Error(ErrorCode::kEmpty, "the server returned no audio");
    }
    write_output(audio_out, response.audio);
    if (!response.audio_transcript.empty()) {
      std::cerr << "transcript: " << response.audio_transcript << '\n';
    }
  }
  return 0;
}

ASRConfig asr_config(const Options& options) {
  ASRConfig config;
  config.base_url = options.get("url", "http://127.0.0.1:8000");
  config.model = options.get("model");
  config.final_base_url = options.get("final-url");
  config.api_key = api_key_from_environment();
  config.stall_timeout = milliseconds(options, "stall-ms", config.stall_timeout.count());
  return config;
}

int transcribe(int argc, char** argv, std::atomic<bool>* cancel) {
  Options options(argc, argv, 2,
                  {"url", "model", "file", "format", "language", "prompt", "audio-layout",
                   "max-new-tokens", "timeout-ms", "stall-ms"},
                  {});
  if (options.help) {
    usage("transcribe", std::cout);
    return 0;
  }
  ASRClient client(asr_config(options));
  TranscribeRequest request;
  request.pcm = audio_pcm(options);
  request.language = options.get("language");
  request.prompt = options.get("prompt");
  request.audio_layout = options.get("audio-layout");
  request.max_new_tokens = static_cast<int>(integer(options, "max-new-tokens", 0, 0, 1000000));
  request.timeout = milliseconds(options, "timeout-ms", 0);
  request.cancel = cancel;
  std::cout << client.transcribe(request).text << '\n';
  return 0;
}

int realtime(int argc, char** argv, std::atomic<bool>* cancel) {
  Options options(argc, argv, 2,
                  {"url", "final-url", "model", "file", "format", "language", "prompt", "audio-layout",
                   "max-new-tokens", "chunk-ms", "final-timeout-ms", "stall-ms",
                   "ping-interval-ms"},
                  {"live-result"});
  if (options.help) {
    usage("realtime", std::cout);
    return 0;
  }
  ASRClient client(asr_config(options));
  RealtimeRequest request;
  request.language = options.get("language");
  request.prompt = options.get("prompt");
  request.audio_layout = options.get("audio-layout");
  request.max_new_tokens = static_cast<int>(integer(options, "max-new-tokens", 0, 0, 1000000));
  request.live_result = options.has("live-result");
  request.final_timeout = milliseconds(options, "final-timeout-ms", request.final_timeout.count());
  request.ping_interval = milliseconds(options, "ping-interval-ms", 0);
  request.cancel = cancel;
  request.on_partial = [](const std::string& text) {
    std::cerr << "partial: " << text << '\n';
  };
  const int64_t chunk_ms = integer(options, "chunk-ms", 100, 20, 5000);
  const size_t chunk_bytes = static_cast<size_t>(chunk_ms * 32);
  const std::string pcm = audio_pcm(options);
  auto session = client.realtime(request);
  // Sleep in 50 ms steps so SIGINT is honoured inside a long chunk interval.
  // ponytail: fixed step, raise it to a waitable event if jitter matters.
  const auto pause = [&](int64_t total_ms) {
    constexpr int64_t kStep = 50;
    for (int64_t slept = 0; slept < total_ms; slept += kStep) {
      if (cancel->load()) return;
      std::this_thread::sleep_for(Milliseconds{std::min(kStep, total_ms - slept)});
    }
  };
  for (size_t offset = 0; offset < pcm.size() && !cancel->load();) {
    const size_t count = std::min(chunk_bytes, pcm.size() - offset);
    session->send(std::string_view(pcm).substr(offset, count));
    offset += count;
    if (offset < pcm.size()) pause(chunk_ms);
  }
  if (cancel->load()) throw Error(ErrorCode::kCancelled, "request cancelled");
  const TranscribeResponse response = session->finish();
  if (request.ping_interval.count() > 0) {
    const auto stats = session->ping_stats();
    std::cerr << "realtime ping sent=" << stats.sent << " pong=" << stats.pongs
              << '\n';
  }
  std::cout << response.text << '\n';
  return 0;
}

int image(int argc, char** argv, std::atomic<bool>* cancel) {
  std::string subcommand = argc >= 3 ? argv[2] : "";
  if (subcommand == "--help" || subcommand == "-h") {
    usage("image", std::cout);
    return 0;
  }
  if (subcommand != "generate" && subcommand != "edit") {
    std::cerr << "error: image requires the subcommand generate or edit\n";
    usage("image", std::cerr);
    return 2;
  }
  const bool generate = subcommand == "generate";
  Options options(argc, argv, 3,
                  {"url", "model", "prompt", "out", "n", "size", "timeout-ms"},
                  {}, generate ? std::set<std::string>{"field"}
                               : std::set<std::string>{"image", "field"});
  if (options.help) {
    usage("image", std::cout);
    return 0;
  }
  // Every required argument and every output target is checked before a
  // request is sent. The generation is billable and cannot be replayed.
  const std::string prompt = options.get("prompt");
  if (prompt.empty()) throw UsageError("--prompt is required");
  const std::string out = options.get("out");
  if (out.empty()) throw UsageError("--out is required");
  const size_t count = static_cast<size_t>(integer(options, "n", 1, 1, 10));
  const std::vector<std::string> paths = artifact_paths(out, count);
  for (const auto& path : paths) check_output(path);
  if (!generate && !options.has("image")) throw UsageError("--image is required");

  const OpenAIConfig config = media_config(options);
  require_model(config);
  OpenAIClient client(config);

  ImageResponse response;
  if (generate) {
    ImageRequest request;
    request.model = options.get("model");
    request.prompt = prompt;
    // A single image keeps the previous body. n above one is the multi-output
    // request.
    request.n = count > 1 ? static_cast<int>(count) : 0;
    request.size = options.get("size");
    add_fields(options, request.extra_body);
    request.timeout = milliseconds(options, "timeout-ms", 0);
    request.cancel = cancel;
    response = client.generate_image(request);
  } else {
    ImageEditRequest request;
    request.model = options.get("model");
    request.prompt = prompt;
    for (const auto& path : options.all("image")) {
      ImagePart part;
      part.data = read_limited(path, 64u << 20, "image");
      part.filename = basename_of(path);
      part.content_type = image_content_type(path);
      request.images.push_back(std::move(part));
    }
    for (const auto& field : options.all("field")) {
      const size_t eq = field.find('=');
      if (eq == std::string::npos || eq == 0) {
        throw UsageError("--field requires NAME=VALUE: " + field);
      }
      request.fields.emplace_back(field.substr(0, eq), field.substr(eq + 1));
    }
    // --n and --size are multipart fields on this route. They are forwarded
    // instead of dropped.
    if (options.has("n")) request.fields.emplace_back("n", std::to_string(count));
    if (options.has("size")) request.fields.emplace_back("size", options.get("size"));
    request.timeout = milliseconds(options, "timeout-ms", 0);
    request.cancel = cancel;
    response = client.edit_image(request);
  }
  if (response.images.size() != count) {
    throw Error(ErrorCode::kBadResponse,
                "expected " + std::to_string(count) + " image(s), the server returned " +
                    std::to_string(response.images.size()));
  }
  write_outputs(paths, response.images);
  return 0;
}

int video(int argc, char** argv, std::atomic<bool>* cancel) {
  std::string subcommand = argc >= 3 ? argv[2] : "";
  if (subcommand == "--help" || subcommand == "-h") {
    usage("video", std::cout);
    return 0;
  }
  const bool create = subcommand == "create";
  const bool status = subcommand == "status";
  const bool content = subcommand == "content";
  const bool remove = subcommand == "delete";
  if (!create && !status && !content && !remove) {
    std::cerr << "error: video requires create, status, content or delete\n";
    usage("video", std::cerr);
    return 2;
  }

  std::set<std::string> values = {"url", "id", "timeout-ms"};
  std::set<std::string> repeated;
  if (create) {
    values = {"url", "model", "prompt", "timeout-ms"};
    repeated = {"field"};
  } else if (content) {
    values.insert("out");
    values.insert("variant");
  }
  Options options(argc, argv, 3, values, {}, repeated);
  if (options.help) {
    usage("video", std::cout);
    return 0;
  }
  // Required arguments and the output target are checked before a request is
  // sent. A missing flag is a usage error, not a server-side rejection.
  const std::string prompt = options.get("prompt");
  if (create && prompt.empty()) throw UsageError("--prompt is required");
  if (content) {
    if (options.get("out").empty()) throw UsageError("--out is required");
    check_output(options.get("out"));
  }
  const OpenAIConfig config = media_config(options);
  if (create) require_model(config);
  OpenAIClient client(config);

  if (create) {
    VideoRequest request;
    request.model = options.get("model");
    request.prompt = prompt;
    add_fields(options, request.body);
    request.timeout = milliseconds(options, "timeout-ms", 0);
    request.cancel = cancel;
    std::cout << client.create_video(request).raw.dump() << '\n';
    return 0;
  }

  const std::string id = options.get("id");
  if (id.empty()) throw UsageError("--id is required");
  if (status) {
    std::cout << client.video_status(id, cancel).raw.dump() << '\n';
  } else if (remove) {
    std::cout << client.delete_video(id, cancel).dump() << '\n';
  } else {
    std::optional<int> variant;
    if (options.has("variant")) {
      variant = static_cast<int>(
          integer(options, "variant", 0, 0, std::numeric_limits<int>::max()));
    }
    client.download_video(id, options.get("out"), variant, cancel);
    std::cout << options.get("out") << '\n';
  }
  return 0;
}

}  // namespace

int run_cli(const std::string& command, int argc, char** argv, std::atomic<bool>* cancel) {
  try {
    if (command == "chat") return chat(argc, argv, cancel);
    if (command == "transcribe") return transcribe(argc, argv, cancel);
    if (command == "realtime") return realtime(argc, argv, cancel);
    if (command == "image") return image(argc, argv, cancel);
    if (command == "video") return video(argc, argv, cancel);
    throw UsageError("unknown client command: " + command);
  } catch (const UsageError& error) {
    std::cerr << "error: " << error.what() << '\n';
    usage(command, std::cerr);
    return 2;
  }
}

}  // namespace umminfer::client

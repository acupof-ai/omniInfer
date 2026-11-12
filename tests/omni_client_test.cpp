// SPDX-License-Identifier: Apache-2.0
// Batch 3 checks for the public client and the fixed-upstream media routes.
// Every request goes to a real local ummInfer server. The configured upstream is
// an unbound loopback port, so forwarded requests fail at connect time. No fake
// upstream, no hand-written protocol response and no inference take part.
// Success against a real Qwen or SGLang upstream is NOT_COVERED.
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "client/cli.hpp"
#include "client/http.hpp"
#include "umminfer/client/client.hpp"
#include "umminfer/serve/openai_api.hpp"

using namespace umminfer;
namespace detail = umminfer::client::detail;

namespace {

int failures = 0;
int checks = 0;

#define CHECK(cond)                                                         \
  do {                                                                      \
    ++checks;                                                               \
    if (!(cond)) {                                                          \
      std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << " " << #cond   \
                << "\n";                                                    \
      ++failures;                                                           \
    }                                                                       \
  } while (0)

constexpr const char* kUpstreamKey = "sk-upstream-CLIENT-SECRET";
constexpr const char* kInboundKey = "inbound-client-key";
constexpr const char* kDeadUpstream = "http://127.0.0.1:9";

struct TestServer {
  explicit TestServer(ApiConfig config)
      : coordinator("ipc:///tmp/omni-client-test-" + std::to_string(::getpid()) +
                        "-" + std::to_string(++counter()) + "-done",
                    "ipc:///tmp/omni-client-test-" + std::to_string(::getpid()) +
                        "-" + std::to_string(counter()) + "-abort"),
        api(coordinator, config),
        inbound_key(config.api_key) {}

  ~TestServer() {
    api.stop();
    if (worker.joinable()) worker.join();
  }

  bool start() {
    if (!api.bind()) return false;
    port = api.bound_port();
    worker = std::thread([this] { api.listen(); });
    for (int attempt = 0; attempt < 100; ++attempt) {
      detail::HttpRequest probe;
      probe.url = "http://127.0.0.1:" + std::to_string(port) + "/health";
      probe.method = detail::HttpMethod::kGet;
      probe.bearer = inbound_key;
      probe.timeout = client::Milliseconds{2000};
      try {
        if (detail::perform(probe).status == 200) return true;
      } catch (const std::exception&) {
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return false;
  }

  static int& counter() {
    static int value = 0;
    return value;
  }

  Coordinator coordinator;
  OpenAIApi api;
  std::string inbound_key;
  std::thread worker;
  int port = 0;
};

ApiConfig base_config() {
  ApiConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.model = "served-model";
  return config;
}

ApiConfig qwen_config() {
  ApiConfig config = base_config();
  config.upstream_kind = "qwen";
  config.upstream_base_url = kDeadUpstream;
  config.upstream_model = "qwen3.5-omni-plus";
  config.upstream_api_key = kUpstreamKey;
  config.upstream_variant = "omni";
  return config;
}

ApiConfig sglang_config(const std::string& variant) {
  ApiConfig config = base_config();
  config.upstream_kind = "sglang";
  config.upstream_base_url = kDeadUpstream;
  config.upstream_model = "MiniMaxAI/MiniMax-H3";
  config.upstream_variant = variant;
  return config;
}

client::OpenAIConfig client_config(int port, const std::string& key,
                                   int attempts = 1) {
  client::OpenAIConfig config;
  config.base_url = "http://127.0.0.1:" + std::to_string(port) + "/v1";
  config.model = "served-model";
  config.api_key = key;
  config.max_attempts = attempts;
  config.retry_base = client::Milliseconds{2000};
  config.retry_max = client::Milliseconds{2000};
  config.connect_timeout = client::Milliseconds{2000};
  config.timeout = client::Milliseconds{5000};
  return config;
}

template <typename Function>
client::ErrorCode error_code(Function&& function, int* status = nullptr) {
  try {
    function();
  } catch (const client::Error& error) {
    if (status != nullptr) *status = error.status();
    return error.code();
  } catch (const std::exception& error) {
    std::cerr << "unexpected exception: " << error.what() << "\n";
    return client::ErrorCode::kEmpty;
  }
  return client::ErrorCode::kEmpty;
}

// Runs one raw request whose HTTP error status is the expected result.
// detail::perform throws on any non-2xx status, so the status and message are
// collected here instead of escaping the test.
int raw_status(detail::HttpRequest request, std::string* text = nullptr) {
  try {
    const detail::HttpResponse response = detail::perform(request);
    if (text != nullptr) *text = response.body;
    return static_cast<int>(response.status);
  } catch (const client::Error& error) {
    if (text != nullptr) *text = error.what();
    return error.status();
  } catch (const std::exception& error) {
    if (text != nullptr) *text = error.what();
    return 0;
  }
}

client::Message text_message(std::string text) {
  client::Message message;
  message.role = "user";
  message.content = std::move(text);
  return message;
}

void test_validation() {
  // The base URL is never contacted: every case must fail before a connection.
  client::OpenAIConfig config;
  config.base_url = "http://127.0.0.1:9";
  config.model = "served-model";
  client::OpenAIClient client(config);

  client::ChatRequest chat;
  chat.messages.push_back(text_message("hi"));
  chat.audio = client::AudioOutput{"Tina", "wav"};
  CHECK(error_code([&] { client.chat(chat); }) == client::ErrorCode::kConfig);

  chat.modalities = {"text", "audio"};
  chat.audio->format = "mp3";
  chat.stream = true;
  CHECK(error_code([&] { client.chat(chat); }) == client::ErrorCode::kConfig);

  chat.audio->format = "wav";
  chat.stream = false;
  CHECK(error_code([&] { client.chat(chat); }) == client::ErrorCode::kConfig);

  chat.stream = true;
  chat.modalities = {"voice"};
  CHECK(error_code([&] { client.chat(chat); }) == client::ErrorCode::kConfig);

  client::ChatRequest both;
  client::Message message;
  message.role = "user";
  message.content = "text";
  client::ContentPart part;
  part.kind = client::ContentPart::Kind::kText;
  part.text = "text";
  message.parts.push_back(part);
  both.messages.push_back(message);
  CHECK(error_code([&] { client.chat(both); }) == client::ErrorCode::kConfig);

  client::ChatRequest empty_image;
  client::Message image_message;
  image_message.role = "user";
  client::ContentPart image_part;
  image_part.kind = client::ContentPart::Kind::kImageUrl;
  image_message.parts.push_back(image_part);
  empty_image.messages.push_back(image_message);
  CHECK(error_code([&] { client.chat(empty_image); }) == client::ErrorCode::kConfig);

  client::ChatRequest fast_video;
  client::Message video_message;
  video_message.role = "user";
  client::ContentPart video_part;
  video_part.kind = client::ContentPart::Kind::kVideoUrl;
  video_part.url = "https://example.com/a.mp4";
  video_part.fps = 20;
  video_message.parts.push_back(video_part);
  fast_video.messages.push_back(video_message);
  CHECK(error_code([&] { client.chat(fast_video); }) == client::ErrorCode::kConfig);

  client::ImageRequest generation;
  CHECK(error_code([&] { client.generate_image(generation); }) ==
        client::ErrorCode::kConfig);
  generation.prompt = "x";
  generation.extra_body["response_format"] = "url";
  CHECK(error_code([&] { client.generate_image(generation); }) ==
        client::ErrorCode::kConfig);

  client::ImageEditRequest edit;
  edit.prompt = "x";
  CHECK(error_code([&] { client.edit_image(edit); }) ==
        client::ErrorCode::kConfig);
  edit.images.push_back({"image-bytes", "a.png", "image/png"});
  edit.fields = {{"size", "1"}, {"size", "2"}};
  CHECK(error_code([&] { client.edit_image(edit); }) ==
        client::ErrorCode::kConfig);
  edit.fields = {{"model", "other"}};
  CHECK(error_code([&] { client.edit_image(edit); }) ==
        client::ErrorCode::kConfig);

  // The edit parser accepts inline bytes only, so response_format is forced to
  // b64_json and a caller field that tries to override it is rejected.
  client::ImageEditRequest format_edit;
  format_edit.prompt = "x";
  format_edit.images.push_back({"image-bytes", "a.png", "image/png"});
  format_edit.fields = {{"response_format", "url"}};
  CHECK(error_code([&] { client.edit_image(format_edit); }) ==
        client::ErrorCode::kConfig);

  client::VideoRequest video;
  CHECK(error_code([&] { client.create_video(video); }) ==
        client::ErrorCode::kConfig);
  video.prompt = "x";
  video.body["model"] = "other";
  CHECK(error_code([&] { client.create_video(video); }) ==
        client::ErrorCode::kConfig);

  CHECK(error_code([&] { client.video_status("a/b"); }) ==
        client::ErrorCode::kConfig);
  CHECK(error_code([&] { client.delete_video(""); }) ==
        client::ErrorCode::kConfig);
  CHECK(error_code([&] { client.download_video("vid-1", ""); }) ==
        client::ErrorCode::kConfig);
  // The zero-based output index must not be negative.
  CHECK(error_code([&] {
          client.download_video("vid-1", "/tmp/omni-client-check/never.mp4", -1);
        }) == client::ErrorCode::kConfig);

  // A cancelled flag fails before any connection.
  std::atomic<bool> cancelled{true};
  chat.cancel = &cancelled;
  chat.modalities = {"text"};
  chat.audio.reset();
  CHECK(error_code([&] { client.chat(chat); }) ==
        client::ErrorCode::kCancelled);
  client::ImageRequest cancelled_image;
  cancelled_image.prompt = "x";
  cancelled_image.cancel = &cancelled;
  CHECK(error_code([&] { client.generate_image(cancelled_image); }) ==
        client::ErrorCode::kCancelled);
  CHECK(error_code([&] { client.video_status("vid-1", &cancelled); }) ==
        client::ErrorCode::kCancelled);
}

void test_qwen_server() {
  ApiConfig server_config = qwen_config();
  server_config.api_key = kInboundKey;
  TestServer server(server_config);
  CHECK(server.start());
  if (server.port == 0) return;

  // Authentication runs before any route logic.
  {
    client::OpenAIClient wrong(client_config(server.port, "wrong-key"));
    client::ChatRequest request;
    request.messages.push_back(text_message("hi"));
    int status = 0;
    CHECK(error_code([&] { wrong.chat(request); }, &status) ==
          client::ErrorCode::kAuthentication);
    CHECK(status == 401);
  }

  client::OpenAIClient client(client_config(server.port, kInboundKey));

  // Text chat still reaches the fixed upstream.
  {
    client::ChatRequest request;
    request.messages.push_back(text_message("hi"));
    int status = 0;
    const auto code = error_code([&] { client.chat(request); }, &status);
    CHECK(code == client::ErrorCode::kServer);
    CHECK(status == 502);
  }

  // Multimodal content parts reach the proxy and fail only at the upstream.
  {
    client::ChatRequest request;
    client::Message message;
    message.role = "user";
    client::ContentPart text;
    text.kind = client::ContentPart::Kind::kText;
    text.text = "what is this";
    client::ContentPart image;
    image.kind = client::ContentPart::Kind::kImageUrl;
    image.url = "https://example.com/a.jpg";
    client::ContentPart audio;
    audio.kind = client::ContentPart::Kind::kInputAudio;
    audio.audio = "AAAA";
    audio.audio_format = "mp3";
    message.parts = {text, image, audio};
    request.messages.push_back(message);
    std::string detail;
    int status = 0;
    try {
      client.chat(request);
    } catch (const client::Error& error) {
      detail = error.what();
      status = error.status();
    } catch (const std::exception& error) {
      detail = error.what();
    }
    CHECK(status == 502);
    CHECK(detail.find(kUpstreamKey) == std::string::npos);
    CHECK(detail.find(kInboundKey) == std::string::npos);
  }

  // Speech output requires streaming and reaches the proxy as a stream.
  {
    client::ChatRequest request;
    request.messages.push_back(text_message("say hi"));
    request.stream = true;
    request.modalities = {"text", "audio"};
    request.audio = client::AudioOutput{"Tina", "wav"};
    bool delta = false;
    request.on_delta = [&](const std::string&) { delta = true; };
    request.timeout = client::Milliseconds{5000};
    int status = 0;
    const auto start = std::chrono::steady_clock::now();
    const auto code = error_code([&] { client.chat(request); }, &status);
    const auto elapsed = std::chrono::steady_clock::now() - start;
    // The stream route commits nothing until the upstream answers. A dead
    // upstream must surface as the public 502 mapping, not as a client timeout
    // after the full request timeout.
    CHECK(code == client::ErrorCode::kServer);
    CHECK(status == 502);
    CHECK(elapsed < std::chrono::milliseconds(3000));
    CHECK(!delta);
  }

  // A fixed-upstream process cannot abort upstream work. /abort must answer 501
  // and never {"success":true}; the client's own cancel is local and never
  // claims an upstream abort.
  {
    detail::HttpRequest request;
    request.url = "http://127.0.0.1:" + std::to_string(server.port) + "/abort";
    request.method = detail::HttpMethod::kPost;
    request.bearer = kInboundKey;
    request.headers.emplace("Content-Type", "application/json");
    request.body = R"({"request_id":"req-1"})";
    request.timeout = client::Milliseconds{2000};
    std::string text;
    const int status = raw_status(std::move(request), &text);
    CHECK(status == 501);
    CHECK(text.find("\"success\":true") == std::string::npos);
    CHECK(text.find("abort is not supported") != std::string::npos);
  }
}

void test_image_server() {
  ApiConfig server_config = sglang_config("image");
  server_config.api_key = kInboundKey;
  TestServer server(server_config);
  CHECK(server.start());
  if (server.port == 0) return;

  client::OpenAIClient client(client_config(server.port, kInboundKey));

  {
    client::ImageRequest request;
    request.prompt = "a red cube";
    std::string detail;
    int status = 0;
    try {
      client.generate_image(request);
    } catch (const client::Error& error) {
      detail = error.what();
      status = error.status();
    } catch (const std::exception& error) {
      detail = error.what();
    }
    CHECK(status == 502);
    CHECK(detail.find(kUpstreamKey) == std::string::npos);
    CHECK(detail.find(kInboundKey) == std::string::npos);
  }

  {
    client::ImageRequest request;
    request.prompt = "a red cube";
    request.model = "other";
    int status = 0;
    CHECK(error_code([&] { client.generate_image(request); }, &status) ==
          client::ErrorCode::kBadRequest);
    CHECK(status == 400);
  }

  {
    client::ImageEditRequest request;
    request.prompt = "make it blue";
    request.images.push_back({"first-image", "first.png", "image/png"});
    request.images.push_back({"second-image", "second.png", "image/png"});
    request.fields = {{"url", "https://example.com/one"}, {"url", "https://example.com/two"}};
    int status = 0;
    CHECK(error_code([&] { client.edit_image(request); }, &status) ==
          client::ErrorCode::kServer);
    CHECK(status == 502);
  }

  // task_type is on the final H3 edit allowlist. The client must forward it
  // instead of dropping it, so the request stops only at the upstream.
  {
    client::ImageEditRequest request;
    request.prompt = "restore this image";
    request.images.push_back({"image-bytes", "a.png", "image/png"});
    request.fields = {{"task_type", "portrait_restoration"}};
    int status = 0;
    CHECK(error_code([&] { client.edit_image(request); }, &status) ==
          client::ErrorCode::kServer);
    CHECK(status == 502);
  }
}

void test_video_server() {
  ApiConfig server_config = sglang_config("video");
  TestServer server(server_config);
  CHECK(server.start());
  if (server.port == 0) return;

  // max_attempts is 3 and the retry delay is 2 s. A creation POST that retried
  // would take at least one second; a single POST fails in milliseconds.
  client::OpenAIClient client(client_config(server.port, "", 3));

  {
    client::VideoRequest request;
    request.prompt = "a running horse";
    request.body["task"] = "t2va";
    const auto start = std::chrono::steady_clock::now();
    int status = 0;
    CHECK(error_code([&] { client.create_video(request); }, &status) ==
          client::ErrorCode::kServer);
    CHECK(status == 502);
    const auto elapsed = std::chrono::steady_clock::now() - start;
    CHECK(elapsed < std::chrono::milliseconds(900));
  }

  // The final H3 create allowlist. The client must forward each field unchanged
  // instead of dropping it, so the request stops only at the upstream and is
  // still sent once.
  {
    client::VideoRequest request;
    request.prompt = "a running horse";
    request.body["task"] = "t2va";
    request.body["num_inference_steps"] = 32;
    request.body["flow_shift"] = 3.0;
    request.body["audio_flow_shift"] = 1.5;
    request.body["reference_resize_mode"] = "stretch";
    const auto start = std::chrono::steady_clock::now();
    int status = 0;
    CHECK(error_code([&] { client.create_video(request); }, &status) ==
          client::ErrorCode::kServer);
    CHECK(status == 502);
    const auto elapsed = std::chrono::steady_clock::now() - start;
    CHECK(elapsed < std::chrono::milliseconds(900));
  }

  {
    int status = 0;
    CHECK(error_code([&] { client.video_status("vid-1"); }, &status) ==
          client::ErrorCode::kServer);
    CHECK(status == 502);
    CHECK(error_code([&] { client.delete_video("vid-1"); }, &status) ==
          client::ErrorCode::kServer);
    CHECK(status == 502);
  }

  // A zero-based output index is forwarded as ?variant=N. Anything that is not
  // a non-negative integer is a local 400 from the proxy, never a forwarded
  // request.
  for (const char* bad : {"abc", "a/b", "bad.id"}) {
    detail::HttpRequest request;
    request.url = "http://127.0.0.1:" + std::to_string(server.port) +
                  "/v1/videos/vid-1/content?variant=" + bad;
    request.method = detail::HttpMethod::kGet;
    request.timeout = client::Milliseconds{2000};
    std::string text;
    const int status = raw_status(std::move(request), &text);
    CHECK(status == 400);
    if (status != 400) std::cerr << "variant=" << bad << " status=" << status << "\n";
  }

  const auto directory = std::filesystem::temp_directory_path() /
                         ("omni-client-test-" + std::to_string(::getpid()));
  std::error_code ignored;
  std::filesystem::create_directories(directory, ignored);
  const auto existing = directory / "existing.mp4";
  {
    std::ofstream out(existing, std::ios::binary);
    out << "keep";
  }
  {
    // An existing artifact is never replaced, even before the network.
    CHECK(error_code([&] { client.download_video("vid-1", existing.string()); }) ==
          client::ErrorCode::kBadRequest);
    std::ifstream in(existing, std::ios::binary);
    std::string content((std::istreambuf_iterator<char>(in)),
                        std::istreambuf_iterator<char>());
    CHECK(content == "keep");
  }
  {
    // A failed download leaves neither the final path nor a part file. A dead
    // upstream must surface as the public 502 mapping before the request
    // timeout, not as a client timeout with no upstream status.
    client::OpenAIClient downloader(client_config(server.port, "", 1));
    const auto fresh = directory / "fresh.mp4";
    int status = 0;
    const auto start = std::chrono::steady_clock::now();
    const auto code = error_code(
        [&] { downloader.download_video("vid-1", fresh.string(), 0); }, &status);
    const auto elapsed = std::chrono::steady_clock::now() - start;
    CHECK(code == client::ErrorCode::kServer);
    CHECK(status == 502);
    CHECK(elapsed < std::chrono::milliseconds(3000));
    CHECK(!std::filesystem::exists(fresh));
    int parts = 0;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
      if (entry.path().filename().string().find(".part.") != std::string::npos) ++parts;
    }
    CHECK(parts == 0);
  }
  std::filesystem::remove_all(directory, ignored);
}

// Runs the real CLI parser. Every case here must fail before any connection,
// so the base URL is never contacted.
int cli(std::vector<std::string> args) {
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (auto& arg : args) argv.push_back(arg.data());
  std::atomic<bool> cancel{false};
  try {
    return client::run_cli(argv[1], static_cast<int>(argv.size()), argv.data(), &cancel);
  } catch (const std::exception&) {
    return 1;
  }
}

void test_cli_usage() {
  const std::string url = "http://127.0.0.1:9/v1";
  const std::string out = "/tmp/omni-client-cli-" + std::to_string(::getpid()) + ".png";
  const std::string input = "/tmp/omni-client-cli-" + std::to_string(::getpid()) + ".jpg";

  // Required media arguments are usage errors, not client configuration errors.
  CHECK(cli({"umm-infer", "image", "generate", "--url", url, "--model", "m", "--out",
             out}) == 2);
  CHECK(cli({"umm-infer", "image", "generate", "--url", url, "--model", "m", "--prompt",
             "p"}) == 2);
  CHECK(cli({"umm-infer", "image", "edit", "--url", url, "--model", "m", "--prompt", "p",
             "--out", out}) == 2);
  CHECK(cli({"umm-infer", "image", "edit", "--url", url, "--model", "m", "--prompt", "p",
             "--image", input}) == 2);
  CHECK(cli({"umm-infer", "video", "create", "--url", url, "--model", "m"}) == 2);
  CHECK(cli({"umm-infer", "video", "content", "--url", url, "--id", "vid-1"}) == 2);

  // An empty value is not an absent flag.
  CHECK(cli({"umm-infer", "chat", "--prompt", "hi", "--temperature="}) == 2);
  CHECK(cli({"umm-infer", "chat", "--prompt", "hi", "--top-p="}) == 2);
  CHECK(cli({"umm-infer", "chat", "--prompt", "hi", "--stream", "--audio-out="}) == 2);

  // The multi-output count is bounded locally.
  CHECK(cli({"umm-infer", "image", "generate", "--url", url, "--model", "m", "--prompt",
             "p", "--out", out, "--n", "11"}) == 2);
  CHECK(cli({"umm-infer", "image", "edit", "--url", url, "--model", "m", "--prompt", "p",
             "--image", input, "--out", out, "--n", "0"}) == 2);
}

// Without a fixed upstream the local routes keep their previous behavior.
void test_no_upstream() {
  TestServer server(base_config());
  CHECK(server.start());
  if (server.port == 0) return;
  client::OpenAIClient client(client_config(server.port, ""));
  client::ChatRequest request;
  request.messages.push_back(text_message("hi"));
  int status = 0;
  CHECK(error_code([&] { client.chat(request); }, &status) ==
        client::ErrorCode::kServer);
  CHECK(status == 503);
}

}  // namespace

int main() {
  test_validation();
  test_cli_usage();
  test_qwen_server();
  test_image_server();
  test_video_server();
  test_no_upstream();
  if (failures != 0) {
    std::cerr << failures << " of " << checks << " checks failed\n";
    return 1;
  }
  std::cout << "all " << checks << " client checks passed\n";
  return 0;
}

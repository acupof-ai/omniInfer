// SPDX-License-Identifier: Apache-2.0
// Batch 2 checks for the fixed-upstream serve proxy: configuration completeness,
// route gating, field validation, path validation, authentication, and key
// scrubbing. Every request goes to a real local ummInfer server. The upstream is
// an unbound loopback port, so a forwarded request fails at connect time; no
// fake upstream server, no hand-written protocol response, no inference.
//
// Scope limit: a 502 here proves that local validation and route gating passed
// and that the transport failed to connect. It does not prove a successful
// upstream forward, request-field rewriting, or upstream key replacement.
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "client/http.hpp"
#include "umminfer/serve/openai_api.hpp"

using namespace umminfer;
namespace detail = umminfer::client::detail;

namespace {

int failures = 0;

#define CHECK(cond)                                                         \
  do {                                                                      \
    if (!(cond)) {                                                          \
      std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << " " << #cond   \
                << "\n";                                                    \
      ++failures;                                                           \
    }                                                                       \
  } while (0)

#define CHECK_EQ(actual, expected)                                          \
  do {                                                                      \
    const auto actual_value = (actual);                                     \
    const auto expected_value = (expected);                                 \
    if (!(actual_value == expected_value)) {                                \
      std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << " " << #actual \
                << " = " << actual_value << ", expected " << expected_value \
                << "\n";                                                    \
      ++failures;                                                           \
    }                                                                       \
  } while (0)

constexpr const char* kUpstreamKey = "sk-upstream-SECRET-0001";
constexpr const char* kInboundKey = "inbound-key-1234";
// Port 9 (discard) is never bound on the test machine, so a forwarded request
// fails at connect time instead of reaching any service. A 502 below therefore
// proves only that local validation passed and the outbound connect was
// attempted; it does not prove the upstream path, an upstream success, or an
// authentication header reaching a real server. Those stay NOT_COVERED until a
// pinned SGLang or an authorized Qwen endpoint exists.
constexpr const char* kDeadUpstream = "http://127.0.0.1:9";

struct Result {
  long status = 0;
  std::string body;
  std::string error;
};

Result call(int port, const std::string& method, const std::string& path,
            const std::string& body = "", const std::string& bearer = "",
            const std::vector<std::pair<std::string, std::string>>& headers = {},
            const std::vector<detail::MimePart>& parts = {}) {
  detail::HttpRequest request;
  request.url = "http://127.0.0.1:" + std::to_string(port) + path;
  if (method == "GET") request.method = detail::HttpMethod::kGet;
  else if (method == "DELETE") request.method = detail::HttpMethod::kDelete;
  else request.method = detail::HttpMethod::kPost;
  request.body = body;
  request.bearer = bearer;
  request.mime_parts = parts;
  request.timeout = client::Milliseconds{10000};
  for (const auto& [name, value] : headers) request.headers.emplace(name, value);
  // cpp-httplib parses a body with no declared type as form-urlencoded and
  // rejects JSON bodies above its 8 KiB form limit with 413. Every JSON request
  // in this file must announce its type; multipart requests keep their own
  // part-derived type.
  if (request.method == detail::HttpMethod::kPost && !body.empty() && parts.empty() &&
      !request.headers.contains("Content-Type") &&
      !request.headers.contains("content-type")) {
    request.headers.emplace("Content-Type", "application/json");
  }
  Result result;
  try {
    const auto response = detail::perform(request);
    result.status = response.status;
    result.body = response.body;
  } catch (const client::Error& error) {
    result.status = error.status() != 0 ? error.status() : -1;
    result.error = error.what();
  } catch (const std::exception& error) {
    result.status = -1;
    result.error = error.what();
  }
  return result;
}

struct TestServer {
  explicit TestServer(ApiConfig config)
      : coordinator("ipc:///tmp/omni-serve-test-" + std::to_string(::getpid()) +
                        "-" + std::to_string(++counter()) + "/done",
                    "ipc:///tmp/omni-serve-test-" + std::to_string(::getpid()) +
                        "-" + std::to_string(counter()) + "/abort"),
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
    // Wait for the accept loop, not for the bind, so a later stop() cannot race
    // with the listener it is meant to end. Authentication covers /health too,
    // so the probe carries the inbound key when one is configured.
    for (int attempt = 0; attempt < 100; ++attempt) {
      if (call(port, "GET", "/health", "", inbound_key).status == 200) return true;
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

// The upstream key is set even though SGLang does not require one, so the
// non-leak assertions below test a credential that actually exists.
ApiConfig sglang_config(const std::string& variant) {
  ApiConfig config = base_config();
  config.upstream_kind = "sglang";
  config.upstream_base_url = kDeadUpstream;
  config.upstream_model = "MiniMaxAI/MiniMax-H3-Image";
  config.upstream_api_key = kUpstreamKey;
  config.upstream_variant = variant;
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

// The production base URL, used only for offline configuration validation.
ApiConfig qwen_https_config() {
  ApiConfig config = qwen_config();
  config.upstream_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1";
  return config;
}

void test_config_validation() {
  CHECK(validate_upstream_config(base_config()).empty());
  // The production HTTPS base URL is validated offline; no request is sent.
  CHECK(validate_upstream_config(qwen_https_config()).empty());

  ApiConfig partial = base_config();
  partial.upstream_kind = "qwen";
  CHECK(!validate_upstream_config(partial).empty());

  ApiConfig no_key = qwen_config();
  no_key.upstream_api_key.clear();
  CHECK(!validate_upstream_config(no_key).empty());

  ApiConfig wrong_variant = qwen_config();
  wrong_variant.upstream_variant = "image";
  CHECK(!validate_upstream_config(wrong_variant).empty());

  ApiConfig sglang_omni = sglang_config("image");
  sglang_omni.upstream_variant = "omni";
  CHECK(!validate_upstream_config(sglang_omni).empty());

  CHECK(validate_upstream_config(sglang_config("image")).empty());
  CHECK(validate_upstream_config(sglang_config("video")).empty());

  ApiConfig plain_remote = sglang_config("image");
  plain_remote.upstream_base_url = "http://example.com";
  CHECK(!validate_upstream_config(plain_remote).empty());

  ApiConfig query = sglang_config("image");
  query.upstream_base_url = "https://example.com/v1?x=1";
  CHECK(!validate_upstream_config(query).empty());

  ApiConfig shared_key = qwen_config();
  shared_key.api_key = kUpstreamKey;
  CHECK(!validate_upstream_config(shared_key).empty());

  ApiConfig unknown_kind = sglang_config("image");
  unknown_kind.upstream_kind = "openai";
  CHECK(!validate_upstream_config(unknown_kind).empty());
}

void test_incomplete_config_fails_bind() {
  ApiConfig config = base_config();
  config.upstream_kind = "sglang";
  Coordinator coordinator("ipc:///tmp/omni-serve-bad-/done",
                          "ipc:///tmp/omni-serve-bad-/abort");
  OpenAIApi api(coordinator, config);
  CHECK(!api.bind());
}

void test_image_server() {
  ApiConfig config = sglang_config("image");
  config.api_key = kInboundKey;
  TestServer server(config);
  CHECK(server.start());
  if (server.port == 0) return;
  const std::string auth = kInboundKey;

  // Authentication covers every route, including the new ones.
  CHECK_EQ(call(server.port, "GET", "/v1/models").status, 401L);
  CHECK_EQ(call(server.port, "POST", "/v1/images/generations", "{}", "wrong").status, 401L);

  // /v1/models names only the served model and leaks neither key nor URL.
  Result models = call(server.port, "GET", "/v1/models", "", auth);
  CHECK_EQ(models.status, 200L);
  CHECK(models.body.find("served-model") != std::string::npos);
  CHECK(models.body.find(kUpstreamKey) == std::string::npos);
  CHECK(models.body.find(kInboundKey) == std::string::npos);
  CHECK(models.body.find("127.0.0.1:9") == std::string::npos);

  // Routes outside the configured variant fail explicitly.
  CHECK_EQ(call(server.port, "POST", "/v1/videos", R"({"prompt":"x"})", auth).status, 501L);
  CHECK_EQ(call(server.port, "GET", "/v1/videos/abc", "", auth).status, 501L);
  CHECK_EQ(call(server.port, "DELETE", "/v1/videos/abc", "", auth).status, 501L);
  CHECK_EQ(call(server.port, "POST", "/v1/chat/completions",
                R"({"messages":[{"role":"user","content":"hi"}]})", auth)
               .status,
           501L);
  CHECK_EQ(call(server.port, "POST", "/v1/audio/speech", "{}", auth).status, 501L);
  // Abort cannot reach a fixed upstream, so it must not answer success.
  CHECK_EQ(call(server.port, "POST", "/abort", R"({"request_id":"r1"})", auth).status,
           501L);

  // Field validation runs before any upstream contact.
  CHECK_EQ(call(server.port, "POST", "/v1/images/generations", "{}", auth).status, 400L);
  CHECK_EQ(call(server.port, "POST", "/v1/images/generations",
                R"({"prompt":"x","width":512})", auth)
               .status,
           400L);
  CHECK_EQ(call(server.port, "POST", "/v1/images/generations",
                R"({"prompt":"x","model":"other"})", auth)
               .status,
           400L);
  CHECK_EQ(call(server.port, "POST", "/v1/images/generations",
                R"({"prompt":"x","response_format":"url"})", auth)
               .status,
           400L);
  CHECK_EQ(call(server.port, "POST", "/v1/images/generations",
                R"({"prompt":"x","response_format":"b64_json"})", auth)
               .status,
           502L);

  // The forwarded failure must not echo the upstream credential.
  Result forwarded = call(server.port, "POST", "/v1/images/generations",
                          R"({"prompt":"x"})", auth);
  CHECK(forwarded.error.find(kUpstreamKey) == std::string::npos);
  CHECK(forwarded.body.find(kUpstreamKey) == std::string::npos);

  // Editing requires ordered multipart data.
  CHECK_EQ(call(server.port, "POST", "/v1/images/edits", "{}", auth).status, 400L);
  std::vector<detail::MimePart> wrong_part(1);
  wrong_part[0].name = "mask";
  wrong_part[0].data = "x";
  CHECK_EQ(call(server.port, "POST", "/v1/images/edits", "", auth, {}, wrong_part).status,
           400L);
  std::vector<detail::MimePart> no_prompt(1);
  no_prompt[0].name = "image";
  no_prompt[0].data = "image-bytes";
  no_prompt[0].filename = "image.png";
  CHECK_EQ(call(server.port, "POST", "/v1/images/edits", "", auth, {}, no_prompt).status,
           400L);
  std::vector<detail::MimePart> valid_edit(2);
  valid_edit[0].name = "image";
  valid_edit[0].data = "image-bytes";
  valid_edit[0].filename = "image.png";
  valid_edit[1].name = "prompt";
  valid_edit[1].data = "make it blue";
  CHECK_EQ(call(server.port, "POST", "/v1/images/edits", "", auth, {}, valid_edit).status,
           502L);
  // task_type is an official edit field.
  std::vector<detail::MimePart> typed_edit(3);
  typed_edit[0].name = "image";
  typed_edit[0].data = "image-bytes";
  typed_edit[0].filename = "image.png";
  typed_edit[1].name = "prompt";
  typed_edit[1].data = "make it blue";
  typed_edit[2].name = "task_type";
  typed_edit[2].data = "edit";
  CHECK_EQ(call(server.port, "POST", "/v1/images/edits", "", auth, {}, typed_edit).status,
           502L);
}

void test_video_server() {
  ApiConfig config = sglang_config("video");
  TestServer server(config);
  CHECK(server.start());
  if (server.port == 0) return;

  CHECK_EQ(call(server.port, "POST", "/v1/images/generations",
                R"({"prompt":"x"})")
               .status,
           501L);
  Result create = call(server.port, "POST", "/v1/videos", R"({"prompt":"x","fps":2})");
  CHECK_EQ(create.status, 502L);
  // Official H3 sampling fields and the video_audio condition pass validation
  // and fail only at connect time.
  CHECK_EQ(call(server.port, "POST", "/v1/videos",
                R"({"prompt":"x","num_inference_steps":8,"flow_shift":1.0,"audio_flow_shift":1.0,"reference_resize_mode":"max"})")
               .status,
           502L);
  CHECK_EQ(call(server.port, "POST", "/v1/videos",
                R"({"prompt":"x","conditions":[{"type":"video_audio","uri":"u","role":"reference"}]})")
               .status,
           502L);
  // A configured upstream key must not appear in the forwarded failure.
  CHECK(create.error.find(kUpstreamKey) == std::string::npos);
  CHECK(create.body.find(kUpstreamKey) == std::string::npos);
  CHECK_EQ(call(server.port, "POST", "/v1/videos",
                R"({"prompt":"x","unsupported":true})")
               .status,
           400L);
  CHECK_EQ(call(server.port, "POST", "/v1/videos", R"({"prompt":"x","task":"t3va"})").status,
           400L);
  CHECK_EQ(call(server.port, "POST", "/v1/videos",
                R"({"prompt":"x","conditions":[{"type":"mesh","uri":"u"}]})")
               .status,
           400L);
  CHECK_EQ(call(server.port, "POST", "/v1/videos",
                R"({"prompt":"x","task":"fl2va","conditions":[{"type":"image","uri":"u","role":"keyframe"}]})")
               .status,
           502L);

  // Status, content, and deletion each reach their own route and fail only at
  // connect time. The exact upstream path is verified by code reading, not by
  // this test.
  CHECK_EQ(call(server.port, "GET", "/v1/videos/vid-1").status, 502L);
  CHECK_EQ(call(server.port, "GET", "/v1/videos/vid-1/content").status, 502L);
  // variant is forwarded as a validated query segment; anything else is
  // rejected instead of being dropped.
  // variant is a zero-based non-negative integer. Valid values reach the
  // upstream path (502 because the test upstream port is unbound); every other
  // form is rejected locally with 400.
  CHECK_EQ(call(server.port, "GET", "/v1/videos/vid-1/content?variant=0").status, 502L);
  CHECK_EQ(call(server.port, "GET", "/v1/videos/vid-1/content?variant=2147483647").status,
           502L);
  CHECK_EQ(call(server.port, "GET", "/v1/videos/vid-1/content?variant=abc").status, 400L);
  CHECK_EQ(call(server.port, "GET", "/v1/videos/vid-1/content?variant=").status, 400L);
  CHECK_EQ(call(server.port, "GET", "/v1/videos/vid-1/content?variant=-1").status, 400L);
  CHECK_EQ(call(server.port, "GET", "/v1/videos/vid-1/content?variant=%2B1").status, 400L);
  CHECK_EQ(call(server.port, "GET", "/v1/videos/vid-1/content?variant=1.5").status, 400L);
  CHECK_EQ(call(server.port, "GET", "/v1/videos/vid-1/content?variant=%201").status, 400L);
  CHECK_EQ(call(server.port, "GET", "/v1/videos/vid-1/content?variant=2147483648").status,
           400L);
  CHECK_EQ(call(server.port, "GET", "/v1/videos/vid-1/content?variant=0&variant=1").status,
           400L);
  CHECK_EQ(call(server.port, "GET", "/v1/videos/vid-1/content?variant=&variant=1").status,
           400L);
  CHECK_EQ(call(server.port, "GET", "/v1/videos/vid-1/content?other=1").status, 400L);
  CHECK_EQ(call(server.port, "DELETE", "/v1/videos/vid-1").status, 502L);

  // A segment outside the identifier class never reaches the handler.
  CHECK_EQ(call(server.port, "GET", "/v1/videos/bad.id").status, 404L);
  CHECK_EQ(call(server.port, "GET", "/v1/videos/a/b").status, 404L);
  CHECK_EQ(call(server.port, "GET", "/v1/generations").status, 404L);
}

void test_qwen_server() {
  ApiConfig config = qwen_config();
  config.upstream_base_url = kDeadUpstream;
  TestServer server(config);
  CHECK(server.start());
  if (server.port == 0) return;

  CHECK_EQ(call(server.port, "POST", "/v1/images/generations",
                R"({"prompt":"x"})")
               .status,
           501L);
  CHECK_EQ(call(server.port, "POST", "/v1/videos", R"({"prompt":"x"})").status, 501L);

  auto chat = [&](const std::string& body) {
    return call(server.port, "POST", "/v1/chat/completions", body);
  };
  CHECK_EQ(chat(R"({"messages":[]})").status, 400L);
  CHECK_EQ(chat(R"({"messages":[{"role":"tool","content":"hi"}]})").status, 400L);
  CHECK_EQ(
      chat(R"({"messages":[{"role":"user","content":[{"type":"input_image","input_image":{"url":"u"}}]}]})")
          .status,
      400L);
  CHECK_EQ(chat(R"({"messages":[{"role":"user","content":"hi"}],"top_k":5})").status,
           400L);
  CHECK_EQ(chat(R"({"messages":[{"role":"user","content":"hi"}],"model":"other"})").status,
           400L);
  // Speech output needs both the audio object and streaming.
  CHECK_EQ(
      chat(R"({"messages":[{"role":"user","content":"hi"}],"modalities":["text","audio"]})")
          .status,
      400L);
  CHECK_EQ(
      chat(R"({"messages":[{"role":"user","content":"hi"}],"modalities":["text","audio"],"audio":{"voice":"Tina","format":"mp3"},"stream":true})")
          .status,
      400L);
  CHECK_EQ(
      chat(R"({"messages":[{"role":"user","content":"hi"}],"modalities":["text","audio"],"audio":{"voice":"Tina","format":"wav"},"stream":false})")
          .status,
      400L);

  // A well-formed request is forwarded and fails only at connect time. The
  // streaming route takes the same path and must map the failure to 502
  // instead of committing a 200 with an empty stream.
  Result forwarded = chat(R"({"messages":[{"role":"user","content":"hi"}]})");
  CHECK_EQ(forwarded.status, 502L);
  CHECK(forwarded.error.find(kUpstreamKey) == std::string::npos);
  Result streamed = chat(
      R"({"messages":[{"role":"user","content":"hi"}],"stream":true})");
  CHECK_EQ(streamed.status, 502L);
  CHECK(streamed.error.find(kUpstreamKey) == std::string::npos);

  Result multimodal = chat(
      R"({"messages":[{"role":"user","content":[{"type":"text","text":"what is this"},{"type":"image_url","image_url":{"url":"https://example.com/a.jpg"}},{"type":"input_audio","input_audio":{"data":"AAAA","format":"mp3"}}]}]})");
  CHECK_EQ(multimodal.status, 502L);

  // Qwen3.5-Omni accepts 2 to 2048 frames.
  CHECK_EQ(chat(R"({"messages":[{"role":"user","content":[{"type":"video","video":["u1"]}]}]})")
               .status,
           400L);
  CHECK_EQ(chat(R"({"messages":[{"role":"user","content":[{"type":"video","video":["u1","u2"]}]}]})")
               .status,
           502L);
  // The upper bound is enforced locally: 2048 frames reach the upstream and
  // fail only at connect time, 2049 is rejected before any transfer.
  auto frames_body = [](int count) {
    std::string frames = "[";
    for (int i = 0; i < count; ++i) {
      if (i != 0) frames += ",";
      frames += R"("u")";
    }
    frames += "]";
    return R"({"messages":[{"role":"user","content":[{"type":"video","video":)" +
           frames + "}]}]}";
  };
  CHECK_EQ(chat(frames_body(2048)).status, 502L);
  CHECK_EQ(chat(frames_body(2049)).status, 400L);
  Result video_part = chat(
      R"({"messages":[{"role":"user","content":[{"type":"video_url","video_url":{"url":"https://example.com/a.mp4"},"fps":2}]}]})");
  CHECK_EQ(video_part.status, 502L);
}

// Without an upstream the existing local routes keep their previous behavior.
void test_no_upstream_unchanged() {
  TestServer server(base_config());
  CHECK(server.start());
  if (server.port == 0) return;
  CHECK_EQ(call(server.port, "GET", "/v1/models").status, 200L);
  CHECK_EQ(call(server.port, "POST", "/v1/audio/speech", "{}").status, 501L);
  // The chat route still runs the coordinator path instead of the proxy gate.
  CHECK_EQ(call(server.port, "POST", "/v1/chat/completions",
                R"({"messages":[{"role":"user","content":"hi"}]})")
               .status,
           503L);
}

// The transfer thread must never own the relay. A failure right after the
// request starts releases the handler reference while the worker still runs;
// the old strong ownership made the worker drop the last reference and join
// itself. Forty concurrent failures keep that window open.
void test_relay_ownership() {
  TestServer server(sglang_config("video"));
  CHECK(server.start());
  if (server.port == 0) return;
  std::atomic<int> wrong{0};
  std::vector<std::thread> clients;
  for (int worker = 0; worker < 8; ++worker) {
    clients.emplace_back([&server, &wrong] {
      for (int i = 0; i < 8; ++i) {
        if (call(server.port, "GET", "/v1/videos/vid-1/content").status != 502) {
          ++wrong;
        }
      }
    });
  }
  for (auto& client : clients) client.join();
  CHECK_EQ(wrong.load(), 0);
  CHECK_EQ(call(server.port, "GET", "/health").status, 200L);
}

}  // namespace

int main() {
  test_config_validation();
  test_incomplete_config_fails_bind();
  test_image_server();
  test_video_server();
  test_qwen_server();
  test_no_upstream_unchanged();
  test_relay_ownership();
  if (failures != 0) {
    std::cerr << failures << " checks failed\n";
    return 1;
  }
  std::cout << "all serve proxy checks passed\n";
  return 0;
}

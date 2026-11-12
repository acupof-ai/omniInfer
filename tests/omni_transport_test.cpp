// SPDX-License-Identifier: Apache-2.0
// Network-free checks for the Batch 1 transport additions: base64, WAV sample
// rate, streamed multipart file callbacks, request validation, and private
// part-file publishing. No fake server and no model are involved.
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "client/http.hpp"
#include "umminfer/audio/audio.hpp"

using namespace umminfer;
using namespace umminfer::client;
namespace detail = umminfer::client::detail;

namespace {

int failures = 0;

#define CHECK(cond)                                                          \
  do {                                                                       \
    if (!(cond)) {                                                           \
      std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << " " << #cond    \
                << "\n";                                                     \
      ++failures;                                                            \
    }                                                                        \
  } while (0)

template <typename Fn>
void expect_error(Fn fn, const char* label) {
  try {
    fn();
  } catch (const std::exception&) {
    return;
  }
  std::cerr << "FAIL expected an error: " << label << "\n";
  ++failures;
}

uint16_t read_u16(std::string_view bytes, size_t offset) {
  return static_cast<uint16_t>(static_cast<unsigned char>(bytes[offset])) |
         static_cast<uint16_t>(static_cast<unsigned char>(bytes[offset + 1]) << 8);
}

uint32_t read_u32(std::string_view bytes, size_t offset) {
  return static_cast<uint32_t>(static_cast<unsigned char>(bytes[offset])) |
         (static_cast<uint32_t>(static_cast<unsigned char>(bytes[offset + 1])) << 8) |
         (static_cast<uint32_t>(static_cast<unsigned char>(bytes[offset + 2])) << 16) |
         (static_cast<uint32_t>(static_cast<unsigned char>(bytes[offset + 3])) << 24);
}

std::string slurp(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

struct TempDir {
  std::string path;
  TempDir() {
    char pattern[] = "/tmp/umminfer-omni-XXXXXX";
    char* made = ::mkdtemp(pattern);
    if (made == nullptr) throw std::runtime_error("mkdtemp failed");
    path = made;
  }
  ~TempDir() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
};

void test_base64() {
  CHECK(detail::base64_decode("", 0).empty());
  CHECK(detail::base64_decode("TQ==", 1) == "M");
  CHECK(detail::base64_decode("TWE=", 2) == "Ma");
  CHECK(detail::base64_decode("TWFu", 3) == "Man");
  CHECK(detail::base64_decode("TWFuTQ==", 4) == "ManM");

  const std::string bytes("\x00\x01\xfe\xffhello", 11);
  CHECK(detail::base64_decode(detail::base64_encode(bytes), 11) == bytes);

  const char* invalid[] = {
      "TQ=",        // length is not a multiple of 4
      "T=Q=",       // padding before the end
      "====",       // padding only
      "A===",       // padding in the second position
      "AA=A",       // padding inside the group
      "AB==",       // non-zero trailing bits
      "AAB=",       // non-zero trailing bits
      "TQ===",      // bad length
      "T WFu",      // whitespace
      "TWFu\n",     // newline
      "TWF*",       // non-alphabet character
  };
  for (const char* text : invalid) {
    expect_error([&] { detail::base64_decode(text, 1024); }, text);
  }
  expect_error([] { detail::base64_decode("TWFu", 2); }, "output limit");
}

void test_wav() {
  const std::string pcm("\x01\x02\x03\x04", 4);
  const std::string wav = audio::wav_from_pcm16(pcm);
  CHECK(wav.size() == 48);
  CHECK(wav.substr(0, 4) == "RIFF");
  CHECK(wav.substr(8, 4) == "WAVE");
  CHECK(read_u32(wav, 4) == 40);
  CHECK(read_u16(wav, 20) == 1);       // PCM
  CHECK(read_u16(wav, 22) == 1);       // mono
  CHECK(read_u32(wav, 24) == 16000);
  CHECK(read_u32(wav, 28) == 32000);
  CHECK(read_u16(wav, 34) == 16);
  CHECK(read_u32(wav, 40) == 4);

  const std::string wav24 = audio::wav_from_pcm16(pcm, 24000);
  CHECK(read_u32(wav24, 24) == 24000);
  CHECK(read_u32(wav24, 28) == 48000);
  CHECK(read_u32(wav24, 40) == 4);

  const std::string empty = audio::wav_from_pcm16("", 8000);
  CHECK(empty.size() == 44);
  CHECK(read_u32(empty, 40) == 0);

  expect_error([&] { audio::wav_from_pcm16("\x01", 16000); }, "odd PCM");
  expect_error([&] { audio::wav_from_pcm16(pcm, 0); }, "zero rate");
  expect_error([&] { audio::wav_from_pcm16(pcm, 0x80000000u); }, "rate overflow");
}

void test_mime_source() {
  const std::string body = "abcdef";
  size_t offset = 0;
  size_t largest = 0;
  detail::MimeSource source(
      [&](char* buffer, size_t max_bytes) {
        largest = std::max(largest, max_bytes);
        const size_t count = std::min<size_t>(max_bytes, body.size() - offset);
        std::memcpy(buffer, body.data() + offset, count);
        offset += count;
        return count;
      },
      body.size());
  char buffer[16];
  CHECK(source.pump(buffer, 2) == 2);
  CHECK(source.pump(buffer, 2) == 2);
  CHECK(source.pump(buffer, 2) == 2);
  CHECK(source.pump(buffer, 2) == 0);
  CHECK(source.consumed() == body.size());
  CHECK(largest <= 2);

  detail::MimeSource short_source([](char*, size_t) { return size_t{0}; }, 4);
  expect_error([&] { short_source.pump(buffer, 4); }, "early end");

  detail::MimeSource overrun([](char*, size_t max_bytes) { return max_bytes + 1; }, 4);
  expect_error([&] { overrun.pump(buffer, 4); }, "overrun");

  detail::MimeSource empty([](char*, size_t) { return size_t{0}; }, 0);
  CHECK(empty.pump(buffer, 4) == 0);

  expect_error([] { detail::MimeSource missing(detail::MimeSource::Read{}, 0); },
               "missing callback");
}

void test_part_writer() {
  TempDir dir;
  const std::string target = dir.path + "/result.bin";

  std::string part_path;
  {
    detail::PartWriter writer(target);
    part_path = writer.part_path();
    CHECK(part_path != target);
    CHECK(part_path.rfind(dir.path + "/.", 0) == 0);
    writer.write("hello");
    writer.write(std::string_view(", world", 7));
    CHECK(!std::filesystem::exists(target));
    CHECK(std::filesystem::exists(part_path));
    CHECK(slurp(part_path) == "hello, world");
    writer.publish();
  }
  CHECK(std::filesystem::exists(target));
  CHECK(!std::filesystem::exists(part_path));
  CHECK(slurp(target) == "hello, world");

  // An existing final path must never be replaced, and the failed writer must
  // remove its own private part.
  std::string rejected_part;
  {
    detail::PartWriter writer(target);
    rejected_part = writer.part_path();
    writer.write("replacement");
    expect_error([&] { writer.publish(); }, "publish over an existing file");
    CHECK(std::filesystem::exists(rejected_part));
  }
  CHECK(slurp(target) == "hello, world");
  CHECK(!std::filesystem::exists(rejected_part));

  // An unpublished part is removed without creating the final path.
  std::string abandoned;
  {
    detail::PartWriter writer(dir.path + "/other.bin");
    abandoned = writer.part_path();
    writer.write("partial");
  }
  CHECK(!std::filesystem::exists(abandoned));
  CHECK(!std::filesystem::exists(dir.path + "/other.bin"));

  expect_error([&] { detail::PartWriter writer(dir.path + "/"); }, "directory target");
}

void test_request_validation() {
  detail::HttpRequest get;
  get.url = "http://127.0.0.1:1/x";
  get.method = detail::HttpMethod::kGet;
  detail::validate_request(get);

  get.body = "x";
  expect_error([&] { detail::validate_request(get); }, "GET body");
  get.body.clear();

  detail::MimePart text;
  text.name = "field";
  text.data = "value";
  get.mime_parts.push_back(text);
  expect_error([&] { detail::validate_request(get); }, "GET multipart");
  get.mime_parts.clear();

  detail::HttpRequest del = get;
  del.method = detail::HttpMethod::kDelete;
  del.body = "x";
  detail::validate_request(del);
  del.mime_parts.push_back(text);
  expect_error([&] { detail::validate_request(del); }, "DELETE multipart");

  detail::HttpRequest post;
  post.url = "http://127.0.0.1:1/x";
  post.stream_limit = 10;
  expect_error([&] { detail::validate_request(post); }, "limit without callback");
  post.on_body = [](std::string_view) { return true; };
  detail::validate_request(post);  // a callback makes stream_limit valid
  post.stream_limit = 0;

  detail::MimePart mixed;
  mixed.name = "file";
  mixed.data = "x";
  mixed.read = [](char*, size_t) { return size_t{0}; };
  mixed.size = 1;
  post.mime_parts.push_back(mixed);
  expect_error([&] { detail::validate_request(post); }, "data and callback");
  post.mime_parts.clear();

  detail::MimePart unnamed;
  unnamed.data = "x";
  post.mime_parts.push_back(unnamed);
  expect_error([&] { detail::validate_request(post); }, "unnamed part");
  post.mime_parts.clear();

  detail::MimePart oversized;
  oversized.name = "file";
  oversized.read = [](char*, size_t) { return size_t{0}; };
  oversized.size = std::numeric_limits<uint64_t>::max();
  post.mime_parts.push_back(oversized);
  expect_error([&] { detail::validate_request(post); }, "oversized part");
  post.mime_parts.clear();

  detail::MimePart streamed;
  streamed.name = "file";
  streamed.read = [](char*, size_t) { return size_t{0}; };
  streamed.size = 3;
  post.mime_parts.push_back(streamed);
  detail::validate_request(post);  // a valid streamed part passes

  detail::HttpRequest empty;
  expect_error([&] { detail::validate_request(empty); }, "missing URL");
}

void test_content_type() {
  CHECK(detail::content_type_matches("application/octet-stream",
                                     "application/octet-stream"));
  CHECK(detail::content_type_matches("Application/Octet-Stream; charset=binary",
                                     "application/octet-stream"));
  CHECK(detail::content_type_matches("audio/wav", "AUDIO/WAV"));
  CHECK(!detail::content_type_matches("text/html", "application/octet-stream"));
  CHECK(!detail::content_type_matches("application/octet-stream", ""));
}

}  // namespace

int main() {
  test_base64();
  test_wav();
  test_mime_source();
  test_part_writer();
  test_request_validation();
  test_content_type();

  if (failures != 0) {
    std::cerr << failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "all transport checks passed\n";
  return 0;
}

// SPDX-License-Identifier: Apache-2.0
#include "umminfer/audio/audio.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace umminfer::audio {
namespace {

constexpr uint32_t kSampleRate = 16000;
constexpr size_t kWavHeaderSize = 44;

[[noreturn]] void invalid_wav() {
  throw std::invalid_argument("WAV must contain one PCM16 mono 16 kHz data chunk");
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

void append_u16(std::string& bytes, uint16_t value) {
  bytes.push_back(static_cast<char>(value & 0xff));
  bytes.push_back(static_cast<char>((value >> 8) & 0xff));
}

void append_u32(std::string& bytes, uint32_t value) {
  bytes.push_back(static_cast<char>(value & 0xff));
  bytes.push_back(static_cast<char>((value >> 8) & 0xff));
  bytes.push_back(static_cast<char>((value >> 16) & 0xff));
  bytes.push_back(static_cast<char>((value >> 24) & 0xff));
}

}  // namespace

std::string wav_from_pcm16(std::string_view pcm) {
  return wav_from_pcm16(pcm, kSampleRate);
}

std::string wav_from_pcm16(std::string_view pcm, uint32_t sample_rate) {
  if (pcm.size() % 2 != 0) {
    throw std::invalid_argument("PCM16 audio must contain whole samples");
  }
  // The fmt chunk stores sample rate and byte rate as uint32 values. Reject a
  // rate whose byte rate cannot be represented instead of truncating it.
  constexpr uint32_t max_sample_rate = std::numeric_limits<uint32_t>::max() / 2u;
  if (sample_rate == 0 || sample_rate > max_sample_rate) {
    throw std::invalid_argument("WAV sample rate is out of range");
  }
  constexpr size_t max_riff_payload = std::numeric_limits<uint32_t>::max() - 36u;
  std::string wav;
  if (pcm.size() > max_riff_payload || pcm.size() > wav.max_size() - kWavHeaderSize) {
    throw std::invalid_argument("PCM audio is too large for WAV");
  }

  wav.reserve(kWavHeaderSize + pcm.size());
  wav.append("RIFF", 4);
  append_u32(wav, static_cast<uint32_t>(36 + pcm.size()));
  wav.append("WAVEfmt ", 8);
  append_u32(wav, 16);
  append_u16(wav, 1);
  append_u16(wav, 1);
  append_u32(wav, sample_rate);
  append_u32(wav, sample_rate * 2u);
  append_u16(wav, 2);
  append_u16(wav, 16);
  wav.append("data", 4);
  append_u32(wav, static_cast<uint32_t>(pcm.size()));
  wav.append(pcm);
  return wav;
}

std::string pcm16_from_wav(std::string_view wav) {
  if (wav.size() < 12 || wav.substr(0, 4) != "RIFF" || wav.substr(8, 4) != "WAVE") {
    invalid_wav();
  }
  const uint32_t riff_size = read_u32(wav, 4);
  if (riff_size < 4 || static_cast<uint64_t>(riff_size) + 8 != wav.size()) invalid_wav();
  const size_t riff_end = static_cast<size_t>(riff_size) + 8;

  bool saw_format = false;
  bool saw_data = false;
  size_t data_offset = 0;
  size_t data_size = 0;
  size_t offset = 12;
  while (offset < riff_end) {
    if (riff_end - offset < 8) invalid_wav();
    const std::string_view chunk_id = wav.substr(offset, 4);
    const uint32_t chunk_size = read_u32(wav, offset + 4);
    const size_t body_offset = offset + 8;
    if (chunk_size > riff_end - body_offset) invalid_wav();
    const size_t body_end = body_offset + static_cast<size_t>(chunk_size);

    if (chunk_id == "fmt ") {
      if (saw_format || chunk_size < 16) invalid_wav();
      const bool valid_format = read_u16(wav, body_offset) == 1 &&
                                read_u16(wav, body_offset + 2) == 1 &&
                                read_u32(wav, body_offset + 4) == kSampleRate &&
                                read_u32(wav, body_offset + 8) == kSampleRate * 2 &&
                                read_u16(wav, body_offset + 12) == 2 &&
                                read_u16(wav, body_offset + 14) == 16;
      if (!valid_format) invalid_wav();
      saw_format = true;
    } else if (chunk_id == "data") {
      if (!saw_format || saw_data || chunk_size % 2 != 0) invalid_wav();
      saw_data = true;
      data_offset = body_offset;
      data_size = chunk_size;
    }

    offset = body_end;
    if (chunk_size % 2 != 0) {
      if (offset == riff_end) invalid_wav();
      ++offset;
    }
  }
  if (!saw_format || !saw_data) invalid_wav();
  return std::string(wav.substr(data_offset, data_size));
}

}  // namespace umminfer::audio

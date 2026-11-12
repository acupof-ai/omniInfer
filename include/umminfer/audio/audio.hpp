// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace umminfer::audio {

// Headerless PCM16 mono sample bytes. wav_from_pcm16() writes 16 kHz for the
// existing ASR path; the overload accepts 1 to 2147483647 Hz.
// Invalid or unsupported input throws std::invalid_argument.
std::string wav_from_pcm16(std::string_view pcm);
std::string wav_from_pcm16(std::string_view pcm, uint32_t sample_rate);
std::string pcm16_from_wav(std::string_view wav);

}  // namespace umminfer::audio

# Roadmap

## Current deliverables

- C++20 control plane with Coordinator, ZeroMQ messages, request aborts,
  ready-gated startup, and C++ stage scheduling.
- C++ stage-to-stage POSIX shared-memory relay. Optional CUDA IPC is available
  as a build path.
- External persistent MLX-LM text stage on Apple Silicon with JSON and SSE.
- Deterministic TinyCTC external stage for real inference-path audio tests.
- Chat and transcription HTTP endpoints, multipart upload, private realtime
  WebSocket protocol, and optional bearer authentication.
- C++ chat and ASR clients and CLI. Chat supports JSON, SSE, retries, timeouts,
  and cancellation. ASR supports multipart, realtime audio, timeouts,
  cancellation, live-result selection, and HTTP Final fallback.
- WAV/PCM16 conversion in the small `umminfer::audio` module, with thin client
  compatibility wrappers.
- Real server/stage/model validation for chat, ASR, authentication, disconnects,
  and subsequent requests.

TinyCTC is test infrastructure. The current deliverable does not include a
trained production ASR model.

## Model support plan

These entries are not current deliverables.

| Model | Status | Target and constraints |
| --- | --- | --- |
| `Qwen3.5-Omni` | **PLANNED** | Only the authorized official API is planned. Verified IDs are `qwen3.5-omni-plus` and `qwen3.5-omni-flash`. They accept text, image, audio, and video input and return text or streaming speech. Do not claim image or video generation. Authorized access is unverified; all API execution cases remain `NOT_COVERED`. |
| `MiniMaxAI/MiniMax-H3` | **IN_PROGRESS — integration planning** | Plan a full-stack BF16 baseline on H20B. The local open H3-Base path is limited to 768p. Its contract covers text, image, video, and audio input, with 768p video and native 32 kHz stereo audio output. |
| `MiniMaxAI/MiniMax-H3-Image` | **IN_PROGRESS — integration planning** | Plan a full-stack BF16 baseline on H20B. Cover text-to-image generation and editing with up to 16 reference images. |

BF16 acceptance covers the complete stack: DiT, complete text encoders, VAE,
activations, offload, and host RAM. Official Hugging Face metadata for the
original H3 FL2VA partition is approximately 144.05 GB. The H3-Image selective
layout for SGLang/vLLM is approximately 135.38 GB. These disk sizes are not VRAM
requirements.

For H3, 2K depends on the hosted Context-IR/Regenerate-2K modules, which are
not fully open. Do not claim a fully local 2K path.

There are no official quantized checkpoints for H3 or H3-Image. Official weights
with runtime online quantization are distinct from third-party converted or
pruned quantized packages. Verify a third-party package's revision, license,
tensor completeness, hashes, and real outputs. Do not call it official MiniMax
quantization. Without a trusted artifact or compatible hardware, mark the case
`BLOCKED` or `NOT_COVERED`.

Quantized tests may run before the H20B baseline. BF16 remains required for
high-precision acceptance and later quality/resource comparisons. Record the
quantization method and precision separately for each run.

There is no H3/H3-Image-specific Mac validation, memory guarantee, or
performance commitment. Generic MPS or offload examples do not establish model
acceptance. Mac runs remain limited to low-resource real-model tests. A model
that cannot run within those limits remains `NOT_COVERED` on Mac. Do not replace
the real model with a simulator.

Final generation acceptance must use the CLI or public API, ummInfer server,
real backend or authorized official API, real model, and decoded artifact. A
direct SGLang run that bypasses ummInfer is not final acceptance. Unimplemented
integrations remain **PLANNED** or **IN_PROGRESS — integration planning**.

## Next priorities

1. Validate H3 and H3-Image through ummInfer with trusted quantized real-model
   artifacts when available. Establish full-stack BF16 baselines on H20B for
   high-precision acceptance and quality/resource comparisons.
2. Add a trained ASR backend with resampling or an explicit production audio
   format contract. Validate multilingual and longer-form speech separately.
3. Add production model execution behind the C++ compute interface if that
   deployment path is required. Keep the external Python protocol independent
   of C++ `Stage`.
4. Add secure server-key injection that does not expose the key in process
   arguments. Add remote `wss://` realtime support before exposing realtime
   outside loopback.
5. Add model usage accounting, dynamic batching, and supported request fields
   only with real backend behavior and end-to-end tests.
6. Run CUDA build, IPC, stress, sanitizer, and model tests on supported CUDA
   hardware before claiming a verified CUDA deliverable.
7. Consolidate real client regression entry points and retain byte-proxy tests
   for network failure conditions.

## Explicitly unverified or unsupported

- CUDA execution, CUDA IPC, CUDA stress, and compute-sanitizer results on this
  machine.
- Production speech recognition quality. TinyCTC has deterministic initialized
  weights and no speech training.
- OpenAI Realtime API compatibility. The implemented realtime protocol is
  private and returns one aggregated result.
- Natural WebSocket control-frame interleaving in the continuation regression.
  No stable real-client trigger exists without injecting frames.
- A first-party HTTP 429 with `Retry-After` from this server. The server does
  not emit one. Retry timing is tested through constrained real-byte network
  conditions and lower-level checks instead.
- Immediate cancellation inside an MLX model call. Abort takes effect at token
  boundaries.
- Mixed Python/C++ shared-memory payload transfer and Python pickle payload
  headers.
- Multimodal chat, tools, JSON response format, text-to-speech, and dynamic
  batching.
- Model-backed execution in the C++ stage host. Its executable uses echo
  compute.
- Unified final integration acceptance across all participants' changes. The
  component results recorded during development do not replace that acceptance.

# ummInfer

ummInfer is an inference runtime for **Unified Multimodal Models (UMM)**.

The target is one engine that serves vision-language (VLM) and speech (ASR)
workloads together: shared sessions, one orchestration path, streaming events,
cancellation, backpressure, error propagation, and output coordination. Models
run in separate worker processes behind that one control plane.

The engine foundation is implemented and tested for text chat and for a
deterministic audio test pipeline. The unified VLM+ASR capability is **not
implemented**. This README separates the two.

## Status

| Layer | State |
| --- | --- |
| C++20 control plane: Coordinator, request abort, ready gating | Implemented; default tests pass |
| ZeroMQ transport with MessagePack control messages | Implemented; default tests pass |
| POSIX shared-memory payload relay with bounded credits | Implemented; default tests pass |
| Optional CUDA IPC relay path | Build path present; unbuilt and unexecuted here (no CUDA toolkit) |
| External persistent Python stage protocol | Implemented; used by the two stages below |
| MLX-LM text chat, JSON and SSE | Implemented; checked against a real local model |
| ASR pipeline | Implemented against the deterministic TinyCTC test model only |
| HTTP, SSE, WebSocket server; CLI; C++ clients | Implemented |
| Native macOS client `clients/ummmlx` | Implemented; text chat only |
| Trained ASR model | Not implemented |
| Real VLM worker | Not implemented |
| Cross-model orchestration (for example ASR to VLM) | Not implemented |
| Shared session across modalities | Not implemented |
| Unified multimodal end-to-end acceptance | Not run |
| H20B verification | BLOCKED; jump host authentication fails |

TinyCTC is test infrastructure. It has deterministic initialized weights, and
it does not recognize speech. The MLX-LM chat path is text-only inference. It is
not a VLM.

Image and video are optional modalities. They attach as separate workers when a
use case needs them, and they do not block the VLM+ASR main path. Today no local
image or video worker exists; the only image and video routes are the
fixed-upstream proxy described below.

See [roadmap](docs/roadmap.md) for the phase plan and acceptance gates, and
[validation](docs/validation.md) for the user journeys that exist today.

Verification has two profiles. The Apple Silicon Mac runs low-consumption
control-plane, stage, and small real-model checks. NVIDIA H20B is for
high-precision acceptance. H20B access currently fails at the jump host
authentication step, so every H20B result is `BLOCKED` / `NOT_RUN`; none is
verified. The split and its gates are in
[roadmap](docs/roadmap.md#verification-split).

## What you can do

| Task | Interface | Backend |
| --- | --- | --- |
| Generate text once | `chat`, `/v1/chat/completions` JSON | External MLX-LM stage |
| Stream generated text | `chat --stream`, chat SSE | External MLX-LM stage |
| Transcribe a WAV or PCM file | `transcribe`, multipart HTTP | TinyCTC test stage |
| Send audio over WebSocket | `realtime`, `/v1/realtime` | TinyCTC test stage |
| Use a bearer key | Environment variables in the client | Optional server authentication |
| Generate or edit an image, create a video | `image`, `video`, `/v1/images/*`, `/v1/videos*` | Fixed upstream only |

The server exposes `/health`, `/ready`, `/v1/models`, and `/abort`.
`/v1/audio/speech` returns HTTP 501.

A second, opt-in server mode forwards one model family to one configured
upstream instead of running a local stack. It adds image, video, and
omni-modality routes. It is a proxy: it does not implement a VLM or a video
model inside ummInfer. See [fixed upstream mode](#fixed-upstream-proxy-mode).
## What is not implemented

- A VLM worker. Local chat accepts text content only and rejects content part
  arrays with HTTP 400.
- A trained ASR model. Only the deterministic TinyCTC stage exists.
- Cross-model orchestration. Chat, transcription, and realtime are separate
  request paths. No request runs two model families in one plan.
- A shared session object across modalities. The realtime WebSocket session is
  private to one audio utterance.
- A unified cancellation, backpressure, and error-propagation contract spanning
  several workers. Abort exists per request. Shared-memory relay credits bound
  one C++ payload edge. Neither is a multi-worker contract.
- Unified multimodal end-to-end acceptance across participants' changes.
  Component results recorded during development do not replace it.

## Requirements and limits

- Text models must be MLX-LM models with a compatible text chat template.
- The provided ASR model is TinyCTC with deterministic initialized weights.
  TinyCTC tests inference and transport. It is not a trained speech recognizer.
- TinyCTC accepts headerless little-endian PCM16, mono, 16 kHz. WAV input must
  contain exactly that format. It accepts 400 samples to 30 seconds of audio.
- Do not pass stereo, compressed audio, another sample rate, or a WAV with
  multiple data chunks. The client rejects unsupported WAV structures.
- Realtime is a private protocol for one aggregated transcription. It is not the
  OpenAI Realtime API. The bundled server sends no partial preview: after the
  stage returns it sends one final `transcription.segment`, then
  `transcription.completed`. Clients can assemble several segments, but this
  server never produces them.
- Chat accepts plain HTTP only on loopback. HTTPS may use a remote host.
- ASR HTTP and HTTPS base URLs, including `--final-url`, are loopback-only.
- Realtime `--url` accepts only loopback `http://` and converts it to `ws://`
  internally. Remote `wss://` is not supported.
- By default, realtime `finish()` sends no `transcription.done`. It closes the
  WebSocket and sends retained PCM through HTTP Final. `--live-result` sends
  `transcription.done` and requests the live completed result. If that result
  fails or exceeds `--final-timeout-ms`, HTTP Final still runs.
- The realtime client retains at most 1 GiB of audio. The bundled WebSocket
  server accumulates at most 64 MiB per session and rejects a larger append.
- Chat supports retries before a stream emits a delta. ASR and realtime do not
  retry inference. Chat supports total, first-token, and stall timeouts. HTTP
  transcription supports total and activity timeouts. Realtime supports setup,
  write, close, and live-result timeouts. Cancellation interrupts transport;
  MLX abort takes effect at token boundaries.
- One external MLX stage processes one request at a time. Other requests queue.
- The C++ stage host contains reference echo compute only. It does not load a
  model. CUDA support is build-time functionality and has not been tested here.
- On the local path, chat message content must be text. Multimodal content
  arrays, tools, `top_p`, `seed`, non-empty `stop`, usage in `stream_options`,
  and JSON response format are rejected by this server.
- Transcription accepts `model`, `stream`, `max_new_tokens`,
  `include_generation_metadata`, `response_format=json`, and `temperature=0`.
  Other response formats and nonzero temperatures return HTTP 400. `language`,
  `prompt`, `audio_layout`, `stop_at_end_of_text`, and `stop_on_token_loop` are
  explicitly rejected by the current backend.
- The control plane includes decode checks for Python-produced MessagePack
  messages. Map keys must be strings. Integers must fit in signed 64 bits.
- Shared-memory `StagePayload` transfer is supported between C++ stages.
  Python-to-C++ shared-memory payload edges are not supported; external Python
  stages use ZeroMQ messages with inline data.
- CUDA IPC carries host-serialized `StagePayload` bytes. It is not zero-copy
  device-tensor transfer and it does not execute a model. It is selected only
  when both stages declare GPU ids on one host, peer access works in both
  directions, and the packed payload is at most `slot_size_mb` (default 512).
- A receiver rejects a `DataRef` above `max_payload_bytes` (default 1 GiB)
  before any allocation. Pending relay acknowledgements time out after 30
  seconds by default (`transfer_timeout_ms`); a timed-out transfer abandons
  its CUDA edge and other targets keep working.

See [architecture](docs/external_stage.md) and [user validation](docs/validation.md).

## Install and build

Use Zig 0.16.0. On macOS with Homebrew:

```sh
brew install zig libzmq msgpack cpp-httplib nlohmann-json curl libwebsockets
python3 -m venv .venv
.venv/bin/python -m pip install mlx-lm pyzmq msgpack websockets
zig build -Doptimize=ReleaseFast -Ddeps-prefix=/opt/homebrew \
  -Dcurl-prefix=/opt/homebrew/opt/curl
zig build test -Ddeps-prefix=/opt/homebrew -Dcurl-prefix=/opt/homebrew/opt/curl
```

The binary is `zig-out/bin/umm-infer`. Apple's system libcurl 8.7.1 lacks the
WebSocket support required by the realtime client. Use `-Ddeps-prefix` when
dependencies are outside the system include and library paths.

`test` runs wire, relay, scheduler, coordinator, client, fixed-upstream proxy,
and two-stage pipeline tests. It builds and runs no model. On this machine it
exits 0: `all checks passed`, `all transport checks passed`, `all serve proxy
checks passed`, `all 82 client checks passed`.

`smoke` checks HTTP validation, authentication, JSON, SSE, 24 concurrent JSON
requests, 12 concurrent SSE requests, external stages, and SIGTERM shutdown.

The finite request-lifecycle model has a TLA check. It does not prove that the
C++ implementation conforms to the model:

```sh
TLA2TOOLS_JAR=/path/to/tla2tools.jar python3 -I tests/tla_test.py
```

The CPU build is the default. CUDA IPC is an optional build:

```sh
zig build -Dcuda -Dcuda-prefix=/usr/local/cuda -Ddeps-prefix=/opt/homebrew
zig build test -Dcuda -Dcuda-prefix=/usr/local/cuda -Ddeps-prefix=/opt/homebrew
```

`-Dcuda-prefix` must contain `include/` and `lib64/`. `-Dcuda` alone uses the
system include and library paths. With `-Dcuda`, `test` also builds and runs
`cuda_ipc_test` and `cuda_pipeline_test`. Both skip themselves in a CPU build.

Three CUDA-only targets stay out of the default test run:

```sh
# Repeated rounds, stops at the first failure.
zig build cuda-stress -Dcuda -Dcuda-prefix=/usr/local/cuda -Ddeps-prefix=/opt/homebrew

# Two-process latency for SHM and CUDA IPC; ReleaseFast is required.
zig build cuda-bench -Dcuda -Dcuda-prefix=/usr/local/cuda -Ddeps-prefix=/opt/homebrew -Doptimize=ReleaseFast

# compute-sanitizer memcheck over both test binaries.
python3 tests/cuda_sanitize.py --ipc zig-out/bin/cuda_ipc_test \
  --pipeline zig-out/bin/cuda_pipeline_test --log-dir /tmp/cuda-sanitize
```

## Run text generation

Use a local MLX model directory, or a Hugging Face model id. The first use of an
id may download model files. Use separate terminals for the server and stage.

```sh
MODEL=/absolute/path/to/mlx-model
IPC_DIR=$(mktemp -d /tmp/umminfer-mlx.XXXXXX)

./zig-out/bin/umm-infer serve --host 127.0.0.1 --port 30000 \
  --model "$MODEL" --entry mlx --entry-endpoint "ipc://$IPC_DIR/entry" \
  --completion "ipc://$IPC_DIR/completion" --abort "ipc://$IPC_DIR/abort"
```

```sh
.venv/bin/python -I python/umminfer_mlx_stage.py \
  --name mlx --recv "ipc://$IPC_DIR/entry" \
  --coordinator "ipc://$IPC_DIR/completion" --abort "ipc://$IPC_DIR/abort" \
  --model "$MODEL"
```

The server queues requests until the stage announces readiness.

```sh
./zig-out/bin/umm-infer chat --url http://127.0.0.1:30000/v1 \
  --model "$MODEL" --prompt 'Give one inference tip.' \
  --max-tokens 64 --temperature 0

./zig-out/bin/umm-infer chat --url http://127.0.0.1:30000/v1 \
  --model "$MODEL" --prompt 'Count to three.' --stream --max-tokens 64
```

The client reads `OPENAI_BASE_URL` and `OPENAI_MODEL` when the corresponding
options are absent. With no `--prompt`, it reads the prompt from standard input.

### Measured stage result

The frozen comparison baseline already caches the tokenizer vocabulary map. The
current stage adds two changes on top of it: a reusable prompt KV snapshot with
a reused detokenizer, and one fused quantized QKV projection per layer instead
of three projections. The fusion applies to quantized Qwen2-architecture models
whose Q, K, and V projections share one quantization layout; on another layout
it is skipped.

Under the project's fixed comparison convention, the recorded result for the
current stage against that baseline is a **61.93% warm speedup on median total
latency**:

| Item | Value |
| --- | --- |
| Metric | warm median `total_ms`, 63.75 ms to 39.37 ms |
| Formula | `(63.75 / 39.37 - 1) * 100`, a ratio of medians |
| Same change as latency reduction | 38.25% fewer milliseconds |
| Configuration | `http_sse`, `max_tokens=64`, 20 interleaved pairs, fresh process per sample, 2 warmups per process, `temperature=0`, Qwen2.5-0.5B-Instruct-4bit |
| Output identity | One output SHA256 across both arms; finish reason `stop` on all 20 pairs |
| Recompute | A second script recomputed the stored medians and matched |

The fused QKV step alone is smaller. In the direct in-process harness, the arm
that already has the vocabulary cache and prompt KV measured 3.92% from adding
the fusion (391.05 to 406.39 output tokens per second). Do not attribute the
61.93% to the fusion.

Read the number with its conditions:

- It is a ratio of medians, not a median of per-pair ratios. The median
  per-pair speedup is 51.11%, and 2 of the 20 pairs were slower with the
  candidate arm.
- Every recorded `cv` is above the 0.10 quiet-window limit: warm `total_ms`
  0.167 baseline and 0.287 candidate; warm `ttft_ms` 0.328 and 2.735. The run
  itself printed four `cv` warnings. By the project rule this is one recorded
  run that asks for a repeat, not a validated result.
- The token counters are absent, not estimated. The frozen binary's chat reply
  carries no `usage`, `generation_tokens`, or `prompt_tokens`, and peak memory
  is unobservable without `--perf-log`. The report records those gaps.
- The report did not pass the `tests/mlx_perf_compare.py` gate set; it has a
  recompute check only.
- Both the HTTP A/B harness and its report are local artifacts. Neither is
  committed here, so the number cannot be re-derived from a clean checkout
  alone.
- The in-repository harnesses for the same stage are
  `tests/mlx_prompt_cache_perf.py` (direct MLX-LM A/B) and `tests/mlx_perf.py`
  (general comparison). Their gates are in [MLX backend](docs/mlx.md).

## Run the audio test pipeline

Generate a new TinyCTC model directory, then use separate terminals:

```sh
ASR_MODEL=$(mktemp -d /tmp/umminfer-tiny-model.XXXXXX)/model
.venv/bin/python -I python/umminfer_tiny_asr_stage.py generate --output "$ASR_MODEL"
ASR_IPC=$(mktemp -d /tmp/umminfer-asr.XXXXXX)

./zig-out/bin/umm-infer serve --host 127.0.0.1 --port 31000 \
  --realtime-port 31001 --model tiny-ctc --entry asr \
  --entry-endpoint "ipc://$ASR_IPC/entry" \
  --completion "ipc://$ASR_IPC/completion" --abort "ipc://$ASR_IPC/abort"
```

```sh
.venv/bin/python -I python/umminfer_tiny_asr_stage.py serve \
  --name asr --recv "ipc://$ASR_IPC/entry" \
  --coordinator "ipc://$ASR_IPC/completion" --abort "ipc://$ASR_IPC/abort" \
  --model "$ASR_MODEL"
```

Use an existing 16 kHz mono PCM16 WAV file. Port 31001 carries the WebSocket
session. `--final-url` sends the default or fallback HTTP Final to port 31000.

```sh
./zig-out/bin/umm-infer transcribe --url http://127.0.0.1:31000 \
  --model tiny-ctc --file audio.wav --max-new-tokens 64

./zig-out/bin/umm-infer realtime --url http://127.0.0.1:31001 --final-url http://127.0.0.1:31000 \
  --model tiny-ctc --file audio.wav --max-new-tokens 64

# Accept the live result, with HTTP Final retained as fallback.
./zig-out/bin/umm-infer realtime --url http://127.0.0.1:31001 --final-url http://127.0.0.1:31000 \
  --model tiny-ctc --file audio.wav --live-result --final-timeout-ms 20000
```

For raw PCM, pass `--format pcm`. A file ending in `.wav` is decoded as WAV;
other file names default to raw PCM. `--file -` reads bytes from standard input.

## In-process demo and C++ stages

The demo runs the echo compute path without an external stage:

```sh
./zig-out/bin/umm-infer serve --demo --port 30000 --ipc-dir /tmp/omni-demo

curl -s localhost:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"omni","messages":[{"role":"user","content":"hi"}]}'
```

A coordinator and two external C++ stages run in separate terminals:

```sh
./zig-out/bin/umm-infer serve --port 30000 --entry entry \
  --entry-endpoint ipc:///tmp/omni/entry \
  --completion ipc:///tmp/omni/completion --abort ipc:///tmp/omni/abort

./zig-out/bin/umm-infer stage --name entry --recv ipc:///tmp/omni/entry \
  --coordinator ipc:///tmp/omni/completion --abort ipc:///tmp/omni/abort \
  --next worker=ipc:///tmp/omni/worker

./zig-out/bin/umm-infer stage --name worker --recv ipc:///tmp/omni/worker \
  --coordinator ipc:///tmp/omni/completion --abort ipc:///tmp/omni/abort \
  --peer entry=ipc:///tmp/omni/entry
```

The downstream `--peer entry=...` endpoint carries relay acknowledgements to
the upstream stage. Without it, upstream transfer credits are not released.
ZMQ does not create the parent directory of an `ipc://` path; the launchers
create it and remove stale socket files for configured endpoints.

### CUDA IPC stages

Two stages on one host can move payloads through CUDA IPC. Declare each stage's
device, and give the sender the device of the stage it sends to:

```sh
./zig-out/bin/umm-infer stage --name entry --gpu-id 0 \
  --recv ipc:///tmp/omni/entry --coordinator ipc:///tmp/omni/completion \
  --abort ipc:///tmp/omni/abort --next worker=ipc:///tmp/omni/worker \
  --peer worker=ipc:///tmp/omni/worker --peer-gpu worker=1

./zig-out/bin/umm-infer stage --name worker --gpu-id 1 \
  --recv ipc:///tmp/omni/worker --coordinator ipc:///tmp/omni/completion \
  --abort ipc:///tmp/omni/abort --peer entry=ipc:///tmp/omni/entry
```

- `--gpu-id N` gives the stage its CUDA device.
- `--peer-gpu name=N` gives the device of a peer stage on the same host.
- The sender checks peer access in both directions before selecting CUDA IPC.
- Two stages on the same device use CUDA IPC without peer access.
- If peer access is unavailable, the payload exceeds the slot limit, or
  `UMMINFER_DISABLE_CUDA_IPC=1` is set, the edge uses POSIX shared memory.
  The fallback is logged once per edge.

None of these commands has been executed on this machine, which has no CUDA
toolkit: `zig build -Dcuda` stops here when `cuda_runtime_api.h` is not found.
The relay code has default-test coverage only; CUDA execution is unverified.

## Fixed upstream proxy mode

`serve` can forward requests to one external model server instead of running a
local stage. This mode is separate from the engine foundation. It adds no local
model, no local orchestration, and no local session.

```sh
# Official compatible API, omni variant.
./zig-out/bin/umm-infer serve --port 30000 --model MY-MODEL \
  --upstream-kind qwen --upstream-base-url https://example.invalid/v1 \
  --upstream-model MODEL-ID --upstream-variant omni

# SGLang server, image or video variant.
./zig-out/bin/umm-infer serve --port 30000 --model MY-MODEL \
  --upstream-kind sglang --upstream-base-url http://127.0.0.1:30010 \
  --upstream-model MODEL-ID --upstream-variant image
```

Rules of this mode:

- One process serves one upstream, one model, and one variant. There is no
  routing or fallback between upstreams.
- The route set follows the variant. `qwen` serves only
  `POST /v1/chat/completions`. `sglang` with `image` serves
  `POST /v1/images/generations` and `POST /v1/images/edits`. `sglang` with
  `video` serves `POST /v1/videos`, `GET /v1/videos/{id}`,
  `GET /v1/videos/{id}/content`, and `DELETE /v1/videos/{id}`.
- Every other media route answers HTTP 501. A server with no upstream flags also
  answers 501 on all image and video routes.
- The upstream credential comes from `--upstream-api-key` or
  `UMMINFER_UPSTREAM_API_KEY`. It is sent only to the upstream. It must differ
  from `--api-key`.
- The request body's `model` field is replaced with `--upstream-model` before
  forwarding.
- `/abort` answers HTTP 501. A fixed upstream is not a coordinator job.
- Remote plain HTTP is rejected. A plain-HTTP upstream must be loopback.
- Requests bypass the Coordinator submission path, C++ `Stage`, and the relay.
  The process still constructs a Coordinator, but no proxy request uses it.
- `/ready` answers 200 immediately when no entry stage is configured.

The CLI multimodal flags need this mode. `chat --image-url`, `--video-url`,
`--audio`, and `--audio-out` send content parts that the local MLX path rejects;
they work only against an `omni` upstream. `image generate`, `image edit`, and
the `video` subcommands need an `image` or `video` upstream.

The default test suite checks configuration validation, route gating, error
mapping, and credential scrubbing against a real local server with an
unreachable upstream. It does not prove an upstream success. Live scripts
`tests/qwen_omni_live.py`, `tests/h3_live.py`, and `tests/h3_image_live.py`
require `UMMINFER_LIVE_OPT_IN=1` and provider credentials. Without them they
print `BLOCKED`/`NOT_COVERED` and make no network request. No live upstream run
is recorded for this revision.

## Authentication

The server enables authentication with `serve --api-key KEY`. It protects HTTP,
health, readiness, and WebSocket upgrade requests. Start it on a private
interface. The server key option can be visible in the local process list.

Do not put client keys in command-line arguments. The client reads
`UMMINFER_API_KEY`, then `OPENAI_API_KEY`:

```sh
export UMMINFER_API_KEY='your-key'
./zig-out/bin/umm-infer transcribe --url http://127.0.0.1:31000 \
  --model tiny-ctc --file audio.wav
./zig-out/bin/umm-infer realtime --url http://127.0.0.1:31001 --final-url http://127.0.0.1:31000 \
  --model tiny-ctc --file audio.wav
```

Realtime WebSocket and its HTTP Final use the same client key.

## C++ API

Include `umminfer/client/client.hpp`. `OpenAIClient::chat` provides JSON and SSE
chat. `ASRClient::transcribe` converts PCM to WAV and uploads it as a multipart
`file` part; it always sets `stream=true` and parses the SSE reply.
`ASRClient::realtime` creates a session with `send`, `finish`, and `close`.
Requests accept cancellation flags.

`umminfer/audio/audio.hpp` provides `wav_from_pcm16` and `pcm16_from_wav`.
The client namespace retains thin wrappers that convert invalid audio errors to
`ErrorCode::kBadRequest`.

## Runtime components

```text
External model path:
CLI/public API -> HTTP or WebSocket -> Coordinator -> ZeroMQ
  -> external Python MLX/TinyASR stage -> model

C++ stage-host path:
C++ Stage -> SimpleScheduler -> compute function

Fixed upstream path:
CLI/public API -> HTTP -> one configured upstream (no Coordinator use)

C++ stage-to-stage payload transfer uses POSIX shared memory, or optional CUDA
IPC where its requirements hold. The external Python path does not use C++
Stage, SimpleScheduler, or the C++ shared-memory relay.
```

## Desktop client

`clients/ummmlx` is a native macOS SwiftUI client for a local MLX server. It
supports incremental SSE chat, Stop, New chat, and light/dark appearance. It
covers text chat only; voice, ASR, image, and video are `NOT_COVERED` there. See
[clients/ummmlx/README.md](clients/ummmlx/README.md).

## Layout

```text
build.zig
include/umminfer/{audio,client,proto,relay,comm,scheduling,pipeline,serve}
src/{audio,client,proto,relay,comm,scheduling,pipeline,serve}
src/main.cpp
python/{umminfer_mlx_stage.py,umminfer_tiny_asr_stage.py}
clients/ummmlx/
tests/self_check.cpp
tests/compat/{wire_compat.cpp,runtime_compat.cpp}
tests/{cuda_ipc_test.cpp,cuda_pipeline_test.cpp,cuda_bench.cpp}
tests/{omni_client_test.cpp,omni_serve_test.cpp,omni_transport_test.cpp}
tests/{cuda_stress.py,cuda_sanitize.py,http_smoke.py,tla_test.py}
tests/{client_smoke.py,client_retry_e2e.py,client_boundary_e2e.py}
tests/{realtime_contract_e2e.py,realtime_client_contract_e2e.py}
tests/{mlx_smoke.py,mlx_stage_test.py,mlx_*_perf.py,client_benchmark.py}
tests/{qwen_omni_live.py,h3_live.py,h3_image_live.py,live_common.py}
tests/tla/{RequestLifecycle.tla,RequestLifecycle.cfg}
docs/{cuda-ipc.md,external_stage.md,mlx.md,client.md,roadmap.md,validation.md}
```

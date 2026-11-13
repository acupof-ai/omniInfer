# User end-to-end validation

Use these journeys to validate the current source. The end-to-end checks below
start their own server, external Python stage, and real model; argument-parsing
checks exit before any process starts. No test below accepts a fake server or
handwritten model response as evidence.

## 1. Prepare the environment

Use an Apple Silicon Mac, Zig 0.16.0, the Homebrew dependencies from README,
and a Python environment with `mlx-lm`, `pyzmq`, `msgpack`, and `websockets`.

Set these paths for the automated checks:

```sh
export UMMINFER_BINARY="$PWD/zig-out/bin/umm-infer"
export MLX_MODEL=/absolute/path/to/local/mlx-model
```

Use a local MLX-LM model with a text chat template. The ASR checks generate a
new deterministic TinyCTC model in a temporary directory. TinyCTC checks do not
measure recognition quality.

Build the runtime:

```sh
zig build -Doptimize=ReleaseFast -Ddeps-prefix=/opt/homebrew \
  -Dcurl-prefix=/opt/homebrew/opt/curl
zig build test -Ddeps-prefix=/opt/homebrew -Dcurl-prefix=/opt/homebrew/opt/curl
```

Expected result: the build exits 0, all C++ checks pass, and the binary exists.
These checks cover wire format, relay, scheduler, coordinator, and C++ pipeline
behavior. They do not load a model.

`runtime_compat` also covers a real non-streaming 504. A request that stays
pending before `stage_ready`, with `request_timeout_s=0.2`, must reach the
public `OpenAIClient` as `kTimeout`, status 504, retryable, with the text
`HTTP 504: request timed out`. The case asserts a configured server wait of at
least 0.15 s and leaves `in_flight` at zero.

## 2. Complete the primary user journey

```sh
python3 -I tests/client_smoke.py "$UMMINFER_BINARY" --model "$MLX_MODEL"
```

Follow the `PASS` and `NOT_COVERED` lines. A zero process exit is required, but
also check that no `FAIL` or `BLOCKED` line exists. The script reaps only the
processes it owns.

| User action | Expected result | Real coverage |
| --- | --- | --- |
| Start chat server and model stage | Readiness changes after stage initialization | Server, Coordinator, ZeroMQ, MLX model |
| Send one chat request | Exit 0 and non-empty model text | Public client, HTTP, model |
| Stream the same prompt | Deltas assemble to the JSON result | SSE parsing and model streaming |
| Send concurrent requests | Four correct answers without cross-answer data | Persistent stage and request isolation |
| Transcribe valid audio | Exit 0 and text from the TinyCTC forward pass | WAV/PCM, multipart, ZeroMQ, model |
| Send two scalar `model` parts | HTTP 400 `duplicate multipart field: model`; `in_flight` returns to zero | Multipart parser, server route |
| Start realtime audio | Live preview and completed result work | Real WebSocket client and server |
| Finish through HTTP Final | Final multipart result is returned | Retained PCM and Final fallback |
| Enable authentication | Missing or wrong key fails; correct key succeeds | HTTP, health/readiness, WS, CLI, Final |
| Stop or kill a streaming client | Request is reaped; a later request succeeds | Disconnect recovery |
| Stop owned processes | Server and stages exit; no owned process remains | Cleanup |

The authentication matrix includes an accepted WebSocket session followed by
an authenticated HTTP Final request. A WebSocket handshake rejection is not
evidence for an HTTP Final rejection.

## 3. Exercise input and client boundaries

```sh
python3 -I tests/client_boundary_e2e.py "$UMMINFER_BINARY" --model "$MLX_MODEL"
python3 -I tests/realtime_client_contract_e2e.py "$UMMINFER_BINARY"
```

Expected results:

- Invalid CLI options return usage exit 2. Invalid realtime `--chunk-ms` values
  are rejected before any network connection.
- Invalid chat and transcription URLs fail before inference.
- Invalid WAV structures and odd PCM16 byte counts fail as bad requests.
- Unsupported server fields retain the server's error reason.
- A request whose generation metadata the real TinyCTC stage omits after a real
  forward returns HTTP 500 with the stage's reason. The transcribe CLI exits 1
  with empty stdout, and the real stage hook emits its omission marker once.
- A real HTTP 400 that echoes the request model under the bearer secret is
  redacted to `unsupported model: [key]`, and the secret does not appear.
- The public `ASRClient` maps a real TinyCTC short-frame failure to
  `kServer`/500/retryable with the stage reason, and a real invalid entry
  endpoint to `kServer`/503/retryable with the `zmq_connect` reason. After the
  500 case, valid audio still transcribes on the same server.
- Final realtime segments do not invoke the public `on_partial` callback.
- Missing optional generation metadata leaves token count null, length-limited
  state null, and finish reason empty in the public result.
- A large append is split into real messages of 1,048,575 and 151,519 bytes;
  the reconstructed PCM hash matches the original input.
- Active close and SIGKILL produce abort events, return `in_flight` to zero, and
  allow a following real request to succeed.
- Real continuation fragments are reassembled.
- A 20-round normal-close regression opens a real WebSocket session through
  the wire proxy and closes it. Every round must record both Close frames and
  no forwarded-error entry. ENOTCONN/EPIPE count as teardown; ECONNRESET counts
  only after an observed Close frame or EOF. Other transport errors and
  protocol assertions remain test failures. `tests/wire_proxy_teardown_test.py`
  pins that classifier without a server.
- `--ping-interval-ms` makes the real CLI send PING control frames while it
  streams audio. The wire proxy observes those frames and injects none. The
  stream keeps its order, ends with exactly one terminal event, and later
  requests succeed.
- `tests/realtime_streaming_live.py` drives the real Qwen3-ASR backend. It
  checks cumulative partials, PING/PONG interleaving, one terminal event, and
  cancellation. The cancelled request must show `SGLANG_ABORT_CONFIRMED` with
  `finish=abort` in the stage log: that is the backend's own acknowledgement,
  not a local socket close.

Check each script's argument help if its local stage defaults differ from your
checkout. Do not use `--skip-real` when validating the end-to-end journey.

## 4. Exercise network retry and stall behavior

```sh
python3 -I tests/client_retry_e2e.py "$UMMINFER_BINARY" --model "$MLX_MODEL"
```

The test inserts a TCP byte relay between the real client and real server. The
relay only forwards, delays, holds, truncates, resets, or observes actual bytes.
It never writes HTTP, SSE, WebSocket, or control-frame payloads of its own.

Both relays ignore connections that are not part of the protocol under test. A
localhost port scanner or health probe sends `GET /` and must not consume a
relay plan slot or bind the WebSocket session. `tests/proxy_isolation_test.py`
pins that invariant without a model, and also proves the pre-fix behaviour still
fails:

```sh
python3 -I tests/proxy_isolation_test.py "$UMMINFER_BINARY"
```

Expected result: retries occur only where the client contract allows them.
Emitted stream content is not silently restarted. Stall and total timeouts
produce client errors. Subsequent requests and process cleanup pass.

These network cases complement the primary smoke. They do not prove immediate
interruption of an MLX forward call.

## 5. Check the MLX backend separately

```sh
python3 -I tests/mlx_smoke.py \
  --server "$UMMINFER_BINARY" \
  --stage python/umminfer_mlx_stage.py \
  --model "$MLX_MODEL"
```

Expected result: JSON, SSE, abort, disconnected-client recovery, and concurrent
requests pass with one persistent stage. Reported latency is a local
measurement, not a performance guarantee.

## 5b. Mac unified low-resource profile

Run the unified ASR -> text path with the local TinyCTC test ASR stage and a
real MLX text stage:

```sh
python3 -I tests/mac_unified_validation.py \
  --model "$MLX_MODEL" --evidence-dir /tmp/umminfer-phase5-mac-evidence
```

See `docs/mac-unified.md` for scope, recorded results, evidence layout, and
limits. TinyCTC is an inference test model; it is not recognition-quality
evidence.

## 6. Manual smoke during development

Start the text or audio processes as shown in README. Then check:

```sh
curl --fail-with-body http://127.0.0.1:30000/health
curl --fail-with-body http://127.0.0.1:30000/ready

curl -N http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Count to three."}],"max_tokens":32,"stream":true}'

./zig-out/bin/umm-infer transcribe --url http://127.0.0.1:31000 \
  --model tiny-ctc --file audio.wav --max-new-tokens 64

./zig-out/bin/umm-infer realtime --url http://127.0.0.1:31001 --final-url http://127.0.0.1:31000 \
  --model tiny-ctc --file audio.wav --live-result --max-new-tokens 64
```

Expected result: readiness is 200 only after the model stage is ready. SSE
contains incremental deltas and a terminating done event. Both audio commands
exit 0 and print the real TinyCTC output.

For authentication, configure the server key through its `serve --api-key`
option in a protected local test environment. Pass the client key only through
`UMMINFER_API_KEY` or `OPENAI_API_KEY`. Repeat transcription and realtime with
no key, a wrong key, and the correct key. Check the server and client logs for
accidental key disclosure.

Stop each owned server and stage with SIGTERM. Confirm that later launches can
bind fresh IPC endpoints and that no test-owned process remains.

## 7. Future model acceptance gates

These gates apply to the model support plan in `roadmap.md`. They are not
current capabilities. A passing HTTP status alone never accepts a model.

Every accepted case must use this real path:

```text
real CLI or public API -> ummInfer server
  -> real backend with a local real model, or authorized official API with a real model
  -> decoded user artifact
```

A direct SGLang run that bypasses ummInfer is supporting backend evidence, not
final acceptance. An unimplemented integration remains `PLANNED` or
`IN_PROGRESS — integration planning`.

Do not use a fake server, fake runtime, fake backend, handwritten model
response, or generated artifact in place of model output. Record the exact
model id and revision, local weights or API endpoint type, backend version,
precision, quantization method, input and output parameters, random seed where
applicable, and artifact SHA256. Record wall time, output duration or count,
peak accelerator memory, host RAM, and resource limits. Mark an unexecuted
case `NOT_COVERED` with the missing hardware, weights, backend, or API access.

### Qwen3.5-Omni

**PLANNED.** Only the authorized official API is planned. The verified model
IDs are `qwen3.5-omni-plus` and `qwen3.5-omni-flash`. They accept text, image,
audio, and video input and return text or streaming speech. Do not claim image
or video generation. Authorized account/API access is unverified and no hosted
execution has been accepted. Until access exists, every API execution case is
`NOT_COVERED`.

After access is verified, exercise each declared input modality through the
complete CLI/public API-to-ummInfer-to-official-API path. Record the provider
model identifier without exposing credentials. Check the returned text. Decode
the streaming speech and check its format, duration, channel and sample layout,
and SHA256.

### MiniMaxAI/MiniMax-H3

**IN_PROGRESS — integration planning.** Establish the full-stack BF16 baseline
on H20B for high-precision acceptance and quality/resource comparisons. Include
DiT, complete text encoders, VAE, activations, offload, and host RAM. Official
Hugging Face metadata for the original H3 FL2VA partition is approximately
144.05 GB. This disk size is not a VRAM requirement.

Cover text, image, video, and audio inputs separately and in supported
combinations. Record the actual input files, hashes, model precision, and all
generation parameters. Decode the real 768p video output and native 32 kHz
stereo audio output. Check video frame decoding, dimensions, frame count and
timing; check audio decoding, sample rate, channel count, sample format and
duration. Save both artifact hashes and the parameters needed to reproduce the
run.

The local open H3-Base acceptance plan covers only the 768p path. 2K requires
hosted Context-IR/Regenerate-2K modules that are not fully open. Do not claim
fully local 2K.

There is no official quantized checkpoint. Official weights with runtime online
quantization are distinct from third-party converted or pruned quantized
packages. Verify a third-party package's revision, license, tensor completeness,
hashes, and real outputs. Do not call it official MiniMax quantization. A trusted
quantized run may execute before the H20B baseline, but it is a separate
precision configuration. Compare its real outputs and resource use with the
BF16 baseline when H20B is available. Without a trusted artifact or compatible
hardware, mark the case `BLOCKED` or `NOT_COVERED`.

There is no H3-specific Mac validation, memory guarantee, or performance
commitment. Generic MPS or offload examples do not establish model acceptance.
Limit Mac execution to low-resource real-model tests. Unsupported or unrun
modalities remain `NOT_COVERED`.

### MiniMaxAI/MiniMax-H3-Image

**IN_PROGRESS — integration planning.** Use the same full-stack BF16 baseline
scope as H3: DiT, complete text encoders, VAE, activations, offload, and host
RAM. The H3-Image selective layout for SGLang/vLLM is approximately 135.38 GB.
This disk size is not a VRAM requirement.

Cover text-to-image generation, editing with one reference image, and editing
with 16 reference images. Record each input image hash and the applicable size,
count, seed, and generation parameters. Decode every returned image with an
actual image decoder. Check its format, pixel dimensions, color mode, and full
artifact SHA256. Do not accept a JSON success field or HTTP 200 as an image
result.

There is no official quantized checkpoint. Apply H3's distinction between
runtime online quantization and third-party packages, including all verification
and `BLOCKED`/`NOT_COVERED` rules. There is no H3-Image-specific Mac validation,
memory guarantee, or performance commitment. Generic MPS or offload examples do
not establish model acceptance. Limit Mac execution to low-resource real-model
tests. Any editing, reference-count, or output-format case that was not executed
remains `NOT_COVERED`.

## 5c. Unified ASR partials and VLM GPU abort

`tests/unified_partial_abort_live.py` drives the real Qwen3-ASR and Qwen3-Omni
workers through public HTTP/SSE. It requires `UMMINFER_LIVE_OPT_IN=1`, the
ReleaseFast binary, both SGLang backends, and real audio.

```sh
UMMINFER_LIVE_OPT_IN=1 python3 -I tests/unified_partial_abort_live.py \
  --server ./zig-out/bin/umm-infer \
  --asr-backend http://127.0.0.1:43100 \
  --omni-backend http://127.0.0.1:43101 \
  --audio asr_zh.wav --cancel-audio asr_en_120s.wav \
  --image man_ironing.png \
  --evidence-dir /tmp/umminfer-phase5-streaming-abort-evidence
```

Expected result, recorded on H20B 2026-11-13: a default request carries one
`asr_final` and no `asr_partial`; `stream_options.include_asr_partials` adds
ordered `asr_partial` deltas before one `asr_final`, then `vlm_delta`, with
gap-free `seq` and one terminal; a cancel during the VLM stream yields exactly
one `cancelled`, no later delta, `/abort` 200, `in_flight=0`, and the Omni
worker logs `SGLANG_ABORT_CONFIRMED ... finish=abort`; the next request
completes. `tests/sglang_abort_probe.py` records the backend-only confirmation:
`/abort_request` 200 and a raw SSE `finish_reason="abort"` frame.

## NOT_COVERED

- A first-party HTTP 429 with `Retry-After`: this server does not emit one.
  Synthetic protocol responses are not validation evidence.
- `other_501_502_public_client`: upstream route status mapping is being changed
  in Batch 2. This binary snapshot is not evaluated for a real public-client 501
  or 502 response, so both remain unverified until the Batch 2 snapshot is
  final.
- Immediate cancellation inside MLX prefill or a model call.
- CUDA execution and CUDA IPC on this Apple Silicon environment.
- Production ASR accuracy and generalization: TinyCTC is an inference test model.
- Full OpenAI Realtime compatibility: that API is not implemented.

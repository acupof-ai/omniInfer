# Roadmap

## Goal

One engine runs VLM and ASR workloads together. The engine is one control plane
and one data plane:

- one session shared by text, speech, and image inputs;
- one orchestration path that can plan several model workers for one request;
- one streaming event contract;
- one cancellation path;
- backpressure at every worker boundary;
- one error-propagation contract;
- output coordination when a request produces more than one artifact.

Image and video are optional modalities. They attach as image-understanding,
image-generation, or video workers when a use case needs them. They do not block
the VLM+ASR main path.

The engine is the existing runtime, extended. It is not a rewrite into another
framework. The C++ Coordinator, the ZeroMQ/MessagePack control messages, the
shared-memory relay, the external Python stage protocol, the HTTP/SSE/WebSocket
adapters, the CLI, and the clients stay. A model runs in its own worker process
behind that control plane. The control plane must not learn model internals.

## Current state

These items are implemented. Each is covered by `zig build test`, by
`zig build smoke`, or by a documented real-model check. They are the foundation
the phases build on.

- C++20 control plane: Coordinator, request ids, `in_flight` accounting, abort,
  ready-gated startup.
- ZeroMQ transport with MessagePack control messages, including Python-produced
  message decode checks.
- POSIX shared-memory payload relay with bounded credits and relay
  acknowledgements between C++ stages. Optional CUDA IPC build path, unbuilt
  and unexecuted here.
- External persistent Python stage protocol. The stage announces `stage_ready`
  after model load and serves one request at a time.
- MLX-LM text chat through the external stage, with JSON and SSE output.
- Deterministic TinyCTC audio stage that exercises the real inference path.
- HTTP endpoints, multipart transcription, private realtime WebSocket protocol,
  optional bearer authentication.
- CLI and C++ clients: chat, transcribe, realtime, image, video.
- Native macOS client `clients/ummmlx` with SSE text chat.
- Fixed-upstream proxy mode: one process, one upstream, one variant, no routing.
  It forwards to an external server; it does not add a local model worker.

Implemented does not mean model-accepted. TinyCTC is test infrastructure with
deterministic initialized weights. The MLX-LM chat path is text inference, not a
VLM.

## Not implemented

- A real VLM worker and a VLM request path.
- A trained ASR model.
- Cross-model orchestration: no request plans ASR then VLM in one session.
- A session object shared by chat, speech, and image turns.
- A unified cancellation, backpressure, and error contract across several
  workers. Per-request abort exists. Relay credits bound one C++ payload edge.
- Unified multimodal end-to-end acceptance.
- Image and video as local workers. The current image and video routes are
  fixed-upstream proxies only.
- RL rollout, trajectory, reward, and training integration.

## Phases

Each phase lists a goal, deliverables, and an acceptance gate. A phase is
complete only when every gate row passes on the named hardware. A partial or
unrun case is recorded as `BLOCKED`, `NOT_RUN`, or `NOT_COVERED`, never as
passed. `NOT_RUN` means no run was attempted; `BLOCKED` means a required
external dependency, such as hardware access or credentials, is missing.
[Validation](validation.md) uses `BLOCKED` and `NOT_COVERED` for the same
purpose.

### Phase 0 — Contract and orchestration

**Goal.** Define the worker and session contract before adding model code.
Existing chat and ASR paths keep working through it.

**Deliverables.**

- A session contract: session id, ordered turns, per-turn modality, and the
  owner of session state. The control plane owns the session; workers own model
  state only.
- A worker contract: capability declaration, input and output schema,
  streaming event types, cancel signal, backpressure signal, and terminal error
  shape. The contract rides on the existing MessagePack control messages; it
  does not add a second protocol stack.
- An orchestration plan: a request is a list of ordered worker steps with input
  and output binding. One step is the current chat or ASR case; several steps
  are the new case. The existing fixed-upstream proxy is not a plan step: it
  serves one route from one upstream and has no Coordinator, session, or
  orchestration.
- One event stream per request that multiplexes step events in plan order.
- Cancel, backpressure, and error propagation rules for a multi-step plan,
  including which step observes a cancel first and how a downstream step is
  informed.
- Contract tests. A two-worker plan runs with two conforming test workers, one
  of which fails, and the test checks event order, cancel, error shape, and
  session isolation.

**Acceptance gate.**

| Check | Rule |
| --- | --- |
| Existing behavior | Chat JSON, chat SSE, transcription, and realtime still pass `zig build test` and the real-model client checks. |
| Plan execution | A two-step plan with test workers returns ordered stream events and one terminal event. |
| Cancel | Cancelling a running two-step plan reaches both workers and leaves `in_flight` at zero. |
| Backpressure | A slow consumer bounds worker output without unbounded buffer growth; the test measures high-water bytes. |
| Error propagation | A failing middle step yields one terminal error with the failing step id, and later steps do not run. |
| Session isolation | Two concurrent sessions do not share turns or output events. |
| No model internals | The contract carries no model-specific field; a test worker and a real worker use the same schema. |

### Phase 1 — Real model workers

**Goal.** Replace test compute with real models behind the Phase 0 contract,
one model family per worker process.

**Deliverables.**

- An ASR worker that runs a trained model and reports real transcription, with
  a declared audio format contract and resampling at the worker boundary where
  the model needs it.
- A VLM worker that accepts text plus image input and produces text or streamed
  text. The worker declares image input in its capability record.
- Worker startup, readiness, health, and shutdown under the existing stage
  protocol.
- Model identity in every response: model id, revision, precision, and
  quantization method, so a later comparison can attribute a result.
- Per-worker resource accounting: peak accelerator memory and host RAM.

**Acceptance gate.**

| Check | Rule |
| --- | --- |
| Real ASR output | A real audio file produces text through CLI and HTTP, with model id and revision recorded, and the text is checked by a human and a reference transcript. |
| Real VLM output | A real image plus a prompt produces a decoded text answer through CLI and HTTP. HTTP 200 alone is not acceptance. |
| Streaming | Both workers stream partial output through the Phase 0 event stream. |
| Cancel | Cancel reaches both worker types and the worker process stays usable for the next request. |
| Readiness | `/ready` answers 503 before worker `stage_ready` and 200 after. |
| No simulator | No test substitutes a fake model, a handwritten response, or a generated artifact for model output. |

### Phase 2 — ASR to VLM orchestration

**Goal.** Run one request through ASR then VLM in one session, with one event
stream and one terminal result.

**Deliverables.**

- A plan that binds ASR output to VLM input.
- Ordered streaming: ASR progress and final text arrive before VLM tokens.
- Output coordination: the client receives both the transcript and the answer
  with clear step attribution.
- Error and cancel semantics for the two-step plan, including a cancel between
  steps.

**Acceptance gate.**

| Check | Rule |
| --- | --- |
| Joint output | One request with real audio and a real image returns a transcript and an answer that uses both inputs. |
| Event order | Stream events show ASR completion before the first VLM token. |
| Cancel between steps | A cancel after ASR and before VLM leaves no VLM work running and `in_flight` at zero. |
| Failure isolation | An ASR failure returns one terminal error and does not start the VLM step. |
| Session reuse | A second turn in the same session sees the first turn's transcript and answer. |

### Phase 3 — Unified multimodal end-to-end acceptance

**Goal.** Accept the unified engine through the public interfaces on real
models, on the named hardware, with the Phase 0 contract in force.

**Deliverables.**

- One acceptance script that drives the real CLI or public C++ API through a
  real server, real workers, and real models, then decodes the real artifacts.
- A recorded run: model ids and revisions, weights hashes, precision,
  quantization method, parameters, seeds, artifact SHA256, wall time, output
  length, peak accelerator memory, host RAM.
- The Mac profile and the H20B profile, recorded separately.

**Acceptance gate.**

| Check | Rule |
| --- | --- |
| Path | CLI or public API to ummInfer server to real workers to real models to decoded artifact. A direct backend run is supporting evidence only. |
| Mac profile | Low-resource real-model run completes and its artifact decodes. |
| H20B profile | High-precision run completes and its artifact decodes. |
| Reproducibility | The recorded parameters reproduce the recorded artifact hash, or the difference is explained. |
| Status honesty | Any unrun modality or model is recorded `BLOCKED` or `NOT_RUN`. |

The H20B row stays `BLOCKED` until access is restored. The phase cannot be
declared complete with that row unrun.

### Phase 4 — Optional image and video

**Goal.** Attach image understanding, image generation, and video workers when a
use case needs them. Do not block the VLM+ASR path.

**Deliverables.**

- A capability record and a plan step per optional worker, on the same
  Phase 0 contract.
- Image input to the VLM worker through the existing content part schema, and
  image output coordination when a plan produces an image artifact.
- Video as a separate worker with its own job lifecycle, if a use case requires
  local video.
- A configuration path that runs the engine with zero optional workers.

**Acceptance gate.**

| Check | Rule |
| --- | --- |
| Main path unaffected | The unified VLM+ASR plan passes its gate with every optional worker absent. |
| Image in | A real image reaches the VLM worker and the answer references its content. |
| Image out | A generated image is decoded by an image decoder; format, dimensions, color mode, and SHA256 are recorded. |
| Video | Unrun cases stay `BLOCKED` or `NOT_RUN`; no claim of local 2K or untested video output. |
| Contract reuse | Optional workers use the same event, cancel, backpressure, and error schema; no worker-specific event channel. |

### Phase 5 — Unified performance baseline and joint optimization

**Goal.** Establish one measurable baseline for the unified engine, then
optimize it without changing output.

**Deliverables.**

- A fixed configuration per path: model snapshot, prompt, `max_tokens`,
  `temperature=0`, sample count, exclusive measurement window.
- Reported metrics per path: first-event latency, total latency, output tokens
  per second, peak memory, cancellation latency, queue depth.
- Attribution per level: direct model, worker, control plane, HTTP/SSE.
- Artifact hashes and binary, stage, harness, model, and weights hashes in every
  report.

**Acceptance gate.**

| Check | Rule |
| --- | --- |
| Baseline recorded | One report per hardware profile with every path present; no skipped path. |
| Output identity | One output hash per path, equal across baseline and candidate runs. |
| No regression | Candidate first-event latency, total latency, and peak memory are not above baseline by more than 5%, or the difference is reported as unchanged. |
| Stability | The coefficient of variation is reported; `cv > 0.10` means the window was not quiet and the run repeats. |
| Attribution | The report separates model time from control-plane time. |

The existing text-chat optimization record keeps its own convention. For the
current MLX stage against the frozen iteration-2 baseline it records a 61.93%
warm speedup on median `total_ms` at `max_tokens=64` over HTTP SSE, 20
interleaved pairs. Three of that record's properties limit what it can support:
every recorded `cv` is above 0.10, so the run asks for a repeat; the metric is
a ratio of medians while the median per-pair speedup is 51.11%; and the whole
stage change includes a prompt KV snapshot and detokenizer reuse, not only the
fused QKV projection. It is one recorded result, not a guarantee, and it is not
a unified-engine baseline. [README](../README.md#measured-stage-result) gives
the full conditions.

### Phase 6 — RL rollout, trajectory, reward, and training

**Goal.** Expose the unified engine as a rollout target for reinforcement
learning, without putting training inside the serving process.

**Deliverables.**

- A rollout record per request: session id, ordered turns, per-step inputs and
  outputs, event timestamps, cancel and failure events, terminal status.
- A trajectory export in a documented format, written by the control plane, not
  by a worker.
- A reward hook at the control-plane boundary: reward is computed from recorded
  trajectories, and a trainer reads the export.
- Independent training as a separate process and repository concern. The
  serving engine does not load optimizer state.
- Determinism controls documented per worker: seed handling, sampling
  parameters, and what cannot be made deterministic.

**Acceptance gate.**

| Check | Rule |
| --- | --- |
| Export | A run produces a trajectory file that parses and contains every turn of the session. |
| Replay | Replaying a recorded trajectory reproduces the recorded outputs within the stated precision, or the difference is explained. |
| Reward boundary | Reward computation reads the export only and does not need access to worker memory. |
| Isolation | Training load does not degrade serving latency in the measured window, or the degradation is measured and reported. |
| No unverified claim | No phase claims an RL result that has not run on real models and recorded hardware. |

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

Final generation acceptance must use the CLI or public API, the ummInfer server,
the real backend or authorized official API, the real model, and the decoded
artifact. A direct SGLang run that bypasses ummInfer is not final acceptance.
Unimplemented integrations remain **PLANNED** or **IN_PROGRESS — integration
planning**.

## Verification split

Two hardware profiles, with different jobs:

| Profile | Job | Limits |
| --- | --- | --- |
| Apple Silicon Mac | Low-consumption development. Control-plane tests, stage contract tests, small real-model chat and audio runs, CLI and client checks. | No CUDA. No H3 or H3-Image acceptance. No performance commitment. A model that cannot run inside the Mac limits stays `NOT_COVERED` on Mac. |
| NVIDIA H20B | High-precision acceptance and quality/resource comparison. BF16 baseline, CUDA build and IPC runs, sanitizer and stress runs. | Only one owner runs builds and benchmarks at a time, in an agreed window. Do not run benchmarks concurrently with sanitizer or stress runs. |

### H20B status

**BLOCKED / NOT_RUN.** Access to H20B fails at the jump host because Kerberos
authentication is denied (GSSAPI) after the ticket expired. No H20B run for the
unified engine exists. No CUDA execution, CUDA IPC run, sanitizer run, stress
run, or BF16 baseline is recorded for this revision.

Two earlier facts do not change this status:

- An earlier round ran CUDA stress and IPC benchmarks on H20 hardware for the
  relay work. That round covers the relay, not a VLM, ASR, or unified-engine
  model run.
- No live H3, H3-Image, or Qwen-Omni run exists on any host. Each attempt prints
  `BLOCKED` because the live opt-in and credentials are absent.

A BLOCKED status is not a pass. Do not write an H20B result as verified until a
run produces a log with a real exit code, the recorded environment, and a
decoded artifact. Restoring access needs a fresh Kerberos ticket from the
account owner. The decision to restore access or to name another high-precision
host belongs to the project owner.

The Mac rows of a phase gate can pass while H20B rows are `BLOCKED`. A phase
that requires both profiles is not complete in that state.

## Next priorities

1. Phase 0: write the session and worker contract, and add the two-worker plan
   test on the existing MessagePack messages.
2. Phase 1: add a trained ASR worker with a declared audio format contract and
   resampling, then a real VLM worker with image input.
3. Phase 2: bind ASR output to VLM input through one plan and one event stream.
4. Restore H20B access, then run the Phase 3 H20B profile and the CUDA build,
   IPC, sanitizer, and stress runs.
5. Phase 5: record one unified baseline per profile before any joint
   optimization claim.
6. Phase 4 and Phase 6 only after the VLM+ASR main path passes its gate.
7. Keep the fixed-upstream proxy independent of the local engine; it is not an
   orchestration path.

## Explicitly unverified or unsupported

- CUDA execution, CUDA IPC, CUDA stress, and compute-sanitizer results on this
  machine.
- H20B access and every H20B result.
- Production speech recognition quality. TinyCTC has deterministic initialized
  weights and no speech training.
- OpenAI Realtime API compatibility. The implemented realtime protocol is
  private and returns one aggregated result.
- Natural WebSocket control-frame interleaving in the continuation regression.
  No stable real-client trigger exists without injecting frames.
- A first-party HTTP 429 with `Retry-After` from this server. The server does
  not emit one.
- Immediate cancellation inside an MLX model call. Abort takes effect at token
  boundaries.
- Mixed Python/C++ shared-memory payload transfer and Python pickle payload
  headers.
- Local multimodal chat, local image generation, local video generation, tools,
  JSON response format, text-to-speech, and dynamic batching.
- Model-backed execution in the C++ stage host. Its executable uses echo
  compute.
- Live upstream runs. The live scripts gate on opt-in and credentials, and no
  live run is recorded for this revision.
- A real public-client 501 or 502 through the fixed-upstream proxy. The proxy
  tests cover configuration, route gating, and a connect failure, not a
  successful upstream forward.
- Unified final integration acceptance across all participants' changes. The
  component results recorded during development do not replace that acceptance.

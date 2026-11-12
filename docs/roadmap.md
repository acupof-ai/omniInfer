# Roadmap

## Goal

One engine runs VLM and ASR workloads together, on one control plane and one
data plane: a session shared by text, speech, and image turns; one orchestration
path that can plan several model workers for one request; one streaming event
contract; one cancellation path; backpressure at every worker boundary; one
error-propagation contract; and output coordination when a request produces more
than one artifact.

Image and video are optional modalities. They attach as image-understanding,
image-generation, or video workers when a use case needs them, and they do not
block the VLM+ASR main path.

The engine is the existing runtime, extended, not a rewrite. The C++ Coordinator,
ZeroMQ/MessagePack control messages, shared-memory relay, external Python stage
protocol, HTTP/SSE/WebSocket adapters, CLI, and clients stay. A model runs in its
own worker process behind that control plane; the control plane must not learn
model internals.

The target architecture figure and the implemented-versus-planned split are in
[README](../README.md#target-architecture). This file holds the phases, their
gates, the model plan, and the verification split.

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
  shape. It rides on the existing MessagePack control messages; it does not add
  a second protocol stack.
- An orchestration plan: a request is a list of ordered worker steps with input
  and output binding. One step is the current chat or ASR case; several steps
  are the new case. The fixed-upstream proxy is not a plan step: it serves one
  route from one upstream and has no Coordinator, session, or orchestration.
- One event stream per request that multiplexes step events in plan order.
- Cancel, backpressure, and error propagation rules for a multi-step plan,
  including which step observes a cancel first and how a downstream step is
  informed.
- Contract tests. A two-worker plan runs with two conforming test workers, one
  of which fails; the test checks event order, cancel, error shape, and session
  isolation.

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

**Goal.** Replace test compute with real models behind the Phase 0 contract, one
model family per worker process.

**Deliverables.**

- An ASR worker that runs a trained model and reports real transcription, with a
  declared audio format contract and resampling at the worker boundary where the
  model needs it.
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
| H20B profile | High-precision run completes and its artifact decodes. The row stays `BLOCKED` until access exists; the phase is not complete with it unrun. |
| Reproducibility | The recorded parameters reproduce the recorded artifact hash, or the difference is explained. |
| Status honesty | Any unrun modality or model is recorded `BLOCKED` or `NOT_RUN`. |

The recorded-run fields and the anti-simulator rules are in
[validation section 7](validation.md#7-future-model-acceptance-gates).

### Phase 4 — Optional image and video

**Goal.** Attach image understanding, image generation, and video workers when a
use case needs them, without blocking the VLM+ASR path.

**Deliverables.**

- A capability record and a plan step per optional worker, on the same Phase 0
  contract.
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

The existing text-chat optimization record keeps its own convention; it is not a
unified-engine baseline. [README](../README.md#measured-stage-result) gives the
number and its conditions.

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

These entries are not current deliverables. Their acceptance procedure,
precision rules, artifact checks, and the quantized-versus-official distinction
are in [validation section 7](validation.md#7-future-model-acceptance-gates).

| Model | Status | Target and constraints |
| --- | --- | --- |
| `Qwen3.5-Omni` | **PLANNED** | Authorized official API only. The documented model IDs are `qwen3.5-omni-plus` and `qwen3.5-omni-flash`; their documented interface accepts text, image, audio, and video input and returns text or streaming speech. Authorized access is unverified, so every execution case is `NOT_COVERED`. Do not claim image or video generation. |
| `MiniMaxAI/MiniMax-H3` | **IN_PROGRESS — integration planning** | Full-stack BF16 baseline on H20B. The open H3-Base path is limited to 768p; 2K needs hosted Context-IR/Regenerate-2K modules that are not fully open. |
| `MiniMaxAI/MiniMax-H3-Image` | **IN_PROGRESS — integration planning** | Full-stack BF16 baseline on H20B. Text-to-image and editing with up to 16 reference images. |

BF16 acceptance covers DiT, complete text encoders, VAE, activations, offload,
and host RAM. Official Hugging Face metadata for the H3 FL2VA partition is
approximately 144.05 GB, and the H3-Image selective layout for SGLang/vLLM is
approximately 135.38 GB; both are disk sizes, not VRAM requirements. There are
no official quantized checkpoints; a third-party package must be verified before
use and must not be called official MiniMax quantization.

Final generation acceptance must use the CLI or public API, the ummInfer server,
the real backend or authorized official API, the real model, and the decoded
artifact. A direct SGLang run that bypasses ummInfer is not final acceptance.

## Verification split

Two hardware profiles, with different jobs:

| Profile | Job | Limits |
| --- | --- | --- |
| Apple Silicon Mac | Low-consumption development. Control-plane tests, stage contract tests, small real-model chat and audio runs, CLI and client checks. | No CUDA. No H3 or H3-Image acceptance. No performance commitment. A model that cannot run inside the Mac limits stays `NOT_COVERED` on Mac. |
| NVIDIA H20B | High-precision acceptance and quality/resource comparison. BF16 baseline, CUDA build and IPC runs, sanitizer and stress runs. | Only one owner runs builds and benchmarks at a time, in an agreed window. Do not run benchmarks concurrently with sanitizer or stress runs. |

Mac runs stay at low resource; generic MPS or offload examples do not establish
model acceptance, and a simulator never replaces the real model.

### H20B status

**BLOCKED / NOT_RUN.** Access fails at the jump host because Kerberos
authentication is denied (GSSAPI) after the ticket expired. No H20B run for the
unified engine exists: no CUDA execution, CUDA IPC run, sanitizer run, stress
run, or BF16 baseline is recorded for this revision.

Two earlier facts do not change this status:

- An earlier round ran CUDA stress and IPC benchmarks on H20 hardware for the
  relay work. That covers the relay, not a VLM, ASR, or unified-engine model run.
- No live H3, H3-Image, or Qwen-Omni run exists on any host. Each attempt prints
  `BLOCKED` because the live opt-in and credentials are absent.

A BLOCKED status is not a pass. An H20B result counts only when a run produces a
log with a real exit code, the recorded environment, and a decoded artifact.
Restoring access needs a fresh Kerberos ticket from the account owner; naming
another high-precision host is the project owner's decision. Mac rows of a gate
can pass while H20B rows are `BLOCKED`, but a phase that needs both profiles is
not complete in that state.

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

These are not repeated from the README limits list.

- CUDA build, execution, IPC, stress, and compute-sanitizer on any host.
- H20B access and every H20B result.
- Natural WebSocket control-frame interleaving in the continuation regression.
  No stable real-client trigger exists without injecting frames.
- A first-party HTTP 429 with `Retry-After` from this server.
- A successful upstream forward through the proxy. The in-tree tests prove
  configuration, route gating, and a connect failure only.
- Dynamic batching, model usage accounting, and per-worker queue statistics.
- Unified final integration acceptance across all participants' changes.

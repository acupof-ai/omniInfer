# Mac unified low-resource validation

This profile runs the unified engine on one Apple Mac with two local stages. It
covers the public HTTP/SSE path, the ASR -> text orchestration, concurrency 1
and 2, cancellation, an ASR failure, and recovery after each of them.

The harness is `tests/mac_unified_validation.py`. It is opt-in: without
`--evidence-dir` it prints usage and exits 0.

## Scope

Two stages run in separate processes:

| Leg | Stage | What it proves |
| --- | --- | --- |
| ASR | `python/umminfer_tiny_asr_stage.py` with the deterministic TinyCTC model | ASR protocol, WAV/PCM handling, stage errors, unified orchestration |
| Text | `python/umminfer_mlx_stage.py` with a real MLX-LM model | The real text leg and SSE streaming |

TinyCTC is an initialized inference test model. It is not trained, and its
output is not recognition-quality evidence. The run proves the contract, not
speech recognition. The MLX model proves the text leg only.

The harness does not contact H20B, CUDA, SGLang, or any remote service. All
traffic is localhost HTTP and local IPC.

## Prerequisites

- Apple Silicon Mac with Zig 0.16.0 and the Homebrew dependencies in README.
- Python with `mlx`, `mlx-lm`, `tokenizers`, `pyzmq`, `msgpack`, and `numpy`.
- A local MLX-LM text model directory with `config.json` and
  `model.safetensors`.

Build the runtime:

```sh
zig build -Doptimize=ReleaseFast -Ddeps-prefix=/opt/homebrew \
  -Dcurl-prefix=/opt/homebrew/opt/curl
```

## Run

```sh
python3 -I tests/mac_unified_validation.py \
  --binary ./zig-out/bin/umm-infer \
  --model /absolute/path/to/mlx-model \
  --evidence-dir /tmp/umminfer-phase5-mac-evidence \
  --samples 5 --concurrency-samples 4 --warmup 1 --max-new-tokens 24
```

The evidence directory must be empty or absent. The harness starts the server,
both stages, and the IPC endpoints. It stops every process it owns. Exit 0
requires zero failed checks and a complete SHA256 verification.

## What it asserts

Each successful unified request must have:

- HTTP 200, a terminating `[DONE]`, and exactly one `complete` terminal event.
- Unified sequence numbers that start at 0 and have no gap.
- Exactly one `asr_final` event, before the first text delta.
- One `vlm_delta` event per content delta.
- The requested `session_id` on every unified event.
- Non-empty transcript and non-empty answer text.

The other checks are:

| Check | Expected result |
| --- | --- |
| Cancel during the text leg | `cancelled` terminal, `[DONE]`, no error, `/health` `in_flight` returns to 0 |
| Request after cancel | `complete` with non-empty text |
| ASR error: WAV with one PCM16 sample | `error` terminal with the stage reason `audio is shorter than one analysis frame` |
| Request after the ASR error | `complete` with non-empty text |

Concurrency 2 carries two requests in flight. Both stages run one request at a
time. The c2 numbers therefore measure server admission, isolation, and
queueing, not throughput.

## Recorded run

One local run on 2026-11-13. Evidence:
`/tmp/umminfer-phase5-mac-evidence`. The table comes from `summary.json`, which
is derived from the raw per-request JSON lines.

| Item | Value |
| --- | --- |
| Host | Mac16,8, Apple M4 Pro, 48 GiB RAM, Darwin 25.3.0 arm64 |
| Python / Zig | 3.11.14 / 0.16.0 |
| MLX / MLX-LM | 0.31.1 / 0.31.2 |
| Binary | `umm-infer` SHA256 `26651a12…c37a984d` |
| Text model | `mlx-community/Qwen2.5-0.5B-Instruct-4bit`, weights SHA256 `ddffab9c…6326d153` |
| ASR test model | TinyCTC, seed 20261111, weights SHA256 `7226b2cf…cd0e5084` |
| Audio | deterministic 440 Hz tone, 0.5 s, 16 kHz mono PCM16, SHA256 `cbe2e5fc…dae44900` |
| Sampling | `temperature=0`, `stream=true`, `max_tokens=24`; the ASR leg inherits it, with `stream=false` and `temperature=0` forced by the engine |
| Warm-up | 1 request, discarded |
| Git | baseline `a412d0d4`, harness files uncommitted during the run |

Latency in milliseconds. Percentiles are nearest-rank over successful samples.

| Scenario | Concurrency | Samples | Failures | Total p50 / p95 | First event p50 / p95 | First delta p50 / p95 | ASR final p50 / p95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `unified-c1` | 1 | 5 | 0 | 69.935 / 71.125 | 7.360 / 7.467 | 13.738 / 13.946 | 7.360 / 7.467 |
| `unified-c2` | 2 | 4 | 0 | 136.129 / 178.437 | 78.169 / 115.990 | 84.649 / 121.654 | 78.169 / 115.990 |

Peak RSS in KiB during each scenario window:

| Scenario | Server | ASR stage | Text stage |
| --- | --- | --- | --- |
| `unified-c1` | 10,528 | 53,456 | 887,744 |
| `unified-c2` | 10,704 | 53,472 | 888,064 |

Checks: cancel passed on the first attempt with 2 text deltas, `[DONE]`, and
`in_flight=0`. Both recovery requests completed. The ASR error propagated the
stage reason. Total failures: 0.

The first-event time includes the ASR leg. At concurrency 2 the second request
waits behind the first in the serial ASR stage, so its first event is later.
Five and four samples are small. p95 equals the maximum. These numbers are not
a performance guarantee and they do not predict H20B behavior.

## Evidence

```text
audio.wav                 deterministic input
env.json                  host, packages, hashes, configuration, scope
tinyctc/                  exact ASR test model used
raw/unified-c1.jsonl      every measured request
raw/unified-c2.jsonl
raw/cancel.jsonl          every cancel attempt
raw/asr-error.jsonl
raw/after-cancel.jsonl
raw/after-error.jsonl
raw/rss_samples.csv       timestamped RSS samples
logs/                     server and stage logs, each capped at 64 KiB
run.log                   harness observations, capped at 64 KiB
summary.json, summary.csv
SHA256SUMS                every file above
```

Verify the evidence:

```sh
cd /tmp/umminfer-phase5-mac-evidence
shasum -a 256 -c SHA256SUMS
find . -type f -size +64k
```

Expected result: every line reports `OK` and the `find` command prints nothing.

## Limits

- TinyCTC is not a trained ASR model. Recognition quality is `NOT_COVERED`.
- The MLX model proves the text leg. It is a 0.5B 4-bit model, not the
  production model.
- Both stages are serial. Concurrency 2 measures queueing and isolation.
- The harness sends no text or image instruction part. `UnifiedEngine` builds a
  content-part array for such a request, and the MLX-LM Qwen2.5 chat template
  accepts a string only, so that local text stage rejects it. This path is
  `NOT_COVERED` on this profile.
- One request, one cancellation, and one error are tested. Sustained load,
  memory leaks, and long-run stability are not.
- Chunked SSE partial ASR through the unified route is not covered: the engine
  submits the ASR leg with `stream=false`. Realtime partials are covered by
  `docs/validation.md`.
- The recorded run used baseline `a412d0d4` with the uncommitted harness. The
  commit adds only the harness, its unit test, and documentation. It changes no
  runtime code, and `env.json` records the binary and model hashes.

## Unit test

```sh
python3 -I tests/mac_unified_validation_test.py
# or
zig build mac-unified-test
```

The unit test covers WAV generation, request-body shape, the event-contract
checker, summary aggregation, log capping, and SHA256SUMS round-trip and
tamper detection. It starts no server and loads no model.

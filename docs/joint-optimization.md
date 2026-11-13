# Joint worker concurrency optimization — real H20B, 2026-11-13

Baseline commit `4b13fe16cfc1055c2cac09ea8b8bfa4a9aae71ec` (branch `main`).
Change under test: branch `phase5/joint-optimization`.
Every number comes from two recorded runs with the same harness, the same real
inputs, the same real Qwen3-ASR and Qwen3-Omni backends, and the same binary
source. Raw records are in
`/tmp/umminfer-phase5-optimization-evidence/{baseline,candidate}/`.

## Conclusion

One Omni worker processed one request at a time. That serial loop was the
bottleneck, not the backend and not the control plane. The same is true for the
ASR worker at unified concurrency above 1.

The direct backend probe measured 1, 2, and 4 concurrent Omni requests:

| Concurrent requests | Wall | Output tokens | Aggregate tok/s |
| ---: | ---: | ---: | ---: |
| 1 | 13.659 s | 256 | 18.74 |
| 2 | 14.245 s | 512 | 35.94 |
| 4 | 14.260 s | 1024 | 71.81 |

The ASR backend behaves the same way: 1, 2, and 4 concurrent requests return in
0.509 s, 0.562 s, and 0.554 s. SGLang already batches; the workers did not.

The baseline unified run confirms the queueing model. At concurrency 4 the
Omni worker served four waves, so total p95 was 54.8 s (`unified-en`) and
55.2 s (`unified-zh`), about 4 × 13.7 s. First-delta p95 was 41.9 s and 42.1 s,
about 3 × 13.7 s plus one ASR leg. The baseline total-latency coefficient of
variation at concurrency 4 was 0.45, which is a four-wave distribution, not
measurement noise.

After the change, both workers run up to 4 upstream requests in parallel. The
public HTTP/SSE/WebSocket surface, the unified event schema, and the
request/session/cancel/error semantics are unchanged.

## What changed

* `python/umminfer_mlx_stage.py`: the stage run loop now has one control thread
  and a bounded request-thread pool (`concurrency`, default 1). The control
  thread keeps exclusive ownership of the ZeroMQ recv, abort, and coordinator
  sockets. Request threads reuse the existing `process()` path with ZeroMQ
  disabled, queue outbound frames, and learn about cancellation from a
  per-request `threading.Event`. An abort that arrives before its submit still
  leaves the existing tombstone. The MLX-LM local stage keeps concurrency 1.
* `python/umminfer_qwen_omni_worker.py`: exposes `--concurrency` and uses the
  shared loop. The model call is an upstream HTTP stream, so threads only wait
  on I/O.
* `python/umminfer_qwen_asr_stage.py`: exposes `--concurrency` and `--perf-log`.
  Each request thread owns one `SGLangBackend`, and a backend owns one HTTP
  connection. The per-request stream mode and backend request id moved to
  thread-local state. Abort still calls the SGLang `/abort_request` endpoint for
  the request's own backend.
* `tests/unified_benchmark.py`: adds `--asr-concurrency`, `--omni-concurrency`,
  and `--worker-perf-log`. At the default concurrency 1 the worker argv is
  identical to the previous harness, so one harness file drove both runs.
* Tests: two concurrency regressions (the Omni fake upstream must report
  `max_active == 2`; two ASR request threads must reach their backend at the
  same time) and one harness argument test.

No model is duplicated. No second engine, transport, or event schema is added.

## Method

* Same harness file (`tests/unified_benchmark.py`, SHA256
  `44765d495e774d26b4319814bd2e6aa47030c948703a688e82eacab8c02b21f3`) in both
  runs.
* Baseline run: worker and server code from `4b13fe1` in a detached worktree,
  worker concurrency 1 (serial path).
* Candidate run: this branch, `--asr-concurrency 4 --omni-concurrency 4`,
  workers started with `--perf-log`.
* `temperature=0`, `max_new_tokens=256`, `max_new_tokens=256` for ASR,
  3 warm-up requests per scenario, 20 successful samples per scenario.
* Unified concurrency 1, 2, and 4, English and Chinese. A batch of `N` requests
  is released at one barrier.
* Percentiles are nearest-rank over successful samples.
* Inputs are the Phase 4 real recordings:
  `asr_en_pcm16.wav` `64fbd9f7…7eedf86d`, `asr_zh.wav` `46dbc998…1160dce50`,
  `man_ironing.png` `e0691718…4cd060e`. The tokenizer is the served Omni
  tokenizer `a7d41145…5d1fab`.
* Both SGLang backends were already running. No backend process was started or
  stopped. The local SSH control socket `/tmp/umm-bench-ssh.sock` was reused.

## Results

All latency values are milliseconds. `B` is baseline, `C` is candidate.
`agg tok/s` is `successes × output_tokens / wall_seconds` for the block.
Every block had 20/20 successes and 0 failures in both runs.

| Scenario | C | p50 B/C | p95 B/C | Δp95 (positive = faster) | fd p95 B/C | ASR final p95 B/C | tok/s per request B/C | agg tok/s B/C | req/s B/C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| unified-en | 1 | 14224/14438 | 14292/14509 | −1.5% | 1279/1292 | 724/740 | 19.73/19.43 | 18.0/17.7 | 0.070/0.069 |
| unified-en | 2 | 14306/15130 | 27849/16613 | −40.3% | 14845/2769 | 1361/2217 | 19.74/18.58 | 18.4/33.4 | 0.072/0.130 |
| unified-en | 4 | 27811/15284 | 54833/15389 | −71.9% | 41863/1574 | 2601/975 | 19.73/18.59 | 18.7/66.9 | 0.073/0.261 |
| unified-zh | 1 | 14101/14270 | 14175/14369 | −1.4% | 1147/1141 | 598/590 | 19.70/19.46 | 18.2/17.9 | 0.071/0.070 |
| unified-zh | 2 | 14282/15020 | 27957/15108 | −46.0% | 14785/1243 | 1083/626 | 19.61/18.49 | 18.4/34.1 | 0.072/0.133 |
| unified-zh | 4 | 27999/14846 | 55184/14882 | −73.0% | 42127/1234 | 2121/638 | 19.58/18.78 | 18.6/68.9 | 0.073/0.269 |

Single-entry controls, which stay at concurrency 1:

| Scenario | p95 B/C | fd p95 B/C | tok/s B/C |
| --- | ---: | ---: | ---: |
| omni-text | 13655/13777 | 585/706 | 19.65/19.58 |
| omni-image | 9390/9508 | 637/653 | 19.52/19.27 |
| asr-en | 674/716 | — | 69.71/68.34 |
| asr-zh | 547/572 | — | 20.58/20.54 |

The candidate at concurrency 4 has p95 15.4 s, against 13.7 s for one Omni
request. The remaining 1.7 s covers one ASR leg, batching contention, and
control-plane work. At concurrency 2 and 4, total latency is now flat instead
of growing in waves.

Per-request decoding speed falls by 5.8% at concurrency 4 (19.73 → 18.59
tok/s). This is the cost of four streams sharing one model. Aggregate output
throughput rises 3.6× (18.7 → 66.9 tok/s in `unified-en`, 18.6 → 68.9 in
`unified-zh`).

## Queue depth and worker-side time

The candidate workers wrote JSON-lines perf records. The log has no timestamps,
so queue depth is reconstructed from event order: `queue_enqueue` adds one,
`request_start` removes one. This counts the stage-local queue between receipt
and request start. It does not count ZMQ backlog or client-side waiting.

| Worker | Requests | Stage TTFT p50/p95 | Stage total p50/p95 | Peak stage queue | Aborts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Omni | 184 | 551/615 ms | 13755/14419 ms | 2 | 0 |
| ASR | 184 | 591/856 ms | 591/856 ms | 3 | 0 |

The peak stays below 4 because the control thread reads a submit and dispatches
it before reading the next submit. The worker is no longer the queue: the stage
total p95 is 14.4 s against a 13.7 s single request.

## Resources

GPU memory is unchanged in every block: peak GPU 0 `89739 MiB`, GPU 1
`87471 MiB`. The resident weights dominate. Mean GPU 0 utilization at
`unified-en` concurrency 4 rose from 10.0% to 11.3%; GPU 1 was 0.0–0.5% in both
runs because ASR legs are short.

Peak RSS at `unified-en` concurrency 4: server 64.3 MB → 59.8 MB, ASR worker
39.1 MB → 42.3 MB, Omni worker 30.6 MB → 31.6 MB. Worker growth is within 10%.
Every failure count is 0.

## Output consistency

Outputs were compared per block by SHA256 over the ordered successful texts.
Eight of ten blocks are byte-identical. The ASR transcript is identical in all
ten blocks (20/20 samples each).

Two blocks differ:

| Block | Identical VLM outputs | Candidate distinct outputs |
| --- | ---: | ---: |
| `unified-en` c2 | 13/20 | 2 |
| `unified-en` c4 | 17/20 | 2 |

The differing samples do not scatter. They collapse to one alternate
completion: the baseline answers `**Carole King**`, the candidate answers
`**Elvis Costello**` from the same first 167 characters. Greedy decoding is not
bit-stable under SGLang batching, so this report does not claim output
identity. Text quality is not measured here.

## Regressions and limits

* Concurrency 1 is 1.0–1.5% slower in p95 for unified blocks. `asr-en` p95 is
  43 ms higher (+6.3%). These are single-run differences at the noise scale;
  causality is not established.
* Per-request tok/s drops 5.8% at concurrency 4. Aggregate throughput rises
  3.6×.
* Two `unified-en` blocks are not byte-identical, as described above.
* Control-plane event latency is not instrumented. The 15.4 s p95 at
  concurrency 4 bounds the combined ASR, batching, and control-plane overhead
  at about 1.7 s; it does not separate them.
* Cancellation latency was not measured in this A/B. Cancellation correctness is
  covered by the existing contract tests and the worker test suite.
* The perf log has no timestamps, so queue depth is ordering evidence, not a
  time series.
* GPU utilization is sampled every 2 s, so short bursts can be missed.
* Model revision: the SGLang `revision` field is `UNKNOWN` (local-path
  launch, no `.git` metadata). All 35 deployed files are byte-identical to
  pinned HF `main` commits with full SHA256 evidence; see
  [performance-baseline.md](performance-baseline.md#model-snapshot-and-provenance)
  and `/tmp/umminfer-phase5-provenance-evidence/`.

## Checks

| Check | Result |
| --- | --- |
| `zig build test -Ddeps-prefix=/opt/homebrew -Dcurl-prefix=/opt/homebrew/opt/curl` | pass |
| `zig build unified-test …` | pass |
| `python3 -I tests/realtime_contract_e2e.py --server zig-out/bin/umm-infer --stage python/umminfer_tiny_asr_stage.py` | pass |
| `python3 -I tests/realtime_client_contract_e2e.py zig-out/bin/umm-infer` | pass |
| `python3 -I tests/unified_benchmark_test.py` | 10 tests pass |
| `python3 -I tests/qwen_omni_worker_test.py` | 3 tests pass |
| `python3 -I tests/qwen_asr_stage_test.py` | 14 tests pass |
| `python3 -I tests/mlx_stage_test.py` | 28 tests pass |
| `git diff --check` | clean |

## Regression found before push (2026-11-13)

A `ReleaseFast` build failed `tests/realtime_client_contract_e2e.py` at
`wait_abort(abort_events, "realtime-1")` in 4 of 5 runs. The request-thread pool
made the abort arrive after `dispatch`, where `record_abort` set the cancel flag
and returned without registering a tombstone. The cancel itself worked; the
registration the contract test observes was missing. Debug timing delivered the
abort before `dispatch`, so the queued path hid the defect.

`record_abort` now registers the tombstone for every request state, and
`drain_outbox` releases it when the request thread finishes.
`test_running_abort_is_registered_and_released` pins the running-request path.

Re-verified on the same H20B backends: 10/10 realtime client runs, 10/10
realtime server runs, all unit suites, and a concurrency-4 unified smoke with
6/6 blocks and 30/30 successes. The worker perf logs show peak request overlap 4
on both workers. A real-backend cancel reported `SGLANG_ABORT_CONFIRMED
finish=abort`, and the next request succeeded. Evidence:
`/tmp/umminfer-phase5-optimization-evidence/regression-20261113/`.

## Reproduce

The exact commands, environment, and file hashes are in
`/tmp/umminfer-phase5-optimization-evidence/COMMANDS.md` and
`ENVIRONMENT.md`. `SHA256SUMS` covers every evidence file.

```sh
cd /tmp/umminfer-phase5-baseline   # 4b13fe1 with the same harness copied in
python3 -I tests/unified_benchmark.py --evidence-dir <baseline> ...  # serial

cd /private/tmp/umminfer-phase5-joint-optimization
python3 -I tests/unified_benchmark.py --evidence-dir <candidate> ... \
  --asr-concurrency 4 --omni-concurrency 4 --worker-perf-log

python3 -I <evidence>/probe/analysis.py <baseline> <candidate> <evidence>
```

## NOT_COVERED

* Mac unified profile.
* Cancellation latency and cancellation behavior under 4-way concurrency as a
  measured quantity. The existing unit test covers concurrent abort isolation
  indirectly; no live cancel was measured in this run.
* Video and image+audio joint input in the A/B. The unified blocks are
  audio-only. Phase 4 joint-image evidence is unchanged.
* Model revision and weights hashes.
* Exclusive host use. GPUs 0 and 1 were used only by these two backends during
  both runs, but other GPU 2/3 work may have existed.

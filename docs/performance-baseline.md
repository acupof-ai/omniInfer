# Unified performance baseline — real H20B, 2026-11-13

Baseline commit `127907e2a12406af3e1074a5f66144ac6f77b20b`, branch `phase4/bench`.
> **Update 2026-11-13.** The queueing described below comes from one serial
> worker per model. A joint worker-concurrency change removes it: at unified
> concurrency 4, total p95 falls from 55.2 s to 14.9 s and aggregate output
> throughput rises 3.6×. See
> [Joint worker concurrency optimization](joint-optimization.md). The numbers
> below stay as the pre-change baseline.

Every number below comes from one recorded run on real models, real audio, and a
real image. Raw per-request records are in
`/tmp/umminfer-phase4-bench-evidence/raw/`; `summary.json` and `summary.csv` are
derived from them.

## Environment

| Item | Value |
| --- | --- |
| Control plane | Apple Silicon Mac, `umm-infer` built by `zig build` |
| Accelerator host | NVIDIA H20B, 8 × H20 97871 MiB, driver 535.161.08 |
| Container | `linyu-dev` |
| ASR backend | SGLang, `qwen3-asr`, GPU 1, `:31000`, PID 34134 |
| VLM backend | SGLang, `qwen3-omni`, GPU 0, `:31001`, PID 35212 |
| SGLang | `0.5.15.post2.dev55+gc6b67186a` |
| Torch / CUDA | `2.11.0+cu130` / 13.0 |
| ASR model | `Qwen3ASRForConditionalGeneration`, `config.json` SHA256 `76d3ae46…1a478c` |
| VLM model | `Qwen3OmniMoeForConditionalGeneration`, dtype `bfloat16`, `config.json` SHA256 `eab5093d…88f2e0d` |
| VLM server flags | `--mem-fraction-static=0.90 --disable-cuda-graph --disable-fast-image-processor --grammar-backend=none --enable-multimodal` |

Both backends were already running and healthy. The benchmark reused them and
did not start or stop any backend process. Local SSH forwards carried
`127.0.0.1:43100 → 172.17.0.2:31000` and `127.0.0.1:43101 → 172.17.0.2:31001`.

Model revisions are `UNKNOWN`: the local model configs contain no commit hash.
Weights hashes were not recorded. This is a baseline of the deployed snapshot,
not of a named upstream revision.

## Path

```
client (test harness) -> HTTP/SSE -> umm-infer serve
  -> Coordinator / UnifiedEngine -> ZeroMQ -> Python workers -> SGLang -> H20 GPU
```

* `omni-text`, `omni-image`: `serve --entry vlm` → `QwenOmniWorker` → SGLang.
* `asr-en`, `asr-zh`: `serve --entry asr` (unified server) → Qwen ASR worker.
* `unified-*`: `serve --entry asr --vlm-entry vlm` → `UnifiedEngine` → ASR
  worker, then Omni worker with the transcript as the prompt.

Client-side latency includes the SSH tunnel and HTTP/SSE framing. Worker-side
model time is not separately instrumented in this baseline; the difference
between `asr_final_ms` and `first_delta_ms` bounds the VLM prefill plus the
control-plane handoff.

## Inputs

| Input | SHA256 | Detail |
| --- | --- | --- |
| `asr_zh.wav` | `46dbc998…1160dce50` | 16 kHz mono PCM16, 134570 B |
| `asr_en.wav` | `f9b4440…9666cea7f` | 24-bit 48 kHz source, 2172458 B |
| `asr_en_pcm16.wav` | `64fbd9f7…7eedf86d` | transcoded to 16 kHz mono PCM16, 481718 B |
| `man_ironing.png` | `e06917184a00b14abd70cd8ea0ff5dca9abfbbad29f7b25c02f97133d4cd060e` | 570×380 JPEG data in a `.png` name |

The English source is transcoded with ffmpeg; no speech is generated. The
harness rebuilds the WAV container from the real PCM so the server's strict
single-`data`-chunk parser accepts it. The image MIME is sniffed from the bytes.

## Method

* `temperature=0`, `stream=true` for chat, `max_new_tokens=256` for both the ASR
  and the VLM leg. `max_new_tokens=256` meets the 256 minimum and avoids the
  128-token truncation that earlier checks used.
* 3 warm-up requests per scenario, discarded.
* 20 successful samples per scenario. Requests that returned HTTP 200 with
  non-empty output count as success; the attempt cap is `2 × samples + 10`.
* Unified concurrency 1, 2, and 4. Each batch of `N` requests is released at the
  same barrier.
* Percentiles are nearest-rank over the successful samples, with no
  interpolation.
* `output_tokens`: exact `generated_token_count` from SGLang for standalone ASR;
  the Omni model's own tokenizer over the delivered text for chat and unified.
* `tokens_per_s`: output tokens divided by first-delta → last-delta for streams,
  and by total latency for the non-streaming ASR response.
* GPU: `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --loop-ms=2000`,
  windowed to each scenario. `memory.used` is dominated by the resident weights.
* RSS: `ps -o rss=` per local process, once per second, peak per window.

## Results

All values are milliseconds unless stated.

| Scenario | Conc | OK | Total p50 | Total p95 | First delta p50 | First delta p95 | ASR final p50 | ASR final p95 | Tokens mean | tok/s p50 | GPU util | RSS peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| omni-text | 1 | 20/20 | 13734 | 13793 | 547 | 588 | — | — | 256 | 19.41 | GPU0 9.7% | srv 22.7 MB / wrk 30.6 MB |
| omni-image | 1 | 20/20 | 9146 | 9538 | 617 | 669 | — | — | 169 | 19.89 | GPU0 10.1% | srv 25.0 MB / wrk 30.8 MB |
| asr-en | 1 | 20/20 | 656 | 723 | — | — | — | — | 46 | 69.96 | GPU1 18.7% | srv 32.7 MB / wrk 33.2 MB |
| asr-zh | 1 | 20/20 | 530 | 547 | — | — | — | — | 11 | 20.72 | GPU1 3.0% | srv 50.2 MB / wrk 27.6 MB |
| unified-en | 1 | 20/20 | 14058 | 14252 | 1259 | 1330 | 709 | 775 | 256 | 20.03 | GPU0 10.0% | srv 35.7 MB / wrk 33.2 + 30.2 MB |
| unified-en | 2 | 20/20 | 14383 | 27336 | 1731 | 14643 | 951 | 1386 | 256 | 20.08 | GPU0 10.0% / GPU1 1.2% | srv 41.3 MB / wrk 26.5 + 20.4 MB |
| unified-en | 4 | 20/20 | 27821 | 54720 | 14824 | 41859 | 1345 | 2574 | 256 | 19.91 | GPU0 10.2% / GPU1 1.2% | srv 52.3 MB / wrk 27.9 + 21.2 MB |
| unified-zh | 1 | 20/20 | 14337 | 14814 | 1103 | 1215 | 558 | 618 | 256 | 19.33 | GPU0 9.7% | srv 49.5 MB / wrk 27.1 + 21.3 MB |
| unified-zh | 2 | 20/20 | 14613 | 28472 | 1205 | 15084 | 652 | 1089 | 256 | 19.15 | GPU0 9.5% / GPU1 0.3% | srv 34.7 MB / wrk 24.9 + 21.3 MB |
| unified-zh | 4 | 20/20 | 29419 | 56290 | 15489 | 42889 | 1123 | 2139 | 256 | 19.14 | GPU0 9.4% / GPU1 0.2% | srv 37.0 MB / wrk 24.9 + 21.3 MB |

Peak GPU memory in every window: GPU 0 `89739 MiB`, GPU 1 `87451 MiB`.

Failure count: 0 in all ten blocks. Finish reasons: `stop` for every ASR and
image request; `length` for all 20 `omni-text` and all 120 unified requests,
which means the answer reached the 256-token cap. Those answers are real but
truncated at the cap; `tokens_per_s` is unaffected because it is measured over
the delivered stream.

Coefficient of variation of total latency: `0.004–0.044` at concurrency 1,
`0.33` at concurrency 2, `0.45` at concurrency 4. The high values at
concurrency above 1 are not measurement noise: each worker processes one
request at a time, so a batch of 4 finishes in four waves. The first-delta
values cluster at about 1.2 s, 14.5 s, 28 s, and 42 s in `unified-en` at
concurrency 4. Every sample is a real request; the distribution is multi-modal
by construction.

Concurrency raises latency and leaves throughput flat: about 20 output tokens/s
per VLM leg at concurrency 1, 2, and 4, because one Omni worker serves one
request at a time. Concurrency 1 is the recommended mode for this deployment.

## What the numbers do not cover

* Mac unified profile: `NOT_COVERED`. This run used the Mac only as the control
  plane.
* This baseline ran at `127907e`, before the Phase 4 joint-input and streaming
  commits. Image through `UnifiedEngine`, joint image + audio, and realtime
  partials were not exercised by this benchmark; their separate H20B evidence is
  in [README](../README.md#run-real-qwen-asr---vlm). ASR partials through the
  unified chat route remain `NOT_COVERED`.
* Cancellation latency, queue depth, and per-worker model-time attribution were
  not measured by this benchmark. Cancellation correctness is covered by the
  separate Phase 4 cancel/failure evidence.
* Model revision and weights hashes: `UNKNOWN`.
* The host was not exclusive for the whole window. Other sessions ran separate
  SGLang instances on GPU 2/3 during part of the run. GPU 0/1 were used only by
  this benchmark's two backends, and GPU utilization on those two GPUs was
  idle between requests.
* GPU utilization is sampled every 2 s, so a short burst can be missed.

## Reproduce

See `COMMANDS.md` in the evidence directory. The summary can be rebuilt from
raw records with:

```sh
python3 -I -c "
import json, math
rows=[json.loads(l) for l in open('raw/unified-en-c4.jsonl')]
ok=[r for r in rows if r['ok']]
v=sorted(r['total_ms'] for r in ok)
print(len(ok), v[max(1, math.ceil(0.95*len(v)))-1])
"
```

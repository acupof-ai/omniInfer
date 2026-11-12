# MLX-LM on Apple Silicon

The optional Python stage runs a persistent MLX-LM model and connects to the
C++ coordinator over ZeroMQ. The C++ process still owns the OpenAI-compatible
HTTP API. The stage loads the model once, returns JSON completions, and emits
incremental SSE deltas. It does not replace shared memory or CUDA IPC between
C++ stages.

## Install

Use an Apple Silicon Mac with Metal support. Keep the Python dependencies in a
virtual environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install mlx-lm pyzmq msgpack
zig build -Doptimize=ReleaseFast -Ddeps-prefix=/opt/homebrew
```

The C++ build and tests do not import MLX. Machines without MLX can still use
the echo stage and the existing CPU/CUDA runtime.

## Run

This example uses a small 4-bit instruct model for a functional check. The
first run downloads it from Hugging Face. Use a local snapshot path or a larger
MLX instruct model for normal use.

```sh
MODEL=mlx-community/Qwen2.5-0.5B-Instruct-4bit
IPC_DIR=$(mktemp -d /tmp/umminfer-mlx.XXXXXX)

./zig-out/bin/umm-infer serve --host 127.0.0.1 --port 30000 \
  --model "$MODEL" --entry mlx --entry-endpoint "ipc://$IPC_DIR/entry" \
  --completion "ipc://$IPC_DIR/completion" --abort "ipc://$IPC_DIR/abort"
```

In another terminal, use the same `MODEL` and `IPC_DIR` values:

```sh
.venv/bin/python -I python/umminfer_mlx_stage.py \
  --name mlx --recv "ipc://$IPC_DIR/entry" \
  --coordinator "ipc://$IPC_DIR/completion" --abort "ipc://$IPC_DIR/abort" \
  --model "$MODEL"
```

After loading the model and initializing both ZeroMQ control inputs, the stage
sends `{"type":"stage_ready","from_stage":NAME}` to the coordinator and prints
its `ready` line. The `serve` command holds requests locally until that message.
Requests may be sent before model loading finishes. An abort of a queued request
is terminal and removes its queued Submit. For a submitted request, the stage
accepts the ordered PUSH abort and the PUB supplement. There is no startup sleep
or timeout fallback.

The CLI `serve` and `stage` commands require this protocol. An older external
stage without `stage_ready` does not work with this startup path. Embedded users
retain the legacy behavior by default: `Coordinator::register_stage(name,
endpoint)` does not wait for ready. Pass `true` as its third argument and set
`Stage::Config::announce_ready = true` to enable the protocol explicitly.
Register all stages before `Coordinator::start()`. Dynamic registration and
re-registration are rejected because readiness belongs to one startup lifecycle.
Caller-supplied request IDs must remain unique for the coordinator's lifetime,
including after `release()`. A delayed abort frame can outlive the result record.

For an offline run with an already cached model, set `HF_HUB_OFFLINE=1` on the
stage command.

JSON completion:

```sh
curl -s http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Give one Metal inference tip."}],"max_tokens":64,"temperature":0}'
```

Incremental SSE:

```sh
curl -N http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Count to ten."}],"max_tokens":64,"stream":true}'
```

Stop the HTTP server and the MLX stage with SIGTERM or Ctrl-C. They are
separate processes; stopping one does not automatically stop the other.

## Parameters and limits

- `--model` accepts a Hugging Face model id or a local model directory.
- Requests support `max_tokens`, `temperature`, and `stream` through the
  existing HTTP API. `max_tokens` maps to MLX-LM's `max_tokens` argument.
- The stage applies the model tokenizer's chat template with a generation
  prompt. The selected model must provide a compatible text chat template.
- One stage executes one request at a time. Concurrent HTTP requests queue.
  Dynamic batching is not implemented.
- Aborts and client disconnects are observed at token boundaries. An in-flight
  prefill or model call is not interrupted inside MLX-LM.
- A failed request returns a failed completion without unloading the model or
  stopping the worker. A malformed control message without a usable request id
  is logged and discarded.
- This backend is text-only. It does not add multimodal input, tool calling, or
  response-usage accounting to the existing HTTP API.

## Verify

The unit tests use a fake generator and do not need MLX or a downloaded model:

```sh
zig build mlx-test -Ddeps-prefix=/opt/homebrew
zig build test -Ddeps-prefix=/opt/homebrew
zig build smoke -Ddeps-prefix=/opt/homebrew
```

The real-model smoke requires an existing local MLX model directory and the MLX
Python dependencies. It forces Hugging Face offline mode and is not part of the
default test or smoke targets. Replace the example model path with a local
snapshot:

```sh
python3 -I tests/mlx_smoke.py \
  --server zig-out/bin/umm-infer \
  --stage python/umminfer_mlx_stage.py \
  --model /absolute/path/to/mlx-model
```

The smoke checks JSON, SSE, abort, a disconnected SSE client, a following request,
and four concurrent requests. It reports first-token latency where the streaming
API exposes it, total latency, output tokens, and end-to-end output tokens per
second. Do not compare model execution time with echo-compute time.

## Performance measurement

Numbers are valid only in an exclusive window. Another process that uses the GPU
or CPU changes every metric below. The harness holds a non-blocking `flock` on
`/tmp/umminfer-mlx-perf.lock` for the whole run, so a second harness stops at
startup. It also reads `ps` before the run and before and after every mode, and
fails if it finds another harness, another stage copy, or the baseline and
candidate stages named on its own command line. Its own process tree is
excluded. A failed `ps` is a failure, not a pass. There is no override flag.

### Fixed configuration

`tests/mlx_perf.py` pins one model snapshot, one prompt, `temperature=0`, one
`max_tokens` budget, and one sample count. A baseline run and a candidate run
use the same command line. The report records the configuration, the model
`model.safetensors` SHA256, the binary SHA256, the stage SHA256, and the MLX and
MLX-LM versions.

### Run

```sh
python3 -I tests/mlx_perf.py --mode all --role baseline \
  --samples 20 --warmup 2 --max-tokens 64 --out /tmp/mlx-perf-baseline.json
```

`--mode` selects one level or all three:

| Mode | Path | Isolates |
| --- | --- | --- |
| `direct` | MLX-LM in the harness process, through the stage's own `load_mlx_backend` | Prefill, decode, model load |
| `stage` | Harness, ZeroMQ, real Python stage, both `sse` and `json` transports | Python stage and ZeroMQ cost |
| `e2e` | Harness over HTTP JSON and SSE, plus the real CLI JSON and SSE | Coordinator, HTTP, SSE cost |

`direct` calls the loader of the stage under test, so the tokenizer cache is
present or absent in exactly the way the measured stage has it. It does not
import MLX-LM on its own path.

Each mode reports the same metrics. `attribution` subtracts like transports:
`direct` against `stage.sse`, `stage.json` against `e2e.json`, `e2e.sse` against
`stage.sse`, and each CLI row against the matching HTTP row. With `--mode all`,
a missing or skipped path fails the run. A single-mode report is partial and
cannot be used for a complete comparison.

### Metrics

| Metric | Definition |
| --- | --- |
| `ttft_ms` | Request send to first content delta. Streaming rows only |
| `total_ms` | Request send to last token |
| `generation_tokens` | The count MLX-LM reports, not an estimate from text |
| `decode_tokens_per_s` | MLX-LM decode rate after the first token |
| `output_tokens_per_s` | `generation_tokens / (total_ms / 1000)`. End-to-end throughput |
| `prompt_tokens_per_s` | Prefill rate as MLX-LM reports it |
| `peak_memory_gb` | `mx.get_peak_memory()`. The Metal peak of the stage process, not RSS |

`cold` is the first real request after the measured instance is ready: `/ready`
answers 200 for `e2e`, `stage_ready` for `stage`, the model load for `direct`.
The flag belongs to the instance, not to the path. Every later transport on the
same warm instance reports `cold` as null, so a second path cannot claim a cold
sample. `warm` is every sample after `--warmup` iterations. Warmup samples are
excluded from every median. A `cv` above 0.10 prints a warning.

### Stage measurement log

`python/umminfer_mlx_stage.py --perf-log PATH` appends one JSON line per model
load and per successful request. The line holds the stage-side timings,
prompt and generation token counts, MLX token rates, finish reason, and peak
memory. The stage writes the request line before it sends `complete`, so a
client that sees the completion can read the record without a race. The option
is off by default and adds no file access when absent.

A stage without this option cannot report peak memory or MLX token rates. The
HTTP API does not expose them.

### Reference metric

The reference metric for an optimization claim is the warm median
`output_tokens_per_s` of the HTTP JSON path. A claim of a given improvement
requires:

- the same command line, model, prompt, `max_tokens`, and machine;
- `--samples` of at least 20 and a reported `cv`;
- both runs in exclusive windows, with the JSON reports kept.

`ttft_ms`, `total_ms`, and the SSE rows are reported with it. They must not
regress. JSON has no first-token boundary, so its `ttft_ms` is empty by
definition.

### Improvement formula

Let `B` be the baseline warm median `output_tokens_per_s` of the HTTP JSON path
and `C` the candidate value from the same configuration:

```
improvement_percent = (C / B - 1) * 100
ratio               = C / B
```

The target is `ratio >= 1.5`. Use the 64-token run. The 16-token run is
supplementary and does not replace it. Concurrent aggregate throughput does not
replace it.

### Acceptance gates

A candidate run is valid only when every gate below holds. A failed gate
invalidates the numbers; it does not get a footnote.

| Gate | Rule |
| --- | --- |
| Output identity | Every sample of every path reports one `output_sha256`. All paths within a report agree, and the harness fails that run on a mismatch. Automated baseline/candidate acceptance must also require the same hash across both reports. |
| Main metric | `ratio >= 1.5` on the 64-token warm median `output_tokens_per_s` of HTTP JSON. |
| TTFT | The candidate warm median `ttft_ms` of the SSE rows is not above the baseline median by more than 5%. JSON rows have no TTFT. |
| Total latency | The candidate warm median `total_ms` of every row is not above the baseline median by more than 5%. |
| Memory | The candidate warm median `peak_memory_gb` is not above the baseline median by more than 5%. The metric covers the stage process only. |
| Stability | Report the `cv` of each median. A `cv` above 0.10 means the window was not quiet; find the competing process and repeat both runs. Do not drop samples. |

The 5% band covers run-to-run noise on this machine. A result inside the band is
reported as unchanged, not as a regression. A result outside the band fails the
gate.

`tests/mlx_perf_compare.py` evaluates every gate above on two reports. It
refuses a partial report: both inputs must be `--mode all`, `completed=true`,
with no skipped path, no foreign process, and unchanged artifact hashes. It also
requires one identical `output_sha256` per path across both reports, the same
binary, model, model weights, prompt, `max_tokens=64`, at least 20 samples, and
matching Python and package versions. A failed gate is named and the command
exits non-zero.

### Final benchmark command

One window, baseline first, then candidate. Pass the same binary and the same
arguments to both. There is no busy override flag. If the lock is held or a
foreign process is visible, fix the window instead of rerunning with a flag.

The binary is the one the master provides for the exclusive window. Do not
reuse an earlier build path as the candidate. Before the first run, record the
SHA256 of all four artifacts:

```sh
shasum -a 256 "$UMMINFER_BINARY" \
  /tmp/umminfer-stage-baseline.py \
  python/umminfer_mlx_stage.py \
  tests/mlx_perf.py
```

```sh
python3 -I tests/mlx_perf.py --mode all --role baseline \
  --samples 20 --warmup 2 --max-tokens 64 \
  --binary "$UMMINFER_BINARY" \
  --stage /tmp/umminfer-stage-baseline.py \
  --out /tmp/umminfer-mlx-exclusive-baseline.json

python3 -I tests/mlx_perf.py --mode all --role candidate \
  --samples 20 --warmup 2 --max-tokens 64 \
  --binary "$UMMINFER_BINARY" \
  --out /tmp/umminfer-mlx-exclusive-candidate.json
```

Keep both reports and their console logs. The reports record the SHA256 of the
model, the binary, the stage, and the harness, so a mismatched pair is visible.

### Measured stage bottleneck

A labelled probe on 2026-11-11 found one dominant removable cost. MLX-LM builds
a streaming detokenizer on every `stream_generate` call, and its token map
comes from `tokenizer.vocab`. transformers rebuilds that vocabulary on each
access: 48 ms of a 104 ms request for this model. The stage now answers
`get_vocab` from a cache built once at load. The same 8-token request then took
42.9 ms.

The probe ran with other work on the machine and is not a baseline. The
baseline and the candidate must be measured with this harness in exclusive
windows.

Steady-state decode measured 400 to 490 output tokens per second. That rate
bounds the gain at long token budgets: the removed cost is constant per
request, so its relative gain falls as `max_tokens` rises.

### Baseline and candidate comparison

The baseline differs from the candidate in one variable: the vocabulary map is
rebuilt on every access. Do not copy the candidate as the baseline, and do not
reuse an older stage that also lacks the abort, metadata, TTFT, and queue
accounting changes. Derive the baseline from the final candidate and remove only
the cache:

```sh
/opt/homebrew/opt/python@3.11/bin/python3.11 -I - <<'PY'
from pathlib import Path
src = Path("python/umminfer_mlx_stage.py").read_text()
old = '''        def build(*args, **kwargs):
            stats["builds"] += 1
            return rebuild(*args, **kwargs)

        vocabulary = build()

        def cached(*args, **kwargs):
            stats["accesses"] += 1
            return vocabulary
'''
new = '''        def cached(*args, **kwargs):
            stats["accesses"] += 1
            stats["builds"] += 1
            return rebuild(*args, **kwargs)
'''
assert src.count(old) == 1
Path("/tmp/umminfer-stage-baseline.py").write_text(src.replace(old, new))
PY
diff -u python/umminfer_mlx_stage.py /tmp/umminfer-stage-baseline.py
shasum -a 256 python/umminfer_mlx_stage.py /tmp/umminfer-stage-baseline.py
```

The counters stay in both files: the baseline increases `builds` and `accesses`
on every call, the candidate builds once at load and only increases `accesses`.

Run the two commands in "Final benchmark command", then compare the reports.
Freeze the environment first; the file is created on the first call and
required to match on every later call:

```sh
python3 -I tests/mlx_perf_compare.py \
  /tmp/umminfer-mlx-exclusive-baseline.json \
  /tmp/umminfer-mlx-exclusive-candidate.json \
  --manifest /tmp/umminfer-mlx-manifest.json \
  --out /tmp/umminfer-mlx-comparison.json
```

The manifest records the interpreter absolute path and the `msgpack`, `mlx`,
`mlx-lm`, and `transformers` versions. The comparison prints one line per gate,
including the main ratio, TTFT, total latency, memory, stability, and output
identity, and exits non-zero if any gate fails.

### Optimization candidates (static analysis)

Each row names the source that causes the cost, the level it affects, and the
smallest experiment that can decide it. None of these numbers is measured until
a run exists in an exclusive window.

| Candidate | Source | Level | Expected effect | Correctness risk | Minimal A/B |
| --- | --- | --- | --- | --- | --- |
| Tokenizer vocabulary cache | `mlx_lm` builds a `StreamingDetokenizer` per `stream_generate`; its token map calls `tokenizer.vocab`, which `transformers` rebuilds per access. Cached in `python/umminfer_mlx_stage.py` at load | stage, and every level above it | Removes one fixed cost per request. Measured: 104 ms to 42.9 ms on an 8-token request | None for a fixed model. The guard makes it a no-op on another `mlx-lm` layout | Baseline stage against candidate stage |
| Detokenizer construction residue | Same constructor, after the vocabulary is cached | stage | Unmeasured; bounded by the former 48 ms minus the vocabulary cost. Likely a few milliseconds | Reusing a stateful detokenizer can carry partial UTF-8 state into the next request | Time the constructor alone in a probe before changing the stage |
| Sampler construction | `make_sampler` runs per request | stage | Below 1 ms | None | Only if a profile shows it |
| Prompt-cache reuse | `stream_generate` allocates a new KV cache per call, so prefill repeats | stage decode path | About 18 ms at the measured 1900 prompt tokens/s for this 35-token prompt | It speeds repeated requests only. The rules require a varied-prompt control run to prove the gain is not request caching | Varied prompts, same token budget |
| Compiled model step | `mlx_lm` calls the model directly; `mx.compile(fun, inputs, outputs, shapeless)` on the installed version rejected a `state=` argument | decode | Decode rate is the dominant term. A 1.5x decode rate alone would reach the target | Compilation can change numerics, and at `temperature=0` a changed number can change the token. The output hash gate decides | Output hash, then decode rate |
| One message per token | The stage emits one `stream` message per delta and the server writes one SSE chunk per message | e2e streaming path | Instrumented stage and e2e differ by about 9 ms in total, so the ceiling is small | Coalescing can delay the first delta if it waits for a full batch | SSE TTFT and total only |

The measured decode rate bounds the target. At 450 output tokens per second, a
64-token request spends about 142 ms in decode and about 18 ms in prefill. The
baseline adds the removed vocabulary cost and request overhead. The vocabulary
cache alone therefore gives roughly +27% at 64 tokens, not +50%. Reaching
`ratio >= 1.5` needs a decode-level change or a lower remaining overhead; the
table shows both.

### Limits

- Peak memory covers the stage process only. The C++ server and the CLI are
  measured separately.
- One stage executes one request at a time. Concurrent wall time measures
  queueing, not throughput, and is not a target metric.
- MLX-LM builds a fresh KV cache per `stream_generate` call. There is no
  cross-request prompt reuse today; an optimization that adds one must repeat
  the run with varied prompts to show the gain is not prompt-cache reuse.
- The harness measures the path it drives. It does not measure a remote client
  or a real network.

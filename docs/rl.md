# RL architecture — phase 1: rollout, reward interface, training worker

Phase 1 builds the architecture only. No training runs, no model weights are
produced, and no RL result is claimed. The pieces delivered are:

1. Rollout recording through the existing public HTTP/SSE exit, in a process
   separate from the server.
2. A complete multimodal trajectory export (JSONL) with ordered ASR/VLM
   events, terminal state, cancellation and error records, and content
   hashes.
3. A pluggable reward interface with no business reward implemented.
4. A training worker that runs as a separate process behind a versioned
   stdio protocol and writes versioned checkpoint metadata.

The serving process is not modified. No second inference engine exists. The
new code files are:

| File | Responsibility |
| --- | --- |
| `python/umminfer_rl_trajectory.py` | Out-of-process recorder, SSE parsing, input references, JSONL export |
| `python/umminfer_rl_reward.py` | Reward abstract class, null reward, plugin loader |
| `python/umminfer_rl_worker.py` | Worker process, NDJSON protocol, checkpoint store with hash chain |
| `tests/rl_architecture_test.py` | Deterministic tests, no model and no server binary required |

## Why the recorder is a client, not a server hook

The recorder calls the public endpoint exactly like any other client. The
alternative, attaching to the C++ `Coordinator` sink seam, requires changing
the serving startup path and puts recording logic inside the serving
process. The phase-1 constraint is the opposite boundary: training-adjacent
code stays out of serving, and the Python stage workers are not touched.

Consequence: only traffic driven through the recorder is captured. A
passive bump-in-the-wire tap that captures all server traffic is future work
and would be a separate sidecar process. It is not built in phase 1.

## Trajectory export

One JSON object per line, one line per terminal request (one turn). Every
record has `schema_version: 1`.

```json
{
  "schema_version": 1,
  "session_id": "sess-1",
  "request_id": "http-1",
  "turn_index": 0,
  "endpoint": "/v1/chat/completions",
  "inputs": [
    {"kind": "text", "ref": null, "media_type": null,
     "sha256": "…", "text": "transcribe this"},
    {"kind": "audio", "ref": "<sha256>.wav", "media_type": "audio/wav",
     "sha256": "…"},
    {"kind": "image", "ref": "<sha256>.png", "media_type": "image/png",
     "sha256": "…"},
    {"kind": "image", "ref": "https://example.test/pic.png",
     "media_type": null, "sha256": null}
  ],
  "request_params": {"model": "…", "stream": true},
  "events": [
    {"seq": 0, "type": "asr_partial", "server_seq": 0, "stage": "asr",
     "is_final": false, "text": "…"},
    {"seq": 1, "type": "asr_final", "server_seq": 1, "stage": "asr",
     "is_final": true, "text": "…"},
    {"seq": 2, "type": "vlm_delta", "server_seq": 2, "stage": "vlm",
     "is_final": false, "text": "…"},
    {"seq": 3, "type": "finish", "finish_reason": "stop"}
  ],
  "terminal_state": "completed",
  "error": null,
  "output_text": "…",
  "output_sha256": "…"
}
```

### Input references

Input bytes never enter the JSONL.

- Text parts are inlined and hashed over their UTF-8 bytes.
- WAV audio and `data:` image bytes are decoded and written to the artifact
  directory under `<sha256>.<ext>`. Identical media is stored once. The
  record keeps the relative file name and the SHA-256 of the decoded bytes.
- Remote `image_url` values stay URL references with `sha256: null`. The
  recorder does not fetch them. Reproducibility of those records depends on
  the URL continuing to serve the same bytes; that is the documented cost of
  referencing instead of inlining.

### Events and terminal states

- Recorder `seq` is gap-free from 0 in arrival order. Unified events also
  keep the server-assigned sequence under `server_seq` for cross-checking;
  the recorder sequence is the ordering authority.
- Event types come from the server vocabulary: `asr_partial`, `asr_final`,
  `vlm_delta`, `complete`, `error`, `cancelled`, plus `output_delta` and
  `finish` for plain OpenAI chunks.
- `output_text` concatenates the final ASR transcript and VLM deltas.
  Provisional `asr_partial` text is excluded because the final event
  replaces it. `output_sha256` hashes that exact string.
- Terminal states: `completed`, `failed`, `cancelled`.
  - A clean stream ends with `[DONE]` and no error frame.
  - An error frame, an HTTP error body, or EOF without `[DONE]` is `failed`.
    The EOF message is `stream ended before [DONE]`, identical to
    `tests/unified_benchmark.py`.
  - A transport failure (refused connection, dropped socket, undecodable
    response bytes) is also `failed` with a `transport error:` message. Every
    attempted turn stays in the export rather than escaping the recorder.
  - `cancelled` is client-attributed. The supported path is the
    `on_request_id` callback on `record_chat_completion()`: when the first
    chunk reveals the server request id, the caller invokes `POST /abort`
    from another thread. The server answers 200 when the coordinator accepts
    the abort (fixed-upstream deployments answer 501 `unsupported_route`,
    in which case the abort is not backend-confirmed and the turn must not
    be labelled cancelled), then emits a final chunk whose `unified.kind` is
    `cancelled` followed by `[DONE]`. The parser maps that terminal to
    `cancelled`. Even a 200 is acceptance, not proof that the model forward
    already stopped.

### Realtime WebSocket exit

`record_realtime()` takes a list of events observed by a WebSocket client.
Each event carries `direction: inbound|outbound`, because the realtime
protocol has error events in both directions. A `transcription.completed`
outbound event completes the turn, an outbound error fails it, and neither
appearing means a client-observed cancellation. Input references are passed
in explicitly through the same shape as `input_references()`; the rule for
concatenating and hashing inbound PCM16 appends is deferred with the live
adapter rather than guessed in phase 1.

The live websockets adapter is **NOT_COVERED** in phase 1: no model or server
binary is available to the deterministic tests, so only the event-list entry
point is tested. A live `/v1/realtime` recording run must be added when a
real environment is available.

## Determinism

- The three modules import nothing time- or randomness-based. There are no
  timestamps, uuids, or host names in any record or checkpoint.
- All hashes use `json.dumps(..., sort_keys=True, separators=(",", ":"),
  allow_nan=False)` over UTF-8 bytes. `allow_nan=False` rejects NaN and
  Infinity, which are not valid JSON.
- Turn indexes count from 0 per session inside one recorder process. A
  trajectory JSONL file has one writer; multiple recorder processes must not
  append to the same path.
- The same request bodies and the same canned response bytes produce
  byte-identical records across runs, which the tests assert.

## Reward interface

`umminfer_rl_reward.Reward` has one method, `score(record)`, returning a
float or `None`. A reward reads only the trajectory record. It has no access
to serving memory.

`NullReward` is the default and returns `None`. The score is stored as
JSON null, never as zero, so "reward missing" and "reward is zero" stay
distinguishable. A concrete reward is loaded from an external module with
`load_reward("module:attr")`; the business reward does not live in this
repository's serving code.

## Training worker and protocol

The worker is a separate operating-system process:

```
python3 -I python/umminfer_rl_worker.py --state-dir <dir>
```

It speaks one JSON object per line on stdin/stdout.

| Request | Response |
| --- | --- |
| `{"type": "ping"}` | `{"type": "pong", "worker_protocol_version": 1}` |
| `{"type": "train", "task_id": "…", "trajectory_path": "…"}` | `{"type": "train_result", ...}` or `{"type": "error", ...}` |
| Any other `type` | `{"type": "error", "error": "unknown request type: …"}` |

Unknown fields on known messages are ignored so optional fields can be added
without a protocol version bump.

A `train` request does not train. It parses the trajectory export and
rejects it on any semantic defect: missing integer `schema_version`,
unknown terminal state, an event `seq` gap, or an `output_sha256` that does
not recompute. A rejected request consumes no checkpoint sequence.

### Checkpoint metadata and hash chain

Checkpoints are written to `<state-dir>/checkpoints/` as
`ckpt-<sequence:08d>.json` plus a `.sha256` sidecar. Sequence starts at 0
and is strictly monotonic. There is no mutable state file: on startup the
worker scans the directory, verifies every sidecar, and walks the
`base_checkpoint_sha256` chain. A tampered manifest fails verification
before any new checkpoint is written.

```json
{
  "checkpoint_schema_version": 1,
  "worker_protocol_version": 1,
  "sequence": 1,
  "algorithm": null,
  "weights_ref": null,
  "reward": {"name": "constant-one"},
  "source": {"trajectory_path": "…", "record_count": 2,
             "trajectory_sha256": "…"},
  "base_checkpoint_sha256": "<sha256 of ckpt-00000000.json>"
}
```

`algorithm` and `weights_ref` are explicit empty slots. One checkpoint
directory has one worker writer; concurrent writers are a deployment error
surfaced as a sequence conflict, not a supported mode. An export that mixes
two different non-null reward names, or carries a non-object reward block,
is rejected rather than attributed ambiguously.

## How PPO/DPO/GRPO plug in later

The phase-1 shapes already contain the slots an algorithm fills; no
algorithm code exists now.

1. Implement `Reward` in an external module and load it through
   `load_reward`.
2. Add a worker request verb (for example `train_step`) carrying an
   `algorithm` name and a config object. Fill `algorithm` and `weights_ref`
   in the manifest and point `weights_ref` at the produced weights. Bump
   `checkpoint_schema_version` if the manifest shape changes incompatibly,
   and document the migration here. Hyperparameters have no phase-1 slot;
   they arrive with the first real algorithm rather than as a reserved
   empty field.
3. PPO consumes `events` plus per-turn reward scores; DPO consumes paired
   records distinguished by session/request ids; GRPO groups records by
   `session_id`. The export already carries those keys. Nothing in this list
   is implemented ahead of a defined objective.

## Open decisions that block real training

These are user inputs. Phase 1 deliberately does not invent defaults.

1. **Training objective.** PPO, DPO, GRPO or another algorithm is not
   chosen.
2. **Dataset.** Which recorded sessions become training data, and any
   held-out split, is undefined.
3. **Reward function.** No business reward exists. `NullReward` records the
   missing-score shape only.
4. **Compute.** Where the worker runs, GPU allocation, and the optimizer and
   its state are undefined. The worker currently writes metadata only.

Two roadmap items are deferred and recorded here rather than silently
dropped:

5. **Per-event timestamps** are listed in `docs/roadmap.md` phase 6. The
   phase-1 export has no time fields at all, because wall-clock values break
   byte-level determinism. Adding them requires a decision on a stable
   clock source and on whether timestamps participate in hashes.
6. **The Replay acceptance gate** is not met. The export stores normalized
   events, not the raw SSE payload bytes, so byte-level replay/diff is not
   possible. A future raw-payload blob store would close this; it has no
   consumer in phase 1 and is not built.

The roadmap Isolation gate (training load versus serving latency) cannot be
measured until real training exists. No RL capability beyond recording and
metadata should be claimed.

## Tests

```
python3 -I tests/rl_architecture_test.py
```

The suite uses a loopback HTTP server with canned JSON/SSE responses and a
real subprocess for the worker boundary. It covers text/audio/image input
references, ordered unified streaming events, EOF truncation, error frames,
the server `cancelled` SSE terminal, a client abort driven through the real
`/abort` exit (including the 501 case), transport failures, realtime event
derivation, null-versus-zero reward handling, NaN rejection, malformed and
mixed reward blocks, checkpoint ordering and both hash-chain tampering
paths, invalid-export rejection, and the subprocess protocol.

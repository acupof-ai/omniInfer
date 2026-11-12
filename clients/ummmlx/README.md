# ummmlx

A small native macOS SwiftUI client for a local ummInfer MLX server.
It has no third-party runtime dependencies.

## Features

- Configurable base URL and model, persisted with `UserDefaults`.
- Current in-memory conversation with completed-turn context.
- Real incremental `POST /chat/completions` SSE text.
- Send, Stop, New chat, completion state, and visible errors.
- Light, dark, and system appearance.
- Requests use `temperature=0`, `max_tokens=64`, and no `stream_options`.

The transport uses `URLSession.AsyncBytes`. The SSE parser accepts arbitrary byte
boundaries, LF/CRLF, comments, and multi-line data. Completion requires `[DONE]`.
Plain HTTP is limited to loopback and redirects are disabled. Stop cancels the
HTTP task; the server observes disconnect at a token boundary.

## Build and test

```sh
cd clients/ummmlx
swift test
Scripts/build.sh release
open build/ummmlx.app
```

The build creates `build/ummmlx.app` without signing, notarization, or a DMG.
The raw executable is `build/ummmlx.app/Contents/MacOS/ummmlx`.

## Required paths

Two inputs have no built-in path: the `umm-infer` server binary and the MLX
model snapshot directory. The repository does not contain them. Get them from
your build output and model cache, then export both:

```sh
export UMMMLX_SERVER=/path/to/umm-infer
export UMMMLX_MODEL=/path/to/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/<revision>
export PYTHON=$(command -v python3)   # must have mlx-lm installed
```

`UMMMLX_MODEL` must be the snapshot directory itself: it must contain
`config.json`, `tokenizer.json`, `tokenizer_config.json` with a chat template,
and one `model.safetensors`.

The acceptance run pins the server binary to SHA256
`d51384fd053fb78d4ff0fa0b49a912d33639f225d727ea61991a9feb06105f01`. A binary
with another hash needs `--allow-unfrozen`, and the report then records
`"frozen": false`.

## Start the real local service

Use separate terminals from the repository root. Terminal 1:

```sh
IPC_DIR=$(mktemp -d "${TMPDIR:-/tmp}/ummmlx-ipc.XXXXXX")
"$UMMMLX_SERVER" serve \
  --host 127.0.0.1 --port 30000 --model Qwen2.5-0.5B-Instruct-4bit \
  --entry mlx --entry-endpoint "ipc://$IPC_DIR/entry" \
  --completion "ipc://$IPC_DIR/completion" --abort "ipc://$IPC_DIR/abort"
```

Terminal 2, with the same `IPC_DIR`:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PYTHON" -I python/umminfer_mlx_stage.py \
  --name mlx --recv "ipc://$IPC_DIR/entry" \
  --coordinator "ipc://$IPC_DIR/completion" --abort "ipc://$IPC_DIR/abort" \
  --model "$UMMMLX_MODEL"
```

After `/ready` returns 200:

```sh
open clients/ummmlx/build/ummmlx.app
```

Set Base URL to `http://127.0.0.1:30000/v1` and Model to
`Qwen2.5-0.5B-Instruct-4bit`. Enter a prompt and select Send. Stop cancels the
active stream. New chat clears the current in-memory session.

## Automated checks

Static light/dark captures do not contact a server. Automation does not persist
endpoint, model, or appearance changes:

```sh
build/ummmlx.app/Contents/MacOS/ummmlx --automation \
  --output=/tmp/ummmlx-static.json --screenshots=/tmp/ummmlx-static
```

The real runner takes the shared `/tmp/umminfer-mlx-perf.lock`, checks process
exclusivity, starts the configured server and the current stage over a private
IPC directory, and drives the GUI. One run covers:

1. Stop after the first real delta, then New chat.
2. Two complete turns in one session. The second request must carry the first
   exchange, so the request history is `user`, `assistant`, `user`.
3. One rejected request with an unserved model name. The frozen server answers
   400 and the client must show that message.
4. Light, dark, and error screenshots, all distinct.

The output directory must not exist. Run the runner with an interpreter that has
`mlx-lm`, `msgpack`, and `pyzmq`:

```sh
OUT=/tmp/ummmlx-e2e-run
"$PYTHON" -I Scripts/real_e2e.py --out "$OUT" \
  --server "$UMMMLX_SERVER" --model "$UMMMLX_MODEL" \
  > "$OUT.console.log" 2>&1 &
E2E_PID=$!
wait "$E2E_PID"
REAL_EXIT=$?
printf 'REAL_EXIT=%s\n' "$REAL_EXIT"
```

`--server`/`--model` fall back to `UMMMLX_SERVER`/`UMMMLX_MODEL`.
`--served-model` defaults to `Qwen2.5-0.5B-Instruct-4bit` (`UMMMLX_SERVED_MODEL`).
The rejected model name is `<served-model>-not-served` unless
`--error-model` sets one.

See `report.json`, `automation.json`, `server.log`, `stage.log`, `app.log`, and
`screenshots/` under that directory. `report.json` carries the before/after
hashes of the server, stage, app, weights, runner, perf helper, model inputs,
and the screenshot SHA256 values. Each captured process log is capped at
64 KiB. A nonzero app exit is preserved as `REAL_EXIT`.

## Voxt reuse

Voxt commit `5645daf0480e9bf572b45a2bba8b8e2f1e642daa` supplied the SSE/error
handling structure and the neutral/teal design tokens. Modified Swift ports and
unmodified reference sources are identified in `ThirdParty/README.md`. Apache
license and NOTICE files are retained. MyGo and the Voxt voice/provider stack
are not used.

## NOT_COVERED

- Voice input, realtime WebSocket, ASR, audio output, image, video, and tools.
- Model downloads, accounts, cloud provider presets, and remote HTTPS E2E.
- Persistent history across launches, multiple saved sessions, and retries.
- Immediate cancellation inside an MLX prefill or model call.
- Code signing, notarization, sandboxing, installer, and DMG.

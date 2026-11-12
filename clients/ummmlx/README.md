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
cd /Users/bytedance/code/omniInfer/clients/ummmlx
swift test
Scripts/build.sh release
open build/ummmlx.app
```

The build creates `build/ummmlx.app` without signing, notarization, or a DMG.
The raw executable is `build/ummmlx.app/Contents/MacOS/ummmlx`.

## Start the real local service

Use separate terminals from the repository root.

Terminal 1:

```sh
IPC_DIR=$(mktemp -d /tmp/ummmlx-ipc.XXXXXX)
/tmp/umm-b3-client-r7/bin/umm-infer serve \
  --host 127.0.0.1 --port 30000 --model Qwen2.5-0.5B-Instruct-4bit \
  --entry mlx --entry-endpoint "ipc://$IPC_DIR/entry" \
  --completion "ipc://$IPC_DIR/completion" --abort "ipc://$IPC_DIR/abort"
```

Terminal 2, with the same `IPC_DIR`:

```sh
MODEL=/Users/bytedance/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /opt/homebrew/opt/python@3.11/bin/python3.11 -I python/umminfer_mlx_stage.py \
  --name mlx --recv "ipc://$IPC_DIR/entry" \
  --coordinator "ipc://$IPC_DIR/completion" --abort "ipc://$IPC_DIR/abort" \
  --model "$MODEL"
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

The real runner uses the existing `/tmp/umminfer-mlx-perf.lock`, process patrol,
frozen server, current stage, fixed local model, and offline environment. It
stops one real stream after a delta, starts a fresh GUI conversation, completes
it, and captures both appearances. The output directory must not exist:

```sh
OUT=/tmp/ummmlx-e2e-manual
/opt/homebrew/opt/python@3.11/bin/python3.11 -I Scripts/real_e2e.py \
  --out "$OUT" > "$OUT.console.log" 2>&1 &
E2E_PID=$!
wait "$E2E_PID"
REAL_EXIT=$?
printf 'REAL_EXIT=%s\n' "$REAL_EXIT"
```

See `report.json`, `automation.json`, `server.log`, `stage.log`, `app.log`, and
`screenshots/` under that directory. Each captured process log is capped at
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

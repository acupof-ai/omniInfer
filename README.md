# omniInfer

omniInfer is a C++20 pipeline runtime with MessagePack control messages,
ZeroMQ transport, POSIX shared-memory payload relay, and an OpenAI-compatible
HTTP API.

## Implemented

- Dynamic MessagePack values, tensors, control messages, `DataRef`,
  `StagePayload`, and `OmniRequest`.
- Serial and batched scheduling.
- Stage execution, request coordination, aborts, and relay acknowledgements.
- POSIX shared-memory transfers between C++ stages.
- `/health`, `/ready`, `/v1/models`, `/v1/chat/completions` (JSON and SSE),
  and `/abort`, with optional bearer authentication.

The executable contains reference echo compute. It does not load a model.
Replace `echo_compute` in `src/main.cpp` with the required compute function.

## Dependencies

- Zig 0.16.0.
- ZeroMQ and msgpack-c.
- cpp-httplib and nlohmann/json.
- Python 3 for the HTTP smoke test.

On macOS with Homebrew:

```sh
brew install zig libzmq msgpack cpp-httplib nlohmann-json
```

## Build and test

Use `-Ddeps-prefix` when dependencies are outside the system include and
library paths. The output is `zig-out/bin/omni-infer`.

```sh
zig build -Doptimize=ReleaseFast -Ddeps-prefix=/opt/homebrew
zig build test -Ddeps-prefix=/opt/homebrew
zig build smoke -Ddeps-prefix=/opt/homebrew
```

`test` runs wire, relay, scheduler, coordinator, and two-stage pipeline tests.
`smoke` checks HTTP validation, authentication, JSON, SSE, 24 concurrent JSON
requests, 12 concurrent SSE requests, external stages, and SIGTERM shutdown.

## Run

Start the in-process demo:

```sh
./zig-out/bin/omni-infer serve --demo --port 30000 --ipc-dir /tmp/omni-demo

curl -s localhost:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"omni","messages":[{"role":"user","content":"hi"}]}'
```

Start a coordinator and two external stages in separate terminals:

```sh
./zig-out/bin/omni-infer serve --port 30000 --entry entry \
  --entry-endpoint ipc:///tmp/omni/entry \
  --completion ipc:///tmp/omni/completion --abort ipc:///tmp/omni/abort

./zig-out/bin/omni-infer stage --name entry --recv ipc:///tmp/omni/entry \
  --coordinator ipc:///tmp/omni/completion --abort ipc:///tmp/omni/abort \
  --next worker=ipc:///tmp/omni/worker

./zig-out/bin/omni-infer stage --name worker --recv ipc:///tmp/omni/worker \
  --coordinator ipc:///tmp/omni/completion --abort ipc:///tmp/omni/abort \
  --peer entry=ipc:///tmp/omni/entry
```

The downstream `--peer entry=...` endpoint carries relay acknowledgements to
the upstream stage. Without it, upstream transfer credits are not released.

## Compatibility and limits

- The control plane supports the implemented MessagePack message schemas and
  includes decode checks for Python-produced messages.
- Shared-memory `StagePayload` transfer is supported between C++ stages.
- Python pickle payload headers are not implemented. A mixed Python/C++
  shared-memory edge is not supported.
- MessagePack map keys must be strings. Integers must fit in signed 64 bits.
- CUDA, model execution, local-object dispatch, and Mooncake transfer are not
  implemented.
- Pending relay acknowledgements have a configurable 30-second default timeout.
  Timed-out transfers fail their requests and release transfer credits.
- `/v1/audio/speech` is not implemented.

ZMQ does not create the parent directory of an `ipc://` path. The launchers
create it and remove stale socket files for configured endpoints.

## Layout

```text
build.zig
include/omniinfer/{proto,relay,comm,scheduling,pipeline,serve}
src/{proto,relay,comm,scheduling,pipeline,serve}
src/main.cpp
tests/self_check.cpp
tests/compat/{wire_compat.cpp,runtime_compat.cpp}
tests/http_smoke.py
```

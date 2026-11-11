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
- Optional: a CUDA toolkit with `cuda_runtime_api.h` and `libcudart` for the
  CUDA IPC build. The default build does not need it.

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

The CPU build is the default. CUDA IPC is an optional build:

```sh
zig build -Dcuda -Dcuda-prefix=/usr/local/cuda -Ddeps-prefix=/opt/homebrew
zig build test -Dcuda -Dcuda-prefix=/usr/local/cuda -Ddeps-prefix=/opt/homebrew
```

`-Dcuda-prefix` must contain `include/` and `lib64/`. `-Dcuda` alone uses the
system include and library paths. With `-Dcuda`, `test` also builds and runs
`cuda_ipc_test` and `cuda_pipeline_test`. Both skip themselves in a CPU build.

Three CUDA-only targets stay out of the default test run:

```sh
# Repeated rounds, stops at the first failure.
zig build cuda-stress -Dcuda -Dcuda-prefix=/usr/local/cuda -Ddeps-prefix=/opt/homebrew

# Two-process latency for SHM and CUDA IPC; ReleaseFast is required.
zig build cuda-bench -Dcuda -Dcuda-prefix=/usr/local/cuda -Ddeps-prefix=/opt/homebrew -Doptimize=ReleaseFast

# compute-sanitizer memcheck over both test binaries.
python3 tests/cuda_sanitize.py --ipc zig-out/bin/cuda_ipc_test \
  --pipeline zig-out/bin/cuda_pipeline_test --log-dir /tmp/cuda-sanitize
```

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

### CUDA IPC stages

Two stages on one host can move payloads through CUDA IPC. Declare each stage's
device, and give the sender the device of the stage it sends to:

```sh
./zig-out/bin/omni-infer stage --name entry --gpu-id 0 \
  --recv ipc:///tmp/omni/entry --coordinator ipc:///tmp/omni/completion \
  --abort ipc:///tmp/omni/abort --next worker=ipc:///tmp/omni/worker \
  --peer worker=ipc:///tmp/omni/worker --peer-gpu worker=1

./zig-out/bin/omni-infer stage --name worker --gpu-id 1 \
  --recv ipc:///tmp/omni/worker --coordinator ipc:///tmp/omni/completion \
  --abort ipc:///tmp/omni/abort --peer entry=ipc:///tmp/omni/entry
```

- `--gpu-id N` gives the stage its CUDA device.
- `--peer-gpu name=N` gives the device of a peer stage on the same host.
- The sender checks peer access in both directions before selecting CUDA IPC.
- Two stages on the same device use CUDA IPC without peer access.
- If peer access is unavailable, the payload exceeds the slot limit, or
  `OMNIINFER_DISABLE_CUDA_IPC=1` is set, the edge uses POSIX shared memory.
  The fallback is logged once per edge.

## Compatibility and limits

- The control plane supports the implemented MessagePack message schemas and
  includes decode checks for Python-produced messages.
- Shared-memory `StagePayload` transfer is supported between C++ stages.
- Python pickle payload headers are not implemented. A mixed Python/C++
  shared-memory edge is not supported.
- MessagePack map keys must be strings. Integers must fit in signed 64 bits.
- CUDA IPC carries host-serialized `StagePayload` bytes. The sender packs the
  payload to host bytes and copies them to its device; the receiver copies the
  bytes back to host memory. It is not zero-copy device-tensor transfer and it
  does not execute a model.
- CUDA IPC is selected only when both stages declare GPU ids on one host, peer
  access works in both directions, and the packed payload is at most
  `slot_size_mb` (default 512). Otherwise the edge uses shared memory.
- A receiver rejects a `DataRef` above `max_payload_bytes` (default 1 GiB)
  before any allocation.
- Pending relay acknowledgements have a configurable 30-second default timeout
  (`transfer_timeout_ms`). A timed-out transfer abandons its CUDA edge: that
  target fails closed and other targets keep working. Restart the sender
  process to rebuild the edge.
- Model execution, local-object dispatch, and Mooncake transfer are not
  implemented.
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
tests/{cuda_ipc_test.cpp,cuda_pipeline_test.cpp,cuda_bench.cpp}
tests/{cuda_stress.py,cuda_sanitize.py,http_smoke.py}
docs/cuda-ipc.md
```

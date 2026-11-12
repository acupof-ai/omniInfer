# External stage architecture

## Normal external-model serving

`umm-infer serve` without `--demo` creates a Coordinator and HTTP/WebSocket
adapters. It does not create a C++ `Stage`.

```text
CLI or public C++ API
        |
        v
HTTP(S) client -> HTTP adapter ----+
                                   |
WebSocket client -> WS adapter ----+--> Coordinator
                                          |
                                     ZeroMQ messages
                                          |
                                          v
                              external Python process
                              MLXStage or TinyASRStage
                                          |
                                          v
                                   real model forward
```

The external Python stage connects directly to the configured ZeroMQ entry,
completion, and abort endpoints. It announces `stage_ready` after loading the
model and initializing its control inputs. The Coordinator queues requests for
an unready entry stage.

The Python process owns model loading and generation. It executes one request
at a time. Abort and disconnect handling reaches MLX at token boundaries.

The external Python stage does not pass through C++ `Stage`, `SimpleScheduler`,
or C++ `CommRouter`. Its request and result data travel in ZeroMQ messages.
Python-to-C++ shared-memory `StagePayload` transfer is not implemented.

The real paths are in `src/main.cpp` (`run_serve`),
`src/serve/openai_api.cpp`, `src/serve/realtime_api.cpp`,
`src/pipeline/pipeline.cpp` (`Coordinator`), and
`python/umminfer_mlx_stage.py`.

## C++ stage host

The separate `umm-infer stage` command and the embedded C++ API create a C++
`Stage`.

```text
Coordinator -> ZeroMQ -> C++ Stage -> SimpleScheduler -> compute function
                             |
                       next C++ Stage, if configured
```

`SimpleScheduler` belongs to the C++ `Stage` object. It executes the supplied
compute function. The current executable supplies `echo_compute`; it does not
load a production model in this path.

Between C++ stages, `Stage` uses `CommRouter` for payload transfer. The default
transport is POSIX shared memory with relay acknowledgements. Optional CUDA IPC
has its own device, peer-access, and slot-size requirements.

The C++ host path is in `src/main.cpp` (`run_stage`),
`include/umminfer/pipeline/pipeline.hpp`, `src/pipeline/pipeline.cpp`,
`src/scheduling/simple_scheduler.cpp`, and `src/comm/comm.cpp`.

## Demo mode

`serve --demo` runs one C++ `Stage` and its echo compute function in the server
process. The request still goes through the Coordinator and ZeroMQ. Demo mode
tests the C++ stage host. It is not an external MLX model server.

## HTTP and WebSocket endpoints

The HTTP adapter owns chat JSON, chat SSE, multipart transcription, health,
readiness, models, and abort. The optional realtime adapter owns the private
single-utterance WebSocket protocol. Both adapters submit requests to the same
Coordinator.

Realtime preview events are not a separate scheduler. The realtime client can
request the completed WebSocket result. Otherwise it sends retained PCM to the
HTTP transcription endpoint for the Final pass.

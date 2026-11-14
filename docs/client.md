# C++ clients

The `umm-infer` executable provides OpenAI-compatible chat and Omni ASR clients.
The public library API is in `include/umminfer/client/client.hpp`.

## Dependency

The client uses libcurl for HTTP(S), multipart forms, SSE, and WebSocket
transport. TLS certificate and hostname verification are enabled. The installed
libcurl must support `ws`. The realtime client opens only `ws://` on loopback;
remote `wss://` is not supported. Apple's system libcurl 8.7.1 does not support
WebSocket; use Homebrew libcurl and pass its prefix explicitly. The realtime
server also requires libwebsockets.

On Linux, check `curl-config --protocols` for `ws` and `wss`. Ubuntu 24.04
ships libcurl 8.5.0 without them. Build curl 8.5.0 with `--enable-websockets`
into a separate prefix and pass it with `-Dcurl-prefix`; do not replace the
system libcurl. [Validation](validation.md) section 3a gives the commands.

The build needs the same dependencies as the root README:

```sh
brew install zig libzmq msgpack cpp-httplib nlohmann-json curl libwebsockets
python3 -m venv .venv
.venv/bin/python -m pip install mlx-lm pyzmq msgpack websockets
zig build -Ddeps-prefix=/opt/homebrew -Dcurl-prefix=/opt/homebrew/opt/curl
zig build test -Ddeps-prefix=/opt/homebrew -Dcurl-prefix=/opt/homebrew/opt/curl
```

## Chat

```sh
export UMMINFER_API_KEY='...'
./zig-out/bin/umm-infer chat \
  --url https://api.openai.com/v1 --model MODEL --prompt 'Hello'

./zig-out/bin/umm-infer chat \
  --url http://127.0.0.1:30000/v1 --model omni --prompt 'Hello' --stream
```

`UMMINFER_API_KEY` takes precedence over `OPENAI_API_KEY`. The CLI also reads
`OPENAI_BASE_URL` and `OPENAI_MODEL`. Keys are not accepted as command-line
arguments. Server error text is redacted before it is truncated or printed.

Plain HTTP is allowed only for a loopback host. Other OpenAI endpoints must use
HTTPS. Redirects are disabled.

Options include `--system`, `--max-tokens`, `--temperature`, `--top-p`,
`--timeout-ms`, `--first-token-ms`, `--stall-ms`, and `--max-attempts`.
Retries use exponential backoff with equal jitter: the delay is drawn from
`[cap/2, cap)`, and `cap` doubles from `retry_base` up to `retry_max`.
`Retry-After` and `Retry-After-Ms` are honored; a wait above `max_retry_after`
is reported as an error. The server wait is not shortened by the attempt
timeout. A retry never starts after a text delta was delivered.

The route accepts an explicit allowlist of top-level chat fields: `model`,
`messages`, `stream`, `temperature`, `max_tokens`, `max_completion_tokens`,
`top_p`, `seed`, `stop`, `response_format` and `stream_options`. Any other
non-null top-level field returns HTTP 400 `unsupported field: NAME`, whether or
not this server knows the name. A null value is treated as absent. A non-null
`top_p`, `seed` or non-empty `stop` is rejected, as is `response_format` other
than type `text`, `stream_options` other than `include_usage: false`, and
message content that is not a text string. A field with the wrong JSON type
returns 400, not 500. The client sends `top_p` and `seed` only when the caller
sets them. It never adds `stream_options`, and it never drops a caller field and
retries: an unsupported field is reported as an error.

## Final ASR

```sh
./zig-out/bin/umm-infer transcribe \
  --url http://127.0.0.1:8000 --model MODEL --file audio.wav

./zig-out/bin/umm-infer transcribe \
  --url http://127.0.0.1:8000 --model MODEL --file audio.pcm --format pcm
```

The endpoint is `POST /v1/audio/transcriptions`. `ASRClient::transcribe` and
the CLI read raw PCM, convert it to WAV with `wav_from_pcm16`, and upload that
WAV as the multipart `file` part. The client always sends `stream=true` and
`response_format=json`, so it parses SSE only; it never reads a plain JSON body.
WAV input to the route must be PCM16, mono, 16 kHz, with one data chunk. Raw PCM
input must contain complete 16-bit samples.

The route accepts `stream=true|false` and `response_format=json`.
A request without `stream`, or with `stream=false`, returns
`{"text": "..."}` as `application/json`. Only `stream=true` returns SSE
(`transcript.text.done` followed by `[DONE]`). Any other `response_format` is
rejected with HTTP 400.

`model` must equal the model the server serves; any other value returns 400.
Multipart parts are checked in two namespaces. The field namespace accepts only
`model`, `stream`, `response_format`, `include_generation_metadata`, `language`,
`prompt`, `max_new_tokens` and `temperature`; any other field name returns 400
`unsupported multipart field: NAME`. The file namespace accepts exactly one part
named `file`; a part with any other name returns 400
`unsupported multipart file part: NAME`, so a scalar field cannot bypass the
field checks by resending itself with a filename. A repeated field name returns
400 `duplicate multipart field: NAME`.

`language` and `prompt` are rejected by presence, including an empty value:
this server does not use them, so accepting them would silently ignore the
caller's intent. The same rule rejects `audio_layout`, `stop_at_end_of_text` and
`stop_on_token_loop` through the field allowlist. An explicit `temperature` is
accepted only as `0`, the only value the entry stage uses. Invalid UTF-8 in a
multipart value returns 400 with a valid JSON body, not 500.

The route reports `generation_metadata` only from values the stage sent in its
`complete.result`: `generated_token_count` and `finish_reason`. The MLX stage
uses the `finish_reason` and cumulative `generation_tokens` from the last
MLX-LM generation response. The TinyASR stage records them from the CTC token
sequence before text stripping. `finish_reason` is `length` only when the
generation limit stopped the run, and `stop` when the model stopped on its
own, including when it stopped exactly at the limit. The field is omitted when the
stage reports neither value. A request with `include_generation_metadata=true`
is answered with HTTP 500 when the stage did not report them; the route never
substitutes a zero count or a default finish reason. The bundled client sends
`include_generation_metadata=true`, so a stage that cannot count tokens makes
the request fail instead of returning an invented `generated_tokens=0`.
`finish_reason=unknown` is reserved for a stage that reports the metadata but
cannot name its stop condition.

Both ASR listeners enforce `Authorization: Bearer KEY` when the server runs with
`--api-key KEY`. The HTTP route, the realtime WebSocket handshake, `/health` and
`/ready` all require it. A missing or wrong key returns HTTP 401
`{"error":{"message":"invalid api key","type":"auth_error"}}`; the WebSocket
handshake is rejected with an HTTP 401 status line. No response or server log
echoes the key. The client reads the key from `UMMINFER_API_KEY`, then
`OPENAI_API_KEY`; the realtime Final request uses the same key.

ASR endpoints are restricted to loopback HTTP. Proxy use is disabled. Use
`--timeout-ms` for the total request deadline and `--stall-ms` for the maximum
interval without response data.

### Real ummInfer ASR server

The HTTP transcription route and realtime WebSocket route submit audio to the
same C++ Coordinator. The external Python stage then runs the model. The HTTP
and WebSocket listeners use separate ports.

The deterministic tiny CTC model uses an FFT, a log-mel filterbank, two dense
layers, and greedy CTC decoding. It is an initialized inference test model, not
a trained speech recognizer. Its output has no recognition-quality guarantee.

Generate its configuration and safetensors weights. The output path must not
exist:

```sh
WORKDIR=$(mktemp -d /tmp/umminfer-asr.XXXXXX)
mkdir "$WORKDIR/ipc"
python3 -I python/umminfer_tiny_asr_stage.py generate \
  --output "$WORKDIR/model"
```

Start the server in one terminal, with the same `WORKDIR`:

```sh
./zig-out/bin/umm-infer serve --host 127.0.0.1 \
  --port 55271 --realtime-port 55272 --model tiny-ctc --entry asr \
  --entry-endpoint "ipc://$WORKDIR/ipc/entry" \
  --completion "ipc://$WORKDIR/ipc/completion" \
  --abort "ipc://$WORKDIR/ipc/abort"
```

Start the real MLX stage in another terminal:

```sh
python3 -I python/umminfer_tiny_asr_stage.py serve --name asr \
  --recv "ipc://$WORKDIR/ipc/entry" \
  --coordinator "ipc://$WORKDIR/ipc/completion" \
  --abort "ipc://$WORKDIR/ipc/abort" --model "$WORKDIR/model"
```

The stage sends `stage_ready` with its `from_stage` name after model loading and
control-plane initialization. The server queues early requests until this event;
an early abort remains terminal and its queued Submit is removed. Start the
server and stage in either order. The CLI startup path requires this protocol and
does not fall back to a timer for older stages.

`GET /ready` reports startup admission for that entry stage. With an external
`--entry-endpoint`, it returns HTTP 503 `{"status":"starting"}` until the entry
stage sends `stage_ready`, then HTTP 200 `{"status":"ready"}`. The in-process
`--demo` path keeps its previous always-200 behavior. The route does not check
downstream stages, model health or per-request liveness.

Use HTTP port 55271 for `transcribe` and WebSocket port 55272 for `realtime`.
The realtime `--final-url` must name the HTTP listener, not the WebSocket
listener:

```sh
./zig-out/bin/umm-infer realtime \
  --url http://127.0.0.1:55272 \
  --final-url http://127.0.0.1:55271 \
  --model tiny-ctc --file audio.wav --chunk-ms 100
```

Omitting `--final-url` selects the base address given by `--url`; it does not
disable the Final request. Use `--final-url` when the WebSocket listener and the
HTTP listener use different ports.

## Realtime ASR

```sh
./zig-out/bin/umm-infer realtime \
  --url http://127.0.0.1:8001 --final-url http://127.0.0.1:8000 \
  --model MODEL --file audio.wav --chunk-ms 100

./zig-out/bin/umm-infer realtime \
  --url http://127.0.0.1:8001 --final-url http://127.0.0.1:8000 \
  --model MODEL --file audio.wav --live-result
```

Realtime transport uses loopback `ws://` and `/v1/realtime`. This protocol is
private to this server. It is not compatible with the OpenAI Realtime API. The
client sends `session.update` and base64 PCM16 append messages. Only
`--live-result` sends `transcription.done`; the default `finish()` closes the
WebSocket and goes straight to HTTP Final.

The `session.update` allowlist is `input_audio_format`, `turn_detection`
and `max_new_tokens`. `language` and `prompt` are rejected by presence,
including an empty string or `null`, because the entry stage does not read
them; accepting them would report a configuration that is not in effect.
The same request sent to the HTTP Final route returns 400, so both paths
refuse the field. `transcription_session.updated` and `transcription.completed`
therefore carry no language. Any other session field returns an `error` event.

The bundled server sends no partial preview events. After the stage returns, it
sends exactly one `transcription.segment` with `is_final: true`, then one
`transcription.completed`, and closes the WebSocket. That completed event is a
single-shot result over the whole retained audio, not a concatenation of the
segments. The client does handle several segments: it groups them by
`event_index` and `segment_id` and passes non-final text to `on_partial`. That
assembly path is library capability, not behavior the bundled server exercises.

By default, `finish()` sends no `transcription.done`. It closes the WebSocket
and performs one HTTP Final ASR request with the retained PCM. Only the PCM is
kept for that request; the preview segments are not used. `--live-result`
instead sends `transcription.done`, accepts `transcription.completed` as the
result, and waits for it up to `--final-timeout-ms`. When that event does not
arrive, when the live path fails, or when the wait times out, the client still
performs the HTTP Final request.

Incoming WebSocket messages are limited to 1 MiB. The client retains at most
1 GiB of realtime audio. The bundled realtime server keeps at most 64 MiB per
session and rejects an append past that limit. SIGINT and SIGTERM cancel active
client operations.

## Library use

Link the client sources with the rest of `umm_core` and libcurl. Construct
`OpenAIClient` or `ASRClient` with an explicit configuration. The library does
not read environment variables itself.

```cpp
#include <umminfer/client/client.hpp>

using namespace umminfer::client;

OpenAIConfig config;
config.base_url = "http://127.0.0.1:30000/v1";
config.model = "omni";
OpenAIClient client(config);

ChatRequest request;
request.messages.push_back({"user", "Hello"});
request.stream = true;
request.on_delta = [](const std::string& delta) { /* consume delta */ };
const ChatResponse response = client.chat(request);
```

Pass a `std::atomic<bool>` through `ChatRequest::cancel`,
`TranscribeRequest::cancel`, or `RealtimeRequest::cancel` to cancel an operation.
All client failures derive from `umminfer::client::Error`; `code()` and
`retryable()` expose the failure classification.

## Verification

Date-specific binary hashes, log paths and latency measurements are recorded in
[validation.md](validation.md). They are snapshots of one run, not a contract.

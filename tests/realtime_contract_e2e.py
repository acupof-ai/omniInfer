#!/usr/bin/env python3
"""Black-box realtime contract E2E: real CLI/WebSocket client, server, stage, model."""

import argparse
import base64
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import wave

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect


MODEL_NAME = "tiny-ctc"
READY_MARKER = b"MLX stage asr ready"
STARTUP_TIMEOUT_S = 120
REQUEST_TIMEOUT_S = 60
SHUTDOWN_TIMEOUT_S = 20
LOG_CAP_BYTES = 65536
RESPONSE_CAP_BYTES = 65536


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def spawn(name, args, cwd, env, processes, logs, readers, ready):
    proc = subprocess.Popen(args, cwd=cwd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    processes[name] = proc
    logs[name] = bytearray()

    def drain():
        while True:
            chunk = proc.stdout.readline(4096)
            if not chunk:
                break
            remaining = LOG_CAP_BYTES - len(logs[name])
            if remaining > 0:
                logs[name].extend(chunk[:remaining])
            if name == "stage" and READY_MARKER in chunk:
                ready.set()
        proc.stdout.close()

    thread = threading.Thread(target=drain, daemon=True)
    thread.start()
    readers.append(thread)


def wait_for_tcp(port, proc):
    for _ in range(400):
        require(proc.poll() is None, "server exited during startup")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise AssertionError(f"server did not listen on port {port} within 20 seconds")


def wait_for_stage(proc, ready):
    for _ in range(STARTUP_TIMEOUT_S * 20):
        if ready.is_set():
            return
        require(proc.poll() is None, "ASR stage exited during startup")
        time.sleep(0.05)
    raise AssertionError(f"ASR stage was not ready within {STARTUP_TIMEOUT_S} seconds")


def stop_owned_processes(processes, readers):
    errors = []
    for proc in reversed(list(processes.values())):
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
    deadline = time.monotonic() + SHUTDOWN_TIMEOUT_S
    for name, proc in processes.items():
        try:
            proc.wait(max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(5)
            errors.append(f"{name} required SIGKILL")
        # The C++ server handles SIGTERM and exits 0. The Python stage has no
        # SIGTERM handler, so signal termination is its normal cleanup.
        allowed = {0, -signal.SIGTERM} if name == "stage" else {0}
        if proc.returncode not in allowed:
            errors.append(f"{name} exit code {proc.returncode}")
    for thread in readers:
        thread.join(2)
        if thread.is_alive():
            errors.append("process output reader did not exit")
    return errors


def http_request(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=REQUEST_TIMEOUT_S)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        response = conn.getresponse()
        data = response.read(RESPONSE_CAP_BYTES + 1)
        require(len(data) <= RESPONSE_CAP_BYTES, "HTTP response exceeded byte cap")
        return response.status, response.getheader("Content-Type", ""), data
    finally:
        conn.close()


def wait_ready(port, proc):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        require(proc.poll() is None, "server exited before /ready")
        try:
            status, _, body = http_request(port, "GET", "/ready")
        except OSError:
            time.sleep(0.05)
            continue
        if (status, body) == (200, b'{"status":"ready"}'):
            return
        time.sleep(0.05)
    raise AssertionError("server did not report ready after stage_ready")


def multipart_transcription(port, fields, wav_bytes):
    boundary = "----umminfer-realtime-contract"
    chunks = []
    for name, value in fields.items():
        chunks.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
            f"{value}\r\n".encode())
    chunks.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        "filename=\"audio.wav\"\r\nContent-Type: application/octet-stream\r\n\r\n".encode())
    chunks.append(wav_bytes)
    chunks.append(f"\r\n--{boundary}--\r\n".encode())
    return http_request(port, "POST", "/v1/audio/transcriptions", b"".join(chunks),
                        {"Content-Type": f"multipart/form-data; boundary={boundary}"})


def raw_handshake(port, target, timeout=1.0):
    """Real HTTP/WebSocket handshake. Reads the server's actual response bytes."""
    request = (f"GET {target} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
               "Upgrade: websocket\r\nConnection: Upgrade\r\n"
               "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
               "Sec-WebSocket-Version: 13\r\n\r\n").encode()
    data = bytearray()
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as sock:
        sock.sendall(request)
        try:
            while len(data) < 8192:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data.extend(chunk)
        except socket.timeout:
            pass
    return bytes(data)


def handshake_status(response):
    line = response.split(b"\r\n", 1)[0]
    parts = line.split()
    require(len(parts) >= 2 and parts[0].startswith(b"HTTP/"), f"bad handshake: {line!r}")
    return int(parts[1]), response


def websocket(port, query):
    return connect(f"ws://127.0.0.1:{port}/v1/realtime?{query}",
                   compression=None, open_timeout=5, close_timeout=5,
                   ping_interval=None, max_size=1 << 20)


def recv_event(ws, timeout=10):
    return json.loads(ws.recv(timeout=timeout))


def expect_close_1000(ws):
    try:
        while True:
            ws.recv(timeout=5)
    except ConnectionClosed as exc:
        require(exc.rcvd is not None and exc.rcvd.code == 1000,
                f"server closed with {exc.rcvd!r}, expected 1000")
        return
    raise AssertionError("server did not close the WebSocket")


def expect_error(port, event, needle):
    with websocket(port, f"intent=transcription&model={MODEL_NAME}") as ws:
        ws.send(json.dumps(event))
        reply = recv_event(ws)
        require(reply.get("type") == "error",
                f"expected an error event, got {reply!r}")
        message = reply.get("error", {}).get("message", "")
        require(needle in message, f"error message {message!r} does not contain {needle!r}")
        expect_close_1000(ws)
    print(f"PASS session_reject needle={needle!r}", flush=True)


def update_session(ws, settings):
    ws.send(json.dumps({"type": "session.update", "session": settings}))
    reply = recv_event(ws)
    require(reply.get("type") == "transcription_session.updated",
            f"unexpected session reply: {reply!r}")
    return reply["session"]


def pcm_from_wav(path):
    with wave.open(str(path), "rb") as wav:
        require((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) ==
                (1, 2, 16000), "test WAV is not PCM16 mono 16 kHz")
        return wav.readframes(wav.getnframes())


def run_contract(port_http, port_ws, audio_path, wav_bytes, binary, env):
    # Handshake rejection is checked with a real handshake and the server's
    # actual response bytes. Successful handshakes use a real WebSocket client.
    # Known limitation (NOT_COVERED): libwebsockets normalizes the URI, so
    # /v1//realtime, dot segments and percent-escaped aliases reach the server
    # as /v1/realtime and are accepted. Only paths that normalize to something
    # else can be rejected. The alias behavior is pinned below so a change is
    # visible, but it is not treated as a passing contract.
    cases = [
        ("/v1/realtime/x?intent=transcription&model=tiny-ctc", 400, b"path must be exactly"),
        ("/v1/realtime?model=tiny-ctc", 400, b"intent must be transcription"),
        ("/v1/realtime?intent=translation&model=tiny-ctc", 400, b"intent must be transcription"),
        ("/v1/realtime?intent=transcription", 400, b"requires a model"),
        ("/v1/realtime?intent=transcription&model=other", 400, b"does not match the served model"),
    ]
    for target, status, needle in cases:
        response = raw_handshake(port_ws, target)
        actual, _ = handshake_status(response)
        require(actual == status, f"{target} returned {actual}: {response[:200]!r}")
        require(needle in response, f"{target} response lacks {needle!r}: {response[:300]!r}")
        print(f"PASS handshake_reject status={actual} target={target}", flush=True)

    encoded = raw_handshake(port_ws,
                            "/v1/realtime?intent=transcription&model=tiny%2Dctc")
    status, _ = handshake_status(encoded)
    require(status == 101, f"percent-encoded served model returned {status}")
    print("PASS handshake_percent_decoded status=101", flush=True)

    # NOT_COVERED known limitation: normalized aliases are accepted by
    # libwebsockets before the handler can see the raw target.
    alias = raw_handshake(port_ws,
                          "/v1//realtime?intent=transcription&model=tiny-ctc")
    alias_status, _ = handshake_status(alias)
    require(alias_status == 101,
            f"alias behavior changed: /v1//realtime returned {alias_status}")
    print("PASS known_limitation_normalized_alias_accepted status=101", flush=True)

    rejections = [
        ({"type": "session.update", "session": {"input_audio_format": "pcm16", "bogus": 1}},
         "unsupported session field: bogus"),
        ({"type": "session.update", "session": {"turn_detection": {"type": "server_vad"}}},
         "turn_detection must be null"),
        ({"type": "session.update", "session": {"audio_layout": "mono16k"}},
         "audio_layout is not supported"),
        ({"type": "session.update", "session": {"input_audio_format": "g711"}},
         "input_audio_format must be pcm16"),
        ({"type": "session.update", "session": {"max_new_tokens": 0}},
         "max_new_tokens must be between 1 and 4096"),
        # The stage never reads language or prompt, so accepting them would be
        # a silent lie. Presence alone is rejected, including empty and null.
        ({"type": "session.update", "session": {"language": "en"}},
         "language is not supported"),
        ({"type": "session.update", "session": {"language": ""}},
         "language is not supported"),
        ({"type": "session.update", "session": {"language": None}},
         "language is not supported"),
        ({"type": "session.update", "session": {"language": 5}},
         "language is not supported"),
        ({"type": "session.update", "session": {"prompt": "contract"}},
         "prompt is not supported"),
        ({"type": "session.update", "session": {"prompt": ""}},
         "prompt is not supported"),
        ({"type": "session.update", "session": {"prompt": None}},
         "prompt is not supported"),
        ({"type": "unsupported.event"}, "unsupported realtime event type"),
        ({"type": "transcription.done"},
         "requires an accepted session and PCM16 audio"),
    ]
    for event, needle in rejections:
        expect_error(port_ws, event, needle)

    def done_then_reject(followup, needle):
        with websocket(port_ws, f"intent=transcription&model={MODEL_NAME}") as ws:
            update_session(ws, {"input_audio_format": "pcm16"})
            # One second of silence keeps the real stage busy long enough that
            # the follow-up frame is read before the result poll fires.
            pcm = b"\x00\x00" * 16000
            ws.send(json.dumps({"type": "input_audio_buffer.append",
                                "audio": base64.b64encode(pcm).decode()}))
            ws.send(json.dumps({"type": "transcription.done"}))
            ws.send(json.dumps(followup))
            reply = recv_event(ws, timeout=REQUEST_TIMEOUT_S)
            require(reply.get("type") == "error",
                    f"expected a done-state error, got {reply!r}")
            require(needle in reply.get("error", {}).get("message", ""),
                    f"done-state error {reply!r} lacks {needle!r}")
            expect_close_1000(ws)
        print(f"PASS done_state_reject needle={needle!r}", flush=True)

    done_then_reject({"type": "input_audio_buffer.append", "audio": "AAAA"},
                     "audio append is not accepted after transcription.done")
    done_then_reject({"type": "session.update", "session": {"language": "fr"}},
                     "session.update is not accepted after transcription.done")

    live_metadata = {}
    live_text = None
    with websocket(port_ws, f"intent=transcription&model={MODEL_NAME}") as ws:
        effective = update_session(
            ws, {"input_audio_format": "pcm16", "turn_detection": None,
                 "max_new_tokens": 64})
        require(effective == {"input_audio_format": "pcm16", "turn_detection": None,
                              "max_new_tokens": 64},
                f"updated session was {effective!r}")
        print("PASS session_updated_effective", flush=True)

        pcm = pcm_from_wav(audio_path)
        for offset in range(0, len(pcm), 3200):
            ws.send(json.dumps({"type": "input_audio_buffer.append",
                                "audio": base64.b64encode(
                                    pcm[offset:offset + 3200]).decode()}))
        ws.send(json.dumps({"type": "transcription.done"}))
        segments = []
        completed = None
        while completed is None:
            event = recv_event(ws, timeout=REQUEST_TIMEOUT_S)
            kind = event.get("type")
            if kind == "transcription.segment":
                segments.append(event)
            elif kind == "transcription.completed":
                completed = event
            elif kind == "error":
                raise AssertionError(f"live transcription failed: {event!r}")
            else:
                raise AssertionError(f"unexpected live event before completed: {event!r}")
        require(len(segments) == 1,
                f"expected exactly one segment before completed, got {segments!r}")
        require(segments[0].get("is_final") is True, "live segment is not final")
        require(segments[0].get("event_index") == 1 and
                segments[0].get("segment_id") == 0,
                f"live segment identifiers were {segments[0]!r}")
        require(isinstance(completed.get("text"), str) and completed["text"],
                f"live completed text was {completed.get('text')!r}")
        # Nothing was ignored, so the result reports no language it never used.
        require("language" not in completed,
                f"completed language was {completed.get('language')!r}")
        live_text = completed["text"]
        live_metadata = completed.get("generation_metadata")
        # The tiny CTC stage is deterministic: one emitted "b" token that
        # stops before the 64-token limit. Assert the real values directly.
        require(live_metadata == {"generated_token_count": 1,
                                  "finish_reason": "stop"},
                f"live metadata was {live_metadata!r}")
        # No further result event is allowed before the server closes.
        try:
            while True:
                event = recv_event(ws, timeout=5)
                raise AssertionError(f"event after completed: {event!r}")
        except ConnectionClosed as exc:
            require(exc.rcvd is not None and exc.rcvd.code == 1000,
                    f"server closed with {exc.rcvd!r}, expected 1000")
    print(f"PASS live_completed text={live_text!r} metadata={live_metadata!r}", flush=True)

    status, content_type, body = multipart_transcription(
        port_http, {"model": MODEL_NAME}, wav_bytes)
    require(status == 200, f"HTTP transcription returned {status}: {body!r}")
    require(content_type.startswith("application/json"), f"content type: {content_type}")
    http_body = json.loads(body)
    http_metadata = http_body.get("generation_metadata", {})
    require(http_body.get("text") == live_text,
            f"live text {live_text!r} differs from HTTP text {http_body.get('text')!r}")
    require(live_metadata == http_metadata,
            f"live metadata {live_metadata!r} differs from HTTP metadata {http_metadata!r}")
    print(f"PASS metadata_matches_http metadata={http_metadata!r}", flush=True)

    expected = (live_text + "\n").encode()
    for label, extra in (
            ("cli_live", ["--live-result", "--final-timeout-ms", "5000"]),
            ("cli_final_fallback", ["--final-url", f"http://127.0.0.1:{port_http}"])):
        command = [str(binary), "realtime", "--url", f"http://127.0.0.1:{port_ws}",
                   "--model", MODEL_NAME, "--file", str(audio_path),
                   "--chunk-ms", "100", "--stall-ms", "20000"] + extra
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                env=env, timeout=REQUEST_TIMEOUT_S + 10)
        require(result.returncode == 0,
                f"{label} exit {result.returncode}: {result.stderr!r}")
        require(result.stdout == expected,
                f"{label} stdout {result.stdout!r}, expected {expected!r}, "
                f"stderr {result.stderr!r}")
        print(f"PASS {label} stdout={result.stdout!r}", flush=True)


def run(server, stage):
    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false"})
    processes = {}
    logs = {}
    readers = []
    ready = threading.Event()
    succeeded = False
    try:
        with tempfile.TemporaryDirectory(prefix="umminfer-realtime-contract-") as root:
            root = Path(root)
            ipc_dir = root / "ipc"
            ipc_dir.mkdir()
            model_dir = root / "model"
            audio_path = root / "audio.wav"
            require(not model_dir.exists(), "model output path already exists")
            generated = subprocess.run(
                [sys.executable, "-I", str(stage), "generate", "--output", str(model_dir)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
                timeout=STARTUP_TIMEOUT_S)
            require(generated.returncode == 0,
                    f"tiny model generation failed: {generated.stderr!r}")
            require((model_dir / "config.json").is_file() and
                    (model_dir / "model.safetensors").is_file(),
                    "tiny model generation did not create its files")

            sample_rate = 16000
            with wave.open(str(audio_path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                for start in range(0, sample_rate // 2, 400):
                    samples = [int(0.2 * 32767 * math.sin(
                        2 * math.pi * 440 * index / sample_rate))
                        for index in range(start, min(start + 400, sample_rate // 2))]
                    wav.writeframes(struct.pack(f"<{len(samples)}h", *samples))
            wav_bytes = audio_path.read_bytes()

            port_http = free_port()
            port_ws = free_port()
            while port_ws == port_http:
                port_ws = free_port()
            entry = f"ipc://{ipc_dir}/entry"
            completion = f"ipc://{ipc_dir}/completion"
            abort_endpoint = f"ipc://{ipc_dir}/abort"
            spawn("server", [str(server), "serve", "--host", "127.0.0.1",
                             "--port", str(port_http), "--realtime-port", str(port_ws),
                             "--model", MODEL_NAME, "--entry", "asr",
                             "--entry-endpoint", entry, "--completion", completion,
                             "--abort", abort_endpoint],
                  server.parent, env, processes, logs, readers, ready)
            wait_for_tcp(port_http, processes["server"])
            wait_for_tcp(port_ws, processes["server"])
            status, _, body = http_request(port_http, "GET", "/ready")
            require((status, body) == (503, b'{"status":"starting"}'),
                    f"/ready before stage was {status} {body!r}")

            spawn("stage", [sys.executable, "-I", str(stage), "serve", "--name", "asr",
                            "--recv", entry, "--coordinator", completion,
                            "--abort", abort_endpoint, "--model", str(model_dir)],
                  stage.parent, env, processes, logs, readers, ready)
            wait_for_stage(processes["stage"], ready)
            wait_ready(port_http, processes["server"])
            print(f"PASS startup http_port={port_http} ws_port={port_ws} "
                  f"stage_pid={processes['stage'].pid}", flush=True)

            run_contract(port_http, port_ws, audio_path, wav_bytes, server, env)

            require(bytes(logs["stage"]).count(READY_MARKER) == 1,
                    "stage emitted more or fewer than one ready marker")
            require(processes["stage"].poll() is None, "contract run stopped the stage")
            print("PASS stage_preserved", flush=True)
            succeeded = True
    finally:
        errors = stop_owned_processes(processes, readers)
        if processes and not errors:
            codes = {name: proc.returncode for name, proc in processes.items()}
            print(f"PASS cleanup owned_processes_reaped exit_codes={codes}", flush=True)
        for error in errors:
            print(f"FAIL cleanup: {error}", file=sys.stderr, flush=True)
        if not succeeded:
            for name, data in logs.items():
                text = bytes(data).decode(errors="replace")
                print(f"{name} log tail:\n{text[-4000:]}", file=sys.stderr, flush=True)
        if succeeded and errors:
            raise RuntimeError("; ".join(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True, type=Path)
    parser.add_argument("--stage", required=True, type=Path)
    args = parser.parse_args()
    server = args.server.resolve()
    stage = args.stage.resolve()
    require(server.is_file(), f"server binary not found: {server}")
    require(stage.is_file(), f"stage script not found: {stage}")
    print(f"SHA256 {server} {hashlib.sha256(server.read_bytes()).hexdigest()}", flush=True)
    run(server, stage)


if __name__ == "__main__":
    main()

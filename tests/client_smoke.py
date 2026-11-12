#!/usr/bin/env python3
"""Run real CLI-to-server-to-MLX-model smoke tests without local responders."""

import argparse
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import select
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor


STARTUP_TIMEOUT_S = 120
REQUEST_TIMEOUT_S = 60
SHUTDOWN_TIMEOUT_S = 20
LOG_CAP_BYTES = 65536
CLI_OUTPUT_CAP_BYTES = 65536
HTTP_BODY_CAP_BYTES = 4096
MODEL_NAME = "Qwen2.5-0.5B-Instruct-4bit"
PROMPT = "Give one concise inference tip."
LONG_PROMPT = "Count from 1 to 100, putting one number on each line."
CONCURRENT_MARKERS = ("ZXQ734J9", "K8V4N6QX", "M7R2X9LP", "T5W8C3HZ")
ASR_MODEL_NAME = "tiny-ctc"
ASR_READY = b"MLX stage asr ready"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def cli_command(binary, port, prompt, max_tokens, stream=False, timeout_ms=60000):
    command = [str(binary), "chat", "--url", f"http://127.0.0.1:{port}/v1",
               "--model", MODEL_NAME, "--prompt", prompt,
               "--max-tokens", str(max_tokens), "--temperature", "0",
               "--timeout-ms", str(timeout_ms)]
    if stream:
        command.append("--stream")
    return command


def check_output(output, error):
    require(len(output) <= CLI_OUTPUT_CAP_BYTES and len(error) <= CLI_OUTPUT_CAP_BYTES,
            "CLI output exceeded byte cap")


def run_cli(binary, port, env, prompt=PROMPT, max_tokens=16, timeout_ms=60000,
            extra=(), expected=0):
    command = cli_command(binary, port, prompt, max_tokens, timeout_ms=timeout_ms)
    command.extend(extra)
    started = time.perf_counter()
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env=env, timeout=REQUEST_TIMEOUT_S + 5)
    elapsed = time.perf_counter() - started
    check_output(result.stdout, result.stderr)
    if expected is not None:
        require(result.returncode == expected,
                f"CLI exit {result.returncode}, expected {expected}: {result.stderr!r}")
    return result, elapsed


def run_concurrent_json(binary, port, env, markers, cli_children):
    batch_started = time.perf_counter()
    pending = []
    for marker in markers:
        prompt = f"Repeat exactly and only this token: {marker}"
        started = time.perf_counter()
        proc = subprocess.Popen(cli_command(binary, port, prompt, 16),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        cli_children.append(proc)
        pending.append((marker, proc, started))

    def read(item):
        marker, proc, started = item
        output, error = proc.communicate(timeout=REQUEST_TIMEOUT_S + 5)
        elapsed = time.perf_counter() - started
        check_output(output, error)
        return {"marker": marker, "returncode": proc.returncode, "stdout": output,
                "stderr": error, "elapsed": elapsed}

    with ThreadPoolExecutor(max_workers=len(markers)) as pool:
        futures = [pool.submit(read, item) for item in pending]
        deadline = batch_started + REQUEST_TIMEOUT_S + 5
        results = [future.result(max(0.1, deadline - time.monotonic()))
                   for future in futures]
    return results, time.perf_counter() - batch_started


def http_json(port, method, path, payload=None, timeout=5, headers=None):
    body = None if payload is None else json.dumps(payload)
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        request_headers = {"Content-Type": "application/json"} if body is not None else {}
        request_headers.update(headers or {})
        headers = request_headers
        conn.request(method, path, body=body, headers=headers)
        response = conn.getresponse()
        data = response.read(HTTP_BODY_CAP_BYTES + 1)
        require(len(data) <= HTTP_BODY_CAP_BYTES, "real server response exceeded byte cap")
        require(response.status == 200, f"{path} returned HTTP {response.status}: {data!r}")
        return json.loads(data)
    finally:
        conn.close()


def http_raw(port, method, path, body=None, headers=None, timeout=30):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        response = conn.getresponse()
        data = response.read(HTTP_BODY_CAP_BYTES + 1)
        require(len(data) <= HTTP_BODY_CAP_BYTES, f"{path} response exceeded byte cap")
        return response.status, dict(response.getheaders()), data
    finally:
        conn.close()


def sse_events(body):
    events = []
    for line in body.decode().splitlines():
        if line.startswith("data: ") and line[len("data: "):] != "[DONE]":
            events.append(json.loads(line[len("data: "):]))
    return events


def chat_request(port, payload, timeout=60):
    status, headers, body = http_raw(
        port, "POST", "/v1/chat/completions", json.dumps(payload).encode(),
        {"Content-Type": "application/json"}, timeout=timeout)
    return status, {k.lower(): v for k, v in headers.items()}, body


def multipart_transcription(port, fields, audio, timeout=30, headers=None, files=None):
    boundary = "----umminfer-smoke-boundary"
    body = bytearray()
    parts = fields.items() if isinstance(fields, dict) else fields
    for name, value in parts:
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.extend((value if isinstance(value, bytes) else str(value).encode()) + b"\r\n")
    for name, filename, content in files or [("file", "audio.wav", audio)]:
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"; '
                    f'filename="{filename}"\r\n'.encode())
        body.extend(b"Content-Type: application/octet-stream\r\n\r\n")
        body.extend(content + b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())
    request_headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    request_headers.update(headers or {})
    return http_raw(port, "POST", "/v1/audio/transcriptions", bytes(body),
                    request_headers, timeout=timeout)


def health_in_flight(port, headers=None):
    return http_json(port, "GET", "/health", timeout=2, headers=headers)["in_flight"]


def wait_for_reap(port, headers=None):
    deadline = time.monotonic() + 10
    for _ in range(200):
        if health_in_flight(port, headers) == 0:
            return
        if time.monotonic() >= deadline:
            break
        time.sleep(0.02)
    raise AssertionError("real server did not reap the request within 10 seconds")


def spawn(name, args, cwd, env, processes, logs, readers, ready,
          ready_marker=b"MLX stage mlx ready"):
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
            if name == "stage" and ready_marker in chunk:
                ready.set()
        proc.stdout.close()

    thread = threading.Thread(target=drain, daemon=True)
    thread.start()
    readers.append(thread)


def wait_for_server(port, proc):
    deadline = time.monotonic() + 20
    for _ in range(400):
        require(proc.poll() is None, "real server exited during startup")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            if time.monotonic() >= deadline:
                break
            time.sleep(0.02)
    raise AssertionError("real server was not ready within 20 seconds")


def wait_for_stage(proc, ready):
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    for _ in range(STARTUP_TIMEOUT_S * 20):
        if ready.is_set():
            return
        require(proc.poll() is None, "MLX stage exited during startup")
        if time.monotonic() >= deadline:
            break
        ready.wait(0.05)
    raise AssertionError(f"MLX stage was not ready within {STARTUP_TIMEOUT_S} seconds")


def cleanup(processes, readers, cli_children, expected_stage_returncode=0):
    errors = []
    for proc in cli_children:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
    for proc in cli_children:
        if proc.poll() is None:
            try:
                proc.wait(5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(5)
                errors.append("interactive CLI required SIGKILL")

    deadline = time.monotonic() + SHUTDOWN_TIMEOUT_S
    for name in ("stage", "server"):
        proc = processes.get(name)
        if proc is not None and proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
    for name, proc in processes.items():
        try:
            proc.wait(max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(5)
            errors.append(f"{name} required SIGKILL")
        expected_returncode = expected_stage_returncode if name == "stage" else 0
        if proc.returncode != expected_returncode:
            errors.append(f"{name} exit code {proc.returncode}")
    for thread in readers:
        thread.join(2)
        if thread.is_alive():
            errors.append("process output reader did not exit")
    return errors


ASR_AUTH_KEY = "asr-auth-key-4c1e"
ASR_AUTH_WRONG = "asr-auth-wrong-key-9d2b"


def ws_realtime(ws_port, model, pcm, headers):
    """Run one real WebSocket transcription session and return its events."""
    from websockets.sync.client import connect

    import base64
    url = (f"ws://127.0.0.1:{ws_port}/v1/realtime"
           f"?intent=transcription&model={model}")
    events = []
    with connect(url, additional_headers=headers, open_timeout=5,
                 proxy=None) as socket_:
        socket_.send(json.dumps({"type": "session.update", "session": {
            "input_audio_format": "pcm16", "turn_detection": None}}))
        events.append(json.loads(socket_.recv(timeout=5)))
        for offset in range(0, len(pcm), 3200):
            chunk = pcm[offset:offset + 3200]
            socket_.send(json.dumps({"type": "input_audio_buffer.append",
                                     "audio": base64.b64encode(chunk).decode()}))
        socket_.send(json.dumps({"type": "transcription.done"}))
        while True:
            try:
                event = json.loads(socket_.recv(timeout=10))
            except Exception:  # the server closes the socket after completed
                break
            events.append(event)
            if event.get("type") == "transcription.completed":
                break
    return events


def ws_handshake(ws_port, model, headers):
    """Send a real WebSocket handshake and return the raw HTTP response head."""
    import base64
    lines = [f"GET /v1/realtime?intent=transcription&model={model} HTTP/1.1",
             f"Host: 127.0.0.1:{ws_port}", "Upgrade: websocket",
             "Connection: Upgrade",
             "Sec-WebSocket-Key: " + base64.b64encode(os.urandom(16)).decode(),
             "Sec-WebSocket-Version: 13"]
    for name, value in headers.items():
        lines.append(f"{name}: {value}")
    with socket.create_connection(("127.0.0.1", ws_port), timeout=5) as sock:
        sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())
        data = b""
        while b"\r\n\r\n" not in data and len(data) < 4096:
            chunk = sock.recv(1024)
            if not chunk:
                break
            data += chunk
    return data


def run_asr_auth_smoke(binary, stage, env, model_dir, audio_path):
    """Real bearer acceptance for the HTTP route, the WebSocket route, the
    HTTP Final request after a WebSocket session, and the CLI."""
    processes = {}
    logs = {}
    readers = []
    ready = threading.Event()
    succeeded = False
    ipc_dir = audio_path.parent / "auth-ipc"
    ipc_dir.mkdir(exist_ok=True)
    entry = f"ipc://{ipc_dir}/entry"
    completion = f"ipc://{ipc_dir}/completion"
    abort_endpoint = f"ipc://{ipc_dir}/abort"
    http_port = free_port()
    ws_port = free_port()
    while ws_port == http_port:
        ws_port = free_port()
    good = {"Authorization": f"Bearer {ASR_AUTH_KEY}"}
    bad = {"Authorization": f"Bearer {ASR_AUTH_WRONG}"}
    audio = audio_path.read_bytes()
    with wave.open(str(audio_path), "rb") as wav:
        pcm = wav.readframes(wav.getnframes())

    def require_no_key_leak(payload, label):
        require(ASR_AUTH_KEY.encode() not in payload,
                f"{label} leaked the api key")
        require(ASR_AUTH_WRONG.encode() not in payload,
                f"{label} leaked the wrong api key")

    try:
        spawn("server", [str(binary), "serve", "--host", "127.0.0.1",
                              "--port", str(http_port), "--realtime-port", str(ws_port),
                              "--model", ASR_MODEL_NAME, "--entry", "asr",
                              "--entry-endpoint", entry, "--completion", completion,
                              "--abort", abort_endpoint, "--api-key", ASR_AUTH_KEY],
              binary.parent, env, processes, logs, readers, ready, ASR_READY)
        wait_for_server(http_port, processes["server"])
        wait_for_server(ws_port, processes["server"])
        spawn("stage", [sys.executable, "-I", str(stage), "serve", "--name", "asr",
                             "--recv", entry, "--coordinator", completion,
                             "--abort", abort_endpoint, "--model", str(model_dir)],
              stage.parent, env, processes, logs, readers, ready, ASR_READY)
        wait_for_stage(processes["stage"], ready)

        # /health and /ready are behind the same handler.
        status, _, body = http_raw(http_port, "GET", "/health")
        require(status == 401, f"/health without key returned {status}: {body!r}")
        require_no_key_leak(body, "unauthenticated /health")
        deadline = time.monotonic() + 15
        while True:
            status, _, body = http_raw(http_port, "GET", "/ready", headers=good, timeout=2)
            if status == 200:
                break
            require(time.monotonic() < deadline, f"/ready with key was {status}: {body!r}")
            time.sleep(0.05)
        status, _, body = http_raw(http_port, "GET", "/ready")
        require(status == 401, f"/ready without key returned {status}: {body!r}")
        require_no_key_leak(body, "unauthenticated /ready")
        print("PASS asr_auth_http_health_ready 401_without_key 200_with_key", flush=True)

        for label, headers in (("missing", {}), ("wrong", bad)):
            status, _, body = multipart_transcription(
                http_port, {"model": ASR_MODEL_NAME}, audio, headers=headers)
            require(status == 401, f"transcription with {label} key returned {status}: {body!r}")
            require(b"invalid api key" in body, f"{label} key error was {body!r}")
            require_no_key_leak(body, f"transcription {label} key")
            print(f"PASS asr_auth_http_transcribe_{label} status=401", flush=True)

        status, _, body = multipart_transcription(
            http_port, {"model": ASR_MODEL_NAME}, audio, headers=good)
        require(status == 200, f"authenticated transcription returned {status}: {body!r}")
        require(json.loads(body).get("text") == "b",
                f"authenticated transcription body was {body!r}")
        wait_for_reap(http_port, good)
        print("PASS asr_auth_http_transcribe_correct status=200 text=b", flush=True)

        for label, headers in (("missing", {}), ("wrong", bad)):
            response = ws_handshake(ws_port, ASR_MODEL_NAME, headers)
            status_line = response.split(b"\r\n", 1)[0]
            require(b" 401 " in status_line,
                    f"WebSocket with {label} key returned {status_line!r}")
            require(b"invalid api key" in response,
                    f"WebSocket with {label} key body was {response!r}")
            require_no_key_leak(response, f"WebSocket {label} key")
            print(f"PASS asr_auth_ws_{label} handshake=401", flush=True)

        events = ws_realtime(ws_port, ASR_MODEL_NAME, pcm, good)
        completed = [event for event in events
                     if event.get("type") == "transcription.completed"]
        require(len(completed) == 1,
                f"authenticated WebSocket completed events: {events!r}")
        require(completed[0].get("text") == "b",
                f"authenticated WebSocket text was {completed[0]!r}")
        require(completed[0].get("generation_metadata") == {
            "generated_token_count": 1, "finish_reason": "stop"},
            f"authenticated WebSocket metadata was {completed[0]!r}")
        wait_for_reap(http_port, good)
        print("PASS asr_auth_ws_correct completed text=b tokens=1", flush=True)

        # The realtime client's Final pass is one HTTP transcription request
        # after the WebSocket session. This is the server segment of it.
        status, _, body = multipart_transcription(
            http_port, {"model": ASR_MODEL_NAME}, audio, headers=good)
        require(status == 200 and json.loads(body).get("text") == "b",
                f"HTTP Final after WebSocket returned {status}: {body!r}")
        wait_for_reap(http_port, good)
        print("PASS asr_auth_ws_then_http_final_server_segment status=200 text=b",
              flush=True)

        # CLI paths. The client must send the key because ASRConfig has no
        # api_key field yet; record the outcome instead of hiding it.
        cli_cases = [
            ("cli_transcribe", [str(binary), "transcribe",
                                "--url", f"http://127.0.0.1:{http_port}",
                                "--model", ASR_MODEL_NAME, "--file", str(audio_path),
                                "--timeout-ms", "20000"]),
            ("cli_realtime_fallback", [str(binary), "realtime",
                                       "--url", f"http://127.0.0.1:{ws_port}",
                                       "--final-url", f"http://127.0.0.1:{http_port}",
                                       "--model", ASR_MODEL_NAME, "--file", str(audio_path),
                                       "--chunk-ms", "100", "--stall-ms", "20000"]),
        ]
        for name, command in cli_cases:
            for key_label, key in (("missing", None), ("wrong", ASR_AUTH_WRONG),
                                   ("correct", ASR_AUTH_KEY)):
                cli_env = dict(env)
                # The ambient environment must not supply a key for the
                # missing-key case.
                cli_env.pop("UMMINFER_API_KEY", None)
                cli_env.pop("OPENAI_API_KEY", None)
                if key is not None:
                    cli_env["UMMINFER_API_KEY"] = key
                result = subprocess.run(command, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, env=cli_env,
                                        timeout=REQUEST_TIMEOUT_S + 5)
                check_output(result.stdout, result.stderr)
                require_no_key_leak(result.stderr, f"{name} {key_label}")
                if key_label != "correct":
                    # HTTP paths report "HTTP 401"; the WebSocket upgrade
                    # reports "Refused WebSocket upgrade: 401".
                    require(result.returncode == 1 and b"401" in result.stderr,
                            f"{name} with {key_label} key exit {result.returncode}: "
                            f"{result.stderr!r}")
                    print(f"PASS asr_auth_{name}_{key_label} status=401 exit=1", flush=True)
                elif result.returncode == 0 and result.stdout == b"b\n":
                    wait_for_reap(http_port, good)
                    print(f"PASS asr_auth_{name}_correct exit=0 text=b", flush=True)
                else:
                    print(f"BLOCKED asr_auth_{name}_correct exit={result.returncode} "
                          f"stderr={result.stderr[:200]!r} "
                          f"reason=client does not send Authorization for ASR",
                          flush=True)

        clean_auth_env = {k: v for k, v in env.items()
                          if k not in ("UMMINFER_API_KEY", "OPENAI_API_KEY")}
        direct = [str(binary), "transcribe",
                  "--url", f"http://127.0.0.1:{http_port}",
                  "--model", ASR_MODEL_NAME, "--file", str(audio_path),
                  "--timeout-ms", "20000"]
        fallback = subprocess.run(
            direct, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env={**clean_auth_env, "OPENAI_API_KEY": ASR_AUTH_KEY},
            timeout=REQUEST_TIMEOUT_S + 5)
        require(fallback.returncode == 0 and fallback.stdout == b"b\n",
                f"OPENAI_API_KEY fallback failed: {fallback.stderr!r}")
        wait_for_reap(http_port, good)
        print("PASS asr_auth_openai_key_fallback output=b", flush=True)
        precedence = subprocess.run(
            direct, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env={**clean_auth_env, "UMMINFER_API_KEY": ASR_AUTH_KEY,
                 "OPENAI_API_KEY": ASR_AUTH_WRONG},
            timeout=REQUEST_TIMEOUT_S + 5)
        require(precedence.returncode == 0 and precedence.stdout == b"b\n",
                f"UMMINFER_API_KEY did not take precedence: {precedence.stderr!r}")
        wait_for_reap(http_port, good)
        print("PASS asr_auth_umm_infer_precedence output=b", flush=True)

        for name in ("server", "stage"):
            require_no_key_leak(bytes(logs[name]), f"{name} log")
        succeeded = True
    finally:
        errors = cleanup(processes, readers, [], -signal.SIGTERM)
        if processes and not errors:
            codes = {name: proc.returncode for name, proc in processes.items()}
            print(f"PASS asr_auth_cleanup owned_processes_reaped exit_codes={codes}",
                  flush=True)
        for error in errors:
            print(f"FAIL asr_auth_cleanup: {error}", file=sys.stderr, flush=True)
        if not succeeded:
            for name, data in logs.items():
                print(f"{name} log tail:\n{bytes(data).decode(errors='replace')[-4000:]}",
                      file=sys.stderr, flush=True)
        if succeeded and errors:
            raise RuntimeError("; ".join(errors))


def run_asr_smoke(binary, stage, env):
    processes = {}
    logs = {}
    readers = []
    ready = threading.Event()
    succeeded = False

    def run_case(name, command, http_port):
        started = time.perf_counter()
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                env=env, timeout=REQUEST_TIMEOUT_S + 5)
        elapsed = time.perf_counter() - started
        check_output(result.stdout, result.stderr)
        require(result.returncode == 0,
                f"{name} exit {result.returncode}: {result.stderr!r}")
        require(result.stdout == b"b\n",
                f"{name} stdout was {result.stdout!r}, stderr was {result.stderr!r}")
        wait_for_reap(http_port)
        stage_pid = processes["stage"].pid
        require(processes["stage"].poll() is None, f"{name} stopped the ASR stage")
        require(bytes(logs["stage"]).count(ASR_READY) == 1,
                f"{name} did not preserve exactly one ASR ready event")
        print(f"PASS {name} latency_ms={elapsed * 1000:.3f} output=b "
              f"ready_events=1 stage_pid={stage_pid} in_flight=0", flush=True)

    try:
        with tempfile.TemporaryDirectory(prefix="umminfer-asr-e2e-") as root:
            root = Path(root)
            ipc_dir = root / "ipc"
            ipc_dir.mkdir()
            model_dir = root / "model"
            audio_path = root / "audio.wav"
            require(not model_dir.exists(), "tiny ASR model output path already exists")
            generated = subprocess.run(
                [sys.executable, "-I", str(stage), "generate", "--output", str(model_dir)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
                timeout=STARTUP_TIMEOUT_S)
            check_output(generated.stdout, generated.stderr)
            require(generated.returncode == 0,
                    f"tiny ASR model generation failed: {generated.stderr!r}")
            require((model_dir / "config.json").is_file() and
                    (model_dir / "model.safetensors").is_file(),
                    "tiny ASR model generation did not create its model files")

            sample_rate = 16000
            with wave.open(str(audio_path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                for start in range(0, sample_rate // 2, 400):
                    samples = [int(0.2 * 32767 * math.sin(2 * math.pi * 440 * index /
                                                          sample_rate))
                               for index in range(start, min(start + 400, sample_rate // 2))]
                    wav.writeframes(struct.pack(f"<{len(samples)}h", *samples))
            with wave.open(str(audio_path), "rb") as wav:
                require((wav.getnchannels(), wav.getsampwidth(), wav.getframerate(),
                         wav.getnframes()) == (1, 2, sample_rate, sample_rate // 2),
                        "generated WAV is not 0.5-second PCM16 mono 16 kHz")

            http_port = free_port()
            ws_port = free_port()
            for _ in range(10):
                if ws_port != http_port:
                    break
                ws_port = free_port()
            require(ws_port != http_port, "ASR HTTP and WebSocket ports are equal")
            entry = f"ipc://{ipc_dir}/entry"
            completion = f"ipc://{ipc_dir}/completion"
            abort_endpoint = f"ipc://{ipc_dir}/abort"
            spawn("server", [str(binary), "serve", "--host", "127.0.0.1",
                             "--port", str(http_port), "--realtime-port", str(ws_port),
                             "--model", ASR_MODEL_NAME, "--entry", "asr",
                             "--entry-endpoint", entry, "--completion", completion,
                             "--abort", abort_endpoint],
                  binary.parent, env, processes, logs, readers, ready, ASR_READY)
            wait_for_server(http_port, processes["server"])
            wait_for_server(ws_port, processes["server"])
            status, _, body = http_raw(http_port, "GET", "/ready", timeout=2)
            require((status, body) == (503, b'{"status":"starting"}'),
                    f"/ready before the ASR stage was {status} {body!r}")
            print("PASS asr_ready_gate_before_stage status=503", flush=True)
            spawn("stage", [sys.executable, "-I", str(stage), "serve", "--name", "asr",
                            "--recv", entry, "--coordinator", completion,
                            "--abort", abort_endpoint, "--model", str(model_dir)],
                  stage.parent, env, processes, logs, readers, ready, ASR_READY)
            wait_for_stage(processes["stage"], ready)
            # The stage prints its marker right after the control-plane send;
            # the coordinator needs the round trip to record stage_ready.
            deadline = time.monotonic() + 15
            while True:
                status, _, body = http_raw(http_port, "GET", "/ready", timeout=2)
                if (status, body) == (200, b'{"status":"ready"}'):
                    break
                require(time.monotonic() < deadline,
                        f"/ready after stage_ready was {status} {body!r}")
                time.sleep(0.05)
            print("PASS asr_ready_gate_after_stage status=200", flush=True)
            print(f"PASS real_asr_startup http_port={http_port} ws_port={ws_port} "
                  f"stage_pid={processes['stage'].pid}", flush=True)

            run_case("real_asr_multipart_wav",
                     [str(binary), "transcribe", "--url", f"http://127.0.0.1:{http_port}",
                      "--model", ASR_MODEL_NAME, "--file", str(audio_path),
                      "--timeout-ms", "20000"], http_port)
            run_case("real_asr_realtime_live",
                     [str(binary), "realtime", "--url", f"http://127.0.0.1:{ws_port}",
                      "--model", ASR_MODEL_NAME, "--file", str(audio_path),
                      "--live-result", "--chunk-ms", "100", "--final-timeout-ms", "5000",
                      "--stall-ms", "20000"], http_port)
            run_case("real_asr_realtime_multipart_fallback",
                     [str(binary), "realtime", "--url", f"http://127.0.0.1:{ws_port}",
                      "--final-url", f"http://127.0.0.1:{http_port}",
                      "--model", ASR_MODEL_NAME, "--file", str(audio_path),
                      "--chunk-ms", "100", "--stall-ms", "20000"], http_port)
            audio = audio_path.read_bytes()
            with wave.open(str(audio_path), "rb") as wav:
                raw_pcm = wav.readframes(wav.getnframes())
            pcm_path = root / "audio.pcm"
            pcm_path.write_bytes(raw_pcm)
            run_case("real_asr_transcribe_pcm",
                     [str(binary), "transcribe", "--url", f"http://127.0.0.1:{http_port}",
                      "--model", ASR_MODEL_NAME, "--file", str(pcm_path),
                      "--format", "pcm", "--timeout-ms", "20000"], http_port)
            for flag, value, needle, label in (
                    ("--language", "en", b"language is not supported", "language"),
                    ("--prompt", "note", b"prompt is not supported", "prompt"),
                    ("--audio-layout", "pcm",
                     b"unsupported multipart field: audio_layout", "audio_layout")):
                result = subprocess.run(
                    [str(binary), "transcribe", "--url", f"http://127.0.0.1:{http_port}",
                     "--model", ASR_MODEL_NAME, "--file", str(audio_path),
                     flag, value, "--timeout-ms", "20000"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
                    timeout=REQUEST_TIMEOUT_S + 5)
                check_output(result.stdout, result.stderr)
                require(result.returncode == 1,
                        f"CLI {flag} exit {result.returncode}: {result.stderr!r}")
                require(b"HTTP 400" in result.stderr and needle in result.stderr,
                        f"CLI {flag} did not reach the server rejection: {result.stderr!r}")
                wait_for_reap(http_port)
                print(f"PASS asr_cli_rejects_{label} status=400 exit=1", flush=True)
            lower_headers = lambda hs: {k.lower(): v for k, v in hs.items()}
            status, headers, body = multipart_transcription(
                http_port, {"model": ASR_MODEL_NAME}, audio)
            require(status == 200, f"default transcription returned {status}: {body!r}")
            require(lower_headers(headers).get("content-type", "").startswith(
                "application/json"), f"default transcription content type: {headers}")
            default = json.loads(body)
            require(default.get("text") == "b", f"default transcription body was {body!r}")
            require(default.get("generation_metadata") == {
                "generated_token_count": 1, "finish_reason": "stop"},
                f"default transcription metadata was {body!r}")
            wait_for_reap(http_port)
            print("PASS asr_default_json content_type=application/json text=b "
                  "tokens=1 finish=stop", flush=True)

            status, headers, body = multipart_transcription(
                http_port, {"model": ASR_MODEL_NAME, "stream": "false"}, audio)
            require(status == 200 and json.loads(body).get("text") == "b",
                    f"stream=false returned {status}: {body!r}")
            wait_for_reap(http_port)
            print("PASS asr_stream_false_json text=b", flush=True)

            status, headers, body = multipart_transcription(
                http_port, {"model": ASR_MODEL_NAME, "stream": "true"}, audio)
            require(status == 200, f"stream=true returned {status}: {body!r}")
            require(lower_headers(headers).get("content-type", "").startswith(
                "text/event-stream"), f"stream=true content type: {headers}")
            require(b'"type":"transcript.text.done"' in body and b'"text":"b"' in body,
                    f"stream=true SSE body was {body!r}")
            event = next(item for item in sse_events(body)
                         if item.get("type") == "transcript.text.done")
            require(event.get("generation_metadata") == {
                "generated_token_count": 1, "finish_reason": "stop"},
                f"stream=true metadata was {body!r}")
            wait_for_reap(http_port)
            print("PASS asr_explicit_stream_sse text=b tokens=1 finish=stop", flush=True)

            status, headers, body = multipart_transcription(
                http_port, {"model": ASR_MODEL_NAME, "include_generation_metadata": "true"},
                audio)
            require(status == 200, f"explicit metadata returned {status}: {body!r}")
            require(json.loads(body).get("generation_metadata", {}).get(
                "generated_token_count") == 1,
                f"explicit metadata body was {body!r}")
            wait_for_reap(http_port)
            print("PASS asr_generation_metadata_requested tokens=1", flush=True)

            # The deterministic model returns exactly six CTC tokens for this
            # chirp: "t8b3z8".
            chirp_path = root / "chirp.wav"
            with wave.open(str(chirp_path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                chirp = [int(0.2 * 32767 * math.sin(
                    2 * math.pi * (200 + 1500 * index / sample_rate) * index / sample_rate))
                    for index in range(sample_rate // 2)]
                wav.writeframes(struct.pack(f"<{len(chirp)}h", *chirp))
            chirp = chirp_path.read_bytes()

            # max_new_tokens=3 stops with three more tokens still available.
            status, _, body = multipart_transcription(
                http_port, {"model": ASR_MODEL_NAME, "max_new_tokens": "3"}, chirp)
            require(status == 200, f"token limit transcription returned {status}: {body!r}")
            limited = json.loads(body).get("generation_metadata", {})
            require(limited.get("finish_reason") == "length" and
                    limited.get("generated_token_count") == 3,
                    f"token limit metadata was {body!r}")
            wait_for_reap(http_port)
            print("PASS asr_token_limit_finish_reason length tokens=3", flush=True)

            # max_new_tokens=6 matches the token count exactly; the model stops
            # at the end of the audio, so the factual reason is stop.
            status, _, body = multipart_transcription(
                http_port, {"model": ASR_MODEL_NAME, "max_new_tokens": "6"}, chirp)
            require(status == 200, f"chirp transcription returned {status}: {body!r}")
            chirp_body = json.loads(body)
            require(chirp_body.get("text") == "t8b3z8",
                    f"chirp transcription text was {body!r}")
            require(chirp_body.get("generation_metadata") == {
                "generated_token_count": 6, "finish_reason": "stop"},
                f"chirp transcription metadata was {body!r}")
            wait_for_reap(http_port)
            print("PASS asr_eos_at_cap_finish_reason stop tokens=6", flush=True)

            rejected = [
                ({"model": ASR_MODEL_NAME, "response_format": "text"},
                 b"unsupported response_format", "response_format_text"),
                ({"model": ASR_MODEL_NAME, "include_generation_metadata": "maybe"},
                 b"include_generation_metadata must be true or false", "metadata_invalid"),
                ({"model": ASR_MODEL_NAME, "stream": "maybe"},
                 b"stream must be true or false", "stream_invalid"),
                ({"model": "other-model"}, b"unsupported model", "model"),
                ({"model": ASR_MODEL_NAME, "temperature": "0.5"},
                 b"temperature is fixed at 0", "temperature"),
                ({"model": ASR_MODEL_NAME, "audio_layout": "pcm"},
                 b"unsupported multipart field: audio_layout", "audio_layout"),
                ({"model": ASR_MODEL_NAME, "stop_at_end_of_text": "true"},
                 b"unsupported multipart field: stop_at_end_of_text", "stop_at_end"),
                ({"model": ASR_MODEL_NAME, "stop_on_token_loop": "true"},
                 b"unsupported multipart field: stop_on_token_loop", "stop_on_loop"),
                ({"model": ASR_MODEL_NAME, "language": "en"},
                 b"language is not supported", "language"),
                # Presence alone is rejected: an empty value is still a request
                # the stage would ignore.
                ({"model": ASR_MODEL_NAME, "language": ""},
                 b"language is not supported", "language_empty"),
                ({"model": ASR_MODEL_NAME, "prompt": "note"},
                 b"prompt is not supported", "prompt"),
                ({"model": ASR_MODEL_NAME, "prompt": ""},
                 b"prompt is not supported", "prompt_empty"),
                # Fields the route does not implement are never silently dropped.
                ({"model": ASR_MODEL_NAME, "timestamp_granularities[]": "word"},
                 b"unsupported multipart field: timestamp_granularities[]",
                 "timestamp_granularities"),
                ({"model": ASR_MODEL_NAME, "chunking_strategy": "auto"},
                 b"unsupported multipart field: chunking_strategy", "chunking_strategy"),
                # Two pure scalar parts with the same name are a protocol error; the
                # route must reject them instead of silently picking one value.
                ([("model", ASR_MODEL_NAME), ("model", ASR_MODEL_NAME)],
                 b"duplicate multipart field: model", "duplicate_scalar_model"),
            ]
            for fields, needle, label in rejected:
                status, _, body = multipart_transcription(http_port, fields, audio)
                require(status == 400, f"{label} returned {status}: {body!r}")
                require(needle in body, f"{label} error was {body!r}")
                wait_for_reap(http_port)
                print(f"PASS asr_rejects_{label} status=400", flush=True)

            # A part that carries a filename lands in the file namespace, so it
            # must not slip past the field allowlist. Only "file" is valid.
            for extra, needle, label in (
                    (("model", "bypass.txt", ASR_MODEL_NAME.encode()),
                     b"unsupported multipart file part: model", "file_part_model"),
                    (("language", "bypass.txt", b"en"),
                     b"unsupported multipart file part: language", "file_part_language")):
                status, _, body = multipart_transcription(
                    http_port, {"model": ASR_MODEL_NAME}, audio,
                    files=[("file", "audio.wav", audio), extra])
                require(status == 400, f"{label} returned {status}: {body!r}")
                require(needle in body, f"{label} error was {body!r}")
                wait_for_reap(http_port)
                print(f"PASS asr_rejects_{label} status=400", flush=True)

            # Raw non-UTF-8 bytes used to reach the JSON error serializer and
            # turn the intended 400 into a 500. The malformed shapes below have
            # correct continuation bytes but are still invalid encodings, so a
            # continuation-only check would let them through.
            for fields, needle, label in (
                    ({"model": b"\xff\xfe-model"}, b"unsupported model",
                     "model_not_utf8"),
                    ({"model": ASR_MODEL_NAME, "response_format": b"\xff\xfe-format"},
                     b"unsupported response_format", "response_format_not_utf8"),
                    ({"model": b"\xe0\x80\x80"}, b"unsupported model",
                     "model_overlong_e0"),
                    ({"model": b"\xed\xa0\x80"}, b"unsupported model",
                     "model_surrogate_ed"),
                    ({"model": b"\xf0\x80\x80\x80"}, b"unsupported model",
                     "model_overlong_f0"),
                    ({"model": b"\xf4\x90\x80\x80"}, b"unsupported model",
                     "model_above_max_f4")):
                status, _, body = multipart_transcription(
                    http_port, fields, audio)
                require(status == 400, f"{label} returned {status}: {body!r}")
                require(needle in body, f"{label} error was {body!r}")
                parsed = json.loads(body)
                require("error" in parsed, f"{label} body is not JSON: {body!r}")
                wait_for_reap(http_port)
                print(f"PASS asr_rejects_{label} status=400 json=ok", flush=True)
            succeeded = True
            run_asr_auth_smoke(binary, stage, env, model_dir, audio_path)
    finally:
        errors = cleanup(processes, readers, [], -signal.SIGTERM)
        if processes and not errors:
            codes = {name: proc.returncode for name, proc in processes.items()}
            print(f"PASS asr_cleanup owned_processes_reaped exit_codes={codes}", flush=True)
        for error in errors:
            print(f"FAIL asr_cleanup: {error}", file=sys.stderr, flush=True)
        if not succeeded:
            for name, data in logs.items():
                print(f"asr_{name} log tail:\n{bytes(data).decode(errors='replace')[-4000:]}",
                      file=sys.stderr, flush=True)
        if succeeded and errors:
            raise RuntimeError("; ".join(errors))


def run_smoke(binary, stage, model, asr_stage):
    port = free_port()
    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                "TOKENIZERS_PARALLELISM": "false"})
    processes = {}
    logs = {}
    readers = []
    cli_children = []
    ready = threading.Event()
    stream_count = 0
    succeeded = False

    def start_stream(prompt, max_tokens):
        nonlocal stream_count
        started = time.perf_counter()
        proc = subprocess.Popen(cli_command(binary, port, prompt, max_tokens, stream=True),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        cli_children.append(proc)
        stream_count += 1
        return proc, started, f"http-{stream_count}"

    def finish_stream(proc, started, action=None):
        try:
            readable, _, _ = select.select([proc.stdout], [], [], REQUEST_TIMEOUT_S)
            require(readable, "real SSE CLI produced no first output within 60 seconds")
            first = os.read(proc.stdout.fileno(), 4096)
            if not first:
                rest, error = proc.communicate(timeout=5)
                raise AssertionError(f"real SSE CLI exited before output: {error!r}")
            ttft = time.perf_counter() - started
            if action is not None:
                action()
            rest, error = proc.communicate(timeout=REQUEST_TIMEOUT_S)
            elapsed = time.perf_counter() - started
            output = first + rest
            check_output(output, error)
            return {"returncode": proc.returncode, "stdout": output, "stderr": error,
                    "ttft": ttft, "elapsed": elapsed}
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate(timeout=5)

    try:
        with tempfile.TemporaryDirectory(prefix="umminfer-client-e2e-") as ipc_dir:
            entry = f"ipc://{ipc_dir}/entry"
            completion = f"ipc://{ipc_dir}/completion"
            abort_endpoint = f"ipc://{ipc_dir}/abort"
            spawn("server", [str(binary), "serve", "--host", "127.0.0.1",
                             "--port", str(port), "--model", MODEL_NAME,
                             "--entry", "mlx", "--entry-endpoint", entry,
                             "--completion", completion, "--abort", abort_endpoint],
                  binary.parent, env, processes, logs, readers, ready)
            wait_for_server(port, processes["server"])
            spawn("stage", [sys.executable, "-I", str(stage), "--name", "mlx",
                            "--recv", entry, "--coordinator", completion,
                            "--abort", abort_endpoint, "--model", str(model)],
                  stage.parent, env, processes, logs, readers, ready)
            wait_for_stage(processes["stage"], ready)
            stage_pid = processes["stage"].pid
            print(f"PASS real_startup port={port} stage_pid={stage_pid}", flush=True)

            json_result, json_elapsed = run_cli(binary, port, env)
            require(json_result.stdout.strip(), "real JSON CLI returned empty output")
            print(f"PASS real_json latency_ms={json_elapsed * 1000:.3f} "
                  f"output={json.dumps(json_result.stdout.decode(), ensure_ascii=False)}",
                  flush=True)

            proc, started, _ = start_stream(PROMPT, 16)
            stream = finish_stream(proc, started)
            require(stream["returncode"] == 0, f"real SSE CLI failed: {stream['stderr']!r}")
            require(stream["stdout"] == json_result.stdout,
                    "real JSON and SSE outputs differ at temperature 0")
            print(f"PASS real_stream_matches_json ttft_ms={stream['ttft'] * 1000:.3f} "
                  f"latency_ms={stream['elapsed'] * 1000:.3f}", flush=True)

            status, _, body = http_raw(port, "GET", "/v1/models", timeout=5)
            models = json.loads(body)
            require(status == 200, f"/v1/models returned {status}: {body!r}")
            require(models["data"][0].get("created") == 0,
                    f"/v1/models created was {body!r}")
            print("PASS models_created_zero", flush=True)

            status, _, body = http_raw(port, "POST", "/v1/audio/speech",
                                       b'{"input":"hello"}',
                                       {"Content-Type": "application/json"}, timeout=5)
            require(status == 501 and
                    b"speech is not implemented by this server" in body,
                    f"speech route returned {status}: {body!r}")
            print("PASS speech_not_implemented status=501", flush=True)

            messages = [{"role": "user", "content": PROMPT}]
            status, _, body = chat_request(
                port, {"model": MODEL_NAME, "messages": messages, "max_tokens": 1})
            require(status == 200, f"limited chat returned {status}: {body!r}")
            limited = json.loads(body)
            require(limited.get("created", None) is not None,
                    f"chat response has no created: {body!r}")
            require(limited["choices"][0].get("finish_reason") == "length",
                    f"max_tokens=1 finish_reason was {body!r}")
            wait_for_reap(port)
            print("PASS chat_finish_reason_length from stage", flush=True)

            status, _, body = chat_request(
                port, {"model": MODEL_NAME, "messages": messages, "max_tokens": 16})
            require(status == 200, f"natural stop chat returned {status}: {body!r}")
            natural = json.loads(body)
            require(natural["choices"][0].get("finish_reason") == "stop",
                    f"natural stop finish_reason was {body!r}")
            wait_for_reap(port)
            print("PASS chat_finish_reason_stop from stage", flush=True)

            status, _, body = chat_request(
                port, {"model": MODEL_NAME, "messages": messages, "max_tokens": 1,
                       "stream": True})
            require(status == 200, f"limited SSE chat returned {status}: {body!r}")
            chunks = [json.loads(line[6:]) for line in body.decode().splitlines()
                      if line.startswith("data: ") and line[6:].strip() != "[DONE]"]
            require(chunks, f"limited SSE chat produced no chunks: {body!r}")
            created_values = {chunk.get("created") for chunk in chunks}
            require(len(created_values) == 1 and None not in created_values,
                    f"SSE chunks disagree on created: {created_values}")
            final = [chunk for chunk in chunks if chunk.get("choices") and
                     chunk["choices"][0].get("finish_reason") is not None]
            require(final and final[0]["choices"][0]["finish_reason"] == "length",
                    f"SSE finish_reason was {body!r}")
            wait_for_reap(port)
            print("PASS chat_sse_created_shared_and_finish_reason_length", flush=True)
            # The raw SSE request above consumed one server-side "http-N" id.
            stream_count += 1

            rejected = [
                ({"model": "other-model", "messages": messages}, b"unsupported model",
                 "model"),
                ({"model": MODEL_NAME, "messages": messages, "top_p": 0.9},
                 b"top_p is not implemented", "top_p"),
                ({"model": MODEL_NAME, "messages": messages, "seed": 7},
                 b"seed is not implemented", "seed"),
                ({"model": MODEL_NAME, "messages": messages, "stop": ["x"]},
                 b"stop is not implemented", "stop"),
                ({"model": MODEL_NAME, "messages": messages,
                  "response_format": {"type": "json_object"}},
                 b"only response_format type text", "response_format"),
                ({"model": MODEL_NAME, "messages": [{"role": "user", "content": None}]},
                 b"content must be a text string", "content_null"),
                ({"model": MODEL_NAME, "messages": messages, "stream": True,
                  "stream_options": {"include_usage": True}},
                 b"stream_options.include_usage is not implemented",
                 "stream_options_usage"),
                ({"model": MODEL_NAME, "messages": messages, "n": 2},
                 b"unsupported field: n", "n"),
                ({"model": MODEL_NAME, "messages": messages,
                  "tools": [{"type": "function"}]},
                 b"unsupported field: tools", "tools"),
                ({"model": MODEL_NAME, "messages": messages,
                  "tool_choice": "auto"},
                 b"unsupported field: tool_choice", "tool_choice"),
                ({"model": MODEL_NAME, "messages": messages, "logprobs": True},
                 b"unsupported field: logprobs", "logprobs"),
                ({"model": MODEL_NAME, "messages": messages, "top_logprobs": 2},
                 b"unsupported field: top_logprobs", "top_logprobs"),
                ({"model": MODEL_NAME, "messages": messages,
                  "logit_bias": {"1": 1}},
                 b"unsupported field: logit_bias", "logit_bias"),
                ({"model": MODEL_NAME, "messages": messages, "presence_penalty": 0.5},
                 b"unsupported field: presence_penalty", "presence_penalty"),
                ({"model": MODEL_NAME, "messages": messages, "frequency_penalty": 0.5},
                 b"unsupported field: frequency_penalty", "frequency_penalty"),
                ({"model": MODEL_NAME, "messages": messages,
                  "response_format": {"type": 1}},
                 b"only response_format type text", "response_format_type"),
                # The route accepts only fields it applies. Anything else is
                # named in a 400 instead of being dropped.
                ({"model": MODEL_NAME, "messages": messages, "reasoning_effort": "low"},
                 b"unsupported field: reasoning_effort", "reasoning_effort"),
                ({"model": MODEL_NAME, "messages": messages, "top_k": 20},
                 b"unsupported field: top_k", "top_k"),
                ({"model": MODEL_NAME, "messages": messages, "not_a_real_field": 1},
                 b"unsupported field: not_a_real_field", "unknown_field"),
            ]
            for payload, needle, label in rejected:
                status, _, body = chat_request(port, payload)
                require(status == 400, f"chat {label} returned {status}: {body!r}")
                require(needle in body, f"chat {label} error was {body!r}")
                print(f"PASS chat_rejects_{label} status=400", flush=True)
            wait_for_reap(port)
            print("PASS chat_validation_errors", flush=True)

            proc, started, request_id = start_stream(LONG_PROMPT, 96)
            aborted = finish_stream(
                proc, started,
                lambda: require(http_json(port, "POST", "/abort",
                                          {"request_id": request_id}).get("success") is True,
                                "real abort was not acknowledged"))
            require(aborted["returncode"] == 1,
                    f"aborted CLI exit was {aborted['returncode']}")
            require(aborted["stdout"].strip(), "aborted CLI emitted no partial model output")
            require(b"request aborted" in aborted["stderr"],
                    f"aborted CLI stderr was {aborted['stderr']!r}")
            wait_for_reap(port)
            print(f"PASS real_abort request_id={request_id} exit=1 "
                  f"partial_bytes={len(aborted['stdout'])} in_flight=0", flush=True)

            after_abort, after_abort_elapsed = run_cli(binary, port, env)
            require(after_abort.stdout == json_result.stdout,
                    "persistent stage changed output after abort")
            require(processes["stage"].pid == stage_pid and processes["stage"].poll() is None,
                    "MLX stage was reloaded after abort")
            print(f"PASS real_request_after_abort latency_ms={after_abort_elapsed * 1000:.3f}",
                  flush=True)

            proc, started, request_id = start_stream(LONG_PROMPT, 96)
            disconnected = finish_stream(proc, started,
                                         lambda: proc.send_signal(signal.SIGTERM))
            require(disconnected["returncode"] == 1,
                    f"cancelled CLI exit was {disconnected['returncode']}")
            require(disconnected["stdout"].strip(), "cancelled CLI emitted no model output")
            require(b"cancelled" in disconnected["stderr"],
                    f"cancelled CLI stderr was {disconnected['stderr']!r}")
            wait_for_reap(port)
            print(f"PASS real_sigterm_disconnect request_id={request_id} "
                  "expected_exit=1 in_flight=0", flush=True)

            after_disconnect, after_disconnect_elapsed = run_cli(binary, port, env)
            require(after_disconnect.stdout == json_result.stdout,
                    "persistent stage changed output after disconnect")
            require(processes["stage"].pid == stage_pid and processes["stage"].poll() is None,
                    "MLX stage was reloaded after disconnect")
            print(f"PASS real_request_after_disconnect "
                  f"latency_ms={after_disconnect_elapsed * 1000:.3f}", flush=True)

            proc, started, request_id = start_stream(LONG_PROMPT, 96)
            killed = finish_stream(proc, started, proc.kill)
            require(killed["returncode"] == -signal.SIGKILL,
                    f"SIGKILL CLI return code was {killed['returncode']}")
            require(killed["stdout"].strip(), "SIGKILL CLI emitted no model output")
            wait_for_reap(port)
            print(f"PASS real_sigkill_disconnect request_id={request_id} "
                  f"expected_returncode={-signal.SIGKILL} in_flight=0", flush=True)

            after_sigkill, after_sigkill_elapsed = run_cli(binary, port, env)
            require(after_sigkill.stdout == json_result.stdout,
                    "persistent stage changed output after SIGKILL")
            require(processes["stage"].pid == stage_pid and processes["stage"].poll() is None,
                    "MLX stage was reloaded after SIGKILL")
            print(f"PASS real_request_after_sigkill "
                  f"latency_ms={after_sigkill_elapsed * 1000:.3f}", flush=True)

            concurrent, concurrent_wall = run_concurrent_json(
                binary, port, env, CONCURRENT_MARKERS, cli_children)
            for item in concurrent:
                text = item["stdout"].decode(errors="replace")
                marker = item["marker"]
                require(item["returncode"] == 0,
                        f"concurrent CLI {marker} failed: {item['stderr']!r}")
                require(marker in text,
                        f"concurrent response for {marker} omitted its marker: {text!r}")
                for other in CONCURRENT_MARKERS:
                    if other != marker:
                        require(other not in text,
                                f"concurrent response for {marker} contained {other}: {text!r}")
                print(f"SAMPLE real_concurrent marker={marker} "
                      f"latency_ms={item['elapsed'] * 1000:.3f} "
                      f"output={json.dumps(text, ensure_ascii=False)}", flush=True)
            require(processes["stage"].pid == stage_pid and processes["stage"].poll() is None,
                    "MLX stage was reloaded during concurrent requests")
            require(bytes(logs["stage"]).count(b"MLX stage mlx ready") == 1,
                    "MLX stage emitted more than one ready event")
            print(f"PASS real_concurrent_no_cross_answer requests=4 "
                  f"wall_ms={concurrent_wall * 1000:.3f}", flush=True)
            print(f"PASS persistent_stage ready_events=1 stage_pid={stage_pid}", flush=True)

            # A prompt on stdin when --prompt is absent.
            clean_chat_env = {k: v for k, v in env.items()
                              if k not in ("OPENAI_BASE_URL", "OPENAI_MODEL",
                                           "UMMINFER_API_KEY", "OPENAI_API_KEY")}
            piped = subprocess.run(
                [str(binary), "chat", "--max-tokens", "16", "--temperature", "0",
                 "--timeout-ms", "60000"],
                input=PROMPT.encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env={**clean_chat_env,
                     "OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1",
                     "OPENAI_MODEL": MODEL_NAME}, timeout=REQUEST_TIMEOUT_S + 5)
            require(piped.returncode == 0 and piped.stdout.strip(),
                    f"chat with a stdin prompt and env base URL failed: {piped.stderr!r}")
            check_output(piped.stdout, piped.stderr)
            print("PASS chat_stdin_and_env_fallback output_nonempty", flush=True)

            total_timeout, _ = run_cli(binary, port, env, timeout_ms=1,
                                       extra=("--max-attempts", "1"), expected=None)
            if total_timeout.returncode == 1 and b"request timeout" in total_timeout.stderr:
                print("PASS real_total_timeout_1ms", flush=True)
            else:
                print(f"NOT_COVERED real_total_timeout_1ms exit={total_timeout.returncode} "
                      f"stderr={total_timeout.stderr[:300]!r}", flush=True)

            first_timeout, _ = run_cli(
                binary, port, env, timeout_ms=10000,
                extra=("--stream", "--first-token-ms", "1", "--stall-ms", "1000",
                       "--max-attempts", "1"), expected=None)
            if (first_timeout.returncode == 1 and
                    b"first_token timeout" in first_timeout.stderr):
                wait_for_reap(port)
                print("PASS real_first_token_timeout_1ms", flush=True)
            else:
                print(f"NOT_COVERED real_first_token_timeout_1ms "
                      f"exit={first_timeout.returncode} "
                      f"stderr={first_timeout.stderr[:300]!r}", flush=True)
            print("NOT_COVERED real_retry_and_stall: no stable real-server trigger", flush=True)
            run_asr_smoke(binary, asr_stage, env)
            succeeded = True
    finally:
        errors = cleanup(processes, readers, cli_children)
        if processes:
            codes = {name: proc.returncode for name, proc in processes.items()}
            if not errors:
                print(f"PASS cleanup owned_processes_reaped exit_codes={codes}", flush=True)
        for error in errors:
            print(f"FAIL cleanup: {error}", file=sys.stderr, flush=True)
        if not succeeded:
            for name, data in logs.items():
                print(f"{name} log tail:\n{bytes(data).decode(errors='replace')[-4000:]}",
                      file=sys.stderr, flush=True)
        if succeeded and errors:
            raise RuntimeError("; ".join(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--stage", type=Path,
                        default=Path("/Users/bytedance/code/omniInfer/python/umminfer_mlx_stage.py"))
    parser.add_argument("--model", type=Path,
                        default=Path("/Users/bytedance/.cache/huggingface/hub/"
                                     "models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/"
                                     "snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"))
    parser.add_argument("--asr-stage", type=Path,
                        default=Path("/Users/bytedance/code/omniInfer/python/"
                                     "umminfer_tiny_asr_stage.py"))
    args = parser.parse_args()
    binary = args.binary.resolve()
    stage = args.stage.resolve()
    model = args.model.resolve()
    asr_stage = args.asr_stage.resolve()
    require(binary.is_file(), f"binary not found: {binary}")
    require(stage.is_file(), f"stage not found: {stage}")
    require(model.is_dir(), f"model directory not found: {model}")
    require(asr_stage.is_file(), f"ASR stage not found: {asr_stage}")
    with binary.open("rb") as source:
        print(f"binary_sha256={hashlib.file_digest(source, 'sha256').hexdigest()}", flush=True)
    run_smoke(binary, stage, model, asr_stage)


if __name__ == "__main__":
    main()

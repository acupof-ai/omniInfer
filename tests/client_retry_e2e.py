#!/usr/bin/env python3
"""Black-box retry, stall and disconnect checks through a real byte tap.

Per case: real umm-infer CLI -> ByteTapProxy -> real umm-infer server -> real
MLX/TinyASR Python stage. The proxy is a plain TCP relay. It copies the real
server bytes and may delay, truncate, hold or reset that byte stream. It never
writes HTTP or WebSocket payloads of its own, so every status line, header and
SSE event in these tests comes from the real server.

The proxy needs two facts about the stream. It finds the end of the request
from Content-Length, and it recognizes the first non-empty SSE delta with a
byte pattern. Both are used only to decide when to cut or hold the real stream.
"""

import argparse
import hashlib
import math
import os
from pathlib import Path
import re
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import wave

STARTUP_TIMEOUT_S = 120
REQUEST_TIMEOUT_S = 90
SHUTDOWN_TIMEOUT_S = 20
LOG_CAP_BYTES = 65536
CLI_OUTPUT_CAP_BYTES = 65536
MODEL_NAME = "Qwen2.5-0.5B-Instruct-4bit"
ASR_MODEL_NAME = "tiny-ctc"
PROMPT = "Give one concise inference tip."
ASR_READY = b"MLX stage asr ready"
MLX_READY = b"MLX stage mlx ready"

DELTA_PATTERN = re.compile(rb'"content":"(?:[^"\\]|\\.)+"')


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def request_complete(buffer):
    """True when the request headers and the declared body have arrived."""
    end = buffer.find(b"\r\n\r\n")
    if end < 0:
        return False
    length = 0
    for line in bytes(buffer[:end]).split(b"\r\n")[1:]:
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":", 1)[1].strip() or b"0")
    return len(buffer) >= end + 4 + length


def reset_socket(sock):
    """Close with SO_LINGER 0 so the peer sees a TCP reset, not a FIN."""
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    except OSError:
        pass
    try:
        sock.close()
    except OSError:
        pass


class ByteTapProxy:
    """TCP relay with one plan per accepted connection.

    Plans: "pass", "reset_after_request", "reset_after_delta",
    "hold_before_response", "hold_after_delta".
    """

    def __init__(self, upstream_port):
        self.upstream_port = upstream_port
        self.plans = []
        self.records = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.listener = socket.socket()
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(16)
        self.listener.settimeout(0.2)
        self.port = self.listener.getsockname()[1]
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)

    def plan(self, *modes):
        self.plans.extend(modes)
        return self

    def start(self):
        self._thread.start()
        return self

    def close(self):
        self._stop.set()
        try:
            self.listener.close()
        except OSError:
            pass
        self._thread.join(2)

    def _accept_loop(self):
        while not self._stop.is_set():
            try:
                client, _ = self.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with self._lock:
                index = len(self.records)
                mode = self.plans[index] if index < len(self.plans) else "pass"
                record = {"index": index, "mode": mode, "request_bytes": 0,
                          "response_bytes": 0, "delta_seen": False,
                          "result": "pending"}
                self.records.append(record)
            threading.Thread(target=self._handle, args=(client, record),
                             daemon=True).start()

    def _handle(self, client, record):
        try:
            upstream = socket.create_connection(("127.0.0.1", self.upstream_port),
                                                timeout=10)
        except OSError:
            record["result"] = "upstream_unavailable"
            client.close()
            return

        mode = record["mode"]
        state = {"forward": True}
        trigger = threading.Event()
        client_closed = threading.Event()

        def pump_request():
            buffer = bytearray()
            try:
                while True:
                    chunk = client.recv(65536)
                    if not chunk:
                        break
                    buffer.extend(chunk)
                    record["request_bytes"] += len(chunk)
                    if state["forward"]:
                        upstream.sendall(chunk)
                    if mode == "reset_after_request" and not trigger.is_set() \
                            and request_complete(buffer):
                        trigger.set()
                        return
            except OSError:
                pass
            finally:
                client_closed.set()

        def pump_response():
            try:
                while True:
                    chunk = upstream.recv(65536)
                    if not chunk:
                        break
                    record["response_bytes"] += len(chunk)
                    if mode == "hold_before_response" and not trigger.is_set():
                        trigger.set()
                        return
                    if not record["delta_seen"] and DELTA_PATTERN.search(chunk):
                        record["delta_seen"] = True
                        if mode in ("reset_after_delta", "hold_after_delta"):
                            client.sendall(chunk)  # deliver the real delta first
                            trigger.set()
                            return
                    if state["forward"]:
                        client.sendall(chunk)
            except OSError:
                pass

        request_thread = threading.Thread(target=pump_request, daemon=True)
        response_thread = threading.Thread(target=pump_response, daemon=True)
        request_thread.start()
        response_thread.start()

        if mode == "pass":
            # The CLI closes its side when it exits, so the request pump ends
            # as soon as the run_cli process is gone. Its exit also means the
            # whole real response was already relayed. The server keeps the
            # connection alive, so shut both directions to end the response
            # pump instead of waiting for the keep-alive timeout.
            request_thread.join(REQUEST_TIMEOUT_S)
            record["result"] = "pass" if record["response_bytes"] > 0 else "no_response"
            for sock in (client, upstream):
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            response_thread.join(2)
            client.close()
            upstream.close()
            return

        if not trigger.wait(45):
            record["result"] = "trigger_timeout"
            state["forward"] = False
            reset_socket(client)
            reset_socket(upstream)
            return

        if mode.startswith("reset"):
            state["forward"] = False
            record["result"] = "reset"
            reset_socket(client)
            reset_socket(upstream)
            return

        # Hold: keep both sockets open with no bytes delivered downstream.
        state["forward"] = False
        record["result"] = "hold"
        client_closed.wait(45)
        client.close()
        upstream.close()


def spawn(name, args, cwd, env, processes, logs, readers, ready,
          ready_marker):
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
    deadline = time.monotonic() + 30
    for _ in range(600):
        require(proc.poll() is None, "real server exited during startup")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            if time.monotonic() >= deadline:
                break
            time.sleep(0.02)
    raise AssertionError("real server was not ready within 30 seconds")


def wait_for_stage(proc, ready):
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    for _ in range(STARTUP_TIMEOUT_S * 20):
        if ready.is_set():
            return
        require(proc.poll() is None, "Python stage exited during startup")
        if time.monotonic() >= deadline:
            break
        ready.wait(0.05)
    raise AssertionError(f"Python stage was not ready within {STARTUP_TIMEOUT_S}s")


def cleanup(processes, readers, expected_stage_returncode=0):
    errors = []
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
        expected = expected_stage_returncode if name == "stage" else 0
        if proc.returncode != expected:
            errors.append(f"{name} exit code {proc.returncode}")
    for thread in readers:
        thread.join(2)
        if thread.is_alive():
            errors.append("process output reader did not exit")
    return errors


def health_in_flight(port):
    with socket.create_connection(("127.0.0.1", port), timeout=2) as conn:
        conn.sendall(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                     b"Connection: close\r\n\r\n")
        data = b""
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            data += chunk
    body = data.split(b"\r\n\r\n", 1)[1]
    text = body.decode("utf-8", "replace")
    return int(text.split('"in_flight"', 1)[1].split(":", 1)[1].split(",")[0].split("}")[0])


def wait_for_reap(port):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if health_in_flight(port) == 0:
            return
        time.sleep(0.05)
    raise AssertionError("real server did not reap the request within 15 seconds")


def run_cli(args, env, timeout=REQUEST_TIMEOUT_S):
    started = time.perf_counter()
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env=env, timeout=timeout)
    elapsed = time.perf_counter() - started
    require(len(result.stdout) <= CLI_OUTPUT_CAP_BYTES and
            len(result.stderr) <= CLI_OUTPUT_CAP_BYTES,
            "CLI output exceeded its byte cap")
    return result, elapsed


def chat_base(binary, port):
    return [str(binary), "chat", "--url", f"http://127.0.0.1:{port}/v1",
            "--model", MODEL_NAME, "--prompt", PROMPT, "--max-tokens", "16",
            "--temperature", "0"]


def check_chat_retry(binary, port, env, processes, stage_pid, baseline):
    proxy = ByteTapProxy(port).plan("reset_after_request", "pass").start()
    try:
        result, elapsed = run_cli(chat_base(binary, proxy.port) +
                                  ["--timeout-ms", "20000", "--max-attempts", "2"],
                                  env)
    finally:
        proxy.close()
    require(result.returncode == 0,
            f"retry after reset failed: exit {result.returncode}, {result.stderr!r}")
    require(result.stdout == baseline,
            f"retry output {result.stdout!r} differs from baseline {baseline!r}")
    require(len(proxy.records) == 2,
            f"expected 2 connections for one retry, saw {len(proxy.records)}")
    require(proxy.records[0]["result"] == "reset" and
            proxy.records[0]["response_bytes"] == 0,
            f"first connection was not reset before a response: {proxy.records[0]}")
    require(proxy.records[1]["result"] == "pass",
            f"second connection did not pass through: {proxy.records[1]}")
    require(elapsed >= 0.2,
            f"retry did not wait for its backoff before reconnecting: {elapsed:.3f}s")
    wait_for_reap(port)
    require(processes["stage"].pid == stage_pid and processes["stage"].poll() is None,
            "MLX stage exited during the retry case")
    print(f"PASS real_retry_after_connection_reset attempts=2 connections=2 "
          f"latency_ms={elapsed * 1000:.1f} output={result.stdout!r}", flush=True)


def check_retry_budget(binary, port, env, processes, stage_pid):
    proxy = ByteTapProxy(port).plan("reset_after_request", "reset_after_request",
                                    "reset_after_request", "pass").start()
    try:
        result, elapsed = run_cli(chat_base(binary, proxy.port) +
                                  ["--timeout-ms", "20000", "--max-attempts", "3"],
                                  env)
    finally:
        proxy.close()
    require(result.returncode == 1,
            f"exhausted retry budget exited {result.returncode}: {result.stderr!r}")
    require(len(proxy.records) == 3,
            f"--max-attempts 3 made {len(proxy.records)} connections, expected 3")
    require(all(record["result"] == "reset" for record in proxy.records),
            f"an attempt was not reset by the proxy: {proxy.records}")
    wait_for_reap(port)
    require(processes["stage"].poll() is None, "MLX stage exited after retry budget")
    print(f"PASS real_retry_budget_exhausted attempts=3 connections=3 "
          f"exit=1 latency_ms={elapsed * 1000:.1f} stderr={result.stderr[-120:]!r}",
          flush=True)


def check_reset_after_delta(binary, port, env, processes, stage_pid):
    proxy = ByteTapProxy(port).plan("reset_after_delta").start()
    try:
        result, elapsed = run_cli(chat_base(binary, proxy.port) +
                                  ["--stream", "--timeout-ms", "20000",
                                   "--max-attempts", "3"], env)
    finally:
        proxy.close()
    require(result.returncode == 1,
            f"reset after delta exited {result.returncode}: {result.stderr!r}")
    require(result.stdout.strip(), "reset after delta produced no partial output")
    require(proxy.records[0]["delta_seen"], "proxy cut the stream before any delta")
    require(len(proxy.records) == 1,
            f"client retried after a delivered delta: {len(proxy.records)} connections")
    wait_for_reap(port)
    require(processes["stage"].poll() is None, "MLX stage exited after mid-stream reset")
    print(f"PASS real_no_retry_after_delta connections=1 exit=1 "
          f"partial_bytes={len(result.stdout)} latency_ms={elapsed * 1000:.1f}",
          flush=True)


def check_first_token_hold(binary, port, env, processes, stage_pid):
    proxy = ByteTapProxy(port).plan("hold_before_response").start()
    try:
        result, elapsed = run_cli(chat_base(binary, proxy.port) +
                                  ["--stream", "--first-token-ms", "900",
                                   "--timeout-ms", "20000", "--max-attempts", "1"],
                                  env, timeout=40)
    finally:
        proxy.close()
    require(result.returncode == 1,
            f"held first token exited {result.returncode}: {result.stderr!r}")
    require(b"first_token timeout" in result.stderr,
            f"held first token stderr was {result.stderr!r}")
    require(result.stdout == b"", "held first token still produced output")
    require(len(proxy.records) == 1,
            f"first-token timeout made {len(proxy.records)} connections")
    wait_for_reap(port)
    require(processes["stage"].poll() is None, "MLX stage exited after first-token hold")
    print(f"PASS real_stalled_first_token connections=1 exit=1 "
          f"latency_ms={elapsed * 1000:.1f}", flush=True)


def check_stall_after_delta(binary, port, env, processes, stage_pid):
    proxy = ByteTapProxy(port).plan("hold_after_delta").start()
    try:
        result, elapsed = run_cli(chat_base(binary, proxy.port) +
                                  ["--stream", "--stall-ms", "900",
                                   "--timeout-ms", "20000", "--max-attempts", "3"],
                                  env, timeout=40)
    finally:
        proxy.close()
    require(result.returncode == 1,
            f"stalled stream exited {result.returncode}: {result.stderr!r}")
    require(b"stall timeout" in result.stderr,
            f"stalled stream stderr was {result.stderr!r}")
    require(result.stdout.strip(), "stalled stream produced no partial output")
    require(len(proxy.records) == 1,
            f"stalled stream retried despite its delta: {len(proxy.records)} connections")
    wait_for_reap(port)
    require(processes["stage"].poll() is None, "MLX stage exited after stall")
    print(f"PASS real_stall_after_delta connections=1 exit=1 "
          f"partial_bytes={len(result.stdout)} latency_ms={elapsed * 1000:.1f}",
          flush=True)


def run_chat_cases(binary, stage, model, env):
    processes, logs, readers = {}, {}, []
    ready = threading.Event()
    succeeded = False
    try:
        with tempfile.TemporaryDirectory(prefix="umminfer-retry-e2e-") as root:
            entry = f"ipc://{root}/entry"
            completion = f"ipc://{root}/completion"
            abort = f"ipc://{root}/abort"
            port = free_port()
            spawn("server", [str(binary), "serve", "--host", "127.0.0.1",
                             "--port", str(port), "--model", MODEL_NAME,
                             "--entry", "mlx", "--entry-endpoint", entry,
                             "--completion", completion, "--abort", abort],
                  binary.parent, env, processes, logs, readers, ready, MLX_READY)
            wait_for_server(port, processes["server"])
            spawn("stage", [sys.executable, "-I", str(stage), "--name", "mlx",
                            "--recv", entry, "--coordinator", completion,
                            "--abort", abort, "--model", str(model)],
                  stage.parent, env, processes, logs, readers, ready, MLX_READY)
            wait_for_stage(processes["stage"], ready)
            stage_pid = processes["stage"].pid
            print(f"PASS real_chain_started port={port} stage_pid={stage_pid}",
                  flush=True)

            baseline, _ = run_cli(chat_base(binary, port) +
                                  ["--timeout-ms", "20000"], env)
            require(baseline.returncode == 0,
                    f"direct baseline failed: {baseline.stderr!r}")
            require(baseline.stdout.strip(), "direct baseline returned no text")
            print(f"PASS real_direct_baseline bytes={len(baseline.stdout)}", flush=True)

            check_chat_retry(binary, port, env, processes, stage_pid, baseline.stdout)
            check_retry_budget(binary, port, env, processes, stage_pid)
            check_reset_after_delta(binary, port, env, processes, stage_pid)
            check_first_token_hold(binary, port, env, processes, stage_pid)
            check_stall_after_delta(binary, port, env, processes, stage_pid)

            recovered, _ = run_cli(chat_base(binary, port) +
                                   ["--timeout-ms", "20000"], env)
            require(recovered.returncode == 0 and recovered.stdout == baseline.stdout,
                    f"chain did not recover after failures: {recovered.stderr!r}")
            require(processes["stage"].pid == stage_pid and
                    processes["stage"].poll() is None,
                    "MLX stage was replaced during recovery")
            print("PASS real_chain_recovered_after_failures stage_persisted", flush=True)
            succeeded = True
    finally:
        errors = cleanup(processes, readers)
        if processes and not errors:
            print(f"PASS real_cleanup exit_codes="
                  f"{ {n: p.returncode for n, p in processes.items()} }", flush=True)
        for error in errors:
            print(f"FAIL real_cleanup: {error}", file=sys.stderr, flush=True)
        if not succeeded:
            for name, data in logs.items():
                print(f"{name} log tail:\n"
                      f"{bytes(data).decode(errors='replace')[-4000:]}",
                      file=sys.stderr, flush=True)
        if succeeded and errors:
            raise RuntimeError("; ".join(errors))


def generate_audio(path):
    sample_rate = 16000
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        frames = []
        for start in range(0, sample_rate // 2, 400):
            samples = [int(0.2 * 32767 *
                           math.sin(2 * math.pi * 440 * index / sample_rate))
                       for index in range(start, min(start + 400, sample_rate // 2))]
            frames.extend(samples)
        wav.writeframes(struct.pack(f"<{len(frames)}h", *frames))


def run_asr_case(binary, asr_stage, env):
    processes, logs, readers = {}, {}, []
    ready = threading.Event()
    succeeded = False
    try:
        with tempfile.TemporaryDirectory(prefix="umminfer-retry-asr-") as root:
            root = Path(root)
            (root / "ipc").mkdir()
            model_dir = root / "model"
            audio = root / "audio.wav"
            generated = subprocess.run(
                [sys.executable, "-I", str(asr_stage), "generate",
                 "--output", str(model_dir)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
                timeout=STARTUP_TIMEOUT_S)
            require(generated.returncode == 0,
                    f"tiny ASR model generation failed: {generated.stderr!r}")
            generate_audio(audio)

            entry = f"ipc://{root}/ipc/entry"
            completion = f"ipc://{root}/ipc/completion"
            abort = f"ipc://{root}/ipc/abort"
            port = free_port()
            spawn("server", [str(binary), "serve", "--host", "127.0.0.1",
                             "--port", str(port), "--model", ASR_MODEL_NAME,
                             "--entry", "asr", "--entry-endpoint", entry,
                             "--completion", completion, "--abort", abort],
                  binary.parent, env, processes, logs, readers, ready, ASR_READY)
            wait_for_server(port, processes["server"])
            spawn("stage", [sys.executable, "-I", str(asr_stage), "serve",
                            "--name", "asr", "--recv", entry,
                            "--coordinator", completion, "--abort", abort,
                            "--model", str(model_dir)],
                  asr_stage.parent, env, processes, logs, readers, ready, ASR_READY)
            wait_for_stage(processes["stage"], ready)
            stage_pid = processes["stage"].pid

            direct = [str(binary), "transcribe", "--url", f"http://127.0.0.1:{port}",
                      "--model", ASR_MODEL_NAME, "--file", str(audio),
                      "--timeout-ms", "20000"]
            baseline, _ = run_cli(direct, env)
            require(baseline.returncode == 0,
                    f"direct transcribe failed: {baseline.stderr!r}")
            print(f"PASS real_asr_baseline stage_pid={stage_pid} "
                  f"output={baseline.stdout!r}", flush=True)

            proxy = ByteTapProxy(port).plan("hold_before_response").start()
            try:
                stalled, elapsed = run_cli(
                    [str(binary), "transcribe",
                     "--url", f"http://127.0.0.1:{proxy.port}",
                     "--model", ASR_MODEL_NAME, "--file", str(audio),
                     "--stall-ms", "900", "--timeout-ms", "20000"],
                    env, timeout=40)
            finally:
                proxy.close()
            require(stalled.returncode == 1,
                    f"stalled transcribe exited {stalled.returncode}: "
                    f"{stalled.stderr!r}")
            require(b"activity timeout" in stalled.stderr,
                    f"stalled transcribe stderr was {stalled.stderr!r}")
            require(len(proxy.records) == 1,
                    f"transcribe retried despite having no retry loop: "
                    f"{len(proxy.records)} connections")
            wait_for_reap(port)
            require(processes["stage"].poll() is None,
                    "TinyASR stage exited after the stalled transcribe")
            print(f"PASS real_asr_stalled_progress connections=1 exit=1 "
                  f"latency_ms={elapsed * 1000:.1f}", flush=True)

            recovered, _ = run_cli(direct, env)
            require(recovered.returncode == 0 and recovered.stdout == baseline.stdout,
                    f"ASR chain did not recover: {recovered.stderr!r}")
            print("PASS real_asr_recovered_after_stall stage_persisted", flush=True)
            succeeded = True
    finally:
        errors = cleanup(processes, readers,
                         -signal.SIGTERM if processes else 0)
        if errors:
            for error in errors:
                print(f"FAIL asr_cleanup: {error}", file=sys.stderr, flush=True)
            if succeeded:
                raise RuntimeError("; ".join(errors))
        if not succeeded:
            for name, data in logs.items():
                print(f"{name} log tail:\n"
                      f"{bytes(data).decode(errors='replace')[-4000:]}",
                      file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--stage", type=Path,
                        default=Path("/Users/bytedance/code/omniInfer/python/"
                                     "umminfer_mlx_stage.py"))
    parser.add_argument("--model", type=Path,
                        default=Path("/Users/bytedance/.cache/huggingface/hub/"
                                     "models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/"
                                     "snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"))
    parser.add_argument("--asr-stage", type=Path,
                        default=Path("/Users/bytedance/code/omniInfer/python/"
                                     "umminfer_tiny_asr_stage.py"))
    parser.add_argument("--skip-asr", action="store_true",
                        help="run only the chat retry and stall cases")
    args = parser.parse_args()
    binary = args.binary.resolve()
    stage = args.stage.resolve()
    model = args.model.resolve()
    asr_stage = args.asr_stage.resolve()
    for path, label in ((binary, "binary"), (stage, "stage"),
                        (model, "model"), (asr_stage, "ASR stage")):
        require(path.exists(), f"{label} not found: {path}")

    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                "TOKENIZERS_PARALLELISM": "false"})
    with binary.open("rb") as source:
        print(f"binary_sha256={hashlib.file_digest(source, 'sha256').hexdigest()}",
              flush=True)
    run_chat_cases(binary, stage, model, env)
    if not args.skip_asr:
        run_asr_case(binary, asr_stage, env)
    print("PASS client_retry_e2e", flush=True)


if __name__ == "__main__":
    main()

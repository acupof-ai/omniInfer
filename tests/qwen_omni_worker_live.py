#!/usr/bin/env python3
"""Opt-in real Qwen3-Omni E2E through CLI/public HTTP and the new Worker."""

import base64
import http.client
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import live_common as live  # noqa: E402

REQUIRED = [
    "UMMINFER_BIN",
    "UMMINFER_QWEN_BASE_URL",
    "UMMINFER_QWEN_MODEL",
    "UMMINFER_QWEN_IMAGE",
]
OPTIONAL = [
    "UMMINFER_QWEN_REVISION",
    "UMMINFER_QWEN_GPU",
    "UMMINFER_QWEN_TOPOLOGY",
]


class Worker:
    def __init__(self, args, workdir, log_name="qwen-worker.log", env=None):
        self.process = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=workdir, env=env)
        self.log_path = os.path.join(workdir, log_name)
        self.bytes = bytearray()
        self.lock = threading.Lock()
        self.reader = threading.Thread(target=self.drain, daemon=True)
        self.reader.start()

    def drain(self):
        while True:
            data = os.read(self.process.stdout.fileno(), 65536)
            if not data:
                break
            with self.lock:
                self.bytes.extend(data)
                if len(self.bytes) > live.LOG_CAP:
                    del self.bytes[:len(self.bytes) - live.LOG_CAP]

    def tail(self):
        with self.lock:
            return bytes(self.bytes).decode(errors="replace")

    def stop(self):
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.reader.join(5)
        self.process.stdout.close()
        tail = self.tail()
        with open(self.log_path, "w") as output:
            output.write(tail)
        live.require(len(tail.encode()) <= live.LOG_CAP, "worker log exceeded its cap")
        return self.process.returncode


class Serve(Worker):
    def __init__(self, binary, args, env, workdir):
        super().__init__([binary, "serve", *args], workdir,
                         log_name="qwen-serve.log", env=env)
        self.port = 0

    def __enter__(self):
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                live.require(self.process.poll() is None,
                             "server exited: " + self.tail())
                match = re.search(r"serving on [^\s:]+:(\d+)", self.tail())
                if match:
                    self.port = int(match.group(1))
                    try:
                        if live.http_get(self.port, "/health")[0] == 200:
                            return self
                    except OSError:
                        pass
                time.sleep(0.05)
            raise live.LiveFail("server startup timed out: " + self.tail())
        except BaseException:
            self.stop()
            raise

    def __exit__(self, *exc):
        self.stop()
        return False


def post_json(port, path, payload, timeout=240):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("POST", path, json.dumps(payload),
                     {"Content-Type": "application/json"})
        response = conn.getresponse()
        return response.status, response.read().decode(errors="replace")
    finally:
        conn.close()


def open_stream(port, model, prompt, max_tokens):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=240)
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": max_tokens, "stream": True}
    conn.request("POST", "/v1/chat/completions", json.dumps(body),
                 {"Content-Type": "application/json"})
    response = conn.getresponse()
    live.require(response.status == 200, f"SSE HTTP {response.status}")
    return conn, response


def next_event(response):
    lines = []
    while True:
        raw = response.readline()
        live.require(bool(raw), "SSE ended before [DONE]")
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if not lines:
                continue
            data = "\n".join(lines)
            return None if data == "[DONE]" else json.loads(data)
        if line.startswith("data:"):
            lines.append(line[5:].lstrip())


def wait_in_flight(port, expected=0, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status, body = live.http_get(port, "/health")
        if status == 200 and json.loads(body)["in_flight"] == expected:
            return body
        time.sleep(0.05)
    raise live.LiveFail(f"in_flight did not reach {expected}: {body}")


def cli_chat(binary, url, model, prompt, env, work, image_url=None):
    args = ["chat", "--url", url, "--model", model,
            "--prompt", prompt, "--temperature", "0", "--max-tokens", "128",
            "--timeout-ms", "240000", "--max-attempts", "1"]
    if image_url is not None:
        args += ["--image-url", image_url]
    started = time.perf_counter()
    result = live.run_cli(binary, args, env, work, timeout_s=260)
    seconds = time.perf_counter() - started
    live.require(result.returncode == 0, "CLI chat failed: " + result.stderr[-2000:])
    return result.stdout.strip(), seconds


def run():
    binary = live.binary_path(os.environ["UMMINFER_BIN"])
    base_url = os.environ["UMMINFER_QWEN_BASE_URL"].rstrip("/")
    model = os.environ["UMMINFER_QWEN_MODEL"]
    served = os.environ.get("UMMINFER_SERVED_MODEL", "qwen3-omni")
    live.require(base_url.startswith("http://127.0.0.1:"),
                 "this E2E requires the documented loopback SSH tunnel")
    image_bytes = Path(os.environ["UMMINFER_QWEN_IMAGE"]).read_bytes()
    image_kind, (image_digest, image_size), image_info = live.check_image(
        os.environ["UMMINFER_QWEN_IMAGE"])
    image_url = ("data:image/" + ("jpeg" if image_kind == "jpeg" else "png")
                 + ";base64," + base64.b64encode(image_bytes).decode())
    worker_path = str(Path(__file__).resolve().parents[1]
                      / "python" / "umminfer_qwen_omni_worker.py")
    keep = "--keep" in sys.argv[1:]
    work = tempfile.mkdtemp(prefix="umm-live-qwen-worker-")
    env = live.cli_env()
    evidence = {
        "chain": os.environ.get(
            "UMMINFER_QWEN_TOPOLOGY",
            "real CLI/public HTTP -> ummInfer serve -> QwenOmniWorker -> SGLang -> real model"),
        "model": model,
        "revision": os.environ.get("UMMINFER_QWEN_REVISION", "NOT_COVERED"),
        "gpu": os.environ.get("UMMINFER_QWEN_GPU", "NOT_COVERED"),
        "image": {"kind": image_kind, "sha256": image_digest,
                  "bytes": image_size, **image_info},
        "checks": {},
        "not_covered": ["audio and video generation", "audio output"],
    }
    try:
        ipc_dir = os.path.join(work, "ipc")
        entry = f"ipc://{ipc_dir}/entry"
        completion = f"ipc://{ipc_dir}/completion"
        abort = f"ipc://{ipc_dir}/abort"
        serve_args = ["--host", "127.0.0.1", "--port", "0", "--model", served,
                      "--entry", "qwen", "--entry-endpoint", entry,
                      "--completion", completion, "--abort", abort]
        with Serve(binary, serve_args, dict(os.environ), work) as server:
            ready_before = live.http_get(server.port, "/ready")
            live.require(ready_before[0] == 503, f"ready before Worker: {ready_before}")
            worker = Worker([
                sys.executable, "-I", worker_path, "--name", "qwen",
                "--recv", entry, "--coordinator", completion, "--abort", abort,
                "--upstream-base-url", base_url, "--upstream-model", model,
            ], work)
            try:
                deadline = time.monotonic() + 30
                while True:
                    ready_after = live.http_get(server.port, "/ready")
                    if ready_after[0] == 200:
                        break
                    live.require(worker.process.poll() is None,
                                 "Worker exited: " + worker.tail())
                    live.require(time.monotonic() < deadline,
                                 "Worker readiness timed out: " + worker.tail())
                    time.sleep(0.05)
                evidence["checks"]["ready"] = {
                    "before_worker_status": ready_before[0],
                    "after_worker_status": ready_after[0],
                    "after_worker_body": json.loads(ready_after[1]),
                }
                url = f"http://127.0.0.1:{server.port}/v1"

                text, seconds = cli_chat(
                    binary, url, served, "Answer with exactly: OMNI_TEXT_OK", env, work)
                live.require(text == "OMNI_TEXT_OK", f"text assertion: {text}")
                evidence["checks"]["text"] = {"output": text, "seconds": seconds}

                image_text, seconds = cli_chat(
                    binary, url, served,
                    "Describe the person, vehicle, and action in this image.",
                    env, work, image_url)
                lower = image_text.lower()
                for label, words in {
                    "person": ("man", "person", "driver"),
                    "vehicle": ("cab", "taxi", "suv", "vehicle", "car"),
                    "action": ("iron", "hang", "cloth", "holding"),
                }.items():
                    live.require(any(word in lower for word in words),
                                 f"image {label} assertion: {image_text}")
                evidence["checks"]["image"] = {"output": image_text, "seconds": seconds}

                started = time.perf_counter()
                conn, response = open_stream(
                    server.port, served, "Answer with exactly: SSE_ALPHA BETA", 64)
                events = []
                while True:
                    event = next_event(response)
                    if event is None:
                        break
                    events.append(event)
                response.close()
                conn.close()
                seconds = time.perf_counter() - started
                live.require(len(events) >= 3, f"too few SSE events: {events}")
                live.require(events[0]["choices"][0]["delta"].get("role") == "assistant",
                             f"first SSE event is not the role: {events[0]}")
                deltas = [choice["delta"].get("content", "")
                          for event in events for choice in event.get("choices", [])]
                stream_text = "".join(deltas)
                live.require("SSE_ALPHA" in stream_text and "BETA" in stream_text,
                             f"SSE content assertion: {stream_text}")
                live.require(stream_text.index("SSE_ALPHA") < stream_text.index("BETA"),
                             f"SSE content order: {stream_text}")
                live.require(events[-1]["choices"][0]["finish_reason"] == "stop",
                             f"last SSE event is not finish=stop: {events[-1]}")
                live.require(not events[-1]["choices"][0]["delta"].get("content"),
                             "content followed the finish event")
                evidence["checks"]["sse"] = {
                    "output": stream_text, "seconds": seconds, "events": len(events),
                    "order": ["role", "content", "finish=stop", "[DONE]"]}

                started = time.perf_counter()
                conn, response = open_stream(
                    server.port, served,
                    "Count from 1 to 100 with one number per line.", 512)
                request_id = None
                while request_id is None:
                    event = next_event(response)
                    live.require(event is not None, "cancel stream ended early")
                    request_id = event.get("id")
                while True:
                    event = next_event(response)
                    if any(choice.get("delta", {}).get("content")
                           for choice in event.get("choices", [])):
                        break
                abort_status, abort_body = post_json(
                    server.port, "/abort", {"request_id": request_id})
                live.require(abort_status == 200,
                             f"abort HTTP {abort_status}: {abort_body}")
                response.close()
                conn.close()
                health = wait_in_flight(server.port)
                seconds = time.perf_counter() - started
                evidence["checks"]["cancel"] = {
                    "request_id": request_id, "abort_status": abort_status,
                    "abort_body": json.loads(abort_body), "health": json.loads(health),
                    "seconds_to_in_flight_zero": seconds}

                followup, seconds = cli_chat(
                    binary, url, served, "Answer with exactly: FOLLOWUP_OK", env, work)
                live.require(followup == "FOLLOWUP_OK", f"followup assertion: {followup}")
                evidence["checks"]["followup"] = {
                    "output": followup, "seconds": seconds,
                    "health": json.loads(wait_in_flight(server.port))}
            finally:
                evidence["worker_exit"] = worker.stop()
            info_url = base_url[:-3] + "/get_server_info" if base_url.endswith("/v1") else None
            if info_url:
                with urllib.request.urlopen(info_url, timeout=10) as response:
                    info = json.load(response)
                evidence["sglang_supporting_metadata"] = {
                    key: info.get(key) for key in (
                        "model_path", "served_model_name", "dtype", "quantization",
                        "version", "weight_version")}
        evidence["server_exit"] = server.process.returncode
        for name in ("qwen-serve.log", "qwen-worker.log"):
            live.require(os.path.getsize(os.path.join(work, name)) <= live.LOG_CAP,
                         f"uncapped log: {name}")
        evidence_path = os.path.join(work, "evidence.json")
        with open(evidence_path, "w") as output:
            json.dump(evidence, output, indent=2, ensure_ascii=False)
            output.write("\n")
        for name, check in evidence["checks"].items():
            print(f"PASS {name}: {json.dumps(check, ensure_ascii=False)}")
        print(f"PASS log caps: {live.LOG_CAP} bytes")
        print(f"EVIDENCE={evidence_path}")
    finally:
        if keep:
            print(f"KEPT workdir: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(live.blocked_main("qwen_omni_worker_live.py", REQUIRED, run, OPTIONAL))

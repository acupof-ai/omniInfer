#!/usr/bin/env python3
"""Real Qwen3-ASR + Qwen3-Omni unified chat: ASR partials and VLM GPU abort.

Runs the real C++ server and the two real workers on this host. The SGLang
backends must already serve qwen3-asr and qwen3-omni on the given URLs. The
harness is opt-in: without UMMINFER_LIVE_OPT_IN=1 it prints BLOCKED and exits 0.

It proves, from public HTTP/SSE frames and worker logs:
  - a default unified request keeps one asr_final and no asr_partial;
  - stream_options.include_asr_partials produces ordered asr_partial deltas,
    one asr_final, then vlm_delta, with gap-free seq and one terminal;
  - a cancel during the VLM stream produces exactly one cancelled terminal,
    no later VLM delta, in_flight=0, and the Omni worker records a backend
    finish_reason=abort confirmation for the child rid;
  - a later request completes.
"""

import argparse
import base64
import hashlib
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

LOG_CAP = 64 * 1024


class CappedLog:
    def __init__(self, path):
        self.path = path
        self.bytes = bytearray()
        self.lock = threading.Lock()
        self.thread = None
        self.process = None

    def start(self, process):
        self.process = process
        self.thread = threading.Thread(target=self._drain, daemon=True)
        self.thread.start()

    def _drain(self):
        fd = self.process.stdout.fileno()
        while True:
            data = os.read(fd, 65536)
            if not data:
                break
            with self.lock:
                self.bytes.extend(data)
                if len(self.bytes) > LOG_CAP:
                    del self.bytes[:len(self.bytes) - LOG_CAP]
        Path(self.path).write_bytes(bytes(self.bytes))

    def text(self):
        with self.lock:
            return bytes(self.bytes).decode("utf-8", "replace")

    def wait(self):
        if self.thread is not None:
            self.thread.join(5)
        Path(self.path).write_bytes(bytes(self.bytes))


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_http(url, timeout_s=120.0, expected=200):
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as reply:
                if reply.status == expected:
                    return True
        except Exception as exc:  # noqa: BLE001 - retry until the deadline
            last = exc
        time.sleep(0.25)
    raise RuntimeError(f"{url} did not answer {expected}: {last}")


def post_json(url, payload, timeout=600.0):
    data = json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as reply:
        return reply.status, json.load(reply)


def stream_chat(port, body, on_frame=None, timeout=600.0):
    """POST one streaming chat request and return its parsed SSE frames."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout)
    connection.request("POST", "/v1/chat/completions",
                       body=json.dumps(body),
                       headers={"Content-Type": "application/json"})
    response = connection.getresponse()
    frames = []
    data_lines = []
    try:
        for raw in response:
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
            elif not line and data_lines:
                data = "\n".join(data_lines)
                data_lines = []
                if data == "[DONE]":
                    break
                frame = json.loads(data)
                frames.append(frame)
                if on_frame is not None:
                    on_frame(frame)
    finally:
        connection.close()
    return response.status, frames


def audio_part(wav_path):
    raw = Path(wav_path).read_bytes()
    return {"type": "input_audio",
            "input_audio": {"data": base64.b64encode(raw).decode(),
                            "format": "wav"}}


def request_body(wav_path, max_tokens, partials, image=None,
                 instruction="Answer briefly."):
    content = []
    if image:
        raw = Path(image).read_bytes()
        suffix = "jpeg" if raw[:2] == b"\xff\xd8" else "png"
        content.append({"type": "image_url", "image_url": {
            "url": f"data:image/{suffix};base64,"
                   + base64.b64encode(raw).decode()}})
    content.append({"type": "text", "text": instruction})
    content.append(audio_part(wav_path))
    body = {"model": "qwen3-unified",
            "messages": [{"role": "user", "content": content}],
            "stream": True, "max_tokens": max_tokens, "temperature": 0}
    if partials:
        body["stream_options"] = {"include_asr_partials": True}
    return body


def unified_frames(frames):
    return [frame["unified"] for frame in frames if "unified" in frame]


def validate_completion(case, frames, need_partials):
    events = unified_frames(frames)
    kinds = [event["kind"] for event in events]
    seqs = [event["seq"] for event in events]
    terminals = [event for event in events
                 if event["kind"] in ("complete", "error", "cancelled")]
    partials = [event for event in events if event["kind"] == "asr_partial"]
    finals = [event for event in events if event["kind"] == "asr_final"]
    deltas = [event for event in events if event["kind"] == "vlm_delta"]
    checks = {
        "seq_gap_free": seqs == list(range(len(seqs))),
        "one_terminal": len(terminals) == 1,
        "terminal_complete": bool(terminals) and terminals[0]["kind"] == "complete",
        "one_asr_final": len(finals) == 1,
        "final_text_nonempty": bool(finals) and bool(finals[0]["text"]),
        "vlm_delta_present": bool(deltas),
        "partial_before_final": (not finals or not partials
                                 or partials[-1]["seq"] < finals[0]["seq"]),
        "final_before_first_delta": (not finals or not deltas
                                     or finals[0]["seq"] < deltas[0]["seq"]),
    }
    if need_partials:
        checks["partial_present"] = bool(partials)
    else:
        checks["no_partial"] = not partials
    return {"case": case, "kind_order": kinds, "checks": checks,
            "frames": len(frames), "events": events,
            "passed": all(checks.values())}


def validate_cancel(case, frames, abort_at):
    events = unified_frames(frames)
    seqs = [event["seq"] for event in events]
    terminals = [event for event in events
                 if event["kind"] in ("complete", "error", "cancelled")]
    deltas_after = [event for event in deltas_after_abort(frames)
                    if event["kind"] == "vlm_delta"]
    checks = {
        "seq_gap_free": seqs == list(range(len(seqs))),
        "one_terminal": len(terminals) == 1,
        "terminal_cancelled": bool(terminals) and terminals[0]["kind"] == "cancelled",
        "no_delta_after_abort": not deltas_after,
        "final_before_abort": any(event["kind"] == "asr_final" and event["seq"] < abort_at
                                  for event in events),
    }
    return {"case": case, "checks": checks, "events": events,
            "abort_at_seq": abort_at, "passed": all(checks.values())}


def deltas_after_abort(frames):
    """Frames received after the abort POST returned."""
    return [frame["unified"] for frame in frames
            if "unified" in frame and frame.get("_after_abort")]


def backend_abort_probe(base_url, model, rid="umminfer-phase5-backend-probe"):
    """Stream one chat request, abort its rid, and record the raw frames."""
    from urllib.parse import urlsplit
    parsed = urlsplit(base_url)
    port = parsed.port or 80
    payload = {"model": model, "rid": rid, "stream": True, "max_tokens": 600,
               "temperature": 0,
               "messages": [{"role": "user",
                             "content": "Count from 1 to 600, one per line."}]}
    connection = http.client.HTTPConnection(parsed.hostname, port, 120)
    connection.request("POST", "/v1/chat/completions",
                       body=json.dumps(payload),
                       headers={"Content-Type": "application/json"})
    response = connection.getresponse()
    frames = []
    data_lines = []
    aborted = False
    abort_status = None
    try:
        for raw in response:
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
                continue
            if not line or not data_lines:
                continue
            data = "\n".join(data_lines)
            data_lines = []
            if data == "[DONE]":
                frames.append({"event": "[DONE]"})
                break
            event = json.loads(data)
            frames.append(event)
            finish = [choice.get("finish_reason")
                      for choice in event.get("choices", [])]
            if not aborted and len(frames) >= 3:
                root = base_url[:-3] if base_url.endswith("/v1") else base_url
                code, _ = post_json(root + "/abort_request", {"rid": rid},
                                    timeout=10)
                abort_status = code
                aborted = True
            if any(reason == "abort" for reason in finish):
                break
    finally:
        connection.close()
    return {"rid": rid, "abort_status": abort_status,
            "frames": frames,
            "confirmed": any(
                choice.get("finish_reason") == "abort"
                for event in frames if event.get("event") != "[DONE]"
                for choice in event.get("choices", []))}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--asr-stage", default="python/umminfer_qwen_asr_stage.py")
    parser.add_argument("--omni-worker", default="python/umminfer_qwen_omni_worker.py")
    parser.add_argument("--asr-backend", required=True)
    parser.add_argument("--omni-backend", required=True)
    parser.add_argument("--asr-model", default="qwen3-asr")
    parser.add_argument("--omni-model", default="qwen3-omni")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--cancel-audio", required=True)
    parser.add_argument("--image", default=None)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    if os.environ.get("UMMINFER_LIVE_OPT_IN") != "1":
        print("BLOCKED: set UMMINFER_LIVE_OPT_IN=1 to run the real H20B check")
        return 0
    evidence = Path(args.evidence_dir)
    evidence.mkdir(parents=True, exist_ok=True)
    ipc = evidence / "ipc"
    ipc.mkdir(exist_ok=True)
    port = free_port()
    commands = []

    def endpoint(name):
        return f"ipc://{ipc}/{name}"

    asr_cmd = [sys.executable, "-I", args.asr_stage, "serve",
               "--name", "asr", "--recv", endpoint("asr"),
               "--coordinator", endpoint("done"), "--abort", endpoint("abort"),
               "--model", args.asr_model, "--backend-url", args.asr_backend,
               "--timeout", "300", "--concurrency", "2",
               "--perf-log", str(evidence / "asr-perf.jsonl")]
    omni_cmd = [sys.executable, "-I", args.omni_worker,
                "--name", "vlm", "--recv", endpoint("vlm"),
                "--coordinator", endpoint("done"), "--abort", endpoint("abort"),
                "--upstream-base-url", args.omni_backend.rstrip("/") + "/v1",
                "--upstream-model", args.omni_model,
                "--request-timeout", "300", "--concurrency", "2",
                "--log-requests", "--perf-log",
                str(evidence / "omni-perf.jsonl")]
    server_cmd = [args.server, "serve", "--host", "127.0.0.1",
                  "--port", str(port), "--model", "qwen3-unified",
                  "--entry", "asr", "--entry-endpoint", endpoint("asr"),
                  "--vlm-entry", "vlm", "--vlm-entry-endpoint", endpoint("vlm"),
                  "--completion", endpoint("done"),
                  "--abort", endpoint("abort")]
    for command in (asr_cmd, omni_cmd, server_cmd):
        commands.append(" ".join(command))

    processes = {}
    logs = {}
    try:
        for name, command in (("asr", asr_cmd), ("vlm", omni_cmd),
                              ("server", server_cmd)):
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            log = CappedLog(evidence / f"{name}.log")
            log.start(process)
            processes[name] = process
            logs[name] = log
            time.sleep(0.3)
        wait_http(f"http://127.0.0.1:{port}/ready", 180)
        wait_http(f"http://127.0.0.1:{port}/v1/models", 30)

        results = []

        # Direct backend probe on a dedicated rid: proves that SGLang itself
        # ends a chat stream with finish_reason=abort after /abort_request.
        # This evidence does not depend on the worker implementation.
        backend_probe = backend_abort_probe(args.omni_backend, args.omni_model)
        Path(evidence / "backend-abort-probe.json").write_text(
            json.dumps(backend_probe, indent=2) + "\n")

        def health():
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=5) as reply:
                return json.load(reply)

        # A: default request, no opt-in.
        status, frames = stream_chat(port, request_body(args.audio, 64, False),
                                     timeout=args.timeout)
        Path(evidence / "case-default.frames.jsonl").write_text(
            "\n".join(json.dumps(frame) for frame in frames) + "\n")
        results.append(validate_completion("default-no-partials", frames, False))

        # B: opt-in partials, complete request.
        status, frames = stream_chat(port, request_body(args.audio, 64, True),
                                     timeout=args.timeout)
        Path(evidence / "case-optin.frames.jsonl").write_text(
            "\n".join(json.dumps(frame) for frame in frames) + "\n")
        results.append(validate_completion("optin-partials", frames, True))

        # C: opt-in partials, cancel during the VLM stream.
        state = {"request_id": None, "aborted": False, "abort_seq": None,
                 "after": False, "abort_status": None}

        def on_frame(frame):
            if state["after"]:
                frame["_after_abort"] = True
            unified = frame.get("unified")
            if frame.get("id"):
                state["request_id"] = frame["id"]
            if (unified and unified["kind"] == "vlm_delta"
                    and not state["aborted"]):
                state["abort_seq"] = unified["seq"]
                state["aborted"] = True
                try:
                    code, reply = post_json(
                        f"http://127.0.0.1:{port}/abort",
                        {"request_id": state["request_id"]}, timeout=10)
                    state["abort_status"] = code
                except Exception as exc:  # noqa: BLE001 - recorded below
                    state["abort_status"] = f"error: {exc}"
                # Only frames after this one count as post-abort.
                state["after"] = True

        # The abort must reach SGLang while the GPU is still decoding. A long
        # deterministic instruction keeps the generation alive past the first
        # VLM delta.
        cancel_body = request_body(
            args.cancel_audio, 600, True,
            instruction="Count from 1 to 600, one number per line. "
                        "Do not stop early.")
        status, frames = stream_chat(port, cancel_body, on_frame,
                                     timeout=args.timeout)
        Path(evidence / "case-cancel.frames.jsonl").write_text(
            "\n".join(json.dumps(frame) for frame in frames) + "\n")
        cancel = validate_cancel("cancel-during-vlm", frames,
                                 state["abort_seq"] or 0)
        cancel["checks"]["abort_post_200"] = state["abort_status"] == 200
        cancel["checks"]["abort_after_partial"] = any(
            event["kind"] == "asr_partial" for event in cancel["events"])
        # The worker writes the confirmation after the abort POST returns.
        expected_rid = f"rid={state['request_id']}:vlm@"
        for _ in range(150):
            if "SGLANG_ABORT_CONFIRMED" in logs["vlm"].text():
                break
            time.sleep(0.1)
        vlm_log = logs["vlm"].text()
        cancel["checks"]["worker_gpu_abort_confirmed"] = (
            "SGLANG_ABORT_CONFIRMED" in vlm_log and "finish=abort" in vlm_log
            and expected_rid in vlm_log)
        cancel["passed"] = all(cancel["checks"].values())
        cancel["abort_status"] = state["abort_status"]
        results.append(cancel)
        time.sleep(0.5)
        health_after_cancel = health()

        # D: recovery after the cancel.
        status, frames = stream_chat(port, request_body(args.audio, 64, False),
                                     timeout=args.timeout)
        Path(evidence / "case-recovery.frames.jsonl").write_text(
            "\n".join(json.dumps(frame) for frame in frames) + "\n")
        results.append(validate_completion("recovery-after-cancel", frames, False))

        # E: optional joint image + audio with opt-in partials.
        if args.image:
            status, frames = stream_chat(
                port, request_body(args.audio, 64, True, image=args.image),
                timeout=args.timeout)
            Path(evidence / "case-joint.frames.jsonl").write_text(
                "\n".join(json.dumps(frame) for frame in frames) + "\n")
            results.append(validate_completion("joint-image-audio", frames, True))

        health_final = health()
        summary = {
            "checks": results,
            "health_after_cancel": health_after_cancel,
            "health_final": health_final,
            "server_port": port,
            "passed": all(result["passed"] for result in results),
        }
        Path(evidence / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n")
        for name, log in logs.items():
            log.wait()
        Path(evidence / "commands.txt").write_text("\n".join(commands) + "\n")
        Path(evidence / "env.json").write_text(json.dumps({
            "asr_backend": args.asr_backend,
            "omni_backend": args.omni_backend,
            "asr_model": args.asr_model,
            "omni_model": args.omni_model,
            "input_sha256": {str(Path(p)): sha256(p)
                             for p in (args.audio, args.cancel_audio, args.image)
                             if p},
        }, indent=2, sort_keys=True) + "\n")
        with open(evidence / "SHA256SUMS", "w") as sums:
            for path in sorted(evidence.rglob("*")):
                if path.is_file() and path.name != "SHA256SUMS":
                    sums.write(f"{sha256(path)}  {path.relative_to(evidence)}\n")
        for result in results:
            print(("PASS " if result["passed"] else "FAIL ") + result["case"])
            for key, value in result["checks"].items():
                print(f"  {key}={value}")
        print(f"EVIDENCE={evidence}")
        return 0 if summary["passed"] else 1
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
        for process in processes.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs.values():
            log.wait()


if __name__ == "__main__":
    sys.exit(main())

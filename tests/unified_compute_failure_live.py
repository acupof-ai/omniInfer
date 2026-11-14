#!/usr/bin/env python3
"""Phase-6 real model-compute failure validation on H20B GPU 3.

Opt-in: without UMMINFER_LIVE_OPT_IN=1 this prints BLOCKED and exits 0. It
needs the H20B host, GPU 3, the model directories, a ReleaseFast umm-infer, and
an independent SGLang copy patched by tests/sglang_nan_injection.py.

One fault leg (ASR or VLM) through the public unified HTTP/SSE path:

  baseline request -> arm the one-shot NaN trigger -> fault request
  -> SGLang NaN detection + crash -> public error, in_flight=0
  -> restart the same SGLang copy with no trigger -> recovery completes.

The patched LogitsProcessor corrupts the final hidden states of one decode step
when the trigger file exists. Only that copy is patched. The other leg uses an
unmodified shared backend.
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
PY = sys.executable
BACKEND_ROOT = Path("/root/phase6-compute-failure/backend")
LAUNCHER = BACKEND_ROOT / "phase6_launch_copy.py"
MODELS = Path("/root/omniinfer-model-validation-20261111-k3/models")


class CappedLog:
    """Keep at most LOG_CAP bytes of a process's combined output."""

    def __init__(self, path):
        self.path = Path(path)
        self.bytes = bytearray()
        self.lock = threading.Lock()
        self.thread = None

    def start(self, process):
        self.thread = threading.Thread(target=self._drain, args=(process,),
                                       daemon=True)
        self.thread.start()

    def _drain(self, process):
        fd = process.stdout.fileno()
        while True:
            data = os.read(fd, 65536)
            if not data:
                break
            with self.lock:
                self.bytes.extend(data)
                if len(self.bytes) > LOG_CAP:
                    del self.bytes[:len(self.bytes) - LOG_CAP]
                self.path.write_bytes(bytes(self.bytes))

    def wait(self):
        if self.thread is not None:
            self.thread.join(10)
        with self.lock:
            self.path.write_bytes(bytes(self.bytes))

    def text(self):
        with self.lock:
            return bytes(self.bytes).decode("utf-8", "replace")


def wait_http(url, timeout_s=120.0, expected=200, proc=None, log=None):
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            tail = log.text()[-2000:] if log is not None else ""
            raise RuntimeError(
                f"process exited {proc.returncode} before {url} answered "
                f"{expected}; log tail:\n{tail}")
        try:
            with urllib.request.urlopen(url, timeout=5) as reply:
                if reply.status == expected:
                    return json.loads(reply.read() or b"{}")
        except Exception as exc:  # noqa: BLE001 - retry until the deadline
            last = exc
        time.sleep(0.5)
    raise RuntimeError(f"{url} did not answer {expected}: {last}")


def require_free_port(port):
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def http_get_json(url):
    with urllib.request.urlopen(url, timeout=10) as reply:
        return reply.status, json.loads(reply.read() or b"{}")


def audio_part(path):
    raw = Path(path).read_bytes()
    return {"type": "input_audio",
            "input_audio": {"data": base64.b64encode(raw).decode(),
                            "format": "wav"}}


def request_body(audio, image=None):
    content = []
    if image:
        raw = Path(image).read_bytes()
        suffix = "jpeg" if raw[:2] == b"\xff\xd8" else "png"
        content.append({"type": "image_url", "image_url": {
            "url": f"data:image/{suffix};base64,"
                   + base64.b64encode(raw).decode()}})
    content.append({"type": "text", "text": "Answer in one short sentence."})
    content.append(audio_part(audio))
    return {"model": "qwen3-unified",
            "messages": [{"role": "user", "content": content}],
            "stream": True, "max_tokens": 32, "temperature": 0}


def stream_chat(port, body, raw_path, timeout=180.0):
    """POST one streaming request; save every raw line; return parsed frames."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout)
    raw_lines = []
    frames = []
    data_lines = []
    started = time.monotonic()
    status = None
    try:
        connection.request("POST", "/v1/chat/completions",
                           body=json.dumps(body),
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        status = response.status
        for line in response:
            text = line.decode("utf-8", "replace")
            raw_lines.append(text)
            bare = text.rstrip("\r\n")
            if bare.startswith("data:"):
                data_lines.append(bare[5:].lstrip())
            elif not bare and data_lines:
                data = "\n".join(data_lines)
                data_lines = []
                if data == "[DONE]":
                    break
                frames.append(json.loads(data))
    except Exception as exc:  # noqa: BLE001 - a broken stream is evidence
        raw_lines.append(f"PHASE6_STREAM_EXCEPTION {type(exc).__name__}: {exc}\n")
    finally:
        connection.close()
        Path(raw_path).write_text("".join(raw_lines))
    return status, frames, time.monotonic() - started


def summarize(status, frames, elapsed):
    events = [f["unified"] for f in frames if "unified" in f]
    terminals = [e for e in events if e["kind"] in ("complete", "error", "cancelled")]
    return {
        "http_status": status,
        "elapsed_s": round(elapsed, 3),
        "kinds": [e["kind"] for e in events],
        "seqs": [e["seq"] for e in events],
        "terminals": len(terminals),
        "terminal": terminals[0] if terminals else None,
        "asr_final": [e for e in events if e["kind"] == "asr_final"],
        "asr_partial": [e for e in events if e["kind"] == "asr_partial"],
        "vlm_delta": [e for e in events if e["kind"] == "vlm_delta"],
        "complete": [e for e in events if e["kind"] == "complete"],
        "error": [e for e in events if e["kind"] == "error"],
    }


def start_process(name, cmd, env, logs, evidence):
    log = CappedLog(evidence / f"{name}.log")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, env=env,
                               start_new_session=True)
    log.start(process)
    logs[name] = log
    return process


def stop_process(process, log):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(10)
    if log is not None:
        log.wait()


def backend_cmd(leg, port, trigger):
    if leg == "asr":
        model_path = MODELS / "asr"
        name = "qwen3-asr"
        # CUDA graph replay skips the Python logits path, so the decode-time
        # injection hook would never run. Eager decode is required here.
        extra = ["--disable-cuda-graph", "--log-level", "info"]
    else:
        model_path = MODELS / "omni"
        name = "qwen3-omni"
        extra = ["--mem-fraction-static=0.90", "--disable-cuda-graph",
                 "--disable-fast-image-processor", "--grammar-backend=none",
                 "--enable-multimodal", "--log-level", "info"]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "3"
    # The container's torch is cu130 against an R535 host driver; the shipped
    # CUDA 13.0 user-mode compat driver makes it work on this host.
    env["LD_LIBRARY_PATH"] = ("/usr/local/cuda-13.0/compat:"
                              "/usr/local/nvidia/lib64")
    env["PHASE6_NAN_TRIGGER"] = str(trigger)
    env["SGLANG_IS_IN_CI"] = "1"
    cmd = [PY, "-I", str(LAUNCHER), "--model-path", str(model_path),
           "--served-model-name", name, "--host", "127.0.0.1",
           "--port", str(port), "--enable-nan-detection"] + extra
    return cmd, env


def environment_report(evidence, commands, result):
    import platform
    lines = [
        f"host_gpu3={os.environ.get('PHASE6_GPU', '3')}",
        f"uname={platform.platform()}",
        f"python={sys.version.split()[0]}",
        f"cuda_visible_devices=3",
        f"ld_library_path=/usr/local/cuda-13.0/compat:/usr/local/nvidia/lib64",
        f"torch_cu130_compat_note=container torch 2.11.0+cu130 with CUDA 13.0 compat driver on R535",
        f"trigger={result['trigger']}",
        f"trigger_fired={result.get('trigger_fired_file_exists')}",
        f"leg={result['leg']}",
    ]
    for command in commands:
        lines.append(f"command={command}")
    Path(evidence / "environment.txt").write_text("\n".join(lines) + "\n")


def write_sums(evidence):
    entries = []
    for path in sorted(evidence.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            entries.append(f"{digest}  {path.relative_to(evidence)}")
    Path(evidence / "SHA256SUMS").write_text("\n".join(entries) + "\n")


def expectations(leg, result):
    fault = result["fault"]
    baseline = result["baseline"]
    recovery = result["recovery"]
    health = result["health_after_fault"]["body"]
    health_recovery = result["health_after_recovery"]["body"]
    health_zero = isinstance(health.get("in_flight"), int) and health["in_flight"] == 0
    recovery_zero = (isinstance(health_recovery.get("in_flight"), int)
                     and health_recovery["in_flight"] == 0)
    terminal = fault["terminal"] or {}
    checks = {
        "baseline_complete": bool(baseline["complete"]) and not baseline["error"],
        "fault_one_terminal": fault["terminals"] == 1,
        "fault_terminal_error": terminal.get("kind") == "error",
        "fault_stage_attribution": terminal.get("stage") == leg,
        "fault_no_complete": not fault["complete"],
        "fault_no_vlm_delta": not fault["vlm_delta"],
        "fault_no_asr_final": (not fault["asr_final"]) if leg == "asr"
                              else True,
        "fault_no_asr_partial": not fault["asr_partial"],
        "fault_seq_gap_free": fault["seqs"] == list(range(len(fault["seqs"]))),
        "health_in_flight_zero": health_zero,
        "detector_inject_line": bool(result["backend_detector"]["inject_line"]),
        "detector_nan_line": bool(result["backend_detector"]["nan_line"]),
        "detector_scheduler_exception":
            bool(result["backend_detector"]["scheduler_exception"]),
        "trigger_consumed": bool(result.get("trigger_fired_file_exists")),
        "recovery_complete": bool(recovery["complete"]) and not recovery["error"],
        "recovery_health_zero": recovery_zero,
    }
    if leg == "vlm":
        checks["fault_asr_final_present"] = bool(fault["asr_final"])
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--leg", choices=("asr", "vlm"), required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--backend-port", type=int, required=True)
    parser.add_argument("--other-backend", required=True)
    parser.add_argument("--unified-port", type=int, default=0)
    parser.add_argument("--launcher",
                        default=str(BACKEND_ROOT / "phase6_launch_copy.py"),
                        help="launcher that puts the patched copy on sys.path")
    parser.add_argument("--trigger-dir",
                        default="/root/phase6-compute-failure")
    args = parser.parse_args()
    if os.environ.get("UMMINFER_LIVE_OPT_IN") != "1":
        print("BLOCKED: set UMMINFER_LIVE_OPT_IN=1 to run the real GPU check")
        return 0
    global LAUNCHER
    LAUNCHER = Path(args.launcher)

    require_free_port(args.backend_port)
    if not args.unified_port:
        args.unified_port = free_port()
    else:
        require_free_port(args.unified_port)

    evidence = Path(args.evidence_dir)
    evidence.mkdir(parents=True, exist_ok=True)
    ipc = evidence / "ipc"
    ipc.mkdir(exist_ok=True)
    for stale in ipc.iterdir():
        stale.unlink()
    trigger = Path(args.trigger_dir) / f"trigger-{args.leg}"
    for leftover in (trigger, Path(str(trigger) + ".fired")):
        if leftover.exists():
            leftover.unlink()

    def endpoint(name):
        return f"ipc://{ipc}/{name}"

    commands = []
    logs = {}
    processes = {}
    result = {"leg": args.leg, "backend_port": args.backend_port,
              "unified_port": args.unified_port,
              "other_backend": args.other_backend, "trigger": str(trigger)}

    backend_cmdline, backend_env = backend_cmd(args.leg, args.backend_port,
                                               trigger)
    commands.append(" ".join(backend_cmdline))

    if args.leg == "asr":
        asr_backend = f"http://127.0.0.1:{args.backend_port}"
        omni_backend = args.other_backend
    else:
        asr_backend = args.other_backend
        omni_backend = f"http://127.0.0.1:{args.backend_port}"
    result["asr_backend"] = asr_backend
    result["omni_backend"] = omni_backend

    asr_cmd = [PY, "-I", str(Path(args.repo) / "python/umminfer_qwen_asr_stage.py"),
               "serve", "--name", "asr", "--recv", endpoint("asr"),
               "--coordinator", endpoint("done"), "--abort", endpoint("abort"),
               "--model", "qwen3-asr", "--backend-url", asr_backend,
               "--timeout", "300", "--concurrency", "2",
               "--perf-log", str(evidence / "asr-perf.jsonl")]
    omni_cmd = [PY, "-I", str(Path(args.repo) / "python/umminfer_qwen_omni_worker.py"),
                "--name", "vlm", "--recv", endpoint("vlm"),
                "--coordinator", endpoint("done"), "--abort", endpoint("abort"),
                "--upstream-base-url", omni_backend + "/v1",
                "--upstream-model", "qwen3-omni", "--request-timeout", "300",
                "--concurrency", "2", "--log-requests",
                "--perf-log", str(evidence / "omni-perf.jsonl")]
    server_cmd = [args.server, "serve", "--host", "127.0.0.1",
                  "--port", str(args.unified_port), "--model", "qwen3-unified",
                  "--entry", "asr", "--entry-endpoint", endpoint("asr"),
                  "--vlm-entry", "vlm", "--vlm-entry-endpoint", endpoint("vlm"),
                  "--completion", endpoint("done"), "--abort", endpoint("abort")]
    commands.extend(" ".join(c) for c in (asr_cmd, omni_cmd, server_cmd))

    try:
        # 1. Backend from the independent copy, on GPU 3.
        started = time.monotonic()
        backend = start_process("backend-fault-run1", backend_cmdline,
                                backend_env, logs, evidence)
        processes["backend"] = backend
        wait_http(f"http://127.0.0.1:{args.backend_port}/health_generate", 600,
                  proc=backend, log=logs["backend-fault-run1"])
        result["backend_ready_s"] = round(time.monotonic() - started, 3)

        for name, cmd in (("asr", asr_cmd), ("vlm", omni_cmd),
                          ("server", server_cmd)):
            processes[name] = start_process(name, cmd, os.environ.copy(),
                                            logs, evidence)
            time.sleep(0.3)
        wait_http(f"http://127.0.0.1:{args.unified_port}/ready", 180,
                  proc=processes["server"], log=logs["server"])
        wait_http(f"http://127.0.0.1:{args.unified_port}/v1/models", 30)

        body = request_body(args.audio,
                            args.image if args.leg == "vlm" else None)

        # 2. Baseline: the same public path completes before injection.
        status, frames, elapsed = stream_chat(
            args.unified_port, body, evidence / "baseline-raw-sse.txt")
        result["baseline"] = summarize(status, frames, elapsed)

        # 3. Arm the one-shot trigger and send the fault request.
        trigger.write_text("armed\n")
        status, frames, elapsed = stream_chat(
            args.unified_port, body, evidence / "fault-raw-sse.txt", timeout=240)
        result["fault"] = summarize(status, frames, elapsed)

        # 4. Public health after the failure.
        health_status, health = http_get_json(
            f"http://127.0.0.1:{args.unified_port}/health")
        result["health_after_fault"] = {"status": health_status, "body": health}

        # 5. Did the detector end the backend on its own?
        deadline = time.monotonic() + 120
        while backend.poll() is None and time.monotonic() < deadline:
            time.sleep(0.25)
        result["backend_exit_code"] = backend.poll()
        result["backend_exited_on_its_own"] = backend.poll() is not None
        stop_process(backend, logs["backend-fault-run1"])
        backend_log = logs["backend-fault-run1"].text()
        result["backend_detector"] = {
            "inject_line": [l for l in backend_log.splitlines()
                            if "PHASE6_NAN_INJECT" in l],
            "nan_line": [l for l in backend_log.splitlines()
                         if "NaN in the logits" in l],
            "scheduler_exception": [l for l in backend_log.splitlines()
                                    if "Scheduler hit an exception" in l],
        }
        result["trigger_fired_file_exists"] = Path(str(trigger) + ".fired").exists()

        # The detector or the harness must have released the listening port.
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                require_free_port(args.backend_port)
                break
            except OSError:
                time.sleep(0.25)

        # 6. Restart the same copy with no trigger file: clean compute path.
        backend2_cmdline, backend2_env = backend_cmd(
            args.leg, args.backend_port, trigger)
        backend2 = start_process("backend-clean-run2", backend2_cmdline,
                                 backend2_env, logs, evidence)
        processes["backend2"] = backend2
        wait_http(f"http://127.0.0.1:{args.backend_port}/health_generate", 600,
                  proc=backend2, log=logs["backend-clean-run2"])

        status, frames, elapsed = stream_chat(
            args.unified_port, body, evidence / "recovery-raw-sse.txt",
            timeout=240)
        result["recovery"] = summarize(status, frames, elapsed)
        health_status, health = http_get_json(
            f"http://127.0.0.1:{args.unified_port}/health")
        result["health_after_recovery"] = {"status": health_status,
                                           "body": health}

        result["commands"] = commands
        result["checks"] = expectations(args.leg, result)
        result["passed"] = all(result["checks"].values())
        environment_report(evidence, commands, result)
        Path(evidence / "result.json").write_text(
            json.dumps(result, indent=2) + "\n")
    finally:
        for name in ("server", "vlm", "asr", "backend2", "backend"):
            if name in processes:
                stop_process(processes[name], logs.get(name))
        for log in logs.values():
            log.wait()
        write_sums(evidence)

    print(json.dumps({"checks": result.get("checks"),
                      "passed": result.get("passed"),
                      "fault": result.get("fault", {}).get("kinds"),
                      "detector": result.get("backend_detector")}, indent=2))
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())

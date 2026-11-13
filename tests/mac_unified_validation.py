#!/usr/bin/env python3
"""Repeatable low-resource unified-engine validation for one Apple Mac.

Every request uses the repository's public path:

  HTTP/SSE -> umm-infer serve -> UnifiedEngine -> ASR stage -> text stage

with two local stages: the deterministic TinyCTC test model for the ASR leg and
a real MLX-LM text model for the text leg. Nothing here contacts another host.

TinyCTC is an initialized inference test model. It proves the ASR protocol and
the ASR -> text orchestration only. It is not a trained ASR model and its output
is not recognition-quality evidence. The MLX model proves only the text leg.

Opt-in: without --evidence-dir the script prints usage and exits 0. With
--evidence-dir it writes raw per-request records, a summary, bounded logs, the
environment, and a complete SHA256SUMS file.
"""

import argparse
import base64
import csv
import http.client
import importlib.metadata
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import struct
import subprocess
import sys
import threading
import time
import wave

LOG_CAP = 64 * 1024
MODEL_NAME = "qwen3-unified"


def _load_benchmark():
    """Reuse the benchmark's tested percentile, SSE, and process helpers."""
    path = Path(__file__).resolve().with_name("unified_benchmark.py")
    spec = importlib.util.spec_from_file_location("unified_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load_benchmark()


# ---- pure helpers (unit-tested) -------------------------------------------

def sine_wav_bytes(seconds=0.5, frequency=440.0, amplitude=0.2):
    """The deterministic audio the existing TinyCTC tests use.

    This is a tone, not speech. It only drives a real CTC forward pass.
    """
    rate = 16000
    count = int(rate * seconds)
    samples = [int(amplitude * 32767 * math.sin(2 * math.pi * frequency * i / rate))
               for i in range(count)]
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(struct.pack(f"<{count}h", *samples))
    return out.getvalue()


def short_wav_bytes():
    """One PCM16 sample: structurally valid WAV, too short for a TinyCTC frame."""
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00")
    return out.getvalue()


def unified_body(wav_bytes, max_tokens, session_id):
    # No text instruction part: UnifiedEngine turns one into a content-part
    # array, and the MLX-LM Qwen2.5 chat template accepts a string only. The
    # limitation is recorded in docs/mac-unified.md, not hidden here.
    content = [{
        "type": "input_audio",
        "input_audio": {
            "data": base64.b64encode(wav_bytes).decode("ascii"),
            "format": "wav",
        },
    }]
    return {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "session_id": session_id,
    }


class CappedLog:
    """Text log whose on-disk form never exceeds LOG_CAP bytes."""

    def __init__(self, cap=LOG_CAP):
        self.cap = cap
        self.buffer = bytearray()

    def write(self, text):
        bench.capped_append(self.buffer, text.encode("utf-8"), self.cap)

    def text(self):
        return bytes(self.buffer).decode("utf-8", errors="replace")


def seq_step_ok(previous, seq):
    """Unified seq starts at 0 and has no gap."""
    return seq == 0 if previous is None else seq == previous + 1


def evaluate(record, expected):
    """Check one request against the unified contract. Returns the errors."""
    errors = []

    def check(condition, message):
        if not condition:
            errors.append(message)

    check(record.get("http_status") == 200,
          f"HTTP {record.get('http_status')}")
    check(record.get("done"), "stream ended without [DONE]")
    check(record.get("terminal") == expected,
          f"terminal {record.get('terminal')!r}, expected {expected!r}")
    check(record.get("terminal_count") == 1,
          "terminal event is not exactly one")
    check(record.get("terminal_is_last"),
          "terminal is not the last unified event")
    check(record.get("seq_gap_free"), "unified seq is not gap-free from 0")
    check(record.get("session_echo_ok", True), "event session_id differs")
    check(record.get("vlm_delta_count") == record.get("content_delta_count"),
          "vlm_delta events and content deltas differ")
    if expected in ("complete", "cancelled"):
        check(record.get("asr_final_count") == 1,
              "ASR final is not exactly one")
        check(record.get("asr_final_before_delta"),
              "ASR final did not precede the first text delta")
    if expected == "complete":
        check(bool(record.get("text")), "empty answer")
        check(bool(record.get("transcript")), "empty transcript")
    if expected == "cancelled":
        check(not record.get("error"),
              f"cancel reported an error: {record.get('error')}")
        check(record.get("vlm_delta_count", 0) >= 1,
              "cancel happened before the text leg produced a delta")
    if expected == "error":
        check(record.get("asr_final_count") == 0,
              "an errored ASR leg still reported an ASR final")
        check(bool(record.get("error")), "error terminal carried no reason")
    record["check_errors"] = errors
    record["ok"] = not errors
    return errors


def summarize(records):
    """Aggregate finished samples. Failures are counted, never dropped."""
    ok = [record for record in records if record.get("ok")]
    return {
        "requests": len(records),
        "successes": len(ok),
        "failures": len(records) - len(ok),
        "total_p50_ms": _pct(ok, "total_ms", 50),
        "total_p95_ms": _pct(ok, "total_ms", 95),
        "first_event_p50_ms": _pct(ok, "first_event_ms", 50),
        "first_event_p95_ms": _pct(ok, "first_event_ms", 95),
        "first_delta_p50_ms": _pct(ok, "first_delta_ms", 50),
        "first_delta_p95_ms": _pct(ok, "first_delta_ms", 95),
        "asr_final_p50_ms": _pct(ok, "asr_final_ms", 50),
        "asr_final_p95_ms": _pct(ok, "asr_final_ms", 95),
        "answer_chars_mean": (round(statistics.fmean(
            [len(r["text"]) for r in ok]), 3) if ok else None),
        "transcripts": sorted({r.get("transcript") for r in ok
                               if r.get("transcript")}),
    }


def _pct(records, key, percentile):
    values = [r[key] for r in records if isinstance(r.get(key), (int, float))]
    value = bench.percentile(values, percentile)
    return None if value is None else round(value, 3)


def write_sha256sums(root):
    root = Path(root)
    files = sorted(
        (path for path in root.rglob("*")
         if path.is_file() and path.name != "SHA256SUMS"),
        key=lambda path: path.relative_to(root).as_posix())
    lines = [f"{bench.sha256_path(path)}  {path.relative_to(root).as_posix()}"
             for path in files]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(files)


def verify_sha256sums(root):
    """Recompute every listed file and report unlisted files too."""
    root = Path(root)
    lines = [line for line in
             (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
             if line.strip()]
    mismatches = []
    listed = set()
    for line in lines:
        digest, name = line.split("  ", 1)
        listed.add(name)
        path = root / name
        if not path.is_file():
            mismatches.append(f"missing: {name}")
        elif bench.sha256_path(path) != digest:
            mismatches.append(f"sha256 mismatch: {name}")
    present = {path.relative_to(root).as_posix()
               for path in root.rglob("*")
               if path.is_file() and path.name != "SHA256SUMS"}
    mismatches.extend(f"not listed: {name}" for name in sorted(present - listed))
    # A socket, FIFO, or device under the evidence root cannot be hashed and
    # must be reported, not silently skipped.
    non_regular = [path.relative_to(root).as_posix()
                   for path in root.rglob("*")
                   if not path.is_dir() and not path.is_file()]
    mismatches.extend(f"not a regular file: {name}"
                      for name in sorted(non_regular))
    return len(lines), mismatches


# ---- request client --------------------------------------------------------

def post_abort(port, request_id, timeout=10):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        connection.request("POST", "/abort",
                           json.dumps({"request_id": request_id}),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        body = json.loads(response.read())
        return response.status == 200 and body.get("success") is True
    except (OSError, ValueError, http.client.HTTPException):
        return False
    finally:
        connection.close()


def in_flight(port, timeout=5):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        connection.request("GET", "/health")
        response = connection.getresponse()
        return json.loads(response.read())["in_flight"]
    finally:
        connection.close()


def wait_zero_in_flight(port, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if in_flight(port) == 0:
                return True
        except (OSError, ValueError, http.client.HTTPException):
            pass
        time.sleep(0.02)
    return False


def stream_unified(port, body, timeout=120, cancel_after_first_delta=False):
    """Drive one public SSE unified request and record contract facts."""
    session_id = body.get("session_id")
    started = time.perf_counter()
    record = {
        "http_status": None, "request_id": None, "session_id": session_id,
        "done": False, "error": None, "text": "", "transcript": "",
        "finish_reason": None, "terminal": None, "terminal_seq": None,
        "terminal_count": 0,
        "asr_final_count": 0, "vlm_delta_count": 0, "content_delta_count": 0,
        "first_event_ms": None, "first_delta_ms": None, "asr_final_ms": None,
        "last_delta_ms": None, "total_ms": None, "last_seq": None,
        "seq_gap_free": True, "terminal_is_last": False,
        "asr_final_before_delta": False, "session_echo_ok": True,
        "abort_sent": False, "abort_success": None, "abort_at_ms": None,
        "kinds": {},
    }
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        connection.request("POST", "/v1/chat/completions",
                           json.dumps(body) + "\n",
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        record["http_status"] = response.status
        if response.status != 200:
            record["error"] = response.read(2000).decode(errors="replace")
            record["total_ms"] = (time.perf_counter() - started) * 1000.0
            return record
        parts = []
        previous_seq = None
        for data in bench.sse_events(
                line.decode("utf-8", errors="replace").rstrip("\r\n")
                for line in response):
            now = time.perf_counter()
            if data == "[DONE]":
                record["done"] = True
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                record["error"] = "invalid SSE JSON"
                break
            if isinstance(chunk.get("id"), str):
                record["request_id"] = chunk["id"]
            if isinstance(chunk.get("error"), dict):
                record["error"] = str(chunk["error"].get("message") or
                                      chunk["error"])
            for choice in chunk.get("choices") or []:
                delta = (choice.get("delta") or {}).get("content")
                if delta:
                    if record["first_delta_ms"] is None:
                        record["first_delta_ms"] = (now - started) * 1000.0
                    parts.append(delta)
                    record["content_delta_count"] += 1
                    record["last_delta_ms"] = (now - started) * 1000.0
                if choice.get("finish_reason") is not None:
                    record["finish_reason"] = choice["finish_reason"]
            event = chunk.get("unified")
            if not isinstance(event, dict):
                continue
            if record["first_event_ms"] is None:
                record["first_event_ms"] = (now - started) * 1000.0
            seq = event.get("seq")
            if isinstance(seq, int):
                if not seq_step_ok(previous_seq, seq):
                    record["seq_gap_free"] = False
                previous_seq = seq
                record["last_seq"] = seq
            kind = event.get("kind")
            if not isinstance(kind, str):
                record["error"] = "unified kind is not a string"
                continue
            record["kinds"][kind] = record["kinds"].get(kind, 0) + 1
            if session_id is not None and event.get("session_id") != session_id:
                record["session_echo_ok"] = False
            if kind == "asr_final":
                record["asr_final_count"] += 1
                record["transcript"] = event.get("text") or ""
                record["asr_final_ms"] = (now - started) * 1000.0
                if record["first_delta_ms"] is None:
                    record["asr_final_before_delta"] = True
            elif kind == "vlm_delta":
                record["vlm_delta_count"] += 1
            elif kind in ("complete", "cancelled", "error"):
                record["terminal_count"] += 1
                record["terminal"] = kind
                record["terminal_seq"] = seq
            if (cancel_after_first_delta and kind == "vlm_delta"
                    and not record["abort_sent"] and record["request_id"]):
                record["abort_sent"] = True
                record["abort_at_ms"] = (now - started) * 1000.0
                record["abort_success"] = post_abort(port,
                                                     record["request_id"])
        record["text"] = "".join(parts)
        record["total_ms"] = (time.perf_counter() - started) * 1000.0
        record["terminal_is_last"] = (
            record["terminal_seq"] is not None
            and record["terminal_seq"] == record["last_seq"])
        return record
    except Exception as error:  # a parser defect must become a failed sample
        record["error"] = f"{type(error).__name__}: {error}"
        record["total_ms"] = (time.perf_counter() - started) * 1000.0
        return record
    finally:
        connection.close()


# ---- process supervision ---------------------------------------------------

class RssSampler(threading.Thread):
    """Sample process RSS often enough to see a sub-second request window."""

    def __init__(self, pids, path, interval=0.2):
        super().__init__(daemon=True, name="rss-sampler")
        self.pids = pids
        self.path = path
        self.interval = interval
        self.stop_flag = threading.Event()

    def run(self):
        with open(self.path, "w", encoding="utf-8") as handle:
            while not self.stop_flag.is_set():
                now = time.time()
                for label, pid in self.pids.items():
                    try:
                        out = subprocess.run(
                            ["ps", "-o", "rss=", "-p", str(pid)],
                            capture_output=True, text=True, timeout=5)
                        value = out.stdout.strip()
                        if value:
                            handle.write(f"{now:.3f},{label},{pid},{value}\n")
                    except (OSError, subprocess.SubprocessError):
                        pass
                handle.flush()
                self.stop_flag.wait(self.interval)

    def stop(self):
        self.stop_flag.set()
        self.join(5)


def start_stage(python, script, name, paths, model, command=()):
    args = [python, "-I", str(script), *command, "--name", name,
            "--recv", paths["asr"] if name == "asr" else paths["vlm"],
            "--coordinator", paths["done"], "--abort", paths["abort"],
            "--model", str(model)]
    return bench.CappedProcess(args, f"{name}-stage",
                               cwd=str(script.parents[1]))


def start_server(binary, paths):
    args = [str(binary), "serve", "--host", "127.0.0.1", "--port", "0",
            "--model", MODEL_NAME,
            "--entry", "asr", "--entry-endpoint", paths["asr"],
            "--vlm-entry", "vlm", "--vlm-entry-endpoint", paths["vlm"],
            "--completion", paths["done"], "--abort", paths["abort"]]
    return bench.CappedProcess(args, "server")


def environment(repo, binary, model, tiny_model, audio_path, config):
    def sysctl(name):
        try:
            return subprocess.run(["sysctl", "-n", name], capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    def git(*args):
        try:
            return subprocess.run(["git", "-C", str(repo), *args],
                                  capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    def hashes(directory):
        out = {}
        for key in ("config.json", "model.safetensors", "tokenizer.json"):
            path = Path(directory) / key
            if path.is_file():
                out[key] = {"sha256": bench.sha256_path(path),
                            "bytes": path.stat().st_size}
        return out

    try:
        zig = subprocess.run(["zig", "version"], capture_output=True,
                             text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        zig = None
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": {"commit": git("rev-parse", "HEAD"),
                "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": bool(git("status", "--porcelain"))},
        "host": {
            "hw_model": sysctl("hw.model"),
            "cpu": sysctl("machdep.cpu.brand_string"),
            "ram_bytes": int(sysctl("hw.memsize") or 0),
            "os": " ".join(os.uname()),
            "python": sys.version.split()[0],
            "zig": zig,
        },
        "packages": {name: version(name) for name in
                     ("mlx", "mlx-lm", "tokenizers", "pyzmq", "msgpack",
                      "numpy")},
        "binary": {"path": str(binary), "sha256": bench.sha256_path(binary)},
        "mlx_text_model": {"path": str(model), "files": hashes(model)},
        "tinyctc_model": {"path": str(tiny_model), "files": hashes(tiny_model),
                          "seed": 20261111,
                          "note": "initialized inference test model, "
                                  "not trained ASR"},
        "audio": {"path": str(audio_path),
                  "kind": "deterministic 440 Hz tone",
                  "sha256": bench.sha256_path(audio_path)},
        "configuration": config,
        "evidence_scope": [
            "TinyCTC proves ASR protocol and orchestration only; it is not a "
            "trained ASR model and gives no recognition-quality evidence.",
            "The real MLX model proves the text leg only.",
            "Both stages run serially in one process each; concurrency 2 "
            "measures server-side admission and isolation, not throughput.",
            "No H20B, CUDA, or SGLang backend was contacted.",
        ],
    }


def run_scenario(name, port, body_factory, samples, concurrency, timeout):
    """Issue `samples` requests with `concurrency` in flight; keep every record."""
    records = []
    lock = threading.Lock()
    index = 0

    def worker():
        nonlocal index
        while True:
            with lock:
                if index >= samples:
                    return
                number = index
                index += 1
            try:
                record = stream_unified(port, body_factory(number), timeout)
            except Exception as error:  # a lost sample must be a counted failure
                record = {
                    "http_status": None, "done": False, "terminal": None,
                    "text": "", "transcript": "", "check_errors": [],
                    "first_event_ms": None, "first_delta_ms": None,
                    "asr_final_ms": None, "total_ms": None,
                    "error": f"{type(error).__name__}: {error}",
                }
            record["scenario"] = name
            record["concurrency"] = concurrency
            evaluate(record, "complete")
            with lock:
                records.append(record)

    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    started = time.time()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if len(records) != samples:
        raise RuntimeError(f"{name}: recorded {len(records)} of {samples} samples")
    return records, started, time.time()


def run_cancel(port, timeout, max_tokens):
    """Cancel during the text leg. A lost race is retried only when the
    completed attempt was itself a clean request."""
    attempts = []
    for _ in range(3):
        body = unified_body(sine_wav_bytes(), max_tokens,
                            session_id=f"mac-cancel-{len(attempts)}")
        record = stream_unified(port, body, timeout,
                                cancel_after_first_delta=True)
        record["scenario"] = "cancel"
        # Judge the attempt by what it actually is. A lost abort race is a
        # complete request, and any protocol defect it has must survive.
        if record.get("terminal") == "complete":
            evaluate(record, "complete")
        else:
            evaluate(record, "cancelled")
        attempts.append(record)
        if record.get("terminal") == "cancelled":
            break
        if record.get("terminal") != "complete" or not record["ok"]:
            break
    return attempts, wait_zero_in_flight(port)


def cancel_passed(attempts, zero):
    """Every attempted request was clean, and the last one was cancelled."""
    return (zero and bool(attempts)
            and attempts[-1].get("terminal") == "cancelled"
            and attempts[-1].get("ok") is True
            and all(attempt.get("ok") for attempt in attempts))


# ---- main ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="./zig-out/bin/umm-infer")
    parser.add_argument("--model", required=False,
                        help="local MLX-LM text model directory")
    parser.add_argument("--evidence-dir", required=False)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--concurrency-samples", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    if not args.evidence_dir:
        parser.print_help()
        print("\nThis harness needs a local MLX text model and real local stages.")
        return 0
    if not args.model:
        parser.error("--model is required with --evidence-dir")
    if args.samples < 1 or args.concurrency_samples < 2:
        parser.error("--samples must be positive and --concurrency-samples >= 2")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")

    repo = Path(__file__).resolve().parents[1]
    binary = Path(args.binary).resolve()
    model = Path(args.model).resolve()
    for path, label in ((binary, "binary"), (model, "MLX model")):
        if not path.exists():
            parser.error(f"{label} not found: {path}")
    if not (model / "config.json").is_file() or not (
            model / "model.safetensors").is_file():
        parser.error(
            f"MLX model needs config.json and model.safetensors: {model}")

    root = Path(args.evidence_dir).resolve()
    if root.exists() and any(root.iterdir()):
        parser.error(f"evidence directory must be empty: {root}")
    raw = root / "raw"
    logs = root / "logs"
    ipc_root = root / "ipc"
    for directory in (raw, logs, ipc_root):
        directory.mkdir(parents=True, exist_ok=True)

    run_log = CappedLog()
    tiny_model = root / "tinyctc"
    generated = subprocess.run(
        [args.python, "-I", str(repo / "python/umminfer_tiny_asr_stage.py"),
         "generate", "--output", str(tiny_model)],
        capture_output=True, text=True, timeout=120)
    if generated.returncode != 0:
        print(f"tiny model generation failed: {generated.stderr}",
              file=sys.stderr)
        return 1

    audio_path = root / "audio.wav"
    audio_path.write_bytes(sine_wav_bytes())
    paths = {name: f"ipc://{ipc_root}/{name}"
             for name in ("asr", "vlm", "done", "abort")}

    processes = {}
    rss = None
    summary = []
    checks = []
    failures = 0
    try:
        server = start_server(binary, paths)
        processes["server"] = server
        port = bench.server_port(server)
        processes["asr-stage"] = start_stage(
            args.python, repo / "python/umminfer_tiny_asr_stage.py",
            "asr", paths, tiny_model, command=("serve",))
        processes["vlm-stage"] = start_stage(
            args.python, repo / "python/umminfer_mlx_stage.py",
            "vlm", paths, model)
        bench.wait_ready(port, server, timeout=300)
        run_log.write(
            f"PASS startup port={port} "
            f"pids={{{', '.join(f'{k}={v.process.pid}' for k, v in processes.items())}}}\n")

        config = {
            "samples": args.samples,
            "concurrency_samples": args.concurrency_samples,
            "warmup": args.warmup,
            "max_new_tokens": args.max_new_tokens,
            "asr_leg": "inherits chat max_tokens; engine forces "
                       "stream=false and temperature=0",
            "stage_concurrency": 1,
            "concurrency": [1, 2],
            "log_cap_bytes": LOG_CAP,
            "sampling": "temperature=0, stream=true",
        }
        (root / "env.json").write_text(
            json.dumps(environment(repo, binary, model, tiny_model, audio_path,
                                   config), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")

        rss = RssSampler({label: process.process.pid
                          for label, process in processes.items()},
                         str(raw / "rss_samples.csv"))
        rss.start()

        for number in range(args.warmup):
            warm = stream_unified(
                port, unified_body(sine_wav_bytes(), args.max_new_tokens,
                                   session_id=f"mac-warmup-{number}"),
                args.timeout)
            evaluate(warm, "complete")
            if not warm["ok"]:
                failures += 1
            run_log.write(f"{'PASS' if warm['ok'] else 'FAIL'} "
                          f"warmup {number} "
                          f"terminal={warm.get('terminal')} "
                          f"check_errors={warm['check_errors']}\n")

        for concurrency, samples in ((1, args.samples),
                                     (2, args.concurrency_samples)):
            name = f"unified-c{concurrency}"
            records, started, finished = run_scenario(
                name, port,
                lambda number: unified_body(
                    sine_wav_bytes(), args.max_new_tokens,
                    session_id=f"{name}-{number}"),
                samples, concurrency, args.timeout)
            bench.write_jsonl(raw / f"{name}.jsonl", records)
            stats = summarize(records)
            stats.update({
                "scenario": name, "concurrency": concurrency,
                "wall_seconds": round(finished - started, 3),
                "rss_peak_kb": bench._rss_window(
                    str(raw / "rss_samples.csv"), started, finished),
                "window_start": started, "window_end": finished,
            })
            summary.append(stats)
            failures += stats["failures"]
            run_log.write(
                f"{name} requests={stats['requests']} "
                f"successes={stats['successes']} failures={stats['failures']} "
                f"total_p50_ms={stats['total_p50_ms']} "
                f"total_p95_ms={stats['total_p95_ms']} "
                f"first_event_p50_ms={stats['first_event_p50_ms']} "
                f"first_event_p95_ms={stats['first_event_p95_ms']} "
                f"transcripts={stats['transcripts']}\n")
            for record in records:
                if not record["ok"]:
                    run_log.write(f"FAIL {name} {record.get('request_id')}: "
                                  f"{record['check_errors']}\n")

        attempts, zero = run_cancel(port, args.timeout,
                                    max(args.max_new_tokens, 96))
        cancelled = attempts[-1]
        bench.write_jsonl(raw / "cancel.jsonl", attempts)
        cancel_ok = cancel_passed(attempts, zero)
        checks.append({"name": "cancel", "ok": cancel_ok,
                       "attempts": len(attempts),
                       "terminal": cancelled.get("terminal"),
                       "abort_success": cancelled.get("abort_success"),
                       "abort_at_ms": cancelled.get("abort_at_ms"),
                       "vlm_delta_count": cancelled.get("vlm_delta_count"),
                       "in_flight_zero": zero,
                       "check_errors": cancelled.get("check_errors", [])})
        failures += 0 if cancel_ok else 1
        run_log.write(f"{'PASS' if cancel_ok else 'FAIL'} cancel "
                      f"attempts={len(attempts)} "
                      f"terminal={cancelled.get('terminal')} "
                      f"abort_success={cancelled.get('abort_success')} "
                      f"vlm_deltas={cancelled.get('vlm_delta_count')} "
                      f"in_flight_zero={zero}\n")

        recovery = stream_unified(
            port, unified_body(sine_wav_bytes(), args.max_new_tokens,
                               session_id="mac-after-cancel"), args.timeout)
        evaluate(recovery, "complete")
        bench.write_jsonl(raw / "after-cancel.jsonl", [recovery])
        checks.append({"name": "after-cancel", "ok": recovery["ok"],
                       "terminal": recovery.get("terminal"),
                       "text_chars": len(recovery.get("text", "")),
                       "check_errors": recovery["check_errors"]})
        failures += 0 if recovery["ok"] else 1
        run_log.write(f"{'PASS' if recovery['ok'] else 'FAIL'} after_cancel "
                      f"terminal={recovery.get('terminal')}\n")

        error_record = stream_unified(
            port, unified_body(short_wav_bytes(), args.max_new_tokens,
                               session_id="mac-asr-error"), args.timeout)
        evaluate(error_record, "error")
        reason_ok = "shorter than one analysis frame" in (
            error_record["error"] or "")
        error_ok = error_record["ok"] and reason_ok
        bench.write_jsonl(raw / "asr-error.jsonl", [error_record])
        checks.append({"name": "asr-error", "ok": error_ok,
                       "terminal": error_record.get("terminal"),
                       "error": error_record.get("error"),
                       "check_errors": error_record["check_errors"] +
                       ([] if reason_ok else ["stage reason not propagated"])})
        failures += 0 if error_ok else 1
        run_log.write(f"{'PASS' if error_ok else 'FAIL'} asr_error "
                      f"terminal={error_record.get('terminal')} "
                      f"error={error_record.get('error')!r}\n")

        recovery = stream_unified(
            port, unified_body(sine_wav_bytes(), args.max_new_tokens,
                               session_id="mac-after-error"), args.timeout)
        evaluate(recovery, "complete")
        bench.write_jsonl(raw / "after-error.jsonl", [recovery])
        checks.append({"name": "after-error", "ok": recovery["ok"],
                       "terminal": recovery.get("terminal"),
                       "text_chars": len(recovery.get("text", "")),
                       "check_errors": recovery["check_errors"]})
        failures += 0 if recovery["ok"] else 1
        run_log.write(f"{'PASS' if recovery['ok'] else 'FAIL'} after_error "
                      f"terminal={recovery.get('terminal')}\n")
    finally:
        if rss is not None:
            rss.stop()
        for name in ("asr-stage", "vlm-stage", "server"):
            process = processes.pop(name, None)
            if process is None:
                continue
            try:
                code = process.stop(str(logs / f"{name}.log"))
                # The TinyCTC stage has no SIGTERM handler, so the harness's
                # own stop signal is reported as -SIGTERM. That is a clean stop.
                if code not in (0, -signal.SIGTERM):
                    failures += 1
                run_log.write(f"stop {name} exit={code}\n")
            except Exception as error:
                failures += 1
                run_log.write(f"FAIL stop {name}: {error}\n")
        # ZeroMQ leaves Unix sockets behind. They are not evidence and cannot
        # be hashed, so the manifest must not silently skip them.
        shutil.rmtree(ipc_root, ignore_errors=True)

    result = {
        "scenarios": summary,
        "checks": checks,
        "failures": failures,
        "evidence_scope": "protocol and orchestration only for TinyCTC; "
                          "real MLX model proves the text leg only",
        "not_evidence": [
            "TinyCTC recognition quality: the model is initialized, not trained.",
            "Mac unified performance under production load or on an accelerator.",
            "Image, video, realtime, and SGLang/Qwen backends on this run.",
        ],
    }
    (root / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    columns = ("scenario", "concurrency", "requests", "successes", "failures",
               "total_p50_ms", "total_p95_ms", "first_event_p50_ms",
               "first_event_p95_ms", "first_delta_p50_ms",
               "first_delta_p95_ms", "asr_final_p50_ms", "asr_final_p95_ms",
               "answer_chars_mean", "wall_seconds", "rss_peak_kb",
               "transcripts")
    with open(root / "summary.csv", "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in summary:
            flat = dict(row)
            flat["transcripts"] = json.dumps(flat.get("transcripts"),
                                             ensure_ascii=False)
            flat["rss_peak_kb"] = json.dumps(flat.get("rss_peak_kb"),
                                             sort_keys=True)
            writer.writerow(flat)

    run_log.write(f"failures={failures}\n")
    (root / "run.log").write_text(run_log.text(), encoding="utf-8")
    listed = write_sha256sums(root)
    _, mismatches = verify_sha256sums(root)
    for mismatch in mismatches:
        print(f"FAIL sha256: {mismatch}", file=sys.stderr)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"evidence={root} files_hashed={listed} "
          f"sha256_mismatches={len(mismatches)}")
    return 0 if failures == 0 and not mismatches else 1


if __name__ == "__main__":
    sys.exit(main())

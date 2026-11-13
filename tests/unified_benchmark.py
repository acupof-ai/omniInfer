#!/usr/bin/env python3
"""Repeatable real-model benchmark for the ummInfer unified engine.

Every measured request travels the repository's own path: HTTP -> umm-infer
serve -> Coordinator/UnifiedEngine -> real Qwen ASR and Qwen Omni workers ->
SGLang backends. No fake server, no synthesized audio, no handwritten answer.

The harness owns the worker and server lifecycle so that every log it produces
is capped at LOG_CAP bytes, and it records raw per-request JSON plus GPU and RSS
samples so every reported percentile can be recomputed.

Opt-in: without --evidence-dir the script prints usage and exits 0.
"""

import argparse
import base64
import csv
import hashlib
import http.client
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import threading
import time
import wave

LOG_CAP = 64 * 1024
SCENARIOS = (
    "asr-en", "asr-zh",
    "omni-text", "omni-image",
    "unified-en", "unified-zh",
)
UNIFIED_CONCURRENCY = (1, 2, 4)


# ---- small pure helpers (unit-tested) -------------------------------------

def percentile(values, p):
    """Nearest-rank percentile: no interpolation, no invented samples."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100.0 * len(ordered)))
    return ordered[rank - 1]


def coefficient_of_variation(values):
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    if mean == 0:
        return None
    return statistics.stdev(values) / mean


def capped_append(buffer, data, cap=LOG_CAP):
    """Keep only the last `cap` bytes; a log must never grow past the cap."""
    buffer.extend(data)
    overflow = len(buffer) - cap
    if overflow > 0:
        del buffer[:overflow]
    return buffer


def image_mime(data):
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    raise ValueError("input image is neither JPEG nor PNG")


def wav16k_mono_pcm16(path):
    """Return a canonical WAV for the one format the workers accept.

    The source recording is real; this only rewrites the container. A source
    with a LIST/metadata chunk is a valid WAV but the server's parser accepts
    exactly one fmt chunk and one data chunk, so the payload is rebuilt here.
    """
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if (channels, width, rate) != (1, 2, 16000):
        raise ValueError(
            f"{path}: need mono 16 kHz PCM16, got channels={channels} "
            f"width={width} rate={rate}")
    if not frames:
        raise ValueError(f"{path}: empty audio")
    out = io.BytesIO()
    with wave.open(out, "wb") as canonical:
        canonical.setnchannels(1)
        canonical.setsampwidth(2)
        canonical.setframerate(16000)
        canonical.writeframes(frames)
    return out.getvalue()


def sse_events(chunks):
    """Turn an iterable of decoded SSE lines into parsed data payloads.

    A comment line is ignored; a blank line ends one event. The caller supplies
    `[DONE]` as the sentinel payload.
    """
    lines = []
    for line in chunks:
        if not line:
            if not lines:
                continue
            payload = "\n".join(lines)
            lines = []
            yield payload
            continue
        if line.startswith("data:"):
            lines.append(line[5:].lstrip())
    if lines:
        yield "\n".join(lines)


def summarize(records, latency_key, first_delta_key="first_delta_ms",
              asr_final_key="asr_final_ms"):
    """Aggregate one scenario's successful samples. Missing fields stay None."""
    ok = [r for r in records if r.get("ok")]
    totals = [r[latency_key] for r in ok if r.get(latency_key) is not None]
    first = [r[first_delta_key] for r in ok
             if r.get(first_delta_key) is not None]
    asr_final = [r[asr_final_key] for r in ok
                 if r.get(asr_final_key) is not None]
    tokens = [r["output_tokens"] for r in ok
              if r.get("output_tokens") is not None]
    tps = [r["tokens_per_s"] for r in ok if r.get("tokens_per_s") is not None]
    return {
        "requests": len(records),
        "successes": len(ok),
        "failures": len(records) - len(ok),
        "total_p50_ms": percentile(totals, 50),
        "total_p95_ms": percentile(totals, 95),
        "total_cv": coefficient_of_variation(totals),
        "first_delta_p50_ms": percentile(first, 50),
        "first_delta_p95_ms": percentile(first, 95),
        "asr_final_p50_ms": percentile(asr_final, 50),
        "asr_final_p95_ms": percentile(asr_final, 95),
        "output_tokens_mean": (statistics.fmean(tokens) if tokens else None),
        "tokens_per_s_p50": percentile(tps, 50),
        "tokens_per_s_p95": percentile(tps, 95),
        "finish_reasons": _counts(r.get("finish_reason") for r in ok),
    }


def _counts(values):
    out = {}
    for value in values:
        key = "none" if value is None else str(value)
        out[key] = out.get(key, 0) + 1
    return out


# ---- process supervision ---------------------------------------------------

class CappedProcess:
    """A child process whose retained output is bounded by LOG_CAP bytes."""

    def __init__(self, args, name, cwd=None):
        self.name = name
        self.buffer = bytearray()
        self.lock = threading.Lock()
        self.process = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd)
        self.reader = threading.Thread(target=self._drain, daemon=True)
        self.reader.start()

    def _drain(self):
        while True:
            try:
                data = os.read(self.process.stdout.fileno(), 65536)
            except OSError:
                return
            if not data:
                return
            with self.lock:
                capped_append(self.buffer, data)

    def tail(self):
        with self.lock:
            return bytes(self.buffer).decode(errors="replace")

    def stop(self, log_path):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.reader.join(5)
        try:
            self.process.stdout.close()
        except OSError:
            pass
        text = self.tail()
        Path(log_path).write_text(text, encoding="utf-8")
        if os.path.getsize(log_path) > LOG_CAP:
            raise RuntimeError(f"{self.name} log exceeded its cap")
        return self.process.returncode


class Sampler(threading.Thread):
    """Timestamp lines as they arrive so metrics can be windowed locally."""

    def __init__(self, command, path, name):
        super().__init__(daemon=True, name=name)
        self.command = command
        self.path = path
        self.stop_flag = threading.Event()
        self.process = None

    def run(self):
        handle = open(self.path, "w", encoding="utf-8")
        try:
            self.process = subprocess.Popen(
                self.command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            for raw in self.process.stdout:
                line = raw.decode(errors="replace").strip()
                if line:
                    handle.write(f"{time.time():.3f},{line}\n")
                    handle.flush()
        finally:
            handle.close()

    def stop(self):
        self.stop_flag.set()
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()


class RssSampler(threading.Thread):
    def __init__(self, pids, path):
        super().__init__(daemon=True, name="rss-sampler")
        self.pids = pids
        self.path = path
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
                        rss = out.stdout.strip()
                        if rss:
                            handle.write(f"{now:.3f},{label},{pid},{rss}\n")
                    except (OSError, subprocess.SubprocessError):
                        pass
                handle.flush()
                self.stop_flag.wait(1.0)

    def stop(self):
        self.stop_flag.set()
        self.join(5)


# ---- request execution -----------------------------------------------------

def _gpu_window(path, start, end):
    peak = {}
    util = {}
    if not os.path.exists(path):
        return peak, util
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            # The sampler prefixes the local receive time to the three
            # nvidia-smi fields: at,index,memory.used,utilization.gpu
            parts = line.rstrip("\n").split(",")
            if len(parts) != 4:
                continue
            try:
                at = float(parts[0])
                index, used, usage = (int(part) for part in parts[1:])
            except ValueError:
                continue
            if not start <= at <= end:
                continue
            peak[index] = max(peak.get(index, 0), used)
            util.setdefault(index, []).append(usage)
    mean_util = {index: statistics.fmean(values)
                 for index, values in util.items() if values}
    return peak, mean_util


def _rss_window(path, start, end):
    peak = {}
    if not os.path.exists(path):
        return peak
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split(",")
            if len(parts) != 4:
                continue
            try:
                at = float(parts[0])
                rss = int(parts[3])
            except ValueError:
                continue
            if not start <= at <= end:
                continue
            label = parts[1]
            peak[label] = max(peak.get(label, 0), rss)
    return peak


class Client:
    def __init__(self, port, model, timeout=300):
        self.port = port
        self.model = model
        self.timeout = timeout

    def _connect(self):
        return http.client.HTTPConnection(
            "127.0.0.1", self.port, timeout=self.timeout)

    def asr(self, wav_bytes, max_new_tokens):
        boundary = "----umminferbench" + hashlib.sha256(
            wav_bytes[:64]).hexdigest()[:16]
        fields = {
            "model": self.model,
            "max_new_tokens": str(max_new_tokens),
            "temperature": "0",
            "include_generation_metadata": "true",
        }
        body = io.BytesIO()
        for name, value in fields.items():
            body.write(f"--{boundary}\r\n".encode())
            body.write(
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
            body.write(value.encode() + b"\r\n")
        body.write(f"--{boundary}\r\n".encode())
        body.write(b'Content-Disposition: form-data; name="file"; '
                   b'filename="audio.wav"\r\n')
        body.write(b"Content-Type: audio/wav\r\n\r\n")
        body.write(wav_bytes + b"\r\n")
        body.write(f"--{boundary}--\r\n".encode())
        payload = body.getvalue()

        started = time.perf_counter()
        connection = self._connect()
        try:
            connection.request(
                "POST", "/v1/audio/transcriptions", payload,
                {"Content-Type": f"multipart/form-data; boundary={boundary}"})
            response = connection.getresponse()
            raw = response.read()
            finished = time.perf_counter()
            status = response.status
        finally:
            connection.close()
        record = {
            "ok": False, "status": status,
            "total_ms": (finished - started) * 1000.0,
            "first_delta_ms": None, "asr_final_ms": None,
            "text": "", "asr_text": "", "output_tokens": None,
            "tokens_per_s": None, "finish_reason": None,
        }
        if status != 200:
            record["error"] = raw[:2000].decode(errors="replace")
            return record
        parsed = json.loads(raw)
        text = parsed.get("text", "")
        metadata = parsed.get("generation_metadata") or {}
        tokens = metadata.get("generated_token_count")
        record.update({
            "ok": bool(text),
            "text": text,
            "output_tokens": tokens,
            "finish_reason": metadata.get("finish_reason"),
        })
        if text and isinstance(tokens, int) and tokens > 0:
            record["tokens_per_s"] = tokens / (record["total_ms"] / 1000.0)
        if not text:
            record["error"] = "empty transcript"
        return record

    def chat_stream(self, payload):
        started = time.perf_counter()
        sent = started
        connection = self._connect()
        record = {
            "ok": False, "status": None, "total_ms": None,
            "first_delta_ms": None, "asr_final_ms": None,
            "text": "", "asr_text": "", "output_tokens": None,
            "tokens_per_s": None, "finish_reason": None, "events": 0,
        }
        try:
            body = json.dumps(payload).encode()
            connection.request(
                "POST", "/v1/chat/completions", body,
                {"Content-Type": "application/json"})
            response = connection.getresponse()
            record["status"] = response.status
            if response.status != 200:
                record["error"] = response.read()[:2000].decode(errors="replace")
                return record
            first_delta = None
            asr_final = None
            last_delta = None
            text_parts = []
            asr_text = None
            finish_reason = None
            done = False
            for data in sse_events(
                    line.decode("utf-8").rstrip("\r\n")
                    for line in response):
                if data == "[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if "error" in event:
                    record["error"] = json.dumps(event["error"])[:2000]
                    return record
                record["events"] += 1
                unified = event.get("unified")
                if unified and unified.get("kind") == "asr_final":
                    asr_final = time.perf_counter()
                    asr_text = unified.get("text")
                for choice in event.get("choices", []):
                    delta = choice.get("delta") or {}
                    content = delta.get("content")
                    if content:
                        if first_delta is None:
                            first_delta = time.perf_counter()
                        last_delta = time.perf_counter()
                        text_parts.append(content)
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
            finished = time.perf_counter()
        finally:
            connection.close()
        text = "".join(text_parts)
        record.update({
            "total_ms": (finished - sent) * 1000.0,
            "first_delta_ms": (None if first_delta is None
                               else (first_delta - sent) * 1000.0),
            "asr_final_ms": (None if asr_final is None
                             else (asr_final - sent) * 1000.0),
            "text": text,
            "asr_text": asr_text or "",
            "finish_reason": finish_reason,
            "ok": bool(done and text),
        })
        if not done:
            record["error"] = "stream ended before [DONE]"
        elif not text:
            record["error"] = "empty answer"
        if first_delta is not None and last_delta is not None:
            record["stream_ms"] = (last_delta - first_delta) * 1000.0
        return record


def count_tokens(tokenizer, text):
    if tokenizer is None or not text:
        return None
    return len(tokenizer.encode(text).ids)


def fill_tokens(record, tokenizer):
    """Count delivered output tokens unless the stage reported an exact count."""
    if record.get("output_tokens") is not None:
        return record
    tokens = count_tokens(tokenizer, record.get("text", ""))
    record["output_tokens"] = tokens
    if tokens is not None and record.get("stream_ms"):
        record["tokens_per_s"] = tokens / (record["stream_ms"] / 1000.0)
    elif tokens is not None and record.get("total_ms"):
        record["tokens_per_s"] = tokens / (record["total_ms"] / 1000.0)
    return record


def run_concurrent(client, make_request, count, concurrency, tokenizer):
    """Issue `count` measured requests with `concurrency` in flight."""
    records = []
    lock = threading.Lock()
    index = 0
    in_flight = min(count, concurrency)
    barrier = threading.Barrier(in_flight)

    def worker():
        nonlocal index
        while True:
            with lock:
                if index >= count:
                    return
                index += 1
            try:
                barrier.wait(timeout=60)
            except threading.BrokenBarrierError:
                pass
            try:
                record = make_request(client)
            except Exception as exc:  # a transport failure is a failed sample
                record = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            fill_tokens(record, tokenizer)
            with lock:
                records.append(record)

    threads = [threading.Thread(target=worker) for _ in range(in_flight)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return records


def run_scenario(name, client, make_request, samples, warmup, concurrency,
                 tokenizer, attempts_cap):
    for _ in range(warmup):
        try:
            fill_tokens(make_request(client), tokenizer)
        except Exception:
            pass  # a failed warmup is not a sample; the measured loop records it
    started = time.time()
    records = []
    remaining = samples
    attempts = 0
    while remaining > 0 and attempts < attempts_cap:
        if attempts >= 3 and not any(r.get("ok") for r in records):
            break  # the path is broken; do not spend the whole attempt cap
        batch = min(remaining, concurrency)
        got = run_concurrent(client, make_request, batch, concurrency, tokenizer)
        records.extend(got)
        attempts += batch
        successes = sum(1 for record in got if record.get("ok"))
        remaining -= successes
        if successes == 0 and attempts >= 3:
            # Three transport or model failures in a row: stop instead of
            # burning the attempt budget on a broken path.
            break
    finished = time.time()
    return records, started, finished


# ---- bodies ----------------------------------------------------------------

def chat_body(model, content, max_tokens):
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
    }


def unified_body(model, wav_bytes, max_tokens):
    return chat_body(model, [{
        "type": "input_audio",
        "input_audio": {
            "data": base64.b64encode(wav_bytes).decode("ascii"),
            "format": "wav",
        },
    }], max_tokens)


# ---- lifecycle -------------------------------------------------------------

def wait_ready(port, server, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.process.poll() is not None:
            raise RuntimeError(f"server exited: {server.tail()[-2000:]}")
        try:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            connection.request("GET", "/ready")
            response = connection.getresponse()
            body = response.read()
            connection.close()
            if response.status == 200:
                return
        except OSError:
            pass
        time.sleep(0.1)
    raise RuntimeError(f"server never became ready: {server.tail()[-2000:]}")


def server_port(server, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.process.poll() is not None:
            raise RuntimeError(f"server exited: {server.tail()[-2000:]}")
        match = re.search(r"serving on [^\s:]+:(\d+)", server.tail())
        if match:
            return int(match.group(1))
        time.sleep(0.05)
    raise RuntimeError("server did not report its port")


def ipc_paths(root):
    return {name: f"ipc://{root}/{name}"
            for name in ("asr", "vlm", "done", "abort")}


def start_workers(python, root, asr_backend, omni_backend,
                  need_asr=True, need_omni=True, asr_concurrency=1,
                  omni_concurrency=1, perf_dir=None):
    paths = ipc_paths(root)
    workers = {}
    if need_asr:
        args = [
            python, "-I", "python/umminfer_qwen_asr_stage.py", "serve",
            "--name", "asr", "--recv", paths["asr"],
            "--coordinator", paths["done"], "--abort", paths["abort"],
            "--model", "qwen3-asr", "--backend-url", asr_backend,
            "--timeout", "300",
        ]
        if asr_concurrency != 1:
            args += ["--concurrency", str(asr_concurrency)]
        if perf_dir:
            args += ["--perf-log", os.path.join(perf_dir,
                                                "asr-worker-perf.jsonl")]
        workers["asr-worker"] = CappedProcess(args, "asr-worker")
    if not need_omni:
        return workers, paths
    args = [
        python, "-I", "python/umminfer_qwen_omni_worker.py",
        "--name", "vlm", "--recv", paths["vlm"],
        "--coordinator", paths["done"], "--abort", paths["abort"],
        "--upstream-base-url", omni_backend + "/v1",
        "--upstream-model", "qwen3-omni", "--request-timeout", "300",
    ]
    if omni_concurrency != 1:
        args += ["--concurrency", str(omni_concurrency)]
    if perf_dir:
        args += ["--perf-log", os.path.join(perf_dir,
                                            "omni-worker-perf.jsonl")]
    workers["omni-worker"] = CappedProcess(args, "omni-worker")
    return workers, paths


def start_server(binary, name, port, paths, unified):
    args = [binary, "serve", "--host", "127.0.0.1", "--port", str(port),
            "--model", "qwen3-unified" if unified else "qwen3-omni"]
    if unified:
        args += ["--entry", "asr", "--entry-endpoint", paths["asr"],
                 "--vlm-entry", "vlm", "--vlm-entry-endpoint", paths["vlm"]]
    else:
        args += ["--entry", "vlm", "--entry-endpoint", paths["vlm"]]
    args += ["--completion", paths["done"], "--abort", paths["abort"]]
    return CappedProcess(args, name)


def sha256_path(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tokenizer(path):
    if not path:
        return None
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(path))


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="./zig-out/bin/umm-infer")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--asr-backend", default="http://127.0.0.1:43100")
    parser.add_argument("--omni-backend", default="http://127.0.0.1:43101")
    parser.add_argument("--audio-en", required=False)
    parser.add_argument("--audio-zh", required=False)
    parser.add_argument("--image", required=False)
    parser.add_argument("--tokenizer", required=False)
    parser.add_argument("--evidence-dir", required=False)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--asr-max-new-tokens", type=int, default=256)
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--gpu-command", default="")
    # Worker-side upstream concurrency. 1 is the serial loop the baseline used.
    parser.add_argument("--asr-concurrency", type=int, default=1)
    parser.add_argument("--omni-concurrency", type=int, default=1)
    # Keep the workers' JSON-lines queue/perf records next to the raw samples.
    parser.add_argument("--worker-perf-log", action="store_true")
    args = parser.parse_args()
    if not args.evidence_dir:
        parser.print_help()
        print("\nThis harness needs real inputs and a running H20B backend.")
        return 0
    if args.max_new_tokens < 256 or args.asr_max_new_tokens < 256:
        parser.error("max_new_tokens must be at least 256")
    if args.samples < 1:
        parser.error("--samples must be positive")
    for name in ("asr_concurrency", "omni_concurrency"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    root = Path(args.evidence_dir)
    raw = root / "raw"
    logs = root / "logs"
    raw.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    ipc_root = root / "ipc"
    ipc_root.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.tokenizer)
    wanted = [name for name in args.scenarios.split(",") if name]
    unknown = [name for name in wanted if name not in SCENARIOS]
    if unknown:
        parser.error("unknown scenario: " + ", ".join(unknown))

    wav_en = wav16k_mono_pcm16(args.audio_en) if args.audio_en else None
    wav_zh = wav16k_mono_pcm16(args.audio_zh) if args.audio_zh else None
    image_bytes = Path(args.image).read_bytes() if args.image else None
    if image_bytes is not None:
        image_url = ("data:" + image_mime(image_bytes) + ";base64,"
                     + base64.b64encode(image_bytes).decode("ascii"))

    env = {
        "binary": os.path.abspath(args.binary),
        "binary_sha256": sha256_path(args.binary),
        "python": args.python,
        "asr_backend": args.asr_backend,
        "omni_backend": args.omni_backend,
        "inputs": {},
        "max_new_tokens_vlm": args.max_new_tokens,
        "max_new_tokens_asr": args.asr_max_new_tokens,
        "samples_per_scenario": args.samples,
        "warmup_per_scenario": args.warmup,
        "unified_concurrency": list(UNIFIED_CONCURRENCY),
        "asr_worker_concurrency": args.asr_concurrency,
        "omni_worker_concurrency": args.omni_concurrency,
        "worker_perf_log": bool(args.worker_perf_log),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    for label, path in (("audio_en", args.audio_en),
                        ("audio_zh", args.audio_zh), ("image", args.image),
                        ("tokenizer", args.tokenizer)):
        if path:
            env["inputs"][label] = {
                "path": os.path.abspath(path),
                "sha256": sha256_path(path),
                "bytes": os.path.getsize(path),
            }
    (root / "env.json").write_text(
        json.dumps(env, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = []
    sampler = None
    if args.gpu_command:
        sampler = Sampler(shlex.split(args.gpu_command),
                          str(raw / "gpu_samples.csv"), "gpu-sampler")
        sampler.start()
    rss = None

    processes = {}
    try:
        # Phase 1: Omni text and image through the single-entry chat path.
        needs_omni = any(name in wanted for name in ("omni-text", "omni-image"))
        if needs_omni:
            omni_root = str(ipc_root / "omni")
            os.makedirs(omni_root, exist_ok=True)
            workers, paths = start_workers(
                args.python, omni_root, args.asr_backend, args.omni_backend,
                need_asr=False,
                omni_concurrency=args.omni_concurrency,
                perf_dir=str(raw) if args.worker_perf_log else None)
            processes.update(workers)
            server = start_server(args.binary, "omni-server", 0, paths, False)
            processes["omni-server"] = server
            port = server_port(server)
            wait_ready(port, server)
            client = Client(port, "qwen3-omni", timeout=300)
            pids = {"server": server.process.pid,
                    "worker": workers["omni-worker"].process.pid}
            rss = RssSampler(pids, str(raw / "rss_samples.csv"))
            rss.start()
            if "omni-text" in wanted:
                body = chat_body(
                    "qwen3-omni",
                    "Describe the trade-offs between latency and throughput "
                    "in an inference server.", args.max_new_tokens)
                records, a, b = run_scenario(
                    "omni-text", client, lambda c: c.chat_stream(body),
                    args.samples, args.warmup, 1, tokenizer,
                    args.samples * 2 + 10)
                _finish_scenario(summary, "omni-text", 1, records, a, b, raw,
                                 str(raw / "rss_samples.csv"))
            if "omni-image" in wanted:
                body = chat_body("qwen3-omni", [
                    {"type": "text",
                     "text": "Describe the person, vehicle, and action in "
                             "this image."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ], args.max_new_tokens)
                records, a, b = run_scenario(
                    "omni-image", client, lambda c: c.chat_stream(body),
                    args.samples, args.warmup, 1, tokenizer,
                    args.samples * 2 + 10)
                _finish_scenario(summary, "omni-image", 1, records, a, b, raw,
                                 str(raw / "rss_samples.csv"))
            rss.stop()
            rss = None
            for name in ("omni-server", "omni-worker"):
                processes.pop(name).stop(str(logs / f"{name}.log"))

        # Phase 2: unified server carries ASR and ASR -> VLM.
        needs_unified = any(name in wanted for name in
                            ("asr-en", "asr-zh", "unified-en", "unified-zh"))
        if needs_unified:
            unified_root = str(ipc_root / "unified")
            os.makedirs(unified_root, exist_ok=True)
            workers, paths = start_workers(
                args.python, unified_root, args.asr_backend, args.omni_backend,
                asr_concurrency=args.asr_concurrency,
                omni_concurrency=args.omni_concurrency,
                perf_dir=str(raw) if args.worker_perf_log else None)
            processes.update(workers)
            server = start_server(args.binary, "unified-server", 0, paths, True)
            processes["unified-server"] = server
            port = server_port(server)
            wait_ready(port, server)
            client = Client(port, "qwen3-unified", timeout=300)
            pids = {"server": server.process.pid,
                    "asr_worker": workers["asr-worker"].process.pid,
                    "omni_worker": workers["omni-worker"].process.pid}
            rss = RssSampler(pids, str(raw / "rss_samples-unified.csv"))
            rss.start()
            for language, wav_bytes in (("en", wav_en), ("zh", wav_zh)):
                if wav_bytes is None:
                    continue
                name = f"asr-{language}"
                if name in wanted:
                    records, a, b = run_scenario(
                        name, client,
                        lambda c, w=wav_bytes: c.asr(
                            w, args.asr_max_new_tokens),
                        args.samples, args.warmup, 1, tokenizer,
                        args.samples * 2 + 10)
                    _finish_scenario(summary, name, 1, records, a, b, raw,
                                     str(raw / "rss_samples-unified.csv"))
                name = f"unified-{language}"
                if name in wanted:
                    body = unified_body(
                        "qwen3-unified", wav_bytes, args.max_new_tokens)
                    for concurrency in UNIFIED_CONCURRENCY:
                        records, a, b = run_scenario(
                            name, client,
                            lambda c, bod=body: c.chat_stream(bod),
                            args.samples, args.warmup, concurrency, tokenizer,
                            args.samples * 2 + 10)
                        _finish_scenario(
                            summary, name, concurrency, records, a, b, raw,
                            str(raw / "rss_samples-unified.csv"))
            rss.stop()
            rss = None
            for name in ("unified-server", "asr-worker", "omni-worker"):
                processes.pop(name).stop(str(logs / f"{name}.log"))
    finally:
        if rss is not None:
            rss.stop()
        for name, process in list(processes.items()):
            try:
                process.stop(str(logs / f"{name}.log"))
            except Exception:
                pass
        if sampler is not None:
            sampler.stop()
            sampler.join(10)

    _write_summary(root, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _finish_scenario(summary, name, concurrency, records, started, finished,
                     raw, rss_path):
    key = name if concurrency == 1 else f"{name}-c{concurrency}"
    write_jsonl(raw / f"{key}.jsonl", records)
    stats = summarize(records, "total_ms")
    if name.startswith("unified") or name.startswith("omni"):
        duration = finished - started
        stats["wall_seconds"] = duration
        stats["request_throughput_per_s"] = (
            stats["successes"] / duration if duration > 0 else None)
    gpu_peak, gpu_util = _gpu_window(
        str(raw / "gpu_samples.csv"), started, finished)
    stats["gpu_peak_mib"] = {str(k): v for k, v in sorted(gpu_peak.items())}
    stats["gpu_mean_util"] = {str(k): round(v, 3)
                              for k, v in sorted(gpu_util.items())}
    stats["rss_peak_kb"] = _rss_window(rss_path, started, finished)
    stats["scenario"] = name
    stats["concurrency"] = concurrency
    stats["window_start"] = started
    stats["window_end"] = finished
    summary.append(stats)


def _write_summary(root, summary):
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    columns = [
        "scenario", "concurrency", "requests", "successes", "failures",
        "total_p50_ms", "total_p95_ms", "total_cv", "first_delta_p50_ms",
        "first_delta_p95_ms", "asr_final_p50_ms", "asr_final_p95_ms",
        "output_tokens_mean", "tokens_per_s_p50", "tokens_per_s_p95",
        "finish_reasons", "gpu_peak_mib", "gpu_mean_util", "rss_peak_kb",
    ]
    with open(root / "summary.csv", "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in summary:
            flat = dict(row)
            for key in ("finish_reasons", "gpu_peak_mib", "gpu_mean_util",
                        "rss_peak_kb"):
                flat[key] = json.dumps(flat.get(key), separators=(",", ":"))
            writer.writerow(flat)


if __name__ == "__main__":
    sys.exit(main())

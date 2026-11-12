#!/usr/bin/env python3
"""Measure MLX inference cost at three levels with one fixed configuration.

Levels:
  direct  MLX-LM in this process. No C++, no ZeroMQ, no HTTP.
  stage   The real Python stage over ZeroMQ, driven by this process as the
          coordinator. Adds Python stage + ZeroMQ to `direct`.
  e2e     The real `umm-infer serve` plus the real stage, driven over HTTP and
          through the real CLI. Adds Coordinator + HTTP + SSE + client to
          `stage`.

Every level reports the same metrics, so each layer's cost is the difference
between two medians. Run this only in an exclusive window: another process that
uses the GPU or CPU changes these numbers.

The configuration is fixed in this file (model snapshot, prompt, temperature 0,
max_tokens, sample count) so a baseline run and a candidate run are comparable.
"""

import argparse
import fcntl
import hashlib
import http.client
from importlib import metadata as importlib_metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time

import msgpack
import zmq


MODEL_PATH = Path(
    "/Users/bytedance/.cache/huggingface/hub/"
    "models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/"
    "snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3")
SERVED_MODEL = "Qwen2.5-0.5B-Instruct-4bit"
PROMPT = "Give one concise inference tip."
TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 64
MAX_TOKENS_CAP = 512
MAX_SAMPLES = 200
MAX_WARMUP = 50
LOG_CAP_BYTES = 65536
OUTPUT_CAP_BYTES = 1 << 20
STAGE_READY_TIMEOUT_S = 240
REQUEST_TIMEOUT_S = 180
# Only these row fields are numeric. A row also carries the output hash, which
# must never enter a median.
NUMERIC_METRICS = ("ttft_ms", "total_ms", "generation_tokens",
                   "decode_tokens_per_s", "output_tokens_per_s", "prompt_tokens",
                   "prompt_tokens_per_s", "peak_memory_gb", "stage_ttft_ms",
                   "stage_total_ms")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE = ROOT / "python" / "umminfer_mlx_stage.py"
DEFAULT_BINARY = ROOT / "zig-out" / "bin" / "umm-infer"
# One fixed path: a second harness, or a second copy of the same stage, fails
# to take the lock and exits before it can measure anything.
LOCK_PATH = Path("/tmp/umminfer-mlx-perf.lock")
# The derived baseline stage is always patrolled, whichever stage this run
# measures. A candidate run must see an externally started baseline process.
BASELINE_STAGE_PATH = Path("/tmp/umminfer-stage-baseline.py")
# Files that determine the model input. Their hashes are pinned before and
# after the run and compared across the baseline and candidate reports.
MODEL_INPUT_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json",
                     "chat_template.jinja", "special_tokens_map.json",
                     "generation_config.json")
# A formal path needs this many warm, non-cold raw samples.
MIN_WARM_SAMPLES = 20


# --------------------------------------------------------------------------
# helpers


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def require_model_layout(model):
    """The pinned model is one `model.safetensors` file.

    A sharded snapshot would be loaded differently, so reject it before the run
    rather than fail on the first file read after measurement.
    """
    require((Path(model) / "model.safetensors").is_file(),
            f"{model} has no model.safetensors: this harness supports only the "
            "current single-file model layout")


def model_input_hashes(model):
    """Hashes of the files that determine the model input.

    config.json, tokenizer.json, and tokenizer_config.json are required: a
    tokenizer or chat template can change the prompt without changing the
    weights, so a missing file is a failure, not an empty map entry.
    """
    model = Path(model)
    hashes = {}
    for name in MODEL_INPUT_FILES:
        path = model / name
        if path.is_file():
            hashes[name] = digest(path)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        require(name in hashes, f"the model has no {name}")
    # The chat template can live in its own file or inside the tokenizer
    # config. Pin whichever the model uses: a template change alters the
    # prompt with no weight or tokenizer file change.
    if "chat_template.jinja" not in hashes:
        path = model / "tokenizer_config.json"
        try:
            config = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"cannot read the chat template from {path}: {exc}")
        if not isinstance(config, dict):
            raise RuntimeError(
                f"{path} is not a JSON object; cannot read the chat template")
        template = config.get("chat_template")
        require(template, "the model has no chat template")
        hashes["chat_template"] = hashlib.sha256(json.dumps(
            template, sort_keys=True).encode("utf-8")).hexdigest()
    return hashes


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def package_version(name):
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def child_env():
    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                "TOKENIZERS_PARALLELISM": "false"})
    return env


def process_table():
    """Every process as (pid, ppid, command). A failure is fatal.

    A measurement window that cannot inspect the process table cannot prove it
    is exclusive, so this raises instead of returning a partial list.
    """
    try:
        result = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,command="],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"ps failed, the window cannot be checked: {exc}")
    require(result.returncode == 0,
            f"ps exited {result.returncode}; the window cannot be checked")
    table = []
    for line in result.stdout.decode(errors="replace").splitlines():
        fields = line.split(None, 2)
        if len(fields) != 3:
            continue
        try:
            table.append((int(fields[0]), int(fields[1]), fields[2]))
        except ValueError:
            continue
    require(table, "ps returned no processes; the window cannot be checked")
    return table


def related_pids(table):
    """This process, its ancestors, and its descendants.

    The launcher of this harness carries the measurement command in its argv.
    Excluding the process tree keeps the check out of its own argv while still
    seeing every process that this run does not own.
    """
    parent = {pid: ppid for pid, ppid, _ in table}

    def is_descendant(pid):
        seen = 0
        while pid > 1 and seen < 256:
            if pid == os.getpid():
                return True
            pid = parent.get(pid, 1)
            seen += 1
        return False

    related = set()
    pid = os.getpid()
    seen = 0
    while pid > 1 and seen < 256:
        related.add(pid)
        pid = parent.get(pid, 1)
        seen += 1
    for candidate in parent:
        if is_descendant(candidate):
            related.add(candidate)
    return related


def foreign_processes(needles):
    table = process_table()
    related = related_pids(table)
    found = []
    for pid, _, command in table:
        if pid in related:
            continue
        if any(needle in command for needle in needles):
            found.append(f"{pid} {command[:160]}")
    return found


def check_exclusive(needles):
    found = foreign_processes(needles)
    require(not found,
            "the window is not exclusive: " + "; ".join(found))


def patrol_needles(binary, stage):
    """Command substrings that mark another MLX process.

    Absolute paths catch a copy launched from anywhere. The bare names catch a
    copy launched with a relative path. This process tree is excluded, so a
    copy does not match itself.
    """
    needles = {"mlx_lm", "mlx_smoke", "umm-infer serve",
               str(Path(__file__).resolve()),
               "umminfer_mlx_stage", "utils/mlx",
               str(BASELINE_STAGE_PATH), BASELINE_STAGE_PATH.name}
    for path in (Path(binary), Path(stage), BASELINE_STAGE_PATH):
        needles.add(str(path.resolve()))
        needles.add(path.name)
    return sorted(needles)


def acquire_lock(path=LOCK_PATH):
    handle = path.open("w", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError(f"another measurement run holds {path}")
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle


def percentile(ordered, fraction):
    index = round(fraction * (len(ordered) - 1))
    return ordered[min(len(ordered) - 1, max(0, index))]


def summarize(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    ordered = sorted(values)
    result = {"count": len(ordered), "median": statistics.median(ordered),
              "min": ordered[0], "max": ordered[-1],
              "p10": percentile(ordered, 0.10), "p90": percentile(ordered, 0.90)}
    if len(ordered) > 1:
        result["stdev"] = statistics.stdev(ordered)
        if result["median"]:
            result["cv"] = result["stdev"] / result["median"]
    return result


def block(rows):
    """Summarize steady state; expose the cold row only when one exists.

    A cold row marks the first real request after one instance became ready.
    The same server serves several transports, so only the first transport
    has a cold row; the others report `cold: null`.
    """
    # A cold row never enters a median, even when `--warmup 0` makes the first
    # request both the cold sample and the first warm one.
    warm = [row for row in rows if row["warm"] and not row.get("cold")]
    summaries = {}
    for metric in NUMERIC_METRICS:
        summary = summarize([row.get(metric) for row in warm])
        if summary is not None:
            summaries[metric] = summary
    return {"cold": next((row for row in rows if row.get("cold")), None),
            "warm": summaries, "samples": rows}


def text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cli_text(output):
    """The chat client prints the completion text and one newline."""
    text = output.decode("utf-8", "replace")
    return text[:-1] if text.endswith("\n") else text


def output_summary(results):
    """One output hash per measured path, plus the cross-path check.

    Every sample of a path must hash the same. Every path must hash the same,
    because the model and the sampling are fixed. A difference means a semantic
    regression, so it is reported instead of averaged away.
    """
    paths = {}

    def collect(name, data):
        if isinstance(data, dict) and data.get("samples"):
            values = {row.get("output_sha256") for row in data["samples"]}
            values.discard(None)
            require(len(values) == 1,
                    f"{name} output changed between samples: {sorted(values)}")
            if values:
                paths[name] = values.pop()
        elif isinstance(data, dict):
            for key, value in data.items():
                collect(f"{name}.{key}", value)

    for mode, data in results.items():
        collect(mode, data)
    if not paths:
        return {}, None, True
    combined = hashlib.sha256("\n".join(
        f"{name}={value}" for name, value in sorted(paths.items())
    ).encode("utf-8")).hexdigest()
    return paths, combined, len(set(paths.values())) == 1


def require_outputs(results):
    """Every measured sample must carry a non-empty output hash.

    A row without one would let a path enter the medians with unverified
    output, so it fails the run.
    """
    def walk(name, data):
        if isinstance(data, dict) and "samples" in data:
            require(data["samples"], f"{name} has no samples")
            for row in data["samples"]:
                value = row.get("output_sha256")
                require(isinstance(value, str) and len(value) == 64,
                        f"{name} sample {row.get('iteration')} has no output hash")
        elif isinstance(data, dict):
            for key, value in data.items():
                walk(f"{name}.{key}", value)

    for mode, data in results.items():
        walk(mode, data)


def require_hash_map(hashes, label, required=("stage",)):
    """Artifact hashes must be present, non-empty, and well formed."""
    require(isinstance(hashes, dict) and hashes, f"{label} hashes are missing")
    for name in required:
        require(name in hashes, f"{label} hashes omit {name}")
    for name, value in hashes.items():
        require(isinstance(value, str) and len(value) == 64
                and all(char in "0123456789abcdef" for char in value),
                f"{label} hash {name} is not a SHA256")


def require_valid_rows(results, max_tokens):
    """Reject invalid samples while preserving natural EOS semantics.

    `max_tokens` is a budget, not a required output length. A short row is
    valid only when MLX-LM reports a stop finish; a full-budget row must report
    length when it carries a finish reason.
    """
    def walk(name, data):
        if isinstance(data, dict) and "samples" in data:
            warm = [row for row in data["samples"]
                    if row.get("warm") and not row.get("cold")]
            require(len(warm) >= MIN_WARM_SAMPLES,
                    f"{name} has {len(warm)} warm non-cold samples; "
                    f"required {MIN_WARM_SAMPLES}")
            for row in warm:
                iteration = row.get("iteration")
                tokens = row.get("generation_tokens")
                require(isinstance(tokens, int) and not isinstance(tokens, bool)
                        and 0 < tokens <= max_tokens,
                        f"{name} sample {iteration} generated {tokens!r} tokens; "
                        f"required 1..{max_tokens}")
                finish = row.get("finish_reason")
                if tokens < max_tokens:
                    require(finish == "stop",
                            f"{name} sample {iteration} stopped at {tokens} tokens "
                            f"with finish_reason={finish!r}")
                elif finish is not None:
                    require(finish == "length",
                            f"{name} sample {iteration} used the token budget "
                            f"with finish_reason={finish!r}")
                total = row.get("total_ms")
                require(isinstance(total, (int, float)) and math.isfinite(total)
                        and total > 0,
                        f"{name} sample {iteration} has total_ms={total!r}")
                memory = row.get("peak_memory_gb")
                require(isinstance(memory, (int, float)) and math.isfinite(memory)
                        and memory > 0,
                        f"{name} sample {iteration} has peak_memory_gb={memory!r}")
                ttft = row.get("ttft_ms")
                if ttft is not None:
                    require(isinstance(ttft, (int, float)) and math.isfinite(ttft)
                            and ttft > 0,
                            f"{name} sample {iteration} has ttft_ms={ttft!r}")
                rate = row.get("output_tokens_per_s")
                require(isinstance(rate, (int, float)) and math.isfinite(rate)
                        and rate > 0,
                        f"{name} sample {iteration} has "
                        f"output_tokens_per_s={rate!r}")
                # Every stored numeric metric must be a real measurement. A
                # negative stage timing or an infinite decode rate would be
                # summarized and printed otherwise.
                for metric in NUMERIC_METRICS:
                    value = row.get(metric)
                    if value is None or metric == "generation_tokens":
                        continue
                    require(isinstance(value, (int, float))
                            and not isinstance(value, bool)
                            and math.isfinite(value) and value > 0,
                            f"{name} sample {iteration} has "
                            f"{metric}={value!r}")
                expected = row["generation_tokens"] / (total / 1000.0)
                require(abs(rate - expected) <= max(1e-6, 1e-9 * abs(expected)),
                        f"{name} sample {iteration} rate {rate!r} does not "
                        f"match generation_tokens/total_ms ({expected!r})")
        elif isinstance(data, dict):
            for key, value in data.items():
                walk(f"{name}.{key}", value)

    for mode, data in results.items():
        walk(mode, data)


def normalize(record):
    """Accept a stage perf-log dict or an MLX-LM GenerationResponse."""
    if record is None:
        return {}
    if isinstance(record, dict):
        return record
    return {"generation_tokens": getattr(record, "generation_tokens", None),
            "prompt_tokens": getattr(record, "prompt_tokens", None),
            "prompt_tps": getattr(record, "prompt_tps", None),
            "generation_tps": getattr(record, "generation_tps", None),
            "peak_memory_gb": getattr(record, "peak_memory", None),
            "finish_reason": getattr(record, "finish_reason", None)}


def make_row(iteration, warmup, ttft, total, record, output):
    values = normalize(record)
    tokens = values.get("generation_tokens")
    stage_ttft = values.get("stage_ttft_seconds")
    stage_total = values.get("stage_total_seconds")
    return {
        "iteration": iteration,
        "cold": False,
        "warm": iteration >= warmup,
        "ttft_ms": None if ttft is None else ttft * 1000.0,
        "total_ms": total * 1000.0,
        "generation_tokens": tokens,
        "finish_reason": values.get("finish_reason"),
        "decode_tokens_per_s": values.get("generation_tps"),
        "output_tokens_per_s": (tokens / total
                                if isinstance(tokens, int) and total > 0 else None),
        "prompt_tokens": values.get("prompt_tokens"),
        "prompt_tokens_per_s": values.get("prompt_tps"),
        "peak_memory_gb": values.get("peak_memory_gb"),
        "stage_ttft_ms": (None if stage_ttft is None else stage_ttft * 1000.0),
        "stage_total_ms": (None if stage_total is None else stage_total * 1000.0),
        "output_sha256": text_sha256(output),
    }


def read_records(path):
    records = []
    if path is None:
        return records
    try:
        with Path(path).open("r", encoding="utf-8") as source:
            for line in source:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return records


class PerfLog:
    """The stage writes one request record before it sends `complete`.

    The cursor keeps a byte offset and reads only appended lines, so a long run
    does not re-parse the whole log for every request.
    """

    def __init__(self, path):
        self.path = None if path is None else Path(path)
        self.offset = 0
        self.partial = b""
        self.pending = []

    def _fill(self):
        if self.path is None:
            return
        try:
            with self.path.open("rb") as source:
                source.seek(self.offset)
                data = source.read()
                self.offset = source.tell()
        except FileNotFoundError:
            return
        lines = (self.partial + data).split(b"\n")
        self.partial = lines.pop()
        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A read can catch the line the stage is still appending.
                continue
            if record.get("event") == "request":
                self.pending.append(record)

    def next(self, expected_id=None):
        if self.path is None:
            return {}
        while not self.pending:
            self._fill()
            if not self.pending:
                require(False, "the stage perf log has no record for the "
                               "finished request")
        record = self.pending.pop(0)
        if expected_id is not None:
            require(record.get("request_id") == expected_id,
                    f"stage record {record.get('request_id')!r} does not match "
                    f"the request {expected_id!r}")
        return record

    def require_exhausted(self):
        """No request record is left over after the last measured request."""
        if self.path is None:
            return
        self._fill()
        require(not self.pending,
                f"the stage perf log has {len(self.pending)} extra request "
                "records after the last measured request")


class Children:
    """Own every child process and cap its captured output."""

    def __init__(self):
        self.procs = {}
        self.logs = {}
        self.readers = []
        self.exited_early = set()
        self.closed = False

    def spawn(self, name, args, env):
        proc = subprocess.Popen(args, cwd=str(ROOT), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.procs[name] = proc
        log = bytearray()
        self.logs[name] = log

        def drain():
            while True:
                chunk = proc.stdout.readline(4096)
                if not chunk:
                    break
                remaining = LOG_CAP_BYTES - len(log)
                if remaining > 0:
                    log.extend(chunk[:remaining])

        thread = threading.Thread(target=drain, daemon=True)
        thread.start()
        self.readers.append(thread)
        return proc

    def close(self):
        """Stop owned children once. A child that already exited is an error.

        Staying up until this call is normal. Exiting before it is a failure
        that would otherwise look like a clean shutdown.
        """
        if self.closed:
            return []
        self.closed = True
        errors = []
        for name, proc in self.procs.items():
            if proc.poll() is not None:
                self.exited_early.add(name)
                errors.append(f"{name} exited before the run ended "
                              f"with code {proc.returncode}")
        for proc in self.procs.values():
            if proc.poll() is None:
                proc.terminate()
        for name, proc in self.procs.items():
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
                errors.append(f"{name} ignored SIGTERM")
        for thread in self.readers:
            thread.join(timeout=5)
        return errors

    def tail(self):
        return {name: bytes(log)[-2000:].decode(errors="replace")
                for name, log in self.logs.items()}


# --------------------------------------------------------------------------
# direct: MLX-LM only


def load_stage_module(stage_path):
    """Load the stage under test so direct mode uses the same loader.

    The vocabulary optimization lives in the stage's `load_mlx_backend`. A
    direct path that called `mlx_lm.load` itself would measure a different
    cost and hide that optimization from the attribution.
    """
    spec = importlib.util.spec_from_file_location(
        "perf_stage_under_test", str(stage_path))
    require(spec is not None and spec.loader is not None,
            f"cannot load the stage module: {stage_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect_generation(generate, prompt, max_tokens, temperature):
    """One generation as (ttft, total, parts, last response).

    First-token latency is the first non-empty delta. An empty
    GenerationResponse is not a token.
    """
    begin = time.perf_counter()
    ttft = None
    last = None
    parts = []
    for response in generate(prompt, max_tokens, temperature):
        if ttft is None and response.text:
            ttft = time.perf_counter() - begin
        last = response
        parts.append(response.text)
    return ttft, time.perf_counter() - begin, parts, last


def run_direct(stage_path, model, max_tokens, warmup, samples):
    module = load_stage_module(stage_path)
    started = time.perf_counter()
    tokenizer, generate = module.load_mlx_backend(str(model))
    load_seconds = time.perf_counter() - started
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}], tokenize=False,
        add_generation_prompt=True)
    rows = []
    for iteration in range(warmup + samples):
        ttft, total, parts, last = collect_generation(
            generate, prompt, max_tokens, TEMPERATURE)
        require(last is not None, "MLX-LM produced no response")
        require("".join(parts), "MLX-LM produced empty output")
        require(isinstance(getattr(last, "generation_tokens", None), int)
                and last.generation_tokens > 0,
                "MLX-LM reported no generation token count")
        row = make_row(iteration, warmup, ttft, total, last, "".join(parts))
        row["cold"] = iteration == 0
        rows.append(row)
    return {"model_load_seconds": load_seconds, **block(rows)}


# --------------------------------------------------------------------------
# stage: the real Python stage over ZeroMQ


def submit_message(request_id, max_tokens, stream):
    return {
        "type": "submit",
        "request_id": request_id,
        "data": {
            "_type": "StagePayload",
            "request_id": request_id,
            "request": {
                "_type": "OmniRequest",
                "inputs": {"messages": [{"role": "user", "content": PROMPT}]},
                "params": {"max_new_tokens": max_tokens,
                           "temperature": TEMPERATURE, "stream": stream},
                "metadata": {},
            },
            "data": None,
        },
    }


def wait_stage_ready(socket_, proc):
    poller = zmq.Poller()
    poller.register(socket_, zmq.POLLIN)
    deadline = time.monotonic() + STAGE_READY_TIMEOUT_S
    while time.monotonic() < deadline:
        require(proc.poll() is None, "the stage exited during startup")
        if poller.poll(200):
            message = msgpack.unpackb(socket_.recv(), raw=False, strict_map_key=False)
            if message.get("type") == "stage_ready":
                return
    raise RuntimeError("the stage did not report ready in time")


def run_stage(stage, model, max_tokens, warmup, samples, perf_path,
              guard=None):
    """Drive the real stage over ZeroMQ on both transports.

    `sse` pairs with direct and with the HTTP/CLI streaming paths. `json`
    pairs with the HTTP JSON path, which is the reference metric. `guard` runs
    before and after each named path, so a competing process is caught per path.
    """
    context = zmq.Context()
    children = Children()
    results = {}
    sockets = []
    try:
        with tempfile.TemporaryDirectory(prefix="umminfer-mlx-perf-") as ipc:
            entry = f"ipc://{ipc}/entry"
            completion = f"ipc://{ipc}/completion"
            abort = f"ipc://{ipc}/abort"
            pull = context.socket(zmq.PULL)
            pull.setsockopt(zmq.RCVHWM, 256)
            pull.bind(completion)
            pub = context.socket(zmq.PUB)
            pub.bind(abort)
            push = context.socket(zmq.PUSH)
            push.setsockopt(zmq.SNDHWM, 256)
            push.connect(entry)
            sockets.extend([pull, pub, push])
            command = [sys.executable, "-I", str(stage), "--name", "mlx",
                       "--recv", entry, "--coordinator", completion,
                       "--abort", abort, "--model", str(model)]
            if perf_path is not None:
                command += ["--perf-log", str(perf_path)]
            proc = children.spawn("stage", command, child_env())
            wait_stage_ready(pull, proc)
            perf = PerfLog(perf_path)
            cold = [True]
            poller = zmq.Poller()
            poller.register(pull, zmq.POLLIN)
            for name, stream in (("sse", True), ("json", False)):
                if guard is not None:
                    guard()
                rows = []
                for iteration in range(warmup + samples):
                    request_id = f"{name}-{iteration}"
                    begin = time.perf_counter()
                    push.send(msgpack.packb(
                        submit_message(request_id, max_tokens, stream),
                        use_bin_type=True))
                    ttft = None
                    deltas = []
                    text = None
                    while True:
                        require(poller.poll(REQUEST_TIMEOUT_S * 1000),
                                f"stage request {request_id} timed out")
                        message = msgpack.unpackb(pull.recv(), raw=False,
                                                  strict_map_key=False)
                        kind = message.get("type")
                        if kind == "stream" and message.get("request_id") == request_id:
                            if ttft is None:
                                ttft = time.perf_counter() - begin
                            deltas.append(message.get("chunk", {}).get("text", ""))
                        elif (kind == "complete"
                              and message.get("request_id") == request_id):
                            require(message.get("success"),
                                    f"stage request failed: {message.get('error')}")
                            text = message.get("result", {}).get("text", "")
                            require(text, "stage returned an empty completion")
                            if stream:
                                # Two views of one completion. A difference is
                                # a stage defect.
                                require("".join(deltas) == text,
                                        "stage deltas do not match the completion text")
                            break
                    total = time.perf_counter() - begin
                    row = make_row(iteration, warmup, ttft, total,
                                   perf.next(request_id), text)
                    row["cold"] = cold[0]
                    cold[0] = False
                    rows.append(row)
                results[name] = block(rows)
                if guard is not None:
                    guard()
            perf.require_exhausted()
        errors = children.close()
        require(not errors, "; ".join(errors))
        return results
    finally:
        # An exception here must not leave a socket held open or a child
        # running: linger 0 closes the socket's queue immediately.
        for socket_ in sockets:
            socket_.close(linger=0)
        context.term()
        errors = children.close()
        if errors:
            print(f"stage cleanup: {'; '.join(errors)}", file=sys.stderr)


# --------------------------------------------------------------------------
# e2e: real server plus real stage, driven over HTTP and the real CLI


def wait_port(port, proc, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        require(proc.poll() is None, "the server exited during startup")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.02)
    raise RuntimeError(f"the server did not listen on {port} in {timeout} seconds")


def wait_ready(port, proc, stage_proc=None, timeout=STAGE_READY_TIMEOUT_S):
    """Wait until `/ready` is 200, which means the stage finished loading.

    A listening socket only proves the HTTP server started. Starting the first
    request before the model exists would put a part of model load into the
    cold sample. With an external stage, a stage exit before ready must fail
    the run instead of holding this wait until its timeout.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        require(proc.poll() is None, "the server exited before it became ready")
        require(stage_proc is None or stage_proc.poll() is None,
                "the stage exited before the server became ready")
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            try:
                conn.request("GET", "/ready")
                response = conn.getresponse()
                response.read(4096)
                if response.status == 200:
                    return
            finally:
                conn.close()
        except OSError:
            pass
        time.sleep(0.05)
    raise RuntimeError(f"/ready did not return 200 in {timeout} seconds")


def http_json(port, max_tokens):
    body = json.dumps({"messages": [{"role": "user", "content": PROMPT}],
                       "max_tokens": max_tokens, "temperature": TEMPERATURE})
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=REQUEST_TIMEOUT_S)
    begin = time.perf_counter()
    try:
        conn.request("POST", "/v1/chat/completions", body=body,
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        data = response.read(OUTPUT_CAP_BYTES + 1)
        total = time.perf_counter() - begin
        require(len(data) <= OUTPUT_CAP_BYTES, "JSON response exceeded the byte cap")
        require(response.status == 200, f"HTTP {response.status}: {data[:400]!r}")
        payload = json.loads(data)
        text = payload["choices"][0]["message"]["content"]
        require(text, "server returned an empty completion")
        return payload.get("id"), None, total, text
    finally:
        conn.close()


def http_sse(port, max_tokens):
    body = json.dumps({"messages": [{"role": "user", "content": PROMPT}],
                       "max_tokens": max_tokens, "temperature": TEMPERATURE,
                       "stream": True})
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=REQUEST_TIMEOUT_S)
    begin = time.perf_counter()
    request_id = None
    ttft = None
    parts = []
    size = 0
    done = False
    deadline = begin + REQUEST_TIMEOUT_S
    try:
        conn.request("POST", "/v1/chat/completions", body=body,
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        require(response.status == 200, f"SSE returned HTTP {response.status}")
        while True:
            require(time.monotonic() < deadline, "the SSE request timed out")
            line = response.readline()
            if not line:
                break
            size += len(line)
            require(size <= OUTPUT_CAP_BYTES, "SSE response exceeded the byte cap")
            if not line.startswith(b"data: "):
                continue
            data = line[len(b"data: "):].strip()
            if data == b"[DONE]":
                done = True
                break
            event = json.loads(data)
            require("error" not in event,
                    f"SSE reported an error event: {event.get('error')!r}")
            request_id = request_id or event.get("id")
            delta = event.get("choices", [{}])[0].get("delta", {})
            text = delta.get("content")
            if text:
                if ttft is None:
                    ttft = time.perf_counter() - begin
                parts.append(text)
        total = time.perf_counter() - begin
        require(done, "the SSE stream ended without [DONE]")
        require("".join(parts), "SSE produced no content delta")
        return request_id, ttft, total, "".join(parts)
    finally:
        conn.close()


def cli_command(binary, port, max_tokens, stream):
    command = [str(binary), "chat", "--url", f"http://127.0.0.1:{port}/v1",
               "--model", SERVED_MODEL, "--prompt", PROMPT,
               "--max-tokens", str(max_tokens), "--temperature", "0",
               "--timeout-ms", str(REQUEST_TIMEOUT_S * 1000)]
    if stream:
        command.append("--stream")
    return command


def has_chat_cli(binary):
    """A binary built before the client landed has no `chat` subcommand."""
    try:
        result = subprocess.run([str(binary), "chat", "--help"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and b"chat" in result.stdout


def cli_request(binary, port, max_tokens, stream):
    """Run the chat client and time its first output byte.

    A streamed CLI prints each delta as it arrives, so the first stdout byte is
    the first content the client received. Reading incrementally is the only
    way to see it; `subprocess.run` reports only the end of the process.
    """
    begin = time.perf_counter()
    proc = subprocess.Popen(cli_command(binary, port, max_tokens, stream),
                            cwd=str(ROOT), env=child_env(),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = bytearray()
    first_byte = [None]
    errors = bytearray()

    def read_stdout():
        while len(output) <= OUTPUT_CAP_BYTES:
            chunk = proc.stdout.read1(65536)
            if not chunk:
                break
            if first_byte[0] is None:
                first_byte[0] = time.perf_counter() - begin
            room = OUTPUT_CAP_BYTES + 1 - len(output)
            output.extend(chunk[:room])

    def read_stderr():
        while True:
            chunk = proc.stderr.read(4096)
            if not chunk:
                return
            if len(errors) < 400:
                errors.extend(chunk[:400 - len(errors)])

    reader = threading.Thread(target=read_stdout, daemon=True)
    error_reader = threading.Thread(target=read_stderr, daemon=True)
    reader.start()
    error_reader.start()
    try:
        proc.wait(timeout=REQUEST_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
        raise RuntimeError("the CLI request timed out")
    total = time.perf_counter() - begin
    reader.join(timeout=10)
    error_reader.join(timeout=10)
    require(len(output) <= OUTPUT_CAP_BYTES, "CLI output exceeded the byte cap")
    require(proc.returncode == 0,
            f"CLI exited {proc.returncode}: {bytes(errors)!r}")
    text = cli_text(bytes(output))
    require(text.strip(), "CLI returned empty output")
    return None, (first_byte[0] if stream else None), total, text


def run_e2e(binary, stage, model, max_tokens, warmup, samples, perf_path,
            guard=None):
    port = free_port()
    children = Children()
    perf = PerfLog(perf_path)
    results = {}
    try:
        with tempfile.TemporaryDirectory(prefix="umminfer-mlx-e2e-") as ipc:
            entry = f"ipc://{ipc}/entry"
            completion = f"ipc://{ipc}/completion"
            abort = f"ipc://{ipc}/abort"
            server = children.spawn("server", [
                str(binary), "serve", "--host", "127.0.0.1", "--port", str(port),
                "--model", SERVED_MODEL, "--entry", "mlx",
                "--entry-endpoint", entry, "--completion", completion,
                "--abort", abort], child_env())
            wait_port(port, server)
            command = [sys.executable, "-I", str(stage), "--name", "mlx",
                       "--recv", entry, "--coordinator", completion,
                       "--abort", abort, "--model", str(model)]
            if perf_path is not None:
                command += ["--perf-log", str(perf_path)]
            stage_proc = children.spawn("stage", command, child_env())
            # With an external entry the API is ready only after the stage
            # announces stage_ready, so /ready is the common start point and
            # the stage must be running before this wait begins. `/ready` 200
            # already means the model was loaded, so no perf-log wait is needed.
            wait_ready(port, server, stage_proc)

            require(has_chat_cli(binary),
                    "the binary has no chat subcommand; a formal run cannot skip paths")
            cold = [True]

            def loop(name, transport):
                if guard is not None:
                    guard()
                rows = []
                for iteration in range(warmup + samples):
                    request_id, ttft, total, output = transport(port, max_tokens)
                    require(output.strip(), f"{name} returned empty output")
                    row = make_row(iteration, warmup, ttft, total,
                                   perf.next(request_id), output)
                    # Only the first request of this server is cold. The same
                    # server serves the other transports, so theirs is null.
                    row["cold"] = cold[0]
                    cold[0] = False
                    rows.append(row)
                if guard is not None:
                    guard()
                return block(rows)

            results["json"] = loop("http_json", http_json)
            results["sse"] = loop("http_sse", http_sse)
            results["cli_json"] = loop("cli_json",
                                       lambda p, m: cli_request(binary, p, m, False))
            results["cli_sse"] = loop("cli_sse",
                                      lambda p, m: cli_request(binary, p, m, True))
            perf.require_exhausted()
        errors = children.close()
        require(not errors, "; ".join(errors))
        return results
    finally:
        errors = children.close()
        if errors:
            print(f"e2e cleanup: {'; '.join(errors)}", file=sys.stderr)
            print(children.tail(), file=sys.stderr)


# --------------------------------------------------------------------------
# report


def median_of(data, metric="total_ms"):
    summary = (data or {}).get("warm", {}).get(metric)
    return None if summary is None else summary.get("median")


def attribution(results):
    """Layer costs from transports that measure the same work.

    Pairs: `direct` and `stage.sse` both stream; `stage.json` and HTTP JSON
    both run without deltas; HTTP SSE and CLI SSE both stream. A difference of
    unlike transports would mix the streaming cost into the comparison.
    """
    direct = results.get("direct") or {}
    stage = results.get("stage") or {}
    e2e = results.get("e2e") or {}
    rows = {}

    def difference(name, high, low):
        high_ms = median_of(high)
        low_ms = median_of(low)
        if high_ms is not None and low_ms is not None:
            rows[name] = high_ms - low_ms

    difference("stage_sse_overhead_ms", stage.get("sse"), direct)
    difference("http_json_overhead_ms", e2e.get("json"), stage.get("json"))
    difference("http_sse_overhead_ms", e2e.get("sse"), stage.get("sse"))
    difference("cli_json_overhead_ms", e2e.get("cli_json"), e2e.get("json"))
    difference("cli_sse_overhead_ms", e2e.get("cli_sse"), e2e.get("sse"))

    direct_rate = median_of(direct, "output_tokens_per_s")
    json_rate = median_of(e2e.get("json"), "output_tokens_per_s")
    if direct_rate and json_rate:
        rows["output_tokens_per_s_direct"] = direct_rate
        rows["output_tokens_per_s_e2e_json"] = json_rate
        rows["e2e_fraction_of_direct"] = json_rate / direct_rate
    return rows


def skipped_paths(results):
    found = []

    def walk(name, data):
        if not isinstance(data, dict):
            return
        if "skipped" in data:
            found.append(name)
            return
        for key, value in data.items():
            walk(f"{name}.{key}", value)

    for mode, data in results.items():
        walk(mode, data)
    return found


def print_block(title, data):
    cold = data.get("cold") or {}
    if cold:
        print(f"COLD {title} " + json.dumps(
            {key: cold.get(key) for key in ("ttft_ms", "total_ms",
                                            "generation_tokens",
                                            "output_tokens_per_s")},
            sort_keys=True))
    for metric, summary in data.get("warm", {}).items():
        print(f"SUMMARY {title} {metric} " + json.dumps(summary, sort_keys=True))
        if summary.get("cv", 0) > 0.10:
            print(f"WARN {title} {metric} cv={summary['cv']:.3f} above 0.10")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("direct", "stage", "e2e", "all"),
                        default="all")
    parser.add_argument("--role", default="unspecified",
                        help="label for the report, for example baseline")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=1,
                        help="formal measurements use exactly one attempt")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not 1 <= args.samples <= MAX_SAMPLES:
        parser.error(f"--samples must be between 1 and {MAX_SAMPLES}")
    if not 0 <= args.warmup <= MAX_WARMUP:
        parser.error(f"--warmup must be between 0 and {MAX_WARMUP}")
    if not 1 <= args.max_tokens <= MAX_TOKENS_CAP:
        parser.error(f"--max-tokens must be between 1 and {MAX_TOKENS_CAP}")
    if args.max_attempts != 1:
        parser.error("--max-attempts must be 1: a retry hides a failed sample")
    model = args.model.resolve()
    stage = args.stage.resolve()
    binary = args.binary.resolve()
    require((model / "tokenizer.json").is_file(), f"model not found: {model}")
    require_model_layout(model)
    require(stage.is_file(), f"stage not found: {stage}")
    if args.mode in ("stage", "e2e", "all"):
        require((model / "config.json").is_file(), "model has no config.json")
    if args.mode in ("e2e", "all"):
        require(binary.is_file(), f"binary not found: {binary}")

    # Hold the lock for the whole run. A second harness, a second stage copy,
    # or any other MLX process makes the numbers invalid, so there is no flag
    # to continue: the run stops instead.
    lock = acquire_lock()
    needles = patrol_needles(binary, stage)
    check_exclusive(needles)

    perf_dir = Path(tempfile.mkdtemp(prefix="umminfer-mlx-perf-log-"))
    # One log per mode. A shared log would make the e2e cursor read the stage
    # mode's records, so an e2e metric could carry a stage-mode timing.
    stage_perf = perf_dir / "stage.jsonl"
    e2e_perf = perf_dir / "e2e.jsonl"
    # The artifacts are pinned before the run and checked again after it, so a
    # rebuild in the middle invalidates the numbers instead of mixing them.
    hashes_before = {"stage": digest(stage),
                     "script": digest(Path(__file__).resolve()),
                     "weights": digest(model / "model.safetensors")}
    if binary.is_file():
        hashes_before["binary"] = digest(binary)
    inputs = model_input_hashes(model)
    for name, value in inputs.items():
        hashes_before[f"model.{name}"] = value
    # A missing or malformed hash would let the after-run check pass by being
    # absent from both sides.
    require_hash_map(hashes_before, "hashes_before",
                     required=("stage", "script", "weights")
                     + tuple(f"model.{name}" for name in inputs))
    guard = lambda: check_exclusive(needles)
    results = {}
    started = time.perf_counter()
    completed = False
    error = None
    try:
        if args.mode in ("direct", "all"):
            print("MODE direct", flush=True)
            check_exclusive(needles)
            results["direct"] = run_direct(stage, model, args.max_tokens,
                                           args.warmup, args.samples)
            check_exclusive(needles)
            print_block("direct", results["direct"])
        if args.mode in ("stage", "all"):
            print("MODE stage", flush=True)
            check_exclusive(needles)
            results["stage"] = run_stage(stage, model, args.max_tokens,
                                         args.warmup, args.samples, stage_perf,
                                         guard)
            check_exclusive(needles)
            for name, data in results["stage"].items():
                print_block(f"stage.{name}", data)
        if args.mode in ("e2e", "all"):
            print("MODE e2e", flush=True)
            check_exclusive(needles)
            results["e2e"] = run_e2e(binary, stage, model, args.max_tokens,
                                     args.warmup, args.samples, e2e_perf,
                                     guard)
            check_exclusive(needles)
            for name, data in results["e2e"].items():
                print_block(name, data)
        # The gates below run on raw rows, not on the summaries: a stored
        # median cannot prove that the samples behind it were valid.
        require_valid_rows(results, args.max_tokens)
        require_outputs(results)
        hashes_after = {"stage": digest(stage),
                        "script": digest(Path(__file__).resolve()),
                        "weights": digest(model / "model.safetensors")}
        if binary.is_file():
            hashes_after["binary"] = digest(binary)
        # Re-enumerate the inputs the same way as before the run. The
        # `chat_template` entry is a hash of the template string, not a file,
        # so a plain digest(model/name) would fail on that layout.
        for name, value in model_input_hashes(model).items():
            hashes_after[f"model.{name}"] = value
        require_hash_map(hashes_after, "hashes_after",
                         required=("stage", "script", "weights")
                     + tuple(f"model.{name}" for name in inputs))
        require(hashes_after == hashes_before,
                "an artifact changed during the run")
        completed = True
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        elapsed = time.perf_counter() - started
        try:
            outputs, output_sha256, output_consistent = output_summary(results)
        except RuntimeError as exc:
            outputs, output_sha256, output_consistent = {}, None, False
            error = error or f"RuntimeError: {exc}"
            completed = False
        skipped = skipped_paths(results)
        try:
            foreign = foreign_processes(needles)
        except Exception as exc:
            foreign = [f"exclusivity check failed: {exc}"]
        if foreign:
            completed = False
            error = error or "the window was not exclusive at the end of the run"
        try:
            hashes_end = {"stage": digest(stage),
                          "script": digest(Path(__file__).resolve()),
                          "weights": digest(model / "model.safetensors")}
            if binary.is_file():
                hashes_end["binary"] = digest(binary)
            for name, value in model_input_hashes(model).items():
                hashes_end[f"model.{name}"] = value
            require_hash_map(hashes_end, "hashes_end",
                             required=("stage", "script", "weights")
                             + tuple(f"model.{name}" for name in inputs))
        except (OSError, RuntimeError) as exc:
            hashes_end = {"error": str(exc)}
            completed = False
            error = error or f"artifact rehash failed: {exc}"
        # Final rehash: a rebuild or edit after the last per-mode check must
        # invalidate the report rather than leave it marked complete.
        if hashes_end != hashes_before and completed:
            completed = False
            error = error or "an artifact changed before the report was written"
        if not output_consistent:
            completed = False
            error = error or "the measured paths did not produce identical output"
        load = next((record for path in (stage_perf, e2e_perf)
                     for record in read_records(path)
                     if record.get("event") == "load"), {})
        report = {
            "role": args.role,
            "completed": completed,
            "error": error,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "wall_seconds": elapsed,
            "configuration": {
                "model": str(model),
                "model_safetensors_sha256": digest(model / "model.safetensors"),
                "prompt": PROMPT,
                "temperature": TEMPERATURE,
                "max_tokens": args.max_tokens,
                "samples": args.samples,
                "warmup": args.warmup,
                "max_attempts": args.max_attempts,
            },
            "environment": {
                "host": platform.node(),
                "machine": platform.machine(),
                "interpreter": sys.executable,
                "python": platform.python_version(),
                "mlx": importlib_metadata.version("mlx"),
                "mlx_lm": importlib_metadata.version("mlx-lm"),
                # The comparator checks these against a frozen manifest. A
                # missing package is recorded as null, not raised.
                "msgpack": package_version("msgpack"),
                "pyzmq": package_version("pyzmq"),
                "tokenizers": package_version("tokenizers"),
                "transformers": package_version("transformers"),
                "binary": None if not binary.is_file() else {
                    "path": str(binary), "sha256": hashes_before.get("binary")},
                "stage": {"path": str(stage),
                          "sha256": hashes_before.get("stage")},
                "direct_loader": "stage.load_mlx_backend",
                "script": {"path": str(Path(__file__).resolve()),
                           "sha256": digest(Path(__file__).resolve())},
            },
            "exclusive_window": {"lock": str(LOCK_PATH),
                                 "foreign_processes": foreign,
                                 "hashes_before": hashes_before,
                                 "hashes_after": hashes_end},
            "stage_model_load_seconds": load.get("seconds"),
            "outputs": outputs,
            "output_sha256": output_sha256,
            "output_consistent": output_consistent,
            "skipped_paths": skipped,
            "results": results,
            "attribution": attribution(results),
        }
        if report["attribution"]:
            print("ATTRIBUTION " + json.dumps(report["attribution"], sort_keys=True))
        print(f"OUTPUT sha256={output_sha256} consistent={output_consistent} "
              + json.dumps(outputs, sort_keys=True))
        if args.out is not None:
            args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            print(f"REPORT {args.out}")
        lock.close()
        shutil.rmtree(perf_dir, ignore_errors=True)
    if not completed:
        print("FAIL the run did not complete" +
              ("" if error is None else f": {error}"), file=sys.stderr)
        return 1
    if skipped:
        print("FAIL the run skipped a path: " + ", ".join(skipped),
              file=sys.stderr)
        return 1
    if not output_consistent:
        print("FAIL the measured paths did not produce identical output",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

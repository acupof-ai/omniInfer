#!/usr/bin/env python3
"""Measure real CLI latency through umm-infer and a persistent MLX model."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import select
import statistics
import subprocess
import sys
import tempfile
import threading
import time

from tokenizers import Tokenizer


MAX_ITERATIONS = 20


def load_smoke():
    path = Path(__file__).resolve().with_name("client_smoke.py")
    spec = importlib.util.spec_from_file_location("real_client_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summary(values):
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--stage", type=Path,
                        default=Path("/Users/bytedance/code/omniInfer/python/umminfer_mlx_stage.py"))
    parser.add_argument("--model", type=Path,
                        default=Path("/Users/bytedance/.cache/huggingface/hub/"
                                     "models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/"
                                     "snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"))
    args = parser.parse_args()
    if not 5 <= args.iterations <= MAX_ITERATIONS:
        parser.error(f"--iterations must be between 5 and {MAX_ITERATIONS}")
    binary = args.binary.resolve()
    stage = args.stage.resolve()
    model = args.model.resolve()
    smoke = load_smoke()
    smoke.require(binary.is_file(), f"binary not found: {binary}")
    smoke.require(stage.is_file(), f"stage not found: {stage}")
    smoke.require((model / "tokenizer.json").is_file(), "model has no tokenizer.json")
    tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
    with binary.open("rb") as source:
        print(f"binary_sha256={hashlib.file_digest(source, 'sha256').hexdigest()}", flush=True)

    import socket
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                "TOKENIZERS_PARALLELISM": "false"})
    processes = {}
    logs = {}
    readers = []
    cli_children = []
    ready = threading.Event()
    samples = {"real_json_total_ms": [], "real_sse_ttft_ms": [], "real_sse_total_ms": []}
    expected_output = None
    succeeded = False

    def run_sse():
        started = time.perf_counter()
        proc = subprocess.Popen(
            smoke.cli_command(binary, port, smoke.PROMPT, 16, stream=True),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        cli_children.append(proc)
        try:
            readable, _, _ = select.select([proc.stdout], [], [], smoke.REQUEST_TIMEOUT_S)
            smoke.require(readable, "real SSE CLI produced no output within 60 seconds")
            first = os.read(proc.stdout.fileno(), 4096)
            smoke.require(first, "real SSE CLI exited before model output")
            ttft_ms = (time.perf_counter() - started) * 1000
            rest, error = proc.communicate(timeout=smoke.REQUEST_TIMEOUT_S)
            total_ms = (time.perf_counter() - started) * 1000
            output = first + rest
            smoke.check_output(output, error)
            smoke.require(proc.returncode == 0, f"real SSE CLI failed: {error!r}")
            return output, ttft_ms, total_ms
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate(timeout=5)

    try:
        with tempfile.TemporaryDirectory(prefix="umminfer-client-benchmark-") as ipc_dir:
            entry = f"ipc://{ipc_dir}/entry"
            completion = f"ipc://{ipc_dir}/completion"
            abort_endpoint = f"ipc://{ipc_dir}/abort"
            server_started = time.perf_counter()
            smoke.spawn("server", [str(binary), "serve", "--host", "127.0.0.1",
                                   "--port", str(port), "--model", smoke.MODEL_NAME,
                                   "--entry", "mlx", "--entry-endpoint", entry,
                                   "--completion", completion, "--abort", abort_endpoint],
                        binary.parent, env, processes, logs, readers, ready)
            smoke.wait_for_server(port, processes["server"])
            server_ready_ms = (time.perf_counter() - server_started) * 1000
            stage_started = time.perf_counter()
            smoke.spawn("stage", [sys.executable, "-I", str(stage), "--name", "mlx",
                                  "--recv", entry, "--coordinator", completion,
                                  "--abort", abort_endpoint, "--model", str(model)],
                        stage.parent, env, processes, logs, readers, ready)
            smoke.wait_for_stage(processes["stage"], ready)
            stage_ready_ms = (time.perf_counter() - stage_started) * 1000
            stage_pid = processes["stage"].pid
            print(f"STARTUP server_ready_ms={server_ready_ms:.3f} "
                  f"stage_ready_ms={stage_ready_ms:.3f}", flush=True)
            print(f"PASS real_benchmark_startup port={port} stage_pid={stage_pid}", flush=True)

            for index in range(args.iterations):
                result, elapsed = smoke.run_cli(binary, port, env)
                smoke.require(result.stdout.strip(), "real JSON CLI returned empty output")
                if expected_output is None:
                    expected_output = result.stdout
                smoke.require(result.stdout == expected_output,
                              "real model output changed across JSON iterations")
                text = result.stdout.decode().removesuffix("\n")
                tokens = len(tokenizer.encode(text, add_special_tokens=False).ids)
                value = elapsed * 1000
                samples["real_json_total_ms"].append(value)
                print(f"SAMPLE real_json iteration={index} total_ms={value:.3f} "
                      f"tokens={tokens}", flush=True)

            for index in range(args.iterations):
                output, ttft_ms, total_ms = run_sse()
                smoke.require(output == expected_output,
                              "real SSE output differs from real JSON output")
                text = output.decode().removesuffix("\n")
                tokens = len(tokenizer.encode(text, add_special_tokens=False).ids)
                samples["real_sse_ttft_ms"].append(ttft_ms)
                samples["real_sse_total_ms"].append(total_ms)
                print(f"SAMPLE real_sse iteration={index} ttft_ms={ttft_ms:.3f} "
                      f"total_ms={total_ms:.3f} tokens={tokens}", flush=True)

            concurrent, concurrent_wall = smoke.run_concurrent_json(
                binary, port, env, smoke.CONCURRENT_MARKERS, cli_children)
            for item in concurrent:
                text = item["stdout"].decode(errors="replace")
                marker = item["marker"]
                smoke.require(item["returncode"] == 0,
                              f"concurrent CLI {marker} failed: {item['stderr']!r}")
                smoke.require(marker in text,
                              f"concurrent response for {marker} omitted its marker: {text!r}")
                for other in smoke.CONCURRENT_MARKERS:
                    if other != marker:
                        smoke.require(other not in text,
                                      f"concurrent response for {marker} contained {other}")
                print(f"SAMPLE real_concurrent_json marker={marker} "
                      f"latency_ms={item['elapsed'] * 1000:.3f}", flush=True)
            print(f"BATCH real_concurrent_json requests=4 "
                  f"wall_ms={concurrent_wall * 1000:.3f}", flush=True)

            smoke.require(processes["stage"].pid == stage_pid and
                          processes["stage"].poll() is None,
                          "persistent MLX stage exited or was replaced")
            smoke.require(bytes(logs["stage"]).count(b"MLX stage mlx ready") == 1,
                          "persistent MLX stage emitted more than one ready event")
            print(f"PASS real_benchmark_persistent_stage "
                  f"requests={args.iterations * 2 + 4} ready_events=1 stage_pid={stage_pid}",
                  flush=True)
            json_values = samples["real_json_total_ms"]
            print(f"COLD_START first_json_ms={json_values[0]:.3f} "
                  f"subsequent_json_median_ms={statistics.median(json_values[1:]):.3f}",
                  flush=True)
            for name, values in samples.items():
                print(f"SUMMARY {name} " + json.dumps(summary(values), sort_keys=True),
                      flush=True)
            succeeded = True
    finally:
        errors = smoke.cleanup(processes, readers, cli_children)
        if processes and not errors:
            codes = {name: proc.returncode for name, proc in processes.items()}
            print(f"PASS cleanup owned_processes_reaped exit_codes={codes}", flush=True)
        for error in errors:
            print(f"FAIL cleanup: {error}", file=sys.stderr, flush=True)
        if not succeeded:
            for name, data in logs.items():
                print(f"{name} log tail:\n{bytes(data).decode(errors='replace')[-4000:]}",
                      file=sys.stderr, flush=True)
        if succeeded and errors:
            raise RuntimeError("; ".join(errors))


if __name__ == "__main__":
    main()

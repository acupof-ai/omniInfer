#!/usr/bin/env python3
"""Black-box client boundary tests.

Two halves:
  * CLI parsing and URL policy, run directly against the real binary.
  * Real regression: real umm-infer server + real MLX stage + real model for
    chat, and a real tiny ASR stage for transcription. No fake server and no
    fabricated responses.
"""

import argparse
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
import time
import wave

STARTUP_TIMEOUT_S = 120
REQUEST_TIMEOUT_S = 120
SHUTDOWN_TIMEOUT_S = 20
MODEL_NAME = "Qwen2.5-0.5B-Instruct-4bit"
PROMPT = "Give one concise inference tip."
ASR_MODEL_NAME = "tiny-ctc"
DEPS_PREFIX = Path(os.environ.get("UMMINFER_DEPS_PREFIX", "/opt/homebrew"))
CURL_PREFIX = Path(os.environ.get("UMMINFER_CURL_PREFIX", "/opt/homebrew/opt/curl"))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def health_in_flight(port):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        conn.request("GET", "/health")
        return json.loads(conn.getresponse().read(4096))["in_flight"]
    finally:
        conn.close()


def wait_in_flight_zero(port, timeout_s=10):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if health_in_flight(port) == 0:
            return
        time.sleep(0.02)
    raise AssertionError("real server in_flight did not return to zero")


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def offline_env():
    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                "TOKENIZERS_PARALLELISM": "false"})
    return env


def run(args, env=None, timeout=REQUEST_TIMEOUT_S):
    return subprocess.run([str(arg) for arg in args], stdin=subprocess.DEVNULL,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          env=env or os.environ.copy(), timeout=timeout)


def expect_usage_error(binary, args, message):
    result = run([binary] + list(args), timeout=30)
    require(result.returncode == 2,
            f"{args} exit {result.returncode}, expected 2: {result.stderr!r}")
    require(message in result.stderr,
            f"{args} stderr lacked {message!r}: {result.stderr!r}")
    return result


def check_chat_parsing(binary):
    expect_usage_error(binary, ["chat", "--bogus", "x"], b"unknown option: --bogus")
    expect_usage_error(binary, ["chat", "hello"], b"unexpected argument: hello")
    expect_usage_error(binary, ["chat", "--prompt", "a", "--prompt", "b"],
                       b"duplicate option: --prompt")
    expect_usage_error(binary, ["chat", "--prompt", "x", "--stream", "--stream"],
                       b"duplicate option: --stream")
    expect_usage_error(binary, ["realtime", "--url", "http://127.0.0.1:1",
                                 "--model", "m", "--file", "/dev/null",
                                 "--live-result", "--live-result"],
                       b"duplicate option: --live-result")
    expect_usage_error(binary, ["chat", "--prompt", "hi", "--top-p", "1.5"],
                       b"invalid --top-p: 1.5")
    expect_usage_error(binary, ["chat", "--prompt", "hi", "--top-p", "-0.1"],
                       b"invalid --top-p: -0.1")
    expect_usage_error(binary, ["chat", "--prompt", "hi", "--top-p", "abc"],
                       b"invalid --top-p: abc")
    expect_usage_error(binary, ["chat", "--prompt", "hi", "--top-p", "nan"],
                       b"invalid --top-p: nan")
    expect_usage_error(binary, ["chat", "--prompt", "hi", "--top-p"],
                       b"--top-p needs a value")
    for flag, value in (("--max-attempts", "0"), ("--max-attempts", "11"),
                        ("--max-attempts", "abc"), ("--max-tokens", "-1"),
                        ("--temperature", "2.5"), ("--temperature", "abc"),
                        ("--timeout-ms", "-1"), ("--first-token-ms", "-1"),
                        ("--stall-ms", "-1")):
        expect_usage_error(binary, ["chat", "--prompt", "hi", flag, value],
                           f"invalid {flag}: {value}".encode())
    help_result = run([binary, "chat", "--help"], timeout=30)
    require(help_result.returncode == 0,
            f"chat --help exit {help_result.returncode}: {help_result.stderr!r}")
    require(b"--top-p" in help_result.stdout,
            f"chat --help omitted --top-p: {help_result.stdout!r}")
    print("PASS chat_cli_parsing top_p_rejected_invalid usage_matches_parser", flush=True)


def check_server_parsing(binary):
    duplicates = {
        "serve": ("--port", ["--port", "1", "--port", "2"]),
        "stage": ("--name", ["--name", "a", "--name", "b"]),
    }
    for command in ("serve", "stage"):
        expect_usage_error(binary, [command, "--bogus", "x"],
                           b"unknown option: --bogus")
        expect_usage_error(binary, [command, "stray"],
                           b"unexpected argument: stray")
        key, args = duplicates[command]
        expect_usage_error(binary, [command] + args, b"duplicate option: " + key.encode())
        expect_usage_error(binary, [command, key], key.encode() + b" needs a value")
        help_result = run([binary, command, "--help"], timeout=30)
        require(help_result.returncode == 0,
                f"{command} --help exit {help_result.returncode}: {help_result.stderr!r}")
        require(b"usage:" in help_result.stdout,
                f"{command} --help omitted usage: {help_result.stdout!r}")
    expect_usage_error(binary, ["serve", "--demo", "--demo"],
                       b"duplicate option: --demo")
    for value in ("false", "0", "arbitrary"):
        expect_usage_error(binary, ["serve", f"--demo={value}"],
                           b"--demo does not take a value")
    expect_usage_error(binary, ["serve", "--next", "a=ipc:///tmp/a"],
                       b"unknown option: --next")
    expect_usage_error(binary, ["stage", "--demo"], b"unknown option: --demo")
    print("PASS serve_stage_parsing unknown positional duplicate help", flush=True)


def check_stage_multi_value(binary, root):
    def stage_command(*extra):
        return [binary, "stage", "--name", "probe",
                "--recv", f"ipc://{root}/probe/in",
                "--coordinator", f"ipc://{root}/probe/done",
                "--abort", f"ipc://{root}/probe/abort", *extra]

    log_path = root / "probe_stage.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            [str(arg) for arg in stage_command(
                "--next", f"worker=ipc://{root}/probe/worker",
                "--peer", f"worker=ipc://{root}/probe/worker",
                "--peer", f"extra=ipc://{root}/probe/extra")],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        try:
            time.sleep(1.5)
            require(process.poll() is None,
                    "stage rejected repeated --next/--peer routing options: "
                    f"{log_path.read_bytes()[-2000:]!r}")
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
            code = process.wait(timeout=SHUTDOWN_TIMEOUT_S)
    require(code == 0, f"stage with routing options exited {code}")
    print("PASS stage_multi_value next_peer_accept_and_repeat", flush=True)

    duplicate = run(stage_command(
        "--gpu-id", "0",
        "--peer", f"worker=ipc://{root}/probe/worker",
        "--peer-gpu", "worker=0", "--peer-gpu", "worker=1"), timeout=30)
    require(duplicate.returncode == 1 and
            b"duplicate --peer-gpu target: worker" in duplicate.stderr,
            f"duplicate --peer-gpu target stderr was {duplicate.stderr!r}")
    malformed = run(stage_command("--peer-gpu", "worker"), timeout=30)
    require(malformed.returncode == 1 and b"requires name=N" in malformed.stderr,
            f"malformed --peer-gpu stderr was {malformed.stderr!r}")
    print("PASS stage_peer_gpu name_value_validation", flush=True)


def check_asr_parsing(binary, root):
    """User-facing audio argument errors exit 2 for usage, 1 for bad audio."""
    odd = root / "odd.pcm"
    odd.write_bytes(b"\x01")
    broken = root / "broken.wav"
    broken.write_bytes(b"not a wav file")
    missing = root / "absent.wav"
    expect_usage_error(binary, ["transcribe", "--url", "http://127.0.0.1:1",
                                "--model", "m"], b"--file is required")
    expect_usage_error(binary, ["transcribe", "--url", "http://127.0.0.1:1",
                                "--model", "m", "--file", str(odd),
                                "--format", "bogus"], b"--format must be wav or pcm")
    expect_usage_error(binary, ["realtime", "--url", "http://127.0.0.1:1",
                                "--model", "m", "--file", str(missing),
                                "--final-timeout-ms", "-1"],
                       b"invalid --final-timeout-ms: -1")
    expect_usage_error(binary, ["realtime", "--url", "http://127.0.0.1:1",
                                "--model", "m", "--file", str(missing),
                                "--max-new-tokens", "-1"],
                       b"invalid --max-new-tokens: -1")
    for label, args, needle in (
            ("missing_file", ["--file", str(missing)], b"audio file is missing or too large"),
            ("odd_pcm", ["--file", str(odd), "--format", "pcm"],
             b"PCM16 audio has an odd byte count"),
            ("broken_wav", ["--file", str(broken), "--format", "wav"],
             b"WAV must contain one PCM16 mono 16 kHz data chunk")):
        result = run([binary, "transcribe", "--url", "http://127.0.0.1:1",
                      "--model", "m"] + args, timeout=30)
        require(result.returncode == 1,
                f"transcribe {label} exit {result.returncode}: {result.stderr!r}")
        require(needle in result.stderr,
                f"transcribe {label} stderr was {result.stderr!r}")
    print("PASS asr_cli_parsing usage_exit_2 bad_audio_exit_1", flush=True)


def check_url_policy(binary):
    for url in ("http://127.example.com:9/v1", "http://127.0.0.1.nip.invalid:9/v1"):
        result = run([binary, "chat", "--url", url, "--prompt", "x"], timeout=30)
        require(result.returncode == 1 and
                b"plain HTTP is allowed only on loopback" in result.stderr,
                f"chat {url} stderr was {result.stderr!r}")
    result = run([binary, "transcribe", "--url", "http://127.example.com:9",
                  "--model", "m", "--file", "/dev/null"], timeout=30)
    require(result.returncode == 1 and b"ASR URL must use a loopback host" in result.stderr,
            f"transcribe DNS-prefix host stderr was {result.stderr!r}")
    for url in ("http://127.0.0.1:1/v1", "http://localhost:1/v1"):
        result = run([binary, "chat", "--url", url, "--prompt", "x",
                      "--timeout-ms", "500", "--max-attempts", "1"], timeout=30)
        require(result.returncode == 1, f"chat {url} exit {result.returncode}")
        require(b"loopback" not in result.stderr,
                f"chat {url} was rejected as non-loopback: {result.stderr!r}")
    print("PASS url_policy dns_prefix_rejected real_loopback_allowed", flush=True)


def wait_http(port, path, expected, process, label, token=None,
              timeout=STARTUP_TIMEOUT_S):
    deadline = time.monotonic() + timeout
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    last = None
    while time.monotonic() < deadline:
        require(process.poll() is None, f"{label} exited before {path} was ready")
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            conn.request("GET", path, headers=headers)
            response = conn.getresponse()
            last = (response.status, response.read(4096))
            conn.close()
            if last[0] == expected:
                return
        except OSError:
            pass
        time.sleep(0.05)
    raise AssertionError(f"{label} {path} never returned {expected}: {last!r}")


def spawn(name, command, cwd, log_dir, env, logs):
    log = (log_dir / f"{name}.log").open("wb")
    process = subprocess.Popen([str(arg) for arg in command], cwd=str(cwd), env=env,
                               stdin=subprocess.DEVNULL, stdout=log,
                               stderr=subprocess.STDOUT)
    logs.append((name, log, process))
    return process


def stop_process(process):
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
    try:
        return process.wait(timeout=SHUTDOWN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.wait(timeout=SHUTDOWN_TIMEOUT_S)


def cleanup(processes, logs, root):
    for process in processes:
        stop_process(process)
    errors = []
    for name, log, process in logs:
        if process.returncode is None:
            errors.append(f"{name} was not reaped")
        if process.returncode not in (0, -signal.SIGTERM):
            errors.append(f"{name} exited with {process.returncode}")
            tail = (root / f"{name}.log").read_bytes()[-4000:]
            print(f"{name} exit {process.returncode} log tail:\n"
                  f"{tail.decode(errors='replace')}", file=sys.stderr, flush=True)
    for _, log, _ in logs:
        log.close()
    return errors


PROBE_SOURCE = r'''
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <utility>

#include "umminfer/client/client.hpp"

using namespace umminfer::client;

const char* code_name(ErrorCode code) {
  switch (code) {
    case ErrorCode::kConfig: return "kConfig";
    case ErrorCode::kAuthentication: return "kAuthentication";
    case ErrorCode::kRateLimited: return "kRateLimited";
    case ErrorCode::kBadRequest: return "kBadRequest";
    case ErrorCode::kServer: return "kServer";
    case ErrorCode::kNetwork: return "kNetwork";
    case ErrorCode::kTimeout: return "kTimeout";
    case ErrorCode::kCancelled: return "kCancelled";
    case ErrorCode::kBadResponse: return "kBadResponse";
    case ErrorCode::kContentFilter: return "kContentFilter";
    case ErrorCode::kEmpty: return "kEmpty";
  }
  return "unknown";
}

void escape(std::string& out, const std::string& text) {
  for (char c : text) {
    if (c == '"' || c == '\\') out += '\\';
    if (c == '\n' || c == '\r') { out += ' '; continue; }
    out += c;
  }
}

int main(int argc, char** argv) {
  if (argc != 4) return 2;
  std::ifstream input(argv[3], std::ios::binary);
  if (!input.good()) return 2;
  std::string pcm((std::istreambuf_iterator<char>(input)),
                  std::istreambuf_iterator<char>());
  ASRConfig config;
  config.base_url = argv[1];
  config.model = argv[2];
  config.connect_timeout = Milliseconds{5000};
  config.stall_timeout = Milliseconds{30000};
  TranscribeRequest request;
  request.pcm = std::move(pcm);
  request.timeout = Milliseconds{20000};
  try {
    const auto response = ASRClient(config).transcribe(request);
    std::string text;
    escape(text, response.text);
    std::cout << "{\"ok\":true,\"text\":\"" << text << "\"}\n";
    return 0;
  } catch (const Error& error) {
    std::string detail;
    escape(detail, error.what());
    std::cout << "{\"ok\":false,\"code\":\"" << code_name(error.code())
              << "\",\"status\":" << error.status()
              << ",\"retryable\":" << (error.retryable() ? "true" : "false")
              << ",\"what\":\"" << detail << "\"}\n";
    return 0;
  }
}
'''


def compile_asr_probe(project, work, deps_prefix, curl_prefix):
    source = work / "asr_client_probe.cpp"
    binary = work / "asr_client_probe"
    source.write_text(PROBE_SOURCE, encoding="utf-8")
    command = ["c++", "-std=c++20", "-O1", str(source),
               str(project / "src/audio/audio.cpp"),
               str(project / "src/client/http.cpp"),
               str(project / "src/client/asr.cpp"),
               "-I", str(project / "include"), "-I", str(project / "src/client"),
               "-I", str(deps_prefix / "include"),
               "-I", str(curl_prefix / "include"), "-L", str(curl_prefix / "lib"),
               "-Wl,-rpath," + str(curl_prefix / "lib"), "-lcurl", "-pthread",
               "-o", str(binary)]
    result = subprocess.run(command, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=180)
    require(result.returncode == 0,
            "ASR probe build failed: " +
            result.stdout.decode(errors="replace")[-4000:])
    return binary


def check_real_chat(binary, stage, model, env, root):
    port = free_port()
    api_key = "secret"
    chat_env = {**env, "UMMINFER_API_KEY": api_key}
    ipc = root / "chat"
    ipc.mkdir()
    entry = f"ipc://{ipc}/entry"
    completion = f"ipc://{ipc}/completion"
    abort = f"ipc://{ipc}/abort"
    processes, logs = [], []
    server = spawn("chat_server", [binary, "serve", "--host", "127.0.0.1",
                                   "--port", str(port), "--model", MODEL_NAME,
                                   "--api-key", api_key, "--entry", "mlx",
                                   "--entry-endpoint", entry,
                                   "--completion", completion, "--abort", abort],
                   root, root, env, logs)
    processes.append(server)
    stage_process = spawn("chat_stage", [sys.executable, "-I", str(stage), "--name", "mlx",
                                         "--recv", entry, "--coordinator", completion,
                                         "--abort", abort, "--model", str(model)],
                          stage.parent, root, env, logs)
    processes.append(stage_process)
    try:
        wait_http(port, "/ready", 200, server, "chat server", token=api_key)
        base = run([binary, "chat", "--url", f"http://127.0.0.1:{port}/v1",
                    "--model", MODEL_NAME, "--prompt", PROMPT,
                    "--max-tokens", "16", "--temperature", "0",
                    "--timeout-ms", "60000"], chat_env)
        require(base.returncode == 0 and base.stdout.strip(),
                f"real chat failed: {base.stderr!r}")
        print(f"PASS real_chat output={base.stdout.decode(errors='replace').strip()!r}",
              flush=True)

        for value in ("0", "0.5", "1"):
            result = run([binary, "chat", "--url", f"http://127.0.0.1:{port}/v1",
                          "--model", MODEL_NAME, "--prompt", PROMPT,
                          "--max-tokens", "16", "--temperature", "0",
                          "--top-p", value, "--max-attempts", "1",
                          "--timeout-ms", "60000"], chat_env)
            require(b"unknown option" not in result.stderr,
                    f"--top-p {value} was rejected as an unknown option: "
                    f"{result.stderr!r}")
            # The current server contract does not implement top_p. A success
            # here would let a missing serialization pass as green.
            require(result.returncode == 1,
                    f"--top-p {value} was not rejected by the server: "
                    f"{result.stdout!r} {result.stderr!r}")
            require(b"HTTP 400" in result.stderr and
                    b"top_p is not implemented by this server" in result.stderr,
                    f"--top-p {value} did not reach the server as an unsupported "
                    f"field: {result.stderr!r}")
            print(f"PASS real_top_p value={value} reached_server status=400 "
                  f"unsupported_field=top_p", flush=True)

        # The same unspaced token is a valid bearer and a rejected model. The
        # server authenticates first, then echoes the model in its 400, so the
        # public client must redact the echoed secret. This request is rejected
        # before any model forward.
        reflected = run([binary, "chat", "--url", f"http://127.0.0.1:{port}/v1",
                         "--model", api_key, "--prompt", PROMPT,
                         "--timeout-ms", "2000", "--max-attempts", "1"], chat_env)
        require(reflected.returncode == 1 and b"HTTP 400" in reflected.stderr,
                f"reflected-key chat stderr was {reflected.stderr!r}")
        require(b"unsupported model: [key]" in reflected.stderr,
                f"reflected-key chat was not redacted: {reflected.stderr!r}")
        require(api_key.encode() not in reflected.stderr,
                f"reflected API key leaked: {reflected.stderr!r}")
        print("PASS chat_response_echo_redacted status=400 model=[key] no_leak",
              flush=True)

        short_key = "xy"
        auth = run([binary, "chat", "--url", f"http://127.0.0.1:{port}/v1",
                    "--model", MODEL_NAME, "--prompt", PROMPT,
                    "--timeout-ms", "2000", "--max-attempts", "1"],
                   {**chat_env, "UMMINFER_API_KEY": short_key})
        require(auth.returncode == 1 and b"HTTP 401" in auth.stderr,
                f"short-key auth stderr was {auth.stderr!r}")
        require(short_key.encode() not in auth.stderr,
                f"short API key leaked into error text: {auth.stderr!r}")
        # This proves only that the first-party server does not echo headers.
        # It does not exercise client redaction of an echoing response.
        no_key_env = {k: v for k, v in chat_env.items()
                      if k not in ("UMMINFER_API_KEY", "OPENAI_API_KEY")}
        missing = run([binary, "chat", "--url", f"http://127.0.0.1:{port}/v1",
                       "--model", MODEL_NAME, "--prompt", PROMPT,
                       "--timeout-ms", "2000", "--max-attempts", "1"], no_key_env)
        require(missing.returncode == 1 and b"HTTP 401" in missing.stderr,
                f"missing-key chat stderr was {missing.stderr!r}")
        require(api_key.encode() not in missing.stderr,
                f"missing-key error leaked the server key: {missing.stderr!r}")
        print("PASS chat_missing_key status=401 no_leak", flush=True)
        print("PASS first_party_401 does_not_echo_request_key", flush=True)
    finally:
        errors = cleanup(processes, logs, root)
        if errors:
            raise RuntimeError("; ".join(errors))


def generate_audio(path):
    sample_rate = 16000
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        frames = []
        for start in range(0, sample_rate // 2, 400):
            samples = [int(0.2 * 32767 *
                           math.sin(2 * math.pi * 440 * index / sample_rate))
                       for index in range(start, min(start + 400, sample_rate // 2))]
            frames.extend(samples)
        audio.writeframes(struct.pack(f"<{len(frames)}h", *frames))


def check_real_asr(binary, asr_stage, env, root, probe):
    port = free_port()
    ipc = root / "asr"
    ipc.mkdir()
    model_dir = root / "asr_model"
    audio = root / "audio.wav"
    generated = run([sys.executable, "-I", str(asr_stage), "generate",
                     "--output", str(model_dir)], env, timeout=STARTUP_TIMEOUT_S)
    require(generated.returncode == 0,
            f"tiny ASR model generation failed: {generated.stderr!r}")
    generate_audio(audio)
    entry = f"ipc://{ipc}/entry"
    completion = f"ipc://{ipc}/completion"
    abort = f"ipc://{ipc}/abort"
    processes, logs = [], []
    server = spawn("asr_server", [binary, "serve", "--host", "127.0.0.1",
                                  "--port", str(port), "--model", ASR_MODEL_NAME,
                                  "--entry", "asr", "--entry-endpoint", entry,
                                  "--completion", completion, "--abort", abort],
                   root, root, env, logs)
    processes.append(server)
    stage_process = spawn("asr_stage", [sys.executable, "-I", str(asr_stage), "serve",
                                        "--name", "asr", "--recv", entry,
                                        "--coordinator", completion, "--abort", abort,
                                        "--model", str(model_dir)],
                          asr_stage.parent, root, env, logs)
    processes.append(stage_process)
    try:
        wait_http(port, "/ready", 200, server, "asr server")
        result = run([binary, "transcribe", "--url", f"http://127.0.0.1:{port}",
                      "--model", ASR_MODEL_NAME, "--file", str(audio),
                      "--timeout-ms", "20000"], env)
        require(result.returncode == 0 and result.stdout == b"b\n",
                f"real transcribe failed: {result.stdout!r} {result.stderr!r}")
        print("PASS real_asr_transcribe output=b", flush=True)

        # One valid PCM16 sample is structurally valid audio. The real TinyCTC
        # stage rejects it after the request reaches its forward path, and the
        # route returns that real failure as HTTP 500. The public ASRClient must
        # map it to kServer/500/retryable.
        pcm_path = root / "asr_audio.pcm"
        with wave.open(str(audio), "rb") as wav:
            pcm_path.write_bytes(wav.readframes(wav.getnframes()))
        short_path = root / "short.pcm"
        short_path.write_bytes(struct.pack("<h", 1000))
        failed = run([str(probe), f"http://127.0.0.1:{port}", ASR_MODEL_NAME,
                      str(short_path)])
        failure = json.loads(failed.stdout)
        require(failure.get("ok") is False and failure.get("code") == "kServer" and
                failure.get("status") == 500 and failure.get("retryable") is True,
                f"short-frame ASR did not map to kServer/500/retryable: {failure!r}")
        require("audio is shorter than one analysis frame" in failure.get("what", ""),
                f"short-frame ASR lost the real stage reason: {failure!r}")
        wait_in_flight_zero(port)
        print("PASS real_asr_client_500 code=kServer status=500 retryable=true "
              "reason=short_frame", flush=True)

        # Recovery control: the same real stage and server accept valid audio
        # and run a real TinyCTC forward after the failed request.
        recovered = run([str(probe), f"http://127.0.0.1:{port}", ASR_MODEL_NAME,
                         str(pcm_path)])
        recovered_result = json.loads(recovered.stdout)
        require(recovered_result.get("ok") is True and
                recovered_result.get("text") == "b",
                f"valid-audio ASR recovery failed: {recovered_result!r}")
        wait_in_flight_zero(port)
        print("PASS real_asr_client_recovery text=b in_flight=0", flush=True)

        # --chunk-ms must be validated before any network work. The URL is the
        # real HTTP listener of this running server, so a parser that runs after
        # the WebSocket attempt fails the upgrade instead of exiting with usage.
        for value in ("0", "-100", "abc"):
            result = run([binary, "realtime", "--url", f"http://127.0.0.1:{port}",
                          "--model", ASR_MODEL_NAME, "--file", str(audio),
                          "--chunk-ms", value],
                         {**env, "UMMINFER_API_KEY": "boundary-secret"}, timeout=30)
            require(result.returncode == 2,
                    f"--chunk-ms {value} exit {result.returncode}: {result.stderr!r}")
            require(f"invalid --chunk-ms: {value}".encode() in result.stderr,
                    f"--chunk-ms {value} stderr was {result.stderr!r}")
            require(b"boundary-secret" not in result.stderr,
                    f"--chunk-ms {value} leaked the environment key: {result.stderr!r}")
            print(f"PASS realtime_chunk_ms_{value} exit=2 parse_before_network",
                  flush=True)
    finally:
        errors = cleanup(processes, logs, root)
        if errors:
            raise RuntimeError("; ".join(errors))


def check_real_503(binary, asr_stage, env, root, probe):
    """A real invalid entry endpoint makes zmq_connect fail after readiness."""
    port = free_port()
    ipc = root / "entry503"
    ipc.mkdir()
    model_dir = root / "entry503_model"
    generated = run([sys.executable, "-I", str(asr_stage), "generate",
                     "--output", str(model_dir)], env, timeout=STARTUP_TIMEOUT_S)
    require(generated.returncode == 0,
            f"503 tiny ASR model generation failed: {generated.stderr!r}")
    pcm_path = root / "entry503_audio.pcm"
    samples = [int(0.2 * 32767 * math.sin(2 * math.pi * 440 * index / 16000))
               for index in range(8000)]
    pcm_path.write_bytes(struct.pack(f"<{len(samples)}h", *samples))
    completion = f"ipc://{ipc}/completion"
    abort = f"ipc://{ipc}/abort"
    processes, logs = [], []
    # The entry endpoint is a real but unsupported transport. The server passes
    # it to the Coordinator unchanged; the stage still announces readiness on
    # its own valid receive endpoint, so /ready becomes 200.
    server = spawn("entry503_server", [binary, "serve", "--host", "127.0.0.1",
                                       "--port", str(port), "--model", ASR_MODEL_NAME,
                                       "--entry", "asr",
                                       "--entry-endpoint", "invalid://umminfer-entry",
                                       "--completion", completion, "--abort", abort],
                   root, root, env, logs)
    processes.append(server)
    stage_process = spawn("entry503_stage", [sys.executable, "-I", str(asr_stage),
                                             "serve", "--name", "asr",
                                             "--recv", f"ipc://{ipc}/stage-in",
                                             "--coordinator", completion,
                                             "--abort", abort,
                                             "--model", str(model_dir)],
                          asr_stage.parent, root, env, logs)
    processes.append(stage_process)
    try:
        wait_http(port, "/ready", 200, server, "503 server")
        result = run([str(probe), f"http://127.0.0.1:{port}", ASR_MODEL_NAME,
                      str(pcm_path)])
        failure = json.loads(result.stdout)
        require(failure.get("ok") is False and failure.get("code") == "kServer" and
                failure.get("status") == 503 and failure.get("retryable") is True,
                f"invalid entry endpoint did not map to kServer/503/retryable: "
                f"{failure!r}")
        require("zmq_connect failed" in failure.get("what", ""),
                f"503 lost the real zmq_connect reason: {failure!r}")
        wait_in_flight_zero(port)
        print("PASS real_asr_client_503 code=kServer status=503 retryable=true "
              "reason=zmq_connect in_flight=0", flush=True)
    finally:
        errors = cleanup(processes, logs, root)
        if errors:
            raise RuntimeError("; ".join(errors))


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
    parser.add_argument("--skip-real", action="store_true",
                        help="run only the CLI parsing and URL policy cases")
    parser.add_argument("--project", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--deps-prefix", type=Path, default=DEPS_PREFIX)
    parser.add_argument("--curl-prefix", type=Path, default=CURL_PREFIX)
    args = parser.parse_args()
    binary = args.binary.resolve()
    stage = args.stage.resolve()
    model = args.model.resolve()
    asr_stage = args.asr_stage.resolve()
    project = args.project.resolve()
    deps_prefix = args.deps_prefix.resolve()
    curl_prefix = args.curl_prefix.resolve()
    require((deps_prefix / "include/nlohmann/json.hpp").is_file(),
            f"nlohmann headers not found under {deps_prefix}")
    require((curl_prefix / "include/curl/curl.h").is_file(),
            f"curl headers not found under {curl_prefix}")
    require(binary.is_file(), f"binary not found: {binary}")
    with binary.open("rb") as source:
        print(f"binary_sha256={hashlib.file_digest(source, 'sha256').hexdigest()}",
              flush=True)

    check_chat_parsing(binary)
    check_server_parsing(binary)
    check_url_policy(binary)

    if args.skip_real:
        print("SKIP real chat and ASR regression (--skip-real)", flush=True)
        return
    for path, label in ((stage, "stage"), (model, "model"), (asr_stage, "ASR stage")):
        require(path.exists(), f"{label} not found: {path}")
    env = offline_env()
    with tempfile.TemporaryDirectory(prefix="umminfer-boundary-") as temp:
        root = Path(temp)
        check_asr_parsing(binary, root)
        probe = compile_asr_probe(project, root, deps_prefix, curl_prefix)
        check_stage_multi_value(binary, root)
        check_real_chat(binary, stage, model, env, root)
        check_real_asr(binary, asr_stage, env, root, probe)
        check_real_503(binary, asr_stage, env, root, probe)
    print("NOT_COVERED other_501_502_public_client: upstream route status "
          "mapping is under change in Batch 2; this binary snapshot is not "
          "evaluated for a real 501 or 502 response.", flush=True)
    print("PASS client_boundary_e2e", flush=True)


if __name__ == "__main__":
    main()

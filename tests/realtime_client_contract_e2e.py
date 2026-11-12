#!/usr/bin/env python3
"""Verify real realtime client contracts with ummInfer and TinyASR inference."""

import argparse
import base64
import hashlib
import http.client
import importlib.util
import json
import math
import os
from pathlib import Path
import queue
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import wave


LOG_CAP = 65536
CLI_CAP = 65536
# The probe links the public client against the same libcurl the build uses.
# Override the prefix when the library lives elsewhere.
DEPS_PREFIX = Path(os.environ.get("UMMINFER_DEPS_PREFIX", "/opt/homebrew"))
CURL_PREFIX = Path(os.environ.get("UMMINFER_CURL_PREFIX", "/opt/homebrew/opt/curl"))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def http_json(port, path):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        conn.request("GET", path)
        response = conn.getresponse()
        body = response.read(4097)
        require(response.status == 200 and len(body) <= 4096,
                f"{path}: HTTP {response.status}, body={body!r}")
        return json.loads(body)
    finally:
        conn.close()


def in_flight(port):
    return http_json(port, "/health")["in_flight"]


def wait_for(predicate, process=None, timeout_s=10, interval_s=0.01):
    for _ in range(max(1, int(timeout_s / interval_s))):
        try:
            if predicate():
                return
        except (ConnectionError, OSError, http.client.HTTPException):
            pass
        if process is not None and process.poll() is not None:
            raise AssertionError(f"process exited early with {process.returncode}")
        time.sleep(interval_s)
    raise AssertionError(f"condition was not met within {timeout_s} seconds")


def spawn(name, args, cwd, env, processes, logs, readers, ready=None, abort_events=None,
          submit_events=None):
    proc = subprocess.Popen(args, cwd=cwd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    processes[name] = proc
    logs[name] = bytearray()

    def drain():
        while True:
            chunk = proc.stdout.readline(4096)
            if not chunk:
                return
            remaining = LOG_CAP - len(logs[name])
            if remaining > 0:
                logs[name].extend(chunk[:remaining])
            if ready is not None and b"MLX stage asr ready" in chunk:
                ready.set()
            marker = b"OBSERVED_ABORT "
            if abort_events is not None and marker in chunk:
                request_id = chunk.split(marker, 1)[1].strip().decode("utf-8")
                abort_events.put_nowait(request_id)
            marker = b"OBSERVED_SUBMIT "
            if submit_events is not None and marker in chunk:
                request_id = chunk.split(marker, 1)[1].strip().decode("utf-8")
                submit_events.put_nowait(request_id)

    thread = threading.Thread(target=drain, name=f"{name}-log", daemon=True)
    thread.start()
    readers.append(thread)
    return proc


def start_cli(args, env, children):
    proc = subprocess.Popen(args, env=env.copy(), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    children.append(proc)
    return proc


def finish_cli(proc, expected=0, timeout=20):
    try:
        output, error = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        output, error = proc.communicate(timeout=5)
        raise AssertionError(f"CLI timed out: {error!r}")
    require(len(output) <= CLI_CAP and len(error) <= CLI_CAP, "CLI output exceeded cap")
    require(proc.returncode == expected,
            f"CLI exit {proc.returncode}, expected {expected}: {error!r}")
    return output, error


def wait_abort(events, expected, timeout_s=10):
    for _ in range(int(timeout_s * 100)):
        try:
            actual = events.get(timeout=0.01)
        except queue.Empty:
            continue
        require(actual == expected, f"abort id {actual!r}, expected {expected!r}")
        return actual
    raise AssertionError(f"did not observe real stage abort {expected}")


def wait_submit(events, expected, timeout_s=10):
    observed = []
    for _ in range(int(timeout_s * 100)):
        try:
            actual = events.get(timeout=0.01)
        except queue.Empty:
            continue
        observed.append(actual)
        if actual == expected:
            return actual
    raise AssertionError(f"did not observe real stage submit {expected}; saw {observed}")



def run_observed_stage(argv):
    parser = argparse.ArgumentParser(description="Run real TinyASR with abort-call logging")
    parser.add_argument("stage", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--recv", required=True)
    parser.add_argument("--coordinator", required=True)
    parser.add_argument("--abort", required=True)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--omit-metadata-request", action="append")
    args = parser.parse_args(argv)
    spec = importlib.util.spec_from_file_location("umminfer_tiny_asr_stage", args.stage.resolve())
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class ObservedTinyASRStage(module.TinyASRStage):
        def process(self, message):
            print(f"OBSERVED_SUBMIT {message.get('request_id')}", flush=True)
            return super().process(message)

        def record_abort(self, request_id):
            present = request_id in self.aborted
            super().record_abort(request_id)
            if not present and request_id in self.aborted:
                print(f"OBSERVED_ABORT {request_id}", flush=True)

        def complete(self, request_id, success, text="", error=None,
                     generated_tokens=None, finish_reason=None):
            if request_id in (args.omit_metadata_request or []):
                generated_tokens = None
                finish_reason = None
                print(f"OBSERVED_METADATA_OMIT {request_id}", flush=True)
            return super().complete(request_id, success, text, error,
                                    generated_tokens, finish_reason)

    stage = ObservedTinyASRStage(args.name, args.recv, args.coordinator, args.abort,
                                 str(args.model.resolve()))
    stage.start()
    stage.run()


def take_frame(buffer):
    if len(buffer) < 2:
        return None
    first, second = buffer[0], buffer[1]
    length = second & 0x7f
    offset = 2
    if length == 126:
        if len(buffer) < 4:
            return None
        length = int.from_bytes(buffer[2:4], "big")
        offset = 4
    elif length == 127:
        if len(buffer) < 10:
            return None
        length = int.from_bytes(buffer[2:10], "big")
        offset = 10
    masked = bool(second & 0x80)
    if masked:
        offset += 4
    require(length <= 4 * 1024 * 1024, "WebSocket frame exceeded capture cap")
    total = offset + length
    if len(buffer) < total:
        return None
    raw = bytes(buffer[:total])
    payload = bytes(buffer[offset:total])
    if masked:
        key = bytes(buffer[offset - 4:offset])
        payload = bytes(value ^ key[index % 4] for index, value in enumerate(payload))
    del buffer[:total]
    return bool(first & 0x80), first & 0x0f, payload, raw


def encode_frame(fin, opcode, payload):
    first = opcode | (0x80 if fin else 0)
    if len(payload) < 126:
        header = bytes((first, len(payload)))
    elif len(payload) <= 65535:
        header = bytes((first, 126)) + len(payload).to_bytes(2, "big")
    else:
        header = bytes((first, 127)) + len(payload).to_bytes(8, "big")
    return header + payload


class FrameParser:
    def __init__(self, proxy, direction):
        self.proxy = proxy
        self.direction = direction
        self.buffer = bytearray()
        self.upgraded = False
        self.message_opcode = None
        self.message_parts = []

    def feed(self, data):
        self.buffer.extend(data)
        if not self.upgraded:
            end = self.buffer.find(b"\r\n\r\n")
            if end < 0:
                require(len(self.buffer) <= 65536, "WebSocket upgrade headers exceeded cap")
                return
            del self.buffer[:end + 4]
            self.upgraded = True
        for _ in range(1024):
            frame = take_frame(self.buffer)
            if frame is None:
                return
            fin, opcode, payload, _ = frame
            self.proxy.record(self.direction, fin, opcode, payload)
            if self.direction != "client" or opcode >= 8:
                continue
            if opcode in (1, 2):
                require(self.message_opcode is None, "nested client WebSocket message")
                self.message_opcode = opcode
                self.message_parts = []
            else:
                require(opcode == 0 and self.message_opcode is not None,
                        "unexpected client continuation frame")
            self.message_parts.append(payload)
            require(sum(map(len, self.message_parts)) <= 4 * 1024 * 1024,
                    "client WebSocket message exceeded capture cap")
            if fin:
                message = b"".join(self.message_parts)
                if self.message_opcode == 1:
                    self.proxy.observe_client_message(message)
                self.message_opcode = None
                self.message_parts = []


class ServerFragmenter:
    """Refragment genuine server data frames without injecting wire bytes."""

    def __init__(self, proxy):
        self.proxy = proxy
        self.buffer = bytearray()
        self.upgraded = False

    def emit(self, frame):
        fin, opcode, payload, raw = frame
        if not fin or opcode not in (1, 2):
            self.proxy.record("server", fin, opcode, payload)
            return raw
        if len(payload) >= 3:
            cuts = (len(payload) // 3, 2 * len(payload) // 3)
        elif len(payload) == 2:
            cuts = (1, 1)
        else:
            cuts = (0, 0)
        parts = (payload[:cuts[0]], payload[cuts[0]:cuts[1]], payload[cuts[1]:])
        output = bytearray()
        for index, part in enumerate(parts):
            part_fin = index == 2
            part_opcode = opcode if index == 0 else 0
            self.proxy.record("server", part_fin, part_opcode, part)
            output.extend(encode_frame(part_fin, part_opcode, part))
        self.proxy.fragmented_messages += 1
        return bytes(output)

    def feed(self, data):
        output = bytearray()
        self.buffer.extend(data)
        if not self.upgraded:
            end = self.buffer.find(b"\r\n\r\n")
            if end < 0:
                require(len(self.buffer) <= 65536, "WebSocket upgrade headers exceeded cap")
                return b""
            output.extend(self.buffer[:end + 4])
            del self.buffer[:end + 4]
            self.upgraded = True
        for _ in range(1024):
            frame = take_frame(self.buffer)
            if frame is None:
                break
            output.extend(self.emit(frame))
        return bytes(output)


class WireProxy:
    """Forward one real connection; optionally refragment genuine server frames."""

    def __init__(self, upstream_port, fragment=False):
        self.upstream_port = upstream_port
        self.fragment = fragment
        self.listener = socket.socket()
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(1)
        self.listener.settimeout(0.1)
        self.port = self.listener.getsockname()[1]
        self.events = []
        self.frame_events = []
        self.errors = []
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.connections = []
        self.forwarders = []
        self.fragmented_messages = 0
        self.append_message_sizes = []
        self.append_pcm_sizes = []
        self.append_sha256 = hashlib.sha256()
        self.session_updates = []
        self.server_text_messages = []
        self.thread = threading.Thread(target=self.accept, name="ws-wire-accept", daemon=True)
        self.thread.start()

    def record(self, direction, fin, opcode, payload):
        with self.lock:
            require(len(self.frame_events) < 10000, "wire event cap reached")
            self.frame_events.append((direction, opcode, fin))
            if direction == "server" and opcode == 1 and fin:
                self.server_text_messages.append(payload)
            if opcode == 8:
                code = int.from_bytes(payload[:2], "big") if len(payload) >= 2 else None
                self.events.append((direction, code))

    def observe_client_message(self, message):
        event = json.loads(message)
        if event.get("type") == "session.update":
            self.session_updates.append(message)
            return
        if event.get("type") != "input_audio_buffer.append":
            return
        pcm = base64.b64decode(event["audio"], validate=True)
        require(len(message) <= 1 << 20,
                f"append JSON exceeds 1 MiB: {len(message)}")
        require(pcm and len(pcm) % 2 == 0, "append does not contain whole PCM16 samples")
        self.append_message_sizes.append(len(message))
        self.append_pcm_sizes.append(len(pcm))
        self.append_sha256.update(pcm)

    def accept(self):
        try:
            for _ in range(200):
                if self.stopping.is_set():
                    return
                try:
                    client, _ = self.listener.accept()
                    break
                except socket.timeout:
                    continue
            else:
                self.errors.append("proxy accept timeout")
                return
            upstream = socket.create_connection(("127.0.0.1", self.upstream_port), timeout=5)
            client.settimeout(None)
            upstream.settimeout(None)
            self.connections = [client, upstream]
            parsers = {"client": FrameParser(self, "client")}
            parsers["server"] = (ServerFragmenter(self) if self.fragment
                                 else FrameParser(self, "server"))
            for source, destination, direction in (
                    (client, upstream, "client"), (upstream, client, "server")):
                thread = threading.Thread(target=self.forward,
                                          args=(source, destination, direction, parsers[direction]),
                                          name=f"ws-wire-{direction}", daemon=True)
                self.forwarders.append(thread)
                thread.start()
        except Exception as exc:
            if not self.stopping.is_set():
                self.errors.append(f"proxy accept: {exc}")

    def forward(self, source, destination, direction, parser):
        try:
            for _ in range(100000):
                data = source.recv(65536)
                if not data:
                    destination.shutdown(socket.SHUT_WR)
                    return
                if direction == "server" and self.fragment:
                    outgoing = parser.feed(data)
                    if outgoing:
                        destination.sendall(outgoing)
                else:
                    parser.feed(data)
                    destination.sendall(data)
            self.errors.append("proxy forwarding iteration cap reached")
        except Exception as exc:
            if not self.stopping.is_set():
                self.errors.append(f"proxy forward: {exc}")

    def close(self):
        self.stopping.set()
        try:
            self.listener.close()
        except OSError:
            pass
        for connection in self.connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                connection.close()
            except OSError:
                pass
        self.thread.join(timeout=2)
        for thread in self.forwarders:
            thread.join(timeout=2)
        require(not self.thread.is_alive() and all(not thread.is_alive()
                                                   for thread in self.forwarders),
                "proxy threads did not exit")
        require(not self.errors, f"proxy errors: {self.errors}")
        with self.lock:
            return list(self.events)


def verify_handshake(events, client_first):
    client = [index for index, event in enumerate(events) if event == ("client", 1000)]
    server = [index for index, event in enumerate(events) if event == ("server", 1000)]
    require(client and server, f"incomplete wire close handshake: {events}")
    if client_first:
        require(client[0] < server[0], f"server close preceded client close: {events}")


PROBE_SOURCE = r'''
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>
#include "umminfer/client/client.hpp"

using namespace umminfer::client;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void live_case(const char* url, const char* path) {
  std::ifstream input(path, std::ios::binary);
  require(input.good(), "cannot open PCM input");
  std::string pcm((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
  require(!pcm.empty() && pcm.size() <= 2 * 1024 * 1024 && pcm.size() % 2 == 0,
          "invalid PCM probe input");
  ASRConfig config;
  config.base_url = url;
  config.model = "tiny-ctc";
  config.connect_timeout = Milliseconds{5000};
  config.stall_timeout = Milliseconds{30000};
  RealtimeRequest request;
  request.live_result = true;
  request.final_timeout = Milliseconds{30000};
  int partial_calls = 0;
  request.on_partial = [&](const std::string&) { ++partial_calls; };
  auto session = ASRClient(config).realtime(request);
  session->send(pcm);
  const auto response = session->finish();
  nlohmann::json out = {{"text", response.text},
                        {"finish_reason", response.finish_reason},
                        {"partial_calls", partial_calls}};
  out["generated_tokens"] = response.generated_tokens
      ? nlohmann::json(*response.generated_tokens) : nlohmann::json(nullptr);
  out["length_limited"] = response.length_limited
      ? nlohmann::json(*response.length_limited) : nlohmann::json(nullptr);
  std::cout << out.dump() << '\n';
}
'''

PROBE_MAIN = r'''
int main(int argc, char** argv) {
  try {
    if (argc == 4 && std::string_view(argv[1]) == "live") {
      live_case(argv[2], argv[3]);
      return 0;
    }
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "probe error: " << error.what() << '\n';
    return 1;
  }
}
'''


def compile_probe(project, work, deps_prefix, curl_prefix):
    source = work / "realtime_client_probe.cpp"
    binary = work / "realtime_client_probe"
    source.write_text(PROBE_SOURCE + PROBE_MAIN, encoding="utf-8")
    command = ["c++", "-std=c++20", "-O1", str(source),
               str(project / "src/audio/audio.cpp"),
               str(project / "src/client/http.cpp"),
               str(project / "src/client/asr.cpp"),
               str(project / "src/client/websocket.cpp"),
               str(project / "src/client/realtime.cpp"),
               "-I", str(project / "include"), "-I", str(project / "src/client"),
               "-I", str(deps_prefix / "include"),
               "-I", str(curl_prefix / "include"), "-L", str(curl_prefix / "lib"),
               "-Wl,-rpath," + str(curl_prefix / "lib"), "-lcurl", "-pthread",
               "-o", str(binary)]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=120)
    require(len(result.stdout) <= LOG_CAP, "probe compiler output exceeded cap")
    require(result.returncode == 0,
            f"probe build failed: {result.stdout.decode(errors='replace')}")
    return binary


def realtime_command(binary, ws_port, audio, final_port=None, final_timeout=5000):
    command = [str(binary), "realtime", "--url", f"http://127.0.0.1:{ws_port}",
               "--model", "tiny-ctc", "--file", str(audio), "--live-result",
               "--chunk-ms", "100", "--final-timeout-ms", str(final_timeout),
               "--stall-ms", "20000"]
    if final_port is not None:
        command.extend(["--final-url", f"http://127.0.0.1:{final_port}"])
    return command


def transcribe_command(binary, http_port, audio):
    return [str(binary), "transcribe", "--url", f"http://127.0.0.1:{http_port}",
            "--model", "tiny-ctc", "--file", str(audio), "--timeout-ms", "20000"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--project", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--deps-prefix", type=Path, default=DEPS_PREFIX)
    parser.add_argument("--curl-prefix", type=Path, default=CURL_PREFIX)
    args = parser.parse_args()
    binary = args.binary.resolve()
    project = args.project.resolve()
    require(binary.is_file() and os.access(binary, os.X_OK), f"not executable: {binary}")
    tiny_stage = project / "python/umminfer_tiny_asr_stage.py"
    require(tiny_stage.is_file(), f"missing stage: {tiny_stage}")
    require((args.deps_prefix / "include/nlohmann/json.hpp").is_file(),
            f"nlohmann headers not found under {args.deps_prefix}")
    require((args.curl_prefix / "include/curl/curl.h").is_file(),
            f"curl headers not found under {args.curl_prefix}")
    print(f"binary_sha256={hashlib.sha256(binary.read_bytes()).hexdigest()}", flush=True)

    processes = {}
    logs = {}
    readers = []
    children = []
    proxies = []
    abort_events = queue.Queue(maxsize=128)
    submit_events = queue.Queue(maxsize=128)
    stage_stopped = False
    succeeded = False

    def resume_stage():
        nonlocal stage_stopped
        if stage_stopped:
            os.kill(processes["stage"].pid, signal.SIGCONT)
            stage_stopped = False

    def stop_stage():
        nonlocal stage_stopped
        os.kill(processes["stage"].pid, signal.SIGSTOP)
        stage_stopped = True

    try:
        with tempfile.TemporaryDirectory(prefix="umminfer-rt-client-", dir="/tmp") as temporary:
            work = Path(temporary)
            ipc = work / "ipc"
            ipc.mkdir()
            subprocess.run([sys.executable, "-I", str(tiny_stage), "generate",
                            "--output", str(work / "model")], check=True,
                           capture_output=True, timeout=60)
            audio = work / "audio.wav"
            samples = [int(0.2 * 32767 * math.sin(2 * math.pi * 440 * index / 16000))
                       for index in range(8000)]
            base_pcm = struct.pack(f"<{len(samples)}h", *samples)
            with wave.open(str(audio), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(base_pcm)
            raw_audio = work / "audio.pcm"
            raw_audio.write_bytes(base_pcm)
            large_audio = work / "large.pcm"
            large_pcm = (base_pcm * ((900000 + len(base_pcm) - 1) // len(base_pcm)))[:900000]
            large_audio.write_bytes(large_pcm)
            # The CLI --file path decodes a container, so the cancel case needs
            # a WAV of the same length rather than the raw probe stream.
            large_wav = work / "large.wav"
            with wave.open(str(large_wav), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(large_pcm)
            probe = compile_probe(project, work, args.deps_prefix.resolve(),
                                  args.curl_prefix.resolve())

            http_port = free_port()
            ws_port = free_port()
            require(http_port != ws_port, "HTTP and WS ports are equal")
            entry = f"ipc://{ipc}/entry"
            completion = f"ipc://{ipc}/completion"
            abort_endpoint = f"ipc://{ipc}/abort"
            env = os.environ.copy()
            env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                        "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"})
            spawn("server", [str(binary), "serve", "--host", "127.0.0.1",
                             "--port", str(http_port), "--realtime-port", str(ws_port),
                             "--model", "tiny-ctc", "--entry", "asr",
                             "--entry-endpoint", entry, "--completion", completion,
                             "--abort", abort_endpoint], binary.parent, env,
                  processes, logs, readers)
            wait_for(lambda: in_flight(http_port) == 0, processes["server"], 30)
            stage_ready = threading.Event()
            spawn("stage", [sys.executable, "-I", str(Path(__file__).resolve()),
                            "--observed-stage", str(tiny_stage), "--name", "asr",
                            "--recv", entry, "--coordinator", completion,
                            "--abort", abort_endpoint, "--model", str(work / "model"),
                            "--omit-metadata-request", "realtime-5",
                            "--omit-metadata-request", "req-4"],
                  project, env, processes, logs, readers, stage_ready, abort_events, submit_events)
            wait_for(stage_ready.is_set, processes["stage"], 120)
            print(f"PASS real_startup http_port={http_port} ws_port={ws_port} "
                  f"stage_pid={processes['stage'].pid}", flush=True)

            for option in ("--max-new-tokens", "--final-timeout-ms", "--stall-ms"):
                rejected = start_cli([str(binary), "realtime", "--url",
                                      f"http://127.0.0.1:{ws_port}", "--model", "tiny-ctc",
                                      "--file", str(audio), option, "-1"], env, children)
                _, error = finish_cli(rejected, expected=2)
                require(f"invalid {option}".encode() in error,
                        f"negative {option} was not explicitly rejected: {error!r}")
            print("PASS cli_negative_options_rejected cases=3 exit_code=2", flush=True)

            layout_proxy = WireProxy(ws_port)
            proxies.append(layout_proxy)
            layout_command = realtime_command(binary, layout_proxy.port, audio) + [
                "--audio-layout", "stereo"]
            layout = start_cli(layout_command, env, children)
            _, layout_error = finish_cli(layout, expected=1)
            layout_proxy.close()
            require(any(b'"audio_layout":"stereo"' in message
                        for message in layout_proxy.session_updates),
                    f"CLI omitted audio_layout: {layout_proxy.session_updates!r}")
            require(any(b"audio_layout is not supported" in message
                        for message in layout_proxy.server_text_messages),
                    f"server did not reject stereo: {layout_proxy.server_text_messages!r}; "
                    f"CLI error={layout_error!r}")
            require(b"audio_layout is not supported" in layout_error,
                    f"CLI did not report the server rejection: {layout_error!r}")
            print("PASS real_audio_layout_forwarded_and_rejected layout=stereo", flush=True)

            active_proxy = WireProxy(ws_port)
            proxies.append(active_proxy)
            active = start_cli(realtime_command(binary, active_proxy.port, audio, http_port, 1),
                               env, children)
            active_output, _ = finish_cli(active)
            active_events = active_proxy.close()
            require(active_output == b"b\n", f"unexpected Final output: {active_output!r}")
            active_abort = wait_abort(abort_events, "realtime-1")
            final_submit = wait_submit(submit_events, "req-1")
            verify_handshake(active_events, client_first=True)
            wait_for(lambda: in_flight(http_port) == 0, timeout_s=10)
            print(f"PASS real_active_close_handshake wire_events={active_events} "
                  f"abort={active_abort} final_output=b in_flight=0", flush=True)
            print(f"PASS real_live_timeout_http_final_fallback final_submit={final_submit} "
                  "live_result=true final_timeout_ms=1 output=b", flush=True)

            after_active = start_cli(transcribe_command(binary, http_port, audio), env, children)
            require(finish_cli(after_active)[0] == b"b\n", "request after active close failed")
            print("PASS real_request_after_active_close output=b in_flight=0", flush=True)

            stop_stage()
            kill_proxy = WireProxy(ws_port)
            proxies.append(kill_proxy)
            killed = start_cli(realtime_command(binary, kill_proxy.port, audio), env, children)
            wait_for(lambda: in_flight(http_port) == 1, killed, 10, 0.001)
            killed.kill()
            require(killed.wait(timeout=5) == -signal.SIGKILL, "SIGKILL exit mismatch")
            wait_for(lambda: in_flight(http_port) == 0, timeout_s=10)
            resume_stage()
            killed_abort = wait_abort(abort_events, "realtime-2")
            killed_events = kill_proxy.close()
            require(all(direction != "client" for direction, _ in killed_events),
                    f"SIGKILL produced a normal client Close: {killed_events}")
            print(f"PASS real_sigkill_disconnect abort={killed_abort} "
                  "expected_returncode=-9 in_flight=0", flush=True)

            after_kill = start_cli(transcribe_command(binary, http_port, audio), env, children)
            require(finish_cli(after_kill)[0] == b"b\n", "request after SIGKILL failed")
            print("PASS real_request_after_sigkill output=b in_flight=0", flush=True)

            live_proxy = WireProxy(ws_port, fragment=True)
            proxies.append(live_proxy)
            live = start_cli(realtime_command(binary, live_proxy.port, audio), env, children)
            try:
                live_output, live_error = finish_cli(live)
            except Exception:
                print(f"DIAG normal_live_wire_events={live_proxy.close()}", flush=True)
                raise
            require(live_output == b"b\n", "normal live result failed")
            require(b"partial:" not in live_error,
                    f"final segment called on_partial or duplicated output: {live_error!r}")
            live_events = live_proxy.close()
            verify_handshake(live_events, client_first=False)
            require(live_proxy.fragmented_messages >= 3,
                    f"too few real messages fragmented: {live_proxy.fragmented_messages}")
            print(f"PASS real_continuation_fragments "
                  f"fragmented_messages={live_proxy.fragmented_messages} "
                  f"wire_events={live_events}", flush=True)
            print("NOT_COVERED real_control_frame_interleaving: no natural real-client "
                  "trigger without injected frames", flush=True)
            wait_for(lambda: in_flight(http_port) == 0, timeout_s=10)
            print("PASS real_final_segment_not_partial output=b "
                  "stderr_no_partial=true in_flight=0", flush=True)

            append_proxy = WireProxy(ws_port)
            proxies.append(append_proxy)
            appended = start_cli([str(probe), "live",
                                  f"http://127.0.0.1:{append_proxy.port}", str(large_audio)],
                                 env, children)
            append_output, _ = finish_cli(appended, timeout=60)
            append_result = json.loads(append_output)
            append_events = append_proxy.close()
            verify_handshake(append_events, client_first=False)
            require(len(append_proxy.append_message_sizes) >= 2,
                    f"large send was not split: {append_proxy.append_message_sizes}")
            require(all(0 < size <= 1 << 20 for size in append_proxy.append_message_sizes),
                    f"append JSON size violation: {append_proxy.append_message_sizes}")
            require(all(size > 0 and size % 2 == 0 for size in append_proxy.append_pcm_sizes),
                    f"append PCM sample boundary violation: {append_proxy.append_pcm_sizes}")
            require(sum(append_proxy.append_pcm_sizes) == len(large_pcm) and
                    append_proxy.append_sha256.hexdigest() == hashlib.sha256(large_pcm).hexdigest(),
                    "append chunks changed the PCM byte stream")
            require(append_result.get("text") and
                    append_result.get("generated_tokens") == 1 and
                    append_result.get("length_limited") is False and
                    append_result.get("finish_reason") == "stop" and
                    append_result.get("partial_calls") == 0,
                    f"real append result or metadata mismatch: {append_result!r}")
            print(f"PASS real_large_append_chunked messages={append_proxy.append_message_sizes} "
                  f"pcm_chunks={append_proxy.append_pcm_sizes} "
                  f"metadata=stop/1/length_limited_false", flush=True)

            unknown_proxy = WireProxy(ws_port)
            proxies.append(unknown_proxy)
            unknown = start_cli([str(probe), "live",
                                 f"http://127.0.0.1:{unknown_proxy.port}", str(raw_audio)],
                                env, children)
            unknown_output, _ = finish_cli(unknown)
            unknown_result = json.loads(unknown_output)
            unknown_events = unknown_proxy.close()
            verify_handshake(unknown_events, client_first=False)
            require(unknown_result.get("text") and
                    unknown_result.get("generated_tokens") is None and
                    unknown_result.get("length_limited") is None and
                    unknown_result.get("finish_reason") == "" and
                    unknown_result.get("partial_calls") == 0,
                    f"missing metadata was synthesized: {unknown_result!r}")
            print("PASS real_missing_generation_metadata_unknown "
                  "generated_tokens=null length_limited=null finish_reason=empty", flush=True)

            # A 5 s chunk interval must not delay cancellation: SIGINT is
            # delivered while the CLI waits between real chunks, and the
            # process must exit within 500 ms and leave the stage idle.
            chunk_pcm = 5000 * 32  # CLI chunk_ms * 32 bytes per ms
            cancel_proxy = WireProxy(ws_port)
            proxies.append(cancel_proxy)
            chunk_pcm = 5000 * 32  # CLI chunk_bytes for --chunk-ms 5000
            slow = start_cli([str(binary), "realtime", "--url",
                              f"http://127.0.0.1:{cancel_proxy.port}", "--model", "tiny-ctc",
                              "--file", str(large_wav), "--live-result",
                              "--chunk-ms", "5000",
                              "--final-timeout-ms", "5000", "--stall-ms", "20000"],
                             env, children)
            # One observed append proves the real session is streaming. The
            # 5 s pause after it is the window SIGINT must cut short.
            wait_for(lambda: bool(cancel_proxy.append_message_sizes), slow, 10, 0.001)
            sent_pcm = sum(cancel_proxy.append_pcm_sizes)
            chunk_pcm = 5000 * 32  # the CLI's chunk-ms to PCM byte mapping
            time.sleep(0.3)
            signalled = time.monotonic()
            slow.send_signal(signal.SIGINT)
            try:
                slow_output, slow_error = slow.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                slow.kill()
                slow.communicate(timeout=5)
                raise AssertionError("SIGINT did not stop the CLI within 5 s")
            latency = time.monotonic() - signalled
            require(slow.returncode == 1,
                    f"SIGINT exit {slow.returncode}, expected 1: {slow_error!r}")
            require(slow_output == b"",
                    f"cancelled realtime wrote stdout: {slow_output!r}")
            require(b"cancelled" in slow_error,
                    f"cancelled realtime stderr was {slow_error!r}")
            require(latency < 0.5,
                    f"cancel latency {latency:.3f}s exceeded 0.5s for chunk_ms=5000")
            require(sent_pcm == chunk_pcm,
                    f"first real append carried {sent_pcm} PCM bytes, expected {chunk_pcm}")
            cancel_events = cancel_proxy.close()
            wait_for(lambda: in_flight(http_port) == 0, timeout_s=10)
            print(f"PASS real_cancel_latency chunk_ms=5000 latency_s={latency:.3f} "
                  f"first_chunk_pcm={sent_pcm} wire_events={cancel_events} "
                  "exit=1 stdout_empty=true in_flight=0", flush=True)

            # The public transcribe CLI always sends stream=true and
            # include_generation_metadata=true. req-4 is the next automatic
            # HTTP request id. The observed stage clears the two metadata
            # fields only after a real TinyCTC forward, so the route answers
            # 500 with the real reason.
            missing_http = start_cli(transcribe_command(binary, http_port, audio),
                                     env, children)
            missing_output, missing_error = finish_cli(missing_http, expected=1)
            require(missing_output == b"",
                    f"HTTP 500 transcribe wrote stdout: {missing_output!r}")
            require(b"HTTP 500" in missing_error,
                    f"HTTP 500 transcribe stderr was {missing_error!r}")
            require(b"the ASR stage did not report generation metadata"
                    in missing_error,
                    f"HTTP 500 transcribe stderr lacked the server reason: "
                    f"{missing_error!r}")
            missing_submit = wait_submit(submit_events, "req-4")

            def metadata_omit_lines():
                return bytes(logs["stage"]).count(
                    b"OBSERVED_METADATA_OMIT req-4\n")

            # The stage log is drained by a background thread, so wait for the
            # complete line instead of reading once. wait_submit only proves
            # OBSERVED_SUBMIT and is not a substitute.
            wait_for(lambda: metadata_omit_lines() >= 1, processes["stage"],
                     10, 0.01)
            require(metadata_omit_lines() == 1,
                    "the real stage metadata hook ran more than once for req-4")
            wait_for(lambda: in_flight(http_port) == 0, timeout_s=10)
            # Re-check after the settlement wait so a late duplicate hook line
            # cannot slip past the earlier single read.
            require(metadata_omit_lines() == 1,
                    "the real stage metadata hook ran more than once for req-4")
            print(f"PASS real_http_metadata_missing status=500 "
                  f"submit={missing_submit} stdout_empty=true "
                  f"metadata_omit={metadata_omit_lines()} in_flight=0", flush=True)
            succeeded = True
    finally:
        errors = []
        try:
            resume_stage()
        except Exception as exc:
            errors.append(f"resume stage: {exc}")
        for proc in children:
            if proc.poll() is None:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except Exception as exc:
                    errors.append(f"CLI reap: {exc}")
        for proxy in proxies:
            try:
                proxy.close()
            except Exception as exc:
                errors.append(f"proxy close: {exc}")
        for name, proc in processes.items():
            if proc.poll() is None:
                proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except Exception as exc:
                    errors.append(f"{name} kill: {exc}")
        for thread in readers:
            thread.join(timeout=2)
            if thread.is_alive():
                errors.append(f"log reader {thread.name} did not exit")
        if succeeded:
            codes = {name: proc.returncode for name, proc in processes.items()}
            if errors:
                raise RuntimeError("; ".join(errors))
            print(f"PASS cleanup owned_processes_reaped exit_codes={codes}", flush=True)
        else:
            for name, data in logs.items():
                print(f"{name} log tail:\n{bytes(data).decode(errors='replace')[-4000:]}",
                      file=sys.stderr, flush=True)
            for error in errors:
                print(f"cleanup error: {error}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--observed-stage":
        run_observed_stage(sys.argv[2:])
    else:
        main()

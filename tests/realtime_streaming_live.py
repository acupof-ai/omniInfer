#!/usr/bin/env python3
"""Real-backend realtime streaming evidence.

Runs the real CLI through the real ummInfer WebSocket server and the real
Qwen3-ASR stage against a serving SGLang backend. It records:

  - incremental `transcription.segment` events with is_final=false,
  - the real client's WebSocket PING/PONG control frames interleaved with the
    ongoing stream,
  - the SGLang GPU-side abort acknowledgement after a client cancellation,
  - one terminal event per request and a usable request afterwards.

The wire proxy only forwards and observes real bytes. It injects no frames.
This script is opt-in: pass --backend-url for a reachable SGLang origin.
"""

import argparse
import base64
import hashlib
import http.client
import json
import os
from pathlib import Path
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent))
from realtime_client_contract_e2e import WireProxy  # noqa: E402

LOG_CAP = 65536
READY_TIMEOUT_S = 180
REQUEST_TIMEOUT_S = 180


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def http_json(port, path):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request("GET", path)
        response = conn.getresponse()
        body = response.read(65537)
        require(len(body) <= LOG_CAP, f"{path} response exceeded {LOG_CAP} bytes")
        return response.status, body
    finally:
        conn.close()


def pcm_from_wav(path):
    with wave.open(str(path), "rb") as wav:
        require((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) ==
                (1, 2, 16000), f"{path} is not PCM16 mono 16 kHz")
        return wav.readframes(wav.getnframes())


def spawn(name, args, cwd, env, processes, logs, readers):
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

    thread = threading.Thread(target=drain, name=f"{name}-log", daemon=True)
    thread.start()
    readers.append(thread)
    return proc


def wait_http(port, path, expected, proc, timeout_s=READY_TIMEOUT_S):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        require(proc.poll() is None, f"server exited before {path} was ready")
        try:
            status, body = http_json(port, path)
        except OSError:
            time.sleep(0.1)
            continue
        if status == expected:
            return body
        time.sleep(0.1)
    raise AssertionError(f"{path} did not return {expected} within {timeout_s}s")


def in_flight(port):
    _, body = http_json(port, "/health")
    return json.loads(body)["in_flight"]


def wait_for(predicate, process=None, timeout_s=20, interval_s=0.01):
    for _ in range(max(1, int(timeout_s / interval_s))):
        if predicate():
            return
        if process is not None and process.poll() is not None:
            detail = b""
            if process.stderr is not None:
                detail = process.stderr.read(4096)
            raise AssertionError(
                f"process exited early with {process.returncode}: {detail!r}")
        time.sleep(interval_s)
    raise AssertionError(f"condition was not met within {timeout_s}s")


def segment_events(messages):
    events = []
    for raw in messages:
        try:
            event = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if event.get("type") in ("transcription.segment", "transcription.completed"):
            events.append(event)
    return events


def realtime_command(binary, port, audio, ping_ms, chunk_ms=50):
    return [str(binary), "realtime", "--url", f"http://127.0.0.1:{port}",
            "--model", "qwen3-asr", "--file", str(audio), "--live-result",
            "--chunk-ms", str(chunk_ms), "--final-timeout-ms", "60000",
            "--stall-ms", "60000", "--max-new-tokens", "4096",
            "--ping-interval-ms", str(ping_ms)]


def run_cli(args, env):
    return subprocess.Popen(args, env=env.copy(), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)


def finish_cli(proc, timeout_s):
    try:
        output, error = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        output, error = proc.communicate(timeout=5)
        raise AssertionError(f"CLI timed out after {timeout_s}s: {error!r}")
    require(len(output) <= LOG_CAP and len(error) <= LOG_CAP,
            "CLI output exceeded the cap")
    return output, error


def frame_summary(proxy):
    frames = list(proxy.frame_events)
    return {
        "client_ping": [i for i, (d, op, _) in enumerate(frames)
                        if d == "client" and op == 9],
        "server_pong": [i for i, (d, op, _) in enumerate(frames)
                        if d == "server" and op == 10],
        "client_data": [i for i, (d, op, _) in enumerate(frames)
                        if d == "client" and op in (1, 2)],
        "server_data": [i for i, (d, op, _) in enumerate(frames)
                        if d == "server" and op in (1, 2)],
        "frames": frames,
    }


def save_frames(path, proxy, summary):
    """Write real frame order and raw payloads under a 48 KiB payload budget.

    Control frames and server text messages carry their full bytes. Client
    audio appends are counted and hashed; only their first and last payloads
    are inlined, because the audio body is large and its integrity is covered
    by the end-to-end PCM SHA256.
    """
    client_data = set(summary["client_data"])
    keep = {min(client_data), max(client_data)} if client_data else set()
    budget = 49152
    wire = []
    for index, (direction, opcode, fin, payload) in enumerate(proxy.frame_payloads):
        entry = {"i": index, "dir": direction, "op": opcode, "fin": fin,
                 "bytes": len(payload)}
        inline = opcode >= 8 or direction == "server" or index in keep
        if inline and budget >= len(payload):
            entry["payload_b64"] = base64.b64encode(payload).decode()
            budget -= len(payload)
        else:
            entry["sha256"] = hashlib.sha256(payload).hexdigest()
        wire.append(entry)
    path.write_text(json.dumps({
        "frames": [list(item) for item in summary["frames"]],
        "client_ping_indices": summary["client_ping"],
        "server_pong_indices": summary["server_pong"],
        "client_data_indices": summary["client_data"],
        "server_data_indices": summary["server_data"],
        "client_text_messages": [m.decode("utf-8", "replace")
                                 for m in proxy.server_text_messages],
        "append_message_sizes": proxy.append_message_sizes,
        "append_pcm_sizes": proxy.append_pcm_sizes,
        "append_pcm_sha256": proxy.append_sha256.hexdigest(),
        "inline_payload_budget_bytes": budget,
        "wire": wire,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def success_case(binary, ws_port, http_port, audio, reference, env, out, label):
    proxy = WireProxy(ws_port)
    try:
        proc = run_cli(realtime_command(binary, proxy.port, audio, 10), env)
        output, error = finish_cli(proc, REQUEST_TIMEOUT_S)
        require(proc.returncode == 0,
                f"successful stream exited {proc.returncode}: {error!r}")
        require(output == (reference + "\n").encode(),
                f"CLI output {output!r}, expected reference {reference!r}")
        messages = list(proxy.server_text_messages)
        events = segment_events(messages)
        partials = [e for e in events if e.get("type") == "transcription.segment"
                    and e.get("is_final") is False]
        finals = [e for e in events if e.get("type") == "transcription.segment"
                  and e.get("is_final") is True]
        completed = [e for e in events if e.get("type") == "transcription.completed"]
        require(len(partials) >= 2,
                f"expected at least two incremental partials, got {len(partials)}")
        require(len(finals) == 1, f"expected one final segment, got {finals!r}")
        require(len(completed) == 1, f"expected one terminal event, got {completed!r}")
        require(events[-1] is completed[0],
                "the terminal event was not last on the wire")
        require(finals[0]["text"] == completed[0]["text"] == reference,
                f"final text {finals[0]['text']!r} / {completed[0]['text']!r} "
                f"differs from reference {reference!r}")
        texts = [p["text"] for p in partials]
        require(all(texts[i].startswith(texts[i - 1]) for i in range(1, len(texts))),
                f"partial text was not cumulative: {texts!r}")
        # A decode that finishes its sentence before the terminal event makes
        # the last partial equal to the final text. Incremental delivery is
        # proven by several strictly growing partials, not by that last one.
        require(texts[0] != reference and any(t != reference for t in texts),
                "every partial already held the final text; the stream was "
                "buffered and replayed, not incremental")
        require(all(p.get("event_index") == 0 and p.get("segment_id") == 0
                    for p in partials),
                "partial identifiers changed mid-stream")
        summary = frame_summary(proxy)
        require(len(summary["client_ping"]) >= 2,
                f"too few real PING frames: {len(summary['client_ping'])}")
        require(len(summary["server_pong"]) >= 2,
                f"too few real PONG frames: {len(summary['server_pong'])}")
        require(any(summary["client_data"][0] < i < summary["client_data"][-1]
                    for i in summary["client_ping"]),
                "PING frames were not interleaved with the client stream")
        require(any(summary["server_data"][0] < i < summary["server_data"][-1]
                    for i in summary["server_pong"]),
                "PONG frames were not interleaved with the server data stream")
        require(b"realtime ping sent=" in error,
                f"CLI did not report ping stats: {error!r}")
        save_frames(out / f"success-frames-{label}.json", proxy, summary)
        wait_for(lambda: in_flight(http_port) == 0, timeout_s=20)
        _, health = http_json(http_port, "/health")
        (out / f"health-after-{label}.json").write_bytes(health + b"\n")
        print(f"PASS real_incremental_partials partials={len(partials)} "
              f"final={finals[0]['text']!r} terminal=1 "
              f"partial_texts={texts!r}", flush=True)
        print(f"PASS real_control_frame_interleaving pings={len(summary['client_ping'])} "
              f"pongs={len(summary['server_pong'])} "
              f"client_data={len(summary['client_data'])} "
              f"server_data={len(summary['server_data'])}", flush=True)
    finally:
        proxy.close()


def cancel_case(ws_port, http_port, audio, env, out, logs):
    """Cancel by closing the real WebSocket right after transcription.done.

    The close is a real client action, not an injected frame. The stage must
    ask SGLang to abort the running request and log the backend's own
    finish_reason=abort acknowledgement.
    """
    from websockets.sync.client import connect

    pcm = pcm_from_wav(audio)
    url = (f"ws://127.0.0.1:{ws_port}/v1/realtime"
           f"?intent=transcription&model=qwen3-asr")
    with connect(url, compression=None, open_timeout=10, close_timeout=10,
                 ping_interval=None, max_size=1 << 20) as ws:
        ws.send(json.dumps({"type": "session.update",
                            "session": {"input_audio_format": "pcm16",
                                        "max_new_tokens": 4096}}))
        reply = json.loads(ws.recv(timeout=10))
        require(reply.get("type") == "transcription_session.updated",
                f"session was not accepted: {reply!r}")
        # 700 KiB PCM per append keeps the base64 JSON under the 1 MiB cap.
        for offset in range(0, len(pcm), 716800):
            chunk = pcm[offset:offset + 716800]
            ws.send(json.dumps({"type": "input_audio_buffer.append",
                                "audio": base64.b64encode(chunk).decode()}))
        ws.send(json.dumps({"type": "transcription.done"}))
        # Wait until the backend is actually decoding (the first real partial
        # reached this client), then close. Closing earlier would cancel a
        # request that had not started; closing later risks losing the race to
        # a fast decode.
        first = json.loads(ws.recv(timeout=30))
        require(first.get("type") == "transcription.segment" and
                first.get("is_final") is False,
                f"expected a first partial before cancelling, got {first!r}")
        ws.close()
    deadline = time.monotonic() + 30
    abort_confirmed = False
    stage_log = b""
    while time.monotonic() < deadline:
        stage_log = bytes(logs["stage"])
        abort_confirmed = (b"SGLANG_ABORT_REQUEST" in stage_log and
                           b"SGLANG_ABORT_CONFIRMED" in stage_log and
                           b"finish=abort" in stage_log)
        if abort_confirmed:
            break
        time.sleep(0.1)
    require(abort_confirmed,
            "no GPU-side abort acknowledgement in the stage log: "
            f"{stage_log[-4000:]!r}")
    wait_for(lambda: in_flight(http_port) == 0, timeout_s=20)
    (out / "cancel-stage-abort.txt").write_text(
        "\n".join(line for line in stage_log.decode("utf-8", "replace").splitlines()
                   if "SGLANG_ABORT" in line) + "\n", encoding="utf-8")
    print("PASS real_cancel_disconnect in_flight=0", flush=True)
    print("PASS real_gpu_abort_confirmed SGLANG_ABORT_REQUEST=true "
          "SGLANG_ABORT_CONFIRMED=true finish=abort", flush=True)


def run(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    processes, logs, readers = {}, {}, []
    http_port = 0
    succeeded = False
    try:
        # ZeroMQ ipc endpoints require an absolute path.
        work = (out / "ipc").resolve()
        work.mkdir(parents=True, exist_ok=True)
        entry = f"ipc://{work}/entry"
        completion = f"ipc://{work}/completion"
        abort = f"ipc://{work}/abort"
        http_port = free_port()
        ws_port = free_port()
        while ws_port == http_port:
            ws_port = free_port()
        spawn("server", [str(args.server), "serve", "--host", "127.0.0.1",
                         "--port", str(http_port), "--realtime-port", str(ws_port),
                         "--model", "qwen3-asr", "--entry", "asr",
                         "--entry-endpoint", entry, "--completion", completion,
                         "--abort", abort],
              args.server.parent, env, processes, logs, readers)
        spawn("stage", [sys.executable, "-I", str(args.stage), "serve",
                        "--name", "asr", "--recv", entry,
                        "--coordinator", completion, "--abort", abort,
                        "--model", "qwen3-asr",
                        "--backend-url", args.backend_url, "--timeout", "120"],
              args.stage.parent, env, processes, logs, readers)
        wait_http(http_port, "/ready", 200, processes["server"])
        with urllib.request.urlopen(args.backend_url + "/v1/models",
                                    timeout=10) as response:
            (out / "backend-models.json").write_bytes(response.read() + b"\n")
        print(f"PASS real_startup http_port={http_port} ws_port={ws_port} "
              f"backend={args.backend_url}", flush=True)

        success_case(args.server, ws_port, http_port, args.audio,
                     args.reference, env, out, "one")
        cancel_case(ws_port, http_port, args.cancel_audio, env, out, logs)
        success_case(args.server, ws_port, http_port, args.audio,
                     args.reference, env, out, "after-cancel")
        print("PASS real_request_after_cancel", flush=True)
        succeeded = True
    finally:
        for proc in reversed(list(processes.values())):
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        for name, proc in processes.items():
            try:
                proc.wait(20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(5)
            print(f"INFO exit {name}={proc.returncode}", flush=True)
        for thread in readers:
            thread.join(2)
        for name, data in logs.items():
            (out / f"{name}.log").write_bytes(bytes(data[:LOG_CAP]))
        if processes:
            try:
                _, health = http_json(http_port, "/health")
                (out / "health.json").write_bytes(health + b"\n")
            except OSError:
                pass
        if not succeeded:
            for name, data in logs.items():
                sys.stderr.write(bytes(data[-4000:]).decode("utf-8", "replace"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True, type=Path)
    parser.add_argument("--stage", required=True, type=Path)
    parser.add_argument("--backend-url", required=True)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--cancel-audio", required=True, type=Path)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.server = args.server.resolve()
    args.binary = args.server
    args.stage = args.stage.resolve()
    require(args.server.is_file() and os.access(args.server, os.X_OK),
            f"server binary not executable: {args.server}")
    require(args.stage.is_file(), f"stage script missing: {args.stage}")
    for audio in (args.audio, args.cancel_audio):
        require(audio.is_file(), f"audio file missing: {audio}")
        payload = pcm_from_wav(audio)
        print(f"INFO audio {audio.name} sha256={hashlib.sha256(payload).hexdigest()} "
              f"seconds={len(payload) / 32000:.3f}", flush=True)
    run(args)


if __name__ == "__main__":
    main()

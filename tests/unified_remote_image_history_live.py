#!/usr/bin/env python3
"""H20B evidence: remote HTTP image URL and real multi-turn session history.

Runs the real C++ server, the real Qwen3-ASR stage and the real Qwen3-Omni
worker against serving SGLang backends. Two server/worker stacks are started
one after the other; the worker pair is restarted because a Coordinator
restart loses the one-shot stage_ready message:

  1. without --session-history: a remote-URL image request must complete and
     the opt-in must be rejected; without the opt-in no history crosses turns;
  2. with --session-history: two sequential turns in one session must carry
     the first transcript and answer into the second VLM prompt, and cancelled
     or failed turns must not enter the next prompt.

The image is served by a local HTTP server, so the URL is reachable and the
served bytes are recorded. The run fails if the model answer does not mention
the image content; an unreachable external URL is never accepted as success.

Opt-in only: without UMMINFER_LIVE_OPT_IN=1 it prints BLOCKED and exits 0.
"""

import argparse
import hashlib
import http.server
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unified_partial_abort_live import (  # noqa: E402
    CappedLog, audio_part, free_port, post_json, sha256, stream_chat,
    wait_http)

LOG_CAP = 64 * 1024
PROMPT_RE = re.compile(r"Qwen Omni request (\S+) received messages=(.*)$")


class ImageServer:
    """Serve files from one directory and record every GET as evidence."""

    def __init__(self, directory, host="0.0.0.0"):
        self.directory = Path(directory)
        self.records = []
        self.lock = threading.Lock()
        payload = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - http.server API
                name = self.path.lstrip("/").split("?", 1)[0]
                target = (payload.directory / name).resolve()
                record = {"path": self.path, "name": name}
                if (target.is_file()
                        and target.parent == payload.directory.resolve()):
                    data = target.read_bytes()
                    record.update({"status": 200, "bytes": len(data),
                                   "sha256": hashlib.sha256(data).hexdigest()})
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    record.update({"status": 404, "bytes": 0})
                    self.send_error(404)
                with payload.lock:
                    payload.records.append(record)

            def log_message(self, *args):
                pass

        self.httpd = http.server.ThreadingHTTPServer((host, 0), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def snapshot(self):
        with self.lock:
            return list(self.records)


def parse_prompts(text):
    """Map worker rid -> forwarded messages from the capped worker log."""
    prompts = {}
    for line in text.splitlines():
        match = PROMPT_RE.search(line)
        if not match:
            continue
        try:
            messages = json.loads(match.group(2))
        except ValueError:
            continue
        prompts[match.group(1)] = messages
    return prompts


def unified_events(frames):
    return [frame["unified"] for frame in frames if "unified" in frame]


def summarize_request(frames):
    events = unified_events(frames)
    terminals = [event for event in events
                 if event["kind"] in ("complete", "error", "cancelled")]
    finals = [event for event in events if event["kind"] == "asr_final"]
    seqs = [event["seq"] for event in events]
    return {
        "request_id": frames[0].get("id") if frames else None,
        "events": events,
        "seq": seqs,
        "seq_gap_free": seqs == list(range(len(seqs))),
        "terminals": terminals,
        "terminal_kind": terminals[0]["kind"] if len(terminals) == 1 else None,
        "transcript": finals[0]["text"] if len(finals) == 1 else None,
        "answer": ("".join(event["text"] for event in events
                           if event["kind"] == "vlm_delta")),
        "session_id": events[0].get("session_id") if events else None,
    }


def request_body(audio, instruction, image_url=None, session_id=None,
                 history=None, max_tokens=96):
    content = []
    if image_url is not None:
        content.append({"type": "image_url",
                        "image_url": {"url": image_url}})
    content.append({"type": "text", "text": instruction})
    content.append(audio_part(audio))
    body = {"model": "qwen3-unified",
            "messages": [{"role": "user", "content": content}],
            "stream": True, "max_tokens": max_tokens, "temperature": 0}
    if session_id is not None:
        body["session_id"] = session_id
    if history is not None:
        body["session_history"] = history
    return body


def messages_of(prompt):
    """Normalize a forwarded prompt to [(role, text)] with image parts kept."""
    out = []
    for message in prompt:
        content = message.get("content")
        if isinstance(content, str):
            out.append((message.get("role"), content))
            continue
        texts = []
        for part in content or []:
            if part.get("type") == "text":
                texts.append(part.get("text", ""))
            elif part.get("type") == "image_url":
                # The worker log summarizes a data URL to a digest but keeps a
                # remote URL in a flat "url" field, so accept both shapes.
                nested = part.get("image_url")
                url = (nested.get("url") if isinstance(nested, dict) else None)
                texts.append("<image:" + str(url or part.get("url", "")) + ">")
        out.append((message.get("role"), " ".join(texts)))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--asr-stage", default="python/umminfer_qwen_asr_stage.py")
    parser.add_argument("--omni-worker", default="python/umminfer_qwen_omni_worker.py")
    parser.add_argument("--asr-backend", required=True)
    parser.add_argument("--omni-backend", required=True)
    parser.add_argument("--asr-model", default="qwen3-asr")
    parser.add_argument("--omni-model", default="qwen3-omni")
    parser.add_argument("--audio-zh", required=True)
    parser.add_argument("--audio-en", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--public-host", default="172.17.0.1",
                        help="host address the SGLang container uses for our HTTP server")
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

    def server_cmd(history):
        command = [args.server, "serve", "--host", "127.0.0.1",
                   "--port", str(port), "--model", "qwen3-unified",
                   "--entry", "asr", "--entry-endpoint", endpoint("asr"),
                   "--vlm-entry", "vlm",
                   "--vlm-entry-endpoint", endpoint("vlm"),
                   "--completion", endpoint("done"),
                   "--abort", endpoint("abort")]
        if history:
            command.append("--session-history")
        return command

    image_server = ImageServer(Path(args.image).parent)
    image_url = (f"http://{args.public_host}:{image_server.port}/"
                 f"{Path(args.image).name}")
    image_server.start()
    commands.append(" ".join(asr_cmd))
    commands.append(" ".join(omni_cmd))
    commands.append(" ".join(server_cmd(False)))
    commands.append(" ".join(server_cmd(True)))
    commands.append(f"serve image dir {Path(args.image).parent} on "
                    f"0.0.0.0:{image_server.port} as {image_url}")

    processes = {}
    logs = {}
    all_logs = []
    results = {}
    server_proc = {"proc": None, "log": None}

    def start_workers(tag):
        # A Coordinator restart loses the one-shot stage_ready message, so the
        # workers are restarted with each server instead of left connected.
        for name, command in (("asr", asr_cmd), ("vlm", omni_cmd)):
            proc = subprocess.Popen(command, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)
            log = CappedLog(evidence / f"{name}-{tag}.log")
            log.start(proc)
            processes[name] = proc
            logs[name] = log
            all_logs.append(log)
            time.sleep(0.3)

    def stop_workers():
        for proc in list(processes.values()):
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        for proc in list(processes.values()):
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        processes.clear()

    def start_server(history, name):
        proc = subprocess.Popen(server_cmd(history), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        log = CappedLog(evidence / f"server-{name}.log")
        log.start(proc)
        server_proc["proc"] = proc
        server_proc["log"] = log
        wait_http(f"http://127.0.0.1:{port}/ready", 180)
        wait_http(f"http://127.0.0.1:{port}/v1/models", 30)
        return proc, log

    def stop_server(proc, log):
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        log.wait()

    def health():
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=5) as reply:
            return json.load(reply)

    def record(name, frames, body):
        (evidence / f"case-{name}.frames.jsonl").write_text(
            "\n".join(json.dumps(frame) for frame in frames) + "\n")
        summarized = json.loads(json.dumps(body))
        for message in summarized["messages"]:
            for part in message.get("content", []):
                audio = part.get("input_audio")
                if isinstance(audio, dict) and "data" in audio:
                    audio["data_bytes"] = len(audio.pop("data"))
        (evidence / f"case-{name}.request.json").write_text(
            json.dumps(summarized, indent=2) + "\n")
        return summarize_request(frames)

    def prompt_for(request_id):
        prompts = parse_prompts(logs["vlm"].text())
        prefix = f"{request_id}:vlm@"
        for rid, messages in prompts.items():
            if rid.startswith(prefix):
                return messages
        return None

    try:
        start_workers("default")

        # --- Server A: default, history disabled. -------------------------
        proc, log = start_server(False, "default")
        health_default = health()
        body = request_body(
            args.audio_zh, "Describe exactly what is in this image: name the "
                           "people and the main object.", image_url=image_url)
        status, frames = stream_chat(port, body, timeout=args.timeout)
        default_image = record("default-remote-image", frames, body)
        default_prompt = prompt_for(default_image["request_id"]) or []
        # Same request without the image: a generic answer that does not
        # change would make the keyword check meaningless.
        control_body = request_body(
            args.audio_zh, "Describe exactly what is in this image: name the "
                           "people and the main object.")
        status, frames = stream_chat(port, control_body, timeout=args.timeout)
        control = record("default-no-image", frames, control_body)
        image_answer = default_image["answer"].lower()
        checks = {
            "terminal_complete": default_image["terminal_kind"] == "complete",
            "seq_gap_free": default_image["seq_gap_free"],
            "one_terminal": len(default_image["terminals"]) == 1,
            "transcript_present": bool(default_image["transcript"]),
            "answer_mentions_image": any(
                word in image_answer
                for word in ("iron", "shirt", "man", "person", "cloth",
                             "table", "board", "garment",
                             "熨", "衬衣", "男子", "男人", "人物", "衣物")),
            "answer_differs_from_no_image":
                default_image["answer"].strip() != control["answer"].strip(),
            "control_completed": control["terminal_kind"] == "complete",
            "worker_received_remote_url": any(
                image_url in "<image:" + text + ">"
                for _, text in messages_of(default_prompt)),
            "image_served": any(item.get("status") == 200
                                and item.get("sha256") == sha256(args.image)
                                for item in image_server.snapshot()),
            "prompt_has_no_history": len(default_prompt) == 1,
        }
        results["default-remote-image"] = {
            "checks": checks, "passed": all(checks.values()),
            "request_id": default_image["request_id"],
            "answer": default_image["answer"],
            "control_answer": control["answer"],
            "transcript": default_image["transcript"],
            "vlm_prompt": messages_of(default_prompt),
        }
        try:
            status, reply = post_json(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                request_body(args.audio_zh, "hello", session_id="s",
                             history=True), timeout=30)
            opt_in_status, opt_in_body = status, reply
        except urllib.error.HTTPError as exc:
            opt_in_status = exc.code
            opt_in_body = exc.read(4096).decode("utf-8", "replace")
        results["default-rejects-opt-in"] = {
            "status": opt_in_status, "body": str(opt_in_body),
            "checks": {"status_400": opt_in_status == 400,
                       "names_switch":
                       "session_history is not enabled on this server"
                       in str(opt_in_body)},
            "passed": (opt_in_status == 400 and
                       "session_history is not enabled on this server"
                       in str(opt_in_body)),
        }
        results["default-remote-image"]["health"] = health_default
        stop_server(proc, log)
        stop_workers()

        # --- Server B: history enabled. -----------------------------------
        start_workers("history")
        proc, log = start_server(True, "history")
        health_history = health()
        session = "h20b-final-history"

        turn1_body = request_body(
            args.audio_zh, "Describe this image in one sentence.",
            image_url=image_url, session_id=session, history=True)
        status, frames = stream_chat(port, turn1_body, timeout=args.timeout)
        turn1 = record("history-turn1", frames, turn1_body)
        turn1_prompt = prompt_for(turn1["request_id"]) or []
        results["history-turn1"] = {
            "checks": {
                "terminal_complete": turn1["terminal_kind"] == "complete",
                "seq_gap_free": turn1["seq_gap_free"],
                "one_terminal": len(turn1["terminals"]) == 1,
                "transcript_present": bool(turn1["transcript"]),
                "worker_received_remote_url": any(
                    image_url in "<image:" + text + ">"
                    for _, text in messages_of(turn1_prompt)),
                "image_served": any(
                    item.get("status") == 200 and
                    item.get("sha256") == sha256(args.image)
                    for item in image_server.snapshot()),
                "no_prior_turn": len(turn1_prompt) == 1,
            },
            "passed": False, "request_id": turn1["request_id"],
            "session_id": turn1["session_id"],
            "transcript": turn1["transcript"], "answer": turn1["answer"],
            "vlm_prompt": messages_of(turn1_prompt),
        }
        results["history-turn1"]["passed"] = all(
            results["history-turn1"]["checks"].values())

        turn2_body = request_body(
            args.audio_zh, "Was the previous answer about an image?",
            session_id=session, history=True)
        status, frames = stream_chat(port, turn2_body, timeout=args.timeout)
        turn2 = record("history-turn2", frames, turn2_body)
        turn2_prompt = messages_of(prompt_for(turn2["request_id"]) or [])
        prior_pairs_ok = (
            len(turn2_prompt) == 3 and
            turn2_prompt[0][0] == "user" and
            turn2_prompt[0][1] == turn1["transcript"] and
            turn2_prompt[1][0] == "assistant" and
            turn2_prompt[1][1] == turn1["answer"])
        results["history-turn2"] = {
            "checks": {
                "terminal_complete": turn2["terminal_kind"] == "complete",
                "seq_gap_free": turn2["seq_gap_free"],
                "one_terminal": len(turn2["terminals"]) == 1,
                "distinct_request_id":
                turn2["request_id"] != turn1["request_id"],
                "same_session": turn2["session_id"] == session,
                "prior_user_transcript": bool(turn2_prompt) and
                turn2_prompt[0][0] == "user" and
                turn2_prompt[0][1] == turn1["transcript"],
                "prior_assistant_answer": len(turn2_prompt) > 1 and
                turn2_prompt[1][0] == "assistant" and
                turn2_prompt[1][1] == turn1["answer"],
                "prior_pair_before_current": prior_pairs_ok,
                "current_turn_last": len(turn2_prompt) == 3 and
                turn2_prompt[2][0] == "user",
            },
            "passed": False, "request_id": turn2["request_id"],
            "session_id": turn2["session_id"],
            "transcript": turn2["transcript"], "answer": turn2["answer"],
            "vlm_prompt": turn2_prompt,
        }
        results["history-turn2"]["passed"] = all(
            results["history-turn2"]["checks"].values())

        # Cancelled turn: long VLM stream, abort after the first delta.
        cancelled = {"request_id": None, "abort_status": None,
                     "aborted": False, "after": False}

        def on_frame(frame):
            if cancelled["after"]:
                frame["_after_abort"] = True
            unified = frame.get("unified")
            if frame.get("id"):
                cancelled["request_id"] = frame["id"]
            if (unified and unified["kind"] == "vlm_delta"
                    and not cancelled["aborted"]):
                cancelled["aborted"] = True
                try:
                    code, _ = post_json(
                        f"http://127.0.0.1:{port}/abort",
                        {"request_id": cancelled["request_id"]}, timeout=10)
                    cancelled["abort_status"] = code
                except Exception as exc:  # noqa: BLE001 - recorded below
                    cancelled["abort_status"] = f"error: {exc}"
                cancelled["after"] = True

        cancel_body = request_body(
            args.audio_en,
            "Count from 1 to 600, one number per line. Do not stop early.",
            session_id=session, history=True, max_tokens=600)
        status, frames = stream_chat(port, cancel_body, on_frame,
                                     timeout=args.timeout)
        cancel = record("history-cancel", frames, cancel_body)
        # The confirmed aborts log line is written after the public cancelled
        # terminal, so wait for it instead of racing the worker thread.
        for _ in range(150):
            if "SGLANG_ABORT_CONFIRMED" in logs["vlm"].text():
                break
            time.sleep(0.1)
        cancel_prompt = messages_of(prompt_for(cancel["request_id"]) or [])
        results["history-cancel"] = {
            "checks": {
                "terminal_cancelled": cancel["terminal_kind"] == "cancelled",
                "one_terminal": len(cancel["terminals"]) == 1,
                "seq_gap_free": cancel["seq_gap_free"],
                "abort_post_200": cancelled["abort_status"] == 200,
                "transcript_present": bool(cancel["transcript"]),
                "prompt_had_prior_turn": any(
                    role == "assistant" and text == turn1["answer"]
                    for role, text in cancel_prompt),
                "worker_gpu_abort_confirmed":
                f"SGLANG_ABORT_CONFIRMED rid={cancel['request_id']}:vlm@"
                in logs["vlm"].text(),
            },
            "passed": False, "request_id": cancel["request_id"],
            "transcript": cancel["transcript"],
            "vlm_prompt": cancel_prompt,
        }
        results["history-cancel"]["passed"] = all(
            results["history-cancel"]["checks"].values())

        # A failed turn: valid ASR, VLM rejects an undecodable image.
        error_body = request_body(
            args.audio_en, "Describe this image.",
            image_url="http://127.0.0.1:1/missing.png",
            session_id=session, history=True)
        status, frames = stream_chat(port, error_body, timeout=args.timeout)
        error_turn = record("history-error", frames, error_body)
        error_prompt = messages_of(prompt_for(error_turn["request_id"]) or [])
        results["history-error"] = {
            "checks": {
                "terminal_error": error_turn["terminal_kind"] == "error",
                "one_terminal": len(error_turn["terminals"]) == 1,
                "seq_gap_free": error_turn["seq_gap_free"],
                "transcript_present": bool(error_turn["transcript"]),
            },
            "passed": False, "request_id": error_turn["request_id"],
            "transcript": error_turn["transcript"],
            "vlm_prompt": error_prompt,
        }
        results["history-error"]["passed"] = all(
            results["history-error"]["checks"].values())

        # Next successful turn must contain only committed turns.
        turn3_body = request_body(
            args.audio_zh, "Summarize the conversation so far.",
            session_id=session, history=True)
        status, frames = stream_chat(port, turn3_body, timeout=args.timeout)
        turn3 = record("history-turn3", frames, turn3_body)
        turn3_prompt = messages_of(prompt_for(turn3["request_id"]) or [])
        cancel_in = any(cancel["transcript"] and cancel["transcript"] in text
                        for _, text in turn3_prompt)
        error_in = any(error_turn["transcript"] and
                       error_turn["transcript"] in text
                       for _, text in turn3_prompt)
        results["history-turn3"] = {
            "checks": {
                "terminal_complete": turn3["terminal_kind"] == "complete",
                "seq_gap_free": turn3["seq_gap_free"],
                "one_terminal": len(turn3["terminals"]) == 1,
                "committed_turns_present": bool(turn3_prompt) and
                turn3_prompt[0][1] == turn1["transcript"],
                "cancelled_turn_absent": not cancel_in,
                "error_turn_absent": not error_in,
            },
            "passed": False, "request_id": turn3["request_id"],
            "transcript": turn3["transcript"],
            "vlm_prompt": turn3_prompt,
        }
        results["history-turn3"]["passed"] = all(
            results["history-turn3"]["checks"].values())

        # Default request on the history-enabled server: no opt-in, no history.
        # The field is absent, not false: this is the default request shape.
        plain_body = request_body(args.audio_zh, "Name one object in this image.",
                                  session_id=session)
        status, frames = stream_chat(port, plain_body, timeout=args.timeout)
        plain = record("history-off-default", frames, plain_body)
        plain_prompt = messages_of(prompt_for(plain["request_id"]) or [])
        results["history-off-default"] = {
            "checks": {
                "terminal_complete": plain["terminal_kind"] == "complete",
                "seq_gap_free": plain["seq_gap_free"],
                "one_terminal": len(plain["terminals"]) == 1,
                "no_prior_turn": len(plain_prompt) == 1,
            },
            "passed": False, "request_id": plain["request_id"],
            "vlm_prompt": plain_prompt,
        }
        results["history-off-default"]["passed"] = all(
            results["history-off-default"]["checks"].values())

        health_final = health()
        prompts = parse_prompts(logs["vlm"].text())
        summary = {
            "checks": results,
            "health_default": health_default,
            "health_history": health_history,
            "health_final": health_final,
            "image_url": image_url,
            "image_access": image_server.snapshot(),
            "image_sha256": sha256(args.image),
            "vlm_prompts": prompts,
            "server_port": port,
            "passed": all(item["passed"] for item in results.values()),
        }
        Path(evidence / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n")
        for log in all_logs:
            log.wait()
        server_log = server_proc["log"]
        if server_log is not None:
            server_log.wait()
        Path(evidence / "commands.txt").write_text("\n".join(commands) + "\n")
        Path(evidence / "env.json").write_text(json.dumps({
            "asr_backend": args.asr_backend,
            "omni_backend": args.omni_backend,
            "asr_model": args.asr_model,
            "omni_model": args.omni_model,
            "public_host": args.public_host,
            "image_url": image_url,
            "input_sha256": {str(Path(p)): sha256(p)
                             for p in (args.audio_zh, args.audio_en,
                                       args.image)},
        }, indent=2, sort_keys=True) + "\n")
        with open(evidence / "SHA256SUMS", "w") as sums:
            for path in sorted(evidence.rglob("*")):
                if path.is_file() and path.name != "SHA256SUMS":
                    sums.write(f"{sha256(path)}  "
                               f"{path.relative_to(evidence)}\n")
        for name, result in results.items():
            print(("PASS " if result["passed"] else "FAIL ") + name)
            for key, value in result["checks"].items():
                print(f"  {key}={value}")
        print(f"EVIDENCE={evidence}")
        return 0 if summary["passed"] else 1
    finally:
        image_server.stop()
        for proc in processes.values():
            proc.send_signal(signal.SIGTERM)
        for proc in processes.values():
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        proc = server_proc["proc"]
        if proc is not None and proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())

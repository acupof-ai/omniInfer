#!/usr/bin/env python3
"""Deterministic tests for the phase-6 RL architecture.

No serving binary and no model are required. A loopback HTTP server emits
canned JSON/SSE bytes so the recorder is exercised through the real public
HTTP contract, and the worker is driven both in-process and as a subprocess.
"""

import base64
import hashlib
import http.server
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import umminfer_rl_reward as reward
import umminfer_rl_trajectory as traj
import umminfer_rl_worker as worker


def sse(payloads):
    return "".join("data: " + p + "\n\n" for p in payloads).encode("utf-8")


class CannedHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        if getattr(self.server, "abort_streaming", False) and \
                self.path == "/v1/chat/completions":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(
                ("data: " + unified_chunk("vlm_delta", 0, "partial")
                 + "\n\n").encode())
            self.wfile.flush()
            self.server.abort_event.wait(5)
            terminal = json.dumps({
                "id": "http-1",
                "choices": [{"index": 0, "delta": {},
                             "finish_reason": None}],
                "unified": {"kind": "cancelled", "stage": None,
                            "seq": 1, "session_id": "sess-1",
                            "is_final": True, "text": ""}})
            self.wfile.write(("data: " + terminal + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        status, content_type, payload = self.server.script(self.path, body)
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


class CannedServer:
    def __init__(self, script):
        self.httpd = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), CannedHandler)
        self.httpd.script = script
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       daemon=True)
        self.thread.start()

    @property
    def base_url(self):
        return "127.0.0.1:{}".format(self.httpd.server_address[1])

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


def chat_body(**overrides):
    body = {"model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True}
    body.update(overrides)
    return body


def unified_chunk(kind, seq, text, is_final=False, request_id="http-1"):
    # Mirror the real server (openai_api.cpp): only VLM deltas carry
    # choices[].delta.content; ASR unified events have an empty delta.
    stage = "asr" if "asr" in kind else "vlm"
    delta = {"content": text} if stage == "vlm" else {}
    return json.dumps({
        "id": request_id, "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        "unified": {"kind": kind, "stage": stage,
                    "seq": seq, "session_id": "sess-1",
                    "is_final": is_final, "text": text}})


class TextResponse:
    """Minimal file-like stand-in for an http.client response body."""

    def __init__(self, raw_bytes):
        self._lines = raw_bytes.decode("utf-8").splitlines(keepends=True)
        self._index = 0

    def readline(self):
        if self._index >= len(self._lines):
            return ""
        line = self._lines[self._index]
        self._index += 1
        return line


class InputReferenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.art = Path(self.tmp.name) / "artifacts"

    def test_text_audio_image_inputs_are_referenced(self):
        wav = b"RIFFxxxxWAVEfmt "
        png = b"\x89PNG\r\n\x1a\n-bytes-"
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "transcribe this"},
            {"type": "input_audio", "input_audio": {
                "format": "wav",
                "data": base64.b64encode(wav).decode()}},
            {"type": "image_url", "image_url": {"url":
                "data:image/png;base64," + base64.b64encode(png).decode()}},
            {"type": "image_url", "image_url": {
                "url": "https://example.test/pic.png"}},
        ]}]
        refs = traj.input_references(messages, self.art)
        self.assertEqual([r["kind"] for r in refs],
                         ["text", "audio", "image", "image"])

        self.assertEqual(refs[0]["text"], "transcribe this")
        self.assertEqual(refs[0]["sha256"],
                         hashlib.sha256(b"transcribe this").hexdigest())

        audio_hash = hashlib.sha256(wav).hexdigest()
        self.assertEqual(refs[1]["sha256"], audio_hash)
        self.assertEqual(refs[1]["ref"], audio_hash + ".wav")
        self.assertEqual((self.art / (audio_hash + ".wav")).read_bytes(), wav)

        image_hash = hashlib.sha256(png).hexdigest()
        self.assertEqual(refs[2]["sha256"], image_hash)
        self.assertTrue((self.art / (image_hash + ".png")).exists())

        self.assertEqual(refs[3]["ref"],
                         "https://example.test/pic.png")
        self.assertIsNone(refs[3]["sha256"])

    def test_identical_media_stored_once(self):
        wav = b"RIFF-duplicate"
        messages = [{"role": "user", "content": [
            {"type": "input_audio", "input_audio": {
                "format": "wav", "data": base64.b64encode(wav).decode()}},
            {"type": "input_audio", "input_audio": {
                "format": "wav", "data": base64.b64encode(wav).decode()}},
        ]}]
        refs = traj.input_references(messages, self.art)
        self.assertEqual(refs[0]["ref"], refs[1]["ref"])
        self.assertEqual(len(list(self.art.iterdir())), 1)


class StreamParsingTests(unittest.TestCase):
    def test_unified_events_keep_order_and_server_seq(self):
        raw = sse([
            unified_chunk("asr_partial", 0, "hel"),
            unified_chunk("asr_final", 1, "hello world", is_final=True),
            unified_chunk("vlm_delta", 2, "answer"),
            json.dumps({"id": "http-1", "choices": [{"index": 0, "delta": {},
                        "finish_reason": "stop"}]}),
            "[DONE]"])
        request_id, events, text, state, error = traj.parse_sse_stream(
            TextResponse(raw))
        self.assertEqual(request_id, "http-1")
        self.assertEqual(state, traj.TERMINAL_COMPLETED)
        self.assertIsNone(error)
        self.assertEqual([e["type"] for e in events],
                         ["asr_partial", "asr_final", "vlm_delta", "finish"])
        self.assertEqual([e["server_seq"] for e in events[:3]], [0, 1, 2])
        self.assertEqual([e["seq"] for e in events], [0, 1, 2, 3])
        self.assertTrue(events[1]["is_final"])
        # ASR partials are provisional; the final transcript and VLM delta
        # form the hashed output.
        self.assertEqual(text, "hello worldanswer")

    def test_eof_without_done_is_failed_with_benchmark_message(self):
        raw = sse([unified_chunk("vlm_delta", 0, "partial")])
        _, events, _, state, error = traj.parse_sse_stream(TextResponse(raw))
        self.assertEqual(state, traj.TERMINAL_FAILED)
        self.assertEqual(error, traj.TRUNCATED_MESSAGE)
        self.assertEqual(events[-1]["type"], "vlm_delta")

    def test_error_chunk_marks_failed(self):
        raw = sse([
            json.dumps({"error": {"message": "boom", "type":
                                  "generation_error", "code": 500}}),
            "[DONE]"])
        _, events, _, state, error = traj.parse_sse_stream(TextResponse(raw))
        self.assertEqual(state, traj.TERMINAL_FAILED)
        self.assertEqual(error, "boom")
        self.assertEqual(events[-1]["type"], "error")

    def test_unified_cancelled_terminal_is_cancelled_not_completed(self):
        # Mirrors the real server's accepted-abort sequence
        # (openai_api.cpp kCancelled path): terminal unified frame then
        # [DONE], with no error key anywhere.
        raw = sse([
            unified_chunk("vlm_delta", 0, "partial"),
            json.dumps({
                "id": "http-1", "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {},
                             "finish_reason": None}],
                "unified": {"kind": "cancelled", "stage": None,
                            "seq": 1, "session_id": "sess-1",
                            "is_final": True, "text": ""}}),
            "[DONE]"])
        _, events, _, state, error = traj.parse_sse_stream(TextResponse(raw))
        self.assertEqual(state, traj.TERMINAL_CANCELLED)
        self.assertIsNone(error)
        self.assertEqual(events[-1]["type"], "cancelled")

    def test_unified_error_terminal_is_failed(self):
        raw = sse([
            json.dumps({
                "id": "http-1",
                "choices": [{"index": 0, "delta": {},
                             "finish_reason": None}],
                "unified": {"kind": "error", "stage": "vlm", "seq": 0,
                            "is_final": True, "text": "stage blew up"}}),
            "[DONE]"])
        _, _, _, state, error = traj.parse_sse_stream(TextResponse(raw))
        self.assertEqual(state, traj.TERMINAL_FAILED)
        self.assertEqual(error, "stage blew up")


class RecorderHttpTests(unittest.TestCase):
    def test_full_streaming_record_is_deterministic(self):
        def script(path, body):
            return 200, "text/event-stream", sse([
                unified_chunk("asr_partial", 0, "hel"),
                unified_chunk("asr_final", 1, "hello", is_final=True),
                "[DONE]"])

        server = CannedServer(script)
        self.addCleanup(server.stop)
        encoded = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as tmp:
                rec = traj.TrajectoryRecorder(
                    Path(tmp) / "traj.jsonl", Path(tmp) / "art")
                record = traj.record_chat_completion(
                    server.base_url, chat_body(), rec, "sess-1")
                encoded.append(json.dumps(record, sort_keys=True,
                                          ensure_ascii=False))
        self.assertEqual(encoded[0], encoded[1])
        record = json.loads(encoded[0])
        self.assertEqual(record["schema_version"],
                         traj.TRAJECTORY_SCHEMA_VERSION)
        self.assertEqual(record["session_id"], "sess-1")
        self.assertEqual(record["turn_index"], 0)
        self.assertEqual(record["terminal_state"],
                         traj.TERMINAL_COMPLETED)
        self.assertEqual(record["output_sha256"],
                         hashlib.sha256("hello".encode()).hexdigest())
        self.assertEqual(record["inputs"][0]["kind"], "text")
        # No wall-clock field may leak into the record.
        self.assertNotIn("created", record)

    def test_turn_index_increments_per_session(self):
        def script(path, body):
            payload = json.dumps({
                "id": "http-9", "object": "chat.completion",
                "choices": [{"index": 0,
                             "message": {"role": "assistant", "content": "ok"},
                             "finish_reason": "stop"}]}).encode()
            return 200, "application/json", payload

        server = CannedServer(script)
        self.addCleanup(server.stop)
        with tempfile.TemporaryDirectory() as tmp:
            rec = traj.TrajectoryRecorder(Path(tmp) / "traj.jsonl")
            body = chat_body(stream=False)
            first = traj.record_chat_completion(
                server.base_url, body, rec, "sess-1")
            second = traj.record_chat_completion(
                server.base_url, body, rec, "sess-1")
            other = traj.record_chat_completion(
                server.base_url, body, rec, "sess-2")
            self.assertEqual([first["turn_index"], second["turn_index"],
                              other["turn_index"]], [0, 1, 0])

    def test_http_error_body_is_failed_record(self):
        def script(path, body):
            payload = json.dumps({"error": {
                "message": "messages must be a non-empty array",
                "type": "invalid_request"}}).encode()
            return 400, "application/json", payload

        server = CannedServer(script)
        self.addCleanup(server.stop)
        with tempfile.TemporaryDirectory() as tmp:
            rec = traj.TrajectoryRecorder(Path(tmp) / "traj.jsonl")
            record = traj.record_chat_completion(
                server.base_url, chat_body(), rec, "sess-1")
            self.assertEqual(record["terminal_state"],
                             traj.TERMINAL_FAILED)
            self.assertEqual(record["events"][0]["type"], "error")

    def test_transport_failure_becomes_failed_record(self):
        import socket
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        dead_port = sock.getsockname()[1]
        sock.close()  # nothing listens: connect must be refused
        with tempfile.TemporaryDirectory() as tmp:
            rec = traj.TrajectoryRecorder(Path(tmp) / "traj.jsonl")
            record = traj.record_chat_completion(
                f"127.0.0.1:{dead_port}", chat_body(), rec, "sess-1",
                timeout=2.0)
            self.assertEqual(record["terminal_state"],
                             traj.TERMINAL_FAILED)
            self.assertTrue(record["error"].startswith("transport error:"))
            self.assertEqual(record["events"][0]["type"], "error")
            # The turn is still in the export.
            self.assertEqual(
                len(Path(rec.jsonl_path).read_text().splitlines()), 1)

    def test_malformed_success_body_is_failed_record(self):
        def script(path, body):
            payload = json.dumps({"object": "chat.completion"}).encode()
            return 200, "application/json", payload

        server = CannedServer(script)
        self.addCleanup(server.stop)
        with tempfile.TemporaryDirectory() as tmp:
            rec = traj.TrajectoryRecorder(Path(tmp) / "traj.jsonl")
            record = traj.record_chat_completion(
                server.base_url, chat_body(stream=False), rec, "sess-1")
            self.assertEqual(record["terminal_state"],
                             traj.TERMINAL_FAILED)
            self.assertEqual(record["events"][0]["type"], "error")

    def test_request_without_messages_is_failed_record(self):
        server = CannedServer(lambda p, b: (200, "application/json", b"{}"))
        self.addCleanup(server.stop)
        with tempfile.TemporaryDirectory() as tmp:
            rec = traj.TrajectoryRecorder(Path(tmp) / "traj.jsonl")
            record = traj.record_chat_completion(
                server.base_url, {"stream": False}, rec, "sess-1")
            self.assertEqual(record["terminal_state"],
                             traj.TERMINAL_FAILED)
            self.assertEqual(record["inputs"], [])

    def test_client_abort_through_public_exit_records_cancelled(self):
        import threading

        def script(path, body):
            if path == "/abort":
                server.httpd.abort_event.set()
                return 200, "application/json", b'{"success":true}'
            return 200, "application/json", b"{}"

        server = CannedServer(script)
        server.httpd.abort_streaming = True
        server.httpd.abort_event = threading.Event()
        self.addCleanup(server.stop)
        with tempfile.TemporaryDirectory() as tmp:
            rec = traj.TrajectoryRecorder(Path(tmp) / "traj.jsonl")
            abort_status = []

            def cancel_when_known(request_id):
                status, _ = traj.abort_request(server.base_url, request_id)
                abort_status.append(status)

            record = traj.record_chat_completion(
                server.base_url, chat_body(), rec, "sess-1",
                on_request_id=cancel_when_known)
            self.assertEqual(abort_status, [200])
            self.assertEqual(record["terminal_state"],
                             traj.TERMINAL_CANCELLED)
            self.assertEqual(record["request_id"], "http-1")


class AbortExitTests(unittest.TestCase):
    def test_abort_uses_public_exit_and_reports_501(self):
        seen = {}

        def script(path, body):
            seen["path"] = path
            seen["body"] = body
            if path == "/abort":
                return 501, "application/json", json.dumps({
                    "error": {"message":
                              "abort is not supported: requests are served "
                              "by a fixed upstream",
                              "type": "unsupported_route"}}).encode()
            return 200, "application/json", b"{}"

        server = CannedServer(script)
        self.addCleanup(server.stop)
        status, body = traj.abort_request(server.base_url, "http-7")
        self.assertEqual(status, 501)
        self.assertEqual(body["error"]["type"], "unsupported_route")
        self.assertEqual(seen, {"path": "/abort",
                                "body": {"request_id": "http-7"}})

    def test_abort_success_body(self):
        def script(path, body):
            return 200, "application/json", b'{"success":true}'

        server = CannedServer(script)
        self.addCleanup(server.stop)
        status, body = traj.abort_request(server.base_url, "http-8")
        self.assertEqual(status, 200)
        self.assertEqual(body, {"success": True})

    def test_cancelled_terminal_state_is_client_attributed(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = traj.TrajectoryRecorder(Path(tmp) / "traj.jsonl")
            record = traj.record_realtime(
                rec, "sess-1", "realtime-1",
                [{"type": "session.update", "direction": "inbound"},
                 {"type": "input_audio_buffer.append",
                  "direction": "inbound"},
                 {"type": "transcription.done", "direction": "inbound"}],
                terminal_state=traj.TERMINAL_CANCELLED)
            self.assertEqual(record["terminal_state"],
                             traj.TERMINAL_CANCELLED)
            self.assertEqual([e["direction"] for e in record["events"]],
                             ["inbound"] * 3)


class RealtimeRecordingTests(unittest.TestCase):
    def test_outbound_events_derive_completed_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = traj.TrajectoryRecorder(Path(tmp) / "traj.jsonl")
            record = traj.record_realtime(
                rec, "sess-1", "realtime-2",
                [{"type": "transcription.segment", "direction": "outbound",
                  "text": "partial", "is_final": False},
                 {"type": "transcription.segment", "direction": "outbound",
                  "text": "full transcript", "is_final": True},
                 {"type": "transcription.completed", "direction": "outbound",
                  "text": "full transcript"}])
            self.assertEqual(record["terminal_state"],
                             traj.TERMINAL_COMPLETED)
            self.assertEqual(record["output_text"], "full transcript")
            self.assertEqual(record["output_sha256"],
                             hashlib.sha256(b"full transcript").hexdigest())
            self.assertEqual([e["seq"] for e in record["events"]], [0, 1, 2])
            self.assertTrue(record["events"][1]["is_final"])

    def test_outbound_error_derives_failed_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = traj.TrajectoryRecorder(Path(tmp) / "traj.jsonl")
            record = traj.record_realtime(
                rec, "sess-1", "realtime-3",
                [{"type": "error", "direction": "outbound",
                  "error": {"message": "realtime transcription timed out"}}])
            self.assertEqual(record["terminal_state"],
                             traj.TERMINAL_FAILED)
            self.assertEqual(record["error"],
                             "realtime transcription timed out")

    def test_realtime_accepts_explicit_input_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = traj.TrajectoryRecorder(Path(tmp) / "traj.jsonl")
            record = traj.record_realtime(
                rec, "sess-1", "realtime-4",
                [{"type": "transcription.completed", "direction": "outbound",
                  "text": "hi"}],
                inputs=[{"kind": "audio", "ref": "pcm-1.bin",
                         "media_type": "audio/pcm16",
                         "sha256": hashlib.sha256(b"pcm").hexdigest()}])
            self.assertEqual(record["inputs"][0]["kind"], "audio")
            self.assertEqual(record["inputs"][0]["media_type"],
                             "audio/pcm16")

    def test_realtime_with_no_outbound_terminal_is_cancelled(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = traj.TrajectoryRecorder(Path(tmp) / "traj.jsonl")
            record = traj.record_realtime(
                rec, "sess-1", "realtime-5",
                [{"type": "transcription.done", "direction": "inbound"}])
            self.assertEqual(record["terminal_state"],
                             traj.TERMINAL_CANCELLED)
            self.assertEqual(record["output_text"], "")


class RewardTests(unittest.TestCase):
    def test_null_reward_is_missing_not_zero(self):
        null = reward.NullReward()
        self.assertIsNone(null.score({"terminal_state": "completed"}))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "traj.jsonl"
            path.write_text(json.dumps({"schema_version": 1,
                                        "terminal_state": "completed"}) + "\n")
            reward.apply_reward(path, null)
            scored = json.loads(path.read_text())
            self.assertIsNone(scored["reward"]["score"])
            self.assertEqual(scored["reward"]["name"], "null")
            self.assertEqual(scored["reward"]["schema_version"],
                             reward.REWARD_SCHEMA_VERSION)

    def test_pluggable_reward_loads_from_module_path(self):
        loaded = reward.load_reward(__name__ + ":ConstantReward")
        self.assertIsInstance(loaded, ConstantReward)
        self.assertEqual(loaded.score({}), 1.0)

    def test_nan_score_cannot_be_written(self):
        class NanReward(reward.Reward):
            name = "nan"

            def score(self, record):
                return float("nan")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "traj.jsonl"
            path.write_text(json.dumps({"schema_version": 1}) + "\n")
            with self.assertRaises(ValueError):
                reward.apply_reward(path, NanReward())


class ConstantReward(reward.Reward):
    name = "constant-one"

    def score(self, record):
        return 1.0


class CheckpointTests(unittest.TestCase):
    def _trajectory(self, tmp, *, reward_name=None, records=1,
                    corrupt_seq=False, corrupt_hash=False,
                    terminal_state="completed"):
        path = Path(tmp) / "traj.jsonl"
        lines = []
        for index in range(records):
            text = f"answer-{index}"
            record = {
                "schema_version": traj.TRAJECTORY_SCHEMA_VERSION,
                "session_id": "s", "request_id": f"r{index}",
                "turn_index": index,
                "events": ([{"seq": 1, "type": "x"}] if corrupt_seq else []),
                "terminal_state": terminal_state,
                "output_text": text,
                "output_sha256": ("deadbeef" if corrupt_hash else
                                  hashlib.sha256(text.encode()).hexdigest())}
            if reward_name is not None:
                record["reward"] = {"name": reward_name, "score": 1.0}
            lines.append(json.dumps(record, sort_keys=True))
        path.write_text("\n".join(lines) + "\n")
        return path

    def test_checkpoint_sequences_are_monotonic_and_versioned(self):
        with tempfile.TemporaryDirectory() as tmp:
            traj_path = self._trajectory(
                tmp, reward_name="constant-one", records=2)
            tw = worker.TrainingWorker(tmp)
            first = tw.handle({"type": "train", "task_id": "t1",
                               "trajectory_path": str(traj_path)})
            second = tw.handle({"type": "train", "task_id": "t2",
                                "trajectory_path": str(traj_path)})
            self.assertEqual(first["status"], "ok")
            self.assertEqual([first["sequence"], second["sequence"]], [0, 1])
            metadata = json.loads(Path(second["checkpoint_path"]).read_text())
            self.assertEqual(metadata["checkpoint_schema_version"],
                             worker.CHECKPOINT_SCHEMA_VERSION)
            self.assertEqual(metadata["worker_protocol_version"],
                             worker.WORKER_PROTOCOL_VERSION)
            self.assertEqual(metadata["sequence"], 1)
            self.assertIsNone(metadata["algorithm"])
            self.assertIsNone(metadata["weights_ref"])
            self.assertEqual(metadata["reward"], {"name": "constant-one"})
            self.assertEqual(metadata["source"]["record_count"], 2)
            first_meta = json.loads(
                Path(first["checkpoint_path"]).read_text())
            self.assertIsNone(first_meta["base_checkpoint_sha256"])
            second_parent = metadata["base_checkpoint_sha256"]
            self.assertEqual(
                second_parent,
                worker.sha256_file(first["checkpoint_path"]))
            self.assertEqual(second_parent,
                             Path(first["checkpoint_path"] + ".sha256")
                             .read_text().strip())
            # Reopening the same directory re-verifies the chain and continues.
            reopened = worker.TrainingWorker(tmp)
            third = reopened.handle(
                {"type": "train", "trajectory_path": str(traj_path)})
            self.assertEqual(third["sequence"], 2)

    def test_checkpoint_chain_detects_tampered_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            traj_path = self._trajectory(tmp)
            tw = worker.TrainingWorker(tmp)
            first = tw.handle({"type": "train",
                               "trajectory_path": str(traj_path)})
            tw.handle({"type": "train", "trajectory_path": str(traj_path)})
            manifest = Path(first["checkpoint_path"])
            manifest.write_text(manifest.read_text().replace(
                '"algorithm": null', '"algorithm": "PPO"'))
            with self.assertRaises(ValueError):
                worker.TrainingWorker(tmp)  # sidecar mismatch on reopen

    def test_forged_sidecar_breaks_parent_hash_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            traj_path = self._trajectory(tmp)
            tw = worker.TrainingWorker(tmp)
            first = tw.handle({"type": "train",
                               "trajectory_path": str(traj_path)})
            tw.handle({"type": "train", "trajectory_path": str(traj_path)})
            # Re-sign the tampered manifest so its own sidecar stays valid;
            # the child manifest's parent hash must then fail verification.
            manifest = Path(first["checkpoint_path"])
            manifest.write_text(manifest.read_text().replace(
                '"algorithm": null', '"algorithm": "PPO"'))
            Path(str(manifest) + ".sha256").write_text(
                worker.sha256_file(manifest) + "\n")
            with self.assertRaises(ValueError):
                worker.TrainingWorker(tmp)

    def test_reward_missing_stays_null_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            traj_path = self._trajectory(tmp)
            response = worker.TrainingWorker(tmp).handle(
                {"type": "train", "trajectory_path": str(traj_path)})
            metadata = json.loads(
                Path(response["checkpoint_path"]).read_text())
            self.assertEqual(metadata["reward"], {"name": None})

    def test_worker_rejects_semantically_bad_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            tw = worker.TrainingWorker(tmp)
            empty = Path(tmp) / "empty.jsonl"
            empty.write_text("")
            self.assertEqual(tw.handle(
                {"type": "train", "trajectory_path": str(empty)})["type"],
                "error")
            bad_seq = self._trajectory(tmp, corrupt_seq=True)
            self.assertEqual(tw.handle(
                {"type": "train", "trajectory_path": str(bad_seq)})["type"],
                "error")
            bad_hash = self._trajectory(tmp, corrupt_hash=True)
            self.assertEqual(tw.handle(
                {"type": "train",
                 "trajectory_path": str(bad_hash)})["type"], "error")
            bad_state = self._trajectory(tmp, terminal_state="zombie")
            self.assertEqual(tw.handle(
                {"type": "train",
                 "trajectory_path": str(bad_state)})["type"], "error")
            # Rejected requests consume no checkpoint sequence.
            self.assertEqual(tw.store.next_sequence, 0)

    def test_worker_handles_malformed_and_mixed_reward_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            tw = worker.TrainingWorker(tmp)
            malformed = Path(tmp) / "malformed.jsonl"
            malformed.write_text(json.dumps({
                "schema_version": 1, "terminal_state": "completed",
                "events": [], "output_text": "",
                "output_sha256": hashlib.sha256(b"").hexdigest(),
                "reward": "not-an-object"}) + "\n")
            response = tw.handle(
                {"type": "train", "trajectory_path": str(malformed)})
            self.assertEqual(response["type"], "error")

            mixed = Path(tmp) / "mixed.jsonl"
            lines = []
            for name in ("reward-a", "reward-b"):
                lines.append(json.dumps({
                    "schema_version": 1, "terminal_state": "completed",
                    "events": [], "output_text": "",
                    "output_sha256": hashlib.sha256(b"").hexdigest(),
                    "reward": {"name": name, "score": 1.0}}))
            mixed.write_text("\n".join(lines) + "\n")
            response = tw.handle(
                {"type": "train", "trajectory_path": str(mixed)})
            self.assertEqual(response["type"], "error")
            self.assertIn("reward", response["error"])
            self.assertEqual(tw.store.next_sequence, 0)

    def test_worker_runs_as_separate_process_over_stdio(self):
        with tempfile.TemporaryDirectory() as tmp:
            traj_path = self._trajectory(tmp, reward_name="constant-one")
            proc = subprocess.run(
                [sys.executable, "-I",
                 str(Path(__file__).resolve().parents[1]
                     / "python" / "umminfer_rl_worker.py"),
                 "--state-dir", str(Path(tmp) / "state")],
                input="\n".join([
                    json.dumps({"type": "ping"}),
                    json.dumps({"type": "train", "task_id": "t",
                                "trajectory_path": str(traj_path)}),
                    json.dumps({"type": "shuffle"}),
                ]) + "\n",
                capture_output=True, text=True, timeout=30, check=True)
            lines = [json.loads(line) for line in
                     proc.stdout.splitlines() if line.strip()]
            self.assertEqual(lines[0]["type"], "hello")
            self.assertEqual(lines[1]["type"], "pong")
            self.assertEqual(lines[1]["worker_protocol_version"],
                             worker.WORKER_PROTOCOL_VERSION)
            self.assertEqual(lines[2]["type"], "train_result")
            self.assertEqual(lines[2]["sequence"], 0)
            self.assertEqual(lines[3]["type"], "error")
            self.assertEqual(proc.stderr, "")


if __name__ == "__main__":
    sys.exit(unittest.main())

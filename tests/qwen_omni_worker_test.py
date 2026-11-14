#!/usr/bin/env python3
"""Protocol unit tests with a fake SGLang; this is not a model E2E test."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import base64
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
import unittest

import zmq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from umminfer_mlx_stage import pack_message, unpack_message
from umminfer_qwen_omni_worker import QwenOmniWorker, summarize_images


class FakeSGLang(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply_json(self, payload):
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def event(self, payload):
        self.wfile.write(b"data: " + json.dumps(payload).encode() + b"\n\n")
        self.wfile.flush()

    def chunk(self, content=None, finish=None):
        delta = {} if content is None else {"content": content}
        self.event({"choices": [{"index": 0, "delta": delta,
                                 "finish_reason": finish}]})

    def do_GET(self):
        if self.path != "/v1/models":
            self.send_error(404)
            return
        self.reply_json({"data": [{"id": "qwen3-omni"}]})

    def do_POST(self):
        if self.path == "/abort_request":
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            self.server.aborts.append(body.get("rid"))
            self.server.cancelled.set()
            self.reply_json({})
            return
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        self.server.requests.append(payload)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        cancel = "CANCEL" in json.dumps(payload["messages"])
        try:
            self.chunk()
            if cancel:
                self.chunk("first")
                for index in range(200):
                    time.sleep(0.02)
                    if self.server.cancelled.is_set():
                        # Same confirmation frame SGLang sends for a graceful
                        # /abort_request: a finish_reason=abort chunk.
                        self.chunk(finish="abort")
                        return
                    self.chunk("x")
                return
            for text in ("a", "", "b"):
                self.chunk(text)
            self.chunk(finish="stop")
            self.event({"choices": [], "usage": {"completion_tokens": 3}})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            if cancel:
                self.server.cancelled.set()

class OverlapSGLang(FakeSGLang):
    """Slow stream that records how many requests are inside it at once."""

    active = 0
    max_active = 0
    lock = threading.Lock()

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        json.loads(self.rfile.read(length))
        with OverlapSGLang.lock:
            OverlapSGLang.active += 1
            OverlapSGLang.max_active = max(OverlapSGLang.max_active,
                                           OverlapSGLang.active)
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.chunk("a")
            time.sleep(0.3)
            self.chunk("b")
            self.chunk(finish="stop")
            self.event({"choices": [], "usage": {"completion_tokens": 2}})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        finally:
            with OverlapSGLang.lock:
                OverlapSGLang.active -= 1


def forwarded_rids(requests):
    """The rid each chat request registered with the backend, in arrival order."""
    return [request.get("rid") for request in requests]


class WorkerProtocolTests(unittest.TestCase):
    def setUp(self):
        self.upstream = ThreadingHTTPServer(("127.0.0.1", 0), FakeSGLang)
        self.upstream.requests = []
        self.upstream.aborts = []
        self.upstream.cancelled = threading.Event()
        self.upstream_thread = threading.Thread(
            target=self.upstream.serve_forever, daemon=True)
        self.upstream_thread.start()

    def tearDown(self):
        self.upstream.shutdown()
        self.upstream.server_close()
        self.upstream_thread.join(3)

    @staticmethod
    def submit(request_id="r", stream=False, messages=None, **params):
        params["stream"] = stream
        return {
            "type": "submit", "request_id": request_id,
            "data": {"request": {
                "inputs": {"messages": messages or [
                    {"role": "user", "content": "hello"}]},
                "params": params,
            }},
        }

    def test_ready_stream_cancel_and_followup(self):
        context = zmq.Context()
        coordinator = context.socket(zmq.PULL)
        coordinator.RCVTIMEO = 5000
        coordinator.bind("tcp://127.0.0.1:*")
        abort = context.socket(zmq.PUB)
        abort.bind("tcp://127.0.0.1:*")
        base_url = f"http://127.0.0.1:{self.upstream.server_port}/v1"
        stage = QwenOmniWorker(
            "qwen", "tcp://127.0.0.1:*", coordinator.last_endpoint.decode(),
            abort.last_endpoint.decode(), base_url, "qwen3-omni")
        sender = None
        thread = None
        try:
            stage.start()
            self.assertEqual(unpack_message(coordinator.recv()), {
                "type": "stage_ready", "from_stage": "qwen"})
            sender = context.socket(zmq.PUSH)
            sender.connect(stage.recv_socket.last_endpoint.decode())
            thread = threading.Thread(target=stage.run)
            thread.start()

            messages = [{"role": "user", "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AA"}},
                # A remote URL is forwarded unchanged; SGLang fetches it.
                {"type": "image_url", "image_url": {
                    "url": "http://172.17.0.1:18080/man_ironing.png"}},
            ]}]
            sender.send(pack_message(self.submit(stream=True, messages=messages,
                                                 max_new_tokens=7, temperature=0.25)))
            received = [unpack_message(coordinator.recv()) for _ in range(3)]
            self.assertEqual([item["type"] for item in received],
                             ["stream", "stream", "complete"])
            self.assertEqual([item["chunk"]["text"] for item in received[:-1]],
                             ["a", "b"])
            self.assertEqual(received[-1]["result"], {
                "text": "ab", "generated_token_count": 3, "finish_reason": "stop"})
            forwarded = self.upstream.requests[0]
            self.assertEqual(forwarded["messages"], messages)
            self.assertEqual(forwarded["model"], "qwen3-omni")
            self.assertEqual(forwarded["max_tokens"], 7)
            self.assertEqual(forwarded["temperature"], 0.25)
            self.assertTrue(forwarded["stream"])
            self.assertEqual(forwarded["stream_options"], {"include_usage": True})

            sender.send(pack_message(self.submit(request_id="plain")))
            plain = unpack_message(coordinator.recv())
            self.assertEqual(plain["type"], "complete")
            self.assertEqual(plain["request_id"], "plain")
            self.assertEqual(plain["result"]["text"], "ab")

            sender.send(pack_message(self.submit(request_id="cancel", stream=True,
                                                 messages=[{"role": "user",
                                                            "content": "CANCEL"}])))
            first = unpack_message(coordinator.recv())
            self.assertEqual(first["request_id"], "cancel")
            self.assertEqual(first["chunk"]["text"], "first")
            sender.send(pack_message({"type": "abort", "request_id": "cancel"}))
            self.assertTrue(self.upstream.cancelled.wait(5),
                            "abort did not reach the upstream")
            # The worker must address the same rid the chat request registered.
            rids = [request["rid"] for request in self.upstream.requests]
            self.assertEqual(len(self.upstream.aborts), 1)
            self.assertEqual(self.upstream.aborts[0], rids[2])
            self.assertTrue(rids[0].startswith("r@"))
            self.assertTrue(rids[1].startswith("plain@"))
            self.assertTrue(rids[2].startswith("cancel@"))
            # Every rid carries the per-process tag, so restarts cannot collide.
            self.assertEqual(len({rid.split("@", 1)[1] for rid in rids}), 1)

            sender.send(pack_message(self.submit(request_id="after")))
            while True:
                result = unpack_message(coordinator.recv())
                if result["type"] == "complete":
                    break
            self.assertEqual(result["request_id"], "after")
            self.assertTrue(result["success"])
            self.assertEqual(result["result"]["text"], "ab")

            sender.send(pack_message({"type": "shutdown"}))
            thread.join(5)
            self.assertFalse(thread.is_alive())
        finally:
            stage.running = False
            if thread is not None:
                thread.join(3)
            stage.close()
            if thread is not None:
                thread.join(1)
            if sender is not None:
                sender.close(linger=0)
            coordinator.close(linger=0)
            abort.close(linger=0)
            context.term()

    def test_concurrency_runs_two_upstream_requests_at_once(self):
        """A serial worker would reach max_active == 1 and take two waves."""
        OverlapSGLang.active = 0
        OverlapSGLang.max_active = 0
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), OverlapSGLang)
        upstream_thread = threading.Thread(target=upstream.serve_forever,
                                           daemon=True)
        upstream_thread.start()
        context = zmq.Context()
        coordinator = context.socket(zmq.PULL)
        coordinator.RCVTIMEO = 5000
        coordinator.bind("tcp://127.0.0.1:*")
        abort = context.socket(zmq.PUB)
        abort.bind("tcp://127.0.0.1:*")
        base_url = f"http://127.0.0.1:{upstream.server_port}/v1"
        stage = QwenOmniWorker(
            "qwen", "tcp://127.0.0.1:*", coordinator.last_endpoint.decode(),
            abort.last_endpoint.decode(), base_url, "qwen3-omni",
            concurrency=2)
        sender = None
        run_thread = None
        try:
            stage.start()
            self.assertEqual(unpack_message(coordinator.recv()),
                             {"type": "stage_ready", "from_stage": "qwen"})
            sender = context.socket(zmq.PUSH)
            sender.connect(stage.recv_socket.last_endpoint.decode())
            run_thread = threading.Thread(target=stage.run)
            run_thread.start()
            sender.send(pack_message(self.submit(request_id="one")))
            sender.send(pack_message(self.submit(request_id="two")))
            completions = []
            while len(completions) < 2:
                message = unpack_message(coordinator.recv())
                if message["type"] == "complete":
                    completions.append(message)
            self.assertEqual({m["request_id"] for m in completions},
                             {"one", "two"})
            self.assertTrue(all(m["success"] for m in completions))
            # Both requests were inside the backend at the same time.
            self.assertEqual(OverlapSGLang.max_active, 2)
            sender.send(pack_message({"type": "shutdown"}))
            run_thread.join(5)
            self.assertFalse(run_thread.is_alive())
        finally:
            stage.running = False
            if run_thread is not None:
                run_thread.join(3)
            stage.close()
            if sender is not None:
                sender.close(linger=0)
            coordinator.close(linger=0)
            abort.close(linger=0)
            context.term()
            upstream.shutdown()
            upstream.server_close()
            upstream_thread.join(3)

    def test_bad_content_fails_without_upstream_request(self):
        stage = QwenOmniWorker("qwen", "in", "out", "abort",
                               "http://127.0.0.1:1/v1", "qwen3-omni")
        sent = []
        stage.coordinator_socket = type("Sink", (), {
            "send": lambda self, data: sent.append(unpack_message(data))})()
        stage.generate = lambda *args: self.fail("bad request reached upstream")
        stage.process(self.submit(messages=[{"role": "user", "content": [None]}]))
        self.assertFalse(sent[-1]["success"])
        self.assertIn("content part", sent[-1]["error"])
        self.assertEqual(self.upstream.requests, [])


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3

import base64
import io
import math
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest import mock
import wave

import zmq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from umminfer_mlx_stage import pack_message, unpack_message
from umminfer_qwen_asr_stage import QwenASRStage, SGLangBackend, pcm16_wav_data_url


PCM = b"\x01\x00\xff\x7f\x00\x80"


def submit(request_id="r", audio=PCM, **params):
    params.setdefault("stream", False)
    return {
        "type": "submit", "request_id": request_id,
        "data": {
            "_type": "StagePayload", "request_id": request_id,
            "request": {
                "_type": "OmniRequest", "inputs": {"audio": audio},
                "params": params, "metadata": {},
            },
            "data": None,
        },
    }


def response(text="actual 中文", count=4, reason="stop"):
    return SimpleNamespace(text=text, generation_tokens=count, finish_reason=reason)


class FakeCoordinator:
    def __init__(self):
        self.messages = []

    def send(self, data):
        self.messages.append(unpack_message(data))


class FakeBackend:
    def __init__(self, outcomes=()):
        self.outcomes = list(outcomes)
        self.calls = []
        self.checked = 0

    def check_model(self):
        self.checked += 1

    def close_active(self):
        pass

    def prepare_request(self):
        pass

    def transcribe(self, pcm, max_tokens, temperature, request_id):
        self.calls.append((pcm, max_tokens, temperature, request_id))
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return response()


class BackendTests(unittest.TestCase):
    def test_pcm16_wav_round_trip(self):
        data_url = pcm16_wav_data_url(PCM)
        prefix, encoded = data_url.split(",", 1)
        self.assertEqual(prefix, "data:audio/wav;base64")
        with wave.open(io.BytesIO(base64.b64decode(encoded)), "rb") as wav:
            self.assertEqual((wav.getnchannels(), wav.getsampwidth(),
                              wav.getframerate()), (1, 2, 16000))
            self.assertEqual(wav.readframes(wav.getnframes()), PCM)
        for invalid in (None, bytearray(PCM), b"", b"\x00"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                pcm16_wav_data_url(invalid)

    def test_generate_payload_and_real_metadata_shape(self):
        backend = SGLangBackend("http://127.0.0.1:13100", "qwen3-asr")
        calls = []

        def request(method, path, payload=None):
            calls.append((method, path, payload))
            return {
                "text": "Chinese<asr_text>甚至出现交易几乎停滞的情况。",
                "meta_info": {
                    "completion_tokens": 11, "prompt_tokens": 61,
                    "finish_reason": {"type": "stop"}, "e2e_latency": 0.109,
                },
            }

        backend._request_json = request
        result = backend.transcribe(PCM, 73, 0.0, "umminfer-req-1")
        method, path, payload = calls[0]
        self.assertEqual((method, path), ("POST", "/generate"))
        self.assertEqual(payload["rid"], "umminfer-req-1")
        self.assertEqual(payload["sampling_params"],
                         {"temperature": 0.0, "max_new_tokens": 73})
        self.assertEqual(payload["modalities"], ["audio"])
        self.assertFalse(payload["stream"])
        encoded = payload["audio_data"].split(",", 1)[1]
        with wave.open(io.BytesIO(base64.b64decode(encoded)), "rb") as wav:
            self.assertEqual(wav.readframes(wav.getnframes()), PCM)
        self.assertEqual(result.text, "甚至出现交易几乎停滞的情况。")
        self.assertEqual((result.generation_tokens, result.finish_reason), (11, "stop"))
        self.assertEqual((result.prompt_tokens, result.backend_e2e_seconds), (61, 0.109))

    def test_missing_or_malformed_metadata_fails(self):
        fixtures = [
            {"text": "x"},
            {"text": "x", "meta_info": {}},
            {"text": "x", "meta_info": {"completion_tokens": True,
                                       "finish_reason": {"type": "stop"}}},
            {"text": "x", "meta_info": {"completion_tokens": 1,
                                       "finish_reason": None}},
            {"text": "x", "meta_info": {"completion_tokens": 1,
                                       "finish_reason": {"type": ""}}},
            {"text": "partial", "meta_info": {"completion_tokens": 2,
                                             "finish_reason": {"type": "abort"}}},
        ]
        for body in fixtures:
            backend = SGLangBackend("http://127.0.0.1", "qwen3-asr")
            backend._request_json = lambda *args, body=body, **kwargs: body
            with self.subTest(body=body), self.assertRaises(ValueError):
                backend.transcribe(PCM, 10, 0.0, "r")

    def test_model_ready_check(self):
        backend = SGLangBackend("http://127.0.0.1", "qwen3-asr")
        backend._request_json = lambda *args: {"data": [{"id": "qwen3-asr"}]}
        backend.check_model()
        backend._request_json = lambda *args: {"data": [{"id": "other"}]}
        with self.assertRaises(ValueError):
            backend.check_model()
    def test_cancel_before_http_registration_does_not_send(self):
        backend = SGLangBackend("http://127.0.0.1", "qwen3-asr")
        calls = []

        class Connection:
            sock = None

            def request(self, *args, **kwargs):
                calls.append((args, kwargs))

            def close(self):
                pass

        backend.close_active()
        with mock.patch("umminfer_qwen_asr_stage.http.client.HTTPConnection",
                        return_value=Connection()):
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                backend._request_json("POST", "/generate", {})
        self.assertEqual(calls, [])
        self.assertIsNone(backend._active)


class StageTests(unittest.TestCase):
    def make_stage(self, backend=None):
        stage = QwenASRStage("test", "in", "out", "abort", "qwen3-asr",
                             "http://127.0.0.1:13100")
        stage.backend = backend or FakeBackend()
        stage.generate = stage._generate
        stage.coordinator_socket = FakeCoordinator()
        stage.poll_interval = 0.01
        return stage

    def test_complete_unicode_and_repeated_requests(self):
        stage = self.make_stage()
        for request_id in ("first", "second"):
            stage.process(submit(request_id=request_id, max_new_tokens=37))
            result = stage.coordinator_socket.messages[-1]
            self.assertEqual(result, {
                "type": "complete", "request_id": request_id, "from_stage": "test",
                "success": True,
                "result": {"text": "actual 中文", "generated_token_count": 4,
                           "finish_reason": "stop"},
                "error": None,
            })
        self.assertEqual(stage.coordinator_socket.messages[0]["result"],
                         stage.coordinator_socket.messages[1]["result"])
        self.assertEqual(stage.backend.calls, [
            (PCM, 37, 0.0, "umminfer-first"),
            (PCM, 37, 0.0, "umminfer-second"),
        ])

    def test_bad_audio_and_params_fail_only_request(self):
        stage = self.make_stage()
        invalid = [submit(audio=value) for value in (None, bytearray(PCM), b"", b"\x00")]
        invalid += [submit(max_new_tokens=value) for value in (True, 0, 4097, "3")]
        invalid += [submit(temperature=value) for value in (True, math.nan, -0.1, 0.1)]
        invalid += [submit(stream=True), submit(stream=1)]
        extra = submit()
        extra["data"]["request"]["inputs"]["language"] = "zh"
        invalid.append(extra)
        for message in invalid:
            with self.subTest(message=message):
                stage.process(message)
                result = stage.coordinator_socket.messages.pop()
                self.assertFalse(result["success"])
                self.assertTrue(result["error"])
        self.assertEqual(stage.backend.calls, [])
        stage.process(submit(request_id="next"))
        self.assertTrue(stage.coordinator_socket.messages[-1]["success"])

    def test_backend_failure_then_success(self):
        stage = self.make_stage(FakeBackend([RuntimeError("backend failed")]))
        stage.process(submit(request_id="failed"))
        self.assertFalse(stage.coordinator_socket.messages[-1]["success"])
        stage.process(submit(request_id="next"))
        self.assertTrue(stage.coordinator_socket.messages[-1]["success"])

    def test_new_request_clears_previous_cancellation(self):
        backend = SGLangBackend("http://127.0.0.1", "qwen3-asr")
        calls = []

        def request(method, path, payload=None):
            self.assertFalse(backend._cancelled.is_set())
            calls.append((method, path, payload))
            return {"text": "ok", "meta_info": {
                "completion_tokens": 1, "finish_reason": {"type": "stop"}}}

        backend._request_json = request
        stage = self.make_stage(backend)
        backend.close_active()
        stage.process(submit())
        self.assertTrue(stage.coordinator_socket.messages[-1]["success"])
        self.assertEqual(len(calls), 1)

    def test_abort_closes_http_and_followup_succeeds(self):
        class BlockingBackend(FakeBackend):
            def __init__(self):
                super().__init__()
                self.started = threading.Event()
                self.cancelled = threading.Event()

            def close_active(self):
                self.cancelled.set()

            def transcribe(self, pcm, max_tokens, temperature, request_id):
                self.calls.append((pcm, max_tokens, temperature, request_id))
                self.started.set()
                if not self.cancelled.wait(3):
                    raise RuntimeError("fake backend was not closed")
                raise RuntimeError("cancelled")

        stage = self.make_stage(BlockingBackend())
        thread = threading.Thread(target=stage.process, args=(submit(),))
        thread.start()
        self.assertTrue(stage.backend.started.wait(1))
        stage.record_abort("r")
        thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertTrue(stage.backend.cancelled.is_set())
        self.assertEqual(stage.coordinator_socket.messages, [])
        stage.backend = FakeBackend()
        stage.process(submit(request_id="after-abort"))
        result = stage.coordinator_socket.messages[-1]
        self.assertTrue(result["success"])
        self.assertEqual(result["request_id"], "after-abort")

    def test_real_zmq_ready_submit_and_shutdown(self):
        context = zmq.Context()
        coordinator = context.socket(zmq.PULL)
        coordinator.RCVTIMEO = 3000
        coordinator.bind("tcp://127.0.0.1:*")
        abort = context.socket(zmq.PUB)
        abort.bind("tcp://127.0.0.1:*")
        stage = self.make_stage()
        stage.generate = None
        stage.recv_endpoint = "tcp://127.0.0.1:*"
        stage.coordinator_endpoint = coordinator.last_endpoint.decode()
        stage.abort_endpoint = abort.last_endpoint.decode()
        thread = None
        sender = None
        try:
            stage.start()
            self.assertEqual(unpack_message(coordinator.recv()),
                             {"type": "stage_ready", "from_stage": "test"})
            self.assertEqual(stage.backend.checked, 1)
            sender = context.socket(zmq.PUSH)
            sender.connect(stage.recv_socket.last_endpoint.decode())
            thread = threading.Thread(target=stage.run)
            thread.start()
            sender.send(pack_message(submit()))
            result = unpack_message(coordinator.recv())
            self.assertTrue(result["success"])
            self.assertEqual(result["result"]["text"], "actual 中文")
            sender.send(pack_message({"type": "shutdown"}))
            thread.join(3)
            self.assertFalse(thread.is_alive())
        finally:
            stage.running = False
            if thread is not None:
                thread.join(3)
            stage.close()
            if sender is not None:
                sender.close(linger=0)
            coordinator.close(linger=0)
            abort.close(linger=0)
            context.term()


if __name__ == "__main__":
    unittest.main(verbosity=2)

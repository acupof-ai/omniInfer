#!/usr/bin/env python3

from collections import deque
import contextlib
import inspect
import math
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace
import unittest

import zmq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import umminfer_mlx_stage
from umminfer_mlx_stage import MLXStage, pack_message, unpack_message


class FakeTokenizer:
    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        self.messages = messages
        return "fake prompt"


class FakeCoordinator:
    def __init__(self):
        self.messages = []

    def send(self, data):
        self.messages.append(unpack_message(data))


def submit(request_id="r", stream=False, **params):
    params["stream"] = stream
    messages = [{"role": "user", "content": "hi"}]
    return {
        "type": "submit",
        "request_id": request_id,
        "data": {
            "_type": "StagePayload",
            "request_id": request_id,
            "request": {
                "_type": "OmniRequest",
                "inputs": {"messages": messages},
                "params": params,
                "metadata": {},
            },
            "data": None,
        },
    }


def response(text, generation_tokens, finish_reason=None):
    """One MLX-LM GenerationResponse, with the fields the stage reads."""
    return SimpleNamespace(text=text, generation_tokens=generation_tokens,
                           finish_reason=finish_reason)


def stream_responses(deltas, finish_reason="stop"):
    """Behave like mlx_lm.stream_generate: only the last response is final.

    Intermediate responses report finish_reason None. The last one carries the
    reason and the count of tokens generated so far, as stream_generate does.
    """
    for index, delta in enumerate(deltas):
        final = index == len(deltas) - 1
        yield response(delta, index + 1, finish_reason if final else None)


class StageTests(unittest.TestCase):
    def make_stage(self, deltas=(), finish_reason="stop"):
        stage = MLXStage("test", "in", "out", "abort", "unused")
        stage.tokenizer = FakeTokenizer()
        stage.calls = []
        stage.coordinator_socket = FakeCoordinator()

        def generate(prompt, max_tokens, temperature):
            stage.calls.append((prompt, max_tokens, temperature))
            yield from stream_responses(deltas, finish_reason)

        stage.generate = generate
        return stage

    def test_load_once(self):
        calls = []
        tokenizer = FakeTokenizer()

        def loader(path):
            calls.append(path)
            return tokenizer, lambda prompt, maximum, temperature: iter(())

        stage = MLXStage("test", "in", "out", "abort", "model", loader)
        stage.load()
        stage.load()
        self.assertEqual(calls, ["model"])
        self.assertIs(stage.tokenizer, tokenizer)

    def test_wire_roundtrip(self):
        message = {"type": "complete", "request_id": "r", "success": True,
                   "result": {"text": "你好"}, "error": None}
        self.assertEqual(unpack_message(pack_message(message)), message)

    def test_non_streaming(self):
        stage = self.make_stage(["hello", " world"])
        stage.process(submit(max_new_tokens=7, temperature=0.25))
        self.assertEqual(stage.calls, [("fake prompt", 7, 0.25)])
        self.assertEqual(len(stage.coordinator_socket.messages), 1)
        result = stage.coordinator_socket.messages[0]
        self.assertEqual(result, {
            "type": "complete", "request_id": "r", "from_stage": "test",
            "success": True,
            "result": {"text": "hello world", "generated_token_count": 2,
                       "finish_reason": "stop"},
            "error": None})

    def test_default_params_and_chat_template(self):
        stage = self.make_stage(["ok"])
        message = submit()
        stage.process(message)
        self.assertEqual(stage.calls, [("fake prompt", 256, 0.0)])
        self.assertEqual(stage.tokenizer.messages,
                         [{"role": "user", "content": "hi"}])

    def test_streaming_deltas(self):
        stage = self.make_stage(["a", "", "b"])
        stage.process(submit(stream=True))
        messages = stage.coordinator_socket.messages
        self.assertEqual([item["type"] for item in messages],
                         ["stream", "stream", "complete"])
        self.assertEqual([item["chunk"]["text"] for item in messages[:-1]],
                         ["a", "b"])
        self.assertEqual(messages[-1]["result"],
                         {"text": "ab", "generated_token_count": 3,
                          "finish_reason": "stop"})

    def test_bad_params_fail_only_request(self):
        stage = self.make_stage(["unused"])
        invalid = [submit(max_new_tokens=True), submit(max_new_tokens=0),
                   submit(max_new_tokens="3"), submit(temperature=True),
                   submit(temperature=math.nan), submit(temperature=-0.1),
                   submit(stream=1)]
        for message in invalid:
            with self.subTest(message=message):
                stage.process(message)
                result = stage.coordinator_socket.messages.pop()
                self.assertFalse(result["success"])
                self.assertTrue(result["error"])
        self.assertEqual(stage.calls, [])

    def test_bad_request_shapes_fail_only_request(self):
        stage = self.make_stage(["ok"])
        invalid = [submit() for _ in range(5)]
        invalid[0]["data"]["request"] = []
        invalid[1]["data"]["request"]["inputs"] = []
        invalid[2]["data"]["request"]["inputs"]["messages"] = {}
        invalid[3]["data"]["request"]["inputs"]["messages"] = []
        invalid[4]["data"]["request"]["inputs"]["messages"] = ["not a map"]
        for message in invalid:
            with self.subTest(message=message):
                stage.process(message)
                result = stage.coordinator_socket.messages.pop()
                self.assertFalse(result["success"])
                self.assertTrue(result["error"])
        stage.process(submit(request_id="next"))
        self.assertTrue(stage.coordinator_socket.messages[-1]["success"])
        self.assertEqual(stage.calls, [("fake prompt", 256, 0.0)])

    def test_generation_failure(self):
        stage = self.make_stage()

        def generate(prompt, maximum, temperature):
            yield SimpleNamespace(text="partial")
            raise RuntimeError("boom")

        stage.generate = generate
        stage.process(submit(stream=True))
        messages = stage.coordinator_socket.messages
        self.assertEqual(messages[0]["chunk"]["text"], "partial")
        self.assertFalse(messages[-1]["success"])
        self.assertEqual(messages[-1]["error"], "boom")
        stage.generate = lambda prompt, maximum, temperature: iter(
            [response("next", 1, "length")])
        stage.process(submit(request_id="after-error"))
        result = stage.coordinator_socket.messages[-1]
        self.assertTrue(result["success"])
        self.assertEqual(result["request_id"], "after-error")
        self.assertEqual(result["result"], {"text": "next",
                                            "generated_token_count": 1,
                                            "finish_reason": "length"})

    def test_length_finish_reason_is_propagated(self):
        stage = self.make_stage(["partial"], finish_reason="length")
        stage.process(submit(max_new_tokens=1))
        result = stage.coordinator_socket.messages[-1]
        self.assertTrue(result["success"])
        self.assertEqual(result["result"]["finish_reason"], "length")
        self.assertEqual(result["result"]["generated_token_count"], 1)

    def test_empty_finish_reason_becomes_unknown(self):
        stage = self.make_stage(["ok"], finish_reason="")
        stage.process(submit())
        result = stage.coordinator_socket.messages[-1]
        self.assertTrue(result["success"])
        self.assertEqual(result["result"]["finish_reason"], "unknown")

    def test_missing_finish_reason_fails_the_request(self):
        stage = self.make_stage()

        def generate(prompt, maximum, temperature):
            yield SimpleNamespace(text="ok")

        stage.generate = generate
        stage.process(submit())
        result = stage.coordinator_socket.messages[-1]
        self.assertFalse(result["success"])
        self.assertIn("finish_reason", result["error"])

    def test_unreported_generation_tokens_fail_the_request(self):
        stage = self.make_stage()

        def generate(prompt, maximum, temperature):
            yield SimpleNamespace(text="a", finish_reason=None)
            yield SimpleNamespace(text="b", finish_reason="stop")

        stage.generate = generate
        stage.process(submit())
        result = stage.coordinator_socket.messages[-1]
        self.assertFalse(result["success"])
        self.assertIn("generation_tokens", result["error"])

    def test_invalid_generation_tokens_fail_the_request(self):
        for bad in (True, -1, "3", 1.0, None):
            with self.subTest(generation_tokens=bad):
                stage = self.make_stage()
                stage.generate = lambda prompt, maximum, temperature: iter(
                    [response("x", bad, "stop")])
                stage.process(submit())
                result = stage.coordinator_socket.messages[-1]
                self.assertFalse(result["success"])
                self.assertIn("generation_tokens", result["error"])

    def test_repeated_requests_produce_identical_output(self):
        stage = self.make_stage(["fixed ", "text"])
        stage.process(submit(request_id="first"))
        first = stage.coordinator_socket.messages[-1]["result"]
        stage.process(submit(request_id="second"))
        second = stage.coordinator_socket.messages[-1]["result"]
        self.assertEqual(first, second)

    def test_abort_stops_before_success(self):
        stage = self.make_stage(["a", "b"])
        polls = []

        def poll(timeout_ms=0):
            polls.append(timeout_ms)
            if len(polls) == 2:
                stage.record_abort("r")

        stage.poll_events = poll
        stage.process(submit(stream=True))
        messages = stage.coordinator_socket.messages
        self.assertEqual([item["type"] for item in messages], ["stream"])
        self.assertNotIn("r", stage.aborted)

    def test_late_abort_suppresses_complete(self):
        stage = self.make_stage(["only token"])
        polls = []

        def poll(timeout_ms=0):
            polls.append(timeout_ms)
            if len(polls) == 2:
                stage.record_abort("r")

        stage.poll_events = poll
        stage.process(submit(stream=True))
        messages = stage.coordinator_socket.messages
        self.assertEqual([item["type"] for item in messages], ["stream"])
        self.assertEqual(messages[0]["chunk"], {"text": "only token"})
        self.assertEqual(len(polls), 2)

    def test_abort_before_submit_skips_generation(self):
        stage = self.make_stage(["unused"])
        stage.record_abort("r")
        self.assertIn("r", stage.aborted)
        stage.process(submit())
        self.assertEqual(stage.calls, [])
        self.assertEqual(stage.coordinator_socket.messages, [])
        self.assertNotIn("r", stage.aborted)
        stage.process(submit(request_id="next"))
        self.assertEqual(stage.calls, [("fake prompt", 256, 0.0)])
        result = stage.coordinator_socket.messages[-1]
        self.assertTrue(result["success"])
        self.assertEqual(result["request_id"], "next")

    def test_queued_abort_removes_the_pending_submit(self):
        stage = self.make_stage(["unused"])
        stage.pending.append(submit(request_id="queued"))
        stage.pending.append(submit(request_id="kept"))
        stage.record_abort("queued")
        self.assertEqual([item["request_id"] for item in stage.pending], ["kept"])

        while stage.pending:
            stage.process(stage.pending.popleft())
        self.assertEqual(stage.calls, [("fake prompt", 256, 0.0)])
        result = stage.coordinator_socket.messages[-1]
        self.assertEqual(result["request_id"], "kept")
        self.assertTrue(result["success"])

    def test_abort_tombstone_expires_without_a_submit(self):
        stage = self.make_stage(["ok"])
        stage.record_abort("late")
        stage.aborted["late"] = 0.0
        stage.expire_aborts()
        self.assertNotIn("late", stage.aborted)
        # An expired tombstone does not suppress the submit that follows.
        stage.process(submit(request_id="late"))
        self.assertTrue(stage.coordinator_socket.messages[-1]["success"])

    def test_abort_decisions_do_not_depend_on_queue_capacity(self):
        stage = self.make_stage(["unused"])
        # Cancellation must not be evicted by traffic to other requests.
        for index in range(1025):
            stage.pending.append(submit(request_id=f"queued-{index}"))
        stage.record_abort("queued-1024")
        self.assertEqual(len(stage.pending), 1024)
        self.assertIn("queued-1024", stage.aborted)

        for index in range(1025):
            stage.record_abort(f"early-{index}")
        # All of them survive, next to the one recorded above.
        self.assertEqual(set(stage.aborted),
                         {f"early-{index}" for index in range(1025)}
                         | {"queued-1024"})
        for index in range(1025):
            stage.process(submit(request_id=f"early-{index}"))
        self.assertEqual(stage.calls, [])
        self.assertEqual(stage.coordinator_socket.messages, [])
        # Every tombstone was released by the submit it cancelled. The queued
        # one never had a submit, so it stays until its deadline.
        self.assertEqual(list(stage.aborted), ["queued-1024"])

    def test_shutdown_stops_generation(self):
        stage = self.make_stage(["a"])

        def poll(timeout_ms=0):
            stage.running = False

        stage.poll_events = poll
        stage.process(submit(stream=True))
        self.assertEqual(stage.coordinator_socket.messages, [])

    def test_real_zmq_submit_and_shutdown(self):
        context = zmq.Context()
        coordinator = context.socket(zmq.PULL)
        coordinator.RCVTIMEO = 3000
        coordinator.bind("tcp://127.0.0.1:*")
        abort = context.socket(zmq.PUB)
        abort.bind("tcp://127.0.0.1:*")
        loads = []
        generations = []

        def loader(path):
            loads.append(path)

            def generate(prompt, maximum, temperature):
                generations.append(prompt)
                yield response("ok", 1, "stop")

            return FakeTokenizer(), generate

        stage = MLXStage("test", "tcp://127.0.0.1:*",
                         coordinator.last_endpoint.decode(),
                         abort.last_endpoint.decode(), "fake", loader)
        thread = None
        sender = None
        try:
            stage.start()
            ready = unpack_message(coordinator.recv())
            self.assertEqual(ready, {"type": "stage_ready", "from_stage": "test"})
            sender = context.socket(zmq.PUSH)
            sender.connect(stage.recv_socket.last_endpoint.decode())
            thread = threading.Thread(target=stage.run)
            thread.start()
            for _ in range(2):
                sender.send(pack_message(submit()))
                result = unpack_message(coordinator.recv())
                self.assertTrue(result["success"])
                self.assertEqual(result["result"]["text"], "ok")
            self.assertEqual(loads, ["fake"])
            sender.send(pack_message({"type": "abort", "request_id": "skipped"}))
            sender.send(pack_message({"type": "abort", "request_id": "skipped"}))
            sender.send(pack_message(submit(request_id="skipped")))
            sender.send(pack_message(submit(request_id="after-abort")))
            result = unpack_message(coordinator.recv())
            self.assertEqual(result["request_id"], "after-abort")
            self.assertTrue(result["success"])
            self.assertEqual(len(generations), 3)
            self.assertNotIn("skipped", stage.aborted)
            sender.send(pack_message({"type": "shutdown"}))
            thread.join(3)
            self.assertFalse(thread.is_alive())
            sender.close(linger=0)
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


class RebuildingTokenizer:
    """Like transformers: every `get_vocab` call builds a new map."""

    def __init__(self, vocabulary=None):
        self.vocabulary = vocabulary or {"hi": 1}
        self.accesses = 0
        self.builds = 0
        # load_mlx_backend caches on this inner object, as transformers nests it.
        self._tokenizer = self

    def get_vocab(self):
        self.accesses += 1
        self.builds += 1
        return dict(self.vocabulary)


@contextlib.contextmanager
def fake_mlx_lm(tokenizer, generated):
    """Stand-in for mlx_lm, which reads the vocabulary once per stream_generate."""
    mlx_lm = ModuleType("mlx_lm")
    sample_utils = ModuleType("mlx_lm.sample_utils")

    def load(path):
        return object(), tokenizer

    def stream_generate(model, tokenizer, prompt, max_tokens, sampler):
        # StreamingDetokenizer's token map reads the vocabulary on construction.
        generated.append(tokenizer._tokenizer.get_vocab())
        yield response("cached ", 1, None)
        yield response("output", 2, "stop")

    def make_sampler(temp):
        return {"temp": temp}

    mlx_lm.load = load
    mlx_lm.stream_generate = stream_generate
    mlx_lm.sample_utils = sample_utils
    sample_utils.make_sampler = make_sampler

    saved = {module.__name__: sys.modules.get(module.__name__)
             for module in (mlx_lm, sample_utils)}
    sys.modules["mlx_lm"] = mlx_lm
    sys.modules["mlx_lm.sample_utils"] = sample_utils
    try:
        yield
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class VocabCacheTests(unittest.TestCase):
    def test_vocabulary_is_built_once_at_load(self):
        tokenizer = RebuildingTokenizer()
        with fake_mlx_lm(tokenizer, []):
            umminfer_mlx_stage.load_mlx_backend("model")
        self.assertEqual(tokenizer.builds, 1)

    def test_generation_does_not_rebuild_the_vocabulary(self):
        tokenizer = RebuildingTokenizer()
        reads = []
        with fake_mlx_lm(tokenizer, reads):
            _, generate = umminfer_mlx_stage.load_mlx_backend("model")
            for _ in range(3):
                list(generate("p", 8, 0.0))
        # Each generate still reads the map, as the detokenizer does.
        self.assertEqual(len(reads), 3)
        self.assertEqual({id(item) for item in reads}, {id(reads[0])})
        # It is built once. Every later read answers from the cache.
        self.assertEqual(tokenizer.builds, 1)

    def test_repeated_requests_return_identical_output(self):
        tokenizer = RebuildingTokenizer()
        with fake_mlx_lm(tokenizer, []):
            _, generate = umminfer_mlx_stage.load_mlx_backend("model")
            texts = ["".join(item.text for item in generate("p", 8, 0.0))
                     for _ in range(3)]
        self.assertEqual(texts, ["cached output"] * 3)

    def test_a_tokenizer_without_the_inner_map_is_used_unchanged(self):
        class Plain:
            def apply_chat_template(self, *args, **kwargs):
                return "p"

        with fake_mlx_lm(Plain(), []):
            tokenizer, _ = umminfer_mlx_stage.load_mlx_backend("model")
        self.assertIsInstance(tokenizer, Plain)


class ResultMetadataContractTests(unittest.TestCase):
    """The base signature and the ASR override must stay one contract.

    tests/client_boundary_e2e.py runs the real ASR path against the unified
    binary. This check is the cheap guard for the failure that path reported:
    a base or override that takes fewer arguments than the caller passes.
    """

    def test_tiny_asr_override_matches_the_base_signature(self):
        try:
            import umminfer_tiny_asr_stage
        except ImportError as exc:
            self.skipTest(f"tiny ASR dependencies are not installed: {exc}")
        base = list(inspect.signature(MLXStage.result_metadata).parameters)
        override = list(inspect.signature(
            umminfer_tiny_asr_stage.TinyASRStage.result_metadata).parameters)
        self.assertEqual(override, base, "result_metadata signatures differ")
        self.assertEqual(base, ["self", "parts", "max_tokens", "last_response"])


if __name__ == "__main__":
    unittest.main()

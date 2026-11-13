#!/usr/bin/env python3
"""Unit tests for the benchmark harness helpers.

These tests use synthetic bytes only to exercise parsing and aggregation. They
never contact a model and are not benchmark evidence.
"""

import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import wave


def load_harness():
    path = Path(__file__).resolve().with_name("unified_benchmark.py")
    spec = importlib.util.spec_from_file_location("unified_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = load_harness()


class PercentileTests(unittest.TestCase):
    def test_nearest_rank(self):
        values = [10, 20, 30, 40, 50]
        self.assertEqual(bench.percentile(values, 50), 30)
        self.assertEqual(bench.percentile(values, 95), 50)
        self.assertEqual(bench.percentile([7], 95), 7)
        self.assertIsNone(bench.percentile([], 50))

    def test_cv(self):
        self.assertIsNone(bench.coefficient_of_variation([1.0]))
        self.assertAlmostEqual(
            bench.coefficient_of_variation([100.0, 100.0]), 0.0)
        self.assertGreater(
            bench.coefficient_of_variation([100.0, 200.0]), 0.1)
        self.assertIsNone(bench.coefficient_of_variation([0.0, 0.0]))


class CapTests(unittest.TestCase):
    def test_keeps_tail_only(self):
        buffer = bytearray()
        bench.capped_append(buffer, b"a" * 40, cap=16)
        bench.capped_append(buffer, b"b" * 40, cap=16)
        self.assertEqual(len(buffer), 16)
        self.assertEqual(bytes(buffer), b"b" * 16)


class InputTests(unittest.TestCase):
    def test_image_mime(self):
        self.assertEqual(bench.image_mime(b"\xff\xd8\xff\xe0"), "image/jpeg")
        self.assertEqual(
            bench.image_mime(b"\x89PNG\r\n\x1a\n"), "image/png")
        with self.assertRaises(ValueError):
            bench.image_mime(b"GIF89a")

    def test_wav_format_is_enforced(self):
        path = Path(tempfile.mkdtemp()) / "ok.wav"
        with wave.open(str(path), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(16000)
            out.writeframes(b"\x00\x00" * 16)
        with wave.open(io.BytesIO(bench.wav16k_mono_pcm16(path)), "rb") as out:
            self.assertEqual(
                (out.getnchannels(), out.getsampwidth(), out.getframerate()),
                (1, 2, 16000))
            self.assertEqual(out.readframes(out.getnframes()), b"\x00\x00" * 16)

        bad = Path(tempfile.mkdtemp()) / "stereo.wav"
        with wave.open(str(bad), "wb") as out:
            out.setnchannels(2)
            out.setsampwidth(2)
            out.setframerate(16000)
            out.writeframes(b"\x00\x00\x00\x00" * 8)
        with self.assertRaises(ValueError):
            bench.wav16k_mono_pcm16(bad)

    def test_bodies(self):
        body = bench.chat_body("m", "hi", 256)
        self.assertEqual(body["messages"][0]["content"], "hi")
        self.assertFalse(body["stream"] is False)
        unified = bench.unified_body("m", b"\x00\x01", 256)
        part = unified["messages"][0]["content"][0]
        self.assertEqual(part["type"], "input_audio")
        self.assertEqual(part["input_audio"]["format"], "wav")
        self.assertEqual(part["input_audio"]["data"], "AAE=")


class SseTests(unittest.TestCase):
    def test_events(self):
        lines = [
            "", ": keepalive",
            "data: {\"a\":1}", "",
            "data: {\"b\":", "data: 2}", "",
            "data: [DONE]", "",
        ]
        self.assertEqual(
            list(bench.sse_events(lines)),
            ['{"a":1}', '{"b":\n2}', "[DONE]"])

    def test_no_blank_line_between_data_lines(self):
        self.assertEqual(
            list(bench.sse_events(['data: {"x": 1}'])), ['{"x": 1}'])


class SummaryTests(unittest.TestCase):
    def test_failures_are_counted_and_missing_fields_are_none(self):
        records = [
            {"ok": True, "total_ms": 100.0, "first_delta_ms": 20.0,
             "asr_final_ms": 40.0, "output_tokens": 10,
             "tokens_per_s": 100.0, "finish_reason": "stop"},
            {"ok": True, "total_ms": 200.0, "first_delta_ms": 30.0,
             "asr_final_ms": None, "output_tokens": 20,
             "tokens_per_s": 100.0, "finish_reason": "length"},
            {"ok": False, "error": "boom"},
        ]
        stats = bench.summarize(records, "total_ms")
        self.assertEqual((stats["requests"], stats["successes"],
                          stats["failures"]), (3, 2, 1))
        self.assertEqual(stats["total_p50_ms"], 100.0)
        self.assertEqual(stats["total_p95_ms"], 200.0)
        self.assertEqual(stats["output_tokens_mean"], 15.0)
        self.assertEqual(stats["finish_reasons"], {"stop": 1, "length": 1})
        self.assertEqual(stats["asr_final_p50_ms"], 40.0)
        self.assertEqual(stats["asr_final_p95_ms"], 40.0)
        stats["requests"] = 3
        json.dumps(stats)


class WorkerArgTests(unittest.TestCase):
    def test_worker_concurrency_and_perf_flags_are_forwarded(self):
        captured = {}

        class Capture:
            def __init__(self, args, name):
                captured[name] = args

        original = bench.CappedProcess
        bench.CappedProcess = Capture
        try:
            bench.start_workers("python3", "/tmp/ipc-root", "http://asr",
                                "http://omni", asr_concurrency=3,
                                omni_concurrency=4, perf_dir="/tmp/raw")
        finally:
            bench.CappedProcess = original
        for name, expected in (("asr-worker", "3"), ("omni-worker", "4")):
            args = captured[name]
            self.assertEqual(args[args.index("--concurrency") + 1], expected)
            perf = args[args.index("--perf-log") + 1]
            self.assertTrue(perf.endswith(f"{name}-perf.jsonl"))


if __name__ == "__main__":
    sys.exit(unittest.main())

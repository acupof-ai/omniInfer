#!/usr/bin/env python3
"""Unit tests for the Mac unified validation harness helpers.

These tests use synthetic records and bytes only. They start no server, load no
model, and are not validation evidence.
"""

import base64
import importlib.util
import io
import os
from pathlib import Path
import tempfile
import unittest
import wave


def load_harness():
    path = Path(__file__).resolve().with_name("mac_unified_validation.py")
    spec = importlib.util.spec_from_file_location("mac_unified_validation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


harness = load_harness()


def complete_record(**overrides):
    record = {
        "http_status": 200, "done": True, "terminal": "complete",
        "terminal_count": 1, "terminal_is_last": True, "seq_gap_free": True,
        "session_echo_ok": True, "vlm_delta_count": 2,
        "content_delta_count": 2, "asr_final_count": 1,
        "asr_final_before_delta": True, "text": "answer",
        "transcript": "b", "error": None, "ok": True,
    }
    record.update(overrides)
    return record


def cancelled_record(**overrides):
    return complete_record(terminal="cancelled", text="", vlm_delta_count=1,
                           content_delta_count=1, **overrides)


class WavTests(unittest.TestCase):
    def test_sine_is_deterministic_pcm16_mono(self):
        first = harness.sine_wav_bytes()
        second = harness.sine_wav_bytes()
        self.assertEqual(first, second)
        with wave.open(io.BytesIO(first), "rb") as wav:
            self.assertEqual((wav.getnchannels(), wav.getsampwidth(),
                              wav.getframerate(), wav.getnframes()),
                             (1, 2, 16000, 8000))

    def test_short_wav_is_one_sample(self):
        with wave.open(io.BytesIO(harness.short_wav_bytes()), "rb") as wav:
            self.assertEqual(wav.getnframes(), 1)
            self.assertEqual(wav.readframes(1), b"\x00\x00")


class BodyTests(unittest.TestCase):
    def test_audio_part_round_trips(self):
        wav_bytes = harness.sine_wav_bytes()
        body = harness.unified_body(wav_bytes, 24, "session-1")
        self.assertEqual(body["model"], "qwen3-unified")
        self.assertTrue(body["stream"])
        self.assertEqual(body["session_id"], "session-1")
        content = body["messages"][0]["content"]
        self.assertEqual(len(content), 1)
        part = content[0]
        self.assertEqual(part["type"], "input_audio")
        self.assertEqual(part["input_audio"]["format"], "wav")
        self.assertEqual(base64.b64decode(part["input_audio"]["data"]),
                         wav_bytes)


class CappedLogTests(unittest.TestCase):
    def test_keeps_tail_within_cap(self):
        log = harness.CappedLog(cap=16)
        log.write("a" * 20)
        log.write("b" * 20)
        self.assertEqual(log.text(), "b" * 16)


class SequenceTests(unittest.TestCase):
    def test_first_seq_must_be_zero(self):
        self.assertTrue(harness.seq_step_ok(None, 0))
        self.assertFalse(harness.seq_step_ok(None, 5))

    def test_seq_must_be_consecutive(self):
        self.assertTrue(harness.seq_step_ok(0, 1))
        self.assertFalse(harness.seq_step_ok(0, 2))


class EvaluateTests(unittest.TestCase):
    def test_complete_passes(self):
        record = complete_record()
        self.assertEqual(harness.evaluate(record, "complete"), [])
        self.assertTrue(record["ok"])

    def test_cancelled_passes(self):
        record = cancelled_record()
        self.assertEqual(harness.evaluate(record, "cancelled"), [])

    def test_error_passes(self):
        record = complete_record(terminal="error", text="", transcript="",
                                 asr_final_count=0, vlm_delta_count=0,
                                 content_delta_count=0, error="stage failed")
        self.assertEqual(harness.evaluate(record, "error"), [])

    def test_sequence_gap_fails(self):
        record = complete_record(seq_gap_free=False)
        self.assertTrue(harness.evaluate(record, "complete"))
        self.assertFalse(record["ok"])

    def test_duplicate_terminal_fails(self):
        record = complete_record(terminal_count=2)
        self.assertTrue(harness.evaluate(record, "complete"))
        self.assertFalse(record["ok"])

    def test_terminal_not_last_fails(self):
        record = complete_record(terminal_is_last=False)
        self.assertTrue(harness.evaluate(record, "complete"))
        self.assertFalse(record["ok"])

    def test_missing_asr_final_fails(self):
        record = complete_record(asr_final_count=0)
        self.assertTrue(harness.evaluate(record, "complete"))
        self.assertFalse(record["ok"])

    def test_wrong_terminal_fails(self):
        record = complete_record(terminal="error", error="x")
        self.assertTrue(harness.evaluate(record, "complete"))
        self.assertFalse(record["ok"])

    def test_session_mismatch_fails(self):
        record = complete_record(session_echo_ok=False)
        self.assertTrue(harness.evaluate(record, "complete"))
        self.assertFalse(record["ok"])

    def test_delta_count_mismatch_fails(self):
        record = complete_record(vlm_delta_count=3)
        self.assertTrue(harness.evaluate(record, "complete"))
        self.assertFalse(record["ok"])


class CancelTests(unittest.TestCase):
    def test_clean_complete_then_cancel_passes(self):
        attempts = [complete_record(), cancelled_record()]
        self.assertTrue(harness.cancel_passed(attempts, True))

    def test_flawed_complete_attempt_fails(self):
        attempts = [complete_record(seq_gap_free=False), cancelled_record()]
        harness.evaluate(attempts[0], "complete")
        self.assertFalse(attempts[0]["ok"])
        self.assertFalse(harness.cancel_passed(attempts, True))

    def test_nonzero_in_flight_fails(self):
        self.assertFalse(harness.cancel_passed([cancelled_record()], False))


class SummaryTests(unittest.TestCase):
    def test_percentiles_count_failures(self):
        records = [
            complete_record(total_ms=10.0, first_event_ms=5.0,
                            first_delta_ms=6.0, asr_final_ms=4.0),
            complete_record(total_ms=20.0, first_event_ms=6.0,
                            first_delta_ms=7.0, asr_final_ms=5.0),
            complete_record(total_ms=30.0, first_event_ms=7.0,
                            first_delta_ms=8.0, asr_final_ms=6.0),
            complete_record(total_ms=40.0, ok=False),
        ]
        stats = harness.summarize(records)
        self.assertEqual((stats["requests"], stats["successes"],
                          stats["failures"]), (4, 3, 1))
        self.assertEqual(stats["total_p50_ms"], 20.0)
        self.assertEqual(stats["total_p95_ms"], 30.0)
        self.assertEqual(stats["first_event_p50_ms"], 6.0)
        self.assertEqual(stats["transcripts"], ["b"])


class ShaTests(unittest.TestCase):
    def test_round_trip_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "raw").mkdir()
            (root / "a.txt").write_text("one\n")
            (root / "raw" / "b.jsonl").write_text('{"x":1}\n')
            listed = harness.write_sha256sums(root)
            self.assertEqual(listed, 2)
            self.assertEqual(harness.verify_sha256sums(root), (2, []))
            (root / "raw" / "b.jsonl").write_text('{"x":2}\n')
            self.assertTrue(harness.verify_sha256sums(root)[1])

    def test_unlisted_file_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.txt").write_text("one\n")
            harness.write_sha256sums(root)
            (root / "extra.txt").write_text("two\n")
            _, mismatches = harness.verify_sha256sums(root)
            self.assertIn("not listed: extra.txt", mismatches)

    def test_non_regular_file_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.txt").write_text("one\n")
            harness.write_sha256sums(root)
            os.mkfifo(root / "pipe")
            _, mismatches = harness.verify_sha256sums(root)
            self.assertIn("not a regular file: pipe", mismatches)


if __name__ == "__main__":
    unittest.main()

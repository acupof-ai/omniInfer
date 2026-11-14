#!/usr/bin/env python3
"""Deterministic tests for the phase-2 RL validation modules.

No torch, no GPU, no network. The policy server is exercised with a fake
policy object so the OpenAI-compatible SSE contract and the recorder
integration are covered on any host. The real optimizer path runs only on
H20B GPU 4 and is recorded as evidence, not faked here.
"""

import http.client
import json
from http.server import HTTPServer
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import umminfer_rl_gsm8k_data as data_mod
import umminfer_rl_gsm8k_reward as reward_mod
import umminfer_rl_policy_serve as serve
import umminfer_rl_reward as reward_iface
import umminfer_rl_run as runner
import umminfer_rl_trajectory as traj
import umminfer_rl_worker as worker

GSM8K_SAMPLE = (
    '{"question": "Jan has 3 apples. She buys 2 more.", '
    '"answer": "3 + 2 = 5\\n#### 5"}\n'
    '{"question": "A costs 1,200. B costs 300. Total?", '
    '"answer": "1200 + 300 = 1500\\n#### 1,500"}\n'
).encode()


class Gsm8kDataTests(unittest.TestCase):
    def test_parse_extracts_integer_gold_with_commas(self):
        items = data_mod.parse_gsm8k(GSM8K_SAMPLE)
        self.assertEqual(items[0]["gold_answer"], 5)
        self.assertEqual(items[0]["gold_text"], "5")
        self.assertEqual(items[1]["gold_answer"], 1500)
        self.assertEqual(len(items), 2)

    def test_fingerprint_hashes_real_bytes(self):
        fingerprint = data_mod.data_fingerprint(GSM8K_SAMPLE)
        self.assertEqual(fingerprint["byte_count"], len(GSM8K_SAMPLE))
        self.assertEqual(fingerprint["total_items"], 2)
        self.assertEqual(len(fingerprint["sha256"]), 64)


class Gsm8kRewardTests(unittest.TestCase):
    def _gold(self, tmp):
        path = Path(tmp) / "gold.jsonl"
        path.write_text(json.dumps({
            "request_id": "r1", "gold_answer": 5}) + "\n" +
            json.dumps({"request_id": "r2", "gold_answer": 1500}) + "\n")
        return path

    def test_extract_answer_variants(self):
        self.assertEqual(reward_mod.extract_answer("reasoning\n#### 42"), 42)
        self.assertEqual(
            reward_mod.extract_answer("work\n#### 1,500\n"), 1500)
        self.assertIsNone(reward_mod.extract_answer("no marker here"))
        self.assertIsNone(reward_mod.extract_answer("#### not a number"))

    def test_exact_match_scores_one_zero_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            reward_obj = reward_mod.GSM8KExactMatchReward(self._gold(tmp))
            self.assertEqual(
                reward_obj.score({"request_id": "r1",
                                  "output_text": "3+2\n#### 5"}), 1.0)
            self.assertEqual(
                reward_obj.score({"request_id": "r1",
                                  "output_text": "#### 9"}), 0.0)
            self.assertEqual(
                reward_obj.score({"request_id": "r1",
                                  "output_text": "no marker"}), 0.0)
            # Unknown request id stays None, never a fabricated zero.
            self.assertIsNone(
                reward_obj.score({"request_id": "other",
                                  "output_text": "#### 5"}))

    def test_loads_through_phase1_loader_with_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            gold = self._gold(tmp)
            import os
            previous = os.environ.get("UMMINFER_RL_GOLD_PATH")
            os.environ["UMMINFER_RL_GOLD_PATH"] = str(gold)
            try:
                loaded = reward_iface.load_reward(
                    "umminfer_rl_gsm8k_reward:load_default")
            finally:
                if previous is None:
                    del os.environ["UMMINFER_RL_GOLD_PATH"]
                else:
                    os.environ["UMMINFER_RL_GOLD_PATH"] = previous
            self.assertIsInstance(loaded,
                                  reward_mod.GSM8KExactMatchReward)
            self.assertEqual(loaded.name, "gsm8k-exact-match-v1")

    def test_apply_reward_round_trip_through_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            gold = self._gold(tmp)
            export = Path(tmp) / "traj.jsonl"
            record = {
                "schema_version": 1, "session_id": "s", "request_id": "r2",
                "turn_index": 0, "events": [],
                "terminal_state": "completed", "error": None,
                "output_text": "sum\n#### 1500",
                "output_sha256": __import__("hashlib").sha256(
                    b"sum\n#### 1500").hexdigest()}
            export.write_text(json.dumps(record) + "\n")
            reward_iface.apply_reward(
                export, reward_mod.GSM8KExactMatchReward(gold))
            scored = json.loads(export.read_text())
            self.assertEqual(scored["reward"]["score"], 1.0)
            self.assertEqual(scored["reward"]["name"],
                             "gsm8k-exact-match-v1")


class CappedWriterTests(unittest.TestCase):
    def test_file_never_exceeds_the_hard_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "capped.log"
            writer = runner.CappedWriter(path, cap=128)
            for _ in range(20):
                writer.write("x" * 16)
            writer.close()
            data = path.read_bytes()
            self.assertLessEqual(len(data), 128)
            self.assertIn(b"truncated", data)

    def test_under_cap_output_is_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "small.log"
            writer = runner.CappedWriter(path, cap=4096)
            writer.write("line one\n")
            writer.write("line two\n")
            writer.close()
            self.assertEqual(path.read_text(), "line one\nline two\n")


class FakePolicy:
    def __init__(self):
        self.calls = 0

    def generate(self, messages, *, temperature, top_p, max_new_tokens):
        self.calls += 1
        return (f"rl-rollout-{self.calls:06d}",
                f"answer {self.calls}", [], [])


class PolicyServerProtocolTests(unittest.TestCase):
    def test_sse_exit_is_recordable_by_the_phase1_recorder(self):
        server = HTTPServer(("127.0.0.1", 0),
                            serve.make_handler(FakePolicy()))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        port = server.server_address[1]
        with tempfile.TemporaryDirectory() as tmp:
            recorder = traj.TrajectoryRecorder(Path(tmp) / "traj.jsonl")
            record = traj.record_chat_completion(
                f"127.0.0.1:{port}",
                {"model": "fake", "stream": True,
                 "messages": [{"role": "user", "content": "q"}]},
                recorder, "sess-1")
            self.assertEqual(record["terminal_state"],
                             traj.TERMINAL_COMPLETED)
            self.assertEqual(record["request_id"], "rl-rollout-000001")
            self.assertEqual(record["output_text"], "answer 1")
            # Re-parse the export like the worker would.
            validated = worker.validate_trajectory(Path(tmp) / "traj.jsonl")
            self.assertEqual(len(validated), 1)

    def test_health_and_unknown_route(self):
        server = HTTPServer(("127.0.0.1", 0),
                            serve.make_handler(FakePolicy()))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        port = server.server_address[1]
        conn = http.client.HTTPConnection("127.0.0.1", port)
        conn.request("GET", "/health")
        self.assertEqual(conn.getresponse().status, 200)
        conn.request("GET", "/nope")
        self.assertEqual(conn.getresponse().status, 404)


class WorkerTrainStepValidationTests(unittest.TestCase):
    def _record(self, *, score=1.0, request_id="r1"):
        text = "work\n#### 5"
        import hashlib
        return {
            "schema_version": 1, "session_id": "s",
            "request_id": request_id, "turn_index": 0, "events": [],
            "terminal_state": "completed", "error": None,
            "output_text": text,
            "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "reward": {"name": "gsm8k-exact-match-v1", "score": score}}

    def test_missing_scores_are_rejected_before_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            tw = worker.TrainingWorker(tmp)
            record = self._record()
            del record["reward"]
            path = Path(tmp) / "traj.jsonl"
            path.write_text(json.dumps(record) + "\n")
            response = tw.handle({
                "type": "train_step", "trajectory_path": str(path),
                "model_path": "/nonexistent",
                "hyperparameters": {"lr": 1e-5, "seed": 1}})
            self.assertEqual(response["type"], "error")
            self.assertIn("numeric reward", response["error"])
            # Rejected before any checkpoint sequence is consumed.
            self.assertEqual(tw.store.next_sequence, 0)

    def test_null_score_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tw = worker.TrainingWorker(tmp)
            record = self._record()
            record["reward"]["score"] = None
            path = Path(tmp) / "traj.jsonl"
            path.write_text(json.dumps(record) + "\n")
            response = tw.handle({
                "type": "train_step", "trajectory_path": str(path),
                "model_path": "/nonexistent",
                "hyperparameters": {"lr": 1e-5, "seed": 1}})
            self.assertEqual(response["type"], "error")
            self.assertEqual(tw.store.next_sequence, 0)

    def test_phase1_train_keeps_null_algorithm_slots(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "traj.jsonl"
            path.write_text(json.dumps(self._record()) + "\n")
            tw = worker.TrainingWorker(tmp)
            response = tw.handle(
                {"type": "train", "trajectory_path": str(path)})
            self.assertEqual(response["type"], "train_result")
            manifest = json.loads(
                Path(response["checkpoint_path"]).read_text())
            self.assertIsNone(manifest["algorithm"])
            self.assertIsNone(manifest["weights_ref"])
            self.assertIsNone(manifest["weights_sha256"])


if __name__ == "__main__":
    sys.exit(unittest.main())

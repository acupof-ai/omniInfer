# SPDX-License-Identifier: Apache-2.0
"""Training worker process: versioned stdio protocol and checkpoints.

The worker runs in its own process. Serving never imports this module and
never loads optimizer state. Transport is one JSON object per line over
stdin/stdout (NDJSON), so any future orchestrator needs no new dependency.

Phase 1 performs no training. A ``train`` request validates the trajectory
export semantically and writes versioned checkpoint metadata. Algorithm name,
weights reference and optimizer state stay null/empty until an algorithm
(PPO/DPO/GRPO/...) is defined by the user; no algorithm code exists here.

Checkpoints form a content hash chain: each manifest records the SHA-256 of
the previous manifest file, and a ``.sha256`` sidecar pins the latest one.
The chain is re-verified every time a worker opens the state directory.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# Some interpreters' -I mode omits the script directory from sys.path; the
# worker imports the sibling trainer module, so pin the directory explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

WORKER_PROTOCOL_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1

_TERMINAL_STATES = frozenset(("completed", "failed", "cancelled"))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_trajectory(path):
    """Parse and semantically validate a trajectory JSONL export.

    Returns the records. Raises ValueError on a structural defect: missing
    schema version, unknown terminal state, gap in event sequence, or an
    output hash that does not recompute. The worker never trusts an export it
    has not checked, regardless of which process wrote it.
    """
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise ValueError("trajectory export contains no records")
    for index, record in enumerate(records):
        prefix = f"record {index}: "
        if not isinstance(record.get("schema_version"), int):
            raise ValueError(prefix + "missing integer schema_version")
        if record.get("terminal_state") not in _TERMINAL_STATES:
            raise ValueError(prefix + "invalid terminal_state")
        events = record.get("events")
        if not isinstance(events, list):
            raise ValueError(prefix + "missing events list")
        for event_index, event in enumerate(events):
            if event.get("seq") != event_index:
                raise ValueError(prefix + f"event seq gap at {event_index}")
        expected = hashlib.sha256(
            record.get("output_text", "").encode("utf-8")).hexdigest()
        if record.get("output_sha256") != expected:
            raise ValueError(prefix + "output_sha256 does not recompute")
    return records


def _reward_name(records):
    """Pick the one reward name across an export, or None when all missing.

    A malformed reward block (present but not an object) is rejected. Two
    different non-null names are also rejected: one checkpoint cannot be
    attributed to more than one reward. Missing blocks (NullReward/never
    scored) coexist with a single real name.
    """
    names = set()
    for record in records:
        block = record.get("reward")
        if block is None:
            continue
        if not isinstance(block, dict):
            raise ValueError("reward block must be an object or null")
        name = block.get("name")
        if name is not None:
            names.add(name)
    if len(names) > 1:
        raise ValueError(
            "trajectory mixes reward names: " + ", ".join(sorted(names)))
    return next(iter(names), None)


class CheckpointStore:
    """Monotonic checkpoints with a content hash chain, no mutable state file."""

    def __init__(self, state_dir):
        self.checkpoint_dir = Path(state_dir) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._latest_sequence, self._latest_hash = self._verify_chain()

    @property
    def next_sequence(self):
        return self._latest_sequence + 1

    def _manifest_paths(self):
        return sorted(self.checkpoint_dir.glob("ckpt-*.json"))

    def _verify_chain(self):
        """Recompute every manifest hash and verify the parent linkage."""
        previous_hash = None
        latest_sequence = -1
        for path in self._manifest_paths():
            sidecar = Path(str(path) + ".sha256")
            if not sidecar.exists():
                raise ValueError(f"missing checksum sidecar for {path.name}")
            actual = sha256_file(path)
            if sidecar.read_text(encoding="utf-8").strip() != actual:
                raise ValueError(f"checksum mismatch for {path.name}")
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest.get("base_checkpoint_sha256") != previous_hash:
                raise ValueError(
                    f"hash chain broken at {path.name}: parent hash mismatch")
            previous_hash = actual
            latest_sequence = manifest["sequence"]
        return latest_sequence, previous_hash

    def write_checkpoint(self, *, trajectory_path, record_count,
                         trajectory_sha256, reward_name,
                         algorithm=None, weights_ref=None,
                         weights_sha256=None, hyperparameters=None,
                         metrics=None):
        sequence = self.next_sequence
        manifest = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "worker_protocol_version": WORKER_PROTOCOL_VERSION,
            "sequence": sequence,
            # Phase 1 leaves algorithm/weights null; a real train_step fills
            # them. Hyperparameters arrive with the first real algorithm.
            "algorithm": algorithm,
            "weights_ref": weights_ref,
            "weights_sha256": weights_sha256,
            "hyperparameters": hyperparameters,
            "metrics": metrics,
            "reward": {"name": reward_name},
            "source": {
                "trajectory_path": str(trajectory_path),
                "record_count": record_count,
                "trajectory_sha256": trajectory_sha256,
            },
            "base_checkpoint_sha256": self._latest_hash,
        }
        path = self.checkpoint_dir / f"ckpt-{sequence:08d}.json"
        if path.exists():
            # Two workers on one directory is a deployment error, not a state
            # machine to support.
            raise ValueError(f"checkpoint sequence conflict: {path.name}")
        content = json.dumps(manifest, sort_keys=True,
                             ensure_ascii=False, allow_nan=False)
        # ponytail: tmp+replace gives atomic publication; directory is
        # single-writer by contract, no locking.
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(content + "\n", encoding="utf-8")
        os.replace(tmp, path)
        digest = sha256_file(path)
        Path(str(path) + ".sha256").write_text(digest + "\n",
                                                encoding="utf-8")
        self._latest_sequence = sequence
        self._latest_hash = digest
        return sequence, path


class TrainingWorker:
    """Request handler. The stdin/stdout loop lives in :func:`main`."""

    def __init__(self, state_dir):
        self.store = CheckpointStore(state_dir)
        # The trainer is created on the first train_step and kept resident, so
        # optimizer state carries across steps. None until a real algorithm
        # runs; phase-1 ``train`` never touches torch.
        self._trainer = None
        self._trainer_key = None

    def handle(self, request):
        req_type = request.get("type")
        if req_type == "ping":
            return {"type": "pong",
                    "worker_protocol_version": WORKER_PROTOCOL_VERSION}
        if req_type == "train":
            return self._handle_train(request)
        if req_type == "train_step":
            return self._handle_train_step(request)
        # Unknown fields on known messages are ignored; only an unknown verb
        # is an error, so adding optional fields later needs no version bump.
        return {"type": "error", "task_id": request.get("task_id"),
                "error": f"unknown request type: {req_type!r}"}

    def _handle_train(self, request):
        task_id = request.get("task_id")
        try:
            trajectory_path = request["trajectory_path"]
            records = validate_trajectory(trajectory_path)
            reward_name = _reward_name(records)
            sequence, path = self.store.write_checkpoint(
                trajectory_path=trajectory_path,
                record_count=len(records),
                trajectory_sha256=sha256_file(trajectory_path),
                reward_name=reward_name)
        except KeyError as exc:
            return {"type": "error", "task_id": task_id,
                    "error": f"missing field: {exc.args[0]}"}
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return {"type": "error", "task_id": task_id, "error": str(exc)}
        return {"type": "train_result", "task_id": task_id, "status": "ok",
                "sequence": sequence, "checkpoint_path": str(path),
                "worker_protocol_version": WORKER_PROTOCOL_VERSION}

    def _handle_train_step(self, request):
        """Real RL update: validate the scored export, run one optimizer step.

        The trainer stays resident between requests so optimizer state carries
        across steps. A config change on a later request is rejected, not
        silently applied to a trainer built with different hyperparameters.
        """
        task_id = request.get("task_id")
        try:
            trajectory_path = request["trajectory_path"]
            model_path = request["model_path"]
            hyperparameters = dict(request.get("hyperparameters") or {})
            records = validate_trajectory(trajectory_path)
            self._require_scores(records)
            reward_name = _reward_name(records)
            trainer_key = (model_path, json.dumps(
                hyperparameters, sort_keys=True))
            if self._trainer is None:
                from umminfer_rl_train import ReinforceTrainer
                self._trainer = ReinforceTrainer(model_path, **hyperparameters)
                self._trainer_key = trainer_key
            elif self._trainer_key != trainer_key:
                raise ValueError(
                    "trainer config is fixed for this state directory; "
                    "open a new one for different model/hyperparameters")
            metrics = self._trainer.train_step(records)
            sequence = self.store.next_sequence
            weights_path = (Path(self.store.checkpoint_dir.parent)
                            / "weights" / f"weights-{sequence:08d}.pt")
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            self._trainer.save_weights(weights_path)
            sequence, path = self.store.write_checkpoint(
                trajectory_path=trajectory_path,
                record_count=len(records),
                trajectory_sha256=sha256_file(trajectory_path),
                reward_name=reward_name,
                algorithm=self._trainer.config["algorithm"],
                weights_ref=str(weights_path),
                weights_sha256=sha256_file(weights_path),
                hyperparameters=self._trainer.config,
                metrics=metrics)
        except KeyError as exc:
            return {"type": "error", "task_id": task_id,
                    "error": f"missing field: {exc.args[0]}"}
        except (OSError, ValueError, TypeError, json.JSONDecodeError,
                RuntimeError, ImportError) as exc:
            return {"type": "error", "task_id": task_id, "error": str(exc)}
        return {"type": "train_step_result", "task_id": task_id,
                "status": "ok", "sequence": sequence,
                "checkpoint_path": str(path),
                "weights_path": str(weights_path), "metrics": metrics,
                "worker_protocol_version": WORKER_PROTOCOL_VERSION}

    @staticmethod
    def _require_scores(records):
        for index, record in enumerate(records):
            block = record.get("reward")
            if not isinstance(block, dict) or block.get("score") is None:
                raise ValueError(
                    f"record {index}: train_step needs a numeric reward score")


def main(argv=None):
    parser = argparse.ArgumentParser(description="ummInfer RL training worker")
    parser.add_argument("--state-dir", required=True)
    args = parser.parse_args(argv)
    worker = TrainingWorker(args.state_dir)
    sys.stdout.write(json.dumps({
        "type": "hello",
        "worker_protocol_version": WORKER_PROTOCOL_VERSION,
    }) + "\n")
    sys.stdout.flush()
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            response = {"type": "error", "error": f"invalid JSON: {exc}"}
        else:
            response = worker.handle(request)
        sys.stdout.write(json.dumps(response, sort_keys=True,
                                    allow_nan=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()

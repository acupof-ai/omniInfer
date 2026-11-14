# SPDX-License-Identifier: Apache-2.0
"""End-to-end RL architecture validation runner.

Pipeline, every stage using the phase-1 interfaces:

1. Load real GSM8K bytes and select train/eval problems.
2. Start the small-model policy server (separate process).
3. Sample grouped rollouts through ``record_chat_completion`` so the
   trajectory export is produced by the existing recorder, not the trainer.
4. Score the export with the verifiable exact-match reward.
5. Drive the separate training worker over its NDJSON protocol for real
   REINFORCE optimizer steps, one scored export per step.
6. Run greedy evaluation in its own process before training and after every
   checkpoint.

The runner itself imports no torch: all model code lives in the server,
worker and eval processes.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import umminfer_rl_gsm8k_data as data_mod
import umminfer_rl_gsm8k_reward as reward_mod
import umminfer_rl_reward as reward_iface
import umminfer_rl_trajectory as traj
import umminfer_rl_worker as worker_mod

build_messages = data_mod.build_messages


def wait_health(port, expected_pid=None, timeout=180):
    import http.client
    deadline = time.time() + timeout
    while time.time() < deadline:
        if expected_pid is not None and expected_pid.poll() is not None:
            raise RuntimeError(
                f"policy server exited before becoming ready "
                f"(rc={expected_pid.returncode}); see its capped log")
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", "/health")
            response = conn.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            # The pid pins the listener to this child: an orphan from an
            # earlier run on a reused port answers with a different pid and
            # is rejected instead of silently serving stale weights.
            if response.status == 200 and (
                    expected_pid is None
                    or payload.get("pid") == expected_pid.pid):
                return
        except (OSError, json.JSONDecodeError, ValueError):
            time.sleep(1)
    raise RuntimeError("policy server did not become ready")


def read_worker_line(worker, timeout):
    """Read one NDJSON line with a hard deadline, turning a worker hang or
    crash into a RuntimeError instead of an indefinite block."""
    import select
    ready, _, _ = select.select([worker.stdout], [], [], timeout)
    if not ready:
        raise RuntimeError(f"training worker did not respond within {timeout}s")
    line = worker.stdout.readline()
    if not line:
        raise RuntimeError("training worker exited mid-run")
    return json.loads(line)


def capture_environment(python):
    """Collect the software/GPU stack from child processes; runner imports
    no torch itself. Failures are recorded, never hidden."""
    code = (
        "import json, platform, sys, torch\n"
        "import transformers\n"
        "print(json.dumps({"
        "'python': sys.version.split()[0],"
        "'platform': platform.platform(),"
        "'torch': torch.__version__,"
        "'transformers': transformers.__version__,"
        "'cuda_available': torch.cuda.is_available(),"
        "'cuda_version': torch.version.cuda,"
        "'gpu_name': torch.cuda.get_device_name(0),"
        "'gpu_count_visible': torch.cuda.device_count()}))")
    proc = subprocess.run([python, "-I", "-c", code],
                          capture_output=True, text=True, timeout=120)
    env = json.loads(proc.stdout.strip().splitlines()[-1])
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=60)
    env["nvidia_smi"] = smi.stdout.strip()
    return env


def sha256_file(path):
    return worker_mod.sha256_file(path)


class CappedWriter:
    """File-like writer with a hard byte cap, matching the repo log rule."""

    _MARKER = b"\n[output truncated at byte cap]\n"

    def __init__(self, path, cap=64 * 1024):
        self.handle = open(path, "wb")
        self.cap = cap
        self.written = 0

    def write(self, data):
        if isinstance(data, str):
            data = data.encode("utf-8")
        if self.written >= self.cap:
            return len(data)
        if self.written >= self.cap:
            return len(data)
        # Marker and data share the cap budget, so total bytes never exceed
        # it even when remaining space is smaller than the marker itself.
        remaining = self.cap - self.written
        take = max(0, remaining - len(self._MARKER))
        if len(data) <= take:
            self.handle.write(data)
            self.written += len(data)
        else:
            self.handle.write(data[:take])
            self.handle.write(self._MARKER[:remaining - take])
            self.written = self.cap
        return len(data)

    def close(self):
        self.handle.close()


def _policy_command(args, step, weights_path):
    command = [args.python, "-I",
               str(Path(__file__).resolve().parent
                   / "umminfer_rl_policy_serve.py"),
               "--model-path", args.model_path, "--port", str(args.port),
               # Distinct generator seed per step, even though weights change.
               "--seed", str(args.seed + step), "--dtype", args.dtype]
    if weights_path:
        command += ["--weights", weights_path]
    return command


def popen_capped(command, log_path):
    """Start a subprocess whose combined output is hard-capped at 64 KiB.

    A stdout file descriptor would let the child bypass any in-process cap,
    so output is piped and a drain thread enforces the cap while preventing
    pipe-buffer deadlock.
    """
    import threading
    capped = CappedWriter(log_path)
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)

    def drain():
        for line in process.stdout:
            capped.write(line)
        process.stdout.close()
        capped.close()

    threading.Thread(target=drain, daemon=True).start()
    return process


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", required=True,
                        help="cached GSM8K test.jsonl bytes")
    parser.add_argument("--data-source", default=data_mod.GSM8K_TEST_URL,
                        help="provenance URL recorded in the manifest")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--python", required=True,
                        help="interpreter for server/worker/eval subprocesses")
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--eval-size", type=int, default=16)
    parser.add_argument("--data-offset", type=int, default=0,
                        help="index of the first GSM8K problem used")
    parser.add_argument("--train-indexes",
                        help="comma-separated dataset indexes for training; "
                             "overrides the contiguous slice")
    parser.add_argument("--eval-indexes",
                        help="comma-separated dataset indexes for evaluation")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261114)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--port", type=int, default=32100)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=400)
    args = parser.parse_args(argv)

    work = Path(args.work_dir)
    rollout_dir = work / "rollouts"
    log_dir = work / "logs"
    state_dir = work / "worker-state"
    eval_dir = work / "evals"
    for directory in (rollout_dir, log_dir, state_dir, eval_dir):
        directory.mkdir(parents=True, exist_ok=True)

    raw = Path(args.data_path).read_bytes()
    items = data_mod.parse_gsm8k(raw)
    if args.train_indexes:
        train_indexes = [int(x) for x in args.train_indexes.split(",")]
    else:
        start = args.data_offset
        train_indexes = list(range(start, start + args.num_prompts))
    if args.eval_indexes:
        eval_indexes = [int(x) for x in args.eval_indexes.split(",")]
    else:
        eval_indexes = list(range(
            train_indexes[-1] + 1,
            train_indexes[-1] + 1 + args.eval_size))
    if max(train_indexes + eval_indexes) >= len(items):
        raise ValueError("a selected index is outside the dataset")
    train_items = [items[i] for i in train_indexes]
    eval_items = [items[i] for i in eval_indexes]

    manifest = {
        "schema_version": 1,
        "kind": "rl-architecture-validation",
        "seed": args.seed,
        "hyperparameters": {
            "num_prompts": len(train_items),
            "group_size": args.group_size,
            "steps": args.steps,
            "eval_size": len(eval_items),
            "lr": args.lr,
            "dtype": args.dtype,
            "max_new_tokens": args.max_new_tokens,
            "sampling": {"temperature": args.temperature, "top_p": 1.0,
                         "do_sample": True},
        },
        "data": data_mod.data_fingerprint(raw, args.data_source),
        "model_path": args.model_path,
        "model_weights_sha256": _model_weights_hash(args.model_path),
        "train_prompt_indexes": train_indexes,
        "eval_prompt_indexes": eval_indexes,
        "steps": [],
    }

    eval_data_path = work / "eval_items.jsonl"
    eval_data_path.write_text("\n".join(
        json.dumps(item, sort_keys=True, ensure_ascii=False)
        for item in eval_items) + "\n", encoding="utf-8")

    # Baseline greedy evaluation before any optimizer step. The eval process
    # loads the base model itself, so no policy server is needed for it.
    eval_results = [run_eval(
        args, eval_data_path, eval_dir / "eval-baseline.json",
        weights_path=None, tag="baseline")]

    # Worker stdout is the NDJSON protocol channel; only stderr is logged,
    # with the same hard cap.
    worker_err = CappedWriter(log_dir / "worker.log")
    worker = subprocess.Popen(
        [args.python, "-I",
         str(Path(__file__).resolve().parent / "umminfer_rl_worker.py"),
         "--state-dir", str(state_dir)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True)
    import threading
    threading.Thread(
        target=lambda: (
            [worker_err.write(line) for line in worker.stderr],
            worker_err.close()),
        daemon=True).start()
    hello = read_worker_line(worker, timeout=60)
    assert hello["type"] == "hello", hello
    worker.stdin.write(json.dumps({"type": "ping"}) + "\n")
    worker.stdin.flush()
    pong = read_worker_line(worker, timeout=60)
    assert pong["type"] == "pong", pong

    previous_weights = None
    try:
        for step in range(args.steps):
            # Restart the policy server on the previous checkpoint each step,
            # so the rollouts the step trains on are sampled from exactly the
            # weights being updated (on-policy).
            server = popen_capped(
                _policy_command(args, step, previous_weights),
                log_dir / f"policy-server-step-{step}.log")
            try:
                wait_health(args.port, expected_pid=server)
                export_path, gold_path = collect_rollouts(
                    args, train_items, step, rollout_dir)
            finally:
                server.terminate()
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server.kill()
            reward_iface.apply_reward(
                export_path,
                reward_mod.GSM8KExactMatchReward(gold_path))
            enrich_with_prompts(export_path, train_items)
            request = {
                "type": "train_step", "task_id": f"step-{step}",
                "trajectory_path": str(export_path),
                "model_path": args.model_path,
                # The trainer seed is fixed for the state directory: it seeds
                # model init and dropout only. Rollout sampling randomness
                # comes from the policy server generator, not from this seed.
                "hyperparameters": {
                    "lr": args.lr, "seed": args.seed + 1,
                    "dtype": args.dtype}}
            worker.stdin.write(json.dumps(request) + "\n")
            worker.stdin.flush()
            result = read_worker_line(worker, timeout=900)
            if result.get("type") != "train_step_result":
                raise RuntimeError(f"worker failed: {result}")
            metrics = result["metrics"]
            # Hard gate on the acceptance property. A step with no gradient
            # signal (every reward group homogeneous) fails loudly instead of
            # producing checkpoints whose only weight change would be noise.
            if metrics["nonzero_grad_params"] == 0:
                raise RuntimeError(
                    "degenerate step: zero non-zero gradient parameters; "
                    "the rollout rewards carry no policy-gradient signal")
            if metrics["weight_delta_max_abs"] <= 0.0:
                raise RuntimeError(
                    "degenerate step: weights did not change after the "
                    "optimizer step")
            weights_path = result["weights_path"]
            eval_results.append(run_eval(
                args, eval_data_path,
                eval_dir / f"eval-step-{step + 1}.json",
                weights_path=weights_path, tag=f"step-{step + 1}"))
            manifest["steps"].append({
                "step": step,
                "policy_seed": args.seed + step,
                "rollout_weights": previous_weights,
                "trajectory_path": str(export_path),
                "trajectory_sha256": sha256_file(export_path),
                "gold_path": str(gold_path),
                "checkpoint_path": result["checkpoint_path"],
                "checkpoint_sha256": sha256_file(result["checkpoint_path"]),
                "weights_path": weights_path,
                "weights_sha256": sha256_file(weights_path),
                "metrics": metrics,
            })
            previous_weights = weights_path

        worker.stdin.close()
        worker.wait(timeout=30)
    finally:
        if worker is not None:
            worker.terminate()
            try:
                worker.wait(timeout=30)
            except subprocess.TimeoutExpired:
                worker.kill()

    manifest["environment"] = capture_environment(args.python)
    manifest["evaluations"] = eval_results
    manifest_path = work / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2,
                   ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")
    write_sha256_sums(work)
    print(json.dumps({"type": "run_complete",
                      "manifest": str(manifest_path)}, sort_keys=True))


def collect_rollouts(args, train_items, step, rollout_dir):
    export_path = rollout_dir / f"rollouts-step-{step}.jsonl"
    gold_path = rollout_dir / f"gold-step-{step}.jsonl"
    recorder = traj.TrajectoryRecorder(export_path)
    gold_rows = []
    for prompt_id, item in enumerate(train_items):
        messages = build_messages(item["question"])
        for _ in range(args.group_size):
            body = {"model": "qwen2.5-0.5b-rl-validation",
                    "messages": messages, "stream": True,
                    "request_params_override": {
                        "temperature": args.temperature, "top_p": 1.0,
                        "max_tokens": args.max_new_tokens}}
            record = traj.record_chat_completion(
                f"127.0.0.1:{args.port}", body, recorder,
                session_id=f"step-{step}-prompt-{prompt_id}")
            if record["terminal_state"] != traj.TERMINAL_COMPLETED:
                raise RuntimeError(
                    f"rollout failed: {record['terminal_state']} "
                    f"{record.get('error')}")
            gold_rows.append({"request_id": record["request_id"],
                              "prompt_id": prompt_id,
                              "gold_answer": item["gold_answer"]})
    gold_path.write_text("\n".join(
        json.dumps(row, sort_keys=True) for row in gold_rows) + "\n",
        encoding="utf-8")
    return export_path, gold_path


def enrich_with_prompts(export_path, train_items):
    """Add prompt_id and the exact messages to each scored record.

    The worker recomputes completion log probabilities from the same messages
    the server received, so the prompt text cannot be reconstructed loosely.
    Session ids encode the prompt index (step-N-prompt-P), which is stable
    because the recorder drives prompts in order.
    """
    rows = []
    for line in export_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        session_id = record["session_id"]
        prompt_id = int(session_id.rsplit("-prompt-", 1)[1])
        record["prompt_id"] = prompt_id
        record["messages"] = build_messages(
            train_items[prompt_id]["question"])
        rows.append(json.dumps(record, sort_keys=True,
                               ensure_ascii=False, allow_nan=False))
    export_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def run_eval(args, eval_data_path, output_path, *, weights_path, tag):
    command = [args.python, "-I",
               str(Path(__file__).resolve().parent / "umminfer_rl_eval.py"),
               "--model-path", args.model_path,
               "--data-path", str(eval_data_path),
               "--output", str(output_path),
               "--dtype", args.dtype, "--tag", tag]
    if weights_path:
        command += ["--weights", weights_path]
    # Evaluation output is capped like every other child log.
    process = popen_capped(
        command, output_path.parent.parent / "logs" / f"eval-{tag}.log")
    try:
        rc = process.wait(timeout=900)
    except subprocess.TimeoutExpired:
        process.kill()
        raise RuntimeError(f"evaluation {tag} timed out")
    if rc != 0:
        raise RuntimeError(f"evaluation {tag} exited with {rc}")
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    return {"tag": tag, "path": str(output_path),
            "exact_match": payload["exact_match"],
            "scored": payload["scored"], "total": payload["total"],
            "weights_sha256": payload.get("weights_sha256")}


def _model_weights_hash(model_path):
    digest = hashlib.sha256()
    model_dir = Path(model_path)
    for path in sorted(model_dir.glob("*.safetensors")):
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def write_sha256_sums(work):
    lines = []
    for path in sorted(work.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            lines.append(f"{sha256_file(path)}  "
                         f"./{path.relative_to(work).as_posix()}")
    (work / "SHA256SUMS").write_text("\n".join(lines) + "\n",
                                     encoding="utf-8")


if __name__ == "__main__":
    main()

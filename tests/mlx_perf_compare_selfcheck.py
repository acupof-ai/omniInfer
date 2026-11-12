#!/usr/bin/env python3
"""Check `tests/mlx_perf_compare.py` with synthetic reports only.

No model, server, or process is started. Each gate is exercised by mutating one
field of a passing pair of reports. Raw rows and their stored summaries are
built together, so a fixture is arithmetically consistent by construction and a
tampered field is the only reason a gate can fail.
"""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("mlx_perf_compare",
                                              HERE / "mlx_perf_compare.py")
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)

HASH = "a" * 64
BINARY = "b" * 64
SCRIPT = "c" * 64
BASE_STAGE = "d" * 64
CAND_STAGE = "e" * 64
MODEL_INPUTS = {"config.json": "1" * 64, "tokenizer.json": "2" * 64,
                "tokenizer_config.json": "3" * 64, "chat_template": "4" * 64}
WEIGHTS = "5" * 64


def check(name, condition):
    print(f"{'PASS' if condition else 'FAIL'} {name}")
    if not condition:
        raise SystemExit(1)


def row(path, iteration, total_ms, ttft_ms, memory, output_hash, warm=True):
    return {"iteration": iteration, "warm": warm,
            "cold": warm is False,
            "ttft_ms": ttft_ms if path in compare.TTFT_PATHS else None,
            "total_ms": total_ms, "generation_tokens": compare.MAIN_TOKENS,
            "output_tokens_per_s": compare.MAIN_TOKENS / (total_ms / 1000.0),
            "prompt_tokens": 35, "peak_memory_gb": memory,
            "output_sha256": output_hash}


def block(path, rate, ttft, memory, output_hash, spread=0.0):
    total = compare.MAIN_TOKENS * 1000.0 / rate
    samples = []
    if path == "direct":
        samples.append(row(path, 0, total, ttft, memory, output_hash,
                           warm=False))
    index = 0
    while len([item for item in samples if item["warm"]]) < compare.MIN_SAMPLES:
        factor = (1 - spread if index % 2 == 0 else 1 + spread)
        samples.append(row(path, index + 1, total * factor, ttft, memory,
                           output_hash))
        index += 1
    measured = [item for item in samples if item["warm"]]
    summaries = {}
    for metric in compare.NUMERIC_METRICS:
        value = compare.recompute(measured, metric)
        if value is not None:
            summaries[metric] = value
    return {"cold": None, "warm": summaries, "samples": samples}


def report(label, rate=100.0, ttft=40.0, memory=0.5, output_hash=HASH,
           spread=0.0, stage=BASE_STAGE, script=SCRIPT, **overrides):
    data = {
        "role": label,
        "completed": True,
        "error": None,
        "output_consistent": True,
        "skipped_paths": [],
        "configuration": {
            "model": "/model", "model_safetensors_sha256": "f" * 64,
            "prompt": "prompt", "temperature": 0.0, "warmup": 2,
            "max_tokens": compare.MAIN_TOKENS, "samples": 20,
            "max_attempts": 1,
        },
        "environment": {
            "host": "test-host", "machine": "arm64",
            "interpreter": "/python", "python": "3.11.14",
            "mlx": "0.31.1", "mlx_lm": "0.31.2", "msgpack": "1.1.0",
            "pyzmq": "27.1.0", "tokenizers": "0.22.1",
            "transformers": "5.12.1",
            "binary": {"path": "/binary", "sha256": BINARY},
            "stage": {"path": "/stage", "sha256": stage},
            "script": {"path": "/harness", "sha256": script},
        },
        "exclusive_window": {
            "foreign_processes": [],
            "hashes_before": {
                "stage": stage, "script": script, "binary": BINARY,
                "weights": WEIGHTS,
                "weights": WEIGHTS,
                **{f"model.{name}": value
                   for name, value in MODEL_INPUTS.items()},
            },
            "hashes_after": {
                "stage": stage, "script": script, "binary": BINARY,
                "weights": WEIGHTS,
                "weights": WEIGHTS,
                **{f"model.{name}": value
                   for name, value in MODEL_INPUTS.items()},
            },
        },
        "results": {
            "direct": block("direct", rate, ttft, memory, output_hash, spread),
            "stage": {
                "json": block("stage.json", rate, ttft, memory, output_hash,
                              spread),
                "sse": block("stage.sse", rate, ttft, memory, output_hash,
                             spread),
            },
            "e2e": {
                "json": block("e2e.json", rate, ttft, memory, output_hash,
                              spread),
                "sse": block("e2e.sse", rate, ttft, memory, output_hash,
                             spread),
                "cli_json": block("e2e.cli_json", rate, ttft, memory,
                                  output_hash, spread),
                "cli_sse": block("e2e.cli_sse", rate, ttft, memory,
                                 output_hash, spread),
            },
        },
    }
    data.update(overrides)
    return data


def failed(gates):
    return [item["gate"] for item in gates if not item["ok"]]


def main():
    baseline = report("baseline", rate=100.0)
    candidate = report("candidate", rate=160.0, stage=CAND_STAGE)
    gates = compare.compare(baseline, candidate)
    check("a passing pair has no failed gate", failed(gates) == [])
    check("the main ratio gate is present",
          any(item["gate"] == "main.output_tokens_per_s" for item in gates))

    broken = copy.deepcopy(candidate)
    broken["completed"] = False
    check("an incomplete report is rejected",
          "candidate.completed" in failed(compare.compare(baseline, broken)))

    broken = copy.deepcopy(candidate)
    broken["output_consistent"] = False
    check("a report with differing outputs is rejected",
          "candidate.output_consistent" in failed(
              compare.compare(baseline, broken)))

    broken = copy.deepcopy(candidate)
    broken["exclusive_window"]["foreign_processes"] = ["123 mlx"]
    check("a foreign process is rejected",
          "candidate.exclusive_window" in failed(
              compare.compare(baseline, broken)))

    broken = copy.deepcopy(candidate)
    broken["exclusive_window"]["hashes_after"]["binary"] = "9" * 64
    check("an artifact that changed during the run is rejected",
          "candidate.artifact_hashes_stable" in failed(
              compare.compare(baseline, broken)))

    broken = copy.deepcopy(candidate)
    del broken["exclusive_window"]["hashes_before"]["stage"]
    changed = failed(compare.compare(baseline, broken))
    check("a missing artifact hash is rejected",
          "candidate.artifact_hashes_valid" in changed)

    broken = copy.deepcopy(candidate)
    broken["exclusive_window"]["hashes_before"]["stage"] = "not-a-hash"
    check("a malformed artifact hash is rejected",
          "candidate.artifact_hashes_valid" in failed(
              compare.compare(baseline, broken)))

    broken = copy.deepcopy(candidate)
    broken["results"]["e2e"]["json"]["samples"][3]["output_sha256"] = "0" * 64
    changed = failed(compare.compare(baseline, broken))
    check("a sample that changes output is rejected",
          "output.candidate.e2e.json" in changed
          and "output.cross_report.e2e.json" in changed)

    broken = copy.deepcopy(candidate)
    for sample in broken["results"]["e2e"]["sse"]["samples"]:
        sample["output_sha256"] = "0" * 64
    changed = failed(compare.compare(baseline, broken))
    check("a path that differs across reports is rejected",
          "output.cross_report.e2e.sse" in changed)

    broken = copy.deepcopy(candidate)
    for sample in broken["results"]["e2e"]["sse"]["samples"]:
        sample["output_sha256"] = "0" * 64
    check("a path that differs from the other paths is rejected",
          "output.candidate.consistent" in failed(
              compare.compare(baseline, broken)))

    check("a ratio below 1.5 is rejected",
          "main.output_tokens_per_s" in failed(
              compare.compare(baseline, report("candidate", rate=140.0,
                                               stage=CAND_STAGE))))

    check("a TTFT regression above 5% is rejected",
          "ttft.e2e.sse" in failed(
              compare.compare(baseline, report("candidate", rate=160.0,
                                               ttft=40.0 * 1.06,
                                               stage=CAND_STAGE))))

    # A slower total on one path while the main ratio still passes: the gate
    # must catch the path, not only the main metric.
    broken = copy.deepcopy(candidate)
    samples = broken["results"]["e2e"]["sse"]["samples"]
    baseline_total = compare.median(baseline["results"]["e2e"]["sse"], "total_ms")
    for sample in samples:
        sample["total_ms"] = baseline_total * 1.06
        sample["output_tokens_per_s"] = (compare.MAIN_TOKENS
                                         / (sample["total_ms"] / 1000.0))
    measured = [item for item in samples if item["warm"]]
    for metric in ("total_ms", "output_tokens_per_s"):
        broken["results"]["e2e"]["sse"]["warm"][metric] = compare.recompute(
            measured, metric)
    check("a total latency regression above 5% is rejected",
          "total.e2e.sse" in failed(compare.compare(baseline, broken)))

    check("a memory regression above 5% is rejected",
          "memory.direct" in failed(
              compare.compare(baseline, report("candidate", rate=160.0,
                                               memory=0.53,
                                               stage=CAND_STAGE))))

    check("an unstable median is rejected",
          "stability.e2e.json.output_tokens_per_s" in failed(
              compare.compare(baseline, report("candidate", rate=160.0,
                                               spread=0.12,
                                               stage=CAND_STAGE))))

    broken = copy.deepcopy(candidate)
    del broken["results"]["stage"]["sse"]
    check("a missing path is rejected",
          "candidate.path_present.stage.sse" in failed(
              compare.compare(baseline, broken)))

    broken = copy.deepcopy(candidate)
    broken["configuration"]["max_tokens"] = 16
    check("a 16-token report is rejected",
          "config.max_tokens" in failed(compare.compare(baseline, broken)))

    broken = copy.deepcopy(candidate)
    broken["configuration"]["samples"] = 19
    check("fewer than 20 samples is rejected",
          "config.samples.candidate" in failed(
              compare.compare(baseline, broken)))

    # One warm row below the 64-token budget: the configured budget is still
    # 64, so only the raw rows can reveal it.
    broken = copy.deepcopy(candidate)
    sample = broken["results"]["e2e"]["json"]["samples"][0]
    sample["generation_tokens"] = 63
    check("a warm row that stopped before 64 tokens is rejected",
          "rows.candidate.e2e.json.generation_tokens" in failed(
              compare.compare(baseline, broken)))

    broken = copy.deepcopy(candidate)
    broken["results"]["e2e"]["json"]["samples"] = [
        item for item in broken["results"]["e2e"]["json"]["samples"]
        if item["warm"]][:compare.MIN_SAMPLES - 1]
    check("19 warm rows are rejected",
          "rows.candidate.e2e.json.warm_count" in failed(
              compare.compare(baseline, broken)))

    broken = copy.deepcopy(candidate)
    broken["results"]["e2e"]["json"]["warm"]["total_ms"]["median"] = 1.0
    check("a stored summary that does not match its rows is rejected",
          "rows.candidate.e2e.json.summary.total_ms" in failed(
              compare.compare(baseline, broken)))

    broken = copy.deepcopy(candidate)
    broken["results"]["e2e"]["json"]["warm"]["total_ms"]["cv"] = -0.01
    broken["results"]["e2e"]["json"]["samples"][0]["output_tokens_per_s"] = (
        float("inf"))
    changed = failed(compare.compare(baseline, broken))
    check("an infinite rate and a negative cv are rejected",
          "rows.candidate.e2e.json.numbers" in changed
          and "stability.e2e.json.total_ms" in changed)

    broken = copy.deepcopy(candidate)
    broken["results"]["e2e"]["json"]["samples"][2]["peak_memory_gb"] = 0.0
    check("a non-positive memory row is rejected",
          "rows.candidate.e2e.json.numbers" in failed(
              compare.compare(baseline, broken)))

    broken = copy.deepcopy(candidate)
    broken["results"]["e2e"]["json"]["samples"][2]["prompt_tokens"] = 36
    check("a prompt that changed within a report is rejected",
          "rows.candidate.e2e.json.prompt_tokens" in failed(
              compare.compare(baseline, broken)))

    broken = copy.deepcopy(candidate)
    for item in broken["results"]["e2e"]["json"]["samples"]:
        item["prompt_tokens"] = 36
    check("a prompt that differs across reports is rejected",
          "prompt.cross_report.e2e.json" in failed(
              compare.compare(baseline, broken)))

    broken = copy.deepcopy(candidate)
    broken["environment"]["binary"] = {"path": "/other", "sha256": "8" * 64}
    check("a different binary is rejected",
          "env.binary_sha256" in failed(compare.compare(baseline, broken)))

    for key in ("host", "machine", "interpreter", "pyzmq", "tokenizers"):
        broken = copy.deepcopy(candidate)
        broken["environment"][key] = "other"
        check(f"a different {key} is rejected",
              f"env.{key}" in failed(compare.compare(baseline, broken)))

    broken = copy.deepcopy(candidate)
    broken["environment"]["script"]["sha256"] = "7" * 64
    check("a different harness is rejected",
          "env.harness_sha256" in failed(compare.compare(baseline, broken)))

    check("the same stage on both sides is rejected",
          "env.stage_distinct" in failed(
              compare.compare(baseline, report("candidate", rate=160.0))))

    broken = copy.deepcopy(candidate)
    del broken["exclusive_window"]["hashes_before"]["model.tokenizer.json"]
    check("a missing model input hash is rejected",
          "env.model_inputs" in failed(compare.compare(baseline, broken)))

    broken = copy.deepcopy(candidate)
    broken["exclusive_window"]["hashes_before"]["model.config.json"] = "4" * 64
    broken["exclusive_window"]["hashes_after"]["model.config.json"] = "4" * 64
    check("a model input that differs across reports is rejected",
          "env.model_inputs" in failed(compare.compare(baseline, broken)))

    # The completeness rule must fail on its own, with both reports missing the
    # same input: otherwise only the cross-report equality is tested.
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template"):
        broken = copy.deepcopy(candidate)
        other = copy.deepcopy(baseline)
        for data in (broken, other):
            del data["exclusive_window"]["hashes_before"][f"model.{name}"]
            del data["exclusive_window"]["hashes_after"][f"model.{name}"]
        check(f"both reports without model.{name} are rejected",
              "env.model_inputs" in failed(compare.compare(other, broken)))

    # Every field of a stored summary is compared with its raw value, not only
    # count, median, and cv.
    for field, value in (("p90", -5.0), ("stdev", -3.0), ("min", -1.0),
                         ("max", 1e9), ("p10", -2.0)):
        broken = copy.deepcopy(candidate)
        broken["results"]["e2e"]["json"]["warm"]["total_ms"][field] = value
        check(f"a stored {field} that no row supports is rejected",
              "rows.candidate.e2e.json.summary.total_ms" in failed(
                  compare.compare(baseline, broken)))

    broken = copy.deepcopy(candidate)
    del broken["results"]["e2e"]["json"]["warm"]["total_ms"]["cv"]
    check("a stored summary without cv is rejected",
          "rows.candidate.e2e.json.summary.total_ms" in failed(
              compare.compare(baseline, broken)))

    broken = copy.deepcopy(candidate)
    broken["results"]["e2e"]["json"]["warm"]["total_ms"]["median"] = float("inf")
    check("an infinite stored median is rejected",
          "rows.candidate.e2e.json.summary.total_ms" in failed(
              compare.compare(baseline, broken)))

    # A metric outside the four main ones: only the full recomputation sees it.
    broken = copy.deepcopy(candidate)
    for item in broken["results"]["e2e"]["json"]["samples"]:
        item["stage_ttft_ms"] = -1000.0
    broken["results"]["e2e"]["json"]["warm"]["stage_ttft_ms"] = (
        compare.recompute(broken["results"]["e2e"]["json"]["samples"],
                          "stage_ttft_ms"))
    check("a negative stage timing is rejected",
          "rows.candidate.e2e.json.numbers" in failed(
              compare.compare(baseline, broken)))

    # An infinite candidate median must not pass the ratio gate.
    broken = copy.deepcopy(candidate)
    for item in broken["results"]["e2e"]["json"]["samples"]:
        item["output_tokens_per_s"] = float("inf")
    broken["results"]["e2e"]["json"]["warm"]["output_tokens_per_s"] = (
        compare.recompute(broken["results"]["e2e"]["json"]["samples"],
                          "output_tokens_per_s"))
    changed = failed(compare.compare(baseline, broken))
    check("an infinite candidate median is rejected",
          "main.output_tokens_per_s" in changed
          and "rows.candidate.e2e.json.numbers" in changed)

    # Every formal report pins stage, harness, weights, and the model inputs.
    for name in ("script", "weights", "binary"):
        broken = copy.deepcopy(candidate)
        del broken["exclusive_window"]["hashes_before"][name]
        del broken["exclusive_window"]["hashes_after"][name]
        check(f"a missing {name} artifact hash is rejected",
              "candidate.artifact_hashes_valid" in failed(
                  compare.compare(baseline, broken)))

    broken = copy.deepcopy(candidate)
    del broken["exclusive_window"]["hashes_before"]["model.chat_template"]
    del broken["exclusive_window"]["hashes_after"]["model.chat_template"]
    check("a report without a chat-template hash is rejected",
          "env.model_inputs" in failed(compare.compare(baseline, broken)))

    broken = copy.deepcopy(candidate)
    broken["environment"]["stage"]["sha256"] = "6" * 64
    check("an environment stage hash that differs from the artifact is rejected",
          "candidate.stage_hash_bound" in failed(
              compare.compare(baseline, broken)))

    broken = copy.deepcopy(candidate)
    broken["environment"]["script"]["sha256"] = "6" * 64
    check("an environment harness hash that differs from the artifact is rejected",
          "candidate.harness_hash_bound" in failed(
              compare.compare(baseline, broken)))

    # The manifest binds the two stages and the harness. Freezing the same
    # stage twice must fail, and a later run with another stage must fail.
    bound, errors = compare.bindings(baseline,
                                     report("candidate", rate=160.0))
    check("a manifest cannot be frozen for one stage", bool(errors))
    bound, errors = compare.bindings(
        baseline, report("candidate", rate=160.0, stage=CAND_STAGE,
                         script="6" * 64))
    check("a manifest cannot be frozen across harnesses", bool(errors))
    bound, errors = compare.bindings(baseline, candidate)
    check("a distinct-stage pair freezes stage and harness hashes",
          not errors and bound["baseline_stage"] == BASE_STAGE
          and bound["candidate_stage"] == CAND_STAGE
          and bound["harness"] == SCRIPT)

    frozen = {**compare.manifest(), **bound}
    check("a report with another package version fails the manifest",
          any(name.startswith("manifest.baseline.mlx")
              for name in failed(compare.manifest_gates(
                  {**baseline, "environment": {**baseline["environment"],
                                               "mlx": "0.0.0"}},
                  candidate, frozen))))
    check("a report from another stage fails the manifest",
          "manifest.candidate.stage" in failed(
              compare.manifest_gates(
                  baseline,
                  {**candidate, "environment": {
                      **candidate["environment"],
                      "stage": {"path": "/stage", "sha256": "5" * 64}}},
                  frozen)))
    check("a report from another harness fails the manifest",
          "manifest.baseline.harness" in failed(
              compare.manifest_gates(
                  {**baseline, "environment": {
                      **baseline["environment"],
                      "script": {"path": "/harness", "sha256": "5" * 64}}},
                  candidate, frozen)))

    # The command line must exit non-zero and name the gate that failed.
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        base_path = directory / "baseline.json"
        cand_path = directory / "candidate.json"
        manifest_path = directory / "manifest.json"
        # The reports must carry the versions of the interpreter that runs
        # them, or the manifest gate fails for a reason unrelated to the test.
        versions = {key: value for key, value in compare.manifest().items()
                    if key != "interpreter"}
        passing_base = report("baseline", rate=100.0)
        passing_base["environment"].update(versions)
        passing_cand = report("candidate", rate=160.0, stage=CAND_STAGE)
        passing_cand["environment"].update(versions)
        passing_base["environment"]["interpreter"] = sys.executable
        passing_cand["environment"]["interpreter"] = sys.executable
        base_path.write_text(json.dumps(passing_base))
        cand_path.write_text(json.dumps(passing_base))
        result = subprocess.run(
            [sys.executable, "-I", str(HERE / "mlx_perf_compare.py"),
             str(base_path), str(cand_path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        check("a failed comparison exits non-zero",
              result.returncode != 0
              and "main.output_tokens_per_s" in result.stderr)

        # A failed pair must not freeze the manifest: the next run would then
        # be bound to a measurement that was never accepted.
        refused = directory / "refused-manifest.json"
        result = subprocess.run(
            [sys.executable, "-I", str(HERE / "mlx_perf_compare.py"),
             str(base_path), str(cand_path), "--manifest", str(refused)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        check("a failed pair does not freeze the manifest",
              result.returncode != 0 and not refused.exists())

        cand_path.write_text(json.dumps(passing_cand))
        # Without --manifest there are no stage or harness bindings to check;
        # the comparison itself must still pass.
        result = subprocess.run(
            [sys.executable, "-I", str(HERE / "mlx_perf_compare.py"),
             str(base_path), str(cand_path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        check("a passing comparison without a manifest exits zero",
              result.returncode == 0 and "PASS all" in result.stdout)

        result = subprocess.run(
            [sys.executable, "-I", str(HERE / "mlx_perf_compare.py"),
             str(base_path), str(cand_path), "--manifest", str(manifest_path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        check("a passing comparison exits zero and freezes the manifest",
              result.returncode == 0 and manifest_path.is_file()
              and "PASS all" in result.stdout)
        frozen_data = json.loads(manifest_path.read_text())
        check("the manifest records the interpreter and package versions",
              Path(frozen_data.get("interpreter", "")).is_absolute()
              and frozen_data.get("msgpack") and frozen_data.get("mlx")
              and frozen_data.get("mlx_lm") and frozen_data.get("pyzmq")
              and frozen_data.get("tokenizers")
              and frozen_data.get("transformers"))
        check("the manifest binds both stages and the harness",
              frozen_data.get("baseline_stage") == BASE_STAGE
              and frozen_data.get("candidate_stage") == CAND_STAGE
              and frozen_data.get("harness") == SCRIPT)

        other = report("candidate", rate=160.0)
        other["environment"].update(versions)
        other["environment"]["interpreter"] = sys.executable
        cand_path.write_text(json.dumps(other))
        result = subprocess.run(
            [sys.executable, "-I", str(HERE / "mlx_perf_compare.py"),
             str(base_path), str(cand_path), "--manifest", str(manifest_path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        check("a frozen manifest rejects the same stage on both sides",
              result.returncode != 0
              and "env.stage_distinct" in result.stderr)

        # A failed pair must not freeze the manifest it was judged against.
        fresh_path = directory / "fresh.json"
        slow = report("candidate", rate=100.0, stage=CAND_STAGE)
        slow["environment"].update(versions)
        slow["environment"]["interpreter"] = sys.executable
        cand_path.write_text(json.dumps(slow))
        result = subprocess.run(
            [sys.executable, "-I", str(HERE / "mlx_perf_compare.py"),
             str(base_path), str(cand_path), "--manifest", str(fresh_path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        check("a failed pair does not freeze a manifest",
              result.returncode != 0 and not fresh_path.exists())
    return 0


if __name__ == "__main__":
    sys.exit(main())

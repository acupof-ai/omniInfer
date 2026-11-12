#!/usr/bin/env python3
"""Compare two complete `tests/mlx_perf.py` reports.

Every gate is evaluated and printed. A failed report exits non-zero and names
the gate and the two values that failed it. No model, server, or process is
started here: the input is two JSON reports and nothing else.

The main metric is the 64-token warm median `output_tokens_per_s` of the HTTP
JSON path. The comparison requires `ratio = candidate / baseline >= 1.5`.
Every stored summary is recomputed from the raw warm rows, and a stored number
that does not match its rows fails the run.
"""

import argparse
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import statistics
import sys


MAIN_TOKENS = 64
MIN_SAMPLES = 20
REQUIRED_RATIO = 1.5
TOLERANCE = 0.05
CV_MAX = 0.10
MAIN_PATH = "e2e.json"
# Every path a `--mode all` run must produce. A partial report cannot be
# compared: a missing path would silently drop one of the gates.
PATHS = ("direct", "stage.json", "stage.sse", "e2e.json", "e2e.sse",
         "e2e.cli_json", "e2e.cli_sse")
# Paths whose rows carry a first-token boundary.
TTFT_PATHS = ("stage.sse", "e2e.sse", "e2e.cli_sse")
# Every numeric field the harness summarizes. Each one is recomputed.
NUMERIC_METRICS = ("ttft_ms", "total_ms", "generation_tokens",
                   "decode_tokens_per_s", "output_tokens_per_s",
                   "prompt_tokens", "prompt_tokens_per_s", "peak_memory_gb",
                   "stage_ttft_ms", "stage_total_ms")
# Every field of a stored summary, each compared with its raw value.
SUMMARY_FIELDS = ("count", "median", "min", "max", "p10", "p90", "stdev", "cv")
# The model input files a formal report must pin. A chat template is recorded
# either as its own file or as `chat_template`, extracted from the tokenizer
# config; at least one of the two must be present.
REQUIRED_MODEL_INPUTS = ("config.json", "tokenizer.json",
                         "tokenizer_config.json")
TEMPLATE_INPUTS = ("chat_template.jinja", "chat_template")
REQUIRED_ARTIFACTS = ("stage", "script", "weights", "binary")
# The environment a comparison pins. host and machine must agree between the
# two reports; they are not frozen in the manifest, because a report may be
# copied between machines for review.
ENVIRONMENT_KEYS = ("host", "machine", "interpreter", "python", "mlx",
                    "mlx_lm", "msgpack", "pyzmq", "tokenizers",
                    "transformers")
MANIFEST_KEYS = ("interpreter", "python", "mlx", "mlx_lm", "msgpack",
                 "pyzmq", "tokenizers", "transformers")
MANIFEST_BINDINGS = ("baseline_stage", "candidate_stage", "harness")
MODEL_INPUT_PREFIX = "model."


def package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def walk(report, path):
    data = report.get("results")
    for part in path.split("."):
        if not isinstance(data, dict):
            return None
        data = data.get(part)
    return data


def warm(row_block, metric):
    summary = (row_block or {}).get("warm", {}).get(metric)
    return None if not isinstance(summary, dict) else summary


def median(row_block, metric):
    summary = warm(row_block, metric)
    return None if summary is None else summary.get("median")


def hex_hash(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def finite_positive(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def gate(gates, name, ok, detail=""):
    gates.append({"gate": name, "ok": bool(ok), "detail": detail})


def rows_of(report, path):
    return (walk(report, path) or {}).get("samples") or []


def measured_rows(report, path):
    """Raw rows that a median may use: warm and not cold."""
    return [row for row in rows_of(report, path)
            if row.get("warm") and not row.get("cold")]


def percentile(ordered, fraction):
    """The harness definition, repeated so the two agree exactly."""
    index = round(fraction * (len(ordered) - 1))
    return ordered[min(len(ordered) - 1, max(0, index))]


def recompute(rows, metric):
    values = [row.get(metric) for row in rows
              if isinstance(row.get(metric), (int, float))
              and not isinstance(row.get(metric), bool)
              and math.isfinite(row.get(metric))]
    if not values:
        return None
    ordered = sorted(values)
    value = statistics.median(ordered)
    result = {"count": len(ordered), "median": value,
              "min": ordered[0], "max": ordered[-1],
              "p10": percentile(ordered, 0.10),
              "p90": percentile(ordered, 0.90)}
    if len(ordered) > 1:
        result["stdev"] = statistics.stdev(ordered)
        if value:
            result["cv"] = result["stdev"] / value
    return result


def same(first, second):
    """Verify the two values agree within float noise.

    Infinity is only equal to itself: a tolerance scaled by an infinite value
    would accept any finite number.
    """
    if isinstance(first, bool) or isinstance(second, bool):
        return first == second
    if not isinstance(first, (int, float)) or not isinstance(second, (int, float)):
        return False
    if not (math.isfinite(first) and math.isfinite(second)):
        return first == second
    return abs(first - second) <= max(1e-12, 1e-9 * abs(first))


def check_rows(gates, label, report, path):
    """Raw rows and stored summaries must agree, field by field."""
    measured = measured_rows(report, path)
    gate(gates, f"rows.{label}.{path}.warm_count",
         len(measured) >= MIN_SAMPLES,
         f"warm non-cold={len(measured)}; required >= {MIN_SAMPLES}")
    tokens_ok = all(row.get("generation_tokens") == MAIN_TOKENS
                    for row in measured)
    gate(gates, f"rows.{label}.{path}.generation_tokens",
         bool(measured) and tokens_ok,
         "every warm row must generate "
         f"{MAIN_TOKENS} tokens; values="
         f"{sorted({row.get('generation_tokens') for row in measured})}")

    numbers_ok = True
    detail = ""
    for row in measured:
        iteration = row.get("iteration")
        total = row.get("total_ms")
        rate = row.get("output_tokens_per_s")
        expected = (row["generation_tokens"] / (total / 1000.0)
                    if finite_positive(total)
                    and isinstance(row.get("generation_tokens"), int)
                    and row["generation_tokens"] else None)
        if not finite_positive(total):
            numbers_ok, detail = False, f"sample {iteration} total_ms={total!r}"
        elif not finite_positive(rate):
            numbers_ok, detail = False, (
                f"sample {iteration} output_tokens_per_s={rate!r}")
        elif not same(rate, expected):
            numbers_ok, detail = False, (
                f"sample {iteration} rate={rate!r} != "
                f"generation_tokens/total_ms={expected!r}")
        elif path in TTFT_PATHS and not finite_positive(row.get("ttft_ms")):
            numbers_ok, detail = False, (
                f"sample {iteration} ttft_ms={row.get('ttft_ms')!r} on a "
                "streaming row")
        else:
            for metric in NUMERIC_METRICS:
                value = row.get(metric)
                if value is None or metric == "generation_tokens":
                    continue
                if not finite_positive(value):
                    numbers_ok, detail = False, (
                        f"sample {iteration} {metric}={value!r}")
                    break
        if not numbers_ok:
            break
    gate(gates, f"rows.{label}.{path}.numbers", numbers_ok,
         detail or "every numeric field is finite and positive")

    block = walk(report, path) or {}
    for metric in NUMERIC_METRICS:
        stored = warm(block, metric)
        expected = recompute(measured, metric)
        if stored is None and expected is None:
            continue
        ok = (stored is not None and expected is not None
              and stored.get("count") == expected["count"]
              and all(field not in expected
                      or same(stored.get(field), expected[field])
                      for field in SUMMARY_FIELDS))
        gate(gates, f"rows.{label}.{path}.summary.{metric}", ok,
             f"stored={stored!r} recomputed={expected!r}")

    prompt = {row.get("prompt_tokens") for row in measured}
    gate(gates, f"rows.{label}.{path}.prompt_tokens",
         len(prompt) == 1 and finite_positive(next(iter(prompt))),
         f"prompt_tokens={sorted(str(value) for value in prompt)}")


def compare(baseline, candidate):
    """Return one entry per gate. Every gate runs, so one failure is not hidden
    by the next."""
    gates = []

    for label, report in (("baseline", baseline), ("candidate", candidate)):
        gate(gates, f"{label}.role", report.get("role") == label,
             f"role={report.get('role')!r}; required {label!r}")
        gate(gates, f"{label}.completed",
             report.get("completed") is True,
             f"completed={report.get('completed')!r} "
             f"error={report.get('error')!r}")
        gate(gates, f"{label}.no_skipped_path",
             report.get("skipped_paths") == [],
             f"skipped_paths={report.get('skipped_paths')!r}")
        gate(gates, f"{label}.output_consistent",
             report.get("output_consistent") is True,
             f"output_consistent={report.get('output_consistent')!r}")
        window = report.get("exclusive_window") or {}
        gate(gates, f"{label}.exclusive_window",
             window.get("foreign_processes") == [],
             f"foreign_processes={window.get('foreign_processes')!r}")
        before, after = window.get("hashes_before"), window.get("hashes_after")
        well_formed = (isinstance(before, dict) and isinstance(after, dict)
                       and before and after
                       and all(hex_hash(value) for value in before.values())
                       and all(hex_hash(value) for value in after.values())
                       and all(name in before for name in REQUIRED_ARTIFACTS))
        gate(gates, f"{label}.artifact_hashes_valid", well_formed,
             f"before={before!r} after={after!r}")
        gate(gates, f"{label}.artifact_hashes_stable", before == after,
             f"before={before!r} after={after!r}")
        environment = report.get("environment") or {}
        # The environment block and the measured artifact hashes are one fact
        # recorded twice. They must agree.
        if isinstance(before, dict):
            gate(gates, f"{label}.stage_hash_bound",
                 (environment.get("stage") or {}).get("sha256")
                 == before.get("stage"),
                 f"environment={(environment.get('stage') or {}).get('sha256')!r} "
                 f"hashes={before.get('stage')!r}")
            gate(gates, f"{label}.harness_hash_bound",
                 (environment.get("script") or {}).get("sha256")
                 == before.get("script"),
                 f"environment={(environment.get('script') or {}).get('sha256')!r} "
                 f"hashes={before.get('script')!r}")
            gate(gates, f"{label}.weights_hash_bound",
                 hex_hash(before.get("weights")),
                 f"hashes={before.get('weights')!r}")
        for path in PATHS:
            block = walk(report, path)
            gate(gates, f"{label}.path_present.{path}",
                 bool((block or {}).get("samples")),
                 f"samples={len((block or {}).get('samples') or [])}")

    configs = [report.get("configuration") or {}
               for report in (baseline, candidate)]
    base_config, cand_config = configs
    for key in ("model", "model_safetensors_sha256", "prompt", "temperature",
                "warmup", "max_attempts"):
        gate(gates, f"config.{key}",
             base_config.get(key) == cand_config.get(key)
             and base_config.get(key) is not None,
             f"baseline={base_config.get(key)!r} "
             f"candidate={cand_config.get(key)!r}")
    gate(gates, "config.max_tokens",
         base_config.get("max_tokens") == cand_config.get("max_tokens") == MAIN_TOKENS,
         f"baseline={base_config.get('max_tokens')!r} "
         f"candidate={cand_config.get('max_tokens')!r}; required {MAIN_TOKENS}")
    for label, config in zip(("baseline", "candidate"), configs):
        gate(gates, f"config.samples.{label}",
             isinstance(config.get("samples"), int)
             and config["samples"] >= MIN_SAMPLES,
             f"samples={config.get('samples')!r}; required >= {MIN_SAMPLES}")

    environments = [report.get("environment") or {}
                    for report in (baseline, candidate)]
    base_env, cand_env = environments
    gate(gates, "env.binary_sha256",
         hex_hash((base_env.get("binary") or {}).get("sha256"))
         and (base_env.get("binary") or {}).get("sha256")
         == (cand_env.get("binary") or {}).get("sha256"),
         f"baseline={(base_env.get('binary') or {}).get('sha256')!r} "
         f"candidate={(cand_env.get('binary') or {}).get('sha256')!r}")
    for key in ENVIRONMENT_KEYS:
        gate(gates, f"env.{key}",
             base_env.get(key) is not None
             and base_env.get(key) == cand_env.get(key),
             f"baseline={base_env.get(key)!r} candidate={cand_env.get(key)!r}")

    # The harness itself is an input to the measurement.
    base_script = (base_env.get("script") or {}).get("sha256")
    cand_script = (cand_env.get("script") or {}).get("sha256")
    gate(gates, "env.harness_sha256",
         hex_hash(base_script) and base_script == cand_script,
         f"baseline={base_script!r} candidate={cand_script!r}")
    # The two stages must be different files; the manifest binds both hashes.
    base_stage = (base_env.get("stage") or {}).get("sha256")
    cand_stage = (cand_env.get("stage") or {}).get("sha256")
    gate(gates, "env.stage_distinct",
         hex_hash(base_stage) and hex_hash(cand_stage)
         and base_stage != cand_stage,
         f"baseline={base_stage!r} candidate={cand_stage!r}")

    # Model input files: required, recorded, and equal across runs.
    def model_inputs(report):
        hashes = (report.get("exclusive_window") or {}).get("hashes_before") or {}
        return {name[len(MODEL_INPUT_PREFIX):]: value
                for name, value in hashes.items()
                if name.startswith(MODEL_INPUT_PREFIX)}

    base_inputs, cand_inputs = model_inputs(baseline), model_inputs(candidate)
    required_inputs = REQUIRED_MODEL_INPUTS
    base_complete = all(name in base_inputs for name in required_inputs) \
        and any(name in base_inputs for name in TEMPLATE_INPUTS)
    cand_complete = all(name in cand_inputs for name in required_inputs) \
        and any(name in cand_inputs for name in TEMPLATE_INPUTS)
    gate(gates, "env.model_inputs",
         base_complete and cand_complete and base_inputs == cand_inputs,
         f"baseline={sorted(base_inputs)} candidate={sorted(cand_inputs)}")

    # Output identity: each path has one hash, all paths agree, both reports
    # agree, and the report's own consistency flag is true.
    baseline_hashes = {}
    candidate_hashes = {}
    for path in PATHS:
        for label, report, collected in (
                ("baseline", baseline, baseline_hashes),
                ("candidate", candidate, candidate_hashes)):
            values = {row.get("output_sha256") for row in rows_of(report, path)}
            collected[path] = values
            gate(gates, f"output.{label}.{path}",
                 len(values) == 1 and hex_hash(next(iter(values))),
                 f"hashes={sorted(str(value) for value in values)}")
    for label, collected in (("baseline", baseline_hashes),
                             ("candidate", candidate_hashes)):
        gate(gates, f"output.{label}.consistent",
             len({next(iter(values)) for values in collected.values()
                  if len(values) == 1}) == 1,
             f"per-path={ {key: sorted(value) for key, value in collected.items()} }")
    for path in PATHS:
        gate(gates, f"output.cross_report.{path}",
             baseline_hashes.get(path) == candidate_hashes.get(path)
             and len(baseline_hashes.get(path) or ()) == 1,
             f"baseline={sorted(baseline_hashes.get(path) or [])} "
             f"candidate={sorted(candidate_hashes.get(path) or [])}")

    for label, report in (("baseline", baseline), ("candidate", candidate)):
        for path in PATHS:
            check_rows(gates, label, report, path)

    # The same prompt must tokenize to the same length in both reports.
    for path in PATHS:
        values = [{row.get("prompt_tokens")
                   for row in measured_rows(report, path)}
                  for report in (baseline, candidate)]
        gate(gates, f"prompt.cross_report.{path}",
             len(values[0]) == 1 and values[0] == values[1]
             and finite_positive(next(iter(values[0]))),
             f"baseline={sorted(str(value) for value in values[0])} "
             f"candidate={sorted(str(value) for value in values[1])}")

    baseline_main = median(walk(baseline, MAIN_PATH), "output_tokens_per_s")
    candidate_main = median(walk(candidate, MAIN_PATH), "output_tokens_per_s")
    ratio = (candidate_main / baseline_main
             if finite_positive(baseline_main) and finite_positive(candidate_main)
             else None)
    gate(gates, "main.output_tokens_per_s",
         ratio is not None and ratio >= REQUIRED_RATIO,
         f"baseline={baseline_main!r} candidate={candidate_main!r} "
         f"ratio={ratio!r}; required >= {REQUIRED_RATIO}")

    for path in TTFT_PATHS:
        base_value = median(walk(baseline, path), "ttft_ms")
        cand_value = median(walk(candidate, path), "ttft_ms")
        gate(gates, f"ttft.{path}",
             finite_positive(base_value) and finite_positive(cand_value)
             and cand_value <= base_value * (1 + TOLERANCE),
             f"baseline={base_value!r} candidate={cand_value!r}; "
             f"allowed <= {None if base_value is None else base_value * (1 + TOLERANCE)!r}")

    for path in PATHS:
        base_value = median(walk(baseline, path), "total_ms")
        cand_value = median(walk(candidate, path), "total_ms")
        gate(gates, f"total.{path}",
             finite_positive(base_value) and finite_positive(cand_value)
             and cand_value <= base_value * (1 + TOLERANCE),
             f"baseline={base_value!r} candidate={cand_value!r}; "
             f"allowed <= {None if base_value is None else base_value * (1 + TOLERANCE)!r}")

    for path in PATHS:
        base_value = median(walk(baseline, path), "peak_memory_gb")
        cand_value = median(walk(candidate, path), "peak_memory_gb")
        gate(gates, f"memory.{path}",
             finite_positive(base_value) and finite_positive(cand_value)
             and cand_value <= base_value * (1 + TOLERANCE),
             f"baseline={base_value!r} candidate={cand_value!r}; "
             f"allowed <= {None if base_value is None else base_value * (1 + TOLERANCE)!r}")

    # Stability: every stored cv must exist, be finite, and stay under the
    # limit. A cv above 0.10 means the window was not exclusive.
    for path in PATHS:
        for metric in NUMERIC_METRICS:
            summaries = [warm(walk(report, path), metric)
                         for report in (baseline, candidate)]
            if all(summary is None for summary in summaries):
                continue
            values = [None if summary is None else summary.get("cv")
                      for summary in summaries]
            gate(gates, f"stability.{path}.{metric}",
                 all(isinstance(value, (int, float))
                     and not isinstance(value, bool)
                     and math.isfinite(value) and 0 <= value <= CV_MAX
                     for value in values),
                 f"cv={values!r}; required 0 <= cv <= {CV_MAX}")

    return gates


def bindings(baseline, candidate):
    """The values the manifest freezes for this baseline/candidate pair.

    Both reports must come from the same harness, and the two stages must be
    different files. Without this, a run could compare a stage with itself.
    """
    base_env = baseline.get("environment") or {}
    cand_env = candidate.get("environment") or {}
    baseline_stage = (base_env.get("stage") or {}).get("sha256")
    candidate_stage = (cand_env.get("stage") or {}).get("sha256")
    base_script = (base_env.get("script") or {}).get("sha256")
    cand_script = (cand_env.get("script") or {}).get("sha256")
    errors = []
    if baseline_stage == candidate_stage:
        errors.append("baseline and candidate use the same stage")
    if not hex_hash(baseline_stage):
        errors.append("the baseline report has no stage hash")
    if not hex_hash(candidate_stage):
        errors.append("the candidate report has no stage hash")
    if base_script != cand_script or not hex_hash(base_script):
        errors.append("the two reports come from different harness files")
    return {"baseline_stage": baseline_stage,
            "candidate_stage": candidate_stage,
            "harness": base_script}, errors


def manifest_gates(baseline, candidate, frozen):
    """Both reports must come from the frozen interpreter, packages, harness,
    and the stages the manifest was frozen with."""
    gates = []
    for label, report in (("baseline", baseline), ("candidate", candidate)):
        environment = report.get("environment") or {}
        for key in MANIFEST_KEYS:
            gate(gates, f"manifest.{label}.{key}",
                 environment.get(key) is not None
                 and environment.get(key) == frozen.get(key),
                 f"report={environment.get(key)!r} manifest={frozen.get(key)!r}")
        gate(gates, f"manifest.{label}.harness",
             (environment.get("script") or {}).get("sha256")
             == frozen.get("harness"),
             f"report={(environment.get('script') or {}).get('sha256')!r} "
             f"manifest={frozen.get('harness')!r}")
        gate(gates, f"manifest.{label}.stage",
             (environment.get("stage") or {}).get("sha256")
             == frozen.get(f"{label}_stage"),
             f"report={(environment.get('stage') or {}).get('sha256')!r} "
             f"manifest={frozen.get(f'{label}_stage')!r}")
    return gates


def manifest():
    """The interpreter and package versions a measurement window is frozen to."""
    return {
        "interpreter": sys.executable,
        "python": platform.python_version(),
        "msgpack": package_version("msgpack"),
        "mlx": package_version("mlx"),
        "mlx_lm": package_version("mlx-lm"),
        "pyzmq": package_version("pyzmq"),
        "tokenizers": package_version("tokenizers"),
        "transformers": package_version("transformers"),
    }


def load(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path, default=None,
                        help="write the gate results and the manifest here")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="freeze the environment on first use; on later "
                             "uses, require the reports to match it")
    args = parser.parse_args()

    baseline = load(args.baseline)
    candidate = load(args.candidate)
    compare_gates = compare(baseline, candidate)
    frozen = manifest()
    bound, errors = bindings(baseline, candidate)
    if args.manifest is not None:
        if args.manifest.is_file():
            expected = load(args.manifest)
            missing = [key for key in MANIFEST_KEYS + MANIFEST_BINDINGS
                       if expected.get(key) is None]
            if missing:
                print("FAIL manifest is missing fields: " + ", ".join(missing),
                      file=sys.stderr)
                return 1
            matches = {key: (expected.get(key), frozen.get(key))
                       for key in MANIFEST_KEYS}
            matches = {key: value for key, value in matches.items()
                       if value[0] != value[1]}
            if matches:
                print("FAIL manifest: " + json.dumps(matches, sort_keys=True),
                      file=sys.stderr)
                return 1
            frozen = expected
            print(f"PASS manifest matched {args.manifest}")
        else:
            # A failed pair must not freeze the manifest: it would bind the
            # next run to a measurement that was never accepted. Every gate,
            # including the environment binding, must pass before the write.
            candidate_frozen = {**frozen, **bound}
            failed_compare = [item["gate"] for item in compare_gates
                              if not item["ok"]]
            failed_binding = [item["gate"]
                              for item in manifest_gates(baseline, candidate,
                                                         candidate_frozen)
                              if not item["ok"]]
            if errors or failed_compare or failed_binding:
                print("FAIL manifest cannot be frozen: "
                      + "; ".join(errors + failed_compare + failed_binding),
                      file=sys.stderr)
                return 1
            frozen = candidate_frozen
            args.manifest.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")
            print(f"PASS manifest frozen at {args.manifest}")

    # Without a frozen manifest there are no stage or harness bindings to
    # check; the comparison below still binds the two reports to each other.
    gates = compare_gates
    if "harness" in frozen:
        gates = manifest_gates(baseline, candidate, frozen) + compare_gates
    for item in gates:
        line = f"{'PASS' if item['ok'] else 'FAIL'} {item['gate']}"
        if item["detail"]:
            line += " " + item["detail"]
        print(line, file=sys.stdout if item["ok"] else sys.stderr)
    failed = [item for item in gates if not item["ok"]]
    if args.out is not None:
        args.out.write_text(json.dumps(
            {"manifest": frozen, "gates": gates,
             "failed": [item["gate"] for item in failed]},
            indent=2, sort_keys=True) + "\n")
        print(f"REPORT {args.out}")
    if failed:
        print(f"FAIL {len(failed)} gate(s): "
              + ", ".join(item["gate"] for item in failed), file=sys.stderr)
        return 1
    print(f"PASS all {len(gates)} gates")
    return 0


if __name__ == "__main__":
    sys.exit(main())

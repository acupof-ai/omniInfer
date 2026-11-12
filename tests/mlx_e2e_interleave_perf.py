#!/usr/bin/env python3
"""Sequential per-sample HTTP A/B with fresh, pre-warmed server processes."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "umminfer_mlx_perf", ROOT / "tests" / "mlx_perf.py")
perf = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(perf)
BASELINE_STAGE = Path("/tmp/umminfer-stage-iteration2-baseline.py")
CANDIDATE_STAGE = ROOT / "python" / "umminfer_mlx_stage.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.samples <= perf.MAX_SAMPLES:
        parser.error("samples outside the harness limit")

    lock = perf.acquire_lock()
    needles = sorted(set(perf.patrol_needles(perf.DEFAULT_BINARY,
                                            CANDIDATE_STAGE))
                     | {str(BASELINE_STAGE), BASELINE_STAGE.name})
    perf.check_exclusive(needles)
    guard = lambda: perf.check_exclusive(needles)
    arms = (("baseline", BASELINE_STAGE), ("candidate", CANDIDATE_STAGE))
    rows = {name: [] for name, _ in arms}
    cold_rows = {name: [] for name, _ in arms}
    artifacts = {"baseline_stage": BASELINE_STAGE,
                 "candidate_stage": CANDIDATE_STAGE,
                 "harness": Path(perf.__file__),
                 "binary": perf.DEFAULT_BINARY,
                 "weights": perf.MODEL_PATH / "model.safetensors"}
    hashes_before = {name: perf.digest(path) for name, path in artifacts.items()}
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="umminfer-e2e-interleave-") as tmp:
            for iteration in range(args.samples):
                order = arms if iteration % 2 == 0 else arms[::-1]
                for name, stage in order:
                    result = perf.run_e2e(
                        perf.DEFAULT_BINARY, stage, perf.MODEL_PATH,
                        perf.DEFAULT_MAX_TOKENS, 2, 1,
                        Path(tmp) / f"{iteration}-{name}.jsonl", guard)
                    perf.require_outputs({"e2e": result})
                    warm = [row for row in result["json"]["samples"]
                            if row["warm"]]
                    perf.require(len(warm) == 1,
                                 "a pre-warmed process did not produce one sample")
                    row = warm[0]
                    perf.require(row["generation_tokens"] == 13
                                 and row["finish_reason"] == "stop"
                                 and row["prompt_tokens"] == 35,
                                 f"unexpected generation metadata: {row!r}")
                    row["pair"] = iteration
                    rows[name].append(row)
                    cold_rows[name].append(result["json"]["cold"])
                expected = rows["baseline"][-1]
                actual = rows["candidate"][-1]
                for field in ("output_sha256", "generation_tokens",
                              "finish_reason", "prompt_tokens"):
                    perf.require(actual[field] == expected[field],
                                 f"pair {iteration} changed {field}")
                print(f"PAIR {iteration} "
                      f"baseline={expected['output_tokens_per_s']:.6f} "
                      f"candidate={actual['output_tokens_per_s']:.6f}",
                      flush=True)

        results = {name: perf.block(values) for name, values in rows.items()}
        perf.require_valid_rows({"e2e": {"json": results["baseline"]}},
                                perf.DEFAULT_MAX_TOKENS)
        perf.require_valid_rows({"e2e": {"json": results["candidate"]}},
                                perf.DEFAULT_MAX_TOKENS)
        medians = {name: perf.median_of(result, "output_tokens_per_s")
                   for name, result in results.items()}
        speedup = 100.0 * (medians["candidate"] / medians["baseline"] - 1.0)
        hashes_after = {name: perf.digest(path)
                        for name, path in artifacts.items()}
        perf.require(hashes_after == hashes_before,
                     "an artifact changed during interleaved measurement")
        report = {
            "configuration": {
                "model": str(perf.MODEL_PATH),
                "prompt": perf.PROMPT,
                "temperature": perf.TEMPERATURE,
                "max_tokens": perf.DEFAULT_MAX_TOKENS,
                "samples_per_arm": args.samples,
                "warmup_per_process": 2,
                "fresh_process_per_sample": True,
                "interleaved": True,
                "concurrent_inference": False,
            },
            "hashes": hashes_before,
            "script_sha256": perf.digest(Path(__file__).resolve()),
            "results": results,
            "cold_samples": cold_rows,
            "warm_median_output_tokens_per_s": medians,
            "speedup_percent": speedup,
            "wall_seconds": time.perf_counter() - started,
            "output_sha256": rows["baseline"][0]["output_sha256"],
            "correctness": "all pairs and all transports within each process match",
        }
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        for name, result in results.items():
            perf.print_block(name, result)
        print(f"SPEEDUP percent={speedup:.9f}")
        print(f"REPORT {args.out}")
        return 0
    finally:
        lock.close()


if __name__ == "__main__":
    sys.exit(main())

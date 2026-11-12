#!/usr/bin/env python3
"""Interleaved direct A/B for reusable prompt KV snapshots."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

from mlx_lm import load


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "python" / "umminfer_mlx_stage.py"
BASELINE_STAGE = Path("/tmp/umminfer-stage-iteration2-baseline.py")
PREVIOUS_STAGE = Path("/tmp/umminfer-stage-iteration2-wired.py")
SPEC = importlib.util.spec_from_file_location(
    "umminfer_mlx_perf", ROOT / "tests" / "mlx_perf.py")
perf = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(perf)


def digest_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def token_hash(tokens):
    return hashlib.sha256(json.dumps(tokens).encode("utf-8")).hexdigest()


def collect(generate, prompt, max_tokens):
    started = time.perf_counter()
    first = None
    parts = []
    tokens = []
    last = None
    for response in generate(prompt, max_tokens):
        if first is None and response.text:
            first = time.perf_counter() - started
        parts.append(response.text)
        tokens.append(response.token)
        last = response
    total = time.perf_counter() - started
    perf.require(last is not None, "generation produced no response")
    return first, total, "".join(parts), tokens, last


def signature(first, total, text, tokens, last):
    return {
        "output_sha256": digest_text(text),
        "token_ids_sha256": token_hash(tokens),
        "generation_tokens": last.generation_tokens,
        "finish_reason": last.finish_reason,
        "prompt_tokens": last.prompt_tokens,
    }


def compare(baseline, candidate, prompt, max_tokens):
    expected = signature(*collect(baseline, prompt, max_tokens))
    actual = signature(*collect(candidate, prompt, max_tokens))
    perf.require(actual == expected,
                 f"prompt cache changed output: expected={expected!r} "
                 f"actual={actual!r}")
    return {"prompt_sha256": digest_text(prompt), "baseline": expected,
            "candidate": actual}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=perf.DEFAULT_MAX_TOKENS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.samples <= perf.MAX_SAMPLES:
        parser.error("samples outside the harness limit")
    if not 2 <= args.warmup <= perf.MAX_WARMUP:
        parser.error("warmup must be at least 2 so cold rows stay separate")
    if not 1 <= args.max_tokens <= perf.MAX_TOKENS_CAP:
        parser.error("max_tokens outside the harness limit")

    lock = perf.acquire_lock()
    needles = perf.patrol_needles("/tmp/umminfer-direct-no-binary", STAGE)
    perf.check_exclusive(needles)
    other_inference = perf.foreign_processes(["umm-infer"])
    stage = perf.load_stage_module(STAGE)
    baseline_stage = perf.load_stage_module(BASELINE_STAGE)
    previous_stage = perf.load_stage_module(PREVIOUS_STAGE)
    started = time.perf_counter()
    model, tokenizer = load(str(perf.MODEL_PATH))
    stage._cache_vocabulary(tokenizer)
    load_seconds = time.perf_counter() - started
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": perf.PROMPT}], tokenize=False,
        add_generation_prompt=True)
    baseline = baseline_stage._make_greedy_generate(
        model, tokenizer, skip_final_lookahead=True)
    prompt_cache = previous_stage._make_greedy_generate(
        model, tokenizer, skip_final_lookahead=True,
        reuse_prompt_cache=True, reuse_detokenizer=True)
    candidate = stage._make_greedy_generate(
        model, tokenizer, skip_final_lookahead=True,
        reuse_prompt_cache=True, reuse_detokenizer=True)
    arms = (("baseline", baseline), ("prompt_cache", prompt_cache),
            ("candidate", candidate))
    rows = {name: [] for name, _ in arms}
    try:
        for iteration in range(args.warmup + args.samples):
            offset = iteration % len(arms)
            order = arms[offset:] + arms[:offset]
            for name, generate in order:
                first, total, text, tokens, last = collect(
                    generate, prompt, args.max_tokens)
                row = perf.make_row(
                    iteration, args.warmup, first, total, last, text)
                row["cold"] = iteration == 0
                row["finish_reason"] = last.finish_reason
                row["token_ids_sha256"] = token_hash(tokens)
                rows[name].append(row)
            fields = ("output_sha256", "token_ids_sha256", "generation_tokens",
                      "finish_reason", "prompt_tokens")
            expected = tuple(rows["baseline"][-1][field] for field in fields)
            for name in ("prompt_cache", "candidate"):
                actual = tuple(rows[name][-1][field] for field in fields)
                perf.require(actual == expected,
                             f"{name} changed output at iteration {iteration}: "
                             f"expected={expected!r} actual={actual!r}")
            perf.require(rows["baseline"][-1]["generation_tokens"] > 0,
                         "the fixed request reported no generated tokens")

        measurement_stats = {
            name: {"prompt_cache": dict(getattr(
                       generate, "prompt_cache_stats", {"hits": 0, "misses": 0})),
                   "detokenizer": dict(getattr(
                       generate, "detokenizer_stats", {"builds": 0, "copies": 0}))}
            for name, generate in arms}
        for name in ("prompt_cache", "candidate"):
            perf.require(measurement_stats[name]["prompt_cache"]["hits"]
                         >= args.warmup + args.samples - 1,
                         f"the repeated prompt did not hit the {name} cache")

        # These controls are outside the measured samples. Alternating unlike
        # prompts must miss and must never reuse another prompt's KV state.
        misses_before = candidate.prompt_cache_stats["misses"]
        controls = [compare(baseline, candidate, item, args.max_tokens)
                    for item in (prompt + "\nAnswer with a different tip.",
                                 prompt,
                                 prompt + "\nUse exactly two words.",
                                 prompt)]
        perf.require(candidate.prompt_cache_stats["misses"] - misses_before
                     == len(controls), "the varied-prompt control hit the cache")

        # Closing a generation after one delta models an aborted stage request.
        # The next request must reuse the untouched master snapshot correctly.
        aborted = candidate(prompt, args.max_tokens)
        next(aborted)
        aborted.close()
        hits_before = candidate.prompt_cache_stats["hits"]
        after_abort = compare(baseline, candidate, prompt, args.max_tokens)
        perf.require(candidate.prompt_cache_stats["hits"] == hits_before + 1,
                     "the post-abort request did not use the cached prompt")

        results = {name: perf.block(values) for name, values in rows.items()}
        medians = {name: perf.median_of(value, "output_tokens_per_s")
                   for name, value in results.items()}
        speedups = {
            "prompt_cache_vs_baseline":
                100.0 * (medians["prompt_cache"] / medians["baseline"] - 1.0),
            "candidate_vs_prompt_cache":
                100.0 * (medians["candidate"] / medians["prompt_cache"] - 1.0),
            "candidate_vs_baseline":
                100.0 * (medians["candidate"] / medians["baseline"] - 1.0),
        }
        speedup = speedups["candidate_vs_baseline"]
        report = {
            "configuration": {
                "model": str(perf.MODEL_PATH),
                "model_safetensors_sha256": perf.digest(
                    perf.MODEL_PATH / "model.safetensors"),
                "prompt": perf.PROMPT, "temperature": perf.TEMPERATURE,
                "max_tokens": args.max_tokens,
                "samples": args.samples, "warmup": args.warmup,
                "interleaved": True, "loaded_model_shared": True,
                "baseline": "frozen iteration-1 greedy argmax, fresh KV and detokenizer",
                "prompt_cache": "previous candidate with request-private shallow KV cache objects",
                "candidate": "previous candidate with process-lifetime wired limit",
                "mlx_exclusive": True,
                "other_inference_processes_at_start": other_inference,
            },
            "hashes": {
                "stage": perf.digest(STAGE),
                "baseline_stage": perf.digest(BASELINE_STAGE),
                "previous_stage": perf.digest(PREVIOUS_STAGE),
                "script": perf.digest(Path(__file__).resolve()),
                "harness": perf.digest(Path(perf.__file__).resolve()),
            },
            "model_load_seconds": load_seconds,
            "correctness": {
                "per_sample_fields": ["token_ids_sha256", "output_sha256",
                                      "generation_tokens", "finish_reason",
                                      "prompt_tokens"],
                "samples_per_arm": len(rows["baseline"]),
                "observed_generation_tokens": {
                    name: sorted({row["generation_tokens"] for row in values})
                    for name, values in rows.items()},
                "observed_finish_reasons": {
                    name: sorted({row["finish_reason"] for row in values})
                    for name, values in rows.items()},
                "all_equal_to_baseline": True,
                "measurement_cache_stats": measurement_stats,
                "varied_prompt_controls": controls,
                "varied_prompts_all_missed": True,
                "after_generator_close": after_abort,
                "after_generator_close_cache_hit": True,
            },
            "results": results,
            "warm_median_output_tokens_per_s": medians,
            "speedup_percent": speedups,
            "speedup_percent_vs_baseline": speedup,
        }
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        for name, result in results.items():
            perf.print_block(name, result)
            print(f"RATE {name} output_tokens_per_s={medians[name]:.6f}")
        for name, value in speedups.items():
            print(f"SPEEDUP {name} percent={value:.9f}")
        print(f"CACHE {json.dumps(measurement_stats, sort_keys=True)}")
        print(f"CORRECTNESS all {len(rows['baseline'])} samples per arm match; "
              f"{len(controls)} varied-prompt controls and post-close match")
        print(f"REPORT {args.out}")
        return 0
    finally:
        lock.close()


if __name__ == "__main__":
    sys.exit(main())

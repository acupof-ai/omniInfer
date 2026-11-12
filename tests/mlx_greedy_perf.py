#!/usr/bin/env python3
"""Interleaved direct A/B for the stage-owned greedy MLX generation path."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "python" / "umminfer_mlx_stage.py"
SPEC = importlib.util.spec_from_file_location(
    "umminfer_mlx_perf", ROOT / "tests" / "mlx_perf.py")
perf = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(perf)


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


def token_hash(tokens):
    return hashlib.sha256(json.dumps(tokens).encode("utf-8")).hexdigest()


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
    needles = perf.patrol_needles(perf.DEFAULT_BINARY, STAGE)
    perf.check_exclusive(needles)
    stage = perf.load_stage_module(STAGE)
    started = time.perf_counter()
    model, tokenizer = load(str(perf.MODEL_PATH))
    stage._cache_vocabulary(tokenizer)
    load_seconds = time.perf_counter() - started
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": perf.PROMPT}], tokenize=False,
        add_generation_prompt=True)

    def stock(prompt_, max_tokens):
        return stream_generate(
            model, tokenizer, prompt_, max_tokens=max_tokens,
            sampler=make_sampler(temp=perf.TEMPERATURE))

    no_logprobs = stage._make_greedy_generate(
        model, tokenizer, skip_final_lookahead=False)
    candidate = stage._make_greedy_generate(
        model, tokenizer, skip_final_lookahead=True)
    arms = (("stock", stock), ("no_logprobs", no_logprobs),
            ("candidate", candidate))
    rows = {name: [] for name, _ in arms}
    try:
        for iteration in range(args.warmup + args.samples):
            # Rotate the order so a fixed arm does not always run first.
            order = arms[iteration % len(arms):] + arms[:iteration % len(arms)]
            reference = None
            for name, generate in order:
                first, total, text, tokens, last = collect(
                    generate, prompt, args.max_tokens)
                row = perf.make_row(
                    iteration, args.warmup, first, total, last, text)
                row["cold"] = iteration == 0
                row["finish_reason"] = last.finish_reason
                row["token_ids_sha256"] = token_hash(tokens)
                rows[name].append(row)
            # A rotated order can run the candidate before stock. Compare again
            # after all three rows exist so every sample is checked.
            stock_row = rows["stock"][-1]
            expected = (stock_row["output_sha256"],
                        stock_row["token_ids_sha256"],
                        stock_row["generation_tokens"],
                        stock_row["finish_reason"], stock_row["prompt_tokens"])
            for name in ("no_logprobs", "candidate"):
                row = rows[name][-1]
                actual = (row["output_sha256"], row["token_ids_sha256"],
                          row["generation_tokens"], row["finish_reason"],
                          row["prompt_tokens"])
                perf.require(actual == expected,
                             f"{name} changed output at iteration {iteration}")
            perf.require(stock_row["generation_tokens"] > 0,
                         "the fixed request reported no generated tokens")

        results = {name: perf.block(values) for name, values in rows.items()}
        medians = {name: perf.median_of(value, "output_tokens_per_s")
                   for name, value in results.items()}
        speedups = {name: 100.0 * (rate / medians["stock"] - 1.0)
                    for name, rate in medians.items()}
        report = {
            "configuration": {
                "model": str(perf.MODEL_PATH),
                "model_safetensors_sha256": perf.digest(
                    perf.MODEL_PATH / "model.safetensors"),
                "prompt": perf.PROMPT, "temperature": perf.TEMPERATURE,
                "max_tokens": args.max_tokens,
                "samples": args.samples, "warmup": args.warmup,
                "interleaved": True, "loaded_model_shared": True,
            },
            "hashes": {
                "stage": perf.digest(STAGE),
                "script": perf.digest(Path(__file__).resolve()),
                "harness": perf.digest(Path(perf.__file__).resolve()),
            },
            "model_load_seconds": load_seconds,
            "correctness": {
                "per_sample_fields": ["token_ids_sha256", "output_sha256",
                                      "generation_tokens", "finish_reason",
                                      "prompt_tokens"],
                "samples_per_arm": len(rows["stock"]),
                "observed_generation_tokens": {
                    name: sorted({row["generation_tokens"] for row in values})
                    for name, values in rows.items()},
                "observed_finish_reasons": {
                    name: sorted({row["finish_reason"] for row in values})
                    for name, values in rows.items()},
                "all_equal_to_stock": True,
            },
            "results": results,
            "warm_median_output_tokens_per_s": medians,
            "speedup_percent_vs_stock": speedups,
        }
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        for name, result in results.items():
            perf.print_block(name, result)
            print(f"RATE {name} output_tokens_per_s={medians[name]:.6f} "
                  f"speedup_percent={speedups[name]:.6f}")
        print(f"CORRECTNESS all {len(rows['stock'])} samples per arm match stock")
        print(f"REPORT {args.out}")
        return 0
    finally:
        lock.close()


if __name__ == "__main__":
    sys.exit(main())

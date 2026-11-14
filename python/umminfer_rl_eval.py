# SPDX-License-Identifier: Apache-2.0
"""Greedy exact-match evaluation on held-out GSM8K problems.

Runs in its own process so evaluation never shares memory with serving or
training. With --weights it loads a checkpoint's model state over the base
model; without it the untouched base model is evaluated. Every answer is
written out, so a wrong claim cannot hide behind a single aggregate number.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

# This interpreter's -I mode omits the script directory from sys.path, so
# bootstrap it explicitly before importing sibling modules.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from umminfer_rl_gsm8k_data import build_messages
from umminfer_rl_gsm8k_reward import extract_answer


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    items = [json.loads(line) for line in
             open(args.data_path, encoding="utf-8") if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=getattr(torch, args.dtype)
    ).to("cuda").eval()
    weights_sha256 = None
    if args.weights:
        with open(args.weights, "rb") as handle:
            weights_sha256 = hashlib.sha256(handle.read()).hexdigest()
        state = torch.load(args.weights, map_location="cuda",
                           weights_only=True)
        model.load_state_dict(state["model"])

    rows = []
    correct = 0
    scored = 0
    with torch.no_grad():
        for item in items:
            messages = build_messages(item["question"])
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False)
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            input_ids = input_ids.to("cuda")
            output = model.generate(
                input_ids, do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(output[0, input_ids.shape[1]:],
                                    skip_special_tokens=True)
            predicted = extract_answer(text)
            is_correct = predicted == item["gold_answer"]
            if predicted is not None:
                scored += 1
                correct += int(is_correct)
            rows.append({"question": item["question"],
                         "gold_answer": item["gold_answer"],
                         "predicted": predicted, "correct": is_correct,
                         "output_text": text})

    result = {
        "tag": args.tag,
        "model_path": args.model_path,
        "weights": args.weights,
        "weights_sha256": weights_sha256,
        "total": len(items),
        "scored": scored,
        "correct": correct,
        "exact_match": correct / len(items),
        "rows": rows,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, sort_keys=True, indent=2,
                  ensure_ascii=False, allow_nan=False)
    print(json.dumps({"tag": args.tag, "exact_match": result["exact_match"],
                      "correct": correct, "total": len(items)}))


if __name__ == "__main__":
    main()

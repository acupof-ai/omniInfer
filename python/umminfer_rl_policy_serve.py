# SPDX-License-Identifier: Apache-2.0
"""Small-model policy server for the RL architecture validation.

A real Hugging Face causal LM served over a loopback OpenAI-compatible
``/v1/chat/completions`` SSE exit, so the phase-1 recorder drives it exactly
like any other serving backend. This process is a plain client of the model
weights and never imports optimizer code; training stays in the worker.

The server is single-threaded on purpose: one loopback orchestrator, one
request at a time, which makes the seeded sampling order reproducible.
"""

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


class Policy:
    def __init__(self, model_path, seed, dtype, weights_path=None):
        # Lazy imports: the protocol layer (make_handler/main wiring) stays
        # importable on hosts without torch, where the deterministic tests run.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        torch_dtype = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_dtype).to("cuda").eval()
        if weights_path:
            # Serve the weights a training checkpoint produced, so rollouts
            # after step N come from the post-step-N policy (on-policy).
            state = torch.load(weights_path, map_location="cuda",
                               weights_only=True)
            self.model.load_state_dict(state["model"])
        self.seed = seed
        self.counter = 0

    def generate(self, messages, *, temperature, top_p, max_new_tokens):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        prompt_ids = prompt_ids.to("cuda")
        # transformers 4.51 generate() does not accept a Generator for this
        # model type. Reseed the global CUDA RNG per call from the base seed
        # and the monotonic counter, so the sampling sequence stays
        # reproducible in single-threaded request order.
        self.torch.manual_seed(self.seed + self.counter)
        with self.torch.no_grad():
            output = self.model.generate(
                prompt_ids,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        completion_ids = output[0, prompt_ids.shape[1]:]
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        self.counter += 1
        request_id = f"rl-rollout-{self.counter:06d}"
        return request_id, text, prompt_ids[0].tolist(), completion_ids.tolist()


def make_handler(policy):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == "/health":
                # PID lets the runner prove it talks to this child, not an
                # orphaned server from an earlier run on a reused port.
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "ok", "pid": os.getpid()}).encode())
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_response(404)
                self.end_headers()
                return
            body = json.loads(self.rfile.read(
                int(self.headers["Content-Length"])))
            params = body.get("request_params_override") or {}
            request_id, text, _, _ = policy.generate(
                body["messages"],
                temperature=float(params.get("temperature", 0.0)),
                top_p=float(params.get("top_p", 1.0)),
                max_new_tokens=int(params.get("max_tokens", 512)))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunk = {
                "id": request_id, "object": "chat.completion.chunk",
                "choices": [{"index": 0,
                             "delta": {"content": text},
                             "finish_reason": None}]}
            self.wfile.write(
                ("data: " + json.dumps(chunk) + "\n\n").encode())
            final = {
                "id": request_id, "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {},
                             "finish_reason": "stop"}]}
            self.wfile.write(
                ("data: " + json.dumps(final) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description="RL validation policy server")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20261114)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--weights",
                        help="optional training checkpoint weights to serve")
    args = parser.parse_args(argv)
    policy = Policy(args.model_path, args.seed, args.dtype,
                    weights_path=args.weights)
    server = HTTPServer((args.host, args.port), make_handler(policy))
    print(json.dumps({"type": "ready", "port": args.port,
                      "seed": args.seed}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

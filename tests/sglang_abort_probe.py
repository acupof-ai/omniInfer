#!/usr/bin/env python3
"""Prove SGLang per-request abort on /v1/chat/completions.

Sends one long streaming chat request with a caller rid, aborts it with
POST /abort_request, and records every SSE frame until finish_reason "abort".
The output JSON is evidence: the abort status, the frames, and whether the
backend confirmed the abort on the generate stream.
"""

import argparse
import http.client
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="qwen3-omni")
    parser.add_argument("--rid", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-tokens", type=int, default=600)
    args = parser.parse_args()
    # SGLang keeps rids across runs; a fixed rid would be a duplicate.
    if not args.rid:
        args.rid = f"umm-phase5-probe-{os.getpid()}-{int(time.time())}"
    host, port = args.base_url.split("//", 1)[1].split(":", 1)
    connection = http.client.HTTPConnection(host, int(port), 120)
    payload = {"model": args.model, "rid": args.rid, "stream": True,
               "max_tokens": args.max_tokens, "temperature": 0,
               "messages": [{"role": "user",
                             "content": "Count from 1 to 600, one per line."}]}
    started = time.monotonic()
    connection.request("POST", "/v1/chat/completions",
                       body=json.dumps(payload),
                       headers={"Content-Type": "application/json"})
    response = connection.getresponse()
    frames = []
    abort_status = None
    data_lines = []
    for raw in response:
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if line:
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
            continue
        if not data_lines:
            continue
        data = "\n".join(data_lines)
        data_lines = []
        if data == "[DONE]":
            break
        event = json.loads(data)
        choice = (event.get("choices") or [{}])[0]
        frames.append({"t": round(time.monotonic() - started, 3),
                       "content": (choice.get("delta") or {}).get("content"),
                       "finish_reason": choice.get("finish_reason")})
        if abort_status is None and len(frames) >= 5:
            abort = http.client.HTTPConnection(host, int(port), 10)
            abort.request("POST", "/abort_request",
                          body=json.dumps({"rid": args.rid}),
                          headers={"Content-Type": "application/json"})
            reply = abort.getresponse()
            reply.read(4096)
            abort_status = reply.status
            abort.close()
        if choice.get("finish_reason") is not None:
            break
    connection.close()
    confirmed = any(frame["finish_reason"] == "abort" for frame in frames)
    result = {"rid": args.rid, "abort_status": abort_status,
              "confirmed": confirmed, "frames": frames}
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(f"ABORT_STATUS={abort_status} CONFIRMED={confirmed} "
          f"FRAMES={len(frames)}")
    return 0 if confirmed and abort_status == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())

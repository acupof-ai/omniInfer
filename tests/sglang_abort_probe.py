#!/usr/bin/env python3
"""Prove SGLang per-request abort on /v1/chat/completions.

Sends one long streaming chat request with a caller rid, aborts it with
POST /abort_request, and records every SSE frame until finish_reason "abort".
It also samples the backend's own request counter (/v1/loads) before the
request, while it is decoding, and after the abort. That counter is the
scheduler's view of the running batch, not a GPU hardware counter: the probe
records it as request-lifecycle evidence and does not claim more.

The output JSON is evidence: the abort status, the frames, the counter samples
and whether the backend confirmed the abort on the generate stream.
"""

import argparse
import http.client
import json
import os
from pathlib import Path
import time
import urllib.request


def get_loads(url, timeout=5.0):
    with urllib.request.urlopen(url, timeout=timeout) as reply:
        body = json.load(reply)
    loads = body.get("loads") or [{}]
    return {
        "at": round(time.time(), 3),
        "total_running_reqs": body.get("aggregate", {}).get(
            "total_running_reqs", loads[0].get("num_running_reqs")),
        "num_running_reqs": loads[0].get("num_running_reqs"),
        "num_waiting_reqs": loads[0].get("num_waiting_reqs"),
        "gen_throughput": loads[0].get("gen_throughput"),
        "num_used_tokens": loads[0].get("num_used_tokens"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="qwen3-omni")
    parser.add_argument("--rid", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--loads-url", default=None,
                        help="defaults to <origin>/v1/loads")
    args = parser.parse_args()
    # SGLang keeps rids across runs; a fixed rid would be a duplicate.
    if not args.rid:
        args.rid = f"umm-phase5-probe-{os.getpid()}-{int(time.time())}"
    host, port = args.base_url.split("//", 1)[1].split(":", 1)
    origin = args.base_url.rstrip("/")
    if origin.endswith("/v1"):
        origin = origin[:-3]
    loads_url = args.loads_url or origin + "/v1/loads"
    connection = http.client.HTTPConnection(host, int(port), 120)
    payload = {"model": args.model, "rid": args.rid, "stream": True,
               "max_tokens": args.max_tokens, "temperature": 0,
               "messages": [{"role": "user",
                             "content": "Count from 1 to 600, one per line."}]}
    loads = []
    try:
        loads.append(dict(get_loads(loads_url), phase="before"))
    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
        loads.append({"phase": "before", "error": str(exc)})
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
            try:
                loads.append(dict(get_loads(loads_url), phase="during"))
            except Exception as exc:  # noqa: BLE001 - recorded, not hidden
                loads.append({"phase": "during", "error": str(exc)})
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
    # The listener removes the rid before the abort frame reaches the client,
    # so the counter is polled until it stops reporting our request.
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        try:
            sample = dict(get_loads(loads_url), phase="after")
        except Exception as exc:  # noqa: BLE001 - recorded, not hidden
            sample = {"phase": "after", "error": str(exc)}
        loads.append(sample)
        if sample.get("num_running_reqs") == 0:
            break
        time.sleep(0.2)
    confirmed = any(frame["finish_reason"] == "abort" for frame in frames)
    during = [item for item in loads if item.get("phase") == "during"]
    after = [item for item in loads if item.get("phase") == "after"]
    before = [item for item in loads if item.get("phase") == "before"]
    running_during = during[0].get("num_running_reqs") if during else None
    running_after = after[-1].get("num_running_reqs") if after else None
    running_before = before[0].get("num_running_reqs") if before else None
    decode_stopped = (
        confirmed and running_during is not None and running_during >= 1 and
        running_after is not None and running_after <= (running_before or 0))
    result = {"rid": args.rid, "abort_status": abort_status,
              "confirmed": confirmed, "frames": frames, "loads": loads,
              "running_before": running_before,
              "running_during": running_during,
              "running_after": running_after,
              "counter_decode_stopped": decode_stopped,
              "counter_scope": "SGLang scheduler /v1/loads, not GPU hardware counters"}
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(f"ABORT_STATUS={abort_status} CONFIRMED={confirmed} "
          f"FRAMES={len(frames)} RUNNING={running_before}->{running_during}"
          f"->{running_after} DECODE_STOPPED={decode_stopped}")
    return 0 if confirmed and abort_status == 200 and decode_stopped else 1


if __name__ == "__main__":
    raise SystemExit(main())

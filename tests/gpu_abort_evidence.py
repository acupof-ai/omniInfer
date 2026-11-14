#!/usr/bin/env python3
"""Hardware-level decode-stop evidence around an SGLang request abort.

Four subcommands, standard library only except ``calibrate`` (torch):

* ``probe``          - one streaming chat request, an ``/abort_request`` after
  a number of frames, and wall-clock/monotonic markers for every phase. It also
  samples ``/v1/loads``; that scheduler counter is request-lifecycle evidence,
  not a hardware counter.
* ``calibrate``      - measures the per-boot offset between
  ``time.monotonic_ns`` and a torch.profiler (Kineto/CUPTI) Chrome-trace
  timestamp, so probe markers can be placed on the GPU timeline of a trace
  recorded by another process.
* ``chrome-analyze`` - joins a SGLang ``/start_profile`` Chrome trace with the
  probe markers: decode-step spans and CUDA kernel counts/GPU-busy time in the
  decode window, and a kernel-free idle window after the abort frame.
* ``dcgm-analyze``   - joins ``dcgmi dmon`` rows (one wall-clock-prefixed CSV
  row per sample) with the probe markers and summarizes SM activity per phase.

The Chrome trace is the SGLang torch.profiler output. Kineto collects CUDA
kernel activity through the CUPTI Activity API inside the scheduler process,
so no external tracer injection is involved.
"""

import argparse
import csv
import gzip
import http.client
import json
import os
from pathlib import Path
import time
import urllib.request


# --------------------------------------------------------------------------
# probe
# --------------------------------------------------------------------------

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
    }


def cmd_probe(args):
    host, port = args.base_url.split("//", 1)[1].split(":", 1)
    port = int(port)
    origin = args.base_url.rstrip("/")
    if origin.endswith("/v1"):
        origin = origin[:-3]
    loads_url = origin + "/v1/loads"
    rid = args.rid or f"umm-phase6-gpu-{os.getpid()}-{int(time.time())}"
    payload = {"model": args.model, "rid": rid, "stream": True,
               "max_tokens": args.max_tokens, "temperature": 0,
               "messages": [{"role": "user",
                             "content": f"Count from 1 to {args.max_tokens}, "
                                        "one per line."}]}
    markers = []

    def mark(name, **extra):
        event = {"name": name, "epoch": round(time.time(), 6),
                 "monotonic": round(time.monotonic(), 6)}
        event.update(extra)
        markers.append(event)

    def sample_loads(phase):
        try:
            sample = dict(get_loads(loads_url), phase=phase)
        except Exception as exc:  # noqa: BLE001 - recorded, not hidden
            sample = {"phase": phase, "error": str(exc)}
        markers.append({"name": f"loads-{phase}",
                        "epoch": round(time.time(), 6),
                        "monotonic": round(time.monotonic(), 6),
                        "sample": sample})
        return sample

    result = {"rid": rid, "markers": markers,
              "request": payload, "frames": [], "loads": []}

    sample_loads("before")
    connection = http.client.HTTPConnection(host, port, 120)
    started_mono = time.monotonic()
    mark("umm-marker-before-baseline")
    time.sleep(args.baseline_s)
    mark("umm-marker-before-request")
    connection.request("POST", "/v1/chat/completions",
                       body=json.dumps(payload),
                       headers={"Content-Type": "application/json"})
    response = connection.getresponse()
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
        frame = {"t": round(time.monotonic() - started_mono, 3),
                 "epoch": round(time.time(), 6),
                 "content": (choice.get("delta") or {}).get("content"),
                 "finish_reason": choice.get("finish_reason")}
        result["frames"].append(frame)
        if abort_status is None and len(result["frames"]) >= args.abort_after:
            result["loads"].append(sample_loads("during"))
            mark("umm-marker-abort-post")
            abort_conn = http.client.HTTPConnection(host, port, 10)
            abort_conn.request("POST", "/abort_request",
                               body=json.dumps({"rid": rid}),
                               headers={"Content-Type": "application/json"})
            abort_reply = abort_conn.getresponse()
            abort_reply.read(4096)
            abort_status = abort_reply.status
            abort_conn.close()
            mark("umm-marker-abort-response", status=abort_status)
        if choice.get("finish_reason") is not None:
            if choice.get("finish_reason") == "abort":
                mark("umm-marker-abort-frame")
            break
    connection.close()

    confirmed = any(f["finish_reason"] == "abort" for f in result["frames"])
    # Idle GPU window after the abort frame; the scheduler has removed rid.
    deadline = time.monotonic() + args.post_s
    while time.monotonic() < deadline:
        try:
            result["loads"].append(dict(get_loads(loads_url), phase="after"))
        except Exception as exc:  # noqa: BLE001
            result["loads"].append({"phase": "after", "error": str(exc)})
        time.sleep(0.2)
    mark("umm-marker-post-end")

    after = [s for s in result["loads"] if s.get("phase") == "after"]
    result.update({
        "abort_status": abort_status,
        "confirmed": confirmed,
        "frame_count": len(result["frames"]),
        "running_after": after[-1].get("num_running_reqs") if after else None,
        "counter_scope": "SGLang scheduler /v1/loads, not GPU hardware counters",
    })
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(f"ABORT_STATUS={abort_status} CONFIRMED={confirmed} "
          f"FRAMES={len(result['frames'])}")
    return 0 if confirmed and abort_status == 200 else 1


# --------------------------------------------------------------------------
# clock calibration (torch only; run inside the SGLang container)
# --------------------------------------------------------------------------

def cmd_calibrate(args):
    import torch
    from torch.profiler import profile, ProfilerActivity, record_function

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) \
            as prof:
        mono_in = time.monotonic_ns()
        with record_function("umm-clock-calibration"):
            torch.zeros(1, device="cuda")
            torch.cuda.synchronize()
        mono_out = time.monotonic_ns()
    prof.export_chrome_trace(args.trace)
    events = json.loads(Path(args.trace).read_text())["traceEvents"]
    marker = next(e for e in events
                  if e.get("name") == "umm-clock-calibration")
    # Kineto Chrome timestamps share CLOCK_MONOTONIC with a constant offset.
    offset_in = mono_in - int(marker["ts"] * 1000)
    offset_out = mono_out - int((marker["ts"] + marker["dur"]) * 1000)
    offset_ns = (offset_in + offset_out) // 2
    out = {
        "clock": "CLOCK_MONOTONIC",
        "rule": "monotonic_ns = chrome_trace_ts_us * 1000 + clock_offset_ns",
        "clock_offset_ns": offset_ns,
        "bracket_ns": [offset_in, offset_out],
        "calibration_epoch": round(time.time(), 6),
        "calibration_monotonic": round(time.monotonic(), 6),
        "trace": os.path.basename(args.trace),
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"CLOCK_OFFSET_NS={offset_ns}")
    return 0


# --------------------------------------------------------------------------
# SGLang torch.profiler Chrome trace analysis (CUPTI kernel activity)
# --------------------------------------------------------------------------

def _load_chrome(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        return json.load(handle)


def cmd_chrome_analyze(args):
    probe = json.loads(Path(args.probe_json).read_text())
    calibration = json.loads(Path(args.calibration).read_text())
    offset_ns = calibration["clock_offset_ns"]
    trace = _load_chrome(args.trace)
    events = trace["traceEvents"]

    def to_trace_us(monotonic_s):
        return (monotonic_s * 1e9 - offset_ns) / 1000.0

    marker = {m["name"]: m["monotonic"] for m in probe["markers"]}
    t_request = to_trace_us(marker["umm-marker-before-request"])
    t_abort = to_trace_us(marker["umm-marker-abort-post"])
    t_frame = to_trace_us(marker["umm-marker-abort-frame"])
    t_end = to_trace_us(marker["umm-marker-post-end"])
    # The abort frame follows the last in-flight decode step. Anything in the
    # first 0.2 s after it is bounded teardown, not a sustained decode stream.
    idle_start = t_frame + args.teardown_us

    kernels = [e for e in events
               if e.get("cat") == "kernel" and e.get("ph") == "X"]
    steps = [e for e in events
             if e.get("name", "").startswith("step[") and e.get("ph") == "X"]
    decode_steps = [s for s in steps if "DECODE" in s["name"]]

    def kernel_stats(lo, hi):
        inside = [k for k in kernels if lo <= k["ts"] < hi]
        return {"kernels": len(inside),
                "gpu_active_us": round(sum(k.get("dur", 0) for k in inside), 3)}

    def step_stats(lo, hi):
        inside = [s for s in steps if lo <= s["ts"] < hi]
        return {"steps": len(inside),
                "decode_steps": sum(1 for s in inside if "DECODE" in s["name"]),
                "extend_steps": sum(1 for s in inside if "EXTEND" in s["name"])}

    windows = [
        dict(window="decode", **step_stats(t_request, t_abort),
             **kernel_stats(t_request, t_abort)),
        dict(window="teardown", **step_stats(t_abort, idle_start),
             **kernel_stats(t_abort, idle_start)),
        dict(window="post-abort-idle", **step_stats(idle_start, t_end),
             **kernel_stats(idle_start, t_end)),
    ]
    idle = windows[-1]
    decode = windows[0]
    idle_span_s = max((t_end - idle_start) / 1e6, 1e-9)
    teardown = windows[1]
    sample_names = sorted({k["name"] for k in kernels
                           if t_request <= k["ts"] < t_abort})[:10]
    idle_names = sorted({k["name"] for k in kernels
                         if idle_start <= k["ts"] < t_end})
    stopped = (decode["decode_steps"] >= 3 and decode["kernels"] >= 100 and
               idle["kernels"] == 0 and idle_span_s >= 2.0)
    result = {
        "rid": probe["rid"],
        "abort_status": probe["abort_status"],
        "sse_confirmed": probe["confirmed"],
        "cupti_version": trace.get("cupti_version"),
        "clock_offset_ns": offset_ns,
        "trace_window_us": {"request": round(t_request, 1),
                            "abort_post": round(t_abort, 1),
                            "abort_frame": round(t_frame, 1),
                            "post_end": round(t_end, 1),
                            "idle_start": round(idle_start, 1)},
        "windows": windows,
        "idle_window_s": round(idle_span_s, 3),
        "teardown_gpu_active_us": teardown["gpu_active_us"],
        "sample_decode_kernel_names": sample_names,
        "idle_window_kernel_names": idle_names,
        "hardware_decode_stopped": bool(stopped),
        "verdict_rule": ">=3 decode steps and >=100 CUDA kernels before the "
                        "abort; zero CUDA kernels for >=2 s after the abort "
                        "frame plus a 0.2 s teardown grace window",
        "evidence_scope": "CUPTI Activity API kernel trace via SGLang "
                          "torch.profiler (Kineto) on the dedicated GPU2 "
                          "scheduler process",
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(f"HARDWARE_DECODE_STOPPED={stopped} "
          f"decode_steps={decode['decode_steps']} "
          f"decode_kernels={decode['kernels']} idle_kernels={idle['kernels']} "
          f"idle_s={round(idle_span_s, 2)}")
    return 0 if stopped else 1


# --------------------------------------------------------------------------
# DCGM analysis
# --------------------------------------------------------------------------

def cmd_dcgm_analyze(args):
    probe = json.loads(Path(args.probe_json).read_text())
    marker_epoch = {m["name"]: m["epoch"] for m in probe["markers"]}
    t_request = marker_epoch["umm-marker-before-request"]
    t_abort = marker_epoch["umm-marker-abort-post"]
    t_end = marker_epoch["umm-marker-post-end"]
    # A DCGM sample is a 100-200 ms duty-cycle average. The first samples after
    # the abort POST straddle in-flight teardown kernels and are reported as a
    # separate "teardown" phase, not the idle post-abort window.
    grace_s = 0.5
    phases = {"baseline": [], "decode": [], "teardown": [], "post-abort": []}
    with open(args.dcgmi_csv, newline="") as handle:
        for row in csv.DictReader(handle):
            if "epoch" not in row:
                continue
            try:
                epoch = float(row["epoch"])
            except (TypeError, ValueError):
                continue
            if epoch < t_request:
                phase = "baseline"
            elif epoch < t_abort:
                phase = "decode"
            elif epoch < t_abort + grace_s:
                phase = "teardown"
            elif epoch <= t_end + 5:
                phase = "post-abort"
            else:
                continue
            phases[phase].append(row)

    def numeric(rows, column):
        out = []
        for row in rows:
            try:
                out.append(float(row[column]))
            except (KeyError, TypeError, ValueError):
                pass
        return out

    summary = {}
    for phase, rows in phases.items():
        smact = numeric(rows, "SMACT")
        gract = numeric(rows, "GRACT")
        summary[phase] = {
            "samples": len(rows),
            "sm_active_min": min(smact) if smact else None,
            "sm_active_mean": (round(sum(smact) / len(smact), 4)
                               if smact else None),
            "sm_active_max": max(smact) if smact else None,
            "graphics_active_mean": (round(sum(gract) / len(gract), 4)
                                     if gract else None),
            "na_samples": sum(1 for row in rows
                              if row.get("SMACT") in ("N/A", "", None)),
        }
    decode_max = summary["decode"]["sm_active_max"]
    post = summary["post-abort"]
    post_max = post["sm_active_max"]
    post_mean = post["sm_active_mean"]
    decode_samples = summary["decode"]["samples"]
    post_samples = post["samples"]
    # Decode is a short burst of per-token kernels, not a sustained 100% duty
    # cycle, so the bar is real activity versus a near-idle post window.
    busy = decode_max is not None and decode_max >= 0.02 and decode_samples >= 1
    stopped = (post_samples >= 5 and post_mean is not None and
               post_mean <= 0.005 and
               (post_max is None or
                (post_max <= 0.02 and post_max <= 0.5 * decode_max)))
    verdict = bool(busy and stopped)
    result = {"rid": probe["rid"],
              "phase_epoch_seconds": {
                  "request": t_request, "abort_post": t_abort,
                  "post_end": t_end},
              "dcgm_fields": "1001 gr_engine_active, 1002 sm_active, "
                             "1004 tensor_active, 1005 dram_active",
              "summary": summary,
              "hardware_decode_stopped": verdict,
              "verdict_rule": "decode max sm_active >= 0.02 with >=1 sample; "
                              "post-abort (after a 0.5 s teardown grace window) "
                              ">=5 samples with mean sm_active <= 0.005, max "
                              "<= 0.02 and <= 50% of the decode max "
                              "(N/A counts as idle)",
              "evidence_scope": "DCGM hardware profiling counters on GPU2"}
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(f"DCGM_DECODE_STOPPED={verdict} decode_max={decode_max} "
          f"post_max={post_max} decode_samples={decode_samples} "
          f"post_samples={post_samples}")
    return 0 if verdict else 1


# --------------------------------------------------------------------------
# stdlib self-check: synthetic traces and DCGM rows, no torch needed
# --------------------------------------------------------------------------

def cmd_selfcheck(args):
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="gpu-evidence-selfcheck-"))
    offset_ns = 1_000_000_000  # 1 ms, arbitrary but internally consistent

    def trace_us(mono_s):
        return (mono_s * 1e9 - offset_ns) / 1000.0

    def write_probe(path, markers, confirmed=True):
        path.write_text(json.dumps({
            "rid": "selfcheck-rid", "abort_status": 200,
            "confirmed": confirmed,
            "markers": [{"name": n, "monotonic": m, "epoch": m}
                        for n, m in markers],
            "frames": [], "loads": [], "request": {}}) + "\n")

    def write_calibration(path):
        path.write_text(json.dumps({"clock_offset_ns": offset_ns}) + "\n")

    def write_trace(path, decode_kernels, idle_kernels, decode_steps):
        events = []
        base = 100.0  # trace ts in us
        for i in range(decode_steps):
            events.append({"name": "step[DECODE bs=1]", "ph": "X", "cat":
                           "user_annotation", "ts": base + i * 1000,
                           "dur": 50})
        for i in range(decode_kernels):
            events.append({"name": "model-kernel", "ph": "X", "cat":
                           "kernel", "ts": base + 200 + i, "dur": 1})
        for i in range(idle_kernels):
            events.append({"name": "rogue-kernel", "ph": "X", "cat":
                           "kernel", "ts": base + 3_000_000 + i, "dur": 1})
        path.write_text(json.dumps({"traceEvents": events,
                                    "cupti_version": "selfcheck"}))

    # Markers in monotonic seconds; trace ts = (mono*1e9 - offset)/1e3.
    markers = [
        ("umm-marker-before-request", 1.0),
        ("umm-marker-abort-post", 2.0),
        ("umm-marker-abort-frame", 2.1),
        ("umm-marker-post-end", 6.0),
    ]
    probe_path = tmp / "probe.json"
    cal_path = tmp / "cal.json"
    trace_path = tmp / "trace.json"
    out_path = tmp / "out.json"
    write_probe(probe_path, markers)
    write_calibration(cal_path)

    # PASS case: many decode kernels/steps, zero idle kernels.
    write_trace(trace_path, decode_kernels=200, idle_kernels=0,
                decode_steps=5)
    ns = argparse.Namespace(trace=str(trace_path), probe_json=str(probe_path),
                            calibration=str(cal_path), out=str(out_path),
                            teardown_us=200_000)
    assert cmd_chrome_analyze(ns) == 0, "PASS case must pass"
    assert json.loads(out_path.read_text())["hardware_decode_stopped"] is True

    # FAIL case: kernels leak into the idle window.
    write_trace(trace_path, decode_kernels=200, idle_kernels=3,
                decode_steps=5)
    assert cmd_chrome_analyze(ns) == 1, "idle-window kernels must fail"

    # FAIL case: too little decode activity.
    write_trace(trace_path, decode_kernels=10, idle_kernels=0,
                decode_steps=1)
    assert cmd_chrome_analyze(ns) == 1, "weak decode must fail"

    # DCGM PASS/FAIL.
    t_request, t_abort, t_end = 1.0, 2.0, 6.0
    rows = ["epoch,GRACT,SMACT,TENSO,DRAMA"]
    for i in range(10):  # baseline idle
        rows.append(f"{0.5 + i * 0.05:.3f},0.0,0.0,0.0,0.0")
    for i in range(5):  # decode busy
        rows.append(f"{1.2 + i * 0.1:.3f},0.1,0.3,0.01,0.02")
    rows.append("2.2,0.05,0.04,0.0,0.01")  # teardown grace
    for i in range(20):  # post-abort idle
        rows.append(f"{2.6 + i * 0.1:.3f},0.0,0.0,0.0,0.0")
    dmon = tmp / "dmon.csv"
    dmon.write_text("\n".join(rows) + "\n")
    dprobe = tmp / "dprobe.json"
    dprobe.write_text(json.dumps({"rid": "selfcheck-rid", "markers": [
        {"name": "umm-marker-before-request", "epoch": t_request},
        {"name": "umm-marker-abort-post", "epoch": t_abort},
        {"name": "umm-marker-post-end", "epoch": t_end}]}) + "\n")
    dout = tmp / "dout.json"
    dn = argparse.Namespace(dcgmi_csv=str(dmon), probe_json=str(dprobe),
                            out=str(dout))
    assert cmd_dcgm_analyze(dn) == 0, "DCGM PASS case must pass"

    # One post-abort sample stays busy -> must fail.
    rows.append("3.0,0.2,0.2,0.01,0.01")
    dmon.write_text("\n".join(rows) + "\n")
    assert cmd_dcgm_analyze(dn) == 1, "busy post-abort sample must fail"
    print("selfcheck: all chrome-trace and DCGM verdict cases pass")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe")
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", default="qwen3-omni")
    p.add_argument("--rid", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--max-tokens", type=int, default=600)
    p.add_argument("--abort-after", type=int, default=12)
    p.add_argument("--baseline-s", type=float, default=2.0)
    p.add_argument("--post-s", type=float, default=3.0)
    p.set_defaults(func=cmd_probe)

    c = sub.add_parser("calibrate")
    c.add_argument("--out", required=True)
    c.add_argument("--trace", required=True)
    c.set_defaults(func=cmd_calibrate)

    a = sub.add_parser("chrome-analyze")
    a.add_argument("--trace", required=True)
    a.add_argument("--probe-json", required=True)
    a.add_argument("--calibration", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--teardown-us", type=float, default=200_000)
    a.set_defaults(func=cmd_chrome_analyze)

    s = sub.add_parser("selfcheck")
    s.set_defaults(func=cmd_selfcheck)

    d = sub.add_parser("dcgm-analyze")
    d.add_argument("--dcgmi-csv", required=True)
    d.add_argument("--probe-json", required=True)
    d.add_argument("--out", required=True)
    d.set_defaults(func=cmd_dcgm_analyze)

    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()

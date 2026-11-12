#!/usr/bin/env python3
"""Opt-in live E2E for an SGLang MiniMax-H3 video upstream through umm-infer.

Chain: frozen umm-infer CLI -> local umm-infer serve (kind=sglang,
variant=video) -> the single configured upstream -> the real model.

Without UMMINFER_LIVE_OPT_IN=1 and every required variable, the script prints
BLOCKED/NOT_COVERED and makes no network request.
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import live_common as live  # noqa: E402

REQUIRED = [
    "UMMINFER_BIN",
    "UMMINFER_H3_BASE_URL",
    "UMMINFER_H3_MODEL",
]
OPTIONAL = ["UMMINFER_H3_API_KEY"]

# Non-terminal statuses polled until the job settles. Every loop is bounded.
PENDING = {"queued", "pending", "running", "processing", "in_progress", "starting"}
TERMINAL_OK = {"succeeded", "success", "completed", "complete", "done"}


def profile():
    """Sampling profile. mac is the low-consumption default; h20b is explicit."""
    name = os.environ.get("UMMINFER_H3_PROFILE", "mac")
    if name == "mac":
        steps = os.environ.get("UMMINFER_H3_MAC_STEPS", "20")
        frames = os.environ.get("UMMINFER_H3_MAC_FRAMES", "49")
        return name, [f"num_inference_steps={steps}", f"num_frames={frames}"]
    if name == "h20b":
        # High-precision documented fields. No quantized or reduced-quality
        # override is sent in either profile.
        return name, ["num_inference_steps=50", "flow_shift=12.0"]
    raise live.LiveFail(f"unknown UMMINFER_H3_PROFILE: {name}")


def base_requires_key(base_url):
    return not (base_url.startswith("http://127.0.0.1")
                or base_url.startswith("http://localhost"))


def wait_job(binary, url, job_id, env, work):
    maximum = int(os.environ.get("UMMINFER_STATUS_POLL_MAX", "120"))
    interval = float(os.environ.get("UMMINFER_STATUS_POLL_S", "5"))
    last = None
    for _ in range(maximum):
        result = live.run_cli(binary, [
            "video", "status", "--url", url, "--id", job_id,
        ], env, work, timeout_s=120)
        live.require(result.returncode == 0,
                     "video status failed: " + result.stderr[-2000:])
        status = live.json_stdout(result)
        last = status
        state = str(status.get("status", "")).lower()
        if state in TERMINAL_OK:
            return status
        live.require(state not in {"failed", "error", "cancelled", "canceled"},
                     f"upstream job failed: {status}")
        live.require(state in PENDING, f"unknown job status: {status}")
        time.sleep(interval)
    raise live.LiveFail(
        f"job {job_id} did not settle in {maximum} polls; last={last}")


def run():
    binary = live.binary_path(os.environ["UMMINFER_BIN"])
    base_url = os.environ["UMMINFER_H3_BASE_URL"]
    model = os.environ["UMMINFER_H3_MODEL"]
    key = os.environ.get("UMMINFER_H3_API_KEY", "")
    if base_requires_key(base_url) and not key:
        raise live.LiveBlocked(
            "UMMINFER_H3_API_KEY is required for a non-loopback upstream")
    served = os.environ.get("UMMINFER_SERVED_MODEL", "omni")
    profile_name, fields = profile()
    variants = int(os.environ.get("UMMINFER_H3_VARIANTS", "2"))
    live.require(variants in (1, 2), "UMMINFER_H3_VARIANTS must be 1 or 2")

    serve_env = dict(os.environ)
    serve_env["UMMINFER_UPSTREAM_API_KEY"] = key
    serve_args = [
        "--host", "127.0.0.1", "--port", "0", "--model", served,
        "--upstream-kind", "sglang",
        "--upstream-base-url", base_url,
        "--upstream-model", model,
        "--upstream-variant", "video",
    ]
    env = live.cli_env()
    passes, gaps = [], []
    keep = "--keep" in sys.argv[1:]
    work = tempfile.mkdtemp(prefix="umm-live-h3-")

    try:
        with live.Serve(binary, serve_args, serve_env, work, "h3") as server:
            url = f"http://127.0.0.1:{server.port}/v1"
            create = [
                "video", "create", "--url", url, "--model", served,
                "--prompt", "A calm ocean at sunrise, slow camera push.",
                "--timeout-ms", os.environ.get("UMMINFER_H3_CREATE_MS", "600000"),
            ]
            for field in fields:
                create += ["--field", field]
            if variants > 1:
                create += ["--field", f"n={variants}"]
            created = live.run_cli(binary, create, env, work, timeout_s=660)
            live.require(created.returncode == 0,
                         "video create failed: " + created.stderr[-2000:])
            body = live.json_stdout(created)
            job_id = str(body.get("id") or body.get("video_id") or "")
            live.require(job_id != "", f"create returned no job id: {body}")
            passes.append(
                f"PASS h3 create: job {job_id} status={body.get('status')} "
                f"profile={profile_name}")

            try:
                status = wait_job(binary, url, job_id, env, work)
                passes.append(f"PASS h3 status: settled status={status.get('status')}")

                downloads = 0
                for index in range(variants):
                    out = os.path.join(work, f"variant-{index}.mp4")
                    got = live.run_cli(binary, [
                        "video", "content", "--url", url, "--id", job_id,
                        "--out", out, "--variant", str(index),
                        "--timeout-ms", os.environ.get("UMMINFER_H3_CONTENT_MS", "600000"),
                    ], env, work, timeout_s=660)
                    if got.returncode != 0 and index > 0:
                        # The real request was made. Only a 404 proves that this
                        # output index does not exist; any other failure stands.
                        live.require(
                            live.http_status_failure(got, 404),
                            f"video content variant {index} failed for a reason "
                            "other than a missing output: " + got.stderr[-2000:])
                        gaps.append(
                            f"NOT_COVERED h3 variant {index}: the real request "
                            "returned 404, so the job has no second artifact")
                        continue
                    live.require(got.returncode == 0,
                                 f"video content variant {index} failed: "
                                 + got.stderr[-2000:])
                    digest, size = live.check_mp4(out)
                    passes.append(
                        f"PASS h3 content variant {index}: mp4 {size} bytes "
                        f"sha256={digest}")
                    downloads += 1
                    live.expect_existing_output_refused(
                        binary,
                        ["video", "content", "--url", url, "--id", job_id,
                         "--out", out, "--variant", str(index)],
                        env, work, out)
                live.require(downloads > 0, "no video artifact was downloaded")
                passes.append("PASS h3 content: existing artifacts refused, unchanged")
            except BaseException:
                # The create call is billable. Try to remove the upstream job
                # before the local serve is stopped and the error propagates.
                live.run_cli(binary, [
                    "video", "delete", "--url", url, "--id", job_id,
                ], env, work, timeout_s=120)
                raise

            removed = live.run_cli(binary, [
                "video", "delete", "--url", url, "--id", job_id,
            ], env, work, timeout_s=120)
            live.require(removed.returncode == 0,
                         "video delete failed: " + removed.stderr[-2000:])
            passes.append("PASS h3 delete: upstream accepted the deletion")

            again = live.run_cli(binary, [
                "video", "status", "--url", url, "--id", job_id,
            ], env, work, timeout_s=120)
            live.require(live.http_status_failure(again, 404),
                         "status after delete did not answer HTTP 404: "
                         + (again.stderr[-500:] or again.stdout[-500:]))
            passes.append("PASS h3 delete: status after delete is HTTP 404")
    finally:
        if not keep:
            import shutil
            shutil.rmtree(work, ignore_errors=True)
        else:
            print(f"KEPT workdir: {work}")

    for line in passes:
        print(line)
    for line in gaps:
        print(line)
    print(f"PASS h3 live: {len(passes)} checks, {len(gaps)} not covered")


if __name__ == "__main__":
    sys.exit(live.blocked_main("h3_live.py", REQUIRED, run, OPTIONAL))

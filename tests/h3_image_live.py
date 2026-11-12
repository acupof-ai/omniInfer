#!/usr/bin/env python3
"""Opt-in live E2E for an SGLang MiniMax-H3 image upstream through umm-infer.

Chain: frozen umm-infer CLI -> local umm-infer serve (kind=sglang,
variant=image) -> the single configured upstream -> the real model.

Without UMMINFER_LIVE_OPT_IN=1 and every required variable, the script prints
BLOCKED/NOT_COVERED and makes no network request.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import live_common as live  # noqa: E402

REQUIRED = [
    "UMMINFER_BIN",
    "UMMINFER_H3_BASE_URL",
    "UMMINFER_H3_MODEL",
]
OPTIONAL = ["UMMINFER_H3_API_KEY"]


def base_requires_key(base_url):
    return not (base_url.startswith("http://127.0.0.1")
                or base_url.startswith("http://localhost"))


def profile_fields():
    """Image sampling profile. mac is the low-consumption default."""
    name = os.environ.get("UMMINFER_H3_PROFILE", "mac")
    default_steps = {"mac": "20", "h20b": "50"}.get(name)
    if default_steps is None:
        raise live.LiveFail(f"unknown UMMINFER_H3_PROFILE: {name}")
    steps = os.environ.get("UMMINFER_H3_IMAGE_STEPS", default_steps)
    return name, [f"num_inference_steps={steps}"]


def extra_fields():
    """Optional documented fields as NAME=VALUE, comma separated."""
    raw = os.environ.get("UMMINFER_H3_IMAGE_FIELDS", "")
    return [item for item in raw.split(",") if item]


def run():
    binary = live.binary_path(os.environ["UMMINFER_BIN"])
    base_url = os.environ["UMMINFER_H3_BASE_URL"]
    model = os.environ["UMMINFER_H3_MODEL"]
    key = os.environ.get("UMMINFER_H3_API_KEY", "")
    if base_requires_key(base_url) and not key:
        raise live.LiveBlocked(
            "UMMINFER_H3_API_KEY is required for a non-loopback upstream")
    served = os.environ.get("UMMINFER_SERVED_MODEL", "omni")
    size = os.environ.get("UMMINFER_H3_IMAGE_SIZE", "")
    profile_name, fields = profile_fields()
    fields += extra_fields()

    serve_env = dict(os.environ)
    serve_env["UMMINFER_UPSTREAM_API_KEY"] = key
    serve_args = [
        "--host", "127.0.0.1", "--port", "0", "--model", served,
        "--upstream-kind", "sglang",
        "--upstream-base-url", base_url,
        "--upstream-model", model,
        "--upstream-variant", "image",
    ]
    env = live.cli_env()
    passes, gaps = [], []
    keep = "--keep" in sys.argv[1:]
    work = tempfile.mkdtemp(prefix="umm-live-h3-image-")

    try:
        with live.Serve(binary, serve_args, serve_env, work, "h3-image") as server:
            url = f"http://127.0.0.1:{server.port}/v1"
            generated = os.path.join(work, "generated.png")
            create = [
                "image", "generate", "--url", url, "--model", served,
                "--prompt", "A single red maple leaf on white paper, studio light.",
                "--out", generated,
                "--timeout-ms", os.environ.get("UMMINFER_H3_CREATE_MS", "600000"),
            ]
            if size:
                create += ["--size", size]
            for field in fields:
                create += ["--field", field]
            result = live.run_cli(binary, create, env, work, timeout_s=660)
            live.require(result.returncode == 0,
                         "image generate failed: " + result.stderr[-2000:])
            kind, (digest, blob_size), info = live.check_image(generated)
            if size:
                width, _, height = size.lower().partition("x")
                if width.isdigit() and height.isdigit():
                    live.require(
                        info.get("width") == int(width) and info.get("height") == int(height),
                        f"generated {kind} is {info.get('width')}x{info.get('height')}, "
                        f"requested {size}")
            passes.append(
                f"PASS h3-image generate: {kind} {blob_size} bytes sha256={digest} "
                f"profile={profile_name}")

            live.expect_existing_output_refused(
                binary,
                ["image", "generate", "--url", url, "--model", served,
                 "--prompt", "A second leaf.", "--out", generated],
                env, work, generated)
            passes.append("PASS h3-image generate: existing artifact refused, unchanged")

            edited = os.path.join(work, "edited.png")
            result = live.run_cli(binary, [
                "image", "edit", "--url", url, "--model", served,
                "--prompt", "Make the background light blue.",
                "--image", generated, "--out", edited,
                "--timeout-ms", os.environ.get("UMMINFER_H3_CREATE_MS", "600000"),
            ], env, work, timeout_s=660)
            live.require(result.returncode == 0,
                         "image edit failed: " + result.stderr[-2000:])
            kind, (digest, blob_size), _ = live.check_image(edited)
            passes.append(
                f"PASS h3-image edit: {kind} {blob_size} bytes sha256={digest}")

            live.expect_existing_output_refused(
                binary,
                ["image", "edit", "--url", url, "--model", served,
                 "--prompt", "Make it darker.", "--image", generated,
                 "--out", edited],
                env, work, edited)
            passes.append("PASS h3-image edit: existing artifact refused, unchanged")

        gaps.append(
            "NOT_COVERED h3-image multi-output naming: the CLI output naming is "
            "not frozen yet, so only a single artifact per call is asserted")
        gaps.append(
            "NOT_COVERED h3-image 16-reference edit: only a one-image edit runs")
        gaps.append(
            "NOT_COVERED h3-image output formats: only the server default "
            "encoding is requested")
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
    print(f"PASS h3-image live: {len(passes)} checks, {len(gaps)} not covered")


if __name__ == "__main__":
    sys.exit(live.blocked_main("h3_image_live.py", REQUIRED, run, OPTIONAL))

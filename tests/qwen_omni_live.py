#!/usr/bin/env python3
"""Opt-in live E2E for a Qwen-Omni upstream through umm-infer.

Chain: frozen umm-infer CLI -> local umm-infer serve (kind=qwen, variant=omni)
-> the single configured upstream -> the real model.

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
    "UMMINFER_QWEN_BASE_URL",
    "UMMINFER_QWEN_MODEL",
    "UMMINFER_QWEN_API_KEY",
    "UMMINFER_QWEN_VOICE",
]


def run():
    binary = live.binary_path(os.environ["UMMINFER_BIN"])
    base_url = os.environ["UMMINFER_QWEN_BASE_URL"]
    model = os.environ["UMMINFER_QWEN_MODEL"]
    voice = os.environ["UMMINFER_QWEN_VOICE"]
    max_tokens = os.environ.get("UMMINFER_LIVE_MAX_TOKENS", "64")
    # The served name is the local alias. The upstream model id is forwarded by
    # the proxy (it rewrites the request body), so the CLI must use this alias.
    served = os.environ.get("UMMINFER_SERVED_MODEL", "omni")
    live.require(base_url.startswith("https://"),
                 "UMMINFER_QWEN_BASE_URL must be https for the hosted upstream")

    serve_env = dict(os.environ)
    serve_env["UMMINFER_UPSTREAM_API_KEY"] = os.environ["UMMINFER_QWEN_API_KEY"]
    serve_args = [
        "--host", "127.0.0.1", "--port", "0", "--model", served,
        "--upstream-kind", "qwen",
        "--upstream-base-url", base_url,
        "--upstream-model", model,
        "--upstream-variant", "omni",
    ]
    env = live.cli_env()
    passes, gaps = [], []

    with tempfile.TemporaryDirectory(prefix="umm-live-qwen-") as work:
        with live.Serve(binary, serve_args, serve_env, work, "qwen") as server:
            url = f"http://127.0.0.1:{server.port}/v1"

            text = live.run_cli(binary, [
                "chat", "--url", url, "--model", served,
                "--prompt", "Reply with one short sentence about the sea.",
                "--max-tokens", max_tokens,
            ], env, work, timeout_s=180)
            live.require(text.returncode == 0, "text chat failed: " + text.stderr[-2000:])
            live.require(text.stdout.strip() != "", "text chat returned empty output")
            passes.append(f"PASS qwen text: non-empty reply ({len(text.stdout.strip())} chars)")

            stream = live.run_cli(binary, [
                "chat", "--url", url, "--model", served, "--stream",
                "--prompt", "Count from one to five.", "--max-tokens", max_tokens,
            ], env, work, timeout_s=180)
            live.require(stream.returncode == 0, "text stream failed: " + stream.stderr[-2000:])
            live.require(stream.stdout.strip() != "", "text stream delivered no delta")
            passes.append("PASS qwen text stream: deltas received, exit 0")

            audio_path = os.path.join(work, "speech.wav")
            audio = live.run_cli(binary, [
                "chat", "--url", url, "--model", served, "--stream",
                "--prompt", "Say hello.", "--max-tokens", max_tokens,
                "--audio-out", audio_path, "--voice", voice,
            ], env, work, timeout_s=300)
            live.require(audio.returncode == 0, "audio stream failed: " + audio.stderr[-2000:])
            digest, size = live.check_wav(audio_path)
            passes.append(f"PASS qwen audio stream: RIFF/WAVE {size} bytes sha256={digest}")

            live.expect_existing_output_refused(
                binary,
                ["chat", "--url", url, "--model", served, "--stream",
                 "--prompt", "Say hello again.", "--audio-out", audio_path,
                 "--voice", voice],
                env, work, audio_path)
            passes.append("PASS qwen audio: existing artifact refused, unchanged")

        gaps.append("NOT_COVERED qwen metadata: the frozen CLI prints text only; "
                    "model/usage/finish_reason are not observable through it")

    for line in passes:
        print(line)
    for line in gaps:
        print(line)
    print(f"PASS qwen live: {len(passes)} checks, {len(gaps)} not covered")


if __name__ == "__main__":
    sys.exit(live.blocked_main("qwen_omni_live.py", REQUIRED, run))

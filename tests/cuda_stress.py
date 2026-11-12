#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Repeat the CUDA test binaries and stop at the first failure.

Standard library only. One line per round, a final count, and one log per
round. Logs are capped at 8 MiB and each round runs in its own process group,
so a timed-out round is killed with its children.
"""
import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

LOG_CAP = 8 * 1024 * 1024
MAX_ROUNDS = 100
MIN_FREE_BYTES = 256 * 1024 * 1024


def run_bounded(cmd, env, timeout_s, log_path, cap=LOG_CAP):
    """Runs cmd, writing at most cap bytes. Returns (rc, timed_out, bytes, cut)."""
    stored = 0
    truncated = False
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )

        def drain():
            nonlocal stored, truncated
            while True:
                chunk = proc.stdout.read(1 << 16)
                if not chunk:
                    return
                room = cap - stored
                if room > 0:
                    log.write(chunk[:room])
                    stored += min(room, len(chunk))
                if len(chunk) > room:
                    truncated = True

        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        timed_out = False
        try:
            rc = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            rc = proc.wait()
        reader.join()
    return rc, timed_out, stored, truncated


def check_disk(log_dir):
    free = shutil.disk_usage(log_dir).free
    if free < MIN_FREE_BYTES:
        raise SystemExit(
            "cuda_stress: only %d MiB free under %s, need %d MiB"
            % (free // (1 << 20), log_dir, MIN_FREE_BYTES // (1 << 20))
        )


def round_label(binary):
    return os.path.basename(binary)


def run_round(index, binary, timeout_s, log_dir, env):
    tag = "%02d-%s" % (index, round_label(binary))
    log_path = os.path.join(log_dir, tag + ".log")
    start = time.time()
    rc, timed_out, stored, cut = run_bounded([binary], env, timeout_s, log_path)
    elapsed = time.time() - start
    ok = rc == 0 and not timed_out
    status = "ok" if ok else ("timeout" if timed_out else "exit=%d" % rc)
    print(
        "round %d %-28s %-8s %6.1fs log=%s bytes=%d%s"
        % (
            index,
            round_label(binary),
            status,
            elapsed,
            os.path.basename(log_path),
            stored,
            " (truncated)" if cut else "",
        ),
        flush=True,
    )
    return ok, log_path


def self_test():
    """Proves the runner detects success, nonzero exit, timeout and log cap."""
    with tempfile.TemporaryDirectory(prefix="cuda-stress-selftest-") as tmp:
        scripts = {
            "ok.sh": "#!/bin/sh\necho ok\nexit 0\n",
            "fail.sh": "#!/bin/sh\necho boom\nexit 3\n",
            "hang.sh": "#!/bin/sh\nsleep 300\n",
            "spam.sh": "#!/bin/sh\ndd if=/dev/zero bs=1048576 count=9 2>/dev/null\n",
        }
        paths = {}
        for name, body in scripts.items():
            path = os.path.join(tmp, name)
            with open(path, "w") as handle:
                handle.write(body)
            os.chmod(path, 0o755)
            paths[name] = path
        env = os.environ.copy()

        rc, timed_out, _, _ = run_bounded([paths["ok.sh"]], env, 5,
                                          os.path.join(tmp, "ok.log"))
        assert rc == 0 and not timed_out, (rc, timed_out)

        rc, timed_out, _, _ = run_bounded([paths["fail.sh"]], env, 5,
                                          os.path.join(tmp, "fail.log"))
        assert rc == 3 and not timed_out, (rc, timed_out)

        rc, timed_out, _, _ = run_bounded([paths["hang.sh"]], env, 1,
                                          os.path.join(tmp, "hang.log"))
        assert timed_out and rc != 0, (rc, timed_out)

        rc, timed_out, stored, cut = run_bounded(
            [paths["spam.sh"]], env, 20, os.path.join(tmp, "spam.log")
        )
        assert not timed_out and cut and stored <= LOG_CAP, (rc, stored, cut)

    try:
        argparse.Namespace(rounds=MAX_ROUNDS + 1)
        assert False, "iteration cap missing"
    except AssertionError:
        pass
    print("self-test ok")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ipc", help="path to cuda_ipc_test")
    parser.add_argument("--pipeline", help="path to cuda_pipeline_test")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="seconds per round")
    parser.add_argument("--log-dir", default="cuda-stress-logs")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0
    if not args.ipc or not args.pipeline:
        parser.error("--ipc and --pipeline are required without --self-test")
    if args.rounds < 1 or args.rounds > MAX_ROUNDS:
        parser.error("--rounds must be between 1 and %d" % MAX_ROUNDS)
    for path in (args.ipc, args.pipeline):
        if not os.access(path, os.X_OK):
            parser.error("not executable: %s" % path)

    os.makedirs(args.log_dir, exist_ok=True)
    check_disk(args.log_dir)

    env = os.environ.copy()
    # Adds the 1 MiB and slot-limit-minus-one cases to cuda_ipc_test.
    env["UMMINFER_CUDA_TEST_STRESS"] = "1"
    binaries = [args.ipc, args.pipeline]

    passed = 0
    for index in range(args.rounds):
        check_disk(args.log_dir)
        binary = binaries[index % len(binaries)]
        ok, log_path = run_round(index + 1, binary, args.timeout,
                                 args.log_dir, env)
        if not ok:
            print("stopped at round %d; log preserved: %s"
                  % (index + 1, log_path), flush=True)
            return 1
        passed += 1

    print("cuda_stress: %d/%d rounds passed" % (passed, args.rounds))
    return 0


if __name__ == "__main__":
    sys.exit(main())

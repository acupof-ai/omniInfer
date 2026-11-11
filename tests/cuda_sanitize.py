#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Runs the CUDA test binaries under compute-sanitizer memcheck.

Standard library only. With --target-processes all, --log-file %p names the
sanitizer process, not the target, so one run produces one aggregate log. Each
run therefore gets a fresh directory, and the check is one summary with zero
errors, not a log count. The target's own stdout/stderr are kept next to it,
and "all checks passed" must appear there.

cuda_ipc_test runs twice. The fatal run uses --report-api-errors no because
test [5] deliberately opens a corrupted IPC handle. The second run keeps the
default API checking: it must exit 99 and report cudaIpcOpenMemHandle, which
proves the child process that opened the handle was tracked.

Large stress cases stay out: this script never sets OMNIINFER_CUDA_TEST_STRESS.

Exit codes: 0 clean, 1 problem, 2 compute-sanitizer not found.
"""
import argparse
import glob
import os
import re
import signal
import shutil
import subprocess
import sys

ERROR_EXIT = 99
CANDIDATES = [
    "compute-sanitizer",
    "/usr/local/cuda/bin/compute-sanitizer",
    "/opt/cuda/bin/compute-sanitizer",
]
SUMMARY = re.compile(r"ERROR SUMMARY:\s+(\d+)\s+error")


def find_tool(explicit):
    for candidate in [explicit] + CANDIDATES:
        if not candidate:
            continue
        if os.sep in candidate:
            if os.path.exists(candidate):
                return candidate
        else:
            path = shutil.which(candidate)
            if path:
                return path
    return None


def fresh_dir(root, tag):
    path = os.path.join(root, tag)
    suffix = 0
    while os.path.exists(path):
        suffix += 1
        path = os.path.join(root, "%s-%d" % (tag, suffix))
    os.makedirs(path)
    return path


def kill_group(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def read_text(path):
    try:
        with open(path, "rb") as handle:
            return handle.read().decode("utf-8", "replace")
    except OSError:
        return ""


class Result:
    def __init__(self):
        self.rc = None
        self.timed_out = False
        self.logs = []
        self.log_bytes = 0
        self.summaries = 0
        self.errors = 0
        self.sanitizer_text = ""
        self.target_text = ""
        self.run_dir = ""


def run_sanitized(tool, binary, root, tag, timeout_s, report_api_errors):
    result = Result()
    result.run_dir = fresh_dir(root, tag)
    log_path = os.path.join(result.run_dir, "sanitizer.log")
    out_path = os.path.join(result.run_dir, "target.stdout")
    err_path = os.path.join(result.run_dir, "target.stderr")

    cmd = [
        tool,
        "--tool",
        "memcheck",
        "--target-processes",
        "all",
        "--error-exitcode",
        str(ERROR_EXIT),
    ]
    if not report_api_errors:
        cmd += ["--report-api-errors", "no"]
    # compute-sanitizer 2023.2.2 parses a bare "--" as an ambiguous option and
    # exits 255, so the program follows the log-file option directly.
    cmd += ["--log-file", log_path, binary]

    with open(out_path, "wb") as out, open(err_path, "wb") as err:
        proc = subprocess.Popen(
            cmd, stdout=out, stderr=err, start_new_session=True
        )
        try:
            result.rc = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            result.timed_out = True
            kill_group(proc)
            result.rc = proc.wait()

    result.logs = sorted(glob.glob(os.path.join(result.run_dir, "*.log")))
    for path in result.logs:
        data = read_text(path)
        result.log_bytes += len(data.encode("utf-8", "replace"))
        result.sanitizer_text += data
        for match in SUMMARY.finditer(data):
            result.summaries += 1
            result.errors += int(match.group(1))
    result.target_text = read_text(out_path) + read_text(err_path)

    print(
        "%-18s rc=%-3s logs=%d summary=%d errors=%d target_bytes=%d%s"
        % (
            tag,
            result.rc,
            len(result.logs),
            result.summaries,
            result.errors,
            len(result.target_text),
            " timeout" if result.timed_out else "",
        ),
        flush=True,
    )
    print("  dir=%s" % result.run_dir, flush=True)
    return result


def common_problems(result, name):
    problems = []
    if result.timed_out:
        problems.append("%s timed out" % name)
    if not result.logs:
        problems.append("%s produced no sanitizer log" % name)
    if result.summaries == 0:
        problems.append("%s has no ERROR SUMMARY" % name)
    if result.errors:
        problems.append("%s has %d sanitizer error(s)" % (name, result.errors))
    if "all checks passed" not in result.target_text:
        problems.append("%s did not report all checks passed" % name)
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ipc", help="path to cuda_ipc_test")
    parser.add_argument("--pipeline", help="path to cuda_pipeline_test")
    parser.add_argument("--log-dir", default="cuda-sanitize-logs")
    parser.add_argument("--timeout", type=float, default=1800.0,
                        help="seconds per sanitized run")
    parser.add_argument("--tool", help="compute-sanitizer path")
    args = parser.parse_args()

    if not args.ipc and not args.pipeline:
        parser.error("give --ipc, --pipeline, or both")
    tool = find_tool(args.tool)
    if tool is None:
        print("cuda_sanitize: compute-sanitizer not found; nothing was checked",
              file=sys.stderr)
        return 2

    os.environ.pop("OMNIINFER_CUDA_TEST_STRESS", None)
    os.makedirs(args.log_dir, exist_ok=True)
    failures = []

    if args.ipc:
        clean = run_sanitized(tool, args.ipc, args.log_dir, "ipc",
                              args.timeout, False)
        failures += common_problems(clean, "ipc")
        if clean.rc != 0:
            failures.append("ipc clean run exited %s" % clean.rc)

        api = run_sanitized(tool, args.ipc, args.log_dir, "ipc-api-default",
                            args.timeout, True)
        # The corrupted handle in test [5] is opened by a child process. Exit
        # 99 with cudaIpcOpenMemHandle in the log proves child tracking.
        if api.rc != ERROR_EXIT:
            failures.append(
                "ipc default-API run exited %s, expected %d"
                % (api.rc, ERROR_EXIT)
            )
        if "cudaIpcOpenMemHandle" not in api.sanitizer_text:
            failures.append(
                "ipc default-API run does not report cudaIpcOpenMemHandle"
            )

    if args.pipeline:
        pipeline = run_sanitized(tool, args.pipeline, args.log_dir, "pipeline",
                                 args.timeout, True)
        failures += common_problems(pipeline, "pipeline")
        if pipeline.rc != 0:
            failures.append("pipeline run exited %s" % pipeline.rc)

    if failures:
        for item in failures:
            print("cuda_sanitize: FAILED: " + item, file=sys.stderr)
        print("cuda_sanitize: logs in %s" % args.log_dir, file=sys.stderr)
        return 1
    print("cuda_sanitize: clean (%s)" % args.log_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

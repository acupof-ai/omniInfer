#!/usr/bin/env python3
"""Run one real frozen-chain ummmlx GUI conversation in an exclusive window.

The server binary and the model directory have no built-in default. Pass
--server and --model, or set UMMMLX_SERVER and UMMMLX_MODEL. The default
acceptance mode requires the server binary to match FROZEN_SHA256; pass
--allow-unfrozen to run another build, which the report records as
"frozen": false.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time

CLIENT = Path(__file__).resolve().parents[1]
ROOT = CLIENT.parents[1]
RUNNER = Path(__file__).resolve()
PERF_HELPER = ROOT / "tests/mlx_perf.py"
SPEC = importlib.util.spec_from_file_location("ummmlx_perf_lock", PERF_HELPER)
perf = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(perf)

FROZEN_SHA256 = "d51384fd053fb78d4ff0fa0b49a912d33639f225d727ea61991a9feb06105f01"
STAGE = ROOT / "python/umminfer_mlx_stage.py"
APP = CLIENT / "build/ummmlx.app/Contents/MacOS/ummmlx"
LOG_CAP = 65536
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PROMPT = "Reply with one short sentence for a real streaming conversation."
FOLLOWUP = "Reply with one short sentence about your previous reply."
# The app waits up to 120 s for each of four requests; leave headroom above 480.
APP_TIMEOUT_S = 600


def drain_app(process, output):
    while True:
        chunk = process.stdout.read(4096)
        if not chunk:
            return
        if len(output) < LOG_CAP:
            output.extend(chunk[:LOG_CAP - len(output)])


def stop_app(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def executable_hashes(binary, model):
    return {
        "server": perf.digest(binary),
        "stage": perf.digest(STAGE),
        "app": perf.digest(APP),
        "weights": perf.digest(model / "model.safetensors"),
        "runner": perf.digest(RUNNER),
        "perf_helper": perf.digest(PERF_HELPER),
    }


def preference_snapshot():
    return subprocess.run(
        ["defaults", "export", "dev.umminfer.ummmlx", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10, check=False)


def require_png(path):
    perf.require(path.stat().st_size > 1000, f"invalid screenshot: {path}")
    with path.open("rb") as image:
        header = image.read(33)
    perf.require(
        len(header) == 33 and header[:8] == PNG_SIGNATURE and
        header[8:12] == b"\x00\x00\x00\x0d" and
        header[12:16] == b"IHDR" and
        int.from_bytes(header[16:20], "big") > 0 and
        int.from_bytes(header[20:24], "big") > 0,
        f"invalid PNG screenshot: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True,
                        help="new output directory; existing paths are refused")
    parser.add_argument("--server", type=Path,
                        default=os.environ.get("UMMMLX_SERVER"),
                        help="umm-infer binary (default: $UMMMLX_SERVER)")
    parser.add_argument("--model", type=Path,
                        default=os.environ.get("UMMMLX_MODEL"),
                        help="MLX model snapshot directory (default: $UMMMLX_MODEL)")
    parser.add_argument("--served-model",
                        default=os.environ.get("UMMMLX_SERVED_MODEL",
                                               perf.SERVED_MODEL),
                        help="model name the server advertises")
    parser.add_argument("--allow-unfrozen", action="store_true",
                        help="accept a server binary that is not FROZEN_SHA256")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--followup-prompt", default=FOLLOWUP)
    parser.add_argument("--error-model", default=None,
                        help="unserved model name for the visible-error step "
                             "(default: <served-model>-not-served)")
    args = parser.parse_args()

    if args.server is None or args.model is None:
        parser.error("pass --server and --model, or set UMMMLX_SERVER and "
                     "UMMMLX_MODEL")
    binary = Path(args.server).expanduser().resolve()
    model = Path(args.model).expanduser().resolve()
    error_model = args.error_model or f"{args.served_model}-not-served"

    args.out = args.out.expanduser().resolve()
    if args.out.exists():
        raise SystemExit(f"output path already exists: {args.out}")
    args.out.mkdir(parents=True, mode=0o700)

    report = {
        "server_binary": str(binary),
        "stage": str(STAGE),
        "model": str(model),
        "served_model": args.served_model,
        "app": str(APP),
        "runner": str(RUNNER),
        "perf_helper": str(PERF_HELPER),
        "python": sys.executable,
        "out": str(args.out),
        "frozen": False,
        "passed": False,
    }
    lock = None
    children = perf.Children()
    app_process = None
    app_thread = None
    app_log = bytearray()
    real_exit = None
    failure = ""
    needles = []
    try:
        perf.require(binary.is_file(), f"server binary not found: {binary}")
        server_sha256 = perf.digest(binary)
        report["server_sha256"] = server_sha256
        report["frozen"] = server_sha256 == FROZEN_SHA256
        if not args.allow_unfrozen:
            perf.require(server_sha256 == FROZEN_SHA256,
                         "the selected server is not the frozen binary; "
                         "pass --allow-unfrozen for a non-acceptance run")
        perf.require((model / "tokenizer.json").is_file(),
                     f"model not found: {model}")
        lock = perf.acquire_lock()
        extra_needles = {
            "/tmp/umm-nolog-harness", "run_nolog_ab", "run_smoke_locked",
            "diag_current_profile", "diag_sync_profile", "diag_vocab",
            "recompute_nolog_report", "fix_report", "mlx_greedy_perf",
            "mlx_prompt_cache_perf", "mlx_e2e_interleave_perf",
            "mlx_perf_compare", "mlx_stage_test", "h3_live", "h3_image_live",
            "qwen_omni_live", str(APP.resolve()), APP.name,
        }
        needles = sorted(set(perf.patrol_needles(binary, STAGE)) | extra_needles)
        perf.check_exclusive(needles)
        perf.require(APP.is_file(), f"build the app first: {APP}")
        perf.require_model_layout(model)
        report["hashes_before"] = executable_hashes(binary, model)
        report["model_input_hashes_before"] = perf.model_input_hashes(model)
        port = perf.free_port()
        endpoint = f"http://127.0.0.1:{port}/v1"
        env = perf.child_env()
        with tempfile.TemporaryDirectory(prefix="ummmlx-e2e-ipc-") as ipc:
            entry = f"ipc://{ipc}/entry"
            completion = f"ipc://{ipc}/completion"
            abort = f"ipc://{ipc}/abort"
            server_command = [
                str(binary), "serve", "--host", "127.0.0.1", "--port", str(port),
                "--model", args.served_model, "--entry", "mlx",
                "--entry-endpoint", entry, "--completion", completion, "--abort", abort,
            ]
            stage_command = [
                sys.executable, "-I", str(STAGE), "--name", "mlx", "--recv", entry,
                "--coordinator", completion, "--abort", abort, "--model", str(model),
            ]
            report["server_command"] = server_command
            report["stage_command"] = stage_command
            server = children.spawn("server", server_command, env)
            perf.wait_port(port, server)
            stage = children.spawn("stage", stage_command, env)
            perf.wait_ready(port, server, stage)
            perf.check_exclusive(needles)

            app_command = [
                str(APP), "--automation", "--stop-first",
                f"--endpoint={endpoint}", f"--model={args.served_model}",
                f"--prompt={args.prompt}",
                f"--followup-prompt={args.followup_prompt}",
                f"--error-model={error_model}",
                f"--screenshots={args.out / 'screenshots'}",
                f"--output={args.out / 'automation.json'}",
            ]
            report["app_command"] = app_command
            defaults_before = preference_snapshot()
            app_process = subprocess.Popen(
                app_command, cwd=str(CLIENT), env=env, start_new_session=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            app_thread = threading.Thread(target=drain_app, args=(app_process, app_log), daemon=True)
            app_thread.start()
            deadline = time.monotonic() + APP_TIMEOUT_S
            for _ in range(APP_TIMEOUT_S):
                if app_process.poll() is not None:
                    break
                if time.monotonic() >= deadline:
                    break
                perf.require(server.poll() is None, "server exited during GUI E2E")
                perf.require(stage.poll() is None, "stage exited during GUI E2E")
                perf.check_exclusive(needles)
                time.sleep(1)
            if app_process.poll() is None:
                stop_app(app_process)
                real_exit = 124
                report["app_raw_returncode"] = app_process.returncode
                raise RuntimeError(f"GUI E2E exceeded {APP_TIMEOUT_S} seconds")
            raw_returncode = app_process.returncode
            report["app_raw_returncode"] = raw_returncode
            real_exit = 128 - raw_returncode if raw_returncode < 0 else raw_returncode
            app_thread.join(timeout=20)
            perf.require(not app_thread.is_alive(), "GUI log reader did not exit")
            perf.check_exclusive(needles)
            report["real_exit"] = real_exit
            report["automation"] = json.loads((args.out / "automation.json").read_text())
            defaults_after = preference_snapshot()
            perf.require(
                (defaults_after.returncode, defaults_after.stdout) ==
                (defaults_before.returncode, defaults_before.stdout),
                "automation changed persistent user preferences")
            perf.require(real_exit == 0, f"GUI app exited {real_exit}")
            automation = report["automation"]
            perf.require(automation.get("passed") is True, "app automation reported failure")
            perf.require(automation.get("endpoint") == endpoint,
                         "automation used a different endpoint")
            perf.require(automation.get("model") == args.served_model,
                         "automation used a different model")
            perf.require(automation.get("stop_checked") is True,
                         "no real stream delta reached the Stop action")
            perf.require(automation.get("stop_status") == "Stopped",
                         "the real Stop action did not finish in Stopped")
            perf.require(automation.get("response"), "the real GUI response is empty")
            perf.require(automation.get("followup_response"),
                         "the second same-session response is empty")
            # Request history: the stopped request, the first turn, the second
            # turn in the same session (which must carry the first exchange),
            # and the rejected request (which carries both exchanges).
            perf.require(
                automation.get("requests") ==
                [["user"], ["user"], ["user", "assistant", "user"],
                 ["user", "assistant", "user", "assistant", "user"]],
                "the same session did not send the expected request history")
            perf.require(
                automation.get("turns") == [
                    {"role": "user", "state": "complete"},
                    {"role": "assistant", "state": "complete"},
                    {"role": "user", "state": "complete"},
                    {"role": "assistant", "state": "complete"},
                ],
                "the same session did not keep two complete exchanges")
            perf.require(automation.get("error_checked") is True,
                         "the visible-error step did not run")
            perf.require(automation.get("error_status") == "Error",
                         "the rejected request did not end in Error")
            error_text = automation.get("error_text") or ""
            perf.require("unsupported model" in error_text and error_model in error_text,
                         f"the server error text is not visible: {error_text!r}")
            expected = [
                str(args.out / "screenshots" / f"ummmlx-{name}.png")
                for name in ("light", "dark", "error")
            ]
            screenshots = automation.get("screenshots", [])
            perf.require(screenshots == expected,
                         "screenshots must be the expected light, dark, and error paths")
            for path in map(Path, screenshots):
                require_png(path)
            digests = {perf.digest(Path(path)) for path in screenshots}
            perf.require(len(digests) == 3,
                         "the captured screenshots are not all distinct")
            report["screenshots"] = screenshots
            report["screenshot_sha256"] = {
                Path(path).name: perf.digest(Path(path)) for path in screenshots
            }

        report["passed"] = True
    except Exception as exc:
        failure = str(exc)
        report["passed"] = False
        report["failure"] = failure
    finally:
        cleanup_errors = []
        try:
            stop_app(app_process)
        except Exception as exc:
            cleanup_errors.append(f"app cleanup failed: {exc}")
        if app_thread is not None:
            try:
                app_thread.join(timeout=20)
            except Exception as exc:
                cleanup_errors.append(f"app log reader join failed: {exc}")
        try:
            (args.out / "app.log").write_bytes(bytes(app_log))
        except OSError as exc:
            cleanup_errors.append(f"app log write failed: {exc}")
        try:
            cleanup_errors.extend(children.close())
        except Exception as exc:
            cleanup_errors.append(f"server/stage cleanup failed: {exc}")
        if app_thread is not None and app_thread.is_alive():
            cleanup_errors.append("app log reader did not exit")
        for name, data in children.logs.items():
            try:
                (args.out / f"{name}.log").write_bytes(bytes(data))
            except OSError as exc:
                cleanup_errors.append(f"{name} log write failed: {exc}")
        report["child_exits"] = {
            name: process.returncode for name, process in children.procs.items()
        }
        for reader in children.readers:
            if reader.is_alive():
                cleanup_errors.append("a server/stage log reader did not exit")
        if lock is not None and needles:
            try:
                perf.check_exclusive(needles)
            except Exception as exc:
                report["passed"] = False
                report["final_patrol_failure"] = str(exc)
                if not failure:
                    failure = str(exc)
        if "hashes_before" in report:
            try:
                report["hashes_after"] = executable_hashes(binary, model)
                report["model_input_hashes_after"] = perf.model_input_hashes(model)
                perf.require(report["hashes_after"] == report["hashes_before"],
                             "an executable or weights changed during E2E")
                perf.require(report["model_input_hashes_after"] ==
                             report["model_input_hashes_before"],
                             "a model input changed during E2E")
            except Exception as exc:
                report["passed"] = False
                report["final_hash_failure"] = str(exc)
                if not failure:
                    failure = str(exc)
        if cleanup_errors:
            report["passed"] = False
            report["cleanup_errors"] = cleanup_errors
            if not failure:
                failure = "; ".join(cleanup_errors)
        if failure:
            report["failure"] = failure
        report["real_exit"] = real_exit
        try:
            (args.out / "report.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n")
        finally:
            if lock is not None:
                lock.close()

    print(f"REPORT={args.out / 'report.json'}")
    print(f"REAL_EXIT={real_exit if real_exit is not None else 'NOT_RUN'}")
    if not report.get("passed"):
        print(f"FAILURE={failure or report.get('cleanup_errors')}", file=sys.stderr)
        return real_exit if real_exit not in (None, 0) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

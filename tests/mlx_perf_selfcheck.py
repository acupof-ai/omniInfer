#!/usr/bin/env python3
"""Check the perf-harness helpers that a model run cannot cheaply prove.

Covers three claims of tests/mlx_perf.py:
  * each mode reads only its own stage perf log;
  * the output hash identifies the text and is never summarized as a number;
  * a hash difference between samples or paths is detected.
"""

import importlib.util
import inspect
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("mlx_perf", HERE / "mlx_perf.py")
perf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(perf)


def check(name, condition):
    print(f"{'PASS' if condition else 'FAIL'} {name}")
    if not condition:
        raise SystemExit(1)


def main():
    with tempfile.TemporaryDirectory() as directory:
        stage_log = Path(directory) / "stage.jsonl"
        e2e_log = Path(directory) / "e2e.jsonl"
        stage_log.write_text(json.dumps({"event": "request", "request_id": "stage-1"}) + "\n")
        e2e_log.write_text(json.dumps({"event": "request", "request_id": "e2e-1"}) + "\n")

        stage = perf.PerfLog(stage_log)
        e2e = perf.PerfLog(e2e_log)
        check("stage cursor reads the stage log",
              stage.next().get("request_id") == "stage-1")
        check("e2e cursor reads the e2e log", e2e.next().get("request_id") == "e2e-1")
        for name, cursor in (("stage", stage), ("e2e", e2e)):
            try:
                cursor.next()
            except RuntimeError:
                print(f"PASS {name} cursor stops at its own end")
            else:
                check(f"{name} cursor stops at its own end", False)

    check("equal text hashes equal",
          perf.text_sha256("tip\n") == perf.text_sha256("tip\n"))
    check("different text hashes differently",
          perf.text_sha256("tip\n") != perf.text_sha256("tip."))
    check("CLI trailing newline is parsed out", perf.cli_text(b"tip\n") == "tip")

    # With `--warmup 0` the first row is both cold and warm. It must not enter
    # a median, or a sample that includes model load becomes the reference.
    first = {"iteration": 0, "warm": True, "cold": True, "total_ms": 999.0}
    later = {"iteration": 1, "warm": True, "cold": False, "total_ms": 10.0}
    summary = perf.block([first, later])
    check("a cold row stays out of the warm median",
          summary["warm"]["total_ms"]["median"] == 10.0)

    row = {"iteration": 5, "warm": True, "ttft_ms": 1.0, "total_ms": 100.0,
           "generation_tokens": 10, "output_tokens_per_s": 100.0,
           "output_sha256": "a" * 64}
    summary = perf.block([row])
    check("medians cover numeric metrics", summary["warm"]["total_ms"]["median"] == 100.0)
    check("output hash is not summarized", "output_sha256" not in summary["warm"])

    same = {"direct": {"samples": [row, row]}}
    hashes, combined, consistent = perf.output_summary(same)
    check("one path yields its hash", hashes == {"direct": "a" * 64})
    check("identical samples are consistent", consistent and combined)

    mixed = {"direct": {"samples": [row]},
             "e2e": {"json": {"samples": [row, {**row, "output_sha256": "a" * 64}]},
                     "sse": {"samples": [{**row, "output_sha256": "def"}]}}}
    _, _, consistent = perf.output_summary(mixed)
    check("a differing path is detected", not consistent)

    changed = {"e2e": {"json": {"samples": [row, {**row, "output_sha256": "def"}]}}}
    try:
        perf.output_summary(changed)
    except RuntimeError:
        print("PASS a sample that changes output is rejected")
    else:
        check("a sample that changes output is rejected", False)

    skipped = {"e2e": {"json": {"samples": [row]},
                       "cli_json": {"skipped": "no chat subcommand"}}}
    hashes, _, consistent = perf.output_summary(skipped)
    check("a skipped path contributes no hash",
          list(hashes.values()) == ["a" * 64] and consistent)
    check("a skipped path is named",
          perf.skipped_paths(skipped) == ["e2e.cli_json"])

    # A sample without a hash must fail the run, not contribute an empty path.
    try:
        perf.require_outputs({"e2e": {"json": {"samples": [{**row, "output_sha256": ""}]}}})
    except RuntimeError:
        print("PASS a sample without an output hash is rejected")
    else:
        check("a sample without an output hash is rejected", False)

    # The cursor reads only appended bytes and matches the expected request id.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "log.jsonl"
        path.write_text(json.dumps({"event": "load", "seconds": 1.0}) + "\n" +
                        json.dumps({"event": "request", "request_id": "a"}) + "\n")
        cursor = perf.PerfLog(path)
        check("cursor skips non-request records",
              cursor.next("a").get("request_id") == "a")
        try:
            cursor.next()
        except RuntimeError:
            print("PASS cursor stops at the end of the log")
        else:
            check("cursor stops at the end of the log", False)
        with path.open("a") as sink:
            sink.write(json.dumps({"event": "request", "request_id": "b"}) + "\n")
        check("cursor reads only appended records",
              cursor.next("b").get("request_id") == "b")
        try:
            perf.PerfLog(path).next("wrong-id")
        except RuntimeError:
            print("PASS a mismatched request id is rejected")
        else:
            check("a mismatched request id is rejected", False)
        leftovers = perf.PerfLog(path)
        try:
            leftovers.require_exhausted()
        except RuntimeError:
            print("PASS an unread request record is reported")
        else:
            check("an unread request record is reported", False)

    missing = {"direct": {"samples": [{"iteration": 0, "warm": True}]}}
    try:
        perf.require_outputs(missing)
    except RuntimeError:
        print("PASS a sample without an output hash is rejected")
    else:
        check("a sample without an output hash is rejected", False)

    # A sharded or missing-single-file snapshot must be rejected before the
    # run, not fail on the first file read after measurement.
    with tempfile.TemporaryDirectory() as directory:
        (Path(directory) / "config.json").write_text("{}")
        try:
            perf.require_model_layout(Path(directory))
        except RuntimeError as exc:
            check("a model without model.safetensors is rejected early",
                  "single-file" in str(exc))
        else:
            check("a model without model.safetensors is rejected early", False)
        (Path(directory) / "model.safetensors").write_bytes(b"x")
        perf.require_model_layout(Path(directory))
        print("PASS a single-file model layout is accepted")

    # First-token latency is the first non-empty delta. A first response with
    # empty text must not stop the clock; it is not a token.
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        (directory / "model.safetensors").write_bytes(b"x")
        fake = directory / "fake_stage.py"
        fake.write_text(
            "import time\n"
            "from types import SimpleNamespace\n"
            "\n"
            "class Tokenizer:\n"
            "    def apply_chat_template(self, *args, **kwargs):\n"
            "        return 'prompt'\n"
            "\n"
            "def load_mlx_backend(path):\n"
            "    def generate(prompt, max_tokens, temperature):\n"
            "        time.sleep(0.15)\n"
            "        yield SimpleNamespace(text='', finish_reason=None,\n"
            "                              generation_tokens=1, prompt_tokens=1,\n"
            "                              prompt_tps=1.0, generation_tps=1.0,\n"
            "                              peak_memory=0.0)\n"
            "        time.sleep(0.15)\n"
            "        yield SimpleNamespace(text='ok', finish_reason='length',\n"
            "                              generation_tokens=2, prompt_tokens=1,\n"
            "                              prompt_tps=1.0, generation_tps=1.0,\n"
            "                              peak_memory=0.0)\n"
            "    return Tokenizer(), generate\n")

        direct = perf.run_direct(fake, directory, 4, 0, 1)
        row = direct["samples"][0]
        check("direct TTFT starts at the first non-empty delta, not the empty one",
              row["ttft_ms"] is not None and row["ttft_ms"] >= 140.0)

        # The real stage class, with the fake loader injected: the fake module
        # has no MLXStage.
        module = perf.load_stage_module(perf.DEFAULT_STAGE)
        fake_module = perf.load_stage_module(fake)
        log = directory / "stage.jsonl"
        stage = module.MLXStage("s", "a", "b", "c", directory,
                                loader=fake_module.load_mlx_backend, perf_log=log)
        stage.coordinator_socket = SimpleNamespace(send=lambda message: None)
        stage.load()
        stage.process({"type": "submit", "request_id": "r1", "data": {
            "request": {"inputs": {"messages": [{"role": "user", "content": "x"}]},
                        "params": {"max_new_tokens": 4, "temperature": 0.0,
                                   "stream": False}}}})
        record = [item for item in perf.read_records(log)
                  if item.get("event") == "request"][0]
        check("stage TTFT starts at the first non-empty delta, not the empty one",
              record["stage_ttft_seconds"] is not None
              and record["stage_ttft_seconds"] >= 0.14)

    # The CLI timeout and the harness timeout are one number: the command must
    # derive its value from REQUEST_TIMEOUT_S, not repeat a literal.
    check("the CLI timeout follows the harness timeout",
          "REQUEST_TIMEOUT_S * 1000" in inspect.getsource(perf.cli_command))
    check("wait_ready watches the stage process",
          "stage_proc.poll()" in inspect.getsource(perf.wait_ready))

    # With an external entry, /ready answers only after the stage announces
    # stage_ready. Waiting for it before the stage starts can never succeed, so
    # the order of these three calls is a property worth pinning.
    source = inspect.getsource(perf.run_e2e)
    wait_port_at = source.index("wait_port(")
    stage_at = source.rindex('children.spawn("stage"')
    ready_at = source.index("wait_ready(")
    check("e2e starts the stage before waiting for /ready",
          wait_port_at < stage_at < ready_at)
    check("e2e hands the stage process to wait_ready",
          "wait_ready(port, server, stage_proc)" in source)

    # The derived baseline is always patrolled, whichever stage this run
    # measures. A candidate run must see an externally started baseline.
    for stage in (perf.DEFAULT_STAGE, perf.BASELINE_STAGE_PATH):
        needles = perf.patrol_needles(perf.DEFAULT_BINARY, stage)
        check(f"patrol covers the baseline path for stage {stage.name}",
              str(perf.BASELINE_STAGE_PATH) in needles
              and perf.BASELINE_STAGE_PATH.name in needles)

    # The model input files decide the prompt, so they are pinned like the
    # weights. A missing required file fails before the run.
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                     "chat_template.jinja"):
            (directory / name).write_text('{"v": 1}')
        first = perf.model_input_hashes(directory)
        check("model input hashes cover every present file",
              set(first) == {"config.json", "tokenizer.json",
                             "tokenizer_config.json", "chat_template.jinja"})
        (directory / "config.json").write_text('{"v": 2}')
        check("a changed model input changes its hash",
              perf.model_input_hashes(directory)["config.json"]
              != first["config.json"])
        (directory / "tokenizer_config.json").unlink()
        try:
            perf.model_input_hashes(directory)
        except RuntimeError:
            print("PASS a missing tokenizer_config.json is rejected")
        else:
            check("a missing tokenizer_config.json is rejected", False)

    # A tokenizer config that decodes to a list has no chat_template key. That
    # must be a named RuntimeError carrying the path, not an AttributeError from
    # config.get() that escapes as an unattributed exception.
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        (directory / "config.json").write_text("{}")
        (directory / "tokenizer.json").write_text("{}")
        (directory / "tokenizer_config.json").write_text("[]")
        try:
            perf.model_input_hashes(directory)
        except RuntimeError as exc:
            check("a non-object tokenizer config is rejected by name",
                  "not a JSON object" in str(exc)
                  and "tokenizer_config.json" in str(exc))
        else:
            check("a non-object tokenizer config is rejected by name", False)

    # Raw rows are the only evidence a median may rest on. Each counterexample
    # below is a report that would otherwise pass with an invalid sample.
    def rows(count=20, **overrides):
        values = []
        for index in range(count):
            row = {"iteration": index, "warm": True, "cold": False,
                   "generation_tokens": 64, "total_ms": 100.0,
                   "output_tokens_per_s": 640.0, "peak_memory_gb": 0.5,
                   "ttft_ms": None, "prompt_tokens": 35,
                   "output_sha256": "a" * 64}
            row.update(overrides)
            values.append(row)
        return values

    def rejects(label, results, max_tokens=64):
        try:
            perf.require_valid_rows(results, max_tokens)
        except RuntimeError:
            print(f"PASS {label} is rejected")
        else:
            check(f"{label} is rejected", False)

    perf.require_valid_rows({"direct": {"samples": rows()}}, 64)
    print("PASS 20 valid warm rows are accepted")

    rejects("19 warm samples", {"direct": {"samples": rows(19)}})
    rejects("a warm sample with 63 generated tokens",
            {"direct": {"samples": rows(generation_tokens=63)}})
    rejects("a rate that does not match tokens/total_ms",
            {"direct": {"samples": rows(output_tokens_per_s=1000.0)}})
    rejects("a zero total_ms",
            {"direct": {"samples": rows(total_ms=0.0)}})
    rejects("an infinite total_ms",
            {"direct": {"samples": rows(total_ms=float("inf"))}})
    rejects("a non-positive peak memory",
            {"direct": {"samples": rows(peak_memory_gb=0.0)}})
    rejects("a non-positive ttft on a streaming row",
            {"stage": {"sse": {"samples": rows(ttft_ms=-1.0)}}})

    cold = rows(20)
    cold.append({**rows(1)[0], "iteration": 20, "cold": True})
    perf.require_valid_rows({"direct": {"samples": cold}}, 64)
    print("PASS a cold row does not count as a warm sample")
    cold = rows(19)
    cold.append({**rows(1)[0], "iteration": 19, "cold": True})
    rejects("19 warm samples plus a cold row", {"direct": {"samples": cold}})

    # The stored summary must equal what its raw rows produce.
    summary = perf.block(rows())["warm"]
    check("a stored median equals the recomputed raw median",
          summary["total_ms"]["median"] == 100.0
          and summary["output_tokens_per_s"]["median"] == 640.0
          and summary["total_ms"]["count"] == 20)
    check("a stable sample set has a non-negative cv",
          summary["total_ms"]["cv"] >= 0)

    skew = rows()
    for row in skew[:11]:
        row["total_ms"] = 200.0
    skew_block = perf.block(skew)["warm"]
    check("changed raw rows change the stored median",
          skew_block["total_ms"]["median"] == 200.0
          and skew_block["total_ms"]["median"]
          != summary["total_ms"]["median"])

    # Every numeric field in a row is summarized, so each one is validated.
    for field, value in (("stage_ttft_ms", -1000.0),
                         ("stage_total_ms", float("inf")),
                         ("decode_tokens_per_s", -7.0),
                         ("prompt_tokens_per_s", float("inf"))):
        rejects(f"a warm row with {field}={value!r}",
                {"direct": {"samples": rows(**{field: value})}})

    # The artifact map stores model inputs as `model.<name>`. The required set
    # must use that same layout, or every run aborts before measurement.
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        (directory / "config.json").write_text('{"v": 1}')
        (directory / "tokenizer.json").write_text('{"v": 1}')
        (directory / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": "template"}))
        (directory / "model.safetensors").write_bytes(b"x")
        inputs = perf.model_input_hashes(directory)
        check("a tokenizer-config template is hashed when no template file exists",
              inputs.get("chat_template") == perf.text_sha256(
                  json.dumps("template", sort_keys=True)))
        hashes = {"stage": "a" * 64, "script": "b" * 64, "weights": "c" * 64,
                  **{f"model.{name}": value for name, value in inputs.items()}}
        perf.require_hash_map(
            hashes, "hashes_before",
            required=("stage", "script", "weights")
            + tuple(f"model.{name}" for name in inputs))
        print("PASS the required hash names match the artifact map keys")

    # The real call sites live in main(). A revert to the bare-name tuple makes
    # every run abort, and only a source check sees it.
    source = inspect.getsource(perf.main)
    check("main requires the prefixed model.* hash names",
          source.count('tuple(f"model.{name}" for name in inputs)') >= 3)
    check("main re-enumerates the model inputs after the run",
          source.count("model_input_hashes(model).items()") == 2)
    # The final rehash runs in `finally`, after the main try. Its label must
    # name hashes_end, or the error points at the wrong map.
    check("the final rehash is labelled hashes_end",
          'require_hash_map(hashes_end, "hashes_end"' in source
          and 'require_hash_map(hashes_end, "hashes_after"' not in source)
    return 0


if __name__ == "__main__":
    sys.exit(main())

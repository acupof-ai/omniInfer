#!/usr/bin/env python3
"""Run an offline real-model smoke test against umm-infer and the MLX stage."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

from tokenizers import Tokenizer


STARTUP_TIMEOUT_S = 120
REQUEST_TIMEOUT_S = 60
SHUTDOWN_TIMEOUT_S = 20
LOG_CAP_BYTES = 65536
RESPONSE_CAP_BYTES = 1024 * 1024
SSE_LINE_CAP_BYTES = 65536
SSE_EVENT_CAP = 160
MAX_GENERATION_TOKENS = 96
# Above the stage's old 1024-entry abort history. Every id must survive.
SWEEP_REQUESTS = 1025
TOKENIZER = None


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close_connection(conn):
    sock = conn.sock
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
    conn.close()


def arm_timeout(conn):
    timer = threading.Timer(REQUEST_TIMEOUT_S, close_connection, args=(conn,))
    timer.daemon = True
    timer.start()
    return timer


def read_body(response):
    data = response.read(RESPONSE_CAP_BYTES + 1)
    require(len(data) <= RESPONSE_CAP_BYTES, "HTTP response exceeded byte cap")
    return data


def token_count(text):
    return len(TOKENIZER.encode(text, add_special_tokens=False).ids)


def result_metrics(text, elapsed, ttft=None):
    count = token_count(text)
    require(count <= MAX_GENERATION_TOKENS, "output exceeded token cap")
    return {"ttft_s": ttft, "total_s": elapsed, "tokens": count,
            "tokens_per_s": count / elapsed if elapsed else 0.0}


def open_chat(port, prompt, max_tokens, stream):
    require(1 <= max_tokens <= MAX_GENERATION_TOKENS, "invalid generation token cap")
    body = {"messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "stream": stream}
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=REQUEST_TIMEOUT_S)
    timer = arm_timeout(conn)
    started = time.perf_counter()
    try:
        conn.request("POST", "/v1/chat/completions", json.dumps(body),
                     {"Content-Type": "application/json"})
        response = conn.getresponse()
        if response.status != 200:
            data = read_body(response)
            raise AssertionError(f"chat HTTP {response.status}: {data[:1000]!r}")
        return conn, timer, response, started
    except Exception:
        timer.cancel()
        close_connection(conn)
        raise


def json_request(port, prompt, max_tokens):
    conn, timer, response, started = open_chat(port, prompt, max_tokens, False)
    try:
        payload = json.loads(read_body(response))
        elapsed = time.perf_counter() - started
        choice = payload["choices"][0]
        text = choice["message"]["content"]
        result = result_metrics(text, elapsed)
        result.update({"id": payload["id"], "text": text,
                       "finish_reason": choice.get("finish_reason")})
        return result
    finally:
        timer.cancel()
        close_connection(conn)


def post_abort(port, request_id):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=REQUEST_TIMEOUT_S)
    timer = arm_timeout(conn)
    try:
        conn.request("POST", "/abort", json.dumps({"request_id": request_id}),
                     {"Content-Type": "application/json"})
        response = conn.getresponse()
        data = read_body(response)
        require(response.status == 200, f"abort HTTP {response.status}: {data!r}")
        require(json.loads(data).get("success") is True, "abort was not acknowledged")
    finally:
        timer.cancel()
        close_connection(conn)


def sse_events(response):
    for _ in range(SSE_EVENT_CAP):
        line = response.readline(SSE_LINE_CAP_BYTES + 1)
        require(len(line) <= SSE_LINE_CAP_BYTES, "SSE line exceeded byte cap")
        if not line:
            return
        if not line.startswith(b"data: "):
            continue
        data = line[6:].strip()
        if data == b"[DONE]":
            yield None
            return
        yield json.loads(data)
    raise AssertionError("SSE response exceeded event cap")


def sse_request(port, prompt, max_tokens, abort_after_first=False,
                disconnect_after_first=False):
    conn, timer, response, started = open_chat(port, prompt, max_tokens, True)
    parts = []
    request_id = None
    stream_error = None
    ttft = None
    done = False
    disconnected = False
    try:
        for event in sse_events(response):
            if event is None:
                done = True
                break
            request_id = event.get("id", request_id)
            if "error" in event:
                stream_error = event["error"].get("message")
                continue
            text = event.get("choices", [{}])[0].get("delta", {}).get("content")
            if not text:
                continue
            if ttft is None:
                ttft = time.perf_counter() - started
            parts.append(text)
            if abort_after_first:
                require(request_id, "SSE response omitted request id")
                post_abort(port, request_id)
                abort_after_first = False
            if disconnect_after_first:
                disconnected = True
                break
        elapsed = time.perf_counter() - started
        require(ttft is not None, "SSE response contained no text")
        require(done or disconnected, "SSE response ended without [DONE]")
        result = result_metrics("".join(parts), elapsed, ttft)
        result.update({"id": request_id, "text": "".join(parts),
                       "stream_error": stream_error, "disconnected": disconnected})
        return result
    finally:
        timer.cancel()
        close_connection(conn)


def stage_records(path):
    """Read the stage's real perf log. A missing file is an empty log."""
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # A read can catch the line the stage is still appending.
            continue
    return records


def stage_events(path, name):
    """Only the records of one event kind. Start, load and completion differ."""
    return [record for record in stage_records(path)
            if record.get("event") == name]


class StageTail:
    """Incremental reader of the stage perf log.

    The sweep waits for one specific record before it acts, and the log grows
    to thousands of lines. Re-reading it per target would make the wait
    quadratic, so this keeps a byte offset.
    """

    def __init__(self, path):
        self.path = path
        self.offset = 0
        self.partial = b""
        self.records = []

    def poll(self):
        if not self.path.exists():
            return self.records
        with self.path.open("rb") as source:
            source.seek(self.offset)
            data = source.read()
            self.offset = source.tell()
        lines = (self.partial + data).split(b"\n")
        self.partial = lines.pop()
        for line in lines:
            if not line.strip():
                continue
            try:
                self.records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return self.records


def wait_enqueued(tail, request_id, timeout=5.0):
    """True when the stage queued this submit, False when it already started.

    The abort must remove a real queued submit. Posting it before the submit
    reaches the stage would leave only a tombstone, which proves nothing about
    pending removal.
    """
    deadline = time.monotonic() + timeout
    while True:
        records = tail.poll()
        if any(record.get("event") == "request_start"
               and record.get("request_id") == request_id for record in records):
            return False
        if any(record.get("event") == "queue_enqueue"
               and record.get("request_id") == request_id for record in records):
            return True
        require(time.monotonic() < deadline,
                f"the stage never queued target {request_id}")
        time.sleep(0.002)


def abort_removed(path, request_id):
    """The stage logged an abort that removed this queued submit."""
    found = [record for record in stage_events(path, "abort")
             if record.get("request_id") == request_id]
    require(found, f"the stage logged no abort for {request_id}")
    return found[-1].get("removed_pending") is True


def sse_abort_queued(port, prompt, max_tokens, tail=None):
    """Abort a stream by request id, before it produces any content.

    The server writes the first SSE event, which carries the id, as soon as the
    stream is set up. At that moment the stage is still busy with an earlier
    request. With `tail`, the abort is sent only after the stage logged that it
    queued the submit, so the abort removes a pending entry.
    """
    conn, timer, response, started = open_chat(port, prompt, max_tokens, True)
    aborted_id = None
    enqueued = None
    parts = []
    stream_error = None
    done = False
    try:
        for event in sse_events(response):
            if event is None:
                done = True
                break
            if "error" in event:
                stream_error = event["error"].get("message")
                continue
            request_id = event.get("id")
            text = event.get("choices", [{}])[0].get("delta", {}).get("content")
            if text:
                parts.append(text)
                continue
            if request_id and aborted_id is None:
                aborted_id = request_id
                enqueued = True if tail is None else wait_enqueued(tail, request_id)
                post_abort(port, request_id)
        return {"id": aborted_id, "text": "".join(parts), "done": done,
                "enqueued": enqueued, "stream_error": stream_error}
    finally:
        timer.cancel()
        close_connection(conn)


def sse_stream(port, prompt, max_tokens, first_token=None):
    """Run one real SSE stream to completion.

    `first_token` is set when the first content delta arrives. A caller uses it
    as proof that the stage is generating, without polling a server counter.
    """
    conn, timer, response, started = open_chat(port, prompt, max_tokens, True)
    parts = []
    request_id = None
    stream_error = None
    done = False
    ttft = None
    try:
        for event in sse_events(response):
            if event is None:
                done = True
                break
            request_id = event.get("id", request_id)
            if "error" in event:
                stream_error = event["error"].get("message")
                continue
            text = event.get("choices", [{}])[0].get("delta", {}).get("content")
            if not text:
                continue
            if ttft is None:
                ttft = time.perf_counter() - started
                if first_token is not None:
                    first_token.set()
            parts.append(text)
        elapsed = time.perf_counter() - started
        require(done, "the blocking SSE stream ended without [DONE]")
        require(stream_error is None, f"the blocking SSE stream failed: {stream_error!r}")
        require(request_id, "the blocking SSE stream carried no request id")
        result = result_metrics("".join(parts), elapsed, ttft)
        result.update({"id": request_id, "text": "".join(parts)})
        return result
    finally:
        timer.cancel()
        close_connection(conn)


def stage_ids(path, name):
    """The request ids of one event kind, in log order.

    A start record and a completion record are different facts. Keeping them
    apart lets a test state that an id entered the model and that it finished.
    """
    return [record.get("request_id") for record in stage_events(path, name)]


def aborted_pending(path):
    """Target ids whose abort record removed a real queued submit.

    A tombstone written before the submit arrived also cancels the request, but
    it does not prove that the stage held it in its queue. Only this set does.
    """
    return {record.get("request_id") for record in stage_events(path, "abort")
            if record.get("removed_pending") is True}


def health_in_flight(port):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=REQUEST_TIMEOUT_S)
    timer = arm_timeout(conn)
    try:
        conn.request("GET", "/health")
        response = conn.getresponse()
        data = read_body(response)
        require(response.status == 200, f"health HTTP {response.status}")
        return json.loads(data)["in_flight"]
    finally:
        timer.cancel()
        close_connection(conn)


def wait_for_disconnect(port):
    for _ in range(200):
        if health_in_flight(port) == 0:
            return
        time.sleep(0.05)
    raise AssertionError("disconnected SSE request remained in flight for 10 seconds")


def print_pass(name, result=None, detail=""):
    fields = ""
    if result is not None:
        ttft = result["ttft_s"]
        fields = (f" ttft_s={ttft:.6f}" if ttft is not None else " ttft_s=n/a")
        fields += (f" total_s={result['total_s']:.6f}"
                   f" tokens={result['tokens']}"
                   f" tokens_per_s={result['tokens_per_s']:.3f}")
    print(f"PASS {name}:{fields}{(' ' + detail) if detail else ''}", flush=True)


def spawn(name, args, cwd, env, processes, logs, readers, ready):
    proc = subprocess.Popen(args, cwd=cwd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    processes[name] = proc
    logs[name] = bytearray()

    def drain():
        while True:
            chunk = proc.stdout.readline(4096)
            if not chunk:
                break
            remaining = LOG_CAP_BYTES - len(logs[name])
            if remaining > 0:
                logs[name].extend(chunk[:remaining])
            if name == "stage" and b"ready" in chunk:
                ready.set()
        proc.stdout.close()

    thread = threading.Thread(target=drain, daemon=True)
    thread.start()
    readers.append(thread)


def wait_for_server(port, proc):
    for _ in range(400):
        require(proc.poll() is None, "server exited during startup")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise AssertionError("server was not ready within 20 seconds")


def wait_for_stage(proc, ready):
    for _ in range(STARTUP_TIMEOUT_S * 20):
        if ready.is_set():
            return
        require(proc.poll() is None, "MLX stage exited during startup")
        time.sleep(0.05)
    raise AssertionError(f"MLX stage was not ready within {STARTUP_TIMEOUT_S} seconds")


def stop_owned_processes(processes, readers):
    errors = []
    for proc in reversed(list(processes.values())):
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
    deadline = time.monotonic() + SHUTDOWN_TIMEOUT_S
    for name, proc in processes.items():
        try:
            proc.wait(max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(5)
            errors.append(f"{name} required SIGKILL")
        if proc.returncode != 0:
            errors.append(f"{name} exit code {proc.returncode}")
    for thread in readers:
        thread.join(2)
        if thread.is_alive():
            errors.append("process output reader did not exit")
    return errors


def run_smoke(server, stage, model, sweep_requests):
    global TOKENIZER
    TOKENIZER = Tokenizer.from_file(str(model / "tokenizer.json"))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                "TOKENIZERS_PARALLELISM": "false"})
    processes = {}
    logs = {}
    readers = []
    ready = threading.Event()
    succeeded = False
    try:
        with tempfile.TemporaryDirectory(prefix="umminfer-mlx-smoke-") as ipc_dir:
            entry = f"ipc://{ipc_dir}/entry"
            completion = f"ipc://{ipc_dir}/completion"
            abort_endpoint = f"ipc://{ipc_dir}/abort"
            spawn("server", [str(server), "serve", "--host", "127.0.0.1",
                             "--port", str(port), "--model", "mlx-smoke",
                             "--entry", "mlx", "--entry-endpoint", entry,
                             "--completion", completion, "--abort", abort_endpoint],
                  server.parent, env, processes, logs, readers, ready)
            wait_for_server(port, processes["server"])
            # The stage writes its real load and request records here. The test
            # uses them as evidence, not as a substitute for a request.
            perf_path = Path(ipc_dir) / "stage.jsonl"
            spawn("stage", [sys.executable, "-I", str(stage), "--name", "mlx",
                            "--recv", entry, "--coordinator", completion,
                            "--abort", abort_endpoint, "--model", str(model),
                            "--perf-log", str(perf_path)],
                  stage.parent, env, processes, logs, readers, ready)
            wait_for_stage(processes["stage"], ready)
            print("PASS startup: server and persistent MLX stage are ready", flush=True)

            prompt = "Give one concise Metal inference tip."
            json_result = json_request(port, prompt, 24)
            print_pass("json", json_result, f"request_id={json_result['id']}")
            sse_result = sse_request(port, prompt, 24)
            require(sse_result["stream_error"] is None, "normal SSE returned an error")
            require(sse_result["text"] == json_result["text"],
                    "JSON and SSE aggregates differ")
            print_pass("sse", sse_result, "aggregate matches JSON")

            abort_records_before = len(stage_events(perf_path, "request"))
            aborted = sse_request(
                port, "Count from 1 to 100, putting one number on each line.",
                MAX_GENERATION_TOKENS, abort_after_first=True)
            require(aborted["stream_error"] == "request aborted",
                    f"unexpected abort result: {aborted['stream_error']!r}")
            require(aborted["tokens"] < MAX_GENERATION_TOKENS,
                    "aborted request generated the full token cap")
            # An in-flight abort may write a start record, because the model
            # began. It must never write a completion record.
            time.sleep(0.3)
            completed_ids = [item.get("request_id")
                             for item in stage_events(perf_path, "request")]
            require(len(completed_ids) == abort_records_before,
                    "an aborted in-flight request wrote a stage completion record")
            require(aborted["id"] not in completed_ids,
                    "the aborted request wrote a completion record")
            print_pass("public_abort", aborted,
                       "HTTP 200, request aborted, no completion record")

            after_abort_marker = "QX7J9K2"
            after_abort = json_request(
                port, f"Repeat exactly: {after_abort_marker}", 16)
            require(after_abort_marker in after_abort["text"],
                    f"request after abort did not return its marker: {after_abort['text']!r}")
            print_pass("request_after_abort", after_abort)

            disconnected = sse_request(
                port, "Count from 1 to 100, putting one number on each line.",
                MAX_GENERATION_TOKENS, disconnect_after_first=True)
            require(disconnected["disconnected"], "SSE client did not disconnect")
            print_pass("sse_client_disconnect", disconnected)
            wait_for_disconnect(port)
            print("PASS disconnect_reaped: in_flight is zero", flush=True)
            after_disconnect_marker = "LK8V4N6"
            after_disconnect = json_request(
                port, f"Repeat exactly: {after_disconnect_marker}", 16)
            require(after_disconnect_marker in after_disconnect["text"],
                    "request after disconnect did not return its marker: "
                    f"{after_disconnect['text']!r}")
            print_pass("request_after_disconnect", after_disconnect)

            markers = [f"ZXQ734{index}" for index in range(4)]
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(json_request, port, f"Repeat exactly: {marker}", 16)
                           for marker in markers]
                deadline = time.monotonic() + REQUEST_TIMEOUT_S + 5
                concurrent = [future.result(max(0.1, deadline - time.monotonic()))
                              for future in futures]
            require(len({item["id"] for item in concurrent}) == 4,
                    "concurrent request ids were not unique")
            for marker, item in zip(markers, concurrent):
                require(marker in item["text"],
                        f"{marker} response omitted its marker: {item['text']!r}")
                for other in markers:
                    if other != marker:
                        require(other not in item["text"],
                                f"{marker} response contained {other}")
                print_pass(marker, item)
            print("PASS concurrent_no_cross_answer: four responses kept their markers",
                  flush=True)

            # Same prompt, same parameters, real model: the text must repeat.
            repeats = [json_request(port, "Repeat exactly: QM4T8", 8)
                       for _ in range(3)]
            require(len({item["text"] for item in repeats}) == 1,
                    f"repeated requests differed: {[i['text'] for i in repeats]!r}")
            require(all(item["finish_reason"] in ("stop", "length")
                        for item in repeats),
                    f"repeated requests reported {[i['finish_reason'] for i in repeats]!r}")
            print(f"PASS repeat_identical text={repeats[0]['text']!r} runs=3 "
                  f"finish={repeats[0]['finish_reason']}", flush=True)

            # A request that reaches its token budget must report `length`, and
            # the stage log must carry the same reason and the real count.
            long_prompt = "Count from 1 to 100, putting one number on each line."
            capped = json_request(port, long_prompt, 4)
            require(capped["finish_reason"] == "length",
                    f"a capped request reported {capped['finish_reason']!r}")
            capped_record = stage_events(perf_path, "request")[-1]
            require(capped_record.get("event") == "request",
                    "the capped request wrote no stage record")
            require(capped_record.get("finish_reason") == "length",
                    f"the stage reported {capped_record.get('finish_reason')!r}")
            require(capped_record.get("generation_tokens") == 4,
                    f"the stage counted {capped_record.get('generation_tokens')} tokens")
            print(f"PASS finish_reason_length http={capped['finish_reason']} "
                  f"stage={capped_record['finish_reason']} "
                  f"tokens={capped_record['generation_tokens']}", flush=True)

            # Queued abort. One request is generating; the next is still queued
            # at the stage. Its abort must drop the submit before any model
            # work, and the running request must finish normally. The stage
            # writes `request_start` only after the abort check, so the queued
            # target must have neither a start nor a completion record, and the
            # runner must have exactly one of each.
            starts_before = set(stage_ids(perf_path, "request_start"))
            completions_before = set(stage_ids(perf_path, "request"))
            tail = StageTail(perf_path)
            long_prompt = "Count from 1 to 100, putting one number on each line."
            with ThreadPoolExecutor(max_workers=2) as pool:
                started = threading.Event()
                running = pool.submit(sse_stream, port, long_prompt,
                                      MAX_GENERATION_TOKENS, started)
                require(started.wait(10),
                        "the blocking request emitted no token")
                queued = sse_abort_queued(port, long_prompt,
                                          MAX_GENERATION_TOKENS, tail)
                require(queued["stream_error"] == "request aborted",
                        f"queued abort gave {queued['stream_error']!r}")
                require(queued["text"] == "",
                        f"queued request generated text: {queued['text']!r}")
                require(queued["id"], "the queued abort had no request id")
                require(queued["enqueued"] is True,
                        "the abort was sent before the stage queued the target")
                running_result = running.result(REQUEST_TIMEOUT_S)
            require(running_result["tokens"] > 1,
                    f"the running request produced {running_result['tokens']} tokens")
            print(f"PASS queued_abort id={queued['id']} "
                  "no_text_before_abort", flush=True)

            deadline = time.monotonic() + 10
            while (time.monotonic() < deadline and
                   running_result["id"] not in stage_ids(perf_path, "request")):
                time.sleep(0.02)
            time.sleep(0.3)
            new_starts = set(stage_ids(perf_path, "request_start")) - starts_before
            new_completions = set(stage_ids(perf_path, "request")) - completions_before
            aborts = [item for item in stage_events(perf_path, "abort")
                      if item.get("request_id") == queued["id"]]
            require(aborts, "the queued target wrote no abort record")
            # The client sends the abort only after the stage logged the
            # enqueue, so the first abort must remove the pending submit.
            # A repeat abort (stream teardown or a second broadcast) finds an
            # empty queue and can only report false. The full sequence is
            # checked, not just the last record.
            sequence = [item.get("removed_pending") for item in aborts]
            require(sequence[0] is True,
                    "the first abort did not remove the target from the stage "
                    f"queue: removed_pending={sequence}")
            require(all(value is False for value in sequence[1:]),
                    "a repeat abort reported a removal: "
                    f"removed_pending={sequence}")
            require(queued["id"] not in new_starts,
                    "the queued target entered the model")
            require(queued["id"] not in new_completions,
                    "the queued target wrote a completion record")
            require(new_starts == {running_result["id"]},
                    f"the pair wrote start records for {sorted(new_starts)}")
            require(new_completions == {running_result["id"]},
                    f"the pair wrote completion records for {sorted(new_completions)}")
            print("PASS queued_abort_reached_model=false "
                  f"runner={running_result['id']} starts=1 requests=1 "
                  f"removed_pending={sequence}", flush=True)

            queued_marker = "AB9K2X"
            after_queued = json_request(port, f"Repeat exactly: {queued_marker}", 8)
            require(queued_marker in after_queued["text"],
                    f"request after queued abort: {after_queued['text']!r}")
            print_pass("request_after_queued_abort", after_queued)

            # More than 1024 distinct request ids, each cancelled through the
            # real /abort while the stage is busy with a real blocking request.
            # The evidence is the stage's own `request_start` record, which is
            # written after the abort check and before the first model step: an
            # id with no start record never entered the model. The coordinator
            # broadcasts aborts on a PUB socket, so an abort has no order with
            # the submit. A target that the stage dequeued first does enter the
            # model; the sweep counts that target as not proven, sends a fresh
            # id, and still ends with more than 1024 proven ids. Each blocking
            # runner must have exactly one start and one completion.
            targets = []
            attempted_ids = []
            runner_ids = []
            starts_before = set(stage_ids(perf_path, "request_start"))
            completions_before = set(stage_ids(perf_path, "request"))
            blocking_requests = 0
            attempts = 0
            sweep_started = time.perf_counter()
            sweep_deadline = sweep_started + 900
            # One target per runner, and the target is sent only after the
            # runner emitted a token, so an abort that does arrive in time has
            # the rest of a generation, about 300 ms here, to reach the stage.
            pool = ThreadPoolExecutor(max_workers=1)
            try:
                while True:
                    while len(targets) < sweep_requests:
                        require(time.monotonic() < sweep_deadline,
                                f"the abort sweep stalled at {len(targets)} "
                                "proven targets")
                        started = threading.Event()
                        runner = pool.submit(sse_stream, port, long_prompt,
                                             MAX_GENERATION_TOKENS, started)
                        require(started.wait(10),
                                "the blocking request did not emit a token")
                        blocking_requests += 1
                        item = sse_abort_queued(port, long_prompt,
                                                MAX_GENERATION_TOKENS, tail)
                        attempts += 1
                        require(item["stream_error"] == "request aborted",
                                f"abort sweep got {item['stream_error']!r}")
                        require(item["done"], "a cancelled stream ended without [DONE]")
                        require(item["id"], "an abort target had no request id")
                        attempted_ids.append(item["id"])
                        targets.append(item["id"])
                        runner_result = runner.result(REQUEST_TIMEOUT_S)
                        require(runner_result["tokens"] > 1,
                                "the blocking request produced no tokens")
                        runner_ids.append(runner_result["id"])
                        if blocking_requests % 10 == 0:
                            print(f"PROGRESS abort_sweep attempts={attempts} "
                                  f"targets={len(targets)} "
                                  f"runners={blocking_requests}", flush=True)
                    time.sleep(0.3)
                    new_starts = set(stage_ids(perf_path, "request_start")) - starts_before
                    enqueued = set(stage_ids(perf_path, "queue_enqueue"))
                    removed = {record.get("request_id")
                               for record in stage_events(perf_path, "abort")
                               if record.get("removed_pending") is True}
                    # Proven means the stage queued the submit and the abort
                    # removed that entry, not that an early tombstone matched.
                    proven = [item for item in targets
                              if item not in new_starts
                              and item in enqueued and item in removed]
                    if len(proven) >= sweep_requests:
                        break
                    targets = proven
            finally:
                pool.shutdown(wait=True)

            new_completions = set(stage_ids(perf_path, "request")) - completions_before
            target_set = set(targets)
            raced_ids = set(attempted_ids) - target_set
            require(len(attempted_ids) == attempts,
                    f"{len(attempted_ids)} attempted ids for {attempts} attempts")
            require(len(set(attempted_ids)) == len(attempted_ids),
                    "attempted target ids were not distinct")
            require(len(targets) >= sweep_requests,
                    f"the sweep proved {len(targets)} cancelled ids")
            require(len(target_set) == len(targets),
                    "proven target ids were not distinct")
            require(target_set.isdisjoint(new_starts),
                    f"proven ids entered the model: "
                    f"{sorted(target_set & new_starts)[:5]}")
            require(target_set.isdisjoint(new_completions),
                    f"proven ids wrote completion records: "
                    f"{sorted(target_set & new_completions)[:5]}")
            require(target_set.isdisjoint(runner_ids),
                    f"a proven target was also a runner: "
                    f"{sorted(target_set & set(runner_ids))[:5]}")
            require(raced_ids.isdisjoint(runner_ids),
                    f"a raced target was also a runner: "
                    f"{sorted(raced_ids & set(runner_ids))[:5]}")
            # A raced target was dequeued before its abort arrived, so it has a
            # start record and neither a completion nor a role as a runner.
            require(raced_ids <= new_starts,
                    f"raced ids without a start record: "
                    f"{sorted(raced_ids - new_starts)[:5]}")
            require(new_starts == set(runner_ids) | raced_ids,
                    f"start records: expected {len(runner_ids)} runners and "
                    f"{len(raced_ids)} raced targets, got {len(new_starts)} ids")
            # Every runner finishes. A raced target may also finish late, but no
            # other id may write a completion record.
            require(set(runner_ids) <= new_completions,
                    f"{len(set(runner_ids) - new_completions)} runners wrote no "
                    "completion record")
            require(new_completions - set(runner_ids) <= raced_ids,
                    f"unexpected completion records: "
                    f"{sorted(new_completions - set(runner_ids) - raced_ids)[:5]}")
            print(f"PASS abort_sweep aborted={len(targets)} distinct={len(target_set)} "
                  f"attempts={attempts} raced={len(raced_ids)} "
                  f"blocking_requests={blocking_requests} "
                  f"runner_starts={len(set(runner_ids))} "
                  f"runner_requests={len(new_completions)} "
                  f"proven_starts=0 proven_requests=0 "
                  f"first={targets[0]} last={targets[-1]} "
                  f"seconds={time.perf_counter() - sweep_started:.1f}", flush=True)

            sweep_marker = "K7M3Q9"
            after_sweep = json_request(port, f"Repeat exactly: {sweep_marker}", 8)
            require(sweep_marker in after_sweep["text"],
                    f"request after the abort sweep: {after_sweep['text']!r}")
            print_pass("request_after_abort_sweep", after_sweep)

            records = stage_records(perf_path)
            loads = [item for item in records if item.get("event") == "load"]
            requests = [item for item in records if item.get("event") == "request"]
            require(len(loads) == 1, f"the stage loaded {len(loads)} times")
            require(loads[0].get("vocab_builds") == 1,
                    f"load built the vocabulary {loads[0].get('vocab_builds')} times")
            # The cache is filled at load and read only when a request runs.
            require(loads[0].get("vocab_accesses") == 0,
                    f"load accessed the vocabulary {loads[0].get('vocab_accesses')} times")
            builds = {item.get("vocab_builds") for item in requests}
            require(builds == {1}, f"requests rebuilt the vocabulary: {builds}")
            accesses = [item.get("vocab_accesses") for item in requests]
            require(all(isinstance(value, int) for value in accesses),
                    "a request record omitted the vocabulary counters")
            require(all(left <= right for left, right in zip(accesses, accesses[1:])),
                    "vocabulary accesses did not grow with the requests")
            require(accesses[-1] > accesses[0],
                    "later requests did not consult the vocabulary cache")
            # Per request, not just the end of a global counter: the entry
            # record and the completion record of one generation must show one
            # build and a strict increase in cache reads.
            starts = {item.get("request_id"): item
                      for item in stage_events(perf_path, "request_start")}
            completions = {item.get("request_id"): item
                           for item in requests}
            shared = sorted(set(starts) & set(completions))
            require(shared, "no request has both a start and a completion record")
            for request_id in shared:
                require(starts[request_id].get("vocab_builds") == 1,
                        f"{request_id} built the vocabulary "
                        f"{starts[request_id].get('vocab_builds')} times")
                require(starts[request_id].get("vocab_accesses")
                        < completions[request_id].get("vocab_accesses"),
                        f"{request_id} did not read the cache during generation")
            print(f"PASS vocab_per_request checked={len(shared)} "
                  "builds=1 accesses_strictly_increase", flush=True)
            print(f"PASS vocab_cache load_builds={loads[0]['vocab_builds']} "
                  f"load_accesses={loads[0]['vocab_accesses']} "
                  f"request_builds={sorted(builds)} "
                  f"accesses={accesses[0]}..{accesses[-1]} "
                  f"requests={len(requests)}", flush=True)

            metrics = {"json": {key: json_result[key]
                                for key in ("total_s", "tokens", "tokens_per_s")},
                       "sse": {key: sse_result[key]
                               for key in ("ttft_s", "total_s", "tokens", "tokens_per_s")}}
            print("METRICS " + json.dumps(metrics, sort_keys=True), flush=True)
            succeeded = True
    finally:
        errors = stop_owned_processes(processes, readers)
        if processes and not errors:
            codes = {name: proc.returncode for name, proc in processes.items()}
            print(f"PASS cleanup: owned processes reaped, exit_codes={codes}", flush=True)
        for error in errors:
            print(f"FAIL cleanup: {error}", file=sys.stderr, flush=True)
        if not succeeded:
            for name, data in logs.items():
                text = bytes(data).decode(errors="replace")
                print(f"{name} log tail:\n{text[-4000:]}", file=sys.stderr, flush=True)
        if succeeded and errors:
            raise RuntimeError("; ".join(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True, type=Path)
    parser.add_argument("--stage", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--sweep-requests", type=int, default=1025,
                        help="distinct real request ids to cancel; each must "
                             "never enter the model")
    args = parser.parse_args()
    server = args.server.resolve()
    stage = args.stage.resolve()
    model = args.model.resolve()
    require(server.is_file(), f"server binary not found: {server}")
    require(stage.is_file(), f"stage script not found: {stage}")
    require(model.is_dir(), f"local model directory not found: {model}")
    require((model / "tokenizer.json").is_file(), "model has no tokenizer.json")
    require((model / "model.safetensors").is_file(), "model has no model.safetensors")
    require(args.sweep_requests > 1024,
            "--sweep-requests must exceed 1024 distinct cancelled request ids")
    run_smoke(server, stage, model, args.sweep_requests)


if __name__ == "__main__":
    main()

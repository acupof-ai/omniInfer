#!/usr/bin/env python3
"""HTTP and process smoke checks for a built omni-infer binary."""

import concurrent.futures
import http.client
import json
import os
import select
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time

API_KEY = "test-key"


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def request(port, payload=None, path="/v1/chat/completions", auth=True):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
    headers = {"Content-Type": "application/json"}
    if auth:
        headers["Authorization"] = f"Bearer {API_KEY}"
    body = None if payload is None else json.dumps(payload)
    conn.request("GET" if payload is None else "POST", path, body=body, headers=headers)
    response = conn.getresponse()
    data = response.read().decode()
    status = response.status
    conn.close()
    return status, data


def chat(port, text, stream=False):
    return request(port, {"messages": [{"role": "user", "content": text}], "stream": stream})


def wait_ready(process, port):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(f"server exited early with {process.returncode}")
        try:
            if request(port, path="/health")[0] == 200:
                return
        except OSError:
            pass
        time.sleep(0.05)
    raise AssertionError("server did not become ready")


def stop_process(process):
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
    code = process.wait(timeout=10)
    if code != 0:
        raise AssertionError(f"process exited with {code}")


def check_demo(binary, root, log):
    port = free_port()
    ipc_dir = os.path.join(root, "demo")
    process = subprocess.Popen(
        [binary, "serve", "--demo", "--host", "127.0.0.1", "--port", str(port),
         "--ipc-dir", ipc_dir, "--api-key", API_KEY],
        stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_ready(process, port)
        assert request(port, path="/health", auth=False)[0] == 401
        assert request(port, path="/health")[0] == 200

        invalid = [
            {"messages": {}},
            {"messages": [], "stream": False},
            {"messages": [{"role": "user", "content": "x"}], "stream": "yes"},
            {"messages": [{"role": "user", "content": "x"}], "max_tokens": "ten"},
            {"messages": [{"role": "user", "content": "x"}], "max_tokens": 0},
            {"messages": [{"role": "user", "content": "x"}], "temperature": -0.01},
            {"messages": [{"role": "user", "content": "x"}], "temperature": 2.01},
            {"messages": [{"role": "user", "content": "x",
                           "n": 18446744073709551615}]},
        ]
        for payload in invalid:
            status, body = request(port, payload)
            assert status == 400, (status, body)
            assert json.loads(body)["error"]["type"] == "invalid_request"

        status, body = chat(port, "single")
        assert status == 200, body
        assert json.loads(body)["choices"][0]["message"]["content"] == "echo: single"

        status, body = chat(port, "stream", stream=True)
        assert status == 200, body
        assert body.count('"content":"echo: stream"') == 1, body
        assert '"finish_reason":"stop"' in body, body
        assert body.endswith("data: [DONE]\n\n"), body
        assert '"error"' not in body, body

        with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
            results = list(pool.map(lambda i: chat(port, f"n{i}"), range(24)))
        for i, (status, body) in enumerate(results):
            assert status == 200, body
            assert json.loads(body)["choices"][0]["message"]["content"] == f"echo: n{i}"

        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            streams = list(pool.map(lambda i: chat(port, f"s{i}", True), range(12)))
        for i, (status, body) in enumerate(streams):
            assert status == 200, body
            assert body.count(f'"content":"echo: s{i}"') == 1, body
            assert body.endswith("data: [DONE]\n\n"), body

        status, body = request(port, path="/health")
        assert status == 200 and json.loads(body)["in_flight"] == 0, body
        stop_process(process)
        print("demo: validation, auth, 24 JSON, 12 SSE, SIGTERM passed")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def check_custom_entry(binary, root, log):
    port = free_port()
    ipc_dir = os.path.join(root, "custom")
    entry = f"ipc://{ipc_dir}/entry"
    completion = f"ipc://{ipc_dir}/completion"
    abort = f"ipc://{ipc_dir}/abort"
    stage = subprocess.Popen(
        [binary, "stage", "--name", "custom", "--recv", entry,
         "--coordinator", completion, "--abort", abort],
        stdout=log, stderr=subprocess.STDOUT)
    server = None
    try:
        server = subprocess.Popen(
            [binary, "serve", "--host", "127.0.0.1", "--port", str(port),
             "--entry", "custom", "--entry-endpoint", entry,
             "--completion", completion, "--abort", abort, "--api-key", API_KEY],
            stdout=log, stderr=subprocess.STDOUT)
        wait_ready(server, port)
        status, body = chat(port, "custom")
        assert status == 200, body
        assert json.loads(body)["choices"][0]["message"]["content"] == "echo: custom"
        stop_process(server)
        stop_process(stage)
        print("external stage: custom entry and SIGTERM passed")
    finally:
        for process in (server, stage):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()


def wait_in_flight(port, expected, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status, body = request(port, path="/health")
        if status == 200 and json.loads(body)["in_flight"] == expected:
            return
        time.sleep(0.02)
    raise AssertionError(f"in_flight did not reach {expected}")


def check_disconnect_and_shutdown(binary, root, log):
    port = free_port()
    ipc_dir = os.path.join(root, "stalled")
    entry = f"ipc://{ipc_dir}/entry"
    process = subprocess.Popen(
        [binary, "serve", "--host", "127.0.0.1", "--port", str(port),
         "--entry-endpoint", entry, "--completion", f"ipc://{ipc_dir}/completion",
         "--abort", f"ipc://{ipc_dir}/abort", "--api-key", API_KEY],
        stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_ready(process, port)
        body = json.dumps(
            {"messages": [{"role": "user", "content": "rst"}], "stream": True})
        for _ in range(3):
            with socket.create_connection(("127.0.0.1", port), timeout=1) as sock:
                sock.sendall(
                    (f"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                     f"Authorization: Bearer {API_KEY}\r\nContent-Type: application/json\r\n"
                     f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n{body}")
                    .encode())
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                                struct.pack("ii", 1, 0))
        wait_in_flight(port, 0)

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
        conn.request("POST", "/v1/chat/completions", body=body,
                     headers={"Content-Type": "application/json",
                              "Authorization": f"Bearer {API_KEY}"})
        response = conn.getresponse()
        assert response.status == 200
        assert response.readline().startswith(b"data: ")
        response.close()
        conn.close()
        wait_in_flight(port, 0)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(chat, port, "stalled")
            wait_in_flight(port, 1)
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=2) == 0
            try:
                pending.result(timeout=3)
            except (OSError, http.client.HTTPException):
                pass
        print("disconnect: RST, idle SSE, active-request SIGTERM passed")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def check_startup_shutdown(binary, root):
    for iteration in range(10):
        ipc_dir = os.path.join(root, f"startup-{iteration}")
        process = subprocess.Popen(
            [binary, "serve", "--host", "127.0.0.1", "--port", "0",
             "--entry-endpoint", f"ipc://{ipc_dir}/entry",
             "--completion", f"ipc://{ipc_dir}/completion",
             "--abort", f"ipc://{ipc_dir}/abort", "--api-key", API_KEY],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            ready, _, _ = select.select([process.stdout], [], [], 2)
            assert ready, "server did not announce startup"
            line = process.stdout.readline()
            assert "serving on" in line, line
            process.send_signal(signal.SIGTERM)
            code = process.wait(timeout=2)
            assert code == 0, (iteration, code)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()
    print("startup: 10 SIGTERM races passed")


def check_cli_errors(binary, root):
    ipc_dir = os.path.join(root, "cli")
    normal = [binary, "serve", "--entry-endpoint", f"ipc://{ipc_dir}/entry",
              "--completion", f"ipc://{ipc_dir}/completion",
              "--abort", f"ipc://{ipc_dir}/abort"]
    for command in ([binary, "serve", "--demo", "--ipc-dir", ipc_dir], normal):
        result = subprocess.run(command + ["--port", "invalid"], capture_output=True,
                                text=True, timeout=3)
        assert result.returncode == 1, result
        assert "invalid --port" in result.stderr, result.stderr
        assert "terminating" not in result.stderr, result.stderr

    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        port = occupied.getsockname()[1]
        for command in ([binary, "serve", "--demo", "--ipc-dir", ipc_dir], normal):
            result = subprocess.run(
                command + ["--host", "127.0.0.1", "--port", str(port)],
                capture_output=True, text=True, timeout=3)
            assert result.returncode == 1, result
            assert "failed to bind" in result.stderr, result.stderr
            assert "serving on" not in result.stdout, result.stdout
    print("CLI: invalid port and occupied port passed")


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: http_smoke.py /path/to/omni-infer")
    binary = os.path.abspath(sys.argv[1])
    with tempfile.TemporaryDirectory(prefix="omni-http-") as root:
        with open(os.path.join(root, "processes.log"), "w") as log:
            check_demo(binary, root, log)
            check_custom_entry(binary, root, log)
            check_disconnect_and_shutdown(binary, root, log)
            check_startup_shutdown(binary, root)
            check_cli_errors(binary, root)


if __name__ == "__main__":
    main()

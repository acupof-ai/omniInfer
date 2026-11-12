#!/usr/bin/env python3
"""Adversarial regression: an ambient non-API probe must not change a proxy case.

A localhost port scanner sends `GET / HTTP/1.1` to every listening port. Both
byte-tap proxies in the E2E harnesses allocate behavior per accepted connection,
so before the fix such a probe consumed a plan slot (`ByteTapProxy`) or bound the
real WebSocket session (`WireProxy`). These tests pin both invariants without a
model, a server stage or the MLX stack:

  1. `ByteTapProxy` ignores non-POST connections and keeps its plan aligned.
  2. `reset_after_request` resets before the response can reach the client even
     when the upstream answers immediately.
  3. `WireProxy` ignores a non-WebSocket handshake and binds the real one.

`LegacyProxy` and `LegacyAccept` below reproduce the old behaviour so the test
also proves the defect it guards against still has teeth.
"""

import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent

PROBE = (b"GET / HTTP/1.1\r\nHost: localhost:1\r\nUser-Agent: port-scan\r\n"
         b"Accept: text/html\r\nConnection: close\r\n\r\n")
UPGRADE = (b"GET /v1/realtime?intent=transcription&model=tiny-ctc HTTP/1.1\r\n"
           b"Host: 127.0.0.1\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
           b"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
           b"Sec-WebSocket-Version: 13\r\n\r\n")
REPLY = json.dumps({"model": "m",
                    "choices": [{"message": {"content": "ok"},
                                 "finish_reason": "stop"}]}).encode()


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def start_upstream(stop, reset_first=0, requests=None):
    """Real HTTP server. Answers every request; optionally resets the first N."""
    import struct

    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(32)
    server.settimeout(0.05)
    port = server.getsockname()[1]
    seen = {"count": 0}

    def handle(conn):
        try:
            conn.settimeout(5.0)
            head = b""
            while b"\r\n\r\n" not in head:
                chunk = conn.recv(65536)
                if not chunk:
                    return
                head += chunk
            length = 0
            for line in head.split(b"\r\n")[1:]:
                if line.lower().startswith(b"content-length:"):
                    length = int(line.split(b":", 1)[1])
            body = head.split(b"\r\n\r\n", 1)[1]
            while len(body) < length:
                chunk = conn.recv(65536)
                if not chunk:
                    return
                body += chunk
            with threading.Lock():
                seen["count"] += 1
                index = seen["count"]
            if requests is not None:
                requests.append(head.split(b"\r\n", 1)[0])
            if index <= reset_first:
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                                struct.pack("ii", 1, 0))
                conn.close()
                return
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                         b"Content-Length: %d\r\n\r\n" % len(REPLY) + REPLY)
            time.sleep(0.2)
        except OSError:
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def loop():
        while not stop.is_set():
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            threading.Thread(target=handle, args=(conn,), daemon=True).start()

    threading.Thread(target=loop, daemon=True).start()
    return port, server


class LegacyProxy:
    """The pre-fix accept loop: any TCP connection consumes a plan slot."""

    def __init__(self, patched, upstream_port):
        self.inner = patched.ByteTapProxy(upstream_port)
        # The constructor builds the accept thread but does not start it, so
        # replacing the target here yields the old behaviour exactly.
        self.inner._thread = threading.Thread(target=self._legacy_accept_loop,
                                              daemon=True)

    def _legacy_accept_loop(self):
        proxy = self.inner
        while not proxy._stop.is_set():
            try:
                client, _ = proxy.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with proxy._lock:
                index = len(proxy.records)
                mode = (proxy.plans[index] if index < len(proxy.plans) else "pass")
                record = {"index": index, "mode": mode, "request_bytes": 0,
                          "response_bytes": 0, "delta_seen": False,
                          "result": "pending"}
                proxy.records.append(record)
            threading.Thread(target=proxy._handle, args=(client, record),
                             daemon=True).start()

    def __getattr__(self, name):
        return getattr(self.inner, name)


def send_probe(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2) as conn:
            conn.sendall(PROBE)
            conn.settimeout(2)
            try:
                conn.recv(256)
            except OSError:
                pass
    except OSError:
        pass


def chat_request(port, cli):
    command = [cli, "chat", "--url", f"http://127.0.0.1:{port}/v1",
               "--model", "m", "--prompt", "hi", "--max-tokens", "4",
               "--timeout-ms", "8000", "--max-attempts", "3"]
    return subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=40)


def test_byte_tap_ignores_probes(retry, cli):
    """Plan stays aligned and every real attempt is reset."""
    stop = threading.Event()
    upstream_port, server = start_upstream(stop)
    proxy = retry.ByteTapProxy(upstream_port).plan(
        "reset_after_request", "reset_after_request", "reset_after_request").start()
    try:
        send_probe(proxy.port)
        send_probe(proxy.port)
        time.sleep(0.05)
        result = chat_request(proxy.port, cli)
        require(result.returncode == 1,
                f"probe shifted the plan: exit {result.returncode} "
                f"{result.stdout!r} {result.stderr!r}")
        require(len(proxy.records) == 3,
                f"expected 3 real attempts, saw {len(proxy.records)}: {proxy.records}")
        require(all(item["mode"] == "reset_after_request" for item in proxy.records),
                f"a non-POST probe consumed a plan slot: {proxy.records}")
        require(all(item["result"] == "reset" for item in proxy.records),
                f"an attempt was not reset: {proxy.records}")
    finally:
        proxy.close()
        server.close()
        stop.set()
    print("PASS probe_does_not_consume_plan attempts=3 probes=2", flush=True)


def test_legacy_proxy_still_fails(retry, cli):
    """The pre-fix behaviour must still reproduce the defect."""
    stop = threading.Event()
    upstream_port, server = start_upstream(stop)
    legacy = LegacyProxy(retry, upstream_port)
    legacy.inner.plan("reset_after_request", "reset_after_request",
                      "reset_after_request", "pass")
    legacy.start()
    try:
        send_probe(legacy.port)
        time.sleep(0.05)
        result = chat_request(legacy.port, cli)
        require(result.returncode == 0,
                "legacy proxy did not reproduce the defect; the regression "
                "guard no longer proves anything")
        require(len(legacy.records) == 4,
                f"legacy proxy recorded {len(legacy.records)} connections, "
                f"expected 4 (probe plus three attempts)")
    finally:
        legacy.close()
        server.close()
        stop.set()
    print("PASS legacy_proxy_reproduces_defect exit=0 connections=4", flush=True)


def test_reset_beats_fast_response(retry, cli):
    """reset_after_request must reset even when upstream answers immediately."""
    for attempt in range(8):
        stop = threading.Event()
        requests = []
        upstream_port, server = start_upstream(stop, requests=requests)
        proxy = retry.ByteTapProxy(upstream_port).plan(
            "reset_after_request", "reset_after_request",
            "reset_after_request").start()
        try:
            result = chat_request(proxy.port, cli)
            require(result.returncode == 1,
                    f"trial {attempt}: response beat the reset: "
                    f"{result.stdout!r} {result.stderr!r}")
            require(len(proxy.records) == 3,
                    f"trial {attempt}: {len(proxy.records)} connections, "
                    f"expected 3")
            require(all(item["result"] == "reset" for item in proxy.records),
                    f"trial {attempt}: {proxy.records}")
        finally:
            proxy.close()
            server.close()
            stop.set()
    print("PASS reset_precedes_fast_response trials=8 attempts=3", flush=True)


def legacy_wire_accept(rt, upstream_port, probes):
    """The pre-fix accept path: the first TCP connection binds the session."""
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(0.1)
    port = listener.getsockname()[1]

    def loop():
        for _ in range(200):
            try:
                client, _ = listener.accept()
                break
            except socket.timeout:
                continue
        else:
            return
        probes.append(client)

    threading.Thread(target=loop, daemon=True).start()
    return port, listener


def test_legacy_wire_proxy_still_fails(rt):
    """The pre-fix accept must still let a probe steal the session."""
    probes = []
    port, listener = legacy_wire_accept(rt, 1, probes)
    try:
        send_probe(port)
        time.sleep(0.3)
        require(len(probes) == 1,
                "legacy accept did not bind the probe; the regression guard "
                "no longer proves anything")
    finally:
        for connection in probes:
            try:
                connection.close()
            except OSError:
                pass
        listener.close()
    print("PASS legacy_wire_proxy_reproduces_defect probe_bound_session=1",
          flush=True)


def test_wire_proxy_ignores_non_websocket(rt):
    """A probe must not bind the session; the real upgrade still is forwarded."""
    stop = threading.Event()
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(8)
    server.settimeout(0.05)
    upstream_port = server.getsockname()[1]
    received = []

    def loop():
        while not stop.is_set():
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            conn.settimeout(2)
            head = b""
            try:
                while b"\r\n\r\n" not in head:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    head += chunk
            except OSError:
                pass
            received.append(head)
            try:
                conn.sendall(b"HTTP/1.1 101 Switching Protocols\r\n"
                             b"Upgrade: websocket\r\nConnection: Upgrade\r\n"
                             b"Sec-WebSocket-Accept: x\r\n\r\n")
            except OSError:
                pass
            conn.close()

    threading.Thread(target=loop, daemon=True).start()
    proxy = rt.WireProxy(upstream_port)
    try:
        send_probe(proxy.port)
        time.sleep(0.2)
        require(not proxy.connections,
                "a non-WebSocket probe bound the real session")
        with socket.create_connection(("127.0.0.1", proxy.port), timeout=2) as conn:
            conn.sendall(UPGRADE)
            conn.settimeout(2)
            try:
                conn.recv(256)
            except OSError:
                pass
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not received:
            time.sleep(0.02)
        require(received, "the real upgrade never reached the upstream server")
        require(received[0].startswith(b"GET /v1/realtime"),
                f"upstream saw {received[0][:60]!r}, not the real handshake")
        require(len(received) == 1,
                f"upstream accepted {len(received)} connections, expected 1")
    finally:
        proxy.stopping.set()
        try:
            proxy.listener.close()
        except OSError:
            pass
        for connection in proxy.connections:
            try:
                connection.close()
            except OSError:
                pass
        server.close()
        stop.set()
    print("PASS wire_proxy_ignores_non_websocket probes=1 real_sessions=1", flush=True)


def main():
    parser_args = sys.argv[1:]
    if not parser_args:
        print("usage: proxy_isolation_test.py <umm-infer binary>", file=sys.stderr)
        return 2
    cli = str(Path(parser_args[0]).resolve())
    retry = load("client_retry_frozen", "client_retry_e2e.py")
    rt = load("realtime_client_contract", "realtime_client_contract_e2e.py")
    test_legacy_wire_proxy_still_fails(rt)
    test_wire_proxy_ignores_non_websocket(rt)
    test_byte_tap_ignores_probes(retry, cli)
    test_legacy_proxy_still_fails(retry, cli)
    test_reset_beats_fast_response(retry, cli)
    print("PASS proxy_isolation_test", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

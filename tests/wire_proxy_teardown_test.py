#!/usr/bin/env python3
"""Unit checks for WireProxy's transport-close classification.

The network part of the regression is the 20-round normal-close loop in
tests/realtime_client_contract_e2e.py, which uses the real server. This file
pins the classifier itself and needs no server, model, or socket.
"""

import errno
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from realtime_client_contract_e2e import WireProxy  # noqa: E402


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def proxy_shaped():
    proxy = object.__new__(WireProxy)
    proxy.close_seen = threading.Event()
    proxy.eof_seen = threading.Event()
    return proxy


def main():
    proxy = proxy_shaped()
    closed = OSError(errno.ENOTCONN, "not connected")
    broken = OSError(errno.EPIPE, "broken pipe")
    reset = OSError(errno.ECONNRESET, "reset")
    other = OSError(errno.EACCES, "denied")

    require(proxy.expected_close_error(closed),
            "ENOTCONN after a local shutdown is teardown")
    require(proxy.expected_close_error(broken),
            "EPIPE after a local shutdown is teardown")
    require(not proxy.expected_close_error(reset),
            "ECONNRESET before any close evidence is a real fault")
    proxy.close_seen.set()
    require(proxy.expected_close_error(reset),
            "ECONNRESET after a WebSocket Close frame is teardown")
    proxy.close_seen.clear()
    proxy.eof_seen.set()
    require(proxy.expected_close_error(reset),
            "ECONNRESET after an observed EOF is teardown")
    require(not proxy.expected_close_error(other),
            "an unrelated OSError stays a real fault")
    require(not proxy.expected_close_error(AssertionError("protocol")),
            "a protocol assertion is never classified as teardown")
    print("PASS wire_proxy_close_classification cases=7", flush=True)


if __name__ == "__main__":
    main()

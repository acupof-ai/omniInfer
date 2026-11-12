#!/usr/bin/env python3
"""Minimal shared helpers for the opt-in real-upstream E2E scripts.

Every script gates on UMMINFER_LIVE_OPT_IN=1 and on a complete, non-empty set
of environment variables before it opens a socket. Without the opt-in the
script prints BLOCKED/NOT_COVERED lines and exits 0. It never contacts an
upstream.

The chain under test is always: frozen umm-infer CLI -> local umm-infer serve ->
the single configured upstream -> the real model. Nothing here starts a fake
server or writes a synthetic response.
"""

import hashlib
import http.client
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import zlib

LOG_CAP = 64 * 1024
# Names whose values are credentials. The provider names are what the scripts
# themselves read; the others are the names the CLI or serve could look up.
SECRET_ENV = (
    "UMMINFER_UPSTREAM_API_KEY",
    "UMMINFER_API_KEY",
    "OPENAI_API_KEY",
    "UMMINFER_QWEN_API_KEY",
    "UMMINFER_H3_API_KEY",
)


class LiveBlocked(Exception):
    """Missing opt-in or configuration. Not a failure of the code under test."""


class LiveFail(AssertionError):
    """A real check failed."""


def gate(required, optional=()):
    """Raise LiveBlocked unless the opt-in and every required value are non-empty."""
    if os.environ.get("UMMINFER_LIVE_OPT_IN") != "1":
        raise LiveBlocked("UMMINFER_LIVE_OPT_IN is not 1")
    missing = [name for name in required if not os.environ.get(name, "")]
    if missing:
        raise LiveBlocked("missing or empty environment: " + ", ".join(missing))
    del optional  # optional names may be absent or empty


def blocked_main(script, required, run, optional=()):
    """Run `run` only after the gate; print BLOCKED and exit 0 otherwise."""
    if "--help" in sys.argv[1:]:
        print(f"usage: {script} [--keep]  (requires UMMINFER_LIVE_OPT_IN=1)")
        print("required environment: " + ", ".join(required))
        if optional:
            print("optional environment: " + ", ".join(optional))
        return 0
    try:
        gate(required, optional)
        run()
    except LiveBlocked as error:
        print(f"BLOCKED: {error}")
        print("NOT_COVERED: no upstream request was made")
    return 0


def binary_path(explicit):
    path = explicit or os.environ.get("UMMINFER_BIN", "")
    if not path:
        raise LiveBlocked("--binary or UMMINFER_BIN is required")
    path = os.path.abspath(path)
    if not os.access(path, os.X_OK):
        raise LiveFail(f"binary is not executable: {path}")
    return path


def sha256_file(path, limit=1 << 30):
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                raise LiveFail(f"artifact is larger than {limit} bytes: {path}")
            digest.update(chunk)
    if size == 0:
        raise LiveFail(f"artifact is empty: {path}")
    return digest.hexdigest(), size


def _poll(process):
    try:
        return process.poll()
    except OSError as error:
        raise LiveFail(f"cannot query child process state: {error}")


def _secrets(env=None):
    values = {os.environ.get(name, "") for name in SECRET_ENV}
    if env:
        values.update(env.get(name, "") for name in SECRET_ENV)
    return {value for value in values if value}


class Serve:
    """One local umm-infer serve process bound to one upstream.

    The upstream key is passed only through the environment. The command line
    never carries a key. The child's output is drained by a reader thread that
    keeps only the last LOG_CAP bytes, so the on-disk log is bounded for the
    whole run, not only after exit.
    """

    def __init__(self, binary, serve_args, env, workdir, name, timeout_s=30.0):
        self.binary = binary
        self.serve_args = list(serve_args)
        self.env = dict(env)
        self.workdir = workdir
        self.name = name
        self.timeout_s = timeout_s
        self.process = None
        self.port = 0
        self.log_path = os.path.join(workdir, f"{name}-serve.log")
        self._tail_bytes = bytearray()
        self._lock = threading.Lock()
        self._reader = None
        self._leak = False

    def __enter__(self):
        joined = " ".join(self.serve_args)
        for secret in _secrets(self.env):
            if secret in joined:
                raise LiveFail("a key reached the serve command line")
            if secret in self.binary:
                raise LiveFail("a key reached the binary path")
        self.process = subprocess.Popen(
            [self.binary, "serve", *self.serve_args],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=self.env,
            cwd=self.workdir)
        self._reader = threading.Thread(target=self._drain, daemon=True)
        self._reader.start()
        try:
            deadline = time.monotonic() + self.timeout_s
            while time.monotonic() < deadline:
                if _poll(self.process) is not None:
                    raise LiveFail(
                        f"{self.name} serve exited early with {self.process.returncode}: "
                        + self._tail())
                match = re.search(rb"serving on [^\s:]+:(\d+)", self._tail())
                if match:
                    self.port = int(match.group(1))
                    break
                time.sleep(0.05)
            if self.port == 0:
                raise LiveFail(f"{self.name} serve did not announce a port: " + self._tail())
            self._wait_health(time.monotonic() + self.timeout_s)
        except BaseException:
            # A with-statement does not call __exit__ when __enter__ raises, so
            # clean up here. The leak scan runs on this path too; a leaked
            # credential replaces the original startup error.
            self._shutdown()
            raise
        return self

    def _drain(self):
        stream = self.process.stdout
        while True:
            try:
                chunk = os.read(stream.fileno(), 65536)
            except (OSError, ValueError):
                break
            if not chunk:
                break
            with self._lock:
                self._tail_bytes.extend(chunk)
                if len(self._tail_bytes) > LOG_CAP:
                    del self._tail_bytes[:len(self._tail_bytes) - LOG_CAP]

    def _wait_health(self, deadline):
        while time.monotonic() < deadline:
            if _poll(self.process) is not None:
                raise LiveFail(f"{self.name} serve died: " + self._tail())
            try:
                if http_get(self.port, "/health")[0] == 200:
                    return
            except OSError:
                pass
            time.sleep(0.1)
        raise LiveFail(f"{self.name} serve never became healthy: " + self._tail())

    def _tail(self):
        with self._lock:
            return bytes(self._tail_bytes).decode(errors="replace")

    def _shutdown(self):
        try:
            if self.process is not None and _poll(self.process) is None:
                self.process.send_signal(signal.SIGTERM)
                try:
                    self.process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            if self._reader is not None:
                self._reader.join(timeout=5)
        finally:
            if self.process is not None and self.process.stdout is not None:
                self.process.stdout.close()
            tail = self._tail()
            # Scan before anything is persisted. A log that contains a
            # credential is never written, so --keep cannot retain it either.
            leaked = any(secret in tail for secret in _secrets(self.env))
            with open(self.log_path, "w") as handle:
                handle.write("[log withheld: credential detected]\n" if leaked else tail)
        if leaked:
            raise LiveFail("a key leaked into the serve log")
        return False

    def __exit__(self, *exc):
        return self._shutdown()


def cli_env(extra=None):
    """Environment for the CLI with every credential removed."""
    env = {k: v for k, v in os.environ.items() if k not in SECRET_ENV}
    if extra:
        env.update(extra)
    return env


def run_cli(binary, args, env, workdir, timeout_s=180.0):
    result = subprocess.run(
        [binary, *args], capture_output=True, text=True, timeout=timeout_s,
        env=env, cwd=workdir)
    for secret in _secrets(env):
        if secret in result.stdout or secret in result.stderr:
            raise LiveFail("a key leaked into CLI output")
    for secret in _secrets():
        if secret in result.stdout or secret in result.stderr:
            raise LiveFail("a key leaked into CLI output")
    return result


def json_stdout(result):
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise LiveFail("CLI returned no JSON: " + result.stderr[-2000:])
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as error:
        raise LiveFail(f"CLI stdout is not JSON ({error}): " + lines[-1][:2000])


def http_status_failure(result, status):
    """True when the CLI failed because the server answered with `status`."""
    return result.returncode != 0 and re.search(
        rf"HTTP {status}\b", result.stderr or "") is not None


def http_get(port, path, timeout=5.0):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("GET", path)
        response = conn.getresponse()
        return response.status, response.read().decode(errors="replace")
    finally:
        conn.close()


def require(condition, message):
    if not condition:
        raise LiveFail(message)


# Independent decoders. Each parses the container and, where possible, decodes
# the payload with the standard library instead of trusting the magic bytes.

def check_wav(path):
    digest, size = sha256_file(path, limit=64 << 20)
    with open(path, "rb") as handle:
        data = handle.read()
    require(len(data) >= 44, f"wav too short: {len(data)} bytes")
    require(data[0:4] == b"RIFF" and data[8:12] == b"WAVE",
            "wav is missing the RIFF/WAVE header")
    require(int.from_bytes(data[4:8], "little") + 8 == len(data),
            "wav RIFF size does not match the file")
    offset, fmt, payload, payload_at, channels, rate = 12, None, 0, None, 0, 0
    while offset + 8 <= len(data):
        name = data[offset:offset + 4]
        length = int.from_bytes(data[offset + 4:offset + 8], "little")
        body = data[offset + 8:offset + 8 + length]
        require(len(body) == length, "wav chunk is truncated")
        if name == b"fmt ":
            require(length >= 16, "wav fmt chunk is too short")
            fmt = int.from_bytes(body[0:2], "little")
            channels = int.from_bytes(body[2:4], "little")
            rate = int.from_bytes(body[4:8], "little")
        elif name == b"data":
            payload = length
            payload_at = offset + 8
        offset += 8 + length + (length & 1)
    require(offset == len(data), "wav has trailing bytes after the last chunk")
    require(fmt in (1, 3), f"wav has no PCM/float format chunk: {fmt}")
    require(channels > 0, "wav channel count is zero")
    require(rate > 0, "wav sample rate is zero")
    require(payload > 0 and payload_at is not None, "wav data chunk is empty")
    samples = data[payload_at:payload_at + payload]
    require(any(byte != 0 for byte in samples), "wav samples are all zero (silence)")
    return digest, size


def _png(data):
    require(data[:8] == b"\x89PNG\r\n\x1a\n", "png signature mismatch")
    offset, seen_ihdr, seen_iend, compressed = 8, False, False, bytearray()
    info = {}
    while offset + 12 <= len(data):
        length = int.from_bytes(data[offset:offset + 4], "big")
        name = data[offset + 4:offset + 8]
        body = data[offset + 8:offset + 8 + length]
        require(len(body) == length, "png chunk is truncated")
        crc = int.from_bytes(data[offset + 8 + length:offset + 12 + length], "big")
        require(zlib.crc32(name + body) & 0xFFFFFFFF == crc,
                f"png chunk CRC mismatch: {name!r}")
        if name == b"IHDR":
            require(length == 13, "png IHDR length is not 13")
            info = {
                "width": int.from_bytes(body[0:4], "big"),
                "height": int.from_bytes(body[4:8], "big"),
                "depth": body[8],
                "color": body[9],
            }
            require(info["width"] > 0 and info["height"] > 0,
                    "png has zero dimensions")
            seen_ihdr = True
        elif name == b"IDAT":
            compressed.extend(body)
        elif name == b"IEND":
            seen_iend = True
            offset += 12 + length
            break
        offset += 12 + length
    require(seen_ihdr, "png has no IHDR")
    require(seen_iend, "png has no IEND")
    require(offset == len(data), "png has trailing bytes after IEND")
    require(compressed, "png has no image data")
    decoder = zlib.decompressobj()
    try:
        raw = decoder.decompress(bytes(compressed), 256 << 20)
    except zlib.error as error:
        raise LiveFail(f"png IDAT does not decode: {error}")
    require(raw, "png IDAT decodes to nothing")
    require(decoder.eof, "png zlib stream is truncated")
    return info


def _jpeg(data):
    require(data[:2] == b"\xff\xd8", "jpeg has no SOI")
    offset, seen_sof, seen_sos, info = 2, False, False, {}
    while offset + 4 <= len(data):
        require(data[offset] == 0xFF, "jpeg marker is not aligned")
        marker = data[offset + 1]
        offset += 2
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            continue
        length = int.from_bytes(data[offset:offset + 2], "big")
        require(length >= 2 and offset + length <= len(data),
                "jpeg segment is truncated")
        if marker in (0xC0, 0xC1, 0xC2):
            require(length >= 7, "jpeg SOF is too short")
            height = int.from_bytes(data[offset + 3:offset + 5], "big")
            width = int.from_bytes(data[offset + 5:offset + 7], "big")
            require(width > 0 and height > 0, "jpeg has zero dimensions")
            info = {"width": width, "height": height, "depth": 8,
                    "color": data[offset + 7]}
            seen_sof = True
        if marker == 0xDA:
            seen_sos = True
            break
        offset += length
    require(seen_sof, "jpeg has no SOF segment")
    require(seen_sos, "jpeg has no SOS segment")
    require(data[-2:] == b"\xff\xd9", "jpeg has no EOI marker")
    return info


def check_image(path):
    digest, size = sha256_file(path, limit=64 << 20)
    with open(path, "rb") as handle:
        data = handle.read()
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png", (digest, size), _png(data)
    if data[:2] == b"\xff\xd8":
        require(size > 1024, "jpeg is too small to be an image")
        return "jpeg", (digest, size), _jpeg(data)
    raise LiveFail(f"artifact is neither PNG nor JPEG: {data[:8]!r}")


def check_mp4(path):
    digest, size = sha256_file(path, limit=1 << 30)
    with open(path, "rb") as handle:
        data = handle.read()
    require(len(data) >= 16, "mp4 is too short")
    offset, boxes = 0, {}
    while offset + 8 <= len(data):
        length = int.from_bytes(data[offset:offset + 4], "big")
        name = data[offset + 4:offset + 8]
        header = 8
        if length == 1:
            require(offset + 16 <= len(data), "mp4 largesize box is truncated")
            length = int.from_bytes(data[offset + 8:offset + 16], "big")
            header = 16
        elif length == 0:
            length = len(data) - offset
        require(length >= header and offset + length <= len(data),
                f"mp4 box {name!r} overruns the file")
        boxes[name] = length - header
        offset += length
    require(offset == len(data), "mp4 has trailing bytes after the last box")
    require(b"ftyp" in boxes, f"mp4 has no ftyp box: {list(boxes)[:8]}")
    require(boxes.get(b"moov", 0) > 0, "mp4 moov box is empty")
    require(boxes.get(b"mdat", 0) > 0, "mp4 mdat box is empty")
    return digest, size


def expect_existing_output_refused(binary, args, env, workdir, path):
    """The CLI must refuse to overwrite an existing artifact before any request."""
    before = sha256_file(path)
    result = run_cli(binary, args, env, workdir, timeout_s=60)
    require(result.returncode != 0,
            "CLI overwrote an existing artifact instead of refusing")
    require(sha256_file(path) == before, "existing artifact changed")

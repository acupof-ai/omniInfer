#!/usr/bin/env python3
"""Real Qwen3-ASR external stage backed by SGLang over HTTP."""

import argparse
import base64
import http.client
import importlib.util
import io
import json
import math
from pathlib import Path
import queue
import signal
import socket
import sys
import threading
import time
from types import SimpleNamespace
from urllib.parse import urlsplit
import wave


def load_pipeline_stage():
    path = Path(__file__).resolve().with_name("umminfer_mlx_stage.py")
    spec = importlib.util.spec_from_file_location("umminfer_mlx_stage", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MLXStage


MLXStage = load_pipeline_stage()
MAX_RESPONSE_BYTES = 4 * 1024 * 1024


def pcm16_wav_data_url(pcm):
    if not isinstance(pcm, bytes) or not pcm or len(pcm) % 2:
        raise ValueError("audio must be nonempty PCM16 with complete samples")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(pcm)
    return "data:audio/wav;base64," + base64.b64encode(output.getvalue()).decode("ascii")


class SGLangBackend:
    def __init__(self, base_url, model, timeout=120.0):
        parsed = urlsplit(base_url)
        if (parsed.scheme != "http" or not parsed.hostname or
                parsed.path not in ("", "/") or parsed.query or parsed.fragment or
                parsed.username or parsed.password):
            raise ValueError("backend URL must be an HTTP origin without credentials")
        self.host = parsed.hostname
        self.port = parsed.port or 80
        self.model = model
        self.timeout = timeout
        self._lock = threading.Lock()
        self._active = None
        self._cancelled = threading.Event()

    def _request_json(self, method, path, payload=None):
        body = None if payload is None else json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        connection = http.client.HTTPConnection(self.host, self.port, self.timeout)
        with self._lock:
            if self._active is not None:
                connection.close()
                raise RuntimeError("another SGLang request is already active")
            self._active = connection
        try:
            headers = {"Content-Type": "application/json"} if body is not None else {}
            with self._lock:
                if self._cancelled.is_set():
                    raise RuntimeError("SGLang request was cancelled")
                connection.request(method, path, body=body, headers=headers)
            if self._cancelled.is_set():
                raise RuntimeError("SGLang request was cancelled")
            response = connection.getresponse()
            data = response.read(MAX_RESPONSE_BYTES + 1)
            if len(data) > MAX_RESPONSE_BYTES:
                raise ValueError("SGLang response exceeds 4 MiB")
            if not 200 <= response.status < 300:
                detail = data[:4096].decode("utf-8", "replace")
                raise RuntimeError(f"SGLang HTTP {response.status}: {detail}")
            if self._cancelled.is_set():
                raise RuntimeError("SGLang request was cancelled")
            return json.loads(data)
        finally:
            with self._lock:
                if self._active is connection:
                    self._active = None
            connection.close()

    def close_active(self):
        self._cancelled.set()
        with self._lock:
            connection = self._active
        if connection is None:
            return
        try:
            if connection.sock is not None:
                connection.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        connection.close()

    def prepare_request(self):
        self._cancelled.clear()

    def check_model(self):
        self.prepare_request()
        body = self._request_json("GET", "/v1/models")
        ids = [item.get("id") for item in body.get("data", [])
               if isinstance(item, dict)] if isinstance(body, dict) else []
        if self.model not in ids:
            raise ValueError(f"SGLang does not serve {self.model!r}: {ids!r}")

    def transcribe(self, pcm, max_tokens, temperature, request_id):
        payload = {
            "text": "",
            "audio_data": pcm16_wav_data_url(pcm),
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_tokens,
            },
            "stream": False,
            "modalities": ["audio"],
            "rid": request_id,
        }
        print(f"SGLang request {request_id} started", flush=True)
        body = self._request_json("POST", "/generate", payload)
        if not isinstance(body, dict) or not isinstance(body.get("text"), str):
            raise ValueError("SGLang did not return transcription text")
        metadata = body.get("meta_info")
        if not isinstance(metadata, dict):
            raise ValueError("SGLang did not return meta_info")
        count = metadata.get("completion_tokens")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("SGLang did not report completion_tokens")
        finish = metadata.get("finish_reason")
        reason = finish.get("type") if isinstance(finish, dict) else None
        if not isinstance(reason, str) or not reason:
            raise ValueError("SGLang did not report finish_reason")
        if reason == "abort":
            raise ValueError("SGLang aborted the transcription")
        text = body["text"].split("<asr_text>", 1)[-1]
        latency = metadata.get("e2e_latency")
        print(f"SGLang request {request_id} finished tokens={count} "
              f"finish={reason} e2e_seconds={latency}", flush=True)
        return SimpleNamespace(
            text=text, generation_tokens=count, finish_reason=reason,
            prompt_tokens=metadata.get("prompt_tokens"), backend_e2e_seconds=latency)


class QwenASRStage(MLXStage):
    def __init__(self, name, recv_endpoint, coordinator_endpoint, abort_endpoint,
                 model_path, backend_url, timeout=120.0):
        super().__init__(name, recv_endpoint, coordinator_endpoint, abort_endpoint,
                         model_path)
        self.backend = SGLangBackend(backend_url, model_path, timeout)
        self.poll_interval = 0.1
        self._backend_request_id = None

    def load(self):
        if self.generate is not None:
            return
        self.backend.check_model()
        self.tokenizer = None
        self.generate = self._generate

    def close(self):
        self.backend.close_active()
        super().close()

    def parse_request(self, message):
        data = message["data"]
        request = data["request"]
        inputs = request["inputs"]
        if not isinstance(inputs, dict) or set(inputs) != {"audio"}:
            raise ValueError("Qwen ASR inputs must contain only audio")
        pcm = inputs["audio"]
        if not isinstance(pcm, bytes) or not pcm or len(pcm) % 2:
            raise ValueError("audio must be nonempty PCM16 with complete samples")
        params = request.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("params must be a map")
        max_tokens = params.get("max_new_tokens", 128)
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise ValueError("max_new_tokens must be an integer")
        if not 1 <= max_tokens <= 4096:
            raise ValueError("max_new_tokens is outside the supported range")
        temperature = params.get("temperature", 0.0)
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise ValueError("temperature must be a number")
        if not math.isfinite(temperature) or temperature != 0:
            raise ValueError("Qwen ASR requires temperature 0")
        stream = params.get("stream", False)
        if not isinstance(stream, bool):
            raise ValueError("stream must be a boolean")
        if stream:
            raise ValueError("incremental ASR streaming is not supported")
        self._backend_request_id = "umminfer-" + message["request_id"]
        return pcm, max_tokens, 0.0, False

    def _generate(self, pcm, max_tokens, temperature):
        results = queue.Queue(maxsize=1)

        def call():
            try:
                results.put(self.backend.transcribe(
                    pcm, max_tokens, temperature, self._backend_request_id))
            except Exception as exc:
                results.put(exc)

        thread = threading.Thread(target=call, name="qwen-asr-http", daemon=True)
        self.backend.prepare_request()
        thread.start()
        try:
            while True:
                try:
                    result = results.get(timeout=self.poll_interval)
                    break
                except queue.Empty:
                    # Empty responses let MLXStage.poll_events observe aborts.
                    yield SimpleNamespace(text="")
            if isinstance(result, Exception):
                raise result
            yield result
        finally:
            self.backend.close_active()
            thread.join(2)
            if thread.is_alive():
                print("SGLang HTTP thread did not exit after connection close",
                      file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="run the real Qwen ASR stage")
    serve.add_argument("--name", required=True)
    serve.add_argument("--recv", required=True)
    serve.add_argument("--coordinator", required=True)
    serve.add_argument("--abort", required=True)
    serve.add_argument("--model", required=True)
    serve.add_argument("--backend-url", required=True)
    serve.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    stage = QwenASRStage(args.name, args.recv, args.coordinator, args.abort,
                         args.model, args.backend_url, args.timeout)

    def stop(signum, frame):
        stage.running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        stage.start()
        stage.run()
    finally:
        stage.close()


if __name__ == "__main__":
    main()

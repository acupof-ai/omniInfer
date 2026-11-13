#!/usr/bin/env python3
"""Real Qwen3-Omni worker using the existing external-stage Coordinator protocol."""

import argparse
import importlib.util
import json
import math
from pathlib import Path
import signal
from types import SimpleNamespace
import urllib.request


def load_pipeline_stage():
    path = Path(__file__).resolve().with_name("umminfer_mlx_stage.py")
    spec = importlib.util.spec_from_file_location("umminfer_mlx_stage", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MLXStage


MLXStage = load_pipeline_stage()


class QwenOmniWorker(MLXStage):
    def __init__(self, name, recv_endpoint, coordinator_endpoint, abort_endpoint,
                 upstream_base_url, upstream_model, request_timeout=300.0,
                 log_requests=False, perf_log=None):
        super().__init__(name, recv_endpoint, coordinator_endpoint, abort_endpoint,
                         upstream_base_url.rstrip("/"), perf_log=perf_log)
        self.upstream_model = upstream_model
        self.request_timeout = request_timeout
        self.log_requests = log_requests

    def open_upstream(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            self.model_path + path, data=data,
            headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(request, timeout=self.request_timeout)

    def load(self):
        if self.generate is not None:
            return
        with self.open_upstream("/models") as response:
            models = json.load(response)
        ids = [item.get("id") for item in models.get("data", [])
               if isinstance(item, dict)]
        if self.upstream_model not in ids:
            raise ValueError(f"upstream model {self.upstream_model!r} is not served")
        self.tokenizer = None
        self.generate = self.generate_upstream

    def parse_request(self, message):
        request = message["data"]["request"]
        messages = request["inputs"]["messages"]
        params = request.get("params", {})
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a nonempty array")
        for item in messages:
            if not isinstance(item, dict) or not isinstance(item.get("role"), str):
                raise ValueError("each message must be a map with a string role")
            content = item.get("content")
            if not isinstance(content, (str, list)):
                raise ValueError("message content must be text or a content part array")
            if isinstance(content, list):
                for part in content:
                    if (not isinstance(part, dict)
                            or not isinstance(part.get("type"), str)):
                        raise ValueError("each content part requires a string type")
        if not isinstance(params, dict):
            raise ValueError("params must be a map")
        max_tokens = params.get("max_new_tokens", 256)
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise ValueError("max_new_tokens must be an integer")
        if not 1 <= max_tokens <= 2**31 - 1:
            raise ValueError("max_new_tokens is outside the supported range")
        temperature = params.get("temperature", 0.0)
        if (isinstance(temperature, bool)
                or not isinstance(temperature, (int, float))):
            raise ValueError("temperature must be a number")
        if not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        stream = params.get("stream", False)
        if not isinstance(stream, bool):
            raise ValueError("stream must be a boolean")
        if self.log_requests:
            print(f"Qwen Omni request {message['request_id']} received "
                  f"messages={json.dumps(messages, ensure_ascii=False)}", flush=True)
        payload = {
            "model": self.upstream_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": float(temperature),
            # One upstream stream serves both public modes and makes abort reach
            # the existing stage loop at token boundaries.
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        return payload, max_tokens, float(temperature), stream

    def complete(self, request_id, success, text="", error=None,
                 generated_tokens=None, finish_reason=None):
        if self.log_requests:
            print(f"Qwen Omni request {request_id} finished success={success} "
                  f"finish={finish_reason}", flush=True)
        super().complete(request_id, success, text=text, error=error,
                         generated_tokens=generated_tokens,
                         finish_reason=finish_reason)

    def generate_upstream(self, payload, max_tokens, temperature):
        del max_tokens, temperature  # Already validated and encoded in payload.
        response = self.open_upstream("/chat/completions", payload)
        usage = None
        finish_reason = None
        done = False
        data_lines = []
        try:
            for raw_line in response:
                line = raw_line.decode("utf-8").rstrip("\r\n")
                if not line:
                    if not data_lines:
                        continue
                    data = "\n".join(data_lines)
                    data_lines = []
                    if data == "[DONE]":
                        done = True
                        break
                    event = json.loads(data)
                    if not isinstance(event, dict):
                        raise ValueError("SGLang SSE event must be a map")
                    if event.get("error") is not None:
                        raise RuntimeError(f"SGLang error: {event['error']}")
                    if isinstance(event.get("usage"), dict):
                        usage = event["usage"]
                    text = ""
                    for choice in event.get("choices", []):
                        reason = choice.get("finish_reason")
                        if reason is not None:
                            finish_reason = reason
                        delta = choice.get("delta") or {}
                        content = delta.get("content")
                        if content is not None and not isinstance(content, str):
                            raise ValueError("SGLang delta content must be text")
                        if content:
                            text += content
                    yield SimpleNamespace(text=text, generation_tokens=None,
                                          finish_reason=None)
                elif line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
            if not done:
                raise RuntimeError("SGLang stream ended before [DONE]")
        finally:
            response.close()
        if usage is None:
            raise RuntimeError("SGLang stream did not report usage")
        generated_tokens = usage.get("completion_tokens")
        yield SimpleNamespace(text="", generation_tokens=generated_tokens,
                              finish_reason=finish_reason)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--recv", required=True)
    parser.add_argument("--coordinator", required=True)
    parser.add_argument("--abort", required=True)
    parser.add_argument("--upstream-base-url", required=True)
    parser.add_argument("--upstream-model", required=True)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--log-requests", action="store_true")
    # Optional JSON-lines record of queue, abort, start and completion events.
    parser.add_argument("--perf-log", default=None)
    args = parser.parse_args()
    if not math.isfinite(args.request_timeout) or args.request_timeout <= 0:
        parser.error("--request-timeout must be positive and finite")
    stage = QwenOmniWorker(
        args.name, args.recv, args.coordinator, args.abort,
        args.upstream_base_url, args.upstream_model, args.request_timeout,
        args.log_requests, args.perf_log)

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

#!/usr/bin/env python3
"""Real Qwen3-Omni worker using the existing external-stage Coordinator protocol."""

import argparse
import base64
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import queue
import signal
import sys
import threading
import time
from types import SimpleNamespace
import urllib.error
import urllib.request


def summarize_images(messages):
    """Replace base64 image payloads with length and SHA256.

    The request log is evidence, not a data dump: one screenshot data URL would
    exceed a usable log bound. The digest still proves which bytes arrived.
    """
    summarized = []
    for message in messages:
        if not isinstance(message, dict) or not isinstance(
                message.get("content"), list):
            summarized.append(message)
            continue
        parts = []
        for part in message["content"]:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                entry = {"type": "image_url"}
                if url.startswith("data:") and "," in url:
                    header, payload = url.split(",", 1)
                    try:
                        raw = base64.b64decode(payload, validate=True)
                    except ValueError:
                        entry.update({"url_prefix": header,
                                      "base64_bytes": len(payload)})
                    else:
                        entry.update({"url_prefix": header, "bytes": len(raw),
                                      "sha256": hashlib.sha256(raw).hexdigest()})
                else:
                    entry["url"] = url
                parts.append(entry)
            else:
                parts.append(part)
        summarized.append({**message, "content": parts})
    return summarized


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
                 log_requests=False, perf_log=None, concurrency=1):
        # The upstream SGLang server batches concurrent requests. Running
        # several request threads is what lets it: the model call is an HTTP
        # stream, not local compute.
        super().__init__(name, recv_endpoint, coordinator_endpoint, abort_endpoint,
                         upstream_base_url.rstrip("/"), perf_log=perf_log,
                         concurrency=concurrency)
        self.upstream_model = upstream_model
        self.request_timeout = request_timeout
        self.log_requests = log_requests
        self._backend_request_id = None
        # SGLang keys rid_to_state by rid. Coordinator ids restart with each
        # server process, so a reused id would collide with a rid the backend
        # still holds. A per-process tag keeps every submitted rid unique.
        self.backend_tag = str(os.getpid())
        # A blocked upstream read yields an empty response so the stage loop
        # can observe an abort between tokens, as the ASR stage already does.
        self.poll_interval = 0.1

    def open_upstream(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            self.model_path + path, data=data,
            headers={"Content-Type": "application/json"})
        try:
            return urllib.request.urlopen(request, timeout=self.request_timeout)
        except urllib.error.HTTPError as exc:
            # Keep the backend's reason. A bare "HTTP Error 400" hides which
            # field the SGLang protocol rejected.
            detail = exc.read(4096).decode("utf-8", "replace")
            raise RuntimeError(
                f"SGLang HTTP {exc.code}: {detail}") from exc

    def abort_url(self):
        """SGLang serves /abort_request at the origin, not under the /v1 base."""
        root = self.model_path
        return (root[:-3] if root.endswith("/v1") else root) + "/abort_request"

    def request_abort(self, request_id):
        """Ask SGLang to stop this rid on the GPU scheduler.

        A local response close drops one HTTP client and stops no GPU work.
        The confirmation is not this call's 200: the reader thread sees the
        generate stream end with finish_reason "abort".
        """
        print(f"SGLANG_ABORT_REQUEST rid={request_id}", flush=True)
        request = urllib.request.Request(
            self.abort_url(),
            data=json.dumps({"rid": request_id}).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=10.0) as reply:
                print(f"SGLANG_ABORT_ACCEPTED rid={request_id} http={reply.status}",
                      flush=True)
                return True
        except Exception as exc:  # noqa: BLE001 - report, never mask the cancel
            print(f"SGLang abort_request failed for {request_id}: {exc}",
                  file=sys.stderr, flush=True)
            return False

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
        # SGLang registers this rid for the upstream request. POST
        # /abort_request needs the same value to stop this request on the
        # scheduler. Request threads keep their own value.
        request_id = f"{message['request_id']}@{self.backend_tag}"
        self._backend_request_id = request_id
        self.local.backend_request_id = request_id
        if self.log_requests:
            print(f"Qwen Omni request {request_id} received "
                  f"messages={json.dumps(summarize_images(messages), ensure_ascii=False)}",
                  flush=True)
        payload = {
            "model": self.upstream_model,
            "rid": request_id,
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
        request_id = getattr(self.local, "backend_request_id",
                             self._backend_request_id)
        # The stage loop closes this generator when it observes an abort. If the
        # read lived here, that close would abandon the HTTP stream before
        # SGLang could send its finish_reason=abort frame. A request thread owns
        # the read instead, exactly as the ASR stage owns its SGLang stream, so
        # the confirmation is still observed after the cancel.
        events = queue.Queue()
        end = object()
        response_holder = {}
        state = {"usage": None, "finish": None, "done": False,
                 "confirmed": False, "failed": False,
                 "abort_requested": threading.Event()}

        def read():
            usage = None
            finish_reason = None
            done = False
            data_lines = []
            try:
                response = self.open_upstream("/chat/completions", payload)
                response_holder["response"] = response
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
                                if reason == "abort":
                                    state["confirmed"] = True
                            delta = choice.get("delta") or {}
                            content = delta.get("content")
                            if content is not None and not isinstance(content, str):
                                raise ValueError("SGLang delta content must be text")
                            if content:
                                text += content
                        events.put(SimpleNamespace(text=text,
                                                   generation_tokens=None,
                                                   finish_reason=None))
                    elif line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())
                if not done and not state["abort_requested"].is_set():
                    raise RuntimeError("SGLang stream ended before [DONE]")
            except Exception as exc:  # noqa: BLE001 - delivered to the consumer
                state["failed"] = True
                events.put(exc)
            finally:
                state["usage"] = usage
                state["finish"] = finish_reason
                state["done"] = done
                events.put(end)

        thread = threading.Thread(target=read, name="qwen-omni-stream",
                                  daemon=True)
        thread.start()
        completed = False
        try:
            while True:
                try:
                    item = events.get(timeout=self.poll_interval)
                except queue.Empty:
                    # Empty responses let the stage loop observe an abort.
                    yield SimpleNamespace(text="")
                    continue
                if item is end:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
            completed = True
            if not state["done"]:
                raise RuntimeError("SGLang stream ended before [DONE]")
            if state["usage"] is None:
                raise RuntimeError("SGLang stream did not report usage")
            generated_tokens = state["usage"].get("completion_tokens")
            yield SimpleNamespace(text="", generation_tokens=generated_tokens,
                                  finish_reason=state["finish"])
        finally:
            if not completed and not state["failed"]:
                state["abort_requested"].set()
                self.request_abort(request_id)
            thread.join(10)
            response = response_holder.get("response")
            if thread.is_alive():
                # The abort did not end the stream. Close the local connection
                # so the reader thread cannot outlive its request.
                if response is not None:
                    response.close()
                thread.join(2)
            if not completed:
                confirmed = state["confirmed"]
                print(f"SGLANG_ABORT_{'CONFIRMED' if confirmed else 'UNCONFIRMED'} "
                      f"rid={request_id} finish={'abort' if confirmed else 'none'}",
                      flush=True)
                if self.perf_log is not None:
                    self.write_perf({"event": "gpu_abort",
                                     "request_id": request_id,
                                     "confirmed": confirmed})
            if response is not None:
                response.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--recv", required=True)
    parser.add_argument("--coordinator", required=True)
    parser.add_argument("--abort", required=True)
    parser.add_argument("--upstream-base-url", required=True)
    parser.add_argument("--upstream-model", required=True)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    # Concurrent upstream requests let SGLang batch. 1 restores the serial
    # worker loop this flag was added to remove.
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--log-requests", action="store_true")
    # Optional JSON-lines record of queue, abort, start and completion events.
    parser.add_argument("--perf-log", default=None)
    args = parser.parse_args()
    if not math.isfinite(args.request_timeout) or args.request_timeout <= 0:
        parser.error("--request-timeout must be positive and finite")
    if args.concurrency < 1:
        parser.error("--concurrency must be a positive integer")
    stage = QwenOmniWorker(
        args.name, args.recv, args.coordinator, args.abort,
        args.upstream_base_url, args.upstream_model, args.request_timeout,
        args.log_requests, args.perf_log, args.concurrency)

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

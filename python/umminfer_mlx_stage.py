#!/usr/bin/env python3
"""Persistent single-request MLX-LM stage for the C++ coordinator."""

import argparse
from collections import deque
import json
import math
import signal
import sys
import time

import msgpack
import zmq


# An abort that arrives before its submit leaves a tombstone. The tombstone is
# removed when the submit arrives, or by this deadline if no submit follows.
# A capacity limit would make a cancellation depend on unrelated traffic.
ABORT_TOMBSTONE_TTL_S = 60.0


def _cache_vocabulary(tokenizer):
    # mlx-lm builds a streaming detokenizer, and that detokenizer's token map,
    # on every stream_generate call. transformers rebuilds the vocabulary on
    # each access: measured 48 ms of a 104 ms request for this model. A loaded
    # tokenizer's vocabulary does not change, so answer get_vocab from a cache
    # built once here. The guard keeps this a no-op on a different mlx-lm
    # layout.
    inner = getattr(tokenizer, "_tokenizer", None)
    if inner is not None and hasattr(inner, "get_vocab"):
        # Count the real calls, so a test can state how often the map was built
        # without changing the tokenizer or the model.
        stats = {"builds": 0, "accesses": 0}
        rebuild = inner.get_vocab

        def build(*args, **kwargs):
            stats["builds"] += 1
            return rebuild(*args, **kwargs)

        vocabulary = build()

        def cached(*args, **kwargs):
            stats["accesses"] += 1
            return vocabulary

        inner.get_vocab = cached
        tokenizer._umminfer_vocab = stats


def _make_greedy_generate(model, tokenizer, skip_final_lookahead=True,
                          reuse_prompt_cache=True, reuse_detokenizer=True):
    """Build the temp=0 path without unused full-vocabulary logprobs."""
    import contextlib
    import copy
    import mlx.core as mx
    from mlx_lm.models import cache
    from mlx_lm.generate import GenerationResponse, generation_stream
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    prompt_snapshot = None
    prompt_cache_stats = {"hits": 0, "misses": 0}
    detokenizer_prototype = None
    detokenizer_vocabulary = None
    detokenizer_stats = {"builds": 0, "copies": 0}

    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)
    detokenizer_class = getattr(tokenizer, "_detokenizer_class", None)
    reuse_detokenizer = (
        reuse_detokenizer
        and getattr(detokenizer_class, "__name__", None)
        == "BPEStreamingDetokenizer")
    if mx.metal.is_available():
        # This factory belongs to a persistent, single-model stage. Setting the
        # limit per request walks all weights and synchronizes the stream.
        mx.set_wired_limit(
            mx.device_info()["max_recommended_working_set_size"])

    def clone_prompt_cache(source):
        cloned = copy.copy(source)
        for index, item in enumerate(source):
            if isinstance(item, cache.KVCache):
                # copy.deepcopy walks the cache object graph on every request.
                # Only its two arrays need independent storage; offset is an int.
                item_copy = copy.copy(item)
                item_copy.keys = mx.array(item.keys)
                item_copy.values = mx.array(item.values)
                cloned[index] = item_copy
            else:
                cloned[index] = copy.deepcopy(item)
        return cloned

    def generate(prompt, max_tokens):
        nonlocal prompt_snapshot, detokenizer_prototype, detokenizer_vocabulary
        if not isinstance(prompt, mx.array):
            if isinstance(prompt, str):
                add_special_tokens = (
                    tokenizer.bos_token is None
                    or not prompt.startswith(tokenizer.bos_token))
                prompt = tokenizer.encode(
                    prompt, add_special_tokens=add_special_tokens)
            prompt = mx.array(prompt)

        if reuse_detokenizer:
            # This read is the cache validity check, not a counter-only access.
            # A replacement vocabulary requires a new token map. The loaded
            # tokenizer's cached vocabulary keeps its identity between requests.
            vocabulary = tokenizer.vocab
            if (detokenizer_prototype is None
                    or vocabulary is not detokenizer_vocabulary):
                detokenizer_prototype = tokenizer.detokenizer
                detokenizer_vocabulary = vocabulary
                detokenizer_stats["builds"] += 1
            # Only the immutable vocabulary map and byte decoder are shared.
            # reset replaces every mutable field used by add_token/finalize.
            detokenizer = copy.copy(detokenizer_prototype)
            detokenizer.reset()
            detokenizer_stats["copies"] += 1
        else:
            detokenizer = tokenizer.detokenizer
        prompt_cache = None

        def model_call(input_tokens):
            return model(input_tokens[None], cache=prompt_cache)

        def step(input_tokens):
            with mx.stream(generation_stream):
                logits = model_call(input_tokens)[:, -1, :]
                return mx.argmax(logits, axis=-1)

        with contextlib.nullcontext():
            tic = time.perf_counter()
            with mx.stream(generation_stream):
                prompt_key = tuple(prompt.tolist())
                if (reuse_prompt_cache and prompt_snapshot is not None
                        and prompt_snapshot[0] == prompt_key):
                    # The master snapshot never enters a model call. Decode
                    # mutates only this request's cloned KV arrays. The first
                    # greedy token is immutable and safe to share.
                    prompt_cache = clone_prompt_cache(prompt_snapshot[1])
                    y = prompt_snapshot[2]
                    prompt_cache_stats["hits"] += 1
                else:
                    prompt_cache = cache.make_prompt_cache(model)
                    remaining = prompt
                    prompt_cache_stats["misses"] += 1
                    while remaining.size > 1:
                        count = min(2048, remaining.size - 1)
                        model_call(remaining[:count])
                        mx.eval([item.state for item in prompt_cache])
                        remaining = remaining[count:]
                        mx.clear_cache()
                    y = step(remaining)
                    if reuse_prompt_cache:
                        # Snapshot the whole prompt before decode mutates its
                        # KV. Only the immutable prompt and its first greedy
                        # token are reused, never generated output text.
                        prompt_snapshot = (prompt_key,
                                           copy.deepcopy(prompt_cache), y)

            mx.async_eval(y)
            n = 0
            while n < max_tokens:
                # Build the lookahead lazily. Terminal tokens do not schedule it.
                next_y = None
                if n + 1 < max_tokens or not skip_final_lookahead:
                    next_y = step(y)
                if n == 0:
                    mx.eval(y)
                    prompt_time = time.perf_counter() - tic
                    prompt_tps = prompt.size / prompt_time
                    tic = time.perf_counter()
                token = y.item()
                finish_reason = (
                    "stop" if token in tokenizer.eos_token_ids else None)
                if finish_reason is None:
                    if next_y is not None:
                        mx.async_eval(next_y)
                    detokenizer.add_token(token)
                    if n + 1 == max_tokens:
                        finish_reason = "length"
                if finish_reason is None:
                    yield GenerationResponse(
                        text=detokenizer.last_segment,
                        token=token,
                        logprobs=None,
                        from_draft=False,
                        prompt_tokens=prompt.size,
                        prompt_tps=prompt_tps,
                        generation_tokens=n + 1,
                        generation_tps=(n + 1) / (time.perf_counter() - tic),
                        peak_memory=mx.get_peak_memory() / 1e9,
                        finish_reason=None,
                    )
                    if n % 256 == 0:
                        mx.clear_cache()
                    y = next_y
                    n += 1
                    continue
                break

            detokenizer.finalize()
            yield GenerationResponse(
                text=detokenizer.last_segment,
                token=token,
                logprobs=None,
                from_draft=False,
                prompt_tokens=prompt.size,
                prompt_tps=prompt_tps,
                generation_tokens=n + 1,
                generation_tps=(n + 1) / (time.perf_counter() - tic),
                peak_memory=mx.get_peak_memory() / 1e9,
                finish_reason=finish_reason,
            )

    generate.prompt_cache_stats = prompt_cache_stats
    generate.detokenizer_stats = detokenizer_stats
    return generate


def _fuse_qwen2_qkv(model):
    """Evaluate one quantized QKV projection instead of three per layer."""
    import copy
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models import qwen2

    if not isinstance(model, qwen2.Model):
        return
    attentions = [layer.self_attn for layer in model.model.layers]
    projections = [(attention.q_proj, attention.k_proj, attention.v_proj)
                   for attention in attentions]
    if not all(isinstance(projection, nn.QuantizedLinear)
               for group in projections for projection in group):
        return
    layouts = [(q.group_size, q.bits, q.mode, q.weight.shape[1:])
               for q, _, _ in projections]
    if any((k.group_size, k.bits, k.mode, k.weight.shape[1:]) != layout or
           (v.group_size, v.bits, v.mode, v.weight.shape[1:]) != layout or
           k.weight.shape[0] != v.weight.shape[0]
           for layout, (_, k, v) in zip(layouts, projections)):
        return

    class FusedAttention(qwen2.Attention):
        def __call__(self, x, mask=None, cache=None):
            B, L, D = x.shape
            qkv = self.qkv_proj(x)
            queries = qkv[..., :self.q_size]
            keys = qkv[..., self.q_size:self.q_size + self.kv_size]
            values = qkv[..., self.q_size + self.kv_size:]
            queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
            keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
            values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
            if cache is not None:
                queries = self.rope(queries, offset=cache.offset)
                keys = self.rope(keys, offset=cache.offset)
                keys, values = cache.update_and_fetch(keys, values)
            else:
                queries = self.rope(queries)
                keys = self.rope(keys)
            output = qwen2.scaled_dot_product_attention(
                queries, keys, values, cache=cache, scale=self.scale, mask=mask)
            output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
            return self.o_proj(output)

    for attention, (q, k, v) in zip(attentions, projections):
        fused = copy.copy(q)
        for name in ("weight", "scales", "biases", "bias"):
            setattr(fused, name, mx.concatenate(
                [getattr(projection, name) for projection in (q, k, v)], axis=0))
        attention.q_size = q.weight.shape[0]
        attention.kv_size = k.weight.shape[0]
        attention.qkv_proj = fused
        del attention.q_proj, attention.k_proj, attention.v_proj
        attention.__class__ = FusedAttention


def load_mlx_backend(path):
    import inspect
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    if "lazy" in inspect.signature(load).parameters:
        model, tokenizer = load(path, lazy=True)
        _fuse_qwen2_qkv(model)
        mx.eval(model.parameters())
    else:
        model, tokenizer = load(path)
    _cache_vocabulary(tokenizer)
    try:
        greedy_generate = _make_greedy_generate(model, tokenizer)
    except ImportError:
        # Lightweight mlx-lm substitutes and older layouts keep the stock path.
        greedy_generate = None

    def generate(prompt, max_tokens, temperature):
        if temperature == 0 and greedy_generate is not None:
            return greedy_generate(prompt, max_tokens)
        return stream_generate(
            model, tokenizer, prompt, max_tokens=max_tokens,
            sampler=make_sampler(temp=temperature))

    return tokenizer, generate


def pack_message(message):
    return msgpack.packb(message, use_bin_type=True)


def unpack_message(data):
    return msgpack.unpackb(data, raw=False, strict_map_key=False)


class MLXStage:
    def __init__(self, name, recv_endpoint, coordinator_endpoint, abort_endpoint,
                 model_path, loader=load_mlx_backend, perf_log=None):
        self.name = name
        self.recv_endpoint = recv_endpoint
        self.coordinator_endpoint = coordinator_endpoint
        self.abort_endpoint = abort_endpoint
        self.model_path = model_path
        self.loader = loader
        # Optional JSON-lines measurement sink. When None, nothing is written.
        self.perf_log = perf_log
        self.perf_sink = None
        self.tokenizer = None
        self.generate = None
        self.context = None
        self.recv_socket = None
        self.coordinator_socket = None
        self.abort_socket = None
        self.poller = None
        self.pending = deque()
        # request id -> tombstone deadline. See ABORT_TOMBSTONE_TTL_S.
        self.aborted = {}
        self.running = True

    def load(self):
        if self.generate is not None:
            return
        started = time.perf_counter()
        self.tokenizer, self.generate = self.loader(self.model_path)
        if self.perf_log is not None:
            record = {"event": "load",
                      "seconds": time.perf_counter() - started}
            self.add_vocab_stats(record)
            self.write_perf(record)

    def add_vocab_stats(self, record):
        """Copy the real generation counters installed by the loader."""
        stats = getattr(self.tokenizer, "_umminfer_vocab", None)
        if isinstance(stats, dict):
            record["vocab_builds"] = stats["builds"]
            record["vocab_accesses"] = stats["accesses"]
        stats = getattr(self.generate, "prompt_cache_stats", None)
        if isinstance(stats, dict):
            record["prompt_cache_hits"] = stats["hits"]
            record["prompt_cache_misses"] = stats["misses"]
        stats = getattr(self.generate, "detokenizer_stats", None)
        if isinstance(stats, dict):
            record["detokenizer_builds"] = stats["builds"]
            record["detokenizer_copies"] = stats["copies"]

    def write_perf(self, record):
        """Append one JSON line. Measurement must never stop a request."""
        try:
            if self.perf_sink is None:
                self.perf_sink = open(self.perf_log, "a", encoding="utf-8")
            self.perf_sink.write(json.dumps(record, sort_keys=True) + "\n")
            self.perf_sink.flush()
        except OSError as exc:
            print(f"MLX stage perf log write failed: {exc}", file=sys.stderr)

    def start(self):
        self.load()
        if not self.running:
            return
        self.context = zmq.Context()
        self.recv_socket = self.context.socket(zmq.PULL)
        self.recv_socket.RCVHWM = 128
        self.recv_socket.LINGER = 0
        self.recv_socket.bind(self.recv_endpoint)
        self.coordinator_socket = self.context.socket(zmq.PUSH)
        self.coordinator_socket.SNDHWM = 128
        self.coordinator_socket.SNDTIMEO = 1000
        self.coordinator_socket.LINGER = 0
        self.coordinator_socket.connect(self.coordinator_endpoint)
        self.abort_socket = self.context.socket(zmq.SUB)
        self.abort_socket.RCVHWM = 128
        self.abort_socket.LINGER = 0
        self.abort_socket.connect(self.abort_endpoint)
        self.abort_socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.poller = zmq.Poller()
        self.poller.register(self.recv_socket, zmq.POLLIN)
        self.poller.register(self.abort_socket, zmq.POLLIN)
        self.emit({"type": "stage_ready", "from_stage": self.name})
        print(f"MLX stage {self.name} ready", flush=True)

    def close(self):
        self.running = False
        sink = self.perf_sink
        self.perf_sink = None
        if sink is not None:
            sink.close()
        sockets = (self.recv_socket, self.coordinator_socket, self.abort_socket)
        self.recv_socket = self.coordinator_socket = self.abort_socket = None
        context = self.context
        self.context = None
        for socket in sockets:
            if socket is not None:
                socket.close(linger=0)
        if context is not None:
            context.term()

    def emit(self, message):
        self.coordinator_socket.send(pack_message(message))

    def result_metadata(self, parts, max_tokens, last_response):
        """Count and finish reason for one request.

        `parts` and `max_tokens` are in the signature because a stage that does
        not use MLX-LM needs them: the tiny ASR stage counts CTC tokens and
        checks its own truncation. This implementation reads only what MLX-LM
        reported on its last response.
        """
        if last_response is None or not hasattr(last_response, "finish_reason"):
            raise ValueError("MLX-LM did not report finish_reason")
        reason = last_response.finish_reason
        if not isinstance(reason, str) or not reason:
            reason = "unknown"
        count = getattr(last_response, "generation_tokens", None)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("MLX-LM did not report generation_tokens")
        return count, reason

    def complete(self, request_id, success, text="", error=None,
                 generated_tokens=None, finish_reason=None):
        result = {"text": text}
        # Only report counts this stage actually observed.
        if generated_tokens is not None:
            result["generated_token_count"] = generated_tokens
        if finish_reason is not None:
            result["finish_reason"] = finish_reason
        self.emit({"type": "complete", "request_id": request_id,
                   "from_stage": self.name, "success": success,
                   "result": result, "error": error})

    def receive(self, socket):
        return unpack_message(socket.recv())

    def poll_events(self, timeout_ms=0):
        if self.poller is None:
            return
        for index in range(64):
            events = dict(self.poller.poll(timeout_ms if index == 0 else 0))
            if not events:
                return
            if self.abort_socket in events:
                try:
                    message = self.receive(self.abort_socket)
                    request_id = message["request_id"]
                    if message.get("type") != "abort" or not isinstance(request_id, str):
                        raise ValueError("invalid abort message")
                    self.record_abort(request_id)
                except Exception as exc:
                    print(f"MLX stage discarded malformed abort: {exc}", file=sys.stderr)
            if self.recv_socket not in events:
                continue
            if len(self.pending) >= 128:
                return
            try:
                message = self.receive(self.recv_socket)
                if not isinstance(message, dict):
                    raise ValueError("control message must be a map")
                message_type = message.get("type")
                if message_type == "shutdown":
                    self.running = False
                elif message_type == "abort":
                    request_id = message["request_id"]
                    if not isinstance(request_id, str):
                        raise ValueError("invalid abort message")
                    self.record_abort(request_id)
                elif message_type == "submit":
                    self.pending.append(message)
                    if self.perf_log is not None:
                        request_id = message.get("request_id")
                        if isinstance(request_id, str) and request_id:
                            # The submit is in this stage's own queue. An abort
                            # record names whether it removed one of these.
                            self.write_perf({"event": "queue_enqueue",
                                             "request_id": request_id})
                else:
                    print(f"MLX stage ignored message type {message_type!r}", file=sys.stderr)
            except Exception as exc:
                print(f"MLX stage discarded malformed control message: {exc}", file=sys.stderr)

    def expire_aborts(self):
        """Release tombstones whose submit never arrived.

        The deadline is the explicit release point. Without it, an abort for a
        request that is never submitted would keep the map growing.
        """
        now = time.monotonic()
        for request_id in [key for key, deadline in self.aborted.items()
                           if deadline <= now]:
            del self.aborted[request_id]

    def record_abort(self, request_id):
        """Cancel a queued submit at once, and keep a tombstone for a submit
        that arrives later. Capacity is not a cancellation condition."""
        self.expire_aborts()
        removed = any(message.get("request_id") == request_id
                      for message in self.pending)
        self.pending = deque(message for message in self.pending
                             if message.get("request_id") != request_id)
        self.aborted[request_id] = time.monotonic() + ABORT_TOMBSTONE_TTL_S
        if self.perf_log is not None:
            # Separate fact from `queue_enqueue`: a cancellation that removed a
            # real queued submit, not one that only left a tombstone.
            self.write_perf({"event": "abort", "request_id": request_id,
                             "removed_pending": removed})

    def consume_abort(self, request_id):
        self.expire_aborts()
        return self.aborted.pop(request_id, None) is not None

    def parse_request(self, message):
        data = message["data"]
        request = data["request"]
        inputs = request["inputs"]
        messages = inputs["messages"]
        params = request.get("params", {})
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a nonempty array")
        if not all(isinstance(item, dict) for item in messages):
            raise ValueError("each message must be a map")
        if not isinstance(params, dict):
            raise ValueError("params must be a map")
        max_tokens = params.get("max_new_tokens", 256)
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise ValueError("max_new_tokens must be an integer")
        if not 1 <= max_tokens <= 2**31 - 1:
            raise ValueError("max_new_tokens is outside the supported range")
        temperature = params.get("temperature", 0.0)
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise ValueError("temperature must be a number")
        if not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        stream = params.get("stream", False)
        if not isinstance(stream, bool):
            raise ValueError("stream must be a boolean")
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("chat template produced an empty prompt")
        return prompt, max_tokens, float(temperature), stream

    def process(self, message):
        request_id = message.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            print("MLX stage discarded submit without a request id", file=sys.stderr)
            return
        if self.consume_abort(request_id):
            return
        try:
            prompt, max_tokens, temperature, stream = self.parse_request(message)
            if self.perf_log is not None:
                # Entry evidence, written after the abort check and the request
                # parse, before the first model step. A request that never
                # reaches the model has no start record; a request aborted
                # mid-generation has one and no completion record.
                start_record = {"event": "request_start",
                                "request_id": request_id}
                self.add_vocab_stats(start_record)
                self.write_perf(start_record)
            started = time.perf_counter()
            first_token_at = None
            parts = []
            last_response = None
            for response in self.generate(prompt, max_tokens, temperature):
                self.poll_events()
                if self.consume_abort(request_id) or not self.running:
                    return
                delta = response.text
                if not isinstance(delta, str):
                    raise ValueError("MLX-LM response text must be a string")
                # First-token latency is the first non-empty delta. An empty
                # GenerationResponse is not a token.
                if self.perf_log is not None and first_token_at is None and delta:
                    first_token_at = time.perf_counter()
                parts.append(delta)
                last_response = response
                if stream and delta:
                    self.emit({"type": "stream", "request_id": request_id,
                               "from_stage": self.name, "chunk": {"text": delta}})
            self.poll_events()
            if self.consume_abort(request_id) or not self.running:
                return
            generated_tokens, finish_reason = self.result_metadata(
                parts, max_tokens, last_response)
            # Write the record before `complete`: a client that sees the
            # completion can then read this record with no race.
            if self.perf_log is not None:
                finished = time.perf_counter()
                record = {
                    "event": "request",
                    "request_id": request_id,
                    "stream": stream,
                    "stage_ttft_seconds": (None if first_token_at is None else
                                           first_token_at - started),
                    "stage_total_seconds": finished - started,
                    "generation_tokens": generated_tokens,
                    "finish_reason": finish_reason,
                    "prompt_tokens": getattr(last_response, "prompt_tokens", None),
                    "prompt_tps": getattr(last_response, "prompt_tps", None),
                    "generation_tps": getattr(last_response, "generation_tps", None),
                    "peak_memory_gb": getattr(last_response, "peak_memory", None),
                }
                self.add_vocab_stats(record)
                self.write_perf(record)
            self.complete(request_id, True, "".join(parts),
                          generated_tokens=generated_tokens,
                          finish_reason=finish_reason)
        except Exception as exc:
            self.poll_events()
            if self.consume_abort(request_id) or not self.running:
                return
            error = str(exc) or type(exc).__name__
            print(f"MLX request {request_id} failed: {error}", file=sys.stderr)
            self.complete(request_id, False, error=error)

    def run(self):
        try:
            while self.running:
                if self.pending:
                    self.process(self.pending.popleft())
                else:
                    self.poll_events(100)
        finally:
            self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--recv", required=True)
    parser.add_argument("--coordinator", required=True)
    parser.add_argument("--abort", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--perf-log", default=None,
                        help="append one JSON line per load and per request")
    args = parser.parse_args()
    stage = MLXStage(args.name, args.recv, args.coordinator, args.abort, args.model,
                     perf_log=args.perf_log)

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

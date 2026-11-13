# SPDX-License-Identifier: Apache-2.0
"""Out-of-process rollout recording and multimodal trajectory export.

The recorder is a normal client of the public HTTP/SSE endpoint. Serving code
is not modified and no inference engine is duplicated. One JSONL line is one
terminal request (one turn). Records contain no wall-clock fields and no
random ids: event order is gap-free sequence numbers, media is
content-addressed by SHA-256, so the same inputs and response bytes always
produce the same record bytes.

Media bytes never live in the JSONL. WAV audio and inline image bytes are
written to an artifact directory under their SHA-256 name; the record keeps a
relative path plus the hash. Remote image URLs stay references with a null
hash: the bytes are never fetched.
"""

import base64
import hashlib
import http.client
import json
from pathlib import Path
from urllib.parse import unquote

TRAJECTORY_SCHEMA_VERSION = 1

# Terminal states observed at the public exit.
TERMINAL_COMPLETED = "completed"
TERMINAL_FAILED = "failed"
TERMINAL_CANCELLED = "cancelled"

# Truncation message kept identical to tests/unified_benchmark.py so both
# consumers classify an EOF without [DONE] the same way.
TRUNCATED_MESSAGE = "stream ended before [DONE]"


def canonical_json(value):
    """Stable serialization for every hash in the RL modules.

    allow_nan=False rejects NaN/Infinity, which are not valid JSON tokens:
    a record containing them could not be parsed by a strict consumer.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_text(text):
    return sha256_bytes(text.encode("utf-8"))


def iter_sse_payloads(response):
    """Yield the decoded payload of each SSE ``data:`` frame.

    Takes any file-like object yielding text lines. A blank line dispatches
    one event, matching tests/unified_benchmark.py:sse_events. ``[DONE]`` is
    yielded as the literal string.
    """
    data_lines = []
    while True:
        line = response.readline()
        if line == b"" or line == "":
            if data_lines:
                yield "\n".join(data_lines)
            return
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        line = line.rstrip("\r\n")
        if line == "":
            if data_lines:
                payload = "\n".join(data_lines)
                data_lines = []
                yield payload
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip(" "))


def parse_sse_stream(response, *, on_request_id=None):
    """Parse an SSE chat response.

    Returns ``(request_id, events, output_text, terminal_state, error)``.
    Unified ASR/VLM events keep their server ``seq`` under ``server_seq``;
    recorder-local ``seq`` is gap-free from 0. A clean ``[DONE]`` completes the
    stream, an error frame fails it, and EOF without ``[DONE]`` fails it with
    the same message the benchmark uses. Cancellation is attributed only when
    the caller itself aborts; see :func:`abort_request`.
    """
    events = []
    output_parts = []
    terminal_state = TERMINAL_FAILED
    error = TRUNCATED_MESSAGE
    request_id = None
    # A server-emitted unified terminal decides the state even when [DONE]
    # follows: the server sends kind=cancelled plus [DONE] for an accepted
    # abort (openai_api.cpp kCancelled path), which must not read completed.
    unified_terminal = None

    def add(event_type, **fields):
        event = {"seq": len(events), "type": event_type}
        event.update(fields)
        events.append(event)

    for payload in iter_sse_payloads(response):
        if payload == "[DONE]":
            if unified_terminal is not None:
                terminal_state = unified_terminal
                if unified_terminal == TERMINAL_CANCELLED:
                    error = None
            elif not any(e["type"] == "error" for e in events):
                terminal_state = TERMINAL_COMPLETED
                error = None
            break
        chunk = json.loads(payload)
        if chunk.get("id"):
            if request_id is None and on_request_id is not None:
                on_request_id(chunk["id"])
            request_id = chunk["id"]
        if "error" in chunk:
            # An explicit SSE error frame is authoritative even if a unified
            # terminal frame also appeared.
            unified_terminal = None
            terminal_state = TERMINAL_FAILED
            error = chunk["error"].get("message", "stream error")
            add("error", text=error)
            continue
        unified = chunk.get("unified")
        if unified is not None:
            kind = unified["kind"]
            text = unified.get("text", "")
            add(kind,
                server_seq=unified.get("seq"),
                stage=unified.get("stage"),
                is_final=bool(unified.get("is_final")),
                text=text)
            if kind == "cancelled":
                unified_terminal = TERMINAL_CANCELLED
            elif kind == "error":
                unified_terminal = TERMINAL_FAILED
                error = text or error
            # Partials are provisional and replaced by the final, so they do
            # not contribute to the hashed output. The final ASR transcript
            # and every VLM delta are assistant output.
            if kind in ("asr_final", "vlm_delta"):
                output_parts.append(text)
            continue
        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            content = delta.get("content")
            if content:
                output_parts.append(content)
                add("output_delta", text=content)
            if choice.get("finish_reason"):
                add("finish", finish_reason=choice["finish_reason"])
    if unified_terminal is not None and terminal_state == TERMINAL_FAILED:
        # EOF before [DONE] after a server terminal still trusts the terminal.
        terminal_state = unified_terminal
        if unified_terminal == TERMINAL_CANCELLED:
            error = None
    return request_id, events, "".join(output_parts), terminal_state, error


def parse_chat_response(body):
    """Non-streaming chat.completion body -> (events, output_text, state, error)."""
    if "error" in body:
        message = body["error"].get("message", "request error")
        return ([{"seq": 0, "type": "error", "text": message}], "",
                TERMINAL_FAILED, message)
    if not body.get("choices"):
        message = "response carried neither choices nor an error"
        return ([{"seq": 0, "type": "error", "text": message}], "",
                TERMINAL_FAILED, message)
    try:
        choice = body["choices"][0]
        text = choice["message"]["content"]
    except (KeyError, TypeError, IndexError):
        message = "response choices[0] is missing message.content"
        return ([{"seq": 0, "type": "error", "text": message}], "",
                TERMINAL_FAILED, message)
    events = [{"seq": 0, "type": "complete", "stage": None,
               "is_final": True, "text": text}]
    if choice.get("finish_reason"):
        events.append({"seq": 1, "type": "finish",
                       "finish_reason": choice["finish_reason"]})
    return events, text, TERMINAL_COMPLETED, None


def input_references(messages, artifact_dir):
    """Reference every input part of a chat request, in request order.

    Returns ``{kind, ref, media_type, sha256, text?}`` dicts. Text is inlined
    and hashed; WAV audio and ``data:`` image bytes are content-addressed into
    ``artifact_dir``; remote image URLs stay references with a null hash.
    """
    artifact_dir = Path(artifact_dir) if artifact_dir else None
    references = []
    for message in messages:
        content = message["content"]
        parts = [content] if isinstance(content, str) else content
        for part in parts:
            if isinstance(part, str):
                references.append(_text_ref(part))
                continue
            part_type = part["type"]
            if part_type == "text":
                references.append(_text_ref(part["text"]))
            elif part_type == "input_audio":
                audio = part["input_audio"]
                raw = base64.b64decode(audio["data"])
                digest = sha256_bytes(raw)
                suffix = "." + audio["format"]
                references.append({
                    "kind": "audio",
                    "ref": _store_artifact(artifact_dir, digest, suffix, raw),
                    "media_type": "audio/" + suffix[1:],
                    "sha256": digest})
            elif part_type == "image_url":
                url = part["image_url"]["url"]
                if url.startswith("data:"):
                    media_type, raw = _decode_data_url(url)
                    digest = sha256_bytes(raw)
                    ext = media_type.split("/", 1)[1].split(";", 1)[0]
                    references.append({
                        "kind": "image",
                        "ref": _store_artifact(artifact_dir, digest,
                                               "." + ext, raw),
                        "media_type": media_type,
                        "sha256": digest})
                else:
                    references.append({"kind": "image", "ref": url,
                                       "media_type": None, "sha256": None})
            else:
                raise ValueError(f"unsupported content part type: {part_type}")
    return references


def _text_ref(text):
    return {"kind": "text", "ref": None, "media_type": None,
            "sha256": sha256_text(text), "text": text}


def _store_artifact(artifact_dir, digest, suffix, raw):
    if artifact_dir is None:
        return None
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / (digest + suffix)
    if not path.exists():
        path.write_bytes(raw)
    return path.name


def _decode_data_url(url):
    header, encoded = url.split(",", 1)
    media_type = header[5:].split(";", 1)[0]
    if "base64" in header:
        raw = base64.b64decode(encoded)
    else:
        raw = unquote(encoded).encode()
    return media_type, raw


class TrajectoryRecorder:
    """Appends immutable trajectory records to one JSONL file.

    Turn counters are process-local state. A trajectory file has one writer;
    multiple recorder processes must not append to the same path.
    """

    def __init__(self, jsonl_path, artifact_dir=None):
        self.jsonl_path = Path(jsonl_path)
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._turns = {}

    def next_turn(self, session_id):
        index = self._turns.get(session_id, 0)
        self._turns[session_id] = index + 1
        return index

    def record(self, *, session_id, request_id, endpoint, inputs, params,
               events, terminal_state, error=None, output_text=""):
        record = {
            "schema_version": TRAJECTORY_SCHEMA_VERSION,
            "session_id": session_id,
            "request_id": request_id,
            "turn_index": self.next_turn(session_id),
            "endpoint": endpoint,
            "inputs": inputs,
            "request_params": params,
            "events": events,
            "terminal_state": terminal_state,
            "error": error,
            "output_text": output_text,
            "output_sha256": sha256_text(output_text),
        }
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True,
                                    ensure_ascii=False, allow_nan=False) + "\n")
        return record


def _http_connection(base_url, timeout):
    return http.client.HTTPConnection(base_url, timeout=timeout)


def abort_request(base_url, request_id, *, headers=None, timeout=30.0):
    """Call the public POST /abort exit.

    Returns ``(status, body)``. Fixed-upstream deployments answer 501
    ``unsupported_route``: the abort is not backend-confirmed there, and the
    caller must not label the turn cancelled. A 200 means the coordinator
    accepted the abort; it is still a client-attributed terminal state, not a
    guarantee that the model forward already stopped.
    """
    headers = dict(headers or {})
    headers["Content-Type"] = "application/json"
    conn = _http_connection(base_url, timeout)
    try:
        conn.request("POST", "/abort",
                     body=json.dumps({"request_id": request_id}),
                     headers=headers)
        response = conn.getresponse()
        return response.status, json.loads(response.read().decode("utf-8"))
    finally:
        conn.close()


def record_chat_completion(base_url, body, recorder, session_id, *,
                           headers=None, timeout=30.0, on_request_id=None):
    """Drive one chat request through the public endpoint and record it.

    Transport failures (refused connection, dropped socket, malformed
    response bytes, a success body missing the documented shape, or a body
    this function cannot serialize) produce a ``failed`` record instead of
    escaping: every attempted turn stays in the export, matching the roadmap
    export gate.

    ``on_request_id`` is invoked with the server request id as soon as the
    first response chunk arrives. A caller can pass a callback that calls
    :func:`abort_request` from another thread: that is the supported way to
    drive the public cancel exit while this function drains the response.
    The server then emits a unified ``cancelled`` terminal and the record
    ends in the ``cancelled`` state.
    """
    headers = dict(headers or {})
    headers["Content-Type"] = "application/json"
    conn = _http_connection(base_url, timeout)
    request_id = None
    inputs = []
    try:
        try:
            inputs = input_references(body["messages"],
                                      recorder.artifact_dir)
            conn.request("POST", "/v1/chat/completions",
                         body=json.dumps(body), headers=headers)
            response = conn.getresponse()
            is_sse = response.getheader("Content-Type", "").startswith(
                "text/event-stream")
            if body.get("stream") and is_sse:
                request_id, events, output_text, state, error = (
                    parse_sse_stream(response, on_request_id=on_request_id))
            else:
                payload = json.loads(response.read().decode("utf-8"))
                request_id = payload.get("id")
                events, output_text, state, error = parse_chat_response(
                    payload)
        except (OSError, http.client.HTTPException, json.JSONDecodeError,
                UnicodeDecodeError, TypeError, KeyError, ValueError) as exc:
            message = f"transport error: {exc}"
            events = [{"seq": 0, "type": "error", "text": message}]
            output_text, state, error = "", TERMINAL_FAILED, message
    finally:
        conn.close()

    return recorder.record(
        session_id=session_id,
        request_id=request_id or _synthetic_request_id(events),
        endpoint="/v1/chat/completions",
        inputs=inputs,
        params={k: v for k, v in body.items() if k != "messages"},
        events=events,
        terminal_state=state,
        error=error,
        output_text=output_text)


def record_realtime(recorder, session_id, request_id, observed_events, *,
                    inputs=None, terminal_state=None, error=None):
    """Record a /v1/realtime turn from events observed by a WS client.

    This function takes no network dependency: the caller supplies the dicts
    it received or sent over the WebSocket, each tagged ``direction``
    (``inbound`` for client-to-server, ``outbound`` for server-to-client).
    ``inputs`` accepts reference dicts shaped like :func:`input_references`
    output; the inbound PCM concatenation/hash rule itself is deferred with
    the live WS adapter (see docs/rl.md). Terminal state is derived from the
    outbound events when not given: ``transcription.completed`` completes, an
    ``error`` fails, and neither appearing means the client observed a
    cancellation.
    """
    events = []
    output_parts = []
    derived_state = TERMINAL_CANCELLED
    derived_error = None
    for index, observed in enumerate(observed_events):
        event_type = observed["type"]
        event = {"seq": index, "type": event_type,
                 "direction": observed["direction"]}
        if "text" in observed:
            event["text"] = observed["text"]
        if observed.get("is_final") is not None:
            event["is_final"] = bool(observed["is_final"])
        events.append(event)
        if observed["direction"] == "outbound":
            if event_type == "transcription.completed":
                derived_state = TERMINAL_COMPLETED
                output_parts.append(observed.get("text", ""))
            elif event_type == "error":
                derived_state = TERMINAL_FAILED
                derived_error = observed.get("error", {}).get("message")
    state = terminal_state or derived_state
    output_text = "".join(output_parts)
    return recorder.record(
        session_id=session_id,
        request_id=request_id,
        endpoint="/v1/realtime",
        inputs=list(inputs or []),
        params={},
        events=events,
        terminal_state=state,
        error=error or derived_error,
        output_text=output_text)


def _synthetic_request_id(events):
    # Deterministic fallback when the exit carried no id (for example a
    # truncated stream). Content-addressed, never random.
    return "rollout-" + sha256_bytes(canonical_json(events))[:16]

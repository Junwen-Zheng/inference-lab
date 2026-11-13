"""Streaming client for the OpenAI-compatible completions endpoint."""
from __future__ import annotations

import http.client
import json
import math
import os
import socket
import threading
import time
from urllib.parse import urlsplit

from .metrics import RequestResult


def iter_sse_data(lines):
    fields = []
    for line in lines:
        line = line.decode("utf-8") if isinstance(line, bytes) else line
        line = line.rstrip("\r\n")
        if not line:
            if fields:
                yield "\n".join(fields)
                fields = []
        elif line.startswith("data:"):
            value = line[5:]
            fields.append(value[1:] if value.startswith(" ") else value)
    if fields:
        yield "\n".join(fields)


def request_completion(base_url: str, model: str, prompt: str, max_tokens: int = 64,
                       timeout_s: float = 60.0, request_id: int = 0,
                       ignore_eos: bool = False) -> RequestResult:
    result = RequestResult(request_id=request_id)
    started = time.perf_counter()
    connection = None
    response = None
    deadline_timer = None
    try:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        if not isinstance(model, str) or not model.strip() or not isinstance(prompt, str):
            raise ValueError("model must be nonempty text and prompt must be text")
        url = urlsplit(base_url)
        if url.scheme not in ("http", "https") or not url.hostname or url.query or url.fragment:
            raise ValueError("base URL must be an http(s) origin, optionally ending in /v1")
        prefix = url.path.rstrip("/")
        if prefix not in ("", "/v1") or url.username or url.password:
            raise ValueError("base URL path must be empty or /v1; use INFERENCE_API_KEY for auth")
        cls = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
        connection = cls(url.hostname, url.port, timeout=timeout_s)
        connection.connect()
        stream_socket = connection.sock

        def close_at_deadline():
            # Interrupt even a peer trickling a line/header without a newline.
            try:
                stream_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        remaining = timeout_s - (time.perf_counter() - started)
        if remaining <= 0:
            raise TimeoutError("connection exceeded request time budget")
        deadline_timer = threading.Timer(remaining, close_at_deadline)
        deadline_timer.daemon = True
        deadline_timer.start()
        payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
                   "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
        if ignore_eos:
            payload["ignore_eos"] = True
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        api_key = os.environ.get("INFERENCE_API_KEY")
        if api_key:
            headers["Authorization"] = "Bearer " + api_key
        connection.request("POST", "/v1/completions", json.dumps(payload), headers)
        response = connection.getresponse()
        result.status_code = response.status
        if response.status != 200:
            # Do not persist server bodies, which may contain prompt data or secrets.
            raise ValueError(f"HTTP {response.status} {response.reason}")
        if "text/event-stream" not in response.getheader("Content-Type", ""):
            raise ValueError("expected text/event-stream response")
        first_at = previous_at = None
        done = False

        def timed_lines():
            # Connection: close can detach the socket from HTTPConnection, so use
            # the socket captured before getresponse to enforce the deadline.
            while True:
                remaining = timeout_s - (time.perf_counter() - started)
                if remaining <= 0:
                    raise TimeoutError("request exceeded total time budget")
                stream_socket.settimeout(remaining)
                line = response.readline()
                if not line:
                    return
                yield line

        for frame in iter_sse_data(timed_lines()):
            now = time.perf_counter()
            if now - started > timeout_s:
                raise TimeoutError("request exceeded total time budget")
            if frame == "[DONE]":
                done = True
                break
            event = json.loads(frame)
            if not isinstance(event, dict):
                raise ValueError("SSE payload must be a JSON object")
            if "error" in event:
                raise ValueError("server returned an SSE error event")
            usage = event.get("usage")
            if usage is not None:
                if not isinstance(usage, dict):
                    raise ValueError("usage must be an object")
                for key in ("prompt_tokens", "completion_tokens"):
                    if not isinstance(usage.get(key), int) or isinstance(usage[key], bool) or usage[key] < 0:
                        raise ValueError("invalid token usage from server")
                result.prompt_tokens = usage["prompt_tokens"]
                result.output_tokens = usage["completion_tokens"]
            choices = event.get("choices", [])
            if not isinstance(choices, list) or any(not isinstance(item, dict) for item in choices):
                raise ValueError("choices must be a list of objects")
            content = choices[0].get("text", "") if choices else ""
            if not isinstance(content, str):
                raise ValueError("completion text must be a string")
            if content:
                if first_at is None:
                    first_at = now
                    result.ttft_s = now - started
                if previous_at is not None:
                    result.chunk_gaps_s.append(now - previous_at)
                previous_at = now
                result.text += content
        if not done:
            raise ValueError("truncated stream: missing [DONE]")
        if result.output_tokens is None:
            raise ValueError("server did not provide usage; cannot measure token throughput")
        if first_at is None or result.output_tokens == 0:
            raise ValueError("no generated text tokens")
        result.e2e_s = time.perf_counter() - started
        if result.output_tokens > 1:
            result.tpot_s = (result.e2e_s - result.ttft_s) / (result.output_tokens - 1)
        result.success = True
    except (OSError, ValueError, TypeError, KeyError, IndexError, http.client.HTTPException) as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        result.e2e_s = time.perf_counter() - started
    finally:
        if deadline_timer:
            deadline_timer.cancel()
        for resource in (response, connection):
            if resource is not None:
                try:
                    resource.close()
                except (OSError, http.client.HTTPException):
                    pass
    return result

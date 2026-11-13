"""Synthetic streaming protocol fixture. This is not an inference engine."""
from __future__ import annotations

import argparse
import json
import math
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class MockHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "InferenceLabMock/1"

    def log_message(self, format, *args):
        pass

    def json_response(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Inference-Lab-Backend", "mock")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self.json_response(200, {"status": "ok", "backend": "mock"})
        elif self.path == "/v1/models":
            self.json_response(200, {"data": [{"id": "inference-model"}]})
        else:
            self.json_response(404, {"error": "unknown endpoint"})

    def do_POST(self):
        if self.path != "/v1/completions":
            self.json_response(404, {"error": "unknown endpoint"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 1_000_000:
                raise ValueError("invalid request size")
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            prompt = request.get("prompt")
            if not isinstance(prompt, str):
                raise ValueError("prompt must be a string")
            n = request.get("max_tokens", 16)
            if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 4096:
                raise ValueError("max_tokens must be an integer from 1 to 4096")
            if request.get("model") != "inference-model" or request.get("stream") is not True:
                raise ValueError("use inference-model with stream=true")
        except (ValueError, TypeError) as exc:
            self.json_response(400, {"error": str(exc)})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Inference-Lab-Backend", "mock")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        def emit(payload):
            value = payload if isinstance(payload, str) else json.dumps(payload)
            self.wfile.write(("data: " + value + "\n\n").encode())
            self.wfile.flush()

        try:
            time.sleep(self.server.first_delay_s)
            for i in range(n):
                if i:
                    time.sleep(self.server.token_delay_s)
                emit({"choices": [{"text": f"synthetic_{i} ", "finish_reason": None}]})
            emit({"choices": [{"text": "", "finish_reason": "length"}]})
            # These counts describe the synthetic fixture, not tokenizer output.
            emit({"choices": [], "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": n, "total_tokens": n + len(prompt.split())}})
            emit("[DONE]")
        except (BrokenPipeError, ConnectionResetError):
            pass


class MockServer(ThreadingHTTPServer):
    # The default backlog of five can reset connection bursts on macOS.
    request_queue_size = 128


def make_server(host="127.0.0.1", port=8000, first_delay_s=.015, token_delay_s=.002):
    if any(isinstance(delay, bool) or not isinstance(delay, (int, float))
           or not math.isfinite(delay) or delay < 0
           for delay in (first_delay_s, token_delay_s)):
        raise ValueError("mock delays must be finite nonnegative numbers")
    server = MockServer((host, port), MockHandler)
    server.daemon_threads = True
    server.first_delay_s = first_delay_s
    server.token_delay_s = token_delay_s
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    server = make_server(args.host, args.port)
    print(f"SYNTHETIC mock listening on http://{args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

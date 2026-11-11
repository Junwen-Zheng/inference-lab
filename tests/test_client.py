import json
import math
import socket
import unittest
from unittest.mock import patch

from inference_lab.client import iter_sse_data, request_completion


def frames(*events):
    lines = []
    for event in events:
        text = event if isinstance(event, str) else json.dumps(event)
        lines.extend([("data: " + text + "\n").encode(), b"\n"])
    return lines


def valid_stream(tokens=3):
    return frames({"choices": [{"text": "hello"}]},
                  {"choices": [], "usage": {"prompt_tokens": 12,
                                             "completion_tokens": tokens}},
                  "[DONE]")


class Response:
    def __init__(self, lines, status=200, content_type="text/event-stream"):
        self.lines = iter(lines)
        self.status = status
        self.reason = "Test response"
        self.content_type = content_type
        self.closed = False

    def getheader(self, name, default=None):
        return self.content_type

    def readline(self):
        item = next(self.lines, b"")
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        self.closed = True


class ClientTests(unittest.TestCase):
    def call(self, lines, *, response=None, **kwargs):
        response = response if response is not None else Response(lines)
        with patch("inference_lab.client.http.client.HTTPConnection") as factory, \
                patch("inference_lab.client.threading.Timer") as timer:
            connection = factory.return_value
            connection.getresponse.return_value = response
            arguments = {"base_url": "http://localhost:8000", "model": "inference-model",
                         "prompt": "Explain batching", **kwargs}
            result = request_completion(**arguments)
            connection.close.assert_called_once()
            timer.return_value.cancel.assert_called_once()
        self.assertTrue(response.closed)
        return result, connection

    def test_sse_handles_comments_multiline_data_and_crlf(self):
        lines = [b": keepalive\r\n", b"event: completion\r\n", b"data: hello\r\n",
                 b"data: world\r\n", b"\r\n"]
        self.assertEqual(list(iter_sse_data(lines)), ["hello\nworld"])

    def test_usage_counts_tokens_instead_of_stream_chunks(self):
        result, _ = self.call(valid_stream(tokens=5))
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.text, "hello")
        self.assertEqual(result.prompt_tokens, 12)
        self.assertEqual(result.output_tokens, 5)
        self.assertEqual(result.chunk_gaps_s, [])

    def test_single_token_has_no_tpot(self):
        result, _ = self.call(valid_stream(tokens=1))
        self.assertTrue(result.success, result.error)
        self.assertIsNone(result.tpot_s)

    def test_truncated_stream_and_missing_usage_are_failures(self):
        for stream, error in ((frames({"choices": [{"text": "hello"}]}), "truncated"),
                              (frames({"choices": [{"text": "hello"}]}, "[DONE]"), "usage")):
            with self.subTest(error=error):
                result, _ = self.call(stream)
                self.assertFalse(result.success)
                self.assertIn(error, result.error)

    def test_malformed_stream_events_are_recorded_as_failures(self):
        events = ["not JSON", "[]", {"usage": [1]}, {"choices": [None]},
                  {"choices": "invalid"}, {"choices": [{"text": 123}]},
                  {"usage": {"prompt_tokens": 1, "completion_tokens": True}},
                  {"usage": {"prompt_tokens": -1, "completion_tokens": 2}},
                  {"error": {"message": "backend failed"}}]
        for event in events:
            with self.subTest(event=event):
                result, _ = self.call(frames(event))
                self.assertFalse(result.success)
                self.assertIsNotNone(result.error)

    def test_http_errors_and_wrong_content_type_do_not_become_successes(self):
        for response in (Response([], status=503), Response([], content_type="application/json")):
            with self.subTest(status=response.status, content_type=response.content_type):
                result, _ = self.call([], response=response)
                self.assertFalse(result.success)
                self.assertIsNotNone(result.error)

    def test_socket_timeout_is_a_request_failure_and_resources_close(self):
        result, _ = self.call([socket.timeout("read timed out")])
        self.assertFalse(result.success)
        self.assertIn("timed out", result.error)

    def test_endpoint_payload_and_optional_authentication(self):
        with patch.dict("os.environ", {"INFERENCE_API_KEY": "test-key"}):
            result, connection = self.call(valid_stream(), base_url="http://localhost:8000/v1",
                                           max_tokens=8, ignore_eos=True)
        method, path, body, headers = connection.request.call_args.args
        self.assertEqual((method, path), ("POST", "/v1/completions"))
        payload = json.loads(body)
        self.assertEqual(payload["max_tokens"], 8)
        self.assertTrue(payload["stream"])
        self.assertTrue(payload["stream_options"]["include_usage"])
        self.assertTrue(payload["ignore_eos"])
        self.assertEqual(headers["Authorization"], "Bearer test-key")
        self.assertNotIn("test-key", repr(result))

    def test_invalid_request_settings_fail_before_connecting(self):
        updates = [{"timeout_s": v} for v in (0, -1, math.nan, math.inf)]
        updates += [{"max_tokens": v} for v in (0, -1, True, 1.5)]
        updates += [{"base_url": "ftp://localhost"}, {"base_url": "http://localhost/wrong"}]
        for update in updates:
            with self.subTest(update=update), \
                    patch("inference_lab.client.http.client.HTTPConnection") as factory:
                arguments = {"base_url": "http://localhost", "model": "m", "prompt": "p", **update}
                result = request_completion(**arguments)
                self.assertFalse(result.success)
                factory.assert_not_called()

    def test_ttft_e2e_tpot_and_chunk_gaps_use_client_clock(self):
        clock = [0.0]
        lines = frames({"choices": [{"text": "A"}]}, {"choices": [{"text": "BC"}]},
                       {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 3}},
                       "[DONE]")
        timings = iter([.10, .10, .30, .30, .31, .31, .32, .32])
        response = Response(lines)
        original_readline = response.readline

        def read():
            clock[0] = next(timings, .32)
            return original_readline()

        response.readline = read
        with patch("inference_lab.client.time.perf_counter", side_effect=lambda: clock[0]):
            result, _ = self.call([], response=response)
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.text, "ABC")
        self.assertAlmostEqual(result.ttft_s, .10)
        self.assertAlmostEqual(result.e2e_s, .32)
        self.assertAlmostEqual(result.tpot_s, .11)
        self.assertAlmostEqual(result.chunk_gaps_s[0], .20)

    def test_total_stream_budget_expires_even_when_lines_keep_arriving(self):
        clock = [0.0]
        response = Response([])

        def read():
            clock[0] += .06
            return b": keepalive\n"

        response.readline = read
        with patch("inference_lab.client.time.perf_counter", side_effect=lambda: clock[0]):
            result, _ = self.call([], response=response, timeout_s=.1)
        self.assertFalse(result.success)
        self.assertIn("time budget", result.error)


if __name__ == "__main__":
    unittest.main()

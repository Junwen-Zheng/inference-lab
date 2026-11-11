"""Exercise the synthetic fixture over real loopback HTTP connections."""
import concurrent.futures
import http.client
import json
import math
import threading
import unittest

from inference_lab.client import iter_sse_data, request_completion
from inference_lab.mock_server import make_server


class MockServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = make_server("127.0.0.1", 0, first_delay_s=.002,
                                 token_delay_s=.001)
        cls.addClassCleanup(cls.server.server_close)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

        def stop():
            cls.server.shutdown()
            cls.thread.join(timeout=5)

        cls.addClassCleanup(stop)
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    def request(self, method, path, body=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port,
                                                 timeout=5)
        try:
            connection.request(method, path, body=body,
                               headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    @staticmethod
    def payload(**updates):
        return {"model": "inference-model", "prompt": "explain the fixture",
                "max_tokens": 3, "stream": True, **updates}

    def test_health_and_models_identify_the_mock_backend(self):
        for path in ("/health", "/v1/models"):
            with self.subTest(path=path):
                status, headers, body = self.request("GET", path)
                self.assertEqual(status, 200)
                self.assertEqual(headers["X-Inference-Lab-Backend"], "mock")
                data = json.loads(body)
                if path == "/health":
                    self.assertEqual(data, {"status": "ok", "backend": "mock"})
                else:
                    self.assertEqual([item["id"] for item in data["data"]],
                                     ["inference-model"])

    def test_stream_has_synthetic_text_usage_and_done(self):
        status, headers, body = self.request(
            "POST", "/v1/completions", json.dumps(self.payload()))
        self.assertEqual(status, 200)
        self.assertIn("text/event-stream", headers["Content-Type"])
        self.assertEqual(headers["X-Inference-Lab-Backend"], "mock")
        frames = list(iter_sse_data(body.splitlines(keepends=True)))
        self.assertEqual(frames[-1], "[DONE]")
        events = [json.loads(frame) for frame in frames[:-1]]
        text = "".join(choice.get("text", "") for event in events
                       for choice in event.get("choices", []))
        self.assertEqual(text, "synthetic_0 synthetic_1 synthetic_2 ")
        self.assertTrue(any(choice.get("finish_reason") == "length"
                            for event in events for choice in event.get("choices", [])))
        self.assertEqual(events[-1]["usage"], {"prompt_tokens": 3,
                                             "completion_tokens": 3,
                                             "total_tokens": 6})

    def test_day_three_client_reads_text_usage_and_timings(self):
        for count in (1, 4):
            with self.subTest(synthetic_pieces=count):
                result = request_completion(self.base_url, "inference-model",
                                            "one two", max_tokens=count, timeout_s=5)
                self.assertTrue(result.success, result.error)
                self.assertEqual(result.status_code, 200)
                self.assertEqual(result.text,
                                 "".join(f"synthetic_{i} " for i in range(count)))
                self.assertEqual(result.prompt_tokens, 2)
                self.assertEqual(result.output_tokens, count)
                self.assertIsNotNone(result.ttft_s)
                self.assertGreaterEqual(result.ttft_s, 0)
                self.assertGreaterEqual(result.e2e_s, result.ttft_s)
                self.assertEqual(len(result.chunk_gaps_s), count - 1)
                self.assertTrue(all(gap >= 0 for gap in result.chunk_gaps_s))
                if count == 1:
                    self.assertIsNone(result.tpot_s)
                else:
                    self.assertIsNotNone(result.tpot_s)
                    self.assertGreaterEqual(result.tpot_s, 0)

    def test_malformed_json_and_non_object_bodies_return_400(self):
        for body in (b"{broken", b"[]", b"null", b"\xff"):
            with self.subTest(body=body):
                status, headers, response = self.request("POST", "/v1/completions", body)
                self.assertEqual(status, 400)
                self.assertEqual(headers["X-Inference-Lab-Backend"], "mock")
                self.assertIn("error", json.loads(response))

    def test_invalid_completion_fields_return_400(self):
        updates = [{"prompt": value} for value in (None, 3, ["text"])]
        updates += [{"max_tokens": value} for value in (0, -1, True, 1.5, "3", 4097)]
        updates += [{"model": "unknown-model"}, {"stream": False}]
        for update in updates:
            with self.subTest(update=update):
                status, _, body = self.request(
                    "POST", "/v1/completions", json.dumps(self.payload(**update)))
                self.assertEqual(status, 400)
                self.assertIn("error", json.loads(body))

    def test_unknown_get_and_post_endpoints_return_404(self):
        for method in ("GET", "POST"):
            with self.subTest(method=method):
                status, _, body = self.request(method, "/unknown", b"{}")
                self.assertEqual(status, 404)
                self.assertIn("error", json.loads(body))

    def test_eight_simultaneous_clients_complete_independently(self):
        barrier = threading.Barrier(8)

        def complete(request_id):
            barrier.wait(timeout=5)
            return request_completion(self.base_url, "inference-model", "burst test",
                                      max_tokens=2, timeout_s=5, request_id=request_id)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(complete, range(8)))
        self.assertEqual({result.request_id for result in results}, set(range(8)))
        for result in results:
            self.assertTrue(result.success, result.error)
            self.assertEqual(result.output_tokens, 2)
            self.assertEqual(result.text, "synthetic_0 synthetic_1 ")

    def test_server_rejects_negative_and_nonfinite_delays(self):
        for setting in ("first_delay_s", "token_delay_s"):
            for value in (-.001, math.nan, math.inf):
                with self.subTest(setting=setting, value=value):
                    with self.assertRaises(ValueError):
                        server = make_server("127.0.0.1", 0, **{setting: value})
                        self.addCleanup(server.server_close)


if __name__ == "__main__":
    unittest.main()

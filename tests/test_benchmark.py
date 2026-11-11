"""Offline benchmark orchestration tests; no server or accelerator is needed."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import random
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from inference_lab import benchmark
from inference_lab.metrics import RequestResult


def successful_result(request_id):
    return RequestResult(
        request_id=request_id, success=True, status_code=200,
        prompt_tokens=12, output_tokens=4, ttft_s=0.01, e2e_s=0.04,
        tpot_s=0.01, chunk_gaps_s=[0.01, 0.01, 0.01], text="test reply",
    )


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.workload = self.root / "prompts.jsonl"
        self.rows = [{"prompt": "Explain batching."}, {"prompt": "Explain latency."}]
        self.raw_workload = "".join(json.dumps(row) + "\n" for row in self.rows).encode()
        self.workload.write_bytes(self.raw_workload)
        self.config = {
            **benchmark.DEFAULTS, "workload": str(self.workload),
            "num_requests": 8, "concurrency": 2, "warmup_requests": 2,
        }

    def run_offline(self, sender=None, config=None, environment=None):
        if sender is None:
            sender = lambda base, model, prompt, tokens, timeout, request_id, ignore_eos: successful_result(request_id)
        with patch.object(benchmark, "request_completion", side_effect=sender), \
                patch.object(benchmark, "check_backend") as health, \
                patch.object(benchmark, "git_revision", return_value="test-revision"):
            result = benchmark.run_benchmark(config or self.config, environment)
        health.assert_called_once_with(
            (config or self.config)["base_url"], (config or self.config)["backend"],
            (config or self.config)["timeout_s"],
        )
        return result

    def test_warmups_are_sequential_and_excluded_from_measurements(self):
        calls = []
        lock = threading.Lock()

        def sender(base, model, prompt, tokens, timeout, request_id, ignore_eos):
            with lock:
                calls.append(request_id)
            return successful_result(request_id)

        result = self.run_offline(sender)
        self.assertEqual(calls[:2], [-1, -2])
        self.assertEqual(sorted(calls[2:]), list(range(8)))
        self.assertEqual(sorted(row["request_id"] for row in result["requests"]), list(range(8)))
        self.assertEqual(result["summary"]["attempted"], 8)
        self.assertEqual(result["summary"]["successful"], 8)

    def test_seed_controls_prompts_and_hash_records_exact_workload_bytes(self):
        def capture(config):
            prompts = {}

            def sender(base, model, prompt, tokens, timeout, request_id, ignore_eos):
                if request_id >= 0:
                    prompts[request_id] = prompt
                return successful_result(request_id)

            return prompts, self.run_offline(sender, config)

        first, result = capture(self.config)
        repeated, _ = capture(self.config)
        changed, _ = capture({**self.config, "seed": self.config["seed"] + 1})
        rng = random.Random(self.config["seed"])
        expected = {
            request_id: rng.choice(self.rows)["prompt"] + f"\nRequest tag: {rng.getrandbits(64):016x}"
            for request_id in range(self.config["num_requests"])
        }
        self.assertEqual(first, expected)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, changed)
        self.assertEqual(result["config"]["workload_sha256"], hashlib.sha256(self.raw_workload).hexdigest())

    def test_failed_measurements_remain_in_artifact_and_error_rate(self):
        def sender(base, model, prompt, tokens, timeout, request_id, ignore_eos):
            if request_id == 3:
                return RequestResult(request_id=3, success=False, status_code=503,
                                     error="HTTP 503", e2e_s=0.02)
            return successful_result(request_id)

        result = self.run_offline(sender)
        failed = [row for row in result["requests"] if not row["success"]]
        self.assertEqual(len(result["requests"]), 8)
        self.assertEqual(failed[0]["request_id"], 3)
        self.assertEqual(failed[0]["error"], "HTTP 503")
        self.assertEqual(result["summary"]["successful"], 7)
        self.assertEqual(result["summary"]["failed"], 1)
        self.assertEqual(result["summary"]["error_rate"], 1 / 8)
        self.assertEqual(result["summary"]["e2e_ms"]["mean"], 40.0)

    def test_concurrency_is_bounded_and_workers_overlap(self):
        active = 0
        peak = 0
        lock = threading.Lock()
        both_started = threading.Event()

        def sender(base, model, prompt, tokens, timeout, request_id, ignore_eos):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                if active == 2:
                    both_started.set()
            try:
                if not both_started.wait(timeout=2):
                    raise AssertionError("two configured workers did not overlap")
                time.sleep(0.005)
                return successful_result(request_id)
            finally:
                with lock:
                    active -= 1

        result = self.run_offline(sender, {**self.config, "warmup_requests": 0})
        self.assertEqual(peak, 2)
        self.assertEqual(active, 0)
        self.assertEqual(result["summary"]["attempted"], 8)

    def test_metadata_and_explicit_environment_are_preserved(self):
        result = self.run_offline(environment={"gpu": "offline test", "server_version": "fixture"})
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["backend"], "mock")
        self.assertEqual(result["model"], self.config["model"])
        self.assertEqual(result["environment"]["gpu"], "offline test")
        self.assertEqual(result["environment"]["server_version"], "fixture")
        self.assertEqual(result["environment"]["git_revision"], "test-revision")
        self.assertTrue(result["run_id"])
        self.assertTrue(result["created_at"])
        self.assertNotIn("workload_sha256", self.config)
        json.dumps(result, allow_nan=False)

    def test_failed_warmup_aborts_before_measurement(self):
        with patch.object(benchmark, "check_backend"), \
                patch.object(benchmark, "request_completion", return_value=RequestResult(
                    request_id=-1, success=False, error="warmup unavailable")) as sender:
            with self.assertRaisesRegex(ValueError, "warmup"):
                benchmark.run_benchmark(self.config)
        self.assertEqual(sender.call_count, 1)
        self.assertEqual(sender.call_args.args[5], -1)

    def test_invalid_configuration_is_rejected_before_requests(self):
        cases = [
            {"num_requests": 0}, {"num_requests": True}, {"concurrency": 9},
            {"concurrency": 0}, {"max_tokens": 0}, {"warmup_requests": -1},
            {"seed": -1}, {"timeout_s": float("inf")}, {"ttft_slo_ms": 0},
            {"e2e_slo_ms": float("nan")}, {"backend": "other"},
            {"ignore_eos": "false"}, {"model": " "}, {"label": ""},
            {"unknown_setting": 1},
        ]
        with patch.object(benchmark, "request_completion") as sender:
            for changes in cases:
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    benchmark.run_benchmark({**self.config, **changes})
        sender.assert_not_called()

    def test_workload_rejects_empty_malformed_and_nonobject_rows(self):
        for content in ("", "\n  \n", "not json\n", "[]\n", "null\n",
                        "42\n", '{}\n', '{"prompt": " "}\n', '{"prompt": 4}\n'):
            with self.subTest(content=content):
                self.workload.write_text(content)
                with self.assertRaises(ValueError):
                    benchmark.load_workload(self.workload)

    def test_health_marker_must_match_backend_label(self):
        for backend, marker, accepted in (
            ("mock", "mock", True), ("vllm", None, True),
            ("mock", None, False), ("vllm", "mock", False),
        ):
            with self.subTest(backend=backend, marker=marker):
                connection = MagicMock()
                response = connection.getresponse.return_value
                response.status = 200
                response.getheader.return_value = marker
                with patch.object(benchmark.http.client, "HTTPConnection", return_value=connection):
                    if accepted:
                        benchmark.check_backend("http://localhost:8000/v1", backend, 3)
                    else:
                        with self.assertRaises(ValueError):
                            benchmark.check_backend("http://localhost:8000/v1", backend, 3)
                connection.request.assert_called_once_with("GET", "/health")
                connection.close.assert_called_once()

    def test_health_rejects_http_failure_and_invalid_urls_without_opening_socket(self):
        connection = MagicMock()
        connection.getresponse.return_value.status = 503
        with patch.object(benchmark.http.client, "HTTPConnection", return_value=connection):
            with self.assertRaisesRegex(ValueError, "503"):
                benchmark.check_backend("http://localhost:8000", "mock", 3)
        connection.close.assert_called_once()
        for url in ("file:///tmp/model", "http://user:secret@localhost:8000",
                    "http://localhost:8000/other", "http://localhost:8000?token=x"):
            with self.subTest(url=url), \
                    patch.object(benchmark.http.client, "HTTPConnection") as http, \
                    patch.object(benchmark.http.client, "HTTPSConnection") as https:
                with self.assertRaises(ValueError):
                    benchmark.check_backend(url, "mock", 3)
                http.assert_not_called()
                https.assert_not_called()

    def test_cli_writes_json_applies_overrides_and_reports_failure_status(self):
        config_path = self.root / "config.json"
        environment_path = self.root / "environment.json"
        config_path.write_text(json.dumps(self.config))
        environment_path.write_text(json.dumps({"gpu": "fixture"}))
        for failed in (0, 1):
            with self.subTest(failed=failed):
                output = self.root / f"nested-{failed}" / "result.json"
                payload = {"backend": "mock", "summary": {"failed": failed}, "requests": []}
                with patch.object(benchmark, "run_benchmark", return_value=payload) as run, \
                        contextlib.redirect_stdout(io.StringIO()):
                    code = benchmark.main([
                        "--config", str(config_path), "--environment", str(environment_path),
                        "--num-requests", "10", "--label", "cli-test", "--output", str(output),
                    ])
                self.assertEqual(code, failed)
                self.assertEqual(json.loads(output.read_text()), payload)
                passed_config, passed_environment = run.call_args.args
                self.assertEqual(passed_config["num_requests"], 10)
                self.assertEqual(passed_config["label"], "cli-test")
                self.assertEqual(passed_config["workload"], str(self.workload))
                self.assertEqual(passed_environment, {"gpu": "fixture"})


if __name__ == "__main__":
    unittest.main()

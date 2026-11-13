from __future__ import annotations

import contextlib
import copy
import csv
import html
import io
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from inference_lab import report
from inference_lab.benchmark import DEFAULTS
from inference_lab.metrics import RequestResult, summarize


def mock_run(label="synthetic baseline", duration=1.0):
    results = [
        RequestResult(request_id=0, success=True, status_code=200,
                      prompt_tokens=12, output_tokens=4, ttft_s=0.01,
                      e2e_s=0.04, tpot_s=0.01),
        RequestResult(request_id=1, success=True, status_code=200,
                      prompt_tokens=12, output_tokens=4, ttft_s=0.02,
                      e2e_s=0.05, tpot_s=0.01),
        RequestResult(request_id=2, success=False, status_code=503,
                      error="synthetic failure", e2e_s=0.03),
    ]
    config = {**DEFAULTS, "num_requests": 3, "concurrency": 1,
              "label": label, "workload_sha256": "a" * 64}
    return {
        "schema_version": 1, "run_id": label, "created_at": "2026-11-13T00:00:00Z",
        "synthetic": True, "label": label, "backend": "mock", "model": config["model"],
        "config": config, "environment": {"fixture": "offline synthetic data"},
        "summary": summarize(results, duration, config["ttft_slo_ms"], config["e2e_slo_ms"]),
        "requests": [asdict(result) for result in results],
    }


class ReportTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def save_run(self, name, run):
        path = self.root / name
        path.write_text(json.dumps(run, allow_nan=False), encoding="utf-8")
        return path

    def test_numbers_and_signed_deltas_preserve_unavailable_values(self):
        for value in (None, True, False, "2", float("nan"), float("inf")):
            with self.subTest(value=value):
                self.assertIsNone(report.number(value))
                self.assertIsNone(report.percent_change(value, 2))
                self.assertIsNone(report.percent_change(2, value))
        self.assertEqual(report.number(0), 0.0)
        self.assertEqual(report.percent_change(6, 4), 50.0)
        self.assertEqual(report.percent_change(2, 4), -50.0)
        self.assertIsNone(report.percent_change(4, 0))

    def test_nested_metrics_and_all_deltas_use_first_run(self):
        baseline = mock_run()
        self.assertEqual(report.metric(baseline, "ttft_ms.p95"), 19.5)
        self.assertEqual(report.metric(baseline, "slo.goodput_requests_per_second"), 2.0)
        self.assertIsNone(report.metric(baseline, "missing.p95"))
        page = report.build_report([baseline, mock_run("synthetic second", 0.5),
                                    mock_run("synthetic third", 0.25)])
        self.assertIn("+100.0%", page)
        self.assertIn("+300.0%", page)
        baseline["summary"]["ttft_ms"]["p95"] = float("nan")
        self.assertIsNone(report.metric(baseline, "ttft_ms.p95"))
        self.assertIn("N/A", report.build_report([baseline]))

    def test_workload_sampling_output_and_slo_must_match(self):
        baseline = mock_run()
        changes = {
            "workload_sha256": "b" * 64, "seed": 43, "num_requests": 4,
            "max_tokens": 64, "ignore_eos": True,
            "ttft_slo_ms": 1000.0, "e2e_slo_ms": 10000.0,
        }
        for key, value in changes.items():
            candidate = copy.deepcopy(baseline)
            candidate["config"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                report.comparison_warnings([baseline, candidate])

    def test_concurrency_and_environment_changes_are_disclosed(self):
        baseline = mock_run()
        candidate = copy.deepcopy(baseline)
        candidate["config"]["concurrency"] = 2
        candidate["environment"]["fixture"] = "another offline fixture"
        warnings = " ".join(report.comparison_warnings([baseline, candidate]))
        self.assertIn("concurrency", warnings)
        self.assertIn("environment", warnings.lower())
        self.assertIn("SYNTHETIC", warnings)

    def test_backends_are_separate_and_model_changes_need_shared_group(self):
        baseline = mock_run()
        candidate = copy.deepcopy(baseline)
        candidate["backend"] = "vllm"
        with self.assertRaisesRegex(ValueError, "mock|backend"):
            report.build_report([baseline, candidate])
        candidate = copy.deepcopy(baseline)
        candidate["model"] = "different-model-id"
        for group in (None, "", "another-group"):
            candidate["config"]["comparison_group"] = group
            with self.subTest(group=group), self.assertRaisesRegex(ValueError, "comparison_group"):
                report.comparison_warnings([baseline, candidate])
        candidate["config"]["comparison_group"] = baseline["config"]["comparison_group"]
        warnings = " ".join(report.comparison_warnings([baseline, candidate]))
        self.assertIn("model", warnings.lower())

    def test_html_and_svg_escape_values_without_external_assets(self):
        run = mock_run('<svg onload="x">')
        run["model"] = '<model>&"'
        run["requests"][-1]["error"] = '<script>alert("x")</script>'
        page = report.build_report([run])
        for value in (run["label"], run["model"], run["requests"][-1]["error"]):
            self.assertIn(html.escape(value), page)
            self.assertNotIn(value, page)
        self.assertEqual(page.count("<svg "), 6)
        self.assertNotRegex(page, r"(?i)<(?:script|link|img)\b|(?:src|href)\s*=|@import\b|url\(")
        self.assertIn("SYNTHETIC", page)
        self.assertIn("503", page)

    def test_csv_records_backend_counts_and_blank_unavailable_metrics(self):
        run = mock_run('synthetic, "quoted"')
        run["summary"]["output_tokens_per_second"] = None
        run["summary"]["ttft_ms"]["p95"] = float("nan")
        destination = self.root / "nested" / "summary.csv"
        report.write_csv([run], destination)
        with destination.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label"], run["label"])
        self.assertEqual(rows[0]["backend"], "mock")
        self.assertEqual([rows[0][key] for key in ("attempted", "successful", "failed")], ["3", "2", "1"])
        self.assertEqual(float(rows[0]["requests_per_second"]), 2.0)
        self.assertEqual(rows[0]["output_tokens_per_second"], "")
        self.assertEqual(rows[0]["ttft_ms.p95"], "")

    def test_load_run_rejects_bad_shapes_and_accepts_benchmark_schema(self):
        valid = mock_run()
        path = self.save_run("run.json", valid)
        self.assertEqual(report.load_run(path), valid)
        invalid = [[], {}, {**valid, "schema_version": True}, {**valid, "model": None},
                   {**valid, "backend": "unknown"}, {**valid, "config": []},
                   {**valid, "summary": []}, {**valid, "requests": {}},
                   {**valid, "synthetic": False},
                   {**valid, "backend": "vllm"},
                   {**valid, "config": {**valid["config"], "backend": "vllm"}},
                   {**valid, "requests": [None]}, {**valid, "requests": [{"success": "false"}]}]
        for value in invalid:
            with self.subTest(value=value):
                path.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaises(ValueError):
                    report.load_run(path)
        path.write_text("{invalid", encoding="utf-8")
        with self.assertRaises(ValueError):
            report.load_run(path)
        with self.assertRaises(OSError):
            report.load_run(self.root / "missing.json")

    def test_cli_reads_json_and_creates_html_and_csv(self):
        first = self.save_run("first.json", mock_run())
        second = self.save_run("second.json", mock_run("synthetic second", 0.5))
        output = self.root / "report" / "index.html"
        summary = self.root / "report" / "summary.csv"
        with contextlib.redirect_stdout(io.StringIO()):
            code = report.main([str(first), str(second), "--output", str(output), "--csv", str(summary)])
        self.assertEqual(code, 0)
        self.assertIn("SYNTHETIC", output.read_text(encoding="utf-8"))
        self.assertIn("+100.0%", output.read_text(encoding="utf-8"))
        with summary.open(encoding="utf-8", newline="") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 2)

    def test_cli_incompatible_runs_exit_two_without_outputs(self):
        candidate = mock_run("synthetic incompatible")
        candidate["config"]["seed"] += 1
        first = self.save_run("first.json", mock_run())
        second = self.save_run("second.json", candidate)
        output, summary = self.root / "bad.html", self.root / "bad.csv"
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            report.main([str(first), str(second), "--output", str(output), "--csv", str(summary)])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("seed", stderr.getvalue())
        self.assertFalse(output.exists())
        self.assertFalse(summary.exists())


if __name__ == "__main__":
    unittest.main()

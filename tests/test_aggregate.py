from __future__ import annotations

import contextlib
import copy
import csv
import hashlib
import io
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from statistics import median
from unittest.mock import patch

from inference_lab import aggregate, report, sweep
from inference_lab.benchmark import DEFAULTS
from inference_lab.metrics import RequestResult, percentile, summarize


class AggregateTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.workload = self.root / "prompts.jsonl"
        self.workload.write_text('{"prompt": "Explain batching."}\n', encoding="utf-8")
        self.digest = hashlib.sha256(self.workload.read_bytes()).hexdigest()
        self.config = {**DEFAULTS, "workload": str(self.workload), "num_requests": 4,
                       "concurrency": 1, "warmup_requests": 1}
        self.environment = {"note": "offline fixture"}
        self.client_metadata = {"python": "3.12.fixture", "client_os": "FixtureOS",
                                "client_machine": "fixture", "git_revision": "fixture"}
        self.sequence = 0

    def make_sweep(self, name="runs", levels=(2, 1), repetitions=2, failed=0,
                   missing_repetition=None, config=None, environment=None, profile="mock"):
        def benchmark(settings, metadata):
            self.sequence += 1
            repetition = metadata["sweep_repetition"]
            latencies = [0.01] * 4 if repetition == 1 else [0.1, 0.1, 0.1, 1.0]
            missing = repetition == missing_repetition
            requests = [
                RequestResult(request_id=index, success=index >= failed,
                              status_code=200 if index >= failed else 503,
                              error=None if index >= failed else "fixture failure",
                              prompt_tokens=8, output_tokens=None if missing else 4,
                              ttft_s=latency / 4, e2e_s=latency,
                              tpot_s=None if missing else latency / 8)
                for index, latency in enumerate(latencies)
            ]
            return {
                "schema_version": 1, "run_id": f"fixture-{self.sequence}",
                "created_at": "2026-11-26T00:00:00Z", "label": settings["label"],
                "backend": settings["backend"], "synthetic": settings["backend"] == "mock",
                "model": settings["model"],
                "config": {**settings, "workload_sha256": self.digest},
                "environment": {**self.client_metadata, **metadata},
                "summary": summarize(requests, float(repetition), settings["ttft_slo_ms"],
                                     settings["e2e_slo_ms"]),
                "requests": [asdict(request) for request in requests],
            }

        output = self.root / name
        with patch.object(sweep, "run_benchmark", side_effect=benchmark), \
                patch.object(sweep.time, "sleep"):
            sweep.run_sweep(config or self.config, output, profile, levels, repetitions, 0,
                            environment=self.environment if environment is None else environment)
        return output / "sweep.json"

    def read(self, path):
        return json.loads(path.read_text(encoding="utf-8"))

    def write(self, path, value):
        path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")

    def first_run(self, index):
        return index.parent / self.read(index)["plan"][0]["filename"]

    def call_cli(self, index, output):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                return aggregate.main([str(index), "--output-dir", str(output)])
            except SystemExit as exc:
                return exc.code

    def snapshot(self, folder):
        return {path.relative_to(folder).as_posix(): path.read_bytes()
                for path in folder.rglob("*") if path.is_file()}

    def test_completed_sweep_groups_repetitions_without_pooling_rates(self):
        index = self.make_sweep()
        manifest = self.read(index)
        before = self.snapshot(index.parent)
        summary = aggregate.build_summary(index)
        self.assertEqual(summary["schema_version"], 1)
        self.assertEqual(summary["kind"], "sweep_summary")
        for key in ("sweep_id", "profile", "backend", "synthetic", "workload_sha256",
                    "base_config", "repetitions"):
            self.assertEqual(summary[key], manifest[key], key)
        self.assertEqual(summary["failed_requests"], 0)
        self.assertEqual([group["concurrency"] for group in summary["groups"]], [1, 2])
        for group in summary["groups"]:
            level = group["concurrency"]
            self.assertEqual(group["repetitions"], 2)
            self.assertEqual(group["run_files"], [f"mock-c{level}-r1.json", f"mock-c{level}-r2.json"])
            self.assertEqual((group["attempted"], group["successful"], group["failed"]), (8, 8, 0))
            self.assertEqual(set(group["metrics"]), {key for key, _, _ in report.METRICS})
            rate = group["metrics"]["requests_per_second"]
            self.assertEqual(rate, {"values": [4.0, 2.0], "observed_runs": 2,
                                    "median": 3.0, "min": 2.0, "max": 4.0})
        self.assertEqual(self.snapshot(index.parent), before)

    def test_latency_statistics_describe_per_run_p95_not_pooled_requests(self):
        index = self.make_sweep(levels=(1,))
        paths = [index.parent / name for name in ("mock-c1-r1.json", "mock-c1-r2.json")]
        runs = [self.read(path) for path in paths]
        expected = [run["summary"]["e2e_ms"]["p95"] for run in runs]
        values = aggregate.build_summary(index)["groups"][0]["metrics"]["e2e_ms.p95"]
        self.assertEqual(values["values"], expected)
        self.assertAlmostEqual(values["median"], median(expected))
        pooled = percentile([request["e2e_s"] * 1000 for run in runs
                             for request in run["requests"]], 95)
        self.assertNotAlmostEqual(values["median"], pooled)

    def test_missing_metrics_keep_repetition_and_make_aggregate_unavailable(self):
        index = self.make_sweep(levels=(1,), missing_repetition=2)
        group = aggregate.build_summary(index)["groups"][0]
        self.assertEqual(group["repetitions"], 2)
        self.assertEqual(group["attempted"], 8)
        for key in ("output_tokens_per_second", "tpot_ms.p95"):
            with self.subTest(metric=key):
                stats = group["metrics"][key]
                self.assertIsNotNone(stats["values"][0])
                self.assertEqual(stats["values"][1], None)
                self.assertEqual(stats["observed_runs"], 1)
                self.assertEqual([stats[key] for key in ("median", "min", "max")], [None] * 3)
        self.assertEqual(group["metrics"]["requests_per_second"]["observed_runs"], 2)

    def test_failed_requests_are_retained_and_cli_returns_one(self):
        index = self.make_sweep(failed=1)
        summary = aggregate.build_summary(index)
        self.assertEqual(summary["failed_requests"], 4)
        for group in summary["groups"]:
            self.assertEqual((group["attempted"], group["successful"], group["failed"]), (8, 6, 2))
            self.assertEqual(group["metrics"]["requests_per_second"]["values"], [3.0, 1.5])
        output = self.root / "failed-summary"
        self.assertEqual(self.call_cli(index, output), 1)
        self.assertEqual(self.read(output / "summary.json"), summary)
        self.assertTrue((output / "summary.csv").is_file())

    def test_single_repetition_has_one_observation_and_equal_bounds(self):
        index = self.make_sweep(levels=(1,), repetitions=1)
        group = aggregate.build_summary(index)["groups"][0]
        self.assertEqual(group["repetitions"], 1)
        for stats in group["metrics"].values():
            self.assertEqual(stats["observed_runs"], 1)
            self.assertEqual([stats[key] for key in ("median", "min", "max")],
                             [stats["values"][0]] * 3)

    def test_portable_artifacts_keep_observed_metadata_without_original_workload(self):
        index = self.make_sweep()
        self.workload.unlink()
        summary = aggregate.build_summary(index)
        self.assertEqual(summary["workload_sha256"], self.digest)
        self.assertEqual(summary["environment"], {**self.client_metadata, **self.environment,
                                                  "deployment_profile": "mock"})
        self.assertFalse(set(summary["environment"]) &
                         {"sweep_id", "sweep_repetition", "sweep_order", "sweep_concurrency"})

    def test_incomplete_sweeps_and_pending_cases_are_rejected(self):
        index = self.make_sweep()
        original = self.read(index)
        for status in ("running", "interrupted", "aborted"):
            with self.subTest(status=status):
                self.write(index, {**original, "status": status})
                with self.assertRaises(ValueError):
                    aggregate.build_summary(index)
        for status in ("pending", "running", "aborted", "interrupted"):
            modified = copy.deepcopy(original)
            modified["plan"][0]["status"] = status
            self.write(index, modified)
            with self.subTest(case_status=status), self.assertRaises(ValueError):
                aggregate.build_summary(index)

    def test_incomplete_or_duplicate_grid_and_orders_are_rejected(self):
        index = self.make_sweep()
        original = self.read(index)
        mutations = [
            lambda value: value["plan"].pop(),
            lambda value: value["plan"].append(copy.deepcopy(value["plan"][0])),
            lambda value: value["plan"][1].update(concurrency=value["plan"][0]["concurrency"]),
            lambda value: value["plan"][1].update(order=value["plan"][0]["order"]),
            lambda value: value.update(repetitions=3),
            lambda value: value.update(concurrency=[1, 1]),
        ]
        for position, mutate in enumerate(mutations):
            value = copy.deepcopy(original)
            mutate(value)
            self.write(index, value)
            with self.subTest(position=position), self.assertRaises(ValueError):
                aggregate.build_summary(index)

    def test_missing_and_escaping_run_paths_are_rejected(self):
        index = self.make_sweep()
        original = self.read(index)
        for filename in ("missing.json", "../outside.json", "/tmp/outside.json", "nested/run.json"):
            manifest = copy.deepcopy(original)
            manifest["plan"][0]["filename"] = filename
            self.write(index, manifest)
            with self.subTest(filename=filename), self.assertRaises((ValueError, OSError)):
                aggregate.build_summary(index)
        self.write(index, original)
        self.first_run(index).unlink()
        with self.assertRaises((ValueError, OSError)):
            aggregate.build_summary(index)

    def test_configuration_and_workload_drift_are_rejected(self):
        index = self.make_sweep()
        path = self.first_run(index)
        original = self.read(path)
        mutations = [
            lambda run: run.update(model="other-model"),
            lambda run: run.update(backend="vllm", synthetic=False),
            lambda run: run["config"].update(workload_sha256="f" * 64),
            lambda run: run["config"].update(max_tokens=17),
            lambda run: run["config"].update(warmup_requests=2),
            lambda run: run["config"].update(base_url="http://different-host:8000/v1"),
        ]
        for position, mutate in enumerate(mutations):
            run = copy.deepcopy(original)
            mutate(run)
            self.write(path, run)
            with self.subTest(position=position), self.assertRaises(ValueError):
                aggregate.build_summary(index)

    def test_runtime_and_client_environment_drift_are_rejected(self):
        environment = {"deployment_profile": "baseline", "model_id": "fixture/model",
                       "model_revision": "fixture-revision", "container_image_id": "fixture-image",
                       "gpu_model": "fixture GPU", "driver_version": "fixture-driver"}
        config = {**self.config, "backend": "vllm"}
        index = self.make_sweep(config=config, profile="baseline", environment=environment)
        self.assertFalse(aggregate.build_summary(index)["synthetic"])
        path = self.first_run(index)
        original = self.read(path)
        for key in ("model_revision", "gpu_model", "python", "client_machine"):
            run = copy.deepcopy(original)
            run["environment"][key] = "changed"
            self.write(path, run)
            with self.subTest(key=key), self.assertRaises(ValueError):
                aggregate.build_summary(index)

    def test_case_identity_must_match_each_raw_run(self):
        index = self.make_sweep()
        path = self.first_run(index)
        original = self.read(path)
        mutations = [
            lambda run: run.update(label="mock-c1-r99"),
            lambda run: run["config"].update(concurrency=4),
            lambda run: run["environment"].update(sweep_id="other-sweep"),
            lambda run: run["environment"].update(sweep_repetition=99),
            lambda run: run["environment"].update(sweep_order=99),
            lambda run: run["environment"].update(sweep_concurrency=4),
        ]
        for position, mutate in enumerate(mutations):
            run = copy.deepcopy(original)
            mutate(run)
            self.write(path, run)
            with self.subTest(position=position), self.assertRaises(ValueError):
                aggregate.build_summary(index)

    def test_failure_flags_and_all_recorded_request_counts_must_reconcile(self):
        index = self.make_sweep(failed=1)
        path = self.first_run(index)
        raw, manifest = self.read(path), self.read(index)
        mutations = [
            lambda run, value: run["summary"].update(attempted=3),
            lambda run, value: run["summary"].update(successful=4),
            lambda run, value: run["summary"].update(failed=0),
            lambda run, value: run["requests"][0].update(success=True),
            lambda run, value: run["requests"].pop(),
            lambda run, value: value["plan"][0].update(failed_requests=0),
            lambda run, value: value.update(failed_requests=0),
        ]
        for position, mutate in enumerate(mutations):
            run, value = copy.deepcopy(raw), copy.deepcopy(manifest)
            mutate(run, value)
            self.write(path, run)
            self.write(index, value)
            with self.subTest(position=position), self.assertRaises(ValueError):
                aggregate.build_summary(index)

    def test_malformed_json_and_unsupported_schemas_fail_without_output(self):
        index = self.make_sweep()
        original = index.read_text()
        for position, text in enumerate(("{", "[]", '{"schema_version": 2}',
                                         original.replace('"schema_version": 1', '"schema_version": true', 1))):
            index.write_text(text)
            output = self.root / f"invalid-summary-{position}"
            with self.subTest(position=position):
                self.assertEqual(self.call_cli(index, output), 2)
                self.assertFalse(output.exists())
        index.write_text(original)
        run = self.first_run(index)
        for position, text in enumerate(("{", "[]", '{"schema_version": 2}')):
            run.write_text(text)
            output = self.root / f"invalid-run-{position}"
            with self.subTest(raw=position):
                self.assertEqual(self.call_cli(index, output), 2)
                self.assertFalse(output.exists())

    def test_cli_writes_json_and_long_csv_with_blank_unavailable_values(self):
        index = self.make_sweep(missing_repetition=2)
        before = self.snapshot(index.parent)
        output = self.root / "summary"
        self.assertEqual(self.call_cli(index, output), 0)
        self.assertEqual({path.name for path in output.iterdir()}, {"summary.json", "summary.csv"})
        self.assertEqual(self.read(output / "summary.json"), aggregate.build_summary(index))
        with (output / "summary.csv").open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            self.assertEqual(reader.fieldnames,
                             ["profile", "backend", "synthetic", "concurrency", "repetitions",
                              "attempted", "successful", "failed", "metric", "observed_runs",
                              "median", "min", "max"])
        self.assertEqual(len(rows), 2 * len(report.METRICS))
        self.assertEqual({row["profile"] for row in rows}, {"mock"})
        self.assertEqual({row["backend"] for row in rows}, {"mock"})
        missing = [row for row in rows if row["metric"] == "output_tokens_per_second"]
        for row in missing:
            self.assertEqual(row["observed_runs"], "1")
            self.assertEqual([row[key] for key in ("median", "min", "max")], [""] * 3)
        self.assertEqual(self.snapshot(index.parent), before)

    def test_cli_refuses_existing_directories_and_symlinks_without_mutation(self):
        index = self.make_sweep()
        empty = self.root / "empty"
        empty.mkdir()
        populated = self.root / "populated"
        populated.mkdir()
        (populated / "summary.json").write_text("keep this")
        target = self.root / "target"
        target.mkdir()
        link = self.root / "link"
        link.symlink_to(target, target_is_directory=True)
        dangling = self.root / "dangling"
        dangling.symlink_to(self.root / "not-created", target_is_directory=True)
        for output in (empty, populated, link, dangling):
            with self.subTest(output=output.name):
                self.assertEqual(self.call_cli(index, output), 2)
        self.assertEqual(list(empty.iterdir()), [])
        self.assertEqual((populated / "summary.json").read_text(), "keep this")
        self.assertEqual(list(target.iterdir()), [])
        self.assertTrue(link.is_symlink())
        self.assertTrue(dangling.is_symlink())
        self.assertFalse((self.root / "not-created").exists())


if __name__ == "__main__":
    unittest.main()

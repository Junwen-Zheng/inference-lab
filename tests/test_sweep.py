from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from inference_lab import report, sweep
from inference_lab.benchmark import DEFAULTS
from inference_lab.metrics import RequestResult, summarize


class SweepTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.workload = self.root / "prompts.jsonl"
        self.workload.write_text('{"prompt": "Explain batching."}\n', encoding="utf-8")
        self.digest = hashlib.sha256(self.workload.read_bytes()).hexdigest()
        self.config = {**DEFAULTS, "workload": str(self.workload),
                       "num_requests": 4, "concurrency": 1, "warmup_requests": 1}
        self.calls = []

    def benchmark(self, config, environment=None, failed=0):
        self.calls.append((copy.deepcopy(config), copy.deepcopy(environment)))
        results = [
            RequestResult(request_id=index, success=index >= failed,
                          status_code=200 if index >= failed else 503,
                          error=None if index >= failed else "synthetic failure",
                          prompt_tokens=8, output_tokens=4, ttft_s=0.01,
                          e2e_s=0.04, tpot_s=0.01)
            for index in range(config["num_requests"])
        ]
        return {
            "schema_version": 1, "run_id": f"fixture-{len(self.calls)}",
            "created_at": "2026-11-20T00:00:00Z", "label": config["label"],
            "backend": config["backend"], "synthetic": config["backend"] == "mock",
            "model": config["model"],
            "config": {**config, "workload_sha256": self.digest},
            "environment": copy.deepcopy(environment or {}),
            "summary": summarize(results, 1.0, config["ttft_slo_ms"], config["e2e_slo_ms"]),
            "requests": [asdict(result) for result in results],
        }

    def read_index(self, output):
        return json.loads((output / "sweep.json").read_text())

    def run_offline(self, output, sender=None, **options):
        settings = {"concurrency": (1, 2), "repetitions": 2, "cooldown_s": 0.25,
                    **options}
        with patch.object(sweep, "run_benchmark", side_effect=sender or self.benchmark), \
                patch.object(sweep.time, "sleep") as sleep:
            manifest = sweep.run_sweep(self.config, output, "mock", **settings)
        return manifest, sleep

    def test_repeated_runs_rotate_order_and_preserve_comparison_controls(self):
        output = self.root / "runs"
        original = copy.deepcopy(self.config)
        environment = {"note": "offline synthetic fixture"}
        manifest, sleep = self.run_offline(output, environment=environment)
        self.assertEqual(manifest["kind"], "concurrency_sweep")
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["failed_requests"], 0)
        self.assertEqual(manifest["workload_sha256"], self.digest)
        self.assertEqual(self.read_index(output), manifest)
        plan = manifest["plan"]
        self.assertEqual([(case["concurrency"], case["repetition"]) for case in plan],
                         [(1, 1), (2, 1), (2, 2), (1, 2)])
        self.assertEqual([case["order"] for case in plan], [1, 2, 3, 4])
        self.assertEqual({case["status"] for case in plan}, {"completed"})
        self.assertEqual({case["filename"] for case in plan},
                         {"mock-c1-r1.json", "mock-c2-r1.json",
                          "mock-c1-r2.json", "mock-c2-r2.json"})
        runs = [report.load_run(output / case["filename"]) for case in plan]
        self.assertEqual(len({run["run_id"] for run in runs}), 4)
        for case, run, (config, metadata) in zip(plan, runs, self.calls):
            self.assertEqual(run["label"], case["filename"][:-5])
            for key in ("seed", "num_requests", "max_tokens", "warmup_requests",
                        "ttft_slo_ms", "e2e_slo_ms", "ignore_eos", "comparison_group"):
                self.assertEqual(config[key], original[key], key)
            self.assertEqual(run["config"]["workload_sha256"], self.digest)
            self.assertEqual(metadata["note"], environment["note"])
            self.assertEqual(metadata["deployment_profile"], "mock")
            self.assertEqual(metadata["sweep_id"], manifest["sweep_id"])
            self.assertEqual(metadata["sweep_repetition"], case["repetition"])
            self.assertEqual(metadata["sweep_order"], case["order"])
            self.assertEqual(metadata["sweep_concurrency"], case["concurrency"])
        self.assertEqual(self.config, original)
        self.assertEqual(environment, {"note": "offline synthetic fixture"})
        self.assertEqual(sleep.call_count, 3)
        self.assertTrue(all(call.args == (0.25,) for call in sleep.call_args_list))

    def test_measured_failures_remain_in_raw_runs_and_do_not_stop_sweep(self):
        output = self.root / "failures"

        def sender(config, environment):
            return self.benchmark(config, environment, failed=1)

        manifest, _ = self.run_offline(output, sender=sender)
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["failed_requests"], 4)
        for case in manifest["plan"]:
            self.assertEqual(case["failed_requests"], 1)
            run = report.load_run(output / case["filename"])
            self.assertEqual(len(run["requests"]), 4)
            self.assertFalse(run["requests"][0]["success"])
            self.assertEqual(run["summary"]["failed"], 1)

    def test_invalid_plan_is_rejected_before_calls_or_directory_creation(self):
        cases = [
            {"profile": "unknown"}, {"profile": "baseline"},
            {"concurrency": (1, 1)}, {"concurrency": ()},
            {"concurrency": (0,)}, {"concurrency": (True,)},
            {"concurrency": (5,)}, {"repetitions": 0},
            {"repetitions": True}, {"repetitions": 1_000_000_000},
            {"cooldown_s": -1}, {"cooldown_s": True},
            {"cooldown_s": float("nan")}, {"cooldown_s": float("inf")},
            {"cooldown_s": 10 ** 1000},
            {"environment": []}, {"environment": {"bad": float("nan")}},
            {"config": {**self.config, "backend": "other"}},
        ]
        with patch.object(sweep, "run_benchmark") as run:
            for index, changes in enumerate(cases):
                output = self.root / f"invalid-{index}"
                options = {"config": self.config, "output_dir": output, "profile": "mock",
                           "concurrency": (1, 2), "repetitions": 2, "cooldown_s": 0,
                           **changes}
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    sweep.run_sweep(**options)
                self.assertFalse(output.exists())
        run.assert_not_called()

    def test_existing_output_is_never_overwritten_even_when_empty(self):
        with patch.object(sweep, "run_benchmark") as run:
            for populated in (False, True):
                output = self.root / f"existing-{populated}"
                output.mkdir()
                sentinel = output / "sweep.json"
                if populated:
                    sentinel.write_text("previous experiment")
                with self.subTest(populated=populated), self.assertRaises(FileExistsError):
                    sweep.run_sweep(self.config, output, "mock", concurrency=(1,))
                self.assertEqual(list(output.iterdir()), [sentinel] if populated else [])
                if populated:
                    self.assertEqual(sentinel.read_text(), "previous experiment")
        run.assert_not_called()

    def test_gpu_profiles_require_matching_explicit_runtime_metadata(self):
        config = {**self.config, "backend": "vllm"}
        environment = {
            "deployment_profile": "baseline", "model_id": "fixture/model",
            "model_revision": "fixture-revision", "container_image_id": "fixture-image",
            "gpu_model": "fixture GPU", "driver_version": "fixture-driver",
        }
        with patch.object(sweep, "run_benchmark", side_effect=self.benchmark) as run, \
                patch.object(sweep.time, "sleep"):
            for index, key in enumerate(environment):
                incomplete = {**environment, key: " "}
                output = self.root / f"missing-{index}"
                with self.subTest(key=key), self.assertRaises(ValueError):
                    sweep.run_sweep(config, output, "baseline", concurrency=(1,),
                                    environment=incomplete)
                self.assertFalse(output.exists())
            for index, (profile, metadata) in enumerate((
                ("mock", environment),
                ("awq", environment),
            )):
                with self.subTest(profile=profile), self.assertRaises(ValueError):
                    sweep.run_sweep(config, self.root / f"mismatch-{index}", profile,
                                    concurrency=(1,), environment=metadata)
            run.assert_not_called()
            output = self.root / "gpu-fixture"
            manifest = sweep.run_sweep(config, output, "baseline", concurrency=(1,),
                                       repetitions=1, cooldown_s=0, environment=environment)
        self.assertFalse(manifest["synthetic"])
        saved = report.load_run(output / "baseline-c1-r1.json")
        for key, value in environment.items():
            self.assertEqual(saved["environment"][key], value)

    def test_setup_failure_preserves_completed_run_and_pending_cases(self):
        output = self.root / "setup-failure"

        def sender(config, environment):
            if self.calls:
                raise ValueError("warmup unavailable")
            return self.benchmark(config, environment)

        with self.assertRaisesRegex(ValueError, "warmup unavailable"):
            self.run_offline(output, sender=sender)
        manifest = self.read_index(output)
        self.assertEqual(manifest["status"], "aborted")
        self.assertEqual([case["status"] for case in manifest["plan"]],
                         ["completed", "aborted", "pending", "pending"])
        self.assertIn("warmup unavailable", manifest["plan"][1]["error"])
        self.assertTrue((output / manifest["plan"][0]["filename"]).exists())
        self.assertFalse((output / manifest["plan"][1]["filename"]).exists())
        self.assertEqual(len(list(output.glob("mock-*.json"))), 1)

    def test_workload_drift_aborts_before_next_benchmark(self):
        output = self.root / "drift"

        def sender(config, environment):
            result = self.benchmark(config, environment)
            self.workload.write_text('{"prompt": "Changed workload."}\n')
            return result

        with self.assertRaisesRegex(ValueError, "workload|hash"):
            self.run_offline(output, sender=sender)
        manifest = self.read_index(output)
        self.assertEqual(manifest["status"], "aborted")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(list(output.glob("mock-*.json"))), 1)
        self.assertEqual(manifest["workload_sha256"], self.digest)
        self.assertTrue(any(case["status"] == "pending" for case in manifest["plan"]))

    def test_returned_hash_mismatch_keeps_actual_artifact_and_aborts(self):
        output = self.root / "returned-drift"

        def sender(config, environment):
            result = self.benchmark(config, environment)
            result["config"]["workload_sha256"] = "f" * 64
            return result

        with self.assertRaisesRegex(ValueError, "workload|hash"):
            self.run_offline(output, sender=sender)
        manifest = self.read_index(output)
        self.assertEqual(manifest["status"], "aborted")
        self.assertEqual(manifest["plan"][0]["status"], "aborted")
        raw = json.loads((output / manifest["plan"][0]["filename"]).read_text())
        self.assertEqual(raw["config"]["workload_sha256"], "f" * 64)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual([case["status"] for case in manifest["plan"]][1:],
                         ["pending", "pending", "pending"])

    def test_interrupt_records_current_case_without_fabricating_raw_output(self):
        output = self.root / "interrupted"

        def sender(config, environment):
            if self.calls:
                raise KeyboardInterrupt
            return self.benchmark(config, environment)

        with self.assertRaises(KeyboardInterrupt):
            self.run_offline(output, sender=sender)
        manifest = self.read_index(output)
        self.assertEqual(manifest["status"], "interrupted")
        self.assertEqual([case["status"] for case in manifest["plan"]],
                         ["completed", "interrupted", "pending", "pending"])
        self.assertEqual(len(list(output.glob("mock-*.json"))), 1)

        output = self.root / "interrupted-cooldown"
        with patch.object(sweep, "run_benchmark", side_effect=self.benchmark), \
                patch.object(sweep.time, "sleep", side_effect=KeyboardInterrupt), \
                self.assertRaises(KeyboardInterrupt):
            sweep.run_sweep(self.config, output, "mock", concurrency=(1, 2),
                            repetitions=1, cooldown_s=1)
        manifest = self.read_index(output)
        self.assertEqual(manifest["status"], "interrupted")
        self.assertEqual([case["status"] for case in manifest["plan"]],
                         ["completed", "pending"])
        self.assertEqual(len(list(output.glob("mock-*.json"))), 1)

    def test_cli_files_feed_report_without_manifest_being_treated_as_run(self):
        config_path = self.root / "config.json"
        environment_path = self.root / "environment.json"
        config_path.write_text(json.dumps(self.config))
        environment_path.write_text(json.dumps({"note": "CLI fixture"}))
        output = self.root / "cli-runs"
        with patch.object(sweep, "run_benchmark", side_effect=self.benchmark), \
                patch.object(sweep.time, "sleep"), \
                contextlib.redirect_stdout(io.StringIO()):
            code = sweep.main(["--config", str(config_path), "--environment", str(environment_path),
                               "--profile", "mock", "--output-dir", str(output),
                               "--concurrency", "1", "2", "--repetitions", "2", "--cooldown-s", "0"])
        self.assertEqual(code, 0)
        manifest = self.read_index(output)
        runs = [report.load_run(output / case["filename"]) for case in manifest["plan"]]
        self.assertEqual(len(runs), 4)
        self.assertEqual(runs[0]["environment"]["note"], "CLI fixture")
        page = report.build_report(runs)
        self.assertEqual(page.count("<svg"), 6)
        self.assertIn("SYNTHETIC", page)
        with self.assertRaises(ValueError):
            report.load_run(output / "sweep.json")

    def test_cli_exit_codes_distinguish_failed_requests_setup_and_interrupt(self):
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(self.config))
        actions = (
            (lambda config, environment: self.benchmark(config, environment, failed=1), 1),
            (ValueError("server unavailable"), 2),
            (KeyboardInterrupt(), 130),
        )
        for index, (action, expected) in enumerate(actions):
            output = self.root / f"exit-{expected}"
            with self.subTest(expected=expected), \
                    patch.object(sweep, "run_benchmark", side_effect=action), \
                    patch.object(sweep.time, "sleep"), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                try:
                    actual = sweep.main(["--config", str(config_path), "--profile", "mock",
                                         "--output-dir", str(output), "--concurrency", "1",
                                         "--repetitions", "1", "--cooldown-s", "0"])
                except SystemExit as exc:
                    actual = exc.code
            self.assertEqual(actual, expected)
            self.assertTrue((output / "sweep.json").exists())

    def test_single_run_has_no_cooldown_and_invalid_cli_json_makes_no_artifacts(self):
        manifest, sleep = self.run_offline(self.root / "single", concurrency=(1,), repetitions=1)
        self.assertEqual(manifest["status"], "completed")
        sleep.assert_not_called()
        config_path = self.root / "invalid.json"
        config_path.write_text("[]")
        output = self.root / "bad-json"
        with patch.object(sweep, "run_benchmark") as run, \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            sweep.main(["--config", str(config_path), "--profile", "mock", "--output-dir", str(output)])
        self.assertEqual(raised.exception.code, 2)
        run.assert_not_called()
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

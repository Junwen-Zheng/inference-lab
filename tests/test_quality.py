from __future__ import annotations

import contextlib
import copy
import hashlib
import inspect
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from inference_lab import quality
from inference_lab.client import request_completion
from inference_lab.metrics import RequestResult


class QualityTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.dataset = self.root / "quality.jsonl"
        self.cases = [
            {"id": "capital", "prompt": "  Return only France's capital.\n",
             "answers": ["Paris", "PARIS"]},
            {"id": "number", "prompt": "Return only 42.", "answers": ["42"]},
            {"id": "punctuation", "prompt": "Return only A-1.", "answers": ["A-1"]},
        ]
        self.write_dataset(self.cases)
        self.config = {**quality.DEFAULTS, "dataset": str(self.dataset),
                       "backend": "mock", "model": "fixture-model",
                       "comparison_group": "quality-fixture"}
        self.calls = []

    def write_dataset(self, cases):
        self.dataset.write_text("".join(json.dumps(row) + "\n" for row in cases),
                                encoding="utf-8")

    def response(self, text, success=True, error=None):
        return RequestResult(request_id=len(self.calls), text=text, success=success,
                             status_code=200 if success else 503, error=error,
                             prompt_tokens=12, output_tokens=3)

    def sender(self, *args, **kwargs):
        arguments = inspect.signature(request_completion).bind(*args, **kwargs)
        arguments.apply_defaults()
        self.calls.append(dict(arguments.arguments))
        return self.response(self.cases[len(self.calls) - 1]["answers"][0])

    def run_offline(self, output, sender=None, **options):
        with patch.object(quality, "check_backend") as backend, \
                patch.object(quality, "request_completion", side_effect=sender or self.sender):
            result = quality.run_quality(self.config, output, "mock", **options)
        backend.assert_called_once_with(self.config["base_url"], "mock",
                                        self.config["timeout_s"])
        return result

    def read_result(self, output):
        return json.loads((output / "quality.json").read_text())

    def cli(self, output, *extra):
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(self.config))
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            try:
                return quality.main(["--config", str(config_path), "--profile", "mock",
                                     "--output-dir", str(output), *extra])
            except SystemExit as exc:
                return exc.code

    def test_normalization_preserves_numbers_punctuation_and_extra_prose(self):
        self.assertEqual(quality.normalize_answer(" \nPaRiS\t France  "), "paris france")
        self.assertEqual(quality.normalize_answer("Straße"), "strasse")
        for left, right in (("42", "4.2"), ("A-1", "A1"), ("Paris", "Paris."),
                            ("Paris", "The answer is Paris"), ("42", "142")):
            with self.subTest(left=left, right=right):
                self.assertNotEqual(quality.normalize_answer(left),
                                    quality.normalize_answer(right))

    def test_dataset_retains_prompts_and_hashes_original_bytes(self):
        rows, digest = quality.load_dataset(self.dataset)
        self.assertEqual(rows, self.cases)
        self.assertEqual(rows[0]["prompt"], "  Return only France's capital.\n")
        self.assertEqual(digest, hashlib.sha256(self.dataset.read_bytes()).hexdigest())

    def test_invalid_datasets_are_rejected_including_duplicates_and_size(self):
        valid = self.cases[0]
        bad_rows = [
            [], [None], [[]], [{**valid, "id": " "}], [{**valid, "id": 1}],
            [{**valid, "prompt": "\t"}], [{**valid, "prompt": False}],
            [{**valid, "answers": []}], [{**valid, "answers": "Paris"}],
            [{**valid, "answers": ["Paris", " "]}], [{**valid, "answers": [4]}],
            [{"id": "x", "prompt": "x"}], [{**valid, "extra": True}],
            [valid, valid],
            [{**valid, "id": str(index)} for index in range(1001)],
        ]
        for rows in bad_rows:
            with self.subTest(rows=rows[:2]), self.assertRaises(ValueError):
                self.write_dataset(rows)
                quality.load_dataset(self.dataset)
        for text in ("{bad json}\n", "   \n"):
            self.dataset.write_text(text)
            with self.subTest(text=text), self.assertRaises(ValueError):
                quality.load_dataset(self.dataset)

    def test_invalid_config_fails_before_network_or_artifact_creation(self):
        changes = [
            {"base_url": "ftp://host"}, {"base_url": "http://host/other"},
            {"base_url": "http://user:secret@host"},
            {"base_url": "http://host?query=1"}, {"base_url": "http://host#part"},
            {"base_url": "http://host:bad"}, {"base_url": "http://["},
            {"base_url": "http://host:99999"}, {"backend": "other"},
            {"model": " "}, {"max_tokens": True}, {"max_tokens": 0},
            {"max_tokens": 1.5}, {"timeout_s": False}, {"timeout_s": 0},
            {"timeout_s": float("nan")}, {"timeout_s": float("inf")},
            {"timeout_s": 10 ** 1000}, {"comparison_group": " "},
            {"dataset": " "}, {"typo": True},
        ]
        with patch.object(quality, "check_backend") as backend, \
                patch.object(quality, "request_completion") as request:
            for index, change in enumerate(changes):
                output = self.root / f"invalid-{index}"
                with self.subTest(change=change), self.assertRaises(ValueError):
                    quality.run_quality({**self.config, **change}, output, "mock")
                self.assertFalse(output.exists())
        backend.assert_not_called()
        request.assert_not_called()

    def test_one_request_per_case_uses_unchanged_prompt_and_fixed_decoding(self):
        config_before = copy.deepcopy(self.config)
        environment = {"note": ["fixture"]}
        output = self.root / "passing"
        result = self.run_offline(output, environment=environment)
        self.assertEqual(len(self.calls), len(self.cases))
        for index, (call, case) in enumerate(zip(self.calls, self.cases)):
            self.assertEqual(call, {"base_url": self.config["base_url"],
                                   "model": "fixture-model", "prompt": case["prompt"],
                                   "max_tokens": self.config["max_tokens"],
                                   "timeout_s": self.config["timeout_s"],
                                   "request_id": index, "ignore_eos": False})
        self.assertEqual(result["kind"], "quality_check")
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["synthetic"])
        self.assertEqual(result["decoding"], {"temperature": 0, "ignore_eos": False})
        self.assertEqual(result["config"]["dataset_sha256"],
                         hashlib.sha256(self.dataset.read_bytes()).hexdigest())
        self.assertEqual(result["summary"], {"total": 3, "attempted": 3, "passed": 3,
                                           "mismatched": 0, "request_errors": 0,
                                           "unattempted": 0, "pass_rate": 1.0})
        self.assertEqual(self.read_result(output), result)
        result["environment"]["note"].append("changed")
        self.assertEqual(environment, {"note": ["fixture"]})
        self.assertEqual(self.config, config_before)

    def test_exact_match_and_request_errors_keep_answer_and_usage_evidence(self):
        responses = [self.response(" \npaRIS\t"), self.response("142"),
                     self.response("A-1", success=False, error="truncated stream")]
        responses_iter = iter(responses)
        result = self.run_offline(self.root / "outcomes",
                                  sender=lambda *args, **kwargs: next(responses_iter))
        self.assertEqual([row["outcome"] for row in result["results"]],
                         ["pass", "mismatch", "request_error"])
        for row, case, response in zip(result["results"], self.cases, responses):
            self.assertEqual(row["id"], case["id"])
            self.assertEqual(row["prompt"], case["prompt"])
            self.assertEqual(row["answers"], case["answers"])
            self.assertEqual(row["answer"], response.text)
            self.assertEqual(row["prompt_tokens"], 12)
            self.assertEqual(row["output_tokens"], 3)
        self.assertEqual(result["results"][2]["error"], "truncated stream")
        self.assertEqual(result["summary"], {"total": 3, "attempted": 3, "passed": 1,
                                           "mismatched": 1, "request_errors": 1,
                                           "unattempted": 0, "pass_rate": 1 / 3})

    def test_progress_is_saved_before_sending_the_next_case(self):
        output = self.root / "progress"

        def sender(*args, **kwargs):
            saved = self.read_result(output)
            self.assertEqual(saved["status"], "running")
            self.assertEqual(saved["summary"]["attempted"], len(self.calls))
            self.assertEqual(saved["summary"]["unattempted"], 3 - len(self.calls))
            return self.sender(*args, **kwargs)

        self.run_offline(output, sender=sender)

    def test_interrupt_keeps_completed_evidence_and_full_dataset_denominator(self):
        output = self.root / "interrupted"
        actions = [self.response("Paris"), KeyboardInterrupt()]
        with patch.object(quality, "check_backend"), \
                patch.object(quality, "request_completion", side_effect=actions) as request, \
                self.assertRaises(KeyboardInterrupt):
            quality.run_quality(self.config, output, "mock")
        self.assertEqual(request.call_count, 2)
        result = self.read_result(output)
        self.assertEqual(result["status"], "interrupted")
        self.assertEqual([row["outcome"] for row in result["results"]],
                         ["pass", "request_error"])
        self.assertEqual(result["summary"], {"total": 3, "attempted": 2, "passed": 1,
                                           "mismatched": 0, "request_errors": 1,
                                           "unattempted": 1, "pass_rate": 1 / 3})
        self.assertEqual(result["results"][0]["answer"], "Paris")
        self.assertEqual(result["results"][1]["answer"], "")
        self.assertIsNone(result["results"][1]["output_tokens"])
        self.assertIn("KeyboardInterrupt", result["results"][1]["error"])

    def test_unexpected_error_saves_aborted_run_and_is_not_swallowed(self):
        output = self.root / "aborted"
        with patch.object(quality, "check_backend"), \
                patch.object(quality, "request_completion", side_effect=[
                    self.response("Paris"), RuntimeError("fixture failure")]), \
                self.assertRaisesRegex(RuntimeError, "fixture failure"):
            quality.run_quality(self.config, output, "mock")
        result = self.read_result(output)
        self.assertEqual(result["status"], "aborted")
        self.assertEqual(result["summary"]["attempted"], 2)
        self.assertEqual(result["summary"]["unattempted"], 1)
        self.assertEqual(result["summary"]["pass_rate"], 1 / 3)
        self.assertEqual(result["results"][0]["answer"], "Paris")
        self.assertIn("fixture failure", result["results"][1]["error"])

    def test_existing_outputs_and_dangling_symlinks_are_never_overwritten(self):
        outputs = [self.root / "empty", self.root / "populated", self.root / "symlink"]
        outputs[0].mkdir()
        outputs[1].mkdir()
        sentinel = outputs[1] / "quality.json"
        sentinel.write_text("previous experiment")
        outputs[2].symlink_to(self.root / "missing", target_is_directory=True)
        with patch.object(quality, "check_backend") as backend, \
                patch.object(quality, "request_completion") as request:
            for output in outputs:
                with self.subTest(output=output), self.assertRaises(FileExistsError):
                    quality.run_quality(self.config, output, "mock")
        backend.assert_not_called()
        request.assert_not_called()
        self.assertEqual(list(outputs[0].iterdir()), [])
        self.assertEqual(sentinel.read_text(), "previous experiment")
        self.assertTrue(outputs[2].is_symlink())
        self.assertFalse((self.root / "missing").exists())

    def test_profile_and_runtime_metadata_preflight_does_not_contact_backend(self):
        gpu_config = {**self.config, "backend": "vllm"}
        environment = {"deployment_profile": "baseline", "model_id": "fixture/model",
                       "model_revision": "fixture-revision", "container_image_id": "fixture-image",
                       "gpu_model": "fixture GPU", "driver_version": "fixture-driver"}
        cases = [(self.config, "baseline", {}), (self.config, "unknown", {}),
                 (gpu_config, "mock", environment), (gpu_config, "awq", environment),
                 (self.config, "mock", []),
                 (self.config, "mock", {"bad": float("nan")})]
        cases.extend((gpu_config, "baseline", {**environment, key: " "})
                     for key in environment)
        with patch.object(quality, "check_backend") as backend, \
                patch.object(quality, "request_completion") as request:
            for index, (config, profile, metadata) in enumerate(cases):
                output = self.root / f"metadata-{index}"
                with self.subTest(profile=profile, metadata=metadata), self.assertRaises(ValueError):
                    quality.run_quality(config, output, profile, environment=metadata)
                self.assertFalse(output.exists())
        backend.assert_not_called()
        request.assert_not_called()

    def test_gpu_run_preserves_runtime_identity_without_mutating_inputs(self):
        config = {**self.config, "backend": "vllm"}
        environment = {"deployment_profile": "awq", "model_id": "fixture/model-awq",
                       "model_revision": "fixture-revision", "container_image_id": "fixture-image",
                       "gpu_model": "fixture GPU", "driver_version": "fixture-driver"}
        before = copy.deepcopy(environment)
        output = self.root / "gpu"
        with patch.object(quality, "check_backend") as backend, \
                patch.object(quality, "request_completion", side_effect=self.sender):
            result = quality.run_quality(config, output, "awq", environment=environment)
        backend.assert_called_once_with(config["base_url"], "vllm", config["timeout_s"])
        self.assertEqual(result["profile"], "awq")
        self.assertEqual(result["backend"], "vllm")
        self.assertFalse(result["synthetic"])
        for key, value in before.items():
            self.assertEqual(result["environment"][key], value)
        self.assertEqual(environment, before)
        self.assertEqual(config["backend"], "vllm")

    def test_backend_failure_keeps_aborted_run_with_all_cases_unattempted(self):
        output = self.root / "offline"
        with patch.object(quality, "check_backend", side_effect=ValueError("unavailable")), \
                patch.object(quality, "request_completion") as request, \
                self.assertRaisesRegex(ValueError, "unavailable"):
            quality.run_quality(self.config, output, "mock")
        result = self.read_result(output)
        self.assertEqual(result["status"], "aborted")
        self.assertEqual(result["results"], [])
        self.assertEqual(result["summary"], {"total": 3, "attempted": 0, "passed": 0,
                                           "mismatched": 0, "request_errors": 0,
                                           "unattempted": 3, "pass_rate": 0.0})
        request.assert_not_called()

    def test_cli_exit_codes_distinguish_pass_fail_setup_and_interrupt(self):
        cases = [([self.response("Paris"), self.response("42"), self.response("A-1")], 0),
                 ([self.response("Paris"), self.response("142"), self.response("A-1")], 1),
                 ([self.response("Paris"), self.response("42"),
                   self.response("A-1", success=False, error="stream error")], 1),
                 (KeyboardInterrupt(), 130)]
        for index, (action, expected) in enumerate(cases):
            output = self.root / f"cli-{index}"
            with self.subTest(expected=expected), patch.object(quality, "check_backend"), \
                    patch.object(quality, "request_completion", side_effect=action):
                self.assertEqual(self.cli(output), expected)
            self.assertTrue((output / "quality.json").exists())
        output = self.root / "cli-setup"
        with patch.object(quality, "check_backend", side_effect=ValueError("unavailable")), \
                patch.object(quality, "request_completion") as request:
            self.assertEqual(self.cli(output), 2)
        self.assertEqual(self.read_result(output)["status"], "aborted")
        request.assert_not_called()

    def test_cli_dataset_override_and_runtime_file_feed_saved_artifact(self):
        alternative = self.root / "alternative.jsonl"
        alternative.write_text(json.dumps({"id": "alternate", "prompt": "Exact prompt\n",
                                            "answers": ["OK"]}) + "\n")
        environment = self.root / "environment.json"
        environment.write_text(json.dumps({"note": "CLI fixture"}))
        output = self.root / "cli-override"
        with patch.object(quality, "check_backend"), \
                patch.object(quality, "request_completion", return_value=self.response("OK")) as request:
            self.assertEqual(self.cli(output, "--dataset", str(alternative),
                                      "--environment", str(environment)), 0)
        request.assert_called_once()
        result = self.read_result(output)
        self.assertEqual(result["results"][0]["prompt"], "Exact prompt\n")
        self.assertEqual(result["results"][0]["id"], "alternate")
        self.assertEqual(result["config"]["dataset_sha256"],
                         hashlib.sha256(alternative.read_bytes()).hexdigest())
        self.assertEqual(result["environment"]["note"], "CLI fixture")

    def test_cli_invalid_json_is_setup_error_without_network_or_files(self):
        path = self.root / "invalid.json"
        with patch.object(quality, "check_backend") as backend, \
                patch.object(quality, "request_completion") as request:
            for index, content in enumerate(("[]", "{bad json}", "null")):
                output = self.root / f"bad-json-{index}"
                path.write_text(content)
                with self.subTest(content=content), contextlib.redirect_stderr(io.StringIO()):
                    try:
                        code = quality.main(["--config", str(path), "--profile", "mock",
                                             "--output-dir", str(output)])
                    except SystemExit as exc:
                        code = exc.code
                self.assertEqual(code, 2)
                self.assertFalse(output.exists())
        backend.assert_not_called()
        request.assert_not_called()


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import contextlib
import copy
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from inference_lab import quality, quality_report
from inference_lab.metrics import RequestResult


class QualityReportTests(unittest.TestCase):
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
            {"id": "colour", "prompt": "Return a blue synonym.", "answers": ["blue", "azure"]},
            {"id": "unicode", "prompt": "Return Straße.", "answers": ["Straße"]},
            {"id": "failure", "prompt": "Return complete.", "answers": ["complete"]},
        ]
        self.dataset.write_text("".join(json.dumps(row) + "\n" for row in self.cases),
                                encoding="utf-8")
        self.config = {**quality.DEFAULTS, "dataset": str(self.dataset),
                       "model": "fixture-model", "comparison_group": "quality-fixture"}
        self.environment = {"model_id": "fixture/model", "model_revision": "revision-1",
                            "container_image_id": "fixture-image", "gpu_model": "fixture GPU",
                            "driver_version": "fixture-driver"}
        self.reference = self.make_run("reference", "baseline", [
            ("Paris", True), ("142", True), ("A-1", False),
            ("blue", True), ("straße", True), ("partial", False),
        ])
        self.candidate = self.make_run("candidate", "awq", [
            ("London", True), ("42", True), ("A-1", True),
            ("blue", False), (" STRASSE\n", True), ("partial", False),
        ])
        self.original_candidate = self.read(self.candidate)

    def make_run(self, name, profile, answers, config=None, environment=None):
        settings = self.config if config is None else config
        metadata = {**self.environment, "deployment_profile": profile}
        if environment is not None:
            metadata.update(environment)
        responses = [RequestResult(request_id=index, text=text, success=success,
                                   status_code=200, error=None if success else "truncated stream",
                                   prompt_tokens=12, output_tokens=3)
                     for index, (text, success) in enumerate(answers)]
        with patch.object(quality, "check_backend"), \
                patch.object(quality, "request_completion", side_effect=responses):
            quality.run_quality(settings, self.root / name, profile, metadata)
        return self.root / name / "quality.json"

    def read(self, path):
        return json.loads(path.read_text(encoding="utf-8"))

    def write(self, path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n",
                        encoding="utf-8")

    def compare(self):
        return quality_report.build_comparison(self.reference, self.candidate)

    def cli(self, output, reference=None, candidate=None):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                return quality_report.main([str(reference or self.reference),
                                            str(candidate or self.candidate),
                                            "--output-dir", str(output)])
            except SystemExit as exc:
                return exc.code

    def reject_candidate(self, mutate):
        value = copy.deepcopy(self.original_candidate)
        mutate(value)
        self.write(self.candidate, value)
        try:
            with self.assertRaises(ValueError):
                self.compare()
        finally:
            self.write(self.candidate, self.original_candidate)

    def test_completed_runs_preserve_evidence_and_classify_transitions(self):
        reference, candidate = self.read(self.reference), self.read(self.candidate)
        source_bytes = [self.reference.read_bytes(), self.candidate.read_bytes()]
        self.dataset.unlink()
        with patch.object(quality, "check_backend") as backend, \
                patch.object(quality, "request_completion") as request:
            result = self.compare()
        backend.assert_not_called()
        request.assert_not_called()
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["kind"], "quality_comparison")
        self.assertEqual(result["backend"], "vllm")
        self.assertFalse(result["synthetic"])
        self.assertEqual(result["dataset_sha256"], reference["config"]["dataset_sha256"])
        for name, source, artifact in (("reference", self.reference, reference),
                                       ("candidate", self.candidate, candidate)):
            self.assertEqual(result[name]["source"], str(source))
            for key in ("run_id", "profile", "backend", "synthetic", "model", "config",
                        "environment", "decoding", "summary", "created_at", "finished_at"):
                self.assertEqual(result[name][key], artifact[key], (name, key))
        self.assertEqual(result["summary"], {
            "total": 6, "reference_passed": 3, "candidate_passed": 3,
            "reference_request_errors": 2, "candidate_request_errors": 2,
            "new_mismatches": 1, "resolved_mismatches": 1,
            "request_error_cases": 3, "answer_changes": 3,
        })
        self.assertEqual([row["transition"] for row in result["cases"]], [
            "pass->mismatch", "mismatch->pass", "request_error->pass",
            "pass->request_error", "pass->pass", "request_error->request_error",
        ])
        for index, row in enumerate(result["cases"]):
            for name, artifact in (("reference", reference), ("candidate", candidate)):
                for key in ("answer", "outcome", "error", "prompt_tokens", "output_tokens",
                            "status_code"):
                    self.assertEqual(row[name][key], artifact["results"][index][key])
        self.assertEqual([self.reference.read_bytes(), self.candidate.read_bytes()], source_bytes)

    def test_cases_match_by_id_in_reference_order(self):
        candidate = self.read(self.candidate)
        candidate["results"] = list(reversed(candidate["results"]))
        self.write(self.candidate, candidate)
        result = self.compare()
        self.assertEqual([row["id"] for row in result["cases"]],
                         [case["id"] for case in self.cases])
        for row, original in zip(result["cases"], self.cases):
            self.assertEqual(row["prompt"], original["prompt"])
            self.assertEqual(row["answers"], original["answers"])
        unicode_case = result["cases"][4]
        self.assertEqual(unicode_case["transition"], "pass->pass")
        self.assertTrue(unicode_case["answer_changed"])
        self.assertEqual(unicode_case["candidate"]["answer"], " STRASSE\n")
        self.assertFalse(result["cases"][2]["answer_changed"])

    def test_environment_and_endpoint_changes_are_disclosed(self):
        reference, candidate = self.read(self.reference), self.read(self.candidate)
        reference["environment"]["reference_note"] = "full precision"
        candidate["environment"].update(gpu_model="another GPU", model_revision="revision-2",
                                        candidate_note="AWQ")
        candidate["config"].update(base_url="http://127.0.0.1:9000/v1",
                                   dataset="missing/elsewhere.jsonl")
        self.write(self.reference, reference)
        self.write(self.candidate, candidate)
        result = self.compare()
        self.assertEqual(result["config_changes"], {
            "base_url": {"reference": self.config["base_url"],
                         "candidate": "http://127.0.0.1:9000/v1"},
            "dataset": {"reference": str(self.dataset), "candidate": "missing/elsewhere.jsonl"},
        })
        changes = result["environment_changes"]
        self.assertEqual(changes["deployment_profile"], {"reference": "baseline", "candidate": "awq"})
        self.assertEqual(changes["gpu_model"], {"reference": "fixture GPU", "candidate": "another GPU"})
        self.assertEqual(changes["reference_note"], {"reference": "full precision", "candidate": None})
        self.assertEqual(changes["candidate_note"], {"reference": None, "candidate": "AWQ"})
        self.assertNotIn("driver_version", changes)

    def test_cli_writes_lossless_json_and_csv_without_touching_sources(self):
        before = [self.reference.read_bytes(), self.candidate.read_bytes()]
        expected = self.compare()
        output = self.root / "comparison"
        self.assertEqual(self.cli(output), 0)
        self.assertEqual(self.read(output / "comparison.json"), expected)
        self.assertEqual({path.name for path in output.iterdir()}, {"comparison.json", "comparison.csv"})
        with (output / "comparison.csv").open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            self.assertEqual(reader.fieldnames, [
                "reference_profile", "candidate_profile", "backend", "synthetic", "id",
                "prompt", "answers", "reference_outcome", "candidate_outcome", "reference_answer",
                "candidate_answer", "transition", "answer_changed", "reference_error", "candidate_error",
            ])
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]["prompt"], self.cases[0]["prompt"])
        self.assertEqual(json.loads(rows[0]["answers"]), ["Paris", "PARIS"])
        self.assertEqual(rows[4]["candidate_answer"], " STRASSE\n")
        self.assertEqual(rows[2]["reference_error"], "truncated stream")
        self.assertEqual(rows[0]["reference_profile"], "baseline")
        self.assertEqual(rows[0]["candidate_profile"], "awq")
        self.assertEqual(rows[0]["backend"], "vllm")
        self.assertEqual([self.reference.read_bytes(), self.candidate.read_bytes()], before)

    def test_existing_directories_and_symlinks_are_not_overwritten(self):
        existing, populated, link = [self.root / name for name in ("empty", "populated", "link")]
        existing.mkdir()
        populated.mkdir()
        sentinel = populated / "comparison.json"
        sentinel.write_text("previous comparison", encoding="utf-8")
        target = self.root / "missing"
        link.symlink_to(target, target_is_directory=True)
        for output in (existing, populated, link, self.reference.parent):
            with self.subTest(output=output):
                self.assertEqual(self.cli(output), 2)
        self.assertEqual(list(existing.iterdir()), [])
        self.assertEqual(sentinel.read_text(), "previous comparison")
        self.assertTrue(link.is_symlink())
        self.assertFalse(target.exists())
        self.assertEqual(self.read(self.candidate), self.original_candidate)

    def test_invalid_inputs_fail_before_creating_output(self):
        invalid = self.root / "invalid.json"
        for index, payload in enumerate(("{bad json", "[]", "null", "{}", "NaN")):
            invalid.write_text(payload, encoding="utf-8")
            output = self.root / f"invalid-output-{index}"
            with self.subTest(payload=payload):
                self.assertEqual(self.cli(output, candidate=invalid), 2)
                self.assertFalse(output.exists())
        missing_output = self.root / "missing-output"
        self.assertEqual(self.cli(missing_output, reference=self.root / "absent.json"), 2)
        self.assertFalse(missing_output.exists())
        for field, value in (("schema_version", 2), ("kind", "benchmark"),
                             ("status", "running"), ("status", "interrupted"),
                             ("status", "aborted"), ("config", []), ("summary", None)):
            with self.subTest(field=field, value=value):
                self.reject_candidate(lambda row, key=field, item=value: row.update({key: item}))

    def test_incomplete_duplicate_or_malformed_cases_are_rejected(self):
        mutations = [
            ("empty", lambda run: run.update(results=[])),
            ("incomplete", lambda run: run["results"].pop()),
            ("duplicate", lambda run: run["results"].append(copy.deepcopy(run["results"][0]))),
            ("blank id", lambda run: run["results"][0].update(id=" ")),
            ("blank prompt", lambda run: run["results"][0].update(prompt="\n")),
            ("empty answers", lambda run: run["results"][0].update(answers=[])),
            ("nontext answer", lambda run: run["results"][0].update(answers=[42])),
            ("nonobject case", lambda run: run["results"].__setitem__(0, None)),
            ("missing evidence", lambda run: run["results"][0].pop("answer")),
            ("too many", lambda run: run.update(results=run["results"] * 167)),
        ]
        for name, mutate in mutations:
            with self.subTest(name=name):
                self.reject_candidate(mutate)

    def test_outcomes_and_summary_must_agree_with_raw_answers(self):
        def forged_pass(run):
            run["results"][0]["outcome"] = "pass"
            run["summary"].update(passed=4, mismatched=0, pass_rate=4 / 6)

        def partial_answer_as_pass(run):
            run["results"][3]["outcome"] = "pass"
            run["summary"].update(passed=4, request_errors=1, pass_rate=4 / 6)

        mutations = [
            ("forged pass", forged_pass),
            ("partial answer as pass", partial_answer_as_pass),
            ("failed HTTP status as pass", lambda run: run["results"][1].update(status_code=503)),
            ("punctuation", lambda run: run["results"][1].update(answer="42.")),
            ("extra prose", lambda run: run["results"][1].update(answer="The answer is 42")),
            ("unknown outcome", lambda run: run["results"][0].update(outcome="unknown")),
            ("wrong count", lambda run: run["summary"].update(passed=4)),
            ("wrong fraction", lambda run: run["summary"].update(pass_rate=1.0)),
            ("unattempted", lambda run: run["summary"].update(unattempted=1)),
            ("boolean count", lambda run: run["summary"].update(request_errors=True)),
            ("negative usage", lambda run: run["results"][1].update(output_tokens=-1)),
            ("boolean usage", lambda run: run["results"][1].update(prompt_tokens=True)),
            ("nontext answer", lambda run: run["results"][1].update(answer=None)),
        ]
        for name, mutate in mutations:
            with self.subTest(name=name):
                self.reject_candidate(mutate)

    def test_incompatible_experiments_cannot_be_compared(self):
        mutations = [
            ("group", lambda run: run["config"].update(comparison_group="another group")),
            ("max tokens", lambda run: run["config"].update(max_tokens=32)),
            ("timeout", lambda run: run["config"].update(timeout_s=60)),
            ("model", lambda run: (run.update(model="another-model"),
                                   run["config"].update(model="another-model"))),
            ("backend", lambda run: (run.update(backend="mock", synthetic=True, profile="mock"),
                                     run["config"].update(backend="mock"))),
            ("synthetic", lambda run: run.update(synthetic=True)),
            ("decoding", lambda run: run["decoding"].update(temperature=0.5)),
            ("ignore eos", lambda run: run["decoding"].update(ignore_eos=True)),
            ("digest", lambda run: run["config"].update(dataset_sha256="0" * 64)),
            ("malformed digest", lambda run: run["config"].update(dataset_sha256="not-a-digest")),
        ]
        for name, mutate in mutations:
            with self.subTest(name=name):
                self.reject_candidate(mutate)

    def test_claimed_digest_does_not_hide_changed_cases(self):
        mutations = [
            ("id", lambda run: run["results"][0].update(id="different")),
            ("raw prompt", lambda run: run["results"][0].update(prompt=self.cases[0]["prompt"].strip())),
            ("raw answers", lambda run: run["results"][0].update(answers=["paris", "PARIS"])),
            ("answer order", lambda run: run["results"][0]["answers"].reverse()),
        ]
        for name, mutate in mutations:
            with self.subTest(name=name):
                self.reject_candidate(mutate)

    def test_runtime_identity_and_self_comparisons_are_validated(self):
        reference = self.read(self.reference)
        mutations = [
            ("same run", lambda run: run.update(run_id=reference["run_id"])),
            ("blank run", lambda run: run.update(run_id=" ")),
            ("unknown profile", lambda run: run.update(profile="unknown")),
            ("mock profile", lambda run: run.update(profile="mock")),
            ("wrong deployment", lambda run: run["environment"].update(deployment_profile="baseline")),
            ("missing GPU metadata", lambda run: run["environment"].pop("gpu_model")),
            ("blank revision", lambda run: run["environment"].update(model_revision=" ")),
            ("mismatched alias", lambda run: run["config"].update(model="different")),
            ("mismatched backend", lambda run: run["config"].update(backend="mock")),
        ]
        for name, mutate in mutations:
            with self.subTest(name=name):
                self.reject_candidate(mutate)
        with self.assertRaises(ValueError):
            quality_report.build_comparison(self.reference, self.reference)

    def test_mock_runs_remain_synthetic_and_nullable_usage_is_preserved(self):
        config = {**self.config, "backend": "mock"}
        answers = [(case["answers"][0], True) for case in self.cases]
        first = self.make_run("mock-first", "mock", answers, config=config)
        second = self.make_run("mock-second", "mock", answers, config=config)
        artifact = self.read(first)
        artifact["results"][0].update(prompt_tokens=None, output_tokens=None, status_code=None)
        self.write(first, artifact)
        result = quality_report.build_comparison(first, second)
        self.assertEqual(result["backend"], "mock")
        self.assertTrue(result["synthetic"])
        self.assertEqual(result["summary"]["candidate_passed"], 6)
        for key in ("prompt_tokens", "output_tokens", "status_code"):
            self.assertIsNone(result["cases"][0]["reference"][key])
        with self.assertRaises(ValueError):
            quality_report.build_comparison(first, self.candidate)


if __name__ == "__main__":
    unittest.main()

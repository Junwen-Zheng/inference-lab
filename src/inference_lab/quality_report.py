"""Compare two completed exact-answer checks without contacting a server."""
from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import math
import re
from pathlib import Path

from .quality import _summary, normalize_answer, validate_config
from .sweep import PROFILES, RUNTIME_FIELDS

RESULT_FIELDS = ("answer", "outcome", "error", "prompt_tokens", "output_tokens", "status_code")
CSV_FIELDS = (
    "reference_profile", "candidate_profile", "backend", "synthetic", "id", "prompt",
    "answers", "reference_outcome", "candidate_outcome", "reference_answer",
    "candidate_answer", "transition", "answer_changed", "reference_error", "candidate_error",
)


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_rows(rows):
    if not isinstance(rows, list) or not 1 <= len(rows) <= 1000:
        raise ValueError("results must contain between 1 and 1000 cases")
    seen = set()
    for row in rows:
        required = {"id", "prompt", "answers", *RESULT_FIELDS}
        if not isinstance(row, dict) or not required <= row.keys():
            raise ValueError("each result must contain the case and response fields")
        if not _text(row["id"]) or not _text(row["prompt"]):
            raise ValueError("case id and prompt must be nonempty text")
        if row["id"] in seen:
            raise ValueError("duplicate case id: " + row["id"])
        seen.add(row["id"])
        answers = row["answers"]
        if not isinstance(answers, list) or not answers or any(not _text(a) for a in answers):
            raise ValueError("answers must be a nonempty list of nonempty strings")
        if not isinstance(row["answer"], str):
            raise ValueError("answer must be text")
        outcome = row["outcome"]
        if outcome not in ("pass", "mismatch", "request_error"):
            raise ValueError("invalid result outcome")
        if row["error"] is not None and not isinstance(row["error"], str):
            raise ValueError("error must be text or null")
        for field in ("prompt_tokens", "output_tokens"):
            value = row[field]
            if value is not None and (not _integer(value) or value < 0):
                raise ValueError(field + " must be a nonnegative integer or null")
        code = row["status_code"]
        if code is not None and (not _integer(code) or not 100 <= code <= 599):
            raise ValueError("status_code must be an HTTP status integer or null")
        if outcome != "request_error":
            if row["error"] is not None or code not in (None, 200):
                raise ValueError("scored answers must have no error and a successful HTTP status")
            passed = normalize_answer(row["answer"]) in {normalize_answer(a) for a in answers}
            if outcome != ("pass" if passed else "mismatch"):
                raise ValueError("outcome disagrees with answer for case: " + row["id"])


def _load(path):
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        raise ValueError("quality artifact must be a JSON object")
    _json(artifact)
    if (not _integer(artifact.get("schema_version")) or artifact["schema_version"] != 1
            or artifact.get("kind") != "quality_check" or artifact.get("status") != "completed"):
        raise ValueError("expected a completed schema version 1 quality_check artifact")
    if not _text(artifact.get("run_id")):
        raise ValueError("run_id must be nonempty text")
    config = artifact.get("config")
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    config = dict(config)
    digest = config.pop("dataset_sha256", None)
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        raise ValueError("dataset_sha256 must be a SHA-256 hex digest")
    validate_config(config)
    profile, backend = artifact.get("profile"), artifact.get("backend")
    if (profile not in PROFILES or backend != config["backend"]
            or (profile == "mock") != (backend == "mock")):
        raise ValueError("profile and backend must match configuration")
    if (type(artifact.get("synthetic")) is not bool
            or artifact["synthetic"] != (backend == "mock")):
        raise ValueError("synthetic must match backend")
    if artifact.get("model") != config["model"]:
        raise ValueError("model must match configuration")
    if _json(artifact.get("decoding")) != _json({"temperature": 0, "ignore_eos": False}):
        raise ValueError("decoding must use temperature 0 and ignore_eos false")
    environment = artifact.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("environment must be a JSON object")
    if backend == "vllm":
        if (environment.get("deployment_profile") != profile
                or any(not _text(environment.get(field)) for field in RUNTIME_FIELDS)):
            raise ValueError("vllm environment requires matching profile and runtime metadata")
    rows = artifact.get("results")
    _validate_rows(rows)
    expected = _summary(len(rows), rows)
    supplied = artifact.get("summary")
    if not isinstance(supplied, dict) or supplied.keys() != expected.keys():
        raise ValueError("summary must contain the quality summary fields")
    for key, value in expected.items():
        actual = supplied[key]
        if key == "pass_rate":
            valid = (not isinstance(actual, bool) and isinstance(actual, (int, float))
                     and actual == value and math.isfinite(actual))
        else:
            valid = _integer(actual) and actual == value
        if not valid:
            raise ValueError("summary disagrees with completed results: " + key)
    return artifact


def _changes(reference, candidate):
    return {key: {"reference": copy.deepcopy(reference.get(key)),
                  "candidate": copy.deepcopy(candidate.get(key))}
            for key in sorted(reference.keys() | candidate.keys())
            if key not in reference or key not in candidate
            or _json(reference[key]) != _json(candidate[key])}


def _metadata(artifact, path):
    fields = ("run_id", "profile", "backend", "synthetic", "model", "config", "environment",
              "decoding", "summary", "created_at", "finished_at")
    return {"source": str(path), **{key: copy.deepcopy(artifact[key])
                                  for key in fields if key in artifact}}


def build_comparison(reference_path, candidate_path):
    reference, candidate = _load(reference_path), _load(candidate_path)
    if reference["run_id"] == candidate["run_id"]:
        raise ValueError("reference and candidate must have distinct run IDs")
    for field in ("backend", "synthetic", "model", "decoding"):
        if reference[field] != candidate[field]:
            raise ValueError("artifacts differ in " + field)
    for field in ("comparison_group", "max_tokens", "timeout_s"):
        if reference["config"][field] != candidate["config"][field]:
            raise ValueError("configurations differ in " + field)
    digest = reference["config"]["dataset_sha256"]
    if digest.lower() != candidate["config"]["dataset_sha256"].lower():
        raise ValueError("artifacts have different dataset hashes")
    candidates = {row["id"]: row for row in candidate["results"]}
    if set(candidates) != {row["id"] for row in reference["results"]}:
        raise ValueError("artifacts must contain the same case IDs")
    cases = []
    for old in reference["results"]:
        new = candidates[old["id"]]
        if old["prompt"] != new["prompt"] or old["answers"] != new["answers"]:
            raise ValueError("case prompt or accepted answers differ: " + old["id"])
        cases.append({"id": old["id"], "prompt": old["prompt"], "answers": old["answers"][:],
                      "reference": {key: old[key] for key in RESULT_FIELDS},
                      "candidate": {key: new[key] for key in RESULT_FIELDS},
                      "transition": old["outcome"] + "->" + new["outcome"],
                      "answer_changed": old["answer"] != new["answer"]})
    summary = {
        "total": len(cases), "reference_passed": reference["summary"]["passed"],
        "candidate_passed": candidate["summary"]["passed"],
        "reference_request_errors": reference["summary"]["request_errors"],
        "candidate_request_errors": candidate["summary"]["request_errors"],
        "new_mismatches": sum(row["transition"] == "pass->mismatch" for row in cases),
        "resolved_mismatches": sum(row["transition"] == "mismatch->pass" for row in cases),
        "request_error_cases": sum("request_error" in row["transition"] for row in cases),
        "answer_changes": sum(row["answer_changed"] for row in cases),
    }
    return {"schema_version": 1, "kind": "quality_comparison", "backend": reference["backend"],
            "synthetic": reference["synthetic"], "dataset_sha256": digest,
            "reference": _metadata(reference, reference_path),
            "candidate": _metadata(candidate, candidate_path), "summary": summary, "cases": cases,
            "environment_changes": _changes(reference["environment"], candidate["environment"]),
            "config_changes": _changes(reference["config"], candidate["config"])}


def _csv(comparison):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for row in comparison["cases"]:
        writer.writerow({
            "reference_profile": comparison["reference"]["profile"],
            "candidate_profile": comparison["candidate"]["profile"],
            "backend": comparison["backend"], "synthetic": comparison["synthetic"],
            **{key: row[key] for key in ("id", "prompt", "transition", "answer_changed")},
            "answers": json.dumps(row["answers"], ensure_ascii=False),
            **{f"{side}_{key}": row[side][key] for side in ("reference", "candidate")
               for key in ("outcome", "answer", "error")},
        })
    return buffer.getvalue()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output_dir.exists() or args.output_dir.is_symlink():
            raise FileExistsError(f"output directory already exists: {args.output_dir}")
        comparison = build_comparison(args.reference, args.candidate)
        payload = json.dumps(comparison, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        csv_payload = _csv(comparison)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "comparison.json").write_text(payload, encoding="utf-8")
        with (args.output_dir / "comparison.csv").open("x", encoding="utf-8", newline="") as handle:
            handle.write(csv_payload)
        print(json.dumps({"output": str(args.output_dir), **comparison["summary"]}))
        return 0
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        parser.exit(2, f"quality_report: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())

"""Run a small exact-answer smoke check against one serving profile."""
from __future__ import annotations

import argparse
import copy
import hashlib
import http.client
import json
import math
import platform
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from .benchmark import check_backend, git_revision
from .client import request_completion
from .sweep import PROFILES, RUNTIME_FIELDS

DEFAULTS = {
    "base_url": "http://127.0.0.1:8000", "model": "inference-model",
    "backend": "vllm", "max_tokens": 64, "timeout_s": 120.0,
    "comparison_group": "qwen2.5-1.5b", "dataset": "data/quality.jsonl",
}


def validate_config(config):
    if not isinstance(config, dict) or any(not isinstance(key, str) for key in config):
        raise ValueError("configuration must be a JSON object with text keys")
    missing, unknown = set(DEFAULTS) - set(config), set(config) - set(DEFAULTS)
    if missing:
        raise ValueError("missing configuration fields: " + ", ".join(sorted(missing)))
    if unknown:
        raise ValueError("unknown configuration fields: " + ", ".join(sorted(unknown)))
    for key in ("base_url", "model", "comparison_group", "dataset"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} must be nonempty text")
    if config["backend"] not in ("mock", "vllm"):
        raise ValueError("backend must be mock or vllm")
    value = config["max_tokens"]
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65536:
        raise ValueError("max_tokens must be an integer between 1 and 65536")
    value = config["timeout_s"]
    try:
        valid_timeout = (not isinstance(value, bool) and isinstance(value, (int, float))
                         and math.isfinite(value) and value > 0)
    except OverflowError:
        valid_timeout = False
    if not valid_timeout:
        raise ValueError("timeout_s must be a positive finite number")
    url = urlsplit(config["base_url"])
    if (url.scheme not in ("http", "https") or not url.hostname or url.username
            or url.password or url.query or url.fragment
            or url.path not in ("", "/", "/v1", "/v1/")
            or any(char.isspace() for char in config["base_url"])):
        raise ValueError("base_url must be an http(s) origin, optionally ending in /v1")
    if url.port is not None and not 1 <= url.port <= 65535:
        raise ValueError("base_url port must be between 1 and 65535")


def normalize_answer(text):
    return " ".join(text.split()).casefold()


def load_dataset(path):
    raw = Path(path).read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if not 1 <= len(rows) <= 1000:
        raise ValueError("dataset must contain between 1 and 1000 cases")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "prompt", "answers"}:
            raise ValueError("each dataset case must contain only id, prompt and answers")
        if any(not isinstance(row[key], str) or not row[key].strip()
               for key in ("id", "prompt")):
            raise ValueError("case id and prompt must be nonempty text")
        if row["id"] in seen:
            raise ValueError("duplicate case id: " + row["id"])
        seen.add(row["id"])
        answers = row["answers"]
        if (not isinstance(answers, list) or not answers
                or any(not isinstance(answer, str) or not answer.strip() for answer in answers)):
            raise ValueError("answers must be a nonempty list of nonempty strings")
    return rows, hashlib.sha256(raw).hexdigest()


def _write_json(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as handle:
            handle.write(payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _summary(total, results):
    counts = {outcome: sum(row["outcome"] == outcome for row in results)
              for outcome in ("pass", "mismatch", "request_error")}
    return {"total": total, "attempted": len(results), "passed": counts["pass"],
            "mismatched": counts["mismatch"], "request_errors": counts["request_error"],
            "unattempted": total - len(results), "pass_rate": counts["pass"] / total}


def run_quality(config, output_dir, profile, environment=None):
    validate_config(config)
    config = copy.deepcopy(config)
    if profile not in PROFILES or (profile == "mock") != (config["backend"] == "mock"):
        raise ValueError("profile must match the mock or vllm backend")
    if environment is not None and not isinstance(environment, dict):
        raise ValueError("environment must be a JSON object")
    environment = copy.deepcopy(environment or {})
    if config["backend"] == "vllm":
        if environment.get("deployment_profile") != profile:
            raise ValueError("environment deployment_profile must match the selected profile")
        if any(not isinstance(environment.get(key), str) or not environment[key].strip()
               for key in RUNTIME_FIELDS):
            raise ValueError("vllm environment requires: " + ", ".join(RUNTIME_FIELDS))
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    rows, digest = load_dataset(config["dataset"])
    json.dumps({"config": config, "environment": environment}, allow_nan=False)
    manifest = {
        "schema_version": 1, "kind": "quality_check", "run_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(), "profile": profile,
        "backend": config["backend"], "model": config["model"],
        "synthetic": config["backend"] == "mock",
        "config": {**config, "dataset_sha256": digest},
        "environment": {"python": platform.python_version(), "client_os": platform.system(),
                        "client_machine": platform.machine(), "git_revision": git_revision(),
                        **environment},
        "decoding": {"temperature": 0, "ignore_eos": False},
        "status": "running", "results": [], "summary": _summary(len(rows), []),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    artifact = output_dir / "quality.json"
    _write_json(artifact, manifest)
    current = None
    try:
        check_backend(config["base_url"], config["backend"], config["timeout_s"])
        for index, row in enumerate(rows):
            current = {**row, "answer": "", "outcome": "request_error", "error": None,
                       "prompt_tokens": None, "output_tokens": None, "status_code": None}
            result = request_completion(config["base_url"], config["model"], row["prompt"],
                                        config["max_tokens"], config["timeout_s"], index, False)
            matches = normalize_answer(result.text) in {
                normalize_answer(answer) for answer in row["answers"]}
            current.update(answer=result.text, error=result.error,
                           prompt_tokens=result.prompt_tokens, output_tokens=result.output_tokens,
                           status_code=result.status_code,
                           outcome=("pass" if matches else "mismatch") if result.success
                           else "request_error")
            manifest["results"].append(current)
            current = None
            manifest["summary"] = _summary(len(rows), manifest["results"])
            _write_json(artifact, manifest)
        manifest["status"] = "completed"
    except (KeyboardInterrupt, Exception) as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "aborted"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        if current is not None:
            current["error"] = manifest["error"]
            manifest["results"].append(current)
        manifest["summary"] = _summary(len(rows), manifest["results"])
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(artifact, manifest)
        raise
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(artifact, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--dataset")
    args = parser.parse_args(argv)
    try:
        supplied = json.loads(args.config.read_text())
        if not isinstance(supplied, dict):
            raise ValueError("configuration file must contain a JSON object")
        config = {**DEFAULTS, **supplied}
        if args.dataset is not None:
            config["dataset"] = args.dataset
        environment = json.loads(args.environment.read_text()) if args.environment else {}
        result = run_quality(config, args.output_dir, args.profile, environment)
        print(json.dumps({"output": str(args.output_dir / "quality.json"),
                          "status": result["status"], **result["summary"]}))
        return 0 if result["summary"]["passed"] == result["summary"]["total"] else 1
    except KeyboardInterrupt:
        print("quality: interrupted; partial artifact retained")
        return 130
    except (OSError, ValueError, TypeError, AttributeError, http.client.HTTPException) as exc:
        parser.exit(2, f"quality: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())

"""Repeat a concurrency grid against one already-running serving profile."""
from __future__ import annotations

import argparse
import copy
import http.client
import json
import math
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .benchmark import DEFAULTS, load_workload, run_benchmark, validate_config

PROFILES = ("mock", "baseline", "batched", "awq")
RUNTIME_FIELDS = ("model_id", "model_revision", "container_image_id", "gpu_model", "driver_version")


def _write_json(path, value, exclusive=False):
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x") as handle:
            handle.write(payload)
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as handle:
            handle.write(payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_sweep(config, output_dir, profile, concurrency=(1, 4, 8), repetitions=3,
              cooldown_s=2.0, environment=None):
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
    try:
        levels = tuple(concurrency)
    except TypeError as exc:
        raise ValueError("concurrency must be a sequence of positive integers") from exc
    if not levels or any(isinstance(n, bool) or not isinstance(n, int)
                         or not 1 <= n <= config["num_requests"] for n in levels):
        raise ValueError("concurrency values must be positive integers no greater than num_requests")
    if len(set(levels)) != len(levels):
        raise ValueError("concurrency values must be unique")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    if len(levels) * repetitions > 10000:
        raise ValueError("a sweep cannot exceed 10000 cases")
    try:
        valid_cooldown = (not isinstance(cooldown_s, bool)
                          and isinstance(cooldown_s, (int, float))
                          and math.isfinite(cooldown_s) and cooldown_s >= 0)
    except OverflowError:
        valid_cooldown = False
    if not valid_cooldown:
        raise ValueError("cooldown_s must be a nonnegative finite number")
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    _, digest = load_workload(config["workload"])
    # Fail on unsupported metadata before creating any artifacts.
    json.dumps({"config": config, "environment": environment}, allow_nan=False)
    sweep_id = str(uuid.uuid4())
    plan = []
    for repetition in range(1, repetitions + 1):
        offset = (repetition - 1) % len(levels)
        for level in levels[offset:] + levels[:offset]:
            plan.append({"filename": f"{profile}-c{level}-r{repetition}.json",
                         "repetition": repetition, "concurrency": level,
                         "order": len(plan) + 1, "status": "pending"})
    manifest = {"schema_version": 1, "kind": "concurrency_sweep", "sweep_id": sweep_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "profile": profile, "backend": config["backend"],
                "synthetic": config["backend"] == "mock", "base_config": config,
                "workload_sha256": digest, "environment": environment,
                "concurrency": list(levels), "repetitions": repetitions,
                "cooldown_s": cooldown_s, "status": "running", "failed_requests": 0,
                "plan": plan}
    output_dir.mkdir(parents=True, exist_ok=False)
    index = output_dir / "sweep.json"
    _write_json(index, manifest)
    current = None
    try:
        for case in plan:
            current = case
            case["status"] = "running"
            _write_json(index, manifest)
            if load_workload(config["workload"])[1] != digest:
                raise ValueError("workload changed during sweep")
            case_config = {**config, "concurrency": case["concurrency"],
                           "label": f"{profile}-c{case['concurrency']}-r{case['repetition']}"}
            case_environment = {**environment, "sweep_id": sweep_id,
                                "deployment_profile": profile,
                                "sweep_repetition": case["repetition"],
                                "sweep_order": case["order"],
                                "sweep_concurrency": case["concurrency"]}
            result = run_benchmark(case_config, case_environment)
            _write_json(output_dir / case["filename"], result, exclusive=True)
            if result.get("config", {}).get("workload_sha256") != digest:
                raise ValueError("recorded workload hash differs from sweep preflight")
            failed = result["summary"]["failed"]
            if isinstance(failed, bool) or not isinstance(failed, int) or failed < 0:
                raise ValueError("benchmark returned an invalid failed request count")
            case.update(status="completed", failed_requests=failed)
            manifest["failed_requests"] += failed
            _write_json(index, manifest)
            current = None
            if case["order"] < len(plan):
                time.sleep(cooldown_s)
        manifest["status"] = "completed"
        _write_json(index, manifest)
        return manifest
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        if current is not None:
            current.update(status="interrupted", error="KeyboardInterrupt")
        _write_json(index, manifest)
        raise
    except Exception as exc:
        manifest["status"] = "aborted"
        if current is not None:
            current.update(status="aborted", error=str(exc))
        _write_json(index, manifest)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--cooldown-s", type=float, default=2.0)
    args = parser.parse_args(argv)
    try:
        supplied = json.loads(args.config.read_text())
        if not isinstance(supplied, dict):
            raise ValueError("configuration file must contain a JSON object")
        config = {**DEFAULTS, **supplied}
        environment = json.loads(args.environment.read_text()) if args.environment else {}
        result = run_sweep(config, args.output_dir, args.profile, args.concurrency,
                           args.repetitions, args.cooldown_s, environment)
        print(json.dumps({"index": str(args.output_dir / "sweep.json"),
                          "status": result["status"], "cases": len(result["plan"]),
                          "failed_requests": result["failed_requests"]}))
        return 1 if result["failed_requests"] else 0
    except KeyboardInterrupt:
        print("sweep: interrupted; completed artifacts retained")
        return 130
    except (OSError, ValueError, TypeError, AttributeError, http.client.HTTPException) as exc:
        parser.exit(2, f"sweep: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())

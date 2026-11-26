"""Summarise repetitions from one completed sweep without contacting a server."""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
from pathlib import Path
from statistics import median

from .benchmark import validate_config
from .report import METRICS, load_run, metric
from .sweep import PROFILES, RUNTIME_FIELDS


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def _same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False)


def _manifest(index):
    if not isinstance(index, dict) or type(index.get("schema_version")) is not int or index["schema_version"] != 1:
        raise ValueError("expected a schema_version=1 sweep object")
    if index.get("kind") != "concurrency_sweep" or index.get("status") != "completed":
        raise ValueError("only completed concurrency sweeps can be summarised")
    config = index.get("base_config")
    validate_config(config)
    profile, backend = index.get("profile"), index.get("backend")
    if profile not in PROFILES or backend != config["backend"] or (profile == "mock") != (backend == "mock"):
        raise ValueError("profile and backend must match base_config")
    if type(index.get("synthetic")) is not bool or index["synthetic"] != (backend == "mock"):
        raise ValueError("synthetic flag must match the backend")
    _text(index.get("sweep_id"), "sweep_id")
    digest = index.get("workload_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        raise ValueError("workload_sha256 must be a 64-character hexadecimal hash")
    environment = index.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("environment must be an object")
    if backend == "vllm":
        if environment.get("deployment_profile") != profile:
            raise ValueError("environment deployment_profile must match the profile")
        for key in RUNTIME_FIELDS:
            _text(environment.get(key), f"environment.{key}")
    levels = index.get("concurrency")
    if not isinstance(levels, list) or not levels:
        raise ValueError("concurrency must be a nonempty list")
    for level in levels:
        if _integer(level, "concurrency", 1) > config["num_requests"]:
            raise ValueError("concurrency cannot exceed num_requests")
    if len(set(levels)) != len(levels):
        raise ValueError("concurrency values must be unique")
    repetitions = _integer(index.get("repetitions"), "repetitions", 1)
    count = len(levels) * repetitions
    if count > 10000:
        raise ValueError("a sweep cannot exceed 10000 cases")
    plan = index.get("plan")
    if not isinstance(plan, list) or len(plan) != count:
        raise ValueError("plan must contain every concurrency/repetition pair")
    _integer(index.get("failed_requests"), "failed_requests")


def _run_counts(run, case, num_requests):
    requests = run["requests"]
    successful = sum(request["success"] for request in requests)
    counts = {"attempted": len(requests), "successful": successful,
              "failed": len(requests) - successful}
    if counts["attempted"] != num_requests:
        raise ValueError("run request count differs from base_config.num_requests")
    for key, value in counts.items():
        if _integer(run["summary"].get(key), f"summary.{key}") != value:
            raise ValueError(f"summary.{key} differs from request records")
    if _integer(case.get("failed_requests"), "case failed_requests") != counts["failed"]:
        raise ValueError("case failed_requests differs from request records")
    return counts


def build_summary(index_path):
    index_path = Path(index_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    _manifest(index)
    profile, config = index["profile"], index["base_config"]
    repetitions, levels = index["repetitions"], index["concurrency"]
    pairs, filenames, run_ids = set(), set(), set()
    grouped = {level: [] for level in levels}
    invariant_environment = None
    for order, case in enumerate(index["plan"], start=1):
        if not isinstance(case, dict) or case.get("status") != "completed":
            raise ValueError("every planned case must be completed")
        repetition = _integer(case.get("repetition"), "case repetition", 1)
        level = _integer(case.get("concurrency"), "case concurrency", 1)
        if repetition > repetitions or level not in levels:
            raise ValueError("case is outside the declared concurrency/repetition grid")
        if _integer(case.get("order"), "case order", 1) != order:
            raise ValueError("case order must match its position in the plan")
        filename = f"{profile}-c{level}-r{repetition}.json"
        if case.get("filename") != filename:
            raise ValueError(f"case filename must be {filename}")
        pair = (level, repetition)
        if pair in pairs or filename in filenames:
            raise ValueError("plan contains a duplicate case or filename")
        pairs.add(pair)
        filenames.add(filename)
        run = load_run(index_path.parent / filename)
        run_id = _text(run.get("run_id"), "run_id")
        if run_id in run_ids:
            raise ValueError("run_id must be unique within the sweep")
        run_ids.add(run_id)
        if (run["backend"] != index["backend"] or run["model"] != config["model"]
                or type(run.get("synthetic")) is not bool or run["synthetic"] != index["synthetic"]):
            raise ValueError("run backend, model and synthetic flag must match the sweep")
        label = filename[:-5]
        expected_config = {**config, "concurrency": level, "label": label,
                           "workload_sha256": index["workload_sha256"]}
        if not _same(run["config"], expected_config) or run.get("label") != label:
            raise ValueError("run configuration or label differs from its planned case")
        environment = run.get("environment")
        if not isinstance(environment, dict):
            raise ValueError("run environment must be an object")
        expected_environment = {**index["environment"], "sweep_id": index["sweep_id"],
                                "deployment_profile": profile,
                                "sweep_repetition": repetition, "sweep_order": order,
                                "sweep_concurrency": level}
        for key, value in expected_environment.items():
            if key not in environment or not _same(environment[key], value):
                raise ValueError(f"run environment.{key} differs from the sweep")
        invariant = {key: value for key, value in environment.items()
                     if key not in ("sweep_id", "sweep_repetition", "sweep_order", "sweep_concurrency")}
        if invariant_environment is not None and not _same(invariant, invariant_environment):
            raise ValueError("run environment changed within the sweep")
        invariant_environment = invariant
        counts = _run_counts(run, case, config["num_requests"])
        grouped[level].append((repetition, filename, run, counts))

    groups = []
    for level in sorted(levels):
        runs = sorted(grouped[level])
        values = {}
        for key, _, _ in METRICS:
            samples = [metric(run, key) for _, _, run, _ in runs]
            complete = all(value is not None for value in samples)
            values[key] = {"values": samples, "observed_runs": sum(value is not None for value in samples),
                           "median": median(samples) if complete else None,
                           "min": min(samples) if complete else None,
                           "max": max(samples) if complete else None}
        groups.append({"concurrency": level, "repetitions": repetitions,
                       "run_files": [filename for _, filename, _, _ in runs],
                       **{key: sum(counts[key] for _, _, _, counts in runs)
                          for key in ("attempted", "successful", "failed")}, "metrics": values})
    failed = sum(group["failed"] for group in groups)
    if index["failed_requests"] != failed:
        raise ValueError("sweep failed_requests differs from request records")
    result = {"schema_version": 1, "kind": "sweep_summary", "source_index": str(index_path),
              **{key: index[key] for key in ("sweep_id", "profile", "backend", "synthetic",
                                            "workload_sha256", "base_config", "repetitions")},
              "environment": invariant_environment, "failed_requests": failed, "groups": groups}
    json.dumps(result, allow_nan=False)
    return result


def _csv(summary):
    output = io.StringIO(newline="")
    fields = ["profile", "backend", "synthetic", "concurrency", "repetitions", "attempted",
              "successful", "failed", "metric", "observed_runs", "median", "min", "max"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for group in summary["groups"]:
        row = {key: summary[key] for key in ("profile", "backend", "synthetic")}
        row.update({key: group[key] for key in ("concurrency", "repetitions", "attempted", "successful", "failed")})
        for key, _, _ in METRICS:
            stats = group["metrics"][key]
            writer.writerow({**row, "metric": key,
                             **{name: stats[name] for name in ("observed_runs", "median", "min", "max")}})
    return output.getvalue()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path, help="Completed sweep.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for summary.json and summary.csv")
    args = parser.parse_args(argv)
    try:
        if args.output_dir.exists() or args.output_dir.is_symlink():
            raise FileExistsError(f"output directory already exists: {args.output_dir}")
        result = build_summary(args.index)
        payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
        csv_payload = _csv(result)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        for filename, content in (("summary.json", payload), ("summary.csv", csv_payload)):
            with (args.output_dir / filename).open("x", encoding="utf-8", newline="") as handle:
                handle.write(content)
        print(f"Wrote {args.output_dir / 'summary.json'} and {args.output_dir / 'summary.csv'}")
        return 1 if result["failed_requests"] else 0
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        parser.exit(2, f"aggregate: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())

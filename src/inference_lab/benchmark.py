"""Fixed-concurrency, closed-loop streaming serving benchmark."""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import platform
import random
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from .client import request_completion
from .metrics import summarize

DEFAULTS = {
    "base_url": "http://127.0.0.1:8000", "model": "inference-model",
    "backend": "mock", "label": "local-demo", "num_requests": 24,
    "concurrency": 4, "max_tokens": 32, "warmup_requests": 2,
    "timeout_s": 60.0, "seed": 42, "ttft_slo_ms": 500.0,
    "e2e_slo_ms": 5000.0, "ignore_eos": False,
    "comparison_group": "qwen2.5-1.5b", "workload": "data/prompts.jsonl",
}


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("configuration must be a JSON object")
    missing = set(DEFAULTS) - set(config)
    if missing:
        raise ValueError("missing configuration fields: " + ", ".join(sorted(missing)))
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError("unknown configuration fields: " + ", ".join(sorted(unknown)))
    for key in ("num_requests", "concurrency", "max_tokens", "warmup_requests", "seed"):
        value = config[key]
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{key} must be an integer")
        if value < (0 if key in ("warmup_requests", "seed") else 1):
            raise ValueError(f"invalid {key}")
    if config["concurrency"] > config["num_requests"]:
        raise ValueError("concurrency cannot exceed num_requests")
    if config["num_requests"] > 1_000_000 or config["max_tokens"] > 65536:
        raise ValueError("request or output limit too large for this teaching harness")
    for key in ("timeout_s", "ttft_slo_ms", "e2e_slo_ms"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be a positive finite number")
    if config["backend"] not in ("mock", "vllm"):
        raise ValueError("backend must be mock or vllm")
    if not isinstance(config["ignore_eos"], bool):
        raise ValueError("ignore_eos must be boolean")
    for key in ("base_url", "model", "label", "comparison_group", "workload"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} must be nonempty text")


def load_workload(path):
    raw = Path(path).read_bytes()
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) or not isinstance(row.get("prompt"), str) or not row["prompt"].strip() for row in rows):
        raise ValueError("workload must contain JSONL objects with nonempty prompt strings")
    return rows, hashlib.sha256(raw).hexdigest()


def check_backend(base_url, backend, timeout_s):
    url = urlsplit(base_url)
    if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password or url.query or url.fragment or url.path.rstrip("/") not in ("", "/v1"):
        raise ValueError("base_url must be an http(s) origin, optionally ending in /v1")
    cls = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    conn = cls(url.hostname, url.port, timeout=timeout_s)
    response = None
    try:
        conn.request("GET", "/health")
        response = conn.getresponse()
        if response.status != 200:
            raise ValueError(f"health check returned HTTP {response.status}")
        is_mock = response.getheader("X-Inference-Lab-Backend") == "mock"
        if is_mock != (backend == "mock"):
            raise ValueError("backend label disagrees with server's synthetic marker")
    finally:
        if response is not None:
            response.close()
        conn.close()


def git_revision():
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=2)
        return completed.stdout.strip() if completed.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def run_benchmark(config, environment=None):
    validate_config(config)
    if environment is not None and not isinstance(environment, dict):
        raise ValueError("environment must be a JSON object")
    rows, digest = load_workload(config["workload"])
    check_backend(config["base_url"], config["backend"], config["timeout_s"])
    rng = random.Random(config["seed"])
    prompts = [rng.choice(rows)["prompt"] + f"\nRequest tag: {rng.getrandbits(64):016x}" for _ in range(config["num_requests"])]

    def send(request_id, prompt):
        return request_completion(config["base_url"], config["model"], prompt,
                                  config["max_tokens"], config["timeout_s"], request_id,
                                  config["ignore_eos"])

    for i in range(config["warmup_requests"]):
        result = send(-i - 1, rows[i % len(rows)]["prompt"] + f"\nWarmup tag: {i}")
        if not result.success:
            raise ValueError("warmup failed: " + str(result.error))
    # Submit one worker per concurrency slot, rather than queueing one Future
    # per request. Each worker sends its next request after the previous ends.
    jobs = iter(enumerate(prompts))
    job_lock = threading.Lock()

    def worker(_):
        batch = []
        while True:
            with job_lock:
                job = next(jobs, None)
            if job is None:
                return batch
            batch.append(send(*job))

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=config["concurrency"]) as pool:
        batches = list(pool.map(worker, range(config["concurrency"])))
    elapsed = time.perf_counter() - started
    results = sorted((result for batch in batches for result in batch),
                     key=lambda result: result.request_id)
    recorded_config = {**config, "workload_sha256": digest}
    return {
        "schema_version": 1, "run_id": str(uuid.uuid4()),
        "synthetic": config["backend"] == "mock",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label": config["label"], "backend": config["backend"], "model": config["model"],
        "config": recorded_config,
        "environment": {"python": platform.python_version(), "client_os": platform.system(),
                        "client_machine": platform.machine(), "git_revision": git_revision(),
                        **(environment or {})},
        "summary": summarize(results, elapsed, config["ttft_slo_ms"], config["e2e_slo_ms"]),
        "requests": [asdict(result) for result in results],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment", type=Path, help="explicit metadata JSON; never collects credentials")
    for key in ("base_url", "model", "backend", "label", "workload"):
        parser.add_argument("--" + key.replace("_", "-"))
    for key in ("num_requests", "concurrency", "max_tokens", "seed"):
        parser.add_argument("--" + key.replace("_", "-"), type=int)
    args = parser.parse_args(argv)
    try:
        supplied = json.loads(args.config.read_text()) if args.config else {}
        if not isinstance(supplied, dict):
            raise ValueError("configuration file must contain a JSON object")
        config = {**DEFAULTS, **supplied}
        for key in DEFAULTS:
            value = getattr(args, key, None)
            if value is not None:
                config[key] = value
        environment = json.loads(args.environment.read_text()) if args.environment else {}
        if not isinstance(environment, dict):
            raise ValueError("environment must be a JSON object")
        result = run_benchmark(config, environment)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"output": str(args.output), "backend": result["backend"], **result["summary"]}, indent=2))
        return 1 if result["summary"]["failed"] else 0
    except (OSError, ValueError, TypeError, AttributeError, http.client.HTTPException) as exc:
        parser.exit(2, f"benchmark: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())

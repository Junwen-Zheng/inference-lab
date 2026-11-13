"""Build an offline HTML comparison and optional CSV from benchmark runs."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


METRICS = (
    ("requests_per_second", "Requests / s", False),
    ("output_tokens_per_second", "Output tokens / s", False),
    ("ttft_ms.p95", "TTFT p95 (ms)", True),
    ("e2e_ms.p95", "End-to-end p95 (ms)", True),
    ("tpot_ms.p95", "TPOT p95 (ms)", True),
    ("slo.goodput_requests_per_second", "SLO goodput / s", False),
)


def number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        converted = float(value)
    except OverflowError:
        return None
    return converted if math.isfinite(converted) and converted >= 0 else None


def metric(run: Mapping[str, Any], key: str) -> float | None:
    value: Any = run.get("summary", {})
    for part in key.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return number(value)


def percent_change(value: Any, baseline: Any) -> float | None:
    current, reference = number(value), number(baseline)
    if current is None or reference is None or reference == 0:
        return None
    change = (current / reference - 1) * 100
    return change if math.isfinite(change) else None


def _validate_run(run: Any) -> None:
    if not isinstance(run, dict) or type(run.get("schema_version")) is not int or run["schema_version"] != 1:
        raise ValueError("expected a schema_version=1 run object")
    if run.get("backend") not in ("mock", "vllm"):
        raise ValueError("backend must be mock or vllm")
    if not isinstance(run.get("model"), str) or not run["model"].strip():
        raise ValueError("model must be nonempty text")
    for field in ("config", "summary"):
        if not isinstance(run.get(field), dict):
            raise ValueError(f"{field} must be an object")
    for field in ("label", "run_id", "created_at"):
        if field in run and (not isinstance(run[field], str) or not run[field].strip()):
            raise ValueError(f"{field} must be nonempty text")
    if "environment" in run and not isinstance(run["environment"], dict):
        raise ValueError("environment must be an object")
    if "synthetic" in run and not isinstance(run["synthetic"], bool):
        raise ValueError("synthetic must be boolean")
    if "synthetic" in run and run["synthetic"] != (run["backend"] == "mock"):
        raise ValueError("synthetic flag must match the backend")

    config = run["config"]
    if "backend" in config and config["backend"] != run["backend"]:
        raise ValueError("config.backend must match the run backend")
    for key in ("num_requests", "concurrency", "max_tokens", "warmup_requests", "seed"):
        if key in config:
            minimum = 0 if key in ("warmup_requests", "seed") else 1
            if type(config[key]) is not int or config[key] < minimum:
                raise ValueError(f"config.{key} must be an integer >= {minimum}")
    if "ignore_eos" in config and not isinstance(config["ignore_eos"], bool):
        raise ValueError("config.ignore_eos must be boolean")
    for key in ("timeout_s", "ttft_slo_ms", "e2e_slo_ms"):
        if key in config and (number(config[key]) is None or config[key] <= 0):
            raise ValueError(f"config.{key} must be a positive finite number")
    for key in ("base_url", "model", "backend", "label", "comparison_group", "workload", "workload_sha256"):
        if key in config and (not isinstance(config[key], str) or not config[key].strip()):
            raise ValueError(f"config.{key} must be nonempty text")

    if not isinstance(run.get("requests"), list):
        raise ValueError("requests must be a list")
    for index, request in enumerate(run["requests"]):
        if not isinstance(request, dict) or not isinstance(request.get("success"), bool):
            raise ValueError(f"requests[{index}] must be an object with boolean success")
        if "request_id" in request and (type(request["request_id"]) is not int or request["request_id"] < 0):
            raise ValueError(f"requests[{index}].request_id must be a nonnegative integer")
        status = request.get("status_code")
        if status is not None and (type(status) is not int or not 100 <= status <= 599):
            raise ValueError(f"requests[{index}].status_code must be an HTTP status or null")
        for field in ("error", "text"):
            if request.get(field) is not None and not isinstance(request[field], str):
                raise ValueError(f"requests[{index}].{field} must be text or null")


def load_run(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        run = json.load(handle)
    try:
        _validate_run(run)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    return run


def comparison_warnings(runs: Sequence[Mapping[str, Any]]) -> list[str]:
    if not runs:
        raise ValueError("at least one benchmark run is required")
    for run in runs:
        _validate_run(run)
    baseline = runs[0]
    baseline_config = baseline["config"]
    warnings: list[str] = []
    strict_keys = ("workload_sha256", "max_tokens", "num_requests", "seed", "ignore_eos",
                   "ttft_slo_ms", "e2e_slo_ms")
    for index, run in enumerate(runs[1:], start=2):
        config = run["config"]
        if run["backend"] != baseline["backend"]:
            raise ValueError("mock and vllm runs cannot be compared; create separate reports")
        for key in strict_keys:
            if (key in config) != (key in baseline_config) or config.get(key) != baseline_config.get(key):
                raise ValueError(f"run {index}: incompatible comparison setting {key}; use matching workload and SLO settings")
        if run["model"] != baseline["model"]:
            group = config.get("comparison_group")
            if not group or group != baseline_config.get("comparison_group"):
                raise ValueError(f"run {index}: different model IDs require the same non-empty config.comparison_group")
            warnings.append(f"Run {index} uses a different model ID in comparison group {group}; this metadata does not establish equivalent model quality.")
        changed = [key for key in sorted(set(config) | set(baseline_config))
                   if config.get(key) != baseline_config.get(key) and key not in strict_keys]
        if changed:
            warnings.append(f"Run {index} differs from the first run in config: {', '.join(changed)}.")
        if run.get("environment") != baseline.get("environment"):
            warnings.append(f"Run {index} has different environment metadata; hardware or serving changes can affect the comparison.")
    if any(not run["config"].get("workload_sha256") for run in runs):
        warnings.append("Workload hash missing: identical prompts cannot be verified.")
    if any(run["backend"] == "mock" for run in runs):
        warnings.insert(0, "SYNTHETIC MOCK RESULTS — these exercise the pipeline and provide no evidence of GPU or model performance.")
    return warnings


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _format(value: Any, decimals: int = 2) -> str:
    converted = number(value)
    return "N/A" if converted is None else f"{converted:,.{decimals}f}"


def _label(run: Mapping[str, Any], index: int) -> str:
    return str(run.get("label") or run.get("run_id") or f"Run {index + 1}")


def _chart(runs: Sequence[Mapping[str, Any]], key: str, title: str, lower: bool) -> str:
    values = [metric(run, key) for run in runs]
    maximum = max((value for value in values if value is not None and value > 0), default=1)
    height = 46 * len(runs) + 30
    rows = []
    for index, (run, value) in enumerate(zip(runs, values)):
        y = 16 + index * 46
        width = (value / maximum * 255) if value is not None else 0
        rows.append(
            f'<text x="8" y="{y + 11}" font-size="12">{_escape(_label(run, index)[:27])}</text>'
            f'<rect x="195" y="{y}" width="{width:.2f}" height="19" fill="#42668b"/>'
            f'<text x="{200 + width:.2f}" y="{y + 14}" font-size="12">{_format(value)}</text>'
        )
    direction = "lower is better" if lower else "higher is better"
    return (f'<section class="chart"><h2>{_escape(title)}</h2><p>{direction}</p>'
            f'<svg viewBox="0 0 560 {height}" role="img" aria-label="{_escape(title)}">'
            + "".join(rows) + "</svg></section>")


def build_report(runs: Sequence[Mapping[str, Any]]) -> str:
    warnings = comparison_warnings(runs)
    warning_html = "".join(f"<li>{_escape(item)}</li>" for item in warnings)
    headings = "".join(f"<th>{_escape(title)}</th>" for _, title, _ in METRICS)
    rows, details = [], []
    for index, run in enumerate(runs):
        label = _escape(_label(run, index))
        summary = run["summary"]
        cells = []
        for key, _, lower in METRICS:
            value = metric(run, key)
            change = percent_change(value, metric(runs[0], key)) if index else None
            change_html = ""
            if change is not None:
                improves = change < 0 if lower else change > 0
                css = "better" if improves else "worse" if change else ""
                change_html = f'<small class="{css}">{change:+.1f}% vs first</small>'
            cells.append(f"<td>{_format(value)}{change_html}</td>")
        attempted = _format(summary.get("attempted"), 0)
        successful = _format(summary.get("successful"), 0)
        failed = _format(summary.get("failed"), 0)
        rows.append(f"<tr><th>{label}<small>{_escape(run['model'])}</small></th>"
                    f"<td>{successful} / {attempted}</td><td>{failed}</td>" + "".join(cells) + "</tr>")
        errors = [request for request in run["requests"] if not request["success"]]
        error_html = "".join(
            f"<li>Request {_escape(request.get('request_id', '?'))}: "
            f"status {_escape(request.get('status_code'))}; {_escape(request.get('error') or 'unspecified failure')}</li>"
            for request in errors
        )
        error_note = ""
        failed_count = number(summary.get("failed"))
        if failed_count is not None and failed_count > len(errors):
            error_note = "<p>The summary reports failures without complete per-request error details.</p>"
        metadata = json.dumps({"run_id": run.get("run_id"), "created_at": run.get("created_at"),
                               "backend": run["backend"], "config": run["config"],
                               "environment": run.get("environment"), "summary": summary}, indent=2)
        details.append(f"<details><summary>{label}: configuration and errors ({len(errors)})</summary>"
                       f"<ul>{error_html}</ul>{error_note}<pre>{_escape(metadata)}</pre></details>")
    charts = "".join(_chart(runs, *entry) for entry in METRICS)
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Inference Lab benchmark comparison</title><style>
body{font:15px system-ui,sans-serif;max-width:1400px;margin:32px auto;padding:0 20px;color:#20252b;background:#fafafa}
h1{font-size:28px;margin-bottom:8px}h2{font-size:17px;margin:0}p{line-height:1.5}.notice{padding:12px 28px;background:#fff5dc;border-left:3px solid #a77519}
.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;background:white}th,td{padding:12px;text-align:right;border-bottom:1px solid #ddd;white-space:nowrap}th:first-child{text-align:left}small{display:block;font-weight:400;font-size:11px;color:#5b6470;margin-top:5px}.better{color:#19653a}.worse{color:#a02e2e}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,360px),1fr));gap:16px;margin:24px 0}.chart{background:#fff;padding:16px;border:1px solid #ddd}.chart p{font-size:12px;color:#5b6470;margin:5px 0}svg{width:100%;height:auto}details{padding:14px;background:#fff;border:1px solid #ddd;margin:10px 0}summary{cursor:pointer;font-weight:600}pre{overflow:auto;font-size:12px;background:#f5f5f5;padding:12px}footer{font-size:13px;color:#505963}
</style></head><body><h1>Inference Lab</h1>
<p>Client-observed performance. The first run is the reference; repeat runs to assess variability.</p>
""" + (f'<ul class="notice">{warning_html}</ul>' if warnings else "") + (
        '<div class="table-wrap"><table><thead><tr><th>Run / model</th><th>Success / attempted</th><th>Failed</th>'
        + headings + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
        + '<div class="charts">' + charts + "</div>" + "".join(details)
        + "<footer><p>N/A means unavailable or invalid. Request throughput counts successful responses. "
          "Token throughput uses server-reported completion tokens. TPOT is (end-to-end − TTFT) / (output tokens − 1) "
          "when at least two tokens are reported. Streaming chunk gaps are transport observations, not inter-token latency (ITL). "
          "SLO goodput counts successful requests meeting both configured latency limits, divided by total run duration.</p>"
          "</footer></body></html>"
    )


def write_csv(runs: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    comparison_warnings(runs)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    keys = [key for key, _, _ in METRICS]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_id", "label", "backend", "model", "attempted", "successful", "failed", *keys])
        writer.writeheader()
        for index, run in enumerate(runs):
            row = {key: run.get(key, "") for key in ("run_id", "backend", "model")}
            row["label"] = _label(run, index)
            for key in ("attempted", "successful", "failed"):
                value = run["summary"].get(key)
                row[key] = value if number(value) is not None else None
            row.update({key: metric(run, key) for key in keys})
            writer.writerow(row)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="Benchmark JSON files; first is the reference")
    parser.add_argument("--output", type=Path, required=True, help="Offline HTML report")
    parser.add_argument("--csv", type=Path, help="Optional summary CSV; unavailable metrics are blank")
    args = parser.parse_args(argv)
    try:
        runs = [load_run(path) for path in args.runs]
        report = build_report(runs)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        if args.csv:
            write_csv(runs, args.csv)
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(2, f"report: {exc}\n")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

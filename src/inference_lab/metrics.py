"""Summarise measured requests; latency inputs are seconds, summaries milliseconds.

Failures remain in the error rate but do not enter successful latency statistics.
Token throughput is unavailable if a successful request has unknown token usage.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import mean
from typing import Iterable


@dataclass
class RequestResult:
    request_id: int
    success: bool = False
    status_code: int | None = None
    error: str | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    ttft_s: float | None = None
    e2e_s: float = 0.0
    tpot_s: float | None = None
    chunk_gaps_s: list[float] = field(default_factory=list)
    text: str = ""


def percentile(values: Iterable[float], q: float) -> float | None:
    """Linear interpolation, including exact endpoints; empty sets are missing."""
    if not 0 <= q <= 100:
        raise ValueError("percentile must be between 0 and 100")
    data = sorted(values)
    if any(not math.isfinite(value) or value < 0 for value in data):
        raise ValueError("latency samples must be finite and nonnegative")
    if not data:
        return None
    index = (len(data) - 1) * q / 100
    low, high = math.floor(index), math.ceil(index)
    return data[low] + (data[high] - data[low]) * (index - low)


def distribution_ms(values: Iterable[float | None]) -> dict:
    data = [v * 1000 for v in values if v is not None]
    return {"mean": mean(data) if data else None,
            **{f"p{q}": percentile(data, q) for q in (50, 95, 99)}}


def summarize(results: list[RequestResult], duration_s: float,
              ttft_slo_ms: float, e2e_slo_ms: float) -> dict:
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("measurement duration must be finite and positive")
    if any(not math.isfinite(limit) or limit < 0 for limit in (ttft_slo_ms, e2e_slo_ms)):
        raise ValueError("SLO limits must be finite and nonnegative")
    good = [r for r in results if r.success]
    for result in good:
        if result.output_tokens is not None and (isinstance(result.output_tokens, bool) or
                not isinstance(result.output_tokens, int) or result.output_tokens < 0):
            raise ValueError("output token counts must be nonnegative integers or None")
    token_rate = (sum(r.output_tokens for r in good) / duration_s
                  if all(r.output_tokens is not None for r in good) else None)
    passing = [r for r in good if r.ttft_s is not None
               and r.ttft_s * 1000 <= ttft_slo_ms
               and r.e2e_s * 1000 <= e2e_slo_ms]
    return {
        "attempted": len(results), "successful": len(good),
        "failed": len(results) - len(good), "duration_s": duration_s,
        "requests_per_second": len(good) / duration_s,
        "output_tokens_per_second": token_rate,
        "error_rate": (len(results) - len(good)) / len(results) if results else 0.0,
        "ttft_ms": distribution_ms(r.ttft_s for r in good),
        "e2e_ms": distribution_ms(r.e2e_s for r in good),
        "tpot_ms": distribution_ms(r.tpot_s for r in good),
        "chunk_gap_ms": distribution_ms(gap for r in good for gap in r.chunk_gaps_s),
        "slo": {"ttft_ms": ttft_slo_ms, "e2e_ms": e2e_slo_ms,
                "good_requests": len(passing),
                "goodput_requests_per_second": len(passing) / duration_s},
    }

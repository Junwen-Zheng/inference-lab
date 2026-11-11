# Inference Lab

A project to deploy an open language model with vLLM on Kubernetes and study
how serving configuration affects latency, throughput, and answer quality.

**Status:** metric calculations and a measured streaming completions client are
implemented and unit tested. Serving deployment, load benchmarking, and
optimisation are planned; no GPU performance results are claimed yet.

## Engineering question

On a fixed GPU and workload, which serving configuration delivers the most
successful responses while meeting a declared latency target?

The experiment will compare:

| Configuration | Purpose |
| --- | --- |
| Full-precision, single-sequence baseline | Establish a reference measurement. |
| Full-precision, larger batching budget | Explore the latency/throughput tradeoff. |
| Quantised weights with aligned batching settings | Examine performance, memory use, and answer quality. |

The planned model family is Qwen2.5-1.5B. Exact runtime versions, model revisions,
and hardware requirements will be recorded when serving is implemented.

## Planned measurements

- Time to first token and end-to-end request latency, including tail percentiles.
- Successful requests per second and generated output tokens per second.
- Errors and SLO goodput: responses meeting the declared latency targets.
- Observed GPU memory use and a separate answer-quality diagnostic.

Comparisons will keep hardware and workload fixed, preserve raw request records,
and include repeated runs. Synthetic protocol checks will be labelled separately
from real model measurements. Quantisation is a hypothesis to evaluate; any
performance or quality conclusion will require recorded evidence.

## Local development

Use Python 3.10+ and Make on macOS or Linux. The current code needs no GPU,
Kubernetes cluster, model download, or third-party Python runtime dependencies.

```bash
make check
```

If your Python executable has a different name:

```bash
make check PYTHON=python3.12
```

This checks Python syntax, imports the package, and runs the metric and streaming-client tests.
Run only the tests with `make test`. The tests use an in-memory HTTP transport; they need no model server or GPU.

## Metric definitions

The metric module accepts per-request observations and the full measured run
length. It reports:

| Metric | Definition |
| --- | --- |
| Successful requests/s | Successful responses divided by run duration. |
| Output tokens/s | Completion tokens from successful responses divided by run duration. |
| Error rate | Failed requests divided by attempted requests. |
| TTFT / E2E / TPOT | Mean and interpolated p50, p95, p99 of successful observations, in milliseconds. |
| SLO goodput | Successful responses meeting both latency limits, divided by run duration. |

Latency inputs are seconds. SLO limits and reported latency statistics are
milliseconds. Missing latency statistics are `None`, not invented zeroes. If any
successful response lacks a token count, token throughput is `None`. Streaming
chunk gaps are transport observations, not true inter-token latency.

A hand-checkable example: two successful requests produce 10 and 20 tokens in a
two-second run, and a third request fails. The run has 1 successful request/s,
15 output tokens/s, and error rate 1/3. If only one response meets both latency
limits, goodput is 0.5 requests/s. The unit tests verify this example.

## Streaming client

`inference_lab.client.request_completion` sends one streamed request to a
vLLM-compatible `/v1/completions` endpoint. It returns the same `RequestResult`
record used by the metric module, including generated text, HTTP status, token
usage, timings, and any transport or protocol error.

- Base URLs may be an HTTP(S) origin or end in `/v1`.
- TTFT measures request start to the first nonempty text event; end-to-end
  latency ends when `[DONE]` arrives.
- Token counts come from the server's final usage, never the number of chunks.
- TPOT is an aggregate estimate: `(E2E - TTFT) / (output_tokens - 1)` for at
  least two output tokens. E2E includes usage and stream-finalisation overhead.
- Gaps between text chunks remain transport observations, not token intervals.
- Socket timeouts and a deadline limit the connected HTTP stream. Missing usage,
  truncated streams, malformed events, and HTTP failures are recorded as failed
  requests rather than successful measurements.

The client uses only the Python standard library. If the server requires a key,
set `INFERENCE_API_KEY` in your environment; it is sent as a Bearer header and
is not part of the result record. A compatible running server is needed for a
real completion; the current unit tests use simulated responses.

## Repository layout

```text
.
├── .gitignore
├── README.md
├── LICENSE
├── Makefile
├── pyproject.toml
├── src/inference_lab/
│   ├── __init__.py
│   ├── metrics.py
│   └── client.py
└── tests/
    ├── test_metrics.py
    └── test_client.py
```

The next implementation step is a local CPU streaming simulator for an
end-to-end HTTP smoke check.

## License

The project code is MIT licensed. Model weights and runtime components will
retain their own licenses.

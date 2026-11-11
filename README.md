# Inference Lab

A project to deploy an open language model with vLLM on Kubernetes and study
how serving configuration affects latency, throughput, and answer quality.

**Status:** metric calculations, a streaming client, a CPU simulator, and a
fixed-concurrency benchmark runner are implemented. Local protocol and runner
checks use no model or GPU. Real model deployment and optimisation experiments
remain planned; no GPU performance results are claimed yet.

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

This checks Python syntax, imports the package, and runs all tests. Run only
the tests with `make test`. Client unit tests use an in-memory HTTP transport;
simulator integration tests open a temporary loopback port. Neither requires
a GPU or model download.

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

## Run the local HTTP smoke check

```bash
make smoke
```

This starts the synthetic server on an unused `127.0.0.1` port, sends one
streaming completion through the real HTTP client, checks four mock output
units and nonempty text, saves `results/smoke.json`, and stops the server even
if the check fails. The JSON includes client timing, usage, output, and any
error, with explicit `backend: mock` and `synthetic: true` labels.

Expected output starts with:

```text
SYNTHETIC HTTP streaming smoke passed: 4 reported mock tokens.
```

To inspect a continuously running simulator, run `make mock` and stop it with
Ctrl+C. It listens on `127.0.0.1:8000` by default. The endpoints are:

| Endpoint | Fixture behaviour |
| --- | --- |
| `GET /health` | Reports healthy status with a mock-backend marker. |
| `GET /v1/models` | Lists the served alias `inference-model`. |
| `POST /v1/completions` | Streams `synthetic_N` text, a usage event, and `[DONE]`. |

The fixture validates request shape, alias, output limit, and streaming flag.
It supports simultaneous local connections and returns JSON errors for invalid
requests. Prompt counts are whitespace counts; output units and delays are
synthetic. The fixture performs no model inference or real tokenisation.

This smoke check verifies HTTP framing, stream parsing, measurement plumbing,
and cleanup. Its timings are not evidence of model or GPU performance.
Generated `results/` remain ignored by Git.

## Run a concurrency benchmark

Start the simulator in one terminal and leave it running:

```bash
make mock
```

From the repository root in a second terminal:

```bash
make benchmark
```

The default `configs/mock.json` sends two warmup requests, then measures 24
requests with at most four in flight and a 32-unit output budget. It samples
`data/prompts.jsonl` deterministically with seed 42 and records its SHA-256.
The server must already be running; the benchmark does not launch or deploy it.

The output is `results/mock-benchmark.json`. It contains per-request results,
configuration, workload hash, client environment, creation time, and a summary
of latency, successful request/token throughput, error rate, and SLO goodput.
Mock output is marked `backend: mock` and `synthetic: true`; its token counts
and timings provide no evidence of model or GPU performance.

Change the concurrency and preserve a separate run:

```bash
PYTHONPATH=src python3 -m inference_lab.benchmark \
  --config configs/mock.json --concurrency 8 --label mock-c8 \
  --output results/mock-c8.json
```

Warmup requests are excluded from both the saved request records and measured
run duration. Failed measured requests remain in the records and error rate.
The CLI saves a completed run and returns exit code 1 if measured requests fail;
configuration, health, and warmup failures return exit code 2. Inspect the error
instead of treating a failed run as successful capacity evidence.

This is a closed-loop workload: each worker waits for its current request before
sending another. It limits simultaneous requests, rather than offering an
independent arrival rate. The measured interval includes client worker startup
and the final drain. Read it as a workload-specific comparison, not an open-loop
production capacity estimate. Output paths can be reused, so choose distinct
filenames when preserving comparisons.

## Validation

```bash
make unit
```

The 32 metric, client, and benchmark orchestration unit tests pass without
opening network sockets. Runner tests use simulated request results to check concurrency,
warmup exclusion, reproducibility, failure retention, and the result artifact.
The existing simulator HTTP integration tests and live `make smoke` / benchmark
commands still require localhost access. They remain pending in the restricted
preparation environment. Run `make check`, `make smoke`, and the commands above
on your machine to verify the complete HTTP path. No GPU is required.

## Repository layout

```text
.
├── .gitignore
├── README.md
├── LICENSE
├── Makefile
├── pyproject.toml
├── configs/
│   └── mock.json
├── data/
│   └── prompts.jsonl
├── src/inference_lab/
│   ├── __init__.py
│   ├── metrics.py
│   ├── client.py
│   ├── mock_server.py
│   └── benchmark.py
├── scripts/
│   └── smoke.py
└── tests/
    ├── test_metrics.py
    ├── test_client.py
    ├── test_mock_server.py
    └── test_benchmark.py
```

The next implementation step is an offline comparison report for the recorded
benchmark JSON files.

## License

The project code is MIT licensed. Model weights and runtime components will
retain their own licenses.

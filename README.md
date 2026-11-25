# Inference Lab

An LLM serving project for comparing latency and throughput as batching and
quantisation settings change. The target is an open model served by vLLM on
Kubernetes; GPU measurements are still pending.

Implemented so far: streaming HTTP client, latency metrics, a synthetic server,
a concurrent benchmark runner, repeated sweeps, offline HTML/CSV reports, and
pinned local vLLM launch profiles. Kubernetes baseline, batching, and AWQ profiles render locally;
they have not been deployed to a cluster. Optional Prometheus monitoring includes
serving metrics and alert rules. No GPU performance results have been collected.

## Local setup

Python 3.10+ and Make are required. The benchmark tooling uses the standard library.
Run commands from the repository root:

```bash
make unit
make check
```

`unit` runs the offline tests. `check` also compiles the code and runs the HTTP
integration tests, which need permission to open a loopback port. Set
`PYTHON=python3.12` on any Make command if needed.

## Run a benchmark

First check the local HTTP path:

```bash
make smoke
```

This starts a temporary synthetic server, checks one streamed completion, saves
`results/smoke.json`, and stops the server. To benchmark, start `make mock` in
one terminal, then run these in a second terminal:

```bash
make benchmark
PYTHONPATH=src python3 -m inference_lab.benchmark \
  --config configs/mock.json --concurrency 8 --label mock-c8 \
  --output results/mock-c8.json
```

The default configuration warms up with two requests, then measures 24 requests
with four in flight and a 32-unit output budget. Prompt selection uses seed 42;
the workload SHA-256, configuration, environment, summary, and request records
are saved in `results/mock-benchmark.json`. Use distinct filenames to keep runs.

Workers send their next request after the previous finishes. This is a
closed-loop concurrency test; it does not simulate an independent arrival rate.
Warmup is excluded, while worker startup and final drain are measured. Failed
requests remain in the output. The CLI returns 1 for measured failures and 2
for setup, health, configuration, or warmup failures.

The simulator performs no inference. Its text, whitespace prompt counts, output
units, and delays are synthetic. Mock artifacts carry `backend: mock` and
`synthetic: true`; their timings cannot establish GPU performance. Generated
`results/` are ignored by Git.

## Compare saved runs

```bash
make report
PYTHONPATH=src python3 -m inference_lab.report \
  results/mock-benchmark.json results/mock-c8.json \
  --output results/comparison.html --csv results/comparison.csv
```

Open the HTML file in a browser. It includes six charts, signed changes from the
first run, request failures, and run metadata. It needs no JavaScript, network
access, or plotting packages. `make report` produces a single-run report at
`results/report.html` and `results/summary.csv`.

Comparisons require matching workload hashes, sample counts, seeds, output
budgets, EOS settings, and latency targets. Concurrency changes are allowed and
disclosed. Mock and vLLM runs cannot be mixed. Different model IDs require the
same explicit `comparison_group`; that setting records intent, not equivalent
answer quality. Environment differences are flagged for inspection. Reports
show missing measurements as `N/A`, with blank CSV cells.

## Repeat concurrency measurements

With `make mock` already running in another terminal:

```bash
make sweep
```

This runs concurrency 1, 4, and 8 three times and saves separate artifacts plus
an index in `results/mock-sweep`. Existing directories are refused; use
`SWEEP_DIR=results/mock-sweep-2` for another attempt. See
[sweeps](docs/sweeps.md) for real-profile metadata, partial runs, and reporting.

## Serve a real model

```bash
make gpu-plan
```

This previews a pinned Docker command without starting anything. The
`baseline`, `batched`, and `awq` profiles require a Linux NVIDIA GPU host to run.
See [local serving](docs/local-serving.md) for prerequisites, launch commands,
runtime capture, and the matching `configs/gpu.json` benchmark.

To inspect the Kubernetes profiles with `kubectl` installed:

```bash
make k8s-render
make k8s-render PROFILE=batched
make k8s-render PROFILE=awq
```

See [Kubernetes serving](docs/kubernetes.md) for the profile differences,
switching deployments, and recording comparable runs. Rendering is offline;
`PROFILE` also selects the Docker command shown by `make gpu-plan`.

## Monitor serving

```bash
make monitoring-render
make monitoring-check
```

Rendering needs `kubectl`; config and rule tests need `promtool` 3.15.1. Neither
command connects to a cluster. See [monitoring](docs/monitoring.md) for deployment,
queries, alert thresholds, and the differences from client benchmark metrics.

## Check answers before comparing speed

A separate quality runner sends eight fixed prompts once each and saves answers,
exact-match outcomes, request errors and runtime metadata. With a real serving
profile running, use [quality checks](docs/quality.md) to collect separate FP16
and AWQ artifacts. This small smoke suite does not establish general model
quality or quantisation equivalence.

## Measurement definitions

| Metric | Definition |
| --- | --- |
| Successful requests/s | Successful requests divided by measured run duration. |
| Output tokens/s | Successful output tokens divided by run duration. Missing if any successful request has unknown usage. |
| Error rate | Failed requests divided by attempted requests. |
| TTFT | Request start to the first nonempty text event. |
| E2E | Request start to receipt of `[DONE]`. |
| TPOT | `(E2E - TTFT) / (output_tokens - 1)` when at least two tokens are reported. |
| SLO goodput | Successful requests meeting both TTFT and E2E limits, divided by run duration. |

Latency summaries report mean and linearly interpolated p50/p95/p99 in
milliseconds, using successful requests only. Missing observations stay missing.
Token counts come from server usage, not chunk counts. TPOT includes stream
finalisation overhead; gaps between text chunks are transport timings rather
than true inter-token latency.

The client accepts HTTP(S) origins, optionally ending in `/v1`, and calls
`/v1/completions`. Set `INFERENCE_API_KEY` for a Bearer header when required.
Timeouts, missing usage, malformed events, truncated streams, and HTTP errors
are recorded as request failures.

## Validation and next experiments

Offline tests cover metrics, stream parsing, benchmark and sweep orchestration,
report generation and quality-check orchestration. Launcher tests use fake Docker
and kubectl commands.
`make unit` also checks Bash syntax. The monitoring manifests render locally;
Promtool rule tests and live metric scraping remain unverified in the preparation
environment. HTTP integration tests and live smoke/benchmark checks also remain
unverified there. Run `make check`, `make smoke`, and the benchmark commands locally
to verify that path.

Real experiments will record GPU hardware, runtime and model revisions, then
compare a baseline, larger batching budgets, and quantised weights on a fixed
workload. Repeated measurements and broader task-specific quality evaluation
are needed before drawing performance or quality conclusions.

Code is MIT licensed. Model weights and runtime components retain their own
licenses.

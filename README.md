# Inference Lab

An LLM serving project for comparing latency and throughput as batching and
quantisation settings change. The target is an open model served by vLLM on
Kubernetes; GPU measurements are still pending.

Implemented so far: streaming HTTP client, latency metrics, a synthetic server,
a concurrent benchmark runner, repeated sweeps, offline HTML/CSV reports, and
pinned local vLLM launch profiles. Kubernetes baseline, batching, and AWQ profiles render locally;
they have not been deployed to a cluster. Optional Prometheus monitoring includes
serving metrics and alert rules. No GPU performance results have been collected.

The [experiment writeup](docs/experiment.md) records the comparison design,
collection commands, current evidence and pending results.

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

## Continuous checks

The [CI workflow](.github/workflows/ci.yml) runs on pushes to `main`, pull
requests and manual dispatch. It checks Python 3.10 and 3.12, including loopback
HTTP integration and smoke tests, renders serving and monitoring manifests, and
checks Prometheus rules. The [latest verified run passed on 2026-11-26](https://github.com/Junwen-Zheng/inference-lab/actions/runs/45742843361),
before the cloud preparation increment.
See [CI checks](docs/ci.md) for local commands and what the jobs cover.

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

After a sweep completes, summarise its repetitions without contacting a server:

```bash
make sweep-summary
```

This writes `summary.json` and `summary.csv` under `results/mock-sweep-summary`,
with per-concurrency median, minimum and maximum for the six report metrics. Raw values,
failed request totals and synthetic labels are retained. Incomplete sweeps are
refused; missing metrics remain unavailable. See
[sweep summaries](docs/sweeps.md#summarise-repetitions) for real runs and limits.

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

The [rented GPU runbook](docs/cloud-gpu.md) prepares a single-node K3s session
on a cloud A10 VM, with a cost proposal, deadline guard, runtime evidence,
larger fixed workload and export/termination steps. Nothing has been rented or
measured yet. `configs/gpu-experiment.json` plans 120 requests per run with a
192-token ceiling; the original `configs/gpu.json` remains a smoke configuration.

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

After collecting batched FP16 and AWQ quality runs, compare their saved answers:

```bash
make quality-report
```

This writes `comparison.json` and `comparison.csv` to `results/quality-comparison`
without contacting a server. It pairs cases by ID, retains both answers and
request errors, and counts new and resolved exact-match mismatches. Incompatible
datasets or decoding settings are refused. See
[answer comparisons](docs/quality.md#compare-saved-answers) for input paths and limits.

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
report generation, sweep summaries and quality-check orchestration. Launcher tests use fake Docker
and kubectl commands.
`make unit` also checks Bash syntax. The recorded hosted CI run passed the Python tests and the
synthetic HTTP smoke check on Python 3.10 and 3.12, all four manifest renders,
Prometheus configuration validation and seven rule scenarios. Real GPU serving,
cluster deployment, live metric scraping and model-quality measurements remain
pending. CI's synthetic checks cannot establish inference performance.

Real experiments will record GPU hardware, runtime and model revisions, then
compare a baseline, larger batching budgets, and quantised weights on a fixed
workload. Repeated measurements and broader task-specific quality evaluation
are needed before drawing performance or quality conclusions. The
[writeup](docs/experiment.md#recorded-results) separates observed validation from
unmeasured hypotheses and explains which artifacts to publish with results.

Code is MIT licensed. Model weights and runtime components retain their own
licenses.

# Batching and quantisation experiment

The serving and measurement tools are implemented. GPU deployment, performance
and answer-quality measurements are pending. This document records the design
and the evidence collected so far; the table below contains no simulated scores.

## Questions and controls

1. Does increasing the scheduler's sequence and token budgets improve throughput
   at a given client concurrency, and what happens to TTFT and SLO goodput?
2. With those budgets fixed, how does an AWQ checkpoint change speed and answers?

These are hypotheses to measure. A higher token rate alone is insufficient if
requests fail, latency targets are missed or answer quality changes.

| Profile | Checkpoint | `max-num-seqs` | `max-num-batched-tokens` |
| --- | --- | ---: | ---: |
| `baseline` | Qwen2.5-1.5B-Instruct, FP16 | 1 | 2048 |
| `batched` | Same FP16 checkpoint and revision | 32 | 4096 |
| `awq` | Qwen2.5-1.5B-Instruct-AWQ, 4-bit weights | 32 | 4096 |

Baseline to batched changes two scheduler settings together, so it cannot
isolate either setting's contribution. Batched to AWQ keeps those budgets fixed
and changes the checkpoint and quantisation flag. It evaluates these particular
checkpoints rather than quantisation in general.

The [manifests](../deploy/k8s/base/deployment.yaml) fix the vLLM image, FP16
activation dtype, 2048-token context limit, seed 42 and disabled prefix caching.
Model revisions are pinned in the [base](../deploy/k8s/base/kustomization.yaml)
and [AWQ overlay](../deploy/k8s/overlays/awq/kustomization.yaml). Record actual
image IDs as well as tags. Keep the GPU, driver, client host, connection path and
background load consistent; record any node or runtime change.

The initial [GPU configuration](../configs/gpu.json) uses 24 measured requests,
two warmups per run, seed 42, temperature 0, a 32-token output budget and EOS
enabled, with a 120-second request timeout. TTFT and E2E targets are 500 ms and
5000 ms. The six prompts in
[`data/prompts.jsonl`](../data/prompts.jsonl) form a small systems-text workload.
The artifact stores its hash. Seeded selection and request tags reproduce the
same measured prompt sequence across profiles. This short workload is a starting
point; 24 requests give unstable tail estimates, especially p99. If increasing the sample
count or changing prompts, use the same revised configuration for every profile.

## Collect runs

Use the [Kubernetes instructions](kubernetes.md) to deploy and verify the
baseline, capture its manifest, Pod image ID, GPU and vLLM version, and connect
the loopback port-forward. Complete `results/baseline-environment.json` from
[`configs/environment.example.json`](../configs/environment.example.json).
Record the repository commit and client/cluster details with that evidence.
The metadata is supplied by the operator; the runner does not verify it remotely.

With baseline already serving, run from the repository root:

```bash
PYTHONPATH=src python3 -m inference_lab.sweep \
  --config configs/gpu.json --profile baseline \
  --environment results/baseline-environment.json \
  --concurrency 1 4 8 --repetitions 3 --cooldown-s 2 \
  --output-dir results/baseline-sweep

make quality PROFILE=baseline \
  QUALITY_ENVIRONMENT=results/baseline-environment.json \
  QUALITY_DIR=results/quality-baseline
```

This schedules nine benchmark runs, each with its own warmup. The sweep rotates
concurrency order within repetitions and retains failures and partial runs; see
[sweeps](sweeps.md). It uses fixed workers that send the next request after the
previous completes, so it does not measure an independent arrival rate or a
production capacity limit. Cold loading and startup time are outside this test.

Switch to batched, reconnect the port-forward, verify the active Pod and capture
fresh `results/batched-environment.json`. Repeat the commands with `batched` in
place of `baseline`, including every output path. Then repeat for `awq`.
All profiles replace the same Deployment and share the `inference-model` alias;
a label or successful health check does not identify which weights are running.
Use fresh directories for repeats. If extending the experiment to multiple
sessions, rotate profile order too and retain the session order in the notes.

The [quality runner](quality.md) uses a separate eight-case dataset and a
64-token budget. It records raw answers, accepted strings and errors. Compare
completed quality artifacts with matching dataset hashes and settings; inspect
every mismatch. Its exact-match score cannot establish general quality parity.

## Compare and publish evidence

After collecting all three complete sweeps, compare the concurrency-4 runs for
batching, then AWQ:

```bash
PYTHONPATH=src python3 -m inference_lab.report \
  results/baseline-sweep/baseline-c4-r*.json \
  results/batched-sweep/batched-c4-r*.json \
  --output results/batching-c4.html --csv results/batching-c4.csv

PYTHONPATH=src python3 -m inference_lab.report \
  results/batched-sweep/batched-c4-r*.json \
  results/awq-sweep/awq-c4-r*.json \
  --output results/awq-c4.html --csv results/awq-c4.csv
```

Repeat for concurrency 1 and 8 by changing `c4` in every path. Pass benchmark
files only; `sweep.json` and `quality.json` are different schemas. The report
shows each run separately, with changes relative to the first input (`r1` with
these filenames), rather than the mean of reference repetitions. Environment
warnings include repetition metadata; inspect actual hardware differences too.
It does not aggregate repetitions or calculate confidence intervals. For each
profile and concurrency, report all three values and their median and range; do not
average per-run percentiles into a pooled percentile or choose the fastest run.

Read token rate, successful requests/s, errors, p95 TTFT/E2E and SLO goodput
together. Missing observations stay unavailable. EOS can produce different
output lengths across profiles, so inspect usage before attributing a speed
change to the runtime. TTFT is measured at the client; TPOT includes stream
finalisation, and text chunk gaps are not individual token timings. Definitions
are in the [README](../README.md#measurement-definitions).

Generated `results/` files are ignored by Git. To publish evidence through the
GitHub website, copy the complete selected sweeps, their indexes, quality JSON,
reports and reviewed environment/manifest records into `docs/evidence/SESSION/`
in your local checkout. Upload that folder under `docs` and add its real links
below. Include failed attempts and explain exclusions. Artifacts contain prompt
and answer text; review them and metadata before publishing. No evidence folder
is supplied yet because real runs have not been collected.

## Recorded results

As of 2026-11-26, the hardware and runtime evidence needed for a performance
comparison has not been collected. The concurrency-4 summary remains pending:

| Profile | Output tokens/s | p95 TTFT / E2E (ms) | Goodput (requests/s) | Errors / attempts | Quality passes / 8 |
| --- | --- | --- | --- | --- | --- |
| Baseline | Pending | Pending | Pending | Pending | Pending |
| Batched | Pending | Pending | Pending | Pending | Pending |
| AWQ | Pending | Pending | Pending | Pending | Pending |

[CI run #3](https://github.com/Junwen-Zheng/inference-lab/actions/runs/45564811881)
passed on 2026-11-25 at
[`3327708`](https://github.com/Junwen-Zheng/inference-lab/commit/3327708b993bacd5496f71efd6561db6dabc2ed2):
99 tests and the synthetic HTTP smoke check on Python 3.10 and 3.12, four
Kustomize renders, Prometheus configuration validation and seven rule scenarios.
These check the measurement pipeline and static configuration. They provide no
GPU speed, cluster readiness or model-quality result.

The [preceding CI run](https://github.com/Junwen-Zheng/inference-lab/actions/runs/45563001965)
found a rate-fixture failure: an expected `30` was computed as
`29.999999999999993`. That exceeds Promtool's one-ULP fuzzy tolerance. The fix
rounds only that fixture's queried rate to six decimal places and adjusts the
expected labels; production recording and alert rules are unchanged. The
[fixture](../tests/test_prometheus_rules.yml) now passes in hosted CI. This was
a test precision issue, with no evidence of a serving throughput regression.

The next evidence needed is a successful completion on the chosen GPU, followed
by matched sweeps and answer checks. Until those artifacts exist, batching gains
and quantisation tradeoffs remain open questions.

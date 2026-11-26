# Repeated concurrency sweeps

A sweep runs against one already-running server. It does not launch, replace,
or configure that server. The same benchmark settings and prompt seed are used
for every run; only concurrency, repetition metadata, and labels change.

## Check the local pipeline

Start `make mock` in one terminal, then run this from the repository root in
another:

```bash
make sweep
```

This measures concurrency 1, 4, and 8 three times each, using the existing mock
configuration and no cooldown. Each run still gets its own warmup. Output goes
to `results/mock-sweep`; the synthetic labels and counts remain visible. This
checks orchestration, not model performance. Live HTTP execution remains
unverified in the restricted preparation environment.

Use a fresh directory for a second attempt:

```bash
make sweep SWEEP_DIR=results/mock-sweep-2
```

Existing directories are refused, including empty ones. The tool does not
resume a partial sweep or remove earlier results.

## Measure a serving profile

Deploy and verify a profile using [Docker](local-serving.md) or
[Kubernetes](kubernetes.md), then leave it running for the whole sweep. Capture
the actual runtime evidence using those instructions. Copy the metadata
template and replace every `null` with the observed value:

```bash
mkdir -p results
cp configs/environment.example.json results/baseline-environment.json
```

| Field | Evidence to use |
| --- | --- |
| `deployment_profile` | Ready Pod profile label or running Docker container label. |
| `model_id`, `model_revision` | Applied model ConfigMap or running container arguments. |
| `container_image_id` | Pod image ID or Docker's inspected image ID. |
| `gpu_model`, `driver_version` | Captured `nvidia-smi` output from the serving container. |

The file must be a JSON object with nonempty text in all six fields. Its profile
must match `--profile`. Extra reviewed fields, such as runtime version, node
name, and GPU memory, are retained. These are operator-supplied records; the
tool cannot establish that they still describe the live server. The shared
`inference-model` alias does not identify the checkpoint.

Once the baseline is serving and the metadata file is complete:

```bash
PYTHONPATH=src python3 -m inference_lab.sweep \
  --config configs/gpu.json --profile baseline \
  --environment results/baseline-environment.json \
  --concurrency 1 4 8 --repetitions 3 --cooldown-s 2 \
  --output-dir results/baseline-sweep
```

Concurrency values must be unique positive integers no larger than the request
count. Repetitions must be positive, with at most 10,000 total cases. Cooldown
must be finite and nonnegative. All settings are validated
before requests begin. Warmup and measured failures follow the benchmark's
existing rules.

## Read the output

Each directory contains full benchmark artifacts such as
`baseline-c4-r2.json`, plus a `sweep.json` index recording the plan, execution
order, settings, workload hash, and case status. The index is saved before
measurement and updated as cases finish.

For `[1, 4, 8]`, repetition order is `[1, 4, 8]`, then `[4, 8, 1]`, then
`[8, 1, 4]`. Rotation reduces a fixed ordering effect; it does not eliminate
thermal, cache, or background-load variation. Cooldown runs between cases,
never after the final case. Workload hash changes abort the sweep.

| Exit code | Meaning |
| --- | --- |
| 0 | Every planned case completed without measured request failures. |
| 1 | Cases completed, but one or more measured requests failed. |
| 2 | Invalid inputs or a setup, warmup, file, or workload consistency error. |
| 130 | Interrupted by Ctrl+C. |

Measured failures are saved and later cases continue. A setup error or
interruption stops the sweep and leaves completed artifacts and index status
available for inspection. The active request may have no complete run file.
Keep partial sweeps separate from complete comparisons.

Generate a report from the run files, excluding the index:

```bash
PYTHONPATH=src python3 -m inference_lab.report \
  results/baseline-sweep/baseline-c*-r*.json \
  --output results/baseline-sweep.html --csv results/baseline-sweep.csv
```

The report displays each repetition separately and compares it with the first
file passed. Inspect the spread, errors, and latency targets together. Do not
select only the fastest repetition or treat a short closed-loop sweep as a
production capacity result.

For batched or AWQ, switch the server first, reconnect any port-forward, capture
fresh metadata, and use a new output directory with the matching `--profile`.
Keep workload, seed, request count, output budget, latency targets, and hardware
fixed across profiles. Quantisation quality still needs a separate check; no
real GPU results have been collected by this project.

## Summarise repetitions

Use a completed sweep's index to calculate statistics across repetitions:

```bash
make sweep-summary
PYTHONPATH=src python3 -m inference_lab.aggregate \
  results/baseline-sweep/sweep.json --output-dir results/baseline-summary
```

The first command reads `results/mock-sweep/sweep.json` and writes to
`results/mock-sweep-summary`. Override `SWEEP_DIR` and `SUMMARY_DIR` for another
location. The second command is for a previously collected baseline sweep;
neither command starts a server or sends requests. Each output directory must
be new, including for a repeat of the same input.

`summary.json` groups runs by concurrency, with raw metric values ordered by
repetition and their median, minimum and maximum for the six HTML report
metrics. It retains profile, workload
hash, configuration, runtime metadata and source filenames. `summary.csv` has
one row per concurrency and metric. Failed requests stay in the totals; runs
with measured failures are included. Mock summaries remain explicitly synthetic.

The tool requires a complete plan and every run file. It rejects incomplete
sweeps, duplicate cases, mismatched settings or runtime metadata, and counts
that disagree with request records. Copy the index and all its run files
together when moving a sweep; the original workload path need not exist.

If any repetition lacks a metric, its raw value is `null` and that metric's
median/minimum/maximum all remain `null` (blank in CSV). `observed_runs` shows
how many values are available; it does not permit silently excluding a run.
These are statistics across run summaries: a median of per-run p95 values is
not the p95 of pooled requests. There are no confidence intervals or automatic
speedup claims. Review each run, error counts and SLO goodput alongside the spread.

Exit codes are 0 for a valid summary without request failures, 1 when the saved
summary includes failed requests, and 2 for invalid input or a file error.
A zero exit code does not mean latency targets were met. Quality artifacts are
separate; summarise each serving profile separately before comparing results.

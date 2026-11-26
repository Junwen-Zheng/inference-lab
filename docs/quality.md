# Answer checks

The eight authored cases in `data/quality.jsonl` cover arithmetic, extraction,
sorting and short-context lookup. Each row has a unique `id`, an unchanged raw
`prompt`, and a list of accepted `answers`. This is a formatting and correctness
smoke test, not a validated benchmark or a measure of general model quality.

The runner sends one request per case, sequentially, through the existing
`/v1/completions` client. It uses temperature 0, allows EOS and has no warmup or
request tags. The default output budget is 64 tokens. Prompts are raw completion
text, without a chat template; prompt formatting can affect instruction models.

## Collect two runs

Start the batched FP16 profile using the [serving instructions](kubernetes.md), then
record its runtime fields in `results/batched-environment.json` using
`configs/environment.example.json`. Set `deployment_profile` to `batched` and
fill every remaining field with the observed value. From the repository root:

```bash
make quality PROFILE=batched QUALITY_ENVIRONMENT=results/batched-environment.json
```

This writes `results/quality-batched/quality.json`. The profile must match the
metadata and backend. Existing output directories are refused; choose a fresh
`QUALITY_DIR=results/quality-batched-2` for a repeat.

Switch the serving deployment to AWQ, reconnect its port-forward, and capture a
new `results/awq-environment.json` with `deployment_profile` set to `awq`. Then:

```bash
make quality PROFILE=awq QUALITY_ENVIRONMENT=results/awq-environment.json
```

This writes `results/quality-awq/quality.json`. The runner records supplied model,
revision, image and GPU metadata; it does not remotely verify the server identity
or switch deployments. The Service alias is shared, so verify the active profile
before each run. `INFERENCE_API_KEY` supplies authentication when required.

To select a different dataset or connection config:

```bash
PYTHONPATH=src python3 -m inference_lab.quality \
  --config configs/quality.json --profile batched \
  --environment results/batched-environment.json \
  --dataset data/quality.jsonl --output-dir results/quality-batched-2
```

## Read the result

Scoring collapses whitespace and ignores case, then compares the whole answer
against the accepted strings. It preserves punctuation and numbers: `42.` does
not match `42`, and an answer containing the expected number in an explanation
does not pass. Inspect mismatches before attributing them to quantisation.

The JSON stores the dataset SHA-256, configuration, decoding settings, runtime
metadata, accepted answers and every returned response. Outcomes are `pass`,
`mismatch`, or `request_error`. Failed streams cannot pass on partial text;
missing usage or `[DONE]` is a request error under the existing client contract.
Use public prompts: the artifact contains prompt and answer text.

`pass_rate` is passed cases divided by the full dataset size. Request errors and
unattempted cases remain in the denominator. Results are saved after each
request; interruption or an unexpected error records a partial artifact with
status `interrupted` or `aborted`. A request interrupted in flight counts as an
attempted request error; its in-flight text is unavailable. Use `completed` runs
for comparison.

CLI exit codes are 0 when every case passes, 1 for mismatches or request errors,
2 for setup failures, and 130 for interruption. Health failures abort before
sending cases. Results use `kind: quality_check` and are separate from benchmark
JSON; do not pass them to the latency report command.

Compare completed batched FP16 and AWQ runs with the same dataset hash, output
budget, prompt format, decoding settings and model alias. Review each case as
well as the overall score, and record actual model revisions and hardware.
`comparison_group` records comparison intent; it proves no quality equivalence.
A score change on eight cases is a signal to investigate, without statistical
evidence of a general regression. GPU answers have not been collected here.

`make unit` checks scoring and orchestration with scripted client results and
no network. A mock-backed run is marked synthetic; the simulator performs no
inference and its scores cannot establish model quality.

## Compare saved answers

Once both quality artifacts exist, run this offline:

```bash
make quality-report
```

It reads `results/quality-batched/quality.json` as the reference and
`results/quality-awq/quality.json` as the candidate. The batched FP16 and AWQ
profiles use the same scheduler budgets. Each output directory must be new;
override `QUALITY_REPORT_DIR` for a repeat. To choose other saved runs:

```bash
PYTHONPATH=src python3 -m inference_lab.quality_report \
  results/quality-baseline/quality.json results/quality-awq/quality.json \
  --output-dir results/baseline-awq-comparison
```

Make also accepts `QUALITY_REFERENCE` and `QUALITY_CANDIDATE` paths. The tool
reads saved JSON only: it does not deploy profiles, send requests, or need the
original dataset file. Inputs must be different, completed runs with matching
backend, model alias, comparison group, dataset hash, output budget, timeout
and decoding settings. Mock and real runs cannot be mixed. Cases are paired
by ID in reference order; their raw prompts and accepted answers must match too.
Recorded totals and scored outcomes must agree with the saved answers and errors.

`comparison.json` retains both runs' metadata, scores, answers, usage and errors,
with per-case outcome transitions and raw answer-change flags. It records
configuration-path and environment differences for review, including checkpoint,
image and hardware changes. Those records are supplied metadata, not proof of
which model actually ran. The CSV has one paired case per row and keeps the
backend and synthetic labels visible.

`new_mismatches` counts pass → mismatch; `resolved_mismatches` counts mismatch
→ pass. Request errors are counted separately, including when partial text
matches an accepted answer. A changed string alone does not establish a quality
change: whitespace or case can differ while both answers pass. Inspect the raw
answers and errors alongside the totals; eight cases cannot establish general
quality equivalence or attribute a change to quantisation.

Exit 0 means the comparison files were written, even if there are mismatches
or request errors. Exit 2 means invalid input or a file error. This reporting
command does not gate model acceptance. Review the files before publishing;
they contain prompt and answer text.

# Inference Lab

A project to deploy an open language model with vLLM on Kubernetes and study
how serving configuration affects latency, throughput, and answer quality.

**Status:** project scaffold. Serving, benchmarking, and optimisation are planned;
no deployment or performance results are claimed yet.

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

Use Python 3.10+ and Make on macOS or Linux. The current scaffold needs no GPU,
Kubernetes cluster, model download, or third-party Python runtime dependencies.

```bash
make check
```

If your Python executable has a different name:

```bash
make check PYTHON=python3.12
```

This checks Python syntax and imports the package. No inference runs yet.

## Repository layout

```text
.
├── README.md
├── LICENSE
├── Makefile
├── pyproject.toml
└── src/
    └── inference_lab/
        └── __init__.py
```

The next implementation step is to define latency, throughput, error-rate, and
SLO-goodput calculations with small, hand-checkable examples.

## License

The project code is MIT licensed. Model weights and runtime components will
retain their own licenses.

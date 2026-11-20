# Serving metrics

Prometheus 3.15.1 scrapes the Kubernetes vLLM Service every five seconds and
evaluates rules every 15 seconds. The names match the pinned vLLM 0.32.0 runtime.
The synthetic server does not expose these metrics.

## Check locally

Install `kubectl` and the `promtool` binary from the matching
[Prometheus release](https://github.com/prometheus/prometheus/releases/tag/v3.15.1),
then run from the repository root:

```bash
make monitoring-render
make monitoring-check
```

Set `PROMTOOL=/absolute/path/to/promtool` if it is not on your PATH. Rendering
and rule tests use no cluster or GPU. The fixtures cover aggregation, counter
resets, excluded finish reasons, alert delays and recovery, idle and missing data.
The manifests were rendered during preparation; Promtool execution and live
scraping are still pending.

## Run on a cluster

First deploy a serving profile using the [Kubernetes instructions](kubernetes.md).
The `inference-lab` namespace must already exist. In that cluster context:

```bash
kubectl apply -k deploy/observability
kubectl -n inference-lab rollout status deployment/prometheus --timeout=180s
kubectl -n inference-lab port-forward --address=127.0.0.1 svc/prometheus 9090:9090
```

Open <http://127.0.0.1:9090>. Check the `vllm` target is UP and the rules loaded
before trusting any graphs. The static target is `vllm.inference-lab.svc:8000`;
it needs cluster DNS and Pod-to-Service access, without Kubernetes discovery or
RBAC permissions. A healthy Prometheus Pod alone does not prove scraping works.

With the vLLM Service forwarded to local port 8000 in another terminal, inspect
the live names and labels if a query is empty:

```bash
curl --fail http://127.0.0.1:8000/metrics
```

## Queries

Paste these expressions into Prometheus. Latency recordings are seconds.

| Signal | PromQL |
| --- | --- |
| Scrape health | `up{job="vllm"}` |
| Running requests | `sum by (model_name) (vllm:num_requests_running{job="vllm"})` |
| Queued requests | `sum by (model_name) (vllm:num_requests_waiting{job="vllm"})` |
| KV cache used (%) | `100 * vllm:kv_cache_usage_perc{job="vllm"}` |
| Generated tokens/s (1m) | `inference_lab:output_tokens_per_second:rate1m` |
| Stop/length completions/s (1m) | `inference_lab:requests_per_second:rate1m` |
| Server p95 TTFT (5m) | `inference_lab:ttft_seconds:p95` |
| Server p95 E2E (5m) | `inference_lab:e2e_seconds:p95` |

Counters are rated before summing across engines. Histogram buckets are summed
before estimating p95; this is not an average of per-engine percentiles. Idle
histograms return NaN, while absent metrics return no series. Neither is a zero
latency measurement. KV cache usage is a fraction of allocated KV cache blocks,
not total GPU memory utilisation.

The completion rate includes only `stop` and `length` endings; it excludes
`abort`, `error`, and `repetition`. Generated tokens include work from requests
that may later fail or abort. Server timings and rolling rates have different
boundaries from the client's measured run, successful output throughput and
SLO goodput. Keep the benchmark JSON as the experiment result.

All profiles use the same Service and model alias. Record profile and UTC start/
end times alongside each run, and wait at least five minutes after switching
profiles before reading a full latency window. A graph spanning the switch can
mix profiles; no profile label is inferred from the Service.

## Alerts and storage

| Alert | Condition |
| --- | --- |
| `VllmTargetUnavailable` | Scrape fails continuously for 2m. |
| `VllmQueueSustained` | More than 4 waiting requests per model for 2m. |
| `VllmTtftHigh` | Five-minute p95 server TTFT exceeds 2s for 5m. |

These are example warning thresholds, not measured SLOs. Alerts appear in the
Prometheus UI only; no Alertmanager or external notifications are configured.
Missing vLLM metrics do not trigger the latency/queue rules. Check target health
and rule evaluation errors when those graphs are empty.

This is a single-Pod lab setup with unauthenticated cluster access; use a trusted
cluster and the loopback port-forward. Samples live in an ephemeral volume and
are lost when the Pod is replaced. Retention is 24h or 512MB of TSDB blocks;
WAL/head data add overhead. The 1Gi volume size limit can cause Pod eviction.

To remove monitoring from the same cluster context:

```bash
kubectl delete -k deploy/observability
```

This removes the monitor and its config, leaving the serving namespace intact.
Reapply the manifests after editing config or rules; the generated ConfigMap
hash changes and replaces the Pod.

Metric definitions: [vLLM 0.32.0](https://docs.vllm.ai/en/v0.32.0/usage/metrics/),
[finish reasons](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/engine/__init__.py),
and [counter suffixes](https://prometheus.github.io/client_python/instrumenting/counter/).

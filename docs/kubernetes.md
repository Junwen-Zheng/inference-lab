# Kubernetes serving profiles

The profiles from [local serving](local-serving.md) run as one GPU Pod behind a
ClusterIP Service. All three manifests render locally; cluster admission,
scheduling, model loading, and inference remain unverified.
For the planned cloud execution, use the [rented GPU runbook](cloud-gpu.md).

| Profile | Checkpoint | Maximum sequences | Batched tokens |
| --- | --- | ---: | ---: |
| `baseline` | Qwen2.5-1.5B-Instruct, FP16 | 1 | 2048 |
| `batched` | Same FP16 checkpoint and revision | 32 | 4096 |
| `awq` | Qwen2.5-1.5B-Instruct-AWQ, 4-bit weights | 32 | 4096 |

Baseline to batched changes the sequence limit and token budget together. Batched
to AWQ holds those settings fixed and changes the checkpoint and quantisation
flag. All profiles retain the image, context limit, FP16 activations, seed, and
disabled prefix caching from Day 7. Model and tokenizer revisions stay pinned.

## Prepare a cluster

Use `kubectl` with Kustomize support and a Linux NVIDIA GPU node. Its driver and
container runtime must support the Day 7 CUDA image. The cluster needs an NVIDIA
device plugin or GPU Operator exposing `nvidia.com/gpu`; see
[GPU scheduling](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/).
Docker Desktop on your Mac does not provide this NVIDIA environment.

The Pod requests one GPU, 4 CPUs, and 8 GiB RAM, with limits of 8 CPUs and 16 GiB
RAM. It also needs a default StorageClass able to bind a 20 GiB ReadWriteOnce
volume, plus access to pull the image and weights. Check capacity and the
selected context before applying:

```bash
kubectl config current-context
kubectl get nodes -o custom-columns='NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu'
kubectl get storageclass
```

The manifest selects Linux nodes. Add GPU pool selectors, tolerations, or
`runtimeClassName: nvidia` only if required by your configured cluster. Set
`storageClassName` in `deploy/k8s/base/cache-pvc.yaml` if no default class is
available. GPU memory fit still needs to be measured on the selected hardware.

## Render and deploy

From the repository root:

```bash
make k8s-render > /tmp/inference-lab-baseline.yaml
make k8s-render PROFILE=batched > /tmp/inference-lab-batched.yaml
make k8s-render PROFILE=awq > /tmp/inference-lab-awq.yaml
```

Rendering is offline and creates no cluster resources. Inspect the result;
it does not prove API compatibility or GPU availability. Kustomize generates a
ConfigMap with a content hash and updates the Deployment reference when settings
change; see [Kustomize](https://kubernetes.io/docs/tasks/manage-kubernetes-objects/kustomization/).

With access to your intended cluster:

```bash
bash scripts/deploy.sh dry-run baseline
bash scripts/deploy.sh apply baseline
```

The client dry run may contact the API for discovery, but does not persist
resources. The apply command creates or updates the `inference-lab` namespace
and its resources, then waits up to 40 minutes for readiness. A 30-minute
[startup probe](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/)
allows cold downloads and compilation. Readiness gates Service traffic;
liveness can restart an unhealthy process after startup.

The Deployment uses `Recreate` to release its single GPU before replacing the
Pod, causing downtime during updates. Model files persist at `/cache` on the
PVC. A 2 GiB memory-backed `/dev/shm` volume counts against Pod memory. The
checkpoint, tokenizer revision, context budget, and scheduler settings match
the matching Day 7 Docker profile. New overlays merge only the intended
ConfigMap values; AWQ also appends `--quantization awq`. The shared base retains
the resource requests, probes, Service, and cache.

## Connect and measure

Keep this running in a separate terminal:

```bash
kubectl -n inference-lab port-forward --address 127.0.0.1 svc/vllm 8000:8000
```

The Service has no public endpoint or API authentication. Use a trusted cluster
and the loopback port-forward. Once the Pod is ready:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
PYTHONPATH=src python3 -m inference_lab.benchmark \
  --config configs/gpu.json --label k8s-baseline-c4 \
  --output results/k8s-baseline/run.json
```

The benchmark warms up before measuring and retains failed requests. A real
completion and inspected run artifact are required before claiming the model
served successfully. Save the rendered manifest and runtime evidence alongside
each run:

```bash
mkdir -p results/k8s-baseline
bash scripts/deploy.sh render baseline > results/k8s-baseline/manifest.yaml
kubectl -n inference-lab get pods -l app.kubernetes.io/name=vllm \
  -o custom-columns='POD:.metadata.name,NODE:.spec.nodeName,IMAGE_ID:.status.containerStatuses[0].imageID' \
  > results/k8s-baseline/pod-images.txt
kubectl -n inference-lab exec deployment/vllm -- nvidia-smi \
  --query-gpu=name,driver_version,memory.total,power.limit \
  --format=csv > results/k8s-baseline/gpu.csv
kubectl -n inference-lab exec deployment/vllm -- python3 -c \
  'import importlib.metadata; print(importlib.metadata.version("vllm"))' \
  > results/k8s-baseline/vllm-version.txt
```

Check those files for collection errors. Tags can be republished; the Pod image
ID records what actually ran. Record any cluster-specific changes and include
reviewed metadata through the benchmark's `--environment` option when comparing
runs. Repeat measurements on fixed hardware before drawing performance claims.

## Switch profiles and compare

All profiles use the same namespace, Deployment, Service, and PVC. Applying one
replaces the current profile; it does not create another GPU server. After saving
the baseline run and runtime evidence, switch to batched:

```bash
bash scripts/deploy.sh apply batched
```

Reconnect the port-forward after Pod replacement. Check health and models again,
then run the same benchmark with a separate label and output path:

```bash
PYTHONPATH=src python3 -m inference_lab.benchmark \
  --config configs/gpu.json --label k8s-batched-c4 \
  --output results/k8s-batched/run.json
```

Capture its manifest and runtime evidence using the commands above with
`batched` and `results/k8s-batched`. Repeat for `awq`, using `k8s-awq-c4` and
`results/k8s-awq`. Keep workload, seed, request count, output limit, client
concurrency, latency targets, and GPU hardware fixed. Do not compare results
from different node GPU types just because the manifests are identical.

After collecting all three real runs:

```bash
PYTHONPATH=src python3 -m inference_lab.report \
  results/k8s-baseline/run.json results/k8s-batched/run.json results/k8s-awq/run.json \
  --output results/k8s-comparison.html --csv results/k8s-comparison.csv
```

The server alias is always `inference-model`; the label does not prove which
checkpoint ran. Check the saved manifest, Pod profile label, and image identity.
These are experiment configurations, not measured improvements. Quantisation
can change answers or reduce speed; a separate quality check and repeated runs
are required before drawing conclusions.

For a Pending or restarting Pod, inspect `kubectl -n inference-lab get pods,pvc`,
`kubectl -n inference-lab describe pod POD_NAME`, and
`kubectl -n inference-lab logs deployment/vllm --tail=100`. Check GPU allocation,
storage binding, image pulls, and memory failures before changing the benchmark.

To release the GPU while preserving cached weights, run
`kubectl -n inference-lab scale deployment/vllm --replicas=0`. Reapplying restores
one replica. Storage may still incur charges. Deleting the namespace also
deletes its PVC; underlying data retention depends on the volume's
[reclaim policy](https://kubernetes.io/docs/concepts/storage/persistent-volumes/).

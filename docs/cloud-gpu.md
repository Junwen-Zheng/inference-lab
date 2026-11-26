# Rented GPU experiment

Preparation only: no instance has been rented and no GPU results exist. The next
execution session will collect all three Kubernetes profiles before releasing
the VM. Analysis and publication happen after billing stops.

## Resource and cost proposal

Use one Lambda on-demand **A10 24 GB** VM with Lambda Stack Ubuntu 22.04/24.04,
SSH and sudo access. Its published configuration is 30 vCPU, 226 GiB RAM and
1.3 TiB included SSD at **US$1.29/hour plus tax**, checked 2026-11-26 on the
[official instance table](https://lambda.com/instances). Four hours costs
US$5.16 before tax; proposed allowance is **US$8 total**. Select no separate
filesystem or other paid resources. Recheck the authenticated quote, quota and
regional stock before launch; do not substitute a more expensive machine.
Check `GET /api/v1/instance-types`: one GPU, x86_64, available region, and
whole-instance `price_cents_per_hour <=129`.
[Billing setup](https://docs.lambda.com/public-cloud/manage-billing/) may place
a separate temporary US$10 card preauthorization.

This is a full VM: Lambda documents
[single-node K3s on its instances](https://docs.lambda.com/education/large-language-models/k8s-ollama-llama-3-2/).
The A10 is Ampere; the pinned vLLM build requires
[Linux and compute capability >=7.5](https://docs.vllm.ai/en/v0.32.0/getting_started/installation/gpu/)
and its [AWQ support includes Ampere](https://docs.vllm.ai/en/v0.32.0/features/quantization/).
Require driver >=575.57.08 as a conservative native
[CUDA 12.9.1 requirement](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html).
The live image, driver, available disk and model fit still need checking.

Account/login, SSH access and approval of the named rental are needed before
launch. The assistant handles setup, execution, troubleshooting and evidence;
keep API keys and private SSH keys out of GitHub and chat.

## Session deadline

Record UTC launch time and a deadline no later than four hours after launch.
Immediately arm this guard in a separate terminal on the local controller,
with `LAMBDA_API_KEY` supplied through its environment:

```bash
mkdir -p results
python3 scripts/lambda-deadline.py --instance-id INSTANCE_UUID \
  --deadline YYYY-MM-DDTHH:MM:SSZ --output results/rental-deadline.jsonl
```

Use actual values. Confirm the `armed` event before setup. The guard checks only
that instance, waits, requests termination through the
[Lambda API](https://docs.lambda.com/api/cloud), and checks the resulting state.
It cannot provision resources. Its private log contains no API key. Keep the
controller awake, connected and the terminal running; supervise the session
with the provider console open. On macOS, prefix the command with `caffeinate -i`.
An interrupted guard must be restarted with the original deadline or the VM
terminated immediately. API/controller failure can defeat the guard: this is
an operational deadline, **not a provider-enforced financial cap**.

Stop starting new measurements after three hours. Export by 3h30, then terminate
early and confirm it. Four hours includes downloads, compilation, retries and
export. If setup fails or the plan will not fit, retain the failed evidence and
terminate; do not extend the rental without a new decision. Never leave the GPU
running until the next daily commit.

## Prepare the remote node

Run the following only on the approved disposable VM, using SSH. Keep ingress
restricted to SSH from the controller; do not publicly expose 8000 or 6443.
Clone the uploaded repository there and record the exact source revision:

```bash
git clone https://github.com/Junwen-Zheng/inference-lab.git
cd inference-lab
mkdir -p results/session
git rev-parse HEAD > results/session/source-revision.txt
python3 --version
nvidia-smi
command -v nvidia-container-runtime
df -h /var/lib
```

Require Python >=3.10, one A10, the driver above and at least 80 GiB free on the
filesystem used by `/var/lib/rancher`. This reserves room for image extraction,
both checkpoints, cache and results; it is a preparation threshold, not a
measured image-size guarantee. The model-cache PVC is 20 GiB. Use the preinstalled
Lambda driver/toolkit; if missing or too old, stop and investigate within the
session limit rather than replacing the stack blindly.

Install pinned K3s with the NVIDIA runtime as default, following
[K3s runtime detection](https://docs.k3s.io/advanced#nvidia-container-runtime).
The default also applies to the device plugin and existing serving manifests.
Local-path storage uses the included VM disk.

```bash
curl --fail --location \
  'https://raw.githubusercontent.com/k3s-io/k3s/v1.37.2+k3s1/install.sh' \
  -o /tmp/install-k3s.sh
sha256sum /tmp/install-k3s.sh > results/session/k3s-installer.sha256
sudo env INSTALL_K3S_VERSION='v1.37.2+k3s1' sh /tmp/install-k3s.sh \
  --default-runtime=nvidia --disable=traefik --disable=servicelb
mkdir -p "$HOME/.kube"
sudo install -m 600 -o "$(id -u)" -g "$(id -g)" \
  /etc/rancher/k3s/k3s.yaml "$HOME/.kube/config"
export KUBECONFIG="$HOME/.kube/config"
kubectl wait --for=condition=Ready node --all --timeout=180s
curl --fail --location \
  https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.20.2/deployments/static/nvidia-device-plugin.yml \
  -o results/session/device-plugin.yaml
kubectl apply -f results/session/device-plugin.yaml
kubectl -n kube-system rollout status daemonset/nvidia-device-plugin-daemonset --timeout=180s
python3 scripts/gpu-evidence.py --output-dir results/node-preflight
```

These commands install software and create cluster resources; they have only
been prepared, not executed here. The preflight makes read-only observations,
retains failed checks, and refuses an existing destination. Exit 0 means its
checks passed, not that inference works. Exit 1 means a failed observation/check;
2 means invalid setup. Check the saved evidence before deploying.

## Fixed measurement plan

`configs/gpu-experiment.json` uses 120 measured requests, 8 warmups, a 192-token
output ceiling, seed 42 and EOS allowed. `data/gpu-prompts.jsonl` contains 24
authored prompts of different lengths. It replaces the 24-request/32-token smoke
configuration for this experiment, while retaining that old configuration.

Each profile uses concurrency 1, 4 and 8, three repetitions and two seconds
between runs: 27 runs, 3,240 measured requests and 216 warmups across baseline,
batched FP16 and AWQ. Each run uses the same seeded prompt sequence. Output
length varies with EOS; retain server token usage and errors. The maximum
measured output budget is 622,080 tokens, not a prediction of completed tokens
or duration. The separate quality checks add 24 requests. This remains a small
closed-loop experiment, not a capacity claim or reliable p99 estimate.

For each profile, set `PROFILE` to `baseline`, then `batched`, then `awq`. Run
from the remote repository, keep the same VM and leave each profile unchanged
through its sweep:

```bash
PROFILE=baseline
bash scripts/deploy.sh render "$PROFILE" > "results/session/$PROFILE-manifest.yaml"
bash scripts/deploy.sh apply "$PROFILE"
python3 scripts/gpu-evidence.py --profile "$PROFILE" \
  --output-dir "results/$PROFILE-runtime"
```

Check the captured profile, pinned checkpoint/revision, actual image ID, GPU,
driver, vLLM 0.32.0, Pod readiness and bound PVC. Profile capture rejects changed
model/scheduler bindings and missing Kubernetes GPU requests; deliberate profile
changes require reviewing the collector's profile expectations too.
Before each sweep, check actual tokenizer
lengths inside the serving container, using the same raw completion prompts
and request tags as the benchmark:

```bash
kubectl -n inference-lab exec -i deployment/vllm -- python3 -c '
import hashlib, json, os, random, sys
from transformers import AutoTokenizer
raw = sys.stdin.buffer.read()
rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
rng = random.Random(42)
prompts = [rng.choice(rows)["prompt"] + f"\nRequest tag: {rng.getrandbits(64):016x}" for _ in range(120)]
prompts += [rows[i % len(rows)]["prompt"] + f"\nWarmup tag: {i}" for i in range(8)]
tokenizer = AutoTokenizer.from_pretrained(os.environ["MODEL_ID"], revision=os.environ["MODEL_REVISION"], trust_remote_code=False)
lengths = [len(tokenizer.encode(prompt, add_special_tokens=True)) for prompt in prompts]
print(json.dumps({"model": os.environ["MODEL_ID"], "revision": os.environ["MODEL_REVISION"], "workload_sha256": hashlib.sha256(raw).hexdigest(), "requests": len(lengths), "min_input_tokens": min(lengths), "max_input_tokens": max(lengths), "output_budget": 192, "context_limit": 2048}))
if max(lengths) + 192 > 2048:
    sys.exit("workload exceeds context limit")
' < data/gpu-prompts.jsonl > "results/$PROFILE-runtime/context-check.json"
```

Stop on a failed check. Start a remote loopback port-forward in a second SSH
terminal, reconnecting after every profile change:

```bash
kubectl -n inference-lab port-forward --address 127.0.0.1 svc/vllm 8000:8000
```

The benchmark client runs on that VM, through the same Service port-forward
path for all profiles. Record this path when interpreting latency.

```bash
curl --fail --max-time 10 http://127.0.0.1:8000/health
curl --fail --max-time 10 http://127.0.0.1:8000/v1/models > "results/$PROFILE-runtime/models.json"
curl --fail --max-time 10 http://127.0.0.1:8000/metrics > "results/$PROFILE-runtime/metrics-before.prom"
PYTHONPATH=src python3 -m inference_lab.sweep \
  --config configs/gpu-experiment.json --profile "$PROFILE" \
  --environment "results/$PROFILE-runtime/environment.json" \
  --concurrency 1 4 8 --repetitions 3 --cooldown-s 2 \
  --output-dir "results/$PROFILE-sweep"
curl --fail --max-time 10 http://127.0.0.1:8000/metrics > "results/$PROFILE-runtime/metrics-after.prom"
make quality PROFILE="$PROFILE" QUALITY_ENVIRONMENT="results/$PROFILE-runtime/environment.json"
kubectl -n inference-lab logs deployment/vllm > "results/$PROFILE-runtime/server.log"
```

Inspect exit status after each command. Sweep exit 1 retains measured failures;
exit 2 is a setup/collection error. Preserve either, including partial attempts.
Quality exit 1 retains mismatches/request errors; exit 2 is invalid setup.
A successful streamed request artifact is the inference proof; `/health` alone
is insufficient. Capture before/after metrics as raw evidence, not interchangeable
with client latency or a clean Prometheus rolling window. Keep failed attempt
directories and use new names for retries. The eight-case quality suite is a
smoke check; compare saved batched/AWQ answers after export using Day 16's tool.

## Export, verify, terminate

Before the deadline, stop measurement work and archive all attempts on the VM:

```bash
tar --exclude='__pycache__' --exclude='*.pyc' \
  -czf "$HOME/inference-lab-evidence.tar.gz" \
  results configs data deploy scripts src tests Makefile README.md pyproject.toml LICENSE
sha256sum "$HOME/inference-lab-evidence.tar.gz"
```

On the controller, `GPU_HOST` is the SSH destination, for example
`ubuntu@ACTUAL_IP`. Download and compare its checksum with the remote value:

```bash
scp "$GPU_HOST:inference-lab-evidence.tar.gz" ./inference-lab-evidence.tar.gz
shasum -a 256 inference-lab-evidence.tar.gz
tar -tzf inference-lab-evidence.tar.gz
```

Confirm the expected sweep JSON files, runtime evidence and quality artifacts
are present, along with the source/input snapshot, and the archive opens locally.
This preserves any troubleshooting changes made after the recorded commit.
Terminate that instance through the
provider console/API immediately; confirm `terminated` and the usage record.
The guard is a backup, not a reason to wait. Provider
[OS shutdown keeps billing active](https://docs.lambda.com/public-cloud/on-demand/creating-managing-instances/);
Kubernetes scale-to-zero or namespace deletion does not release the VM either.
Included root-disk data disappears at termination. Check that no separate
filesystem, replacement VM or other billable resource remains. Preserve the
local termination log and final bill privately; review artifacts before selecting
files for a later GitHub upload. No API keys, kubeconfig or SSH keys belong in
the evidence archive.

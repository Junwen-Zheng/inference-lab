# Local vLLM serving

The launcher defines three single-GPU experiments. It has been checked without
a GPU; a successful model load and completion remain pending.

## Requirements

Run the server on Linux with Docker, NVIDIA Container Toolkit, and a compatible
NVIDIA driver. The pinned CUDA build requires compute capability 7.5 or higher;
check the [vLLM GPU requirements](https://docs.vllm.ai/en/v0.32.0/getting_started/installation/gpu/)
and [Container Toolkit setup](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Leave GPU 0 free and check that its driver supports CUDA 12.9. Allow disk space
and outbound access for the image and model download. Memory fit depends on the
GPU and runtime; it has not been established by this project.

The CUDA launcher does not run on macOS. You can inspect its commands and run
the offline tests on your Mac. Use a prepared remote Linux GPU host for serving;
the launcher does not provision or rent one.

## Profiles

| Profile | Weights | Maximum sequences | Batched tokens |
| --- | --- | ---: | ---: |
| `baseline` | Qwen2.5-1.5B-Instruct, FP16 | 1 | 2048 |
| `batched` | Same FP16 checkpoint | 32 | 4096 |
| `awq` | Qwen2.5-1.5B-Instruct-AWQ, 4-bit | 32 | 4096 |

All use a 2048-token context (prompt plus output), FP16 activations, seed 42,
85% GPU memory utilisation, disabled prefix caching, and vLLM generation defaults.
The baseline-to-batched comparison changes two scheduler settings together.
Compare batched FP16 to AWQ with the same workload, concurrency, and hardware;
quantisation is not assumed to improve speed or preserve answer quality.

The image is `vllm/vllm-openai:v0.32.0-cu129`, listed in the
[official release](https://github.com/vllm-project/vllm/releases/tag/v0.32.0).
Model and tokenizer revisions are fixed in `deploy/docker/run.sh`:

- FP16: [`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/commit/989aa7980e4cf806f80c7fef2b1adb7bc71aa306).
- AWQ: [`3ecffa0ceb27851800f45519bab9c457a04405e1`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-AWQ/commit/3ecffa0ceb27851800f45519bab9c457a04405e1).

Change those pins in code when changing the experiment. Image tags can be
republished, so record the resolved image identity for each real run. Launch
flags follow the [versioned CLI reference](https://docs.vllm.ai/en/v0.32.0/cli/serve/).

## Start and check

From the repository root, inspect a command without Docker or a GPU:

```bash
make gpu-plan
bash deploy/docker/run.sh awq --dry-run
```

On the Linux GPU host, start one profile in the foreground:

```bash
bash deploy/docker/run.sh baseline
```

First startup downloads weights and compiles kernels. The server publishes
`127.0.0.1:8000`; it has no API authentication, so keep this loopback binding.
The default persistent cache is `.cache/model` in the repository. Set an
absolute `MODEL_CACHE_DIR` to store it elsewhere. Ctrl+C stops the container;
cached files remain. Stop it before starting another profile on the same port.

In a second terminal on that host:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
curl --fail http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"inference-model","prompt":"Explain continuous batching briefly.","max_tokens":32,"temperature":0}'
```

Wait for health to return 200 before benchmarking. The served alias is shared
by all profiles, so capture the actual checkpoint and launch arguments:

```bash
mkdir -p results/baseline
docker inspect --format '{"image_id":{{json .Image}},"image_tag":{{json .Config.Image}},"server_args":{{json .Config.Cmd}}}' \
  inference-lab-vllm > results/baseline/environment.json
docker image inspect vllm/vllm-openai:v0.32.0-cu129 \
  --format '{{json .RepoDigests}}' > results/baseline/image-digests.json
docker exec inference-lab-vllm nvidia-smi \
  --query-gpu=name,driver_version,memory.total,power.limit \
  --format=csv > results/baseline/gpu.csv
docker exec inference-lab-vllm python3 -c \
  'import importlib.metadata; print(importlib.metadata.version("vllm"))' \
  > results/baseline/vllm-version.txt
PYTHONPATH=src python3 -m inference_lab.benchmark \
  --config configs/gpu.json --environment results/baseline/environment.json \
  --output results/baseline/run.json
```

Inspect every capture for errors or empty fields. `.Image` is Docker's local
image ID; `RepoDigests` records registry digests when available. Keep the GPU
and runtime files beside the run. For later profiles, change the output folder
and benchmark `--label`, keeping measurement settings fixed. Repeat runs before
drawing conclusions; this short workload is a pipeline check. Kubernetes
deployment and a separate answer-quality diagnostic are later increments.

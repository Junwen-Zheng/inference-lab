#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: bash deploy/docker/run.sh {baseline|batched|awq} [--dry-run]"
  echo "Use --dry-run to print the Docker command without starting it."
}

if [[ $# -eq 1 && $1 == --help ]]; then usage; exit 0; fi
if [[ $# -lt 1 || $# -gt 2 ]]; then usage >&2; exit 2; fi
profile=$1
dry_run=false
if [[ $# -eq 2 ]]; then
  if [[ $2 != --dry-run ]]; then usage >&2; exit 2; fi
  dry_run=true
fi

model=Qwen/Qwen2.5-1.5B-Instruct
revision=989aa7980e4cf806f80c7fef2b1adb7bc71aa306
max_seqs=1
batch_tokens=2048
case "$profile" in
  baseline) ;;
  batched) max_seqs=32; batch_tokens=4096 ;;
  awq)
    model=Qwen/Qwen2.5-1.5B-Instruct-AWQ
    revision=3ecffa0ceb27851800f45519bab9c457a04405e1
    max_seqs=32
    batch_tokens=4096
    ;;
  *) echo "Unknown profile: $profile" >&2; usage >&2; exit 2 ;;
esac

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cache_dir=${MODEL_CACHE_DIR:-"$repo_dir/.cache/model"}
case "$cache_dir" in
  /*) ;;
  *) cache_dir="$PWD/$cache_dir" ;;
esac
if [[ $cache_dir == *,* ]]; then
  echo "MODEL_CACHE_DIR must not contain a comma." >&2
  exit 2
fi

if [[ $dry_run == false ]]; then
  if [[ $(uname -s) != Linux ]]; then
    echo "GPU serving requires a Linux host with an NVIDIA GPU." >&2
    exit 1
  fi
  if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is required." >&2
    exit 1
  fi
  mkdir -p -- "$cache_dir"
  cache_dir=$(cd -- "$cache_dir" && pwd -P)
  if [[ $cache_dir == *,* ]]; then
    echo "Resolved MODEL_CACHE_DIR must not contain a comma." >&2
    exit 2
  fi
fi

server_args=(
  --model "$model" --revision "$revision" --tokenizer-revision "$revision"
  --served-model-name inference-model --host 0.0.0.0 --port 8000
  --dtype float16 --max-model-len 2048 --max-num-seqs "$max_seqs"
  --max-num-batched-tokens "$batch_tokens" --gpu-memory-utilization 0.85
  --no-enable-prefix-caching --generation-config vllm --seed 42
)
if [[ $profile == awq ]]; then server_args+=(--quantization awq); fi
cmd=(
  docker run --rm --name inference-lab-vllm --gpus device=0
  --label "inference-lab.profile=$profile" --shm-size=2g
  --publish 127.0.0.1:8000:8000
  --mount "type=bind,source=$cache_dir,target=/cache"
  --env HF_HOME=/cache/huggingface --env VLLM_NO_USAGE_STATS=1
  vllm/vllm-openai:v0.32.0-cu129 "${server_args[@]}"
)
if [[ $dry_run == true ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
else
  exec "${cmd[@]}"
fi

#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'Usage: bash scripts/deploy.sh {render|dry-run|apply} baseline\n'
}

if [[ $# -eq 1 && $1 == --help ]]; then
    usage
    exit 0
fi
if [[ $# -ne 2 ]]; then
    usage >&2
    exit 2
fi

action=$1
profile=$2
case "$action" in
    render|dry-run|apply) ;;
    *) usage >&2; exit 2 ;;
esac
if [[ $profile != baseline ]]; then
    printf 'Unsupported profile: %s (expected baseline)\n' "$profile" >&2
    exit 2
fi
if ! command -v kubectl >/dev/null 2>&1; then
    printf 'kubectl is required.\n' >&2
    exit 127
fi

script_dir=${BASH_SOURCE[0]%/*}
if [[ $script_dir == "${BASH_SOURCE[0]}" ]]; then script_dir=.; fi
repo_dir=$(cd -- "$script_dir/.." && pwd)
overlay="$repo_dir/deploy/k8s/overlays/$profile"
case "$action" in
    render) kubectl kustomize "$overlay" ;;
    dry-run) kubectl apply --dry-run=client --validate=false -k "$overlay" ;;
    apply)
        context=$(kubectl config current-context)
        printf 'Applying %s to context %s\n' "$profile" "$context" >&2
        kubectl apply -k "$overlay"
        kubectl -n inference-lab rollout status deployment/vllm --timeout=40m
        ;;
esac

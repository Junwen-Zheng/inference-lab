#!/usr/bin/env python3
import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal
import io
import json
from pathlib import Path
import re
import shutil
import subprocess


NAMESPACE = "inference-lab"
GPU_QUERY = ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,compute_cap",
             "--format=csv,noheader,nounits"]
CONFIG_KEYS = ("MODEL_ID", "MODEL_REVISION", "MAX_NUM_SEQS", "MAX_NUM_BATCHED_TOKENS")
PROFILES = {
    "baseline": ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306", "1", "2048"),
    "batched": ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306", "32", "4096"),
    "awq": ("Qwen/Qwen2.5-1.5B-Instruct-AWQ", "3ecffa0ceb27851800f45519bab9c457a04405e1", "32", "4096"),
}


def serving_bindings(container, profile):
    args = container.get("args", [])
    if not isinstance(args, list) or any(not isinstance(arg, str) for arg in args):
        return False
    bindings = {"--model": "$(MODEL_ID)", "--revision": "$(MODEL_REVISION)",
                "--tokenizer-revision": "$(MODEL_REVISION)", "--max-num-seqs": "$(MAX_NUM_SEQS)",
                "--max-num-batched-tokens": "$(MAX_NUM_BATCHED_TOKENS)"}
    bindings["--quantization"] = "awq" if profile == "awq" else None
    for flag, expected in bindings.items():
        positions = [index for index, arg in enumerate(args) if arg == flag or arg.startswith(flag + "=")]
        if expected is None:
            if positions:
                return False
        elif (len(positions) != 1 or args[positions[0]] != flag or positions[0] + 1 >= len(args)
              or args[positions[0] + 1] != expected):
            return False
    refs = container.get("envFrom", [])
    resources = container.get("resources", {})
    try:
        allocated = all(quantity(resources.get(kind, {}).get("nvidia.com/gpu", "0")) == 1
                        for kind in ("requests", "limits"))
    except ValueError:
        allocated = False
    return (not container.get("command") and len(refs) == 1 and not refs[0].get("prefix")
            and bool(refs[0].get("configMapRef", {}).get("name"))
            and allocated
            and not any(item.get("name") in CONFIG_KEYS for item in container.get("env", [])))


def quantity(value):
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([a-zA-Z]*)", str(value))
    scales = {"": Decimal(1), "m": Decimal("0.001")}
    scales.update({unit: Decimal(1000) ** power for power, unit in enumerate("kMGTPE", 1)})
    scales.update({unit + "i": Decimal(1024) ** power
                   for power, unit in enumerate("KMGTPE", 1)})
    if not match or match[2] not in scales:
        raise ValueError(f"unsupported resource quantity: {value}")
    return Decimal(match[1]) * scales[match[2]]


def gpu_rows(text):
    rows = []
    for row in csv.reader(io.StringIO(text), skipinitialspace=True):
        if len(row) != 5:
            raise ValueError("nvidia-smi must return five fields per GPU")
        rows.append(dict(zip(("name", "uuid", "driver_version", "memory_mib", "compute_cap"),
                             (field.strip() for field in row))))
    return rows


def collect(output_dir, profile=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "captured_at": datetime.now(timezone.utc).isoformat(),
              "profile": profile, "commands": [], "checks": []}

    def check(name, passed, detail):
        report["checks"].append({"name": name, "passed": bool(passed), "detail": detail})

    def command(args, parse=None):
        record = {"argv": args}
        report["commands"].append(record)
        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=30, check=False)
            record["returncode"] = result.returncode
            if result.returncode:
                raise ValueError(f"command exited {result.returncode}")
            value = parse(result.stdout) if parse else result.stdout.strip()
            if parse is json.loads:
                if not isinstance(value, dict):
                    raise ValueError("Kubernetes response must be a JSON object")
                if "items" in value and (not isinstance(value["items"], list)
                                           or any(not isinstance(item, dict) for item in value["items"])):
                    raise ValueError("Kubernetes items must be a list of objects")
            # API objects can carry credentials in annotations or environment variables.
            if parse is not json.loads:
                record["output"] = value
            return value
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            record["error"] = str(exc)
            check("command", False, " ".join(args) + ": " + str(exc))
            return None

    gpu = command(GPU_QUERY, gpu_rows) or []
    report["host_gpus"] = gpu
    check("one_host_gpu", len(gpu) == 1, f"observed {len(gpu)}")
    if len(gpu) == 1:
        try:
            driver = tuple(int(part) for part in gpu[0]["driver_version"].split("."))
            driver = driver + (0,) * (3 - len(driver))
            compatible = driver >= (575, 57, 8) and Decimal(gpu[0]["compute_cap"]) >= Decimal("7.5")
        except (ValueError, ArithmeticError):
            compatible = False
        check("cuda_12_9_host", compatible, "requires driver >=575.57.08 and compute capability >=7.5")
    try:
        disk = shutil.disk_usage("/var/lib/rancher/k3s")
        report["disk"] = {"path": "/var/lib/rancher/k3s", "total_bytes": disk.total,
                          "free_bytes": disk.free}
        check("disk_space", disk.free >= 80 * 1024 ** 3, "requires at least 80 GiB free for images and cache")
    except OSError as exc:
        check("disk_space", False, str(exc))
    report["kubectl_version"] = command(["kubectl", "version", "-o", "json"], json.loads)
    nodes = command(["kubectl", "get", "nodes", "-o", "json"], json.loads) or {}
    report["nodes"] = [{"name": node.get("metadata", {}).get("name"),
                        "os": node.get("metadata", {}).get("labels", {}).get("kubernetes.io/os"),
                        "allocatable": node.get("status", {}).get("allocatable", {}),
                        "conditions": node.get("status", {}).get("conditions", [])}
                       for node in nodes.get("items", [])]
    check("one_node", len(report["nodes"]) == 1, f"observed {len(report['nodes'])}")
    if len(report["nodes"]) == 1:
        node = report["nodes"][0]
        check("ready_linux_node", node["os"] == "linux" and any(
            item.get("type") == "Ready" and item.get("status") == "True"
            for item in node["conditions"]), str(node["name"]))
        try:
            alloc = node["allocatable"]
            enough = (quantity(alloc.get("nvidia.com/gpu", "0")) == 1
                      and quantity(alloc.get("cpu", "0")) >= 4
                      and quantity(alloc.get("memory", "0")) >= 8 * 1024 ** 3)
        except ValueError:
            enough = False
        check("node_resources", enough, "requires one GPU, 4 allocatable CPU and 8 GiB RAM")
    storage = command(["kubectl", "get", "storageclasses", "-o", "json"], json.loads) or {}
    report["storageclasses"] = [{"name": item.get("metadata", {}).get("name"),
                                  "provisioner": item.get("provisioner"),
                                  "default": item.get("metadata", {}).get("annotations", {}).get(
                                      "storageclass.kubernetes.io/is-default-class") == "true",
                                  "reclaim_policy": item.get("reclaimPolicy")}
                                 for item in storage.get("items", [])]
    defaults = [item for item in report["storageclasses"] if item["default"]]
    check("local_path_storage", len(defaults) == 1 and defaults[0]["provisioner"] == "rancher.io/local-path",
          "requires exactly one default local-path storage class")
    if profile:
        capture_profile(report, profile, gpu, command, check, output_dir)
    report["ready"] = all(item["passed"] for item in report["checks"])
    (output_dir / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def capture_profile(report, profile, host_gpus, command, check, output_dir):
    prefix = ["kubectl", "-n", NAMESPACE]
    deployment = command(prefix + ["get", "deployment/vllm", "-o", "json"], json.loads) or {}
    template = deployment.get("spec", {}).get("template", {})
    containers = template.get("spec", {}).get("containers", [])
    container = next((item for item in containers if item.get("name") == "vllm"), {})
    allowed_args = {"--model", "--revision", "--tokenizer-revision", "--served-model-name", "--host",
                    "--port", "--dtype", "--max-model-len", "--max-num-seqs", "--max-num-batched-tokens",
                    "--gpu-memory-utilization", "--no-enable-prefix-caching", "--generation-config",
                    "--seed", "--quantization"}
    args = container.get("args", [])
    safe_args = []
    for index, arg in enumerate(args):
        if arg in allowed_args:
            safe_args.append(arg)
            if index + 1 < len(args) and not args[index + 1].startswith("--"):
                safe_args.append(args[index + 1])
    report["deployment"] = {"name": deployment.get("metadata", {}).get("name"),
                            "profile": template.get("metadata", {}).get("labels", {}).get("inference-lab/profile"),
                            "image": container.get("image"), "args": safe_args,
                            "runtime_class": template.get("spec", {}).get("runtimeClassName"),
                            "resources": container.get("resources", {})}
    check("deployment_profile", report["deployment"]["profile"] == profile, profile)
    status = deployment.get("status", {})
    check("deployment_ready", status.get("observedGeneration") == deployment.get("metadata", {}).get("generation")
          and status.get("updatedReplicas") == 1 and status.get("availableReplicas") == 1,
          "requires the observed single-replica rollout")
    refs = [item.get("configMapRef", {}).get("name") for item in container.get("envFrom", [])]
    config = {}
    if len(refs) == 1 and refs[0] and refs[0].startswith("vllm-config-"):
        value = command(prefix + ["get", "configmap/" + refs[0], "-o", "json"], json.loads) or {}
        config = {key: value.get("data", {}).get(key) for key in CONFIG_KEYS}
    report["serving_config"] = config
    check("serving_config", tuple(config.get(key) for key in CONFIG_KEYS) == PROFILES[profile],
          "requires pinned profile checkpoint, revision and scheduler budgets")
    check("deployment_bindings", serving_bindings(container, profile),
          "requires unique serving argument bindings, no environment override and one requested/limited GPU")
    pods = command(prefix + ["get", "pods", "-l", "app.kubernetes.io/name=vllm", "-o", "json"], json.loads) or {}
    report["pods"] = [{"name": item.get("metadata", {}).get("name"),
                       "profile": item.get("metadata", {}).get("labels", {}).get("inference-lab/profile"),
                       "node": item.get("spec", {}).get("nodeName"),
                       "phase": item.get("status", {}).get("phase"),
                       "configmaps": [ref.get("configMapRef", {}).get("name")
                                      for candidate in item.get("spec", {}).get("containers", [])
                                      if candidate.get("name") == "vllm" for ref in candidate.get("envFrom", [])],
                       "containers": [{key: status.get(key) for key in
                                       ("name", "image", "imageID", "ready", "restartCount")}
                                      for status in item.get("status", {}).get("containerStatuses", [])]}
                      for item in pods.get("items", [])]
    ready = [item for item in report["pods"] if item["profile"] == profile and item["phase"] == "Running"
             and any(status.get("name") == "vllm" and status.get("ready")
                     for status in item["containers"])]
    check("ready_profile_pod", len(ready) == 1 and len(report["pods"]) == 1, "requires exactly one Ready profile Pod")
    pvc = command(prefix + ["get", "pvc/model-cache", "-o", "json"], json.loads) or {}
    report["cache_pvc"] = {"name": pvc.get("metadata", {}).get("name"),
                           "storage_class": pvc.get("spec", {}).get("storageClassName"),
                           "phase": pvc.get("status", {}).get("phase"),
                           "capacity": pvc.get("status", {}).get("capacity")}
    check("cache_pvc_bound", report["cache_pvc"]["phase"] == "Bound", "model-cache PVC")
    if len(ready) != 1:
        return
    pod = ready[0]
    raw_pod = next(item for item in pods.get("items", []) if item.get("metadata", {}).get("name") == pod["name"])
    pod_container = next((item for item in raw_pod.get("spec", {}).get("containers", [])
                          if item.get("name") == "vllm"), {})
    check("pod_serving_spec", serving_bindings(pod_container, profile)
          and pod_container.get("args") == container.get("args")
          and pod_container.get("image") == container.get("image")
          and pod_container.get("envFrom") == container.get("envFrom")
          and pod_container.get("resources") == container.get("resources"),
          "Ready Pod serving arguments, image, configuration sources and resources must match the Deployment")
    report["pod_resources"] = pod_container.get("resources", {})
    check("pod_config", pod["configmaps"] == refs, "Ready Pod must use the Deployment ConfigMap")
    check("pod_node", len(report["nodes"]) == 1 and pod["node"] == report["nodes"][0]["name"],
          "Ready Pod must run on the inspected node")
    exec_prefix = prefix + ["exec", pod["name"], "-c", "vllm", "--"]
    pod_gpus = command(exec_prefix + GPU_QUERY, gpu_rows) or []
    report["pod_gpus"] = pod_gpus
    version = command(exec_prefix + ["python3", "-c",
                      "import importlib.metadata; print(importlib.metadata.version('vllm'))"])
    report["vllm_version"] = version
    check("pod_gpu", len(pod_gpus) == 1 and len(host_gpus) == 1
          and pod_gpus[0]["uuid"] == host_gpus[0]["uuid"], "Pod GPU must match the single host GPU")
    statuses = [item for item in pod["containers"] if item.get("name") == "vllm"]
    image_id = statuses[0].get("imageID") if len(statuses) == 1 else None
    check("runtime_identity", bool(version and image_id and "sha256:" in image_id), "requires vLLM version and resolved image digest")
    if all(item["passed"] for item in report["checks"]):
        environment = {"deployment_profile": profile, "model_id": config["MODEL_ID"],
                       "model_revision": config["MODEL_REVISION"], "container_image_id": image_id,
                       "gpu_model": pod_gpus[0]["name"], "driver_version": pod_gpus[0]["driver_version"]}
        (output_dir / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Read-only checks and selected runtime evidence for one K3s GPU host.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--profile", choices=("baseline", "batched", "awq"))
    args = parser.parse_args()
    try:
        report = collect(args.output_dir, args.profile)
    except OSError as exc:
        parser.exit(2, f"{exc}\n")
    print(args.output_dir / "preflight.json")
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

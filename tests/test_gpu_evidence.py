import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/gpu-evidence.py"
SPEC = importlib.util.spec_from_file_location("gpu_evidence", SCRIPT)
EVIDENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVIDENCE)
GIB = 1024 ** 3


class GpuEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "new evidence"
        self.gpu = "NVIDIA A10, GPU-test, 575.57.08, 23028, 8.6\n"
        self.disk = SimpleNamespace(total=1000 * GIB, free=900 * GIB)
        self.resources = {
            "nodes": {"items": [{"metadata": {"name": "gpu-node", "labels": {"kubernetes.io/os": "linux"},
                                              "annotations": {"private": "SECRET_TOKEN"}},
                                  "status": {"allocatable": {"cpu": "30000m", "memory": "226Gi", "nvidia.com/gpu": "1"},
                                             "conditions": [{"type": "Ready", "status": "True"}]}}]},
            "storageclasses": {"items": [{"metadata": {"name": "local-path", "annotations": {
                "storageclass.kubernetes.io/is-default-class": "true"}},
                "provisioner": "rancher.io/local-path", "reclaimPolicy": "Delete"}]},
            "deployment/vllm": {"metadata": {"name": "vllm", "generation": 2},
                                "status": {"observedGeneration": 2, "updatedReplicas": 1, "availableReplicas": 1},
                                "spec": {"template": {"metadata": {"labels": {"inference-lab/profile": "batched"}},
                                                       "spec": {"containers": [{"name": "vllm", "image": "vllm/vllm-openai:tag",
                                                                                "args": ["--model", "$(MODEL_ID)", "--revision", "$(MODEL_REVISION)",
                                                                                         "--tokenizer-revision", "$(MODEL_REVISION)",
                                                                                         "--max-num-seqs", "$(MAX_NUM_SEQS)",
                                                                                         "--max-num-batched-tokens", "$(MAX_NUM_BATCHED_TOKENS)",
                                                                                         "--api-key", "SECRET_TOKEN"],
                                                                                "env": [{"name": "HF_TOKEN", "value": "SECRET_TOKEN"}],
                                                                                "resources": {"requests": {"cpu": "4", "memory": "8Gi", "nvidia.com/gpu": "1"},
                                                                                              "limits": {"cpu": "8", "memory": "16Gi", "nvidia.com/gpu": "1"}},
                                                                                "envFrom": [{"configMapRef": {"name": "vllm-config-abc"}}]}]}}}},
            "configmap/vllm-config-abc": {"data": {"MODEL_ID": "Qwen/Qwen2.5-1.5B-Instruct",
                                                   "MODEL_REVISION": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
                                                   "MAX_NUM_SEQS": "32", "MAX_NUM_BATCHED_TOKENS": "4096", "PRIVATE": "SECRET_TOKEN"}},
            "pods": {"items": [{"metadata": {"name": "vllm-123", "labels": {"inference-lab/profile": "batched"}},
                                 "spec": {"nodeName": "gpu-node", "containers": [{"name": "vllm", "envFrom": [
                                     {"configMapRef": {"name": "vllm-config-abc"}}]}]},
                                 "status": {"phase": "Running", "containerStatuses": [
                                     {"name": "vllm", "image": "vllm/vllm-openai:tag", "imageID": "docker-pullable://vllm@sha256:abc",
                                      "ready": True, "restartCount": 0, "state": {"terminated": {"message": "SECRET_TOKEN"}}}]}}]},
            "pvc/model-cache": {"metadata": {"name": "model-cache"}, "spec": {"storageClassName": "local-path"},
                                "status": {"phase": "Bound", "capacity": {"storage": "20Gi"}}},
        }
        self.deployment_container = self.resources["deployment/vllm"]["spec"]["template"]["spec"]["containers"][0]
        self.resources["pods"]["items"][0]["spec"]["containers"] = [copy.deepcopy(self.deployment_container)]
        self.pod_container = self.resources["pods"]["items"][0]["spec"]["containers"][0]

    def command(self, argv, **kwargs):
        self.assertEqual(kwargs, {"capture_output": True, "text": True, "timeout": 30, "check": False})
        if "nvidia-smi" in argv:
            value = self.gpu
        elif argv[:2] == ["kubectl", "version"]:
            value = json.dumps({"serverVersion": {"gitVersion": "v1.35.0+k3s1"}})
        elif "python3" in argv:
            value = "0.32.0\n"
        else:
            value = json.dumps(self.resources[argv[argv.index("get") + 1]])
        return subprocess.CompletedProcess(argv, 0, value, "")

    def collect(self, profile=None, command=None):
        with patch.object(EVIDENCE.subprocess, "run", side_effect=command or self.command), \
                patch.object(EVIDENCE.shutil, "disk_usage", return_value=self.disk):
            report = EVIDENCE.collect(self.output, profile)
        self.assertEqual(json.loads((self.output / "preflight.json").read_text()), report)
        return report

    def test_ready_host_handles_cpu_millicores_and_binary_memory(self):
        report = self.collect()
        self.assertTrue(report["ready"])
        self.assertEqual(report["host_gpus"][0]["compute_cap"], "8.6")
        self.assertEqual(report["disk"]["free_bytes"], 900 * GIB)
        self.assertFalse((self.output / "environment.json").exists())
        self.assertNotIn("SECRET_TOKEN", (self.output / "preflight.json").read_text())

    def test_driver_and_compute_capability_reject_incompatible_gpu(self):
        for driver, capability in (("575.57.07", "8.6"), ("570.200.0", "8.6"), ("580.1.0", "7.0"), ("unknown", "8.6")):
            with self.subTest(driver=driver, capability=capability):
                self.output = self.output.parent / (driver + capability)
                self.gpu = f"NVIDIA A10, GPU-test, {driver}, 23028, {capability}\n"
                self.assertFalse(self.collect()["ready"])

    def test_missing_host_gpu_and_missing_node_gpu_are_recorded(self):
        self.gpu = ""
        self.resources["nodes"]["items"][0]["status"]["allocatable"].pop("nvidia.com/gpu")
        report = self.collect()
        failed = {check["name"] for check in report["checks"] if not check["passed"]}
        self.assertIn("one_host_gpu", failed)
        self.assertIn("node_resources", failed)

    def test_not_ready_or_multiple_nodes_fail(self):
        node = self.resources["nodes"]["items"][0]
        node["status"]["conditions"][0]["status"] = "False"
        self.assertFalse(self.collect()["ready"])
        self.output = self.output.parent / "second"
        self.resources["nodes"]["items"].append(copy.deepcopy(node))
        self.assertFalse(self.collect()["ready"])

    def test_small_disk_fails_and_preserves_evidence(self):
        self.disk.free = 79 * GIB
        self.assertFalse(self.collect()["ready"])

    def test_insufficient_or_unsupported_quantities_fail(self):
        for cpu, memory in (("3999m", "8Gi"), ("4", "7Gi"), ("4garbage", "8Gi"), ("4", "8000000000")):
            with self.subTest(cpu=cpu, memory=memory):
                self.output = self.output.parent / (cpu + memory)
                self.resources["nodes"]["items"][0]["status"]["allocatable"].update(cpu=cpu, memory=memory)
                self.assertFalse(self.collect()["ready"])

    def test_no_default_or_duplicate_default_storage_fails(self):
        self.resources["storageclasses"]["items"][0]["metadata"]["annotations"] = {}
        self.assertFalse(self.collect()["ready"])
        self.output = self.output.parent / "second"
        item = self.resources["storageclasses"]["items"][0]
        item["metadata"]["annotations"] = {"storageclass.kubernetes.io/is-default-class": "true"}
        self.resources["storageclasses"]["items"].append(copy.deepcopy(item))
        self.assertFalse(self.collect()["ready"])

    def test_failed_commands_and_timeout_write_failure_without_stderr(self):
        for problem in (subprocess.TimeoutExpired(["kubectl"], 30), FileNotFoundError("missing command"),
                        subprocess.CompletedProcess([], 1, "", "SECRET_TOKEN")):
            with self.subTest(problem=type(problem).__name__):
                self.output = self.output.parent / type(problem).__name__
                def failed(argv, **kwargs):
                    if isinstance(problem, BaseException):
                        raise problem
                    return problem
                self.assertFalse(self.collect(command=failed)["ready"])
                self.assertNotIn("SECRET_TOKEN", (self.output / "preflight.json").read_text())

    def test_existing_directory_is_not_modified_and_no_commands_run(self):
        self.output.mkdir()
        marker = self.output / "keep"
        marker.write_text("original")
        with patch.object(EVIDENCE.subprocess, "run") as run:
            with self.assertRaises(FileExistsError):
                EVIDENCE.collect(self.output)
            run.assert_not_called()
        self.assertEqual(marker.read_text(), "original")

    def test_malformed_api_response_is_saved_as_a_failure(self):
        for index, response in enumerate(("[]", '"string"', '{"items": "wrong"}', '{"items": [1]}', "bad JSON")):
            with self.subTest(response=response):
                self.output = self.output.parent / f"malformed-{index}"
                def malformed(argv, **kwargs):
                    if "nodes" in argv:
                        return subprocess.CompletedProcess(argv, 0, response, "")
                    return self.command(argv, **kwargs)
                report = self.collect(command=malformed)
                self.assertFalse(report["ready"])
                self.assertTrue(any("error" in command for command in report["commands"]))

    def test_profile_capture_writes_actual_environment_without_credentials(self):
        report = self.collect("batched")
        self.assertTrue(report["ready"])
        environment = json.loads((self.output / "environment.json").read_text())
        self.assertEqual(environment["deployment_profile"], "batched")
        self.assertEqual(environment["model_revision"], "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
        self.assertEqual(environment["container_image_id"], "docker-pullable://vllm@sha256:abc")
        self.assertNotIn("SECRET_TOKEN", (self.output / "preflight.json").read_text())

    def test_stale_rollout_or_different_profile_does_not_write_environment(self):
        self.resources["deployment/vllm"]["status"]["observedGeneration"] = 1
        report = self.collect("baseline")
        self.assertFalse(report["ready"])
        self.assertFalse((self.output / "environment.json").exists())

    def test_pod_using_old_config_does_not_write_environment(self):
        self.resources["pods"]["items"][0]["spec"]["containers"][0]["envFrom"][0]["configMapRef"]["name"] = "old-config"
        self.assertFalse(self.collect("batched")["ready"])
        self.assertFalse((self.output / "environment.json").exists())

    def test_unbound_cache_or_unresolved_image_does_not_write_environment(self):
        self.resources["pvc/model-cache"]["status"]["phase"] = "Pending"
        self.resources["pods"]["items"][0]["status"]["containerStatuses"][0]["imageID"] = ""
        self.assertFalse(self.collect("batched")["ready"])
        self.assertFalse((self.output / "environment.json").exists())

    def test_literal_model_argument_does_not_claim_configmap_model(self):
        for container in (self.deployment_container, self.pod_container):
            container["args"][1] = "different/model"
        self.assertFalse(self.collect("batched")["ready"])
        self.assertFalse((self.output / "environment.json").exists())

    def test_explicit_environment_override_does_not_claim_configmap_values(self):
        for index, container in enumerate((self.deployment_container, self.pod_container)):
            with self.subTest(container=index):
                self.output = self.output.parent / f"environment-override-{index}"
                container["env"].append({"name": "MODEL_REVISION", "value": "other-revision"})
                self.assertFalse(self.collect("batched")["ready"])
                self.assertFalse((self.output / "environment.json").exists())
                container["env"].pop()

    def test_duplicate_argument_overrides_are_rejected(self):
        original = copy.deepcopy(self.deployment_container["args"])
        for index, extra in enumerate((["--model", "other/model"], ["--revision=other-revision"],
                                       ["--max-num-seqs=1"], ["--quantization", "awq"])):
            with self.subTest(extra=extra):
                self.output = self.output.parent / f"duplicate-{index}"
                for container in (self.deployment_container, self.pod_container):
                    container["args"] = original + extra
                self.assertFalse(self.collect("batched")["ready"])
                self.assertFalse((self.output / "environment.json").exists())

    def test_ready_pod_argument_drift_is_rejected(self):
        self.pod_container["args"].extend(["--seed", "123"])
        self.assertFalse(self.collect("batched")["ready"])
        self.assertFalse((self.output / "environment.json").exists())

    def test_changed_checkpoint_or_scheduler_budget_is_rejected(self):
        data = self.resources["configmap/vllm-config-abc"]["data"]
        original = data.copy()
        for key in EVIDENCE.CONFIG_KEYS:
            with self.subTest(key=key):
                self.output = self.output.parent / key
                data.update(original)
                data[key] = "changed"
                self.assertFalse(self.collect("batched")["ready"])
                self.assertFalse((self.output / "environment.json").exists())

    def test_prefixed_configmap_or_command_override_is_rejected(self):
        self.pod_container["envFrom"][0]["prefix"] = "IGNORED_"
        self.assertFalse(self.collect("batched")["ready"])
        self.assertFalse((self.output / "environment.json").exists())
        self.output = self.output.parent / "command-override"
        self.pod_container["envFrom"][0].pop("prefix")
        self.pod_container["command"] = ["other-server"]
        self.assertFalse(self.collect("batched")["ready"])
        self.assertFalse((self.output / "environment.json").exists())

    def test_baseline_and_awq_canonical_bindings_are_accepted(self):
        for profile in ("baseline", "awq"):
            with self.subTest(profile=profile):
                self.output = self.output.parent / profile
                self.resources["deployment/vllm"]["spec"]["template"]["metadata"]["labels"]["inference-lab/profile"] = profile
                self.resources["pods"]["items"][0]["metadata"]["labels"]["inference-lab/profile"] = profile
                self.resources["configmap/vllm-config-abc"]["data"].update(zip(EVIDENCE.CONFIG_KEYS, EVIDENCE.PROFILES[profile]))
                if profile == "awq":
                    for container in (self.deployment_container, self.pod_container):
                        container["args"].extend(["--quantization", "awq"])
                self.assertTrue(self.collect(profile)["ready"])
                self.assertEqual(json.loads((self.output / "environment.json").read_text())["deployment_profile"], profile)

    def test_visible_host_gpu_without_kubernetes_allocation_is_rejected(self):
        for index, kind in enumerate(("requests", "limits")):
            with self.subTest(kind=kind):
                self.output = self.output.parent / f"gpu-allocation-{index}"
                for container in (self.deployment_container, self.pod_container):
                    container["resources"][kind].pop("nvidia.com/gpu", None)
                self.assertFalse(self.collect("batched")["ready"])
                self.assertFalse((self.output / "environment.json").exists())
                for container in (self.deployment_container, self.pod_container):
                    container["resources"][kind]["nvidia.com/gpu"] = "1"

    def test_ready_pod_resource_drift_is_rejected(self):
        self.pod_container["resources"]["requests"]["cpu"] = "2"
        self.assertFalse(self.collect("batched")["ready"])
        self.assertFalse((self.output / "environment.json").exists())


if __name__ == "__main__":
    unittest.main()

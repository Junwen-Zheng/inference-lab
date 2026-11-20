import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "scripts/deploy.sh"


class DeployTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.bin = self.folder / "bin"
        self.bin.mkdir()
        self.record = self.folder / "kubectl.jsonl"
        self.stub = self.bin / "kubectl"
        helper = self.folder / "kubectl_stub.py"
        helper.write_text(
            "import json, os, sys\n"
            "args = sys.argv[1:]\n"
            "with open(os.environ['DEPLOY_RECORD'], 'a') as log:\n"
            "    log.write(json.dumps(args) + '\\n')\n"
            "if args == ['config', 'current-context']:\n"
            "    print('test-gpu-context')\n"
            "    sys.exit(int(os.environ.get('CONTEXT_EXIT', '0')))\n"
            "if args and args[0] == 'apply':\n"
            "    sys.exit(int(os.environ.get('APPLY_EXIT', '0')))\n"
            "if args and args[0] == 'kustomize':\n"
            "    print('kind: Deployment')\n"
        )
        self.stub.write_text(
            f"#!/bin/bash\nexec {shlex.quote(sys.executable)} {shlex.quote(str(helper))} \"$@\"\n"
        )
        self.stub.chmod(0o755)
        self.env = os.environ.copy()
        self.env.update(PATH=str(self.bin), DEPLOY_RECORD=str(self.record))
        self.env.pop("CONTEXT_EXIT", None)
        self.env.pop("APPLY_EXIT", None)

    def run_deploy(self, *args, script=DEPLOY):
        return subprocess.run(
            ["/bin/bash", str(script), *args],
            cwd=self.folder,
            env=self.env,
            text=True,
            capture_output=True,
            timeout=5,
        )

    def calls(self):
        if not self.record.exists():
            return []
        return [json.loads(line) for line in self.record.read_text().splitlines()]

    def overlay(self, repo=ROOT):
        return str(repo / "deploy/k8s/overlays/baseline")

    def test_render_outputs_manifests_without_apply(self):
        result = self.run_deploy("render", "baseline")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "kind: Deployment")
        self.assertEqual(self.calls(), [["kustomize", self.overlay()]])

    def test_dry_run_uses_client_mode_without_schema_download(self):
        result = self.run_deploy("dry-run", "baseline")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.calls(),
            [["apply", "--dry-run=client", "--validate=false", "-k", self.overlay()]],
        )

    def test_apply_shows_context_and_waits_for_rollout(self):
        result = self.run_deploy("apply", "baseline")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("test-gpu-context", result.stderr)
        self.assertEqual(
            self.calls(),
            [
                ["config", "current-context"],
                ["apply", "-k", self.overlay()],
                ["-n", "inference-lab", "rollout", "status", "deployment/vllm", "--timeout=40m"],
            ],
        )

    def test_help_and_argument_validation_need_no_kubectl(self):
        self.stub.unlink()
        help_result = self.run_deploy("--help")
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("baseline", help_result.stdout + help_result.stderr)
        for args in (
            (),
            ("render",),
            ("baseline",),
            ("unknown", "baseline"),
            ("render", "awq"),
            ("render", "baseline", "extra"),
            ("--help", "baseline"),
        ):
            with self.subTest(args=args):
                result = self.run_deploy(*args)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertTrue(result.stderr.strip())
        self.assertEqual(self.calls(), [])

    def test_repository_path_with_spaces_is_one_argument_from_other_directory(self):
        repo = self.folder / "repository with spaces"
        copied = repo / "scripts/deploy.sh"
        copied.parent.mkdir(parents=True)
        copied.write_bytes(DEPLOY.read_bytes())
        (repo / "deploy/k8s/overlays/baseline").mkdir(parents=True)
        result = self.run_deploy("render", "baseline", script=copied)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls(), [["kustomize", self.overlay(repo)]])

    def test_missing_kubectl_has_clear_error(self):
        self.stub.unlink()
        result = self.run_deploy("render", "baseline")
        self.assertEqual(result.returncode, 127, result.stderr)
        self.assertIn("kubectl", result.stderr)
        self.assertEqual(self.calls(), [])

    def test_context_failure_prevents_apply(self):
        self.env["CONTEXT_EXIT"] = "31"
        result = self.run_deploy("apply", "baseline")
        self.assertEqual(result.returncode, 31, result.stderr)
        self.assertEqual(self.calls(), [["config", "current-context"]])

    def test_apply_failure_prevents_rollout(self):
        self.env["APPLY_EXIT"] = "37"
        result = self.run_deploy("apply", "baseline")
        self.assertEqual(result.returncode, 37, result.stderr)
        self.assertEqual(
            self.calls(), [["config", "current-context"], ["apply", "-k", self.overlay()]]
        )


if __name__ == "__main__":
    unittest.main()

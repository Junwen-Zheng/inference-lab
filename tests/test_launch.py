import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "deploy/docker/run.sh"
IMAGE = "vllm/vllm-openai:v0.32.0-cu129"
FP16_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
AWQ_REVISION = "3ecffa0ceb27851800f45519bab9c457a04405e1"


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.cache = self.folder / "model cache"
        self.record = self.folder / "docker-args"
        self.bin = self.folder / "bin"
        self.bin.mkdir()
        self.env = os.environ.copy()
        self.env.update(
            MODEL_CACHE_DIR=str(self.cache),
            LAUNCH_RECORD=str(self.record),
            PATH=f"{self.bin}:{os.defpath}",
        )
        self.stub("uname", "printf 'Darwin\\n'\n")
        self.stub("docker", "printf '%s\\0' \"$@\" > \"$LAUNCH_RECORD\"\nexit 17\n")

    def stub(self, name, body):
        path = self.bin / name
        path.write_text("#!/bin/bash\n" + body)
        path.chmod(0o755)

    def launch(self, *args, launcher=LAUNCHER):
        return subprocess.run(
            ["/bin/bash", str(launcher), *args],
            cwd=ROOT,
            env=self.env,
            text=True,
            capture_output=True,
            timeout=5,
        )

    def dry_run(self, profile):
        result = self.launch(profile, "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        return shlex.split(result.stdout)

    def flag(self, argv, name):
        return argv[argv.index(name) + 1]

    def assert_shared_settings(self, argv):
        self.assertEqual(argv[:3], ["docker", "run", "--rm"])
        self.assertEqual(argv.count(IMAGE), 1)
        self.assertEqual(self.flag(argv, "--publish"), "127.0.0.1:8000:8000")
        self.assertEqual(self.flag(argv, "--gpus"), "device=0")
        for name, value in {
            "--served-model-name": "inference-model",
            "--host": "0.0.0.0",
            "--port": "8000",
            "--dtype": "float16",
            "--max-model-len": "2048",
            "--gpu-memory-utilization": "0.85",
            "--generation-config": "vllm",
            "--seed": "42",
        }.items():
            self.assertEqual(self.flag(argv, name), value)
        self.assertIn("--no-enable-prefix-caching", argv)
        self.assertIn("HF_HOME=/cache/huggingface", argv)
        self.assertNotIn("--privileged", argv)

    def test_baseline_pins_model_and_single_sequence(self):
        argv = self.dry_run("baseline")
        self.assert_shared_settings(argv)
        self.assertEqual(self.flag(argv, "--model"), "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(self.flag(argv, "--revision"), FP16_REVISION)
        self.assertEqual(self.flag(argv, "--tokenizer-revision"), FP16_REVISION)
        self.assertEqual(self.flag(argv, "--max-num-seqs"), "1")
        self.assertEqual(self.flag(argv, "--max-num-batched-tokens"), "2048")
        self.assertNotIn("--quantization", argv)

    def test_batched_changes_capacity_without_changing_model(self):
        baseline = self.dry_run("baseline")
        batched = self.dry_run("batched")
        self.assert_shared_settings(batched)
        self.assertEqual(self.flag(batched, "--max-num-seqs"), "32")
        self.assertEqual(self.flag(batched, "--max-num-batched-tokens"), "4096")
        for flag in ("--max-num-seqs", "--max-num-batched-tokens", "--label"):
            baseline[baseline.index(flag) + 1] = batched[batched.index(flag) + 1]
        self.assertEqual(baseline, batched)

    def test_awq_uses_matching_quantized_revision(self):
        argv = self.dry_run("awq")
        self.assert_shared_settings(argv)
        self.assertEqual(self.flag(argv, "--model"), "Qwen/Qwen2.5-1.5B-Instruct-AWQ")
        self.assertEqual(self.flag(argv, "--revision"), AWQ_REVISION)
        self.assertEqual(self.flag(argv, "--tokenizer-revision"), AWQ_REVISION)
        self.assertEqual(self.flag(argv, "--quantization"), "awq")
        self.assertEqual(self.flag(argv, "--max-num-seqs"), "32")
        self.assertEqual(self.flag(argv, "--max-num-batched-tokens"), "4096")

    def test_dry_run_on_mac_preserves_spaced_cache_without_side_effects(self):
        (self.bin / "docker").unlink()
        argv = self.dry_run("baseline")
        self.assertEqual(
            self.flag(argv, "--mount"), f"type=bind,source={self.cache},target=/cache"
        )
        self.assertFalse(self.cache.exists())
        self.assertFalse(self.record.exists())

    def test_default_cache_is_relative_to_repository(self):
        repo = self.folder / "repository with spaces"
        copied = repo / "deploy/docker/run.sh"
        copied.parent.mkdir(parents=True)
        copied.write_bytes(LAUNCHER.read_bytes())
        self.env.pop("MODEL_CACHE_DIR")
        result = self.launch("baseline", "--dry-run", launcher=copied)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.flag(shlex.split(result.stdout), "--mount"),
            f"type=bind,source={repo}/.cache/model,target=/cache",
        )
        self.assertFalse((repo / ".cache").exists())

    def test_environment_text_cannot_add_flags_or_execute_commands(self):
        marker = self.folder / "unexpected-file"
        cache = str(self.folder / "cache") + f" $(touch {marker})"
        self.env.update(
            MODEL_CACHE_DIR=cache,
            MODEL_REVISION="unreviewed-revision",
            VLLM_ARGS=f"--trust-remote-code; touch {marker}",
        )
        argv = self.dry_run("baseline")
        self.assertEqual(self.flag(argv, "--revision"), FP16_REVISION)
        self.assertEqual(self.flag(argv, "--mount"), f"type=bind,source={cache},target=/cache")
        self.assertNotIn("--trust-remote-code", argv)
        self.assertFalse(marker.exists())
        self.assertFalse(Path(cache).exists())

    def test_invalid_arguments_exit_two(self):
        for args in (
            (),
            ("unknown",),
            ("baseline", "--invalid"),
            ("baseline", "--dry-run", "extra"),
            ("--dry-run", "baseline"),
        ):
            with self.subTest(args=args):
                result = self.launch(*args)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertTrue(result.stderr.strip())
        self.assertFalse(self.record.exists())

    def test_mount_path_with_comma_is_rejected(self):
        self.env["MODEL_CACHE_DIR"] = str(self.folder / "cache,readonly")
        result = self.launch("baseline", "--dry-run")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertFalse(self.record.exists())

    def test_real_launch_refuses_mac_before_invoking_docker(self):
        result = self.launch("baseline")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Linux", result.stderr)
        self.assertFalse(self.record.exists())
        self.assertFalse(self.cache.exists())

    def test_linux_launch_passes_exact_arguments_and_propagates_exit_status(self):
        expected = self.dry_run("batched")[1:]
        self.stub("uname", "printf 'Linux\\n'\n")
        result = self.launch("batched")
        self.assertEqual(result.returncode, 17, result.stderr)
        self.assertTrue(self.cache.is_dir())
        expected[expected.index("--mount") + 1] = (
            f"type=bind,source={self.cache.resolve()},target=/cache"
        )
        actual = self.record.read_bytes().decode().split("\0")[:-1]
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()

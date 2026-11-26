from datetime import datetime, timedelta, timezone
from http.client import BadStatusLine, IncompleteRead
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "lambda_deadline", Path(__file__).resolve().parents[1] / "scripts/lambda-deadline.py")
deadline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(deadline)
INSTANCE = "0920582c7ff041399e34823a0be62549"
NOW = datetime(2026, 11, 26, 1, tzinfo=timezone.utc)


class FakeClock:
    def __init__(self):
        self.seconds = 0
        self.sleeps = []

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.seconds += seconds


class FakeAPI:
    def __init__(self, statuses):
        self.statuses = iter(statuses)
        self.calls = []
        self.fail_posts = 0

    def __call__(self, method, path, payload=None):
        self.calls.append((method, path, payload))
        if method == "POST":
            if self.fail_posts:
                self.fail_posts -= 1
                raise RuntimeError("sensitive provider message")
            return {"data": {}}
        status = next(self.statuses)
        if isinstance(status, Exception):
            raise status
        return {"data": {"id": INSTANCE, "status": status, "jupyter_token": "secret"}}


class LambdaDeadlineTests(unittest.TestCase):
    def run_guard(self, api, when=None):
        self.clock = FakeClock()
        self.events = []
        deadline.guard(INSTANCE, when or NOW, lambda event, **details: self.events.append(
            {"event": event, **details}), api, now=lambda: NOW,
            sleep=self.clock.sleep, monotonic=lambda: self.clock.seconds)

    def test_utc_timestamp_required(self):
        for value in ("2026-11-26", "2026-11-26T01:00:00", "2026-11-26T01:00:00+11:00", "nan"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                deadline.parse_deadline(value)
        self.assertEqual(deadline.parse_deadline("2026-11-26T01:00:00Z"), NOW)
        self.assertEqual(deadline.parse_deadline("2026-11-26T01:00:00+00:00"), NOW)

    def test_future_beyond_four_hours_rejected_before_api(self):
        api = FakeAPI([])
        with self.assertRaises(ValueError):
            self.run_guard(api, NOW + timedelta(hours=4, seconds=1))
        self.assertEqual(api.calls, [])

    def test_invalid_uuid_rejected_before_api(self):
        api = FakeAPI([])
        with self.assertRaises(ValueError):
            deadline.guard("invalid", NOW, lambda *args, **kwargs: None, api)
        self.assertEqual(api.calls, [])

    def test_initial_instance_must_match_and_status_be_known(self):
        for record in ({"data": {"id": "0" * 32, "status": "active"}},
                       {"data": {"id": INSTANCE, "status": "unknown"}}, {}, {"data": None}):
            with self.subTest(record=record), self.assertRaises(ValueError):
                self.run_guard(lambda *args: record)
        self.assertEqual(self.events, [])

    def test_wait_then_terminate_exact_target(self):
        api = FakeAPI(["active", "terminated"])
        self.run_guard(api, NOW + timedelta(seconds=125))
        self.assertEqual(self.clock.sleeps, [60, 60, 5])
        self.assertEqual(api.calls, [
            ("GET", "/instances/" + INSTANCE, None),
            ("POST", "/instance-operations/terminate", {"instance_ids": [INSTANCE]}),
            ("GET", "/instances/" + INSTANCE, None)])
        self.assertEqual(self.events[-1]["event"], "terminated")

    def test_past_deadline_terminates_immediately(self):
        self.run_guard(FakeAPI(["active", "terminated"]), NOW - timedelta(minutes=1))
        self.assertEqual(self.clock.sleeps, [])

    def test_already_terminated_does_not_post(self):
        api = FakeAPI(["terminated"])
        self.run_guard(api)
        self.assertEqual(len(api.calls), 1)

    def test_failed_post_still_verifies_then_retries(self):
        api = FakeAPI(["active", "active", "terminated"])
        api.fail_posts = 1
        self.run_guard(api)
        self.assertEqual(self.clock.sleeps, [5])
        self.assertEqual(sum(call[0] == "POST" for call in api.calls), 2)
        self.assertNotIn("secret", json.dumps(self.events))
        self.assertNotIn("sensitive", json.dumps(self.events))

    def test_successful_post_not_treated_as_confirmed_termination(self):
        api = FakeAPI(["active", "terminating", "terminated"])
        self.run_guard(api)
        self.assertEqual(self.clock.sleeps, [5])
        self.assertEqual(self.events[-1]["status"], "terminated")

    def test_verification_failures_are_bounded(self):
        api = FakeAPI(["active"] + [RuntimeError("private")] * 12)
        with self.assertRaisesRegex(RuntimeError, "termination unconfirmed"):
            self.run_guard(api)
        self.assertEqual(len(self.clock.sleeps), 11)
        self.assertEqual(sum(call[0] == "POST" for call in api.calls), 12)

    def test_initial_api_latency_does_not_extend_deadline(self):
        clock = FakeClock()
        api = FakeAPI(["active", "terminated"])

        def delayed_api(*args):
            clock.seconds += 10
            return api(*args)

        deadline.guard(INSTANCE, NOW + timedelta(seconds=65), lambda *args, **kwargs: None,
                       delayed_api, now=lambda: NOW, sleep=clock.sleep,
                       monotonic=lambda: clock.seconds)
        self.assertEqual(clock.sleeps, [55])

    def test_api_timeout_and_error_redaction(self):
        api = deadline.LambdaAPI("secret")
        with patch.object(api.opener, "open", side_effect=deadline.HTTPError(
                "https://private", 503, "secret provider text", None, None)) as request:
            with self.assertRaisesRegex(RuntimeError, "^HTTP 503$"):
                api("GET", "/instances/" + INSTANCE)
            self.assertEqual(request.call_args.kwargs, {"timeout": 10})
            sent = request.call_args.args[0]
            self.assertEqual(sent.full_url, deadline.API_URL + "/instances/" + INSTANCE)
            self.assertEqual(sent.get_header("Authorization"), "Bearer secret")
        self.assertIsNone(deadline.NoRedirect().redirect_request(None, None, 302, None, None,
                                                                "https://elsewhere"))

    def test_incomplete_http_response_retries_termination(self):
        active = json.dumps({"data": {"id": INSTANCE, "status": "active"}}).encode()
        terminated = json.dumps({"data": {"id": INSTANCE, "status": "terminated"}}).encode()
        for failure in (IncompleteRead(b"secret", 100), BadStatusLine("secret")):
            with self.subTest(failure=type(failure).__name__):
                api = deadline.LambdaAPI("secret")
                with patch.object(api.opener, "open") as opened:
                    opened.return_value.__enter__.return_value.read.side_effect = [
                        active, failure, active, b"{}", terminated]
                    self.run_guard(api)
                methods = [call.args[0].method for call in opened.call_args_list]
                self.assertEqual(methods, ["GET", "POST", "GET", "POST", "GET"])
                self.assertEqual(self.clock.sleeps, [5])
                self.assertEqual(self.events[-1]["event"], "terminated")
                self.assertNotIn("secret", json.dumps(self.events))

    def test_cli_requires_key_and_refuses_existing_or_symlink_output(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True), \
                patch("sys.stderr", new_callable=io.StringIO), patch.object(deadline, "LambdaAPI") as api:
            path = Path(directory) / "guard.jsonl"
            args = ["--instance-id", INSTANCE, "--deadline", NOW.isoformat(), "--output", str(path)]
            self.assertEqual(deadline.main(args), 2)
            self.assertFalse(path.exists())
            os.environ["LAMBDA_API_KEY"] = "secret"
            path.write_text("preserve")
            self.assertEqual(deadline.main(args), 2)
            path.unlink()
            path.symlink_to(Path(directory) / "missing")
            self.assertEqual(deadline.main(args), 2)
            api.assert_not_called()

    def test_cli_output_permissions_and_secret_redaction(self):
        api = FakeAPI(["active", "terminated"])
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"LAMBDA_API_KEY": "secret"}), \
                patch.object(deadline, "LambdaAPI", return_value=api), \
                patch("sys.stdout", new_callable=io.StringIO) as stdout, \
                patch("sys.stderr", new_callable=io.StringIO) as stderr:
            path = Path(directory) / "guard.jsonl"
            self.assertEqual(deadline.main(["--instance-id", INSTANCE, "--deadline", NOW.isoformat(),
                                            "--output", str(path)]), 0)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(records[-1]["event"], "terminated")
            self.assertIn("not a hard financial cap", stderr.getvalue())
            self.assertNotIn("secret", path.read_text() + stdout.getvalue() + stderr.getvalue())

    def test_cli_interruption_demands_manual_termination(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"LAMBDA_API_KEY": "secret"}), \
                patch.object(deadline, "LambdaAPI"), \
                patch.object(deadline, "guard", side_effect=KeyboardInterrupt), \
                patch("sys.stdout", new_callable=io.StringIO), \
                patch("sys.stderr", new_callable=io.StringIO) as stderr:
            path = Path(directory) / "guard.jsonl"
            self.assertEqual(deadline.main(["--instance-id", INSTANCE, "--deadline", NOW.isoformat(),
                                            "--output", str(path)]), 130)
            self.assertIn("Terminate the instance", stderr.getvalue())
            self.assertEqual(json.loads(path.read_text())["event"], "interrupted")


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
import argparse
from datetime import datetime, timezone
from http.client import HTTPException
import json
import math
import os
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import UUID


API_URL = "https://cloud.lambda.com/api/v1"
STATUSES = {"booting", "active", "unhealthy", "terminated", "terminating", "preempted"}
WARNING = "Keep supervising this guard. Controller/network failure can prevent termination; this is not a hard financial cap."


def parse_deadline(value):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
            raise ValueError
        return parsed.astimezone(timezone.utc)
    except (ValueError, TypeError):
        raise ValueError("deadline must be an ISO timestamp with UTC Z or +00:00") from None


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class LambdaAPI:
    def __init__(self, key):
        self.key = key
        self.opener = build_opener(NoRedirect())

    def __call__(self, method, path, payload=None):
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(API_URL + path, data=body, method=method, headers={
            "Authorization": "Bearer " + self.key,
            "Accept": "application/json", "Content-Type": "application/json",
        })
        try:
            with self.opener.open(request, timeout=10) as response:
                raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise ValueError
            return json.loads(raw)
        except HTTPError as error:
            raise RuntimeError("HTTP " + str(error.code)) from None
        except (URLError, OSError, ValueError, HTTPException):
            raise RuntimeError("API transport or JSON error") from None


def instance_status(response, instance_id):
    try:
        data = response["data"]
        if UUID(data["id"]).hex != instance_id or data["status"] not in STATUSES:
            raise ValueError
        return data["status"]
    except (KeyError, TypeError, ValueError, AttributeError):
        raise ValueError("API returned an unexpected instance ID or status") from None


def guard(instance_id, deadline, emit, api, *, now=None, sleep=time.sleep,
          monotonic=time.monotonic):
    now = now or (lambda: datetime.now(timezone.utc))
    instance_id = UUID(instance_id).hex
    remaining = (deadline - now()).total_seconds()
    if not math.isfinite(remaining) or remaining > 4 * 3600:
        raise ValueError("deadline must be at most four hours from now")
    # Include API latency, and do not extend the interval for wall-clock corrections.
    stop_at = monotonic() + max(0, remaining)
    status = instance_status(api("GET", "/instances/" + instance_id), instance_id)
    emit("armed", instance_id=instance_id, deadline=deadline.isoformat(), status=status)
    if status == "terminated":
        emit("terminated", status=status)
        return
    while True:
        wait = stop_at - monotonic()
        if wait <= 0:
            break
        sleep(min(60, wait))
    emit("deadline_reached")
    for attempt in range(1, 13):
        try:
            api("POST", "/instance-operations/terminate", {"instance_ids": [instance_id]})
            emit("termination_requested", attempt=attempt)
        except RuntimeError:
            emit("termination_request_failed", attempt=attempt)
        try:
            status = instance_status(api("GET", "/instances/" + instance_id), instance_id)
            emit("observed", attempt=attempt, status=status)
            if status == "terminated":
                emit("terminated", status=status)
                return
        except (RuntimeError, ValueError):
            emit("verification_failed", attempt=attempt)
        if attempt < 12:
            sleep(5)
    raise RuntimeError("termination unconfirmed: terminate this instance in the Lambda console now")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Terminate one approved Lambda instance at a UTC deadline.")
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        instance_id = UUID(args.instance_id).hex
        deadline = parse_deadline(args.deadline)
        remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
        if remaining > 4 * 3600:
            raise ValueError("deadline must be at most four hours from now")
        key = os.environ.get("LAMBDA_API_KEY", "").strip()
        if not key:
            raise ValueError("set LAMBDA_API_KEY in this terminal")
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            def emit(event, **details):
                record = {"time": datetime.now(timezone.utc).isoformat(), "event": event, **details}
                output.write(json.dumps(record) + "\n")
                output.flush()
                print(event, flush=True)

            print(WARNING, file=sys.stderr, flush=True)
            try:
                guard(instance_id, deadline, emit, LambdaAPI(key))
            except KeyboardInterrupt:
                emit("interrupted")
                print("Guard stopped. Terminate the instance in the Lambda console now.", file=sys.stderr)
                return 130
            except (ValueError, RuntimeError) as error:
                emit("failed")
                print(str(error) + ". Check the Lambda console immediately.", file=sys.stderr)
                return 2
        return 0
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

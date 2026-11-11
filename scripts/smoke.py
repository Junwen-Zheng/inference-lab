#!/usr/bin/env python3
"""Exercise real loopback HTTP streaming against the synthetic fixture."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import threading

from inference_lab.client import request_completion
from inference_lab.mock_server import make_server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/smoke.json"))
    args = parser.parse_args(argv)
    server = make_server(port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = request_completion(
            f"http://127.0.0.1:{server.server_port}", "inference-model",
            "Synthetic protocol smoke", max_tokens=4, timeout_s=5,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "schema_version": 1, "backend": "mock", "synthetic": True,
            "request": asdict(result),
        }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if not result.success or result.output_tokens != 4 or result.ttft_s is None or not result.text:
            raise RuntimeError(f"streaming smoke failed; saved {args.output}: {result.error}")
        print("SYNTHETIC HTTP streaming smoke passed: 4 reported mock tokens.")
        print(f"Saved {args.output}. These are synthetic protocol measurements.")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

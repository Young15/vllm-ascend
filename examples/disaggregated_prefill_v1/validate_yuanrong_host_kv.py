# SPDX-License-Identifier: Apache-2.0
"""Validate the real PD request path and Decode-side Host KV counters."""

import argparse
import ast
import json
import time
import urllib.request
from pathlib import Path

STATS_MARKER = "Yuanrong Host KV cumulative Decode stats: "


def read_stats(log: Path, offset: int) -> list[dict]:
    with log.open() as stream:
        stream.seek(offset)
        lines = stream.readlines()
    return [ast.literal_eval(line.split(STATS_MARKER, 1)[1]) for line in lines if STATS_MARKER in line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-url", required=True, help="Running PD proxy base URL")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--decode-log", required=True, type=Path, help="Log of a fresh, idle Decode instance")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    if args.max_tokens < 3:
        parser.error("at least three generated tokens are needed to exercise subsequent Decode cache reuse")
    if read_stats(args.decode_log, 0):
        parser.error("use a fresh idle Decode instance so cumulative counters describe this validation request")
    log_offset = args.decode_log.stat().st_size
    request = urllib.request.Request(
        args.proxy_url.rstrip("/") + "/v1/completions",
        data=json.dumps(
            {
                "model": args.model,
                "prompt": args.prompt_file.read_text(),
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "ignore_eos": True,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        result = json.load(response)
    if not result.get("choices"):
        raise RuntimeError("PD proxy returned no completion")
    deadline = time.monotonic() + 10
    stats = []
    while time.monotonic() < deadline:
        stats = read_stats(args.decode_log, log_offset)
        if stats:
            break
        time.sleep(0.1)
    required = (
        "completed_decode_layers",
        "get_calls",
        "host_copy_bytes",
        "h2d_bytes",
        "host_hits",
        "verified_h2d_bytes",
    )
    if not stats or any(not any(row.get(key, 0) > 0 for row in stats) for key in required):
        raise RuntimeError("Decode did not prove Get, owned copy, H2D, attention preparation and Host reuse")
    if any(row.get("owned_bytes") != 0 or row.get("decode_put_calls") != 0 for row in stats):
        raise RuntimeError("Decode leaked request-owned Host objects or wrote generated KV to the store")
    report = {
        "status": "pd_path_checked",
        "usage": result.get("usage"),
        "decode_stats": stats,
        "limits": "Generated-history KV is masked after its current forward; this is not a quality/performance test.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()

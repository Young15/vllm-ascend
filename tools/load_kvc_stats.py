import argparse
import glob
import json
import math
import re
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

KVC_TIMING_PATTERN = re.compile(
    r"(?P<backend>Yuanrong(?:Backend)?|Mooncake(?:Backend)?|Memcache(?:Backend)?)\s+"
    r"(?P<op>load_kvc|store_kvc|exists)\s+took\s+"
    r"(?P<elapsed_ms>\d+(?:\.\d+)?)\s+ms"
    r"(?:,\s+bytes=(?P<bytes>\d+))?"
)

BYTES_PER_MB = 1000 * 1000


def normalize_backend_name(name: str) -> str:
    return name.removesuffix("Backend")


def group_key(backend: str, op: str) -> str:
    return f"{backend}/{op}"


def iter_input_paths(patterns: list[str], recursive: bool) -> Iterable[Path]:
    for pattern in patterns:
        path = Path(pattern)
        if path.exists():
            if path.is_dir():
                glob_pattern = "**/*" if recursive else "*"
                yield from (p for p in path.glob(glob_pattern) if p.is_file())
            else:
                yield path
            continue

        matched_paths = [Path(p) for p in glob.glob(pattern) if Path(p).is_file()]
        if not matched_paths:
            raise FileNotFoundError(f"No files matched: {pattern}")
        yield from matched_paths


def parse_lines(lines: Iterable[str]) -> dict[str, list[dict[str, int | float]]]:
    records: dict[str, list[dict[str, int | float]]] = defaultdict(list)
    for line in lines:
        match = KVC_TIMING_PATTERN.search(line)
        if match is None:
            continue
        backend = normalize_backend_name(match.group("backend"))
        key = group_key(backend, match.group("op"))
        record: dict[str, int | float] = {"elapsed_ms": float(match.group("elapsed_ms"))}
        if match.group("bytes") is not None:
            record["bytes"] = int(match.group("bytes"))
        records[key].append(record)
    return dict(records)


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    rank = math.ceil(percent / 100 * len(sorted_values))
    index = min(max(rank - 1, 0), len(sorted_values) - 1)
    return sorted_values[index]


def summarize(records: list[dict[str, int | float]]) -> dict[str, int | float]:
    elapsed_values = [float(record["elapsed_ms"]) for record in records]
    total_elapsed = sum(elapsed_values)
    total_bytes = sum(int(record.get("bytes", 0)) for record in records)
    count = len(elapsed_values)
    throughput_mb_s = 0.0
    if total_elapsed > 0:
        throughput_mb_s = total_bytes / BYTES_PER_MB / (total_elapsed / 1000)
    return {
        "count": count,
        "avg_ms": total_elapsed / count if count else 0.0,
        "p50_ms": percentile(elapsed_values, 50),
        "p90_ms": percentile(elapsed_values, 90),
        "p95_ms": percentile(elapsed_values, 95),
        "p99_ms": percentile(elapsed_values, 99),
        "max_ms": max(elapsed_values) if elapsed_values else 0.0,
        "total_mb": total_bytes / BYTES_PER_MB,
        "throughput_mb_s": throughput_mb_s,
    }


def merge_records(records_by_backend: dict[str, list[dict[str, int | float]]]) -> list[dict[str, int | float]]:
    merged: list[dict[str, int | float]] = []
    for records in records_by_backend.values():
        merged.extend(records)
    return merged


def format_table(summaries: dict[str, dict[str, int | float]]) -> str:
    headers = [
        "backend/op",
        "count",
        "avg_ms",
        "p50_ms",
        "p90_ms",
        "p95_ms",
        "p99_ms",
        "max_ms",
        "total_mb",
        "throughput_mb_s",
    ]
    rows = []
    for backend, summary in summaries.items():
        rows.append(
            [
                backend,
                str(summary["count"]),
                f"{summary['avg_ms']:.3f}",
                f"{summary['p50_ms']:.3f}",
                f"{summary['p90_ms']:.3f}",
                f"{summary['p95_ms']:.3f}",
                f"{summary['p99_ms']:.3f}",
                f"{summary['max_ms']:.3f}",
                f"{summary['total_mb']:.3f}",
                f"{summary['throughput_mb_s']:.3f}",
            ]
        )
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]
    lines = ["  ".join(header.ljust(width) for header, width in zip(headers, widths))]
    lines.append("  ".join("-" * width for width in widths))
    lines.extend("  ".join(cell.ljust(width) for cell, width in zip(row, widths)) for row in rows)
    return "\n".join(lines)


def read_file_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as file:
        yield from file


def collect_records(args: argparse.Namespace) -> dict[str, list[dict[str, int | float]]]:
    records_by_backend: dict[str, list[dict[str, int | float]]] = defaultdict(list)
    if not args.paths or args.paths == ["-"]:
        parsed = parse_lines(sys.stdin)
        for backend, records in parsed.items():
            records_by_backend[backend].extend(records)
        return dict(records_by_backend)

    for path in iter_input_paths(args.paths, args.recursive):
        parsed = parse_lines(read_file_lines(path))
        for backend, records in parsed.items():
            records_by_backend[backend].extend(records)
    return dict(records_by_backend)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize Yuanrong/Mooncake/Memcache load_kvc, store_kvc, and exists latency from service logs."
    )
    parser.add_argument("paths", nargs="*", help="Log files, directories, globs, or '-' for stdin.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan directories.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a table.")
    args = parser.parse_args()

    records_by_backend = collect_records(args)
    if not records_by_backend:
        print("No KVC timing records found.", file=sys.stderr)
        return 1

    summaries = {backend: summarize(records) for backend, records in sorted(records_by_backend.items())}
    summaries["overall"] = summarize(merge_records(records_by_backend))

    if args.json:
        print(json.dumps(summaries, indent=2, sort_keys=True))
    else:
        print(format_table(summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

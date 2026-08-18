from tools.load_kvc_stats import parse_lines, summarize


def test_parse_exists_timing_for_all_backends():
    records = parse_lines(
        [
            "Yuanrong exists took 1.250 ms, keys=2",
            "MooncakeBackend exists took 2.500 ms, keys=3",
            "Memcache exists took 3.750 ms, keys=4",
        ]
    )

    assert records == {
        "Yuanrong/exists": [{"elapsed_ms": 1.25}],
        "Mooncake/exists": [{"elapsed_ms": 2.5}],
        "Memcache/exists": [{"elapsed_ms": 3.75}],
    }


def test_exists_summary_has_latency_without_transfer_throughput():
    summary = summarize([{"elapsed_ms": 1.0}, {"elapsed_ms": 3.0}])

    assert summary["count"] == 2
    assert summary["avg_ms"] == 2.0
    assert summary["total_mb"] == 0.0
    assert summary["throughput_mb_s"] == 0.0


def test_parse_transfer_logs_and_ignore_unrelated_lines():
    records = parse_lines(
        [
            "INFO Yuanrong load_kvc took 2.000 ms, bytes=1000000",
            "MooncakeBackend store_kvc took 4.000 ms, bytes=2000000",
            "Memcache load_kvc took 1.000 ms, bytes=0",
            "unrelated message",
        ]
    )
    assert records == {
        "Yuanrong/load_kvc": [{"elapsed_ms": 2.0, "bytes": 1000000}],
        "Mooncake/store_kvc": [{"elapsed_ms": 4.0, "bytes": 2000000}],
        "Memcache/load_kvc": [{"elapsed_ms": 1.0, "bytes": 0}],
    }


def test_summary_percentiles_and_throughput():
    summary = summarize([{"elapsed_ms": 1.0, "bytes": 1000000}, {"elapsed_ms": 3.0, "bytes": 2000000}])
    assert summary == {
        "count": 2,
        "avg_ms": 2.0,
        "p50_ms": 1.0,
        "p90_ms": 3.0,
        "p95_ms": 3.0,
        "p99_ms": 3.0,
        "max_ms": 3.0,
        "total_mb": 3.0,
        "throughput_mb_s": 750.0,
    }


def test_empty_and_zero_duration_summary():
    assert all(value == 0 for value in summarize([]).values())
    summary = summarize([{"elapsed_ms": 0.0, "bytes": 1000000}])
    assert summary["count"] == 1
    assert summary["total_mb"] == 1.0
    assert summary["throughput_mb_s"] == 0.0

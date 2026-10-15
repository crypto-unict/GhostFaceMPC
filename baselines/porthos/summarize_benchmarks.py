#!/usr/bin/env python3
"""Aggregate the Porthos pool reports of one run into a single summary.

Reads every ``pool-<n>.json`` that ``run_porthos_mtls.py`` wrote into
``--pools``, and writes ``report.json`` (the full summary, the setup phase and
the per-request barrier included) and ``samples.csv`` (one row per measured
request) into ``--output``.  The summary is computed from the pool files rather
than by hand, so it cannot drift from them.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
from hashlib import sha256
from pathlib import Path

from ghostmpc.backends.porthos import (
    load_pool_samples,
    summarize_offline,
    summarize_phase,
    summarize_samples,
)

POOL = re.compile(r"^pool-(\d+)\.json$")
CSV_FIELDS = ("pool", "request", "wall_seconds", "cpu_seconds_total",
              "bytes_total", "rounds", "rounds_by_party", "send_calls",
              "recv_calls", "peak_rss_bytes", "bytes_reconciled",
              "causal_rounds_validated")


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def summarize(pools_dir: Path) -> dict[str, object]:
    pools = sorted((path for path in pools_dir.iterdir()
                    if POOL.match(path.name)),
                   key=lambda path: int(POOL.match(path.name).group(1)))
    if not pools:
        raise SystemExit(f"{pools_dir}: no pool-<n>.json reports")
    samples: list[dict] = []
    headers = []
    for pool in pools:
        samples += load_pool_samples(pool)
        headers.append(json.loads(pool.read_text()))
    header = headers[0]
    for other in headers[1:]:
        for key in ("backend", "threads_per_party", "score"):
            if other.get(key) != header.get(key):
                raise SystemExit(f"pools disagree on {key!r}: "
                                 f"{header.get(key)!r} != {other.get(key)!r}")
    rounds = header.get("rounds") or {}
    validated = all(bool((item.get("rounds") or {}).get("validated"))
                    for item in headers)
    summary = summarize_samples(samples)
    if not validated:
        summary["rounds"] = {"median": None, "p5": None, "p95": None,
                             "observations": 0, "status": "not-available"}
    warmups = [sum(1 for row in item["samples"] if row["warmup"])
               for item in headers]
    trials = [len(item["samples"]) - count for item, count in zip(headers, warmups)]
    reconciled = all(row["bytes_reconciled"] for row in samples)
    return {
        "schema_version": 1,
        "backend": header["backend"],
        "truncation": header.get("truncation"),
        "score": bool(header.get("score")),
        "transport": header.get("transport"),
        "measurement": {
            "pools": len(pools),
            "warmups_per_pool": warmups,
            "trials_per_pool": trials,
            "measured_samples": len(samples),
            "fresh_key_sets": len(pools),
            "threads_per_party": header.get("threads_per_party"),
            "thread_knob": header.get("thread_knob"),
            "bytes": "payload sent by P0, P1 and P2, counted once at the sender",
            "bytes_sender_receiver_reconciled": reconciled,
            "rounds": {
                "method": rounds.get("method"),
                "validated": validated,
                "build_manifests": sorted({str((item.get("rounds") or {}).get(
                    "build_manifest")) for item in headers}),
            },
            "excluded_from_online": [
                "mTLS handshake", "socket setup", "key loading",
                "weight and gallery loading", "per-request synchronization barrier"],
        },
        # The figures one verification request costs: the medians over every
        # measured request of every pool.
        "request": {
            "bytes_total": summary["bytes_total"]["median"],
            "rounds": summary["rounds"]["median"],
            "wall_seconds": summary["wall_seconds"]["median"],
            "cpu_seconds_total": summary["cpu_seconds_total"]["median"],
        },
        "summary": summary,
        # Both phases sit outside `start_m()`/`end_m()` and stay outside the
        # summary above; they are reported so the exclusion is visible.
        "offline": summarize_offline(samples),
        "sync_barrier": summarize_phase(samples, "sync_barrier"),
        "pool_reports": [{"path": str(path), "sha256": digest(path)}
                         for path in pools],
        "host": {"platform": platform.platform(),
                 "processor": platform.processor()},
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pools", type=Path, required=True,
                        help="directory holding pool-<n>.json")
    parser.add_argument("--output", type=Path,
                        help="where to write report.json and samples.csv "
                             "(default: the --pools directory)")
    args = parser.parse_args()
    output = args.output or args.pools
    report = summarize(args.pools)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    with (output / "samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS,
                                extrasaction="ignore")
        writer.writeheader()
        for row in report["samples"]:
            writer.writerow({**row, "rounds_by_party": json.dumps(
                row.get("rounds_by_party"), separators=(",", ":"))})
    request = report["request"]
    measurement = report["measurement"]
    print(f"{report['backend']}, {measurement['threads_per_party']} threads "
          f"per party, {measurement['pools']} pool(s), "
          f"{measurement['measured_samples']} measured request(s)")
    print(f"  bytes per request (median):   {request['bytes_total']:,.0f}")
    print("  online rounds (median):       "
          + ("n/a" if request["rounds"] is None else f"{request['rounds']:,.0f}"))
    print(f"  wall seconds (median):        {request['wall_seconds']:.3f}")
    print(f"  CPU seconds, 3 parties:       {request['cpu_seconds_total']:.3f}")
    print(f"  bytes reconciled:             "
          f"{measurement['bytes_sender_receiver_reconciled']}")
    print(output / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""CSV and JSON reports written after a benchmark run."""

from __future__ import annotations

import csv
import json
import os
import platform
import statistics
import sys
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from ..core.model import Manifest
from .stats import Trial, percentile, summarize


def write_report(trials: list[Trial], manifest: Manifest, output_dir: Path,
                 mpspdz_commit: str, *,
                 protocol_runtime_args: Sequence[str] | None = None,
                 score: object | None = None,
                 match: object | None = None,
                 identify: object | None = None,
                 network: dict[str, object] | None = None,
                 threads: int | None = None,
                 optimizations: dict[str, object] | None = None
                 ) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(trials)
    cold_by_pool: dict[str, float] = {}
    for trial in trials:
        if trial.layer != "end_to_end" or trial.phase != "total":
            continue
        cold_by_pool[trial.pool_id] = cold_by_pool.get(trial.pool_id, 0.0) + trial.wall_seconds
    cold_values = list(cold_by_pool.values())
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "model_id": manifest.model_id, "checkpoint_sha256": manifest.checkpoint_sha256,
        "weights_sha256": manifest.weights_sha256, "mpspdz_commit": mpspdz_commit,
        "host": {"platform": platform.platform(), "python": platform.python_version(),
                 "cpu_count": os.cpu_count()},
        # The compute budget the measured program was emitted at.  Bytes and
        # rounds do not move with it; wall and CPU do.
        "threads_per_party": threads,
        # The circuit optimizations the program was emitted with, and the
        # truncation fusions they selected; null for a program that has no
        # switches (the baselines).
        "optimizations": optimizations,
        # Always present, so a reader never has to infer from a missing key
        # whether a run was shaped. Outside the emulated network the profile is
        # `co-host` and the emulation is `none`.
        "network": network if network is not None else {
            "profile": "co-host", "emulation": "none", "physical_hosts": False},
        "definition": {
            "wall_seconds": "per-request server execution elapsed time; excludes compilation",
            "cpu_seconds": "per-process CPU time; excludes I/O wait",
            "bytes": "only inter-server traffic; client traffic excluded",
            "rounds": "runtime protocol rounds reported by MP-SPDZ, maximum across parties",
            "phase": ("total is the complete measured timer; online is the "
                      "input-dependent work; offline is secure preprocessing"),
            "pool_id": "isolated execution allocation; a consumed id can never return to ready state",
            "layer": ("end_to_end is one whole request; input_sharing, "
                      "embedding and score are the operations inside it and "
                      "sum to it; weight_load and gallery_load are "
                      "once-per-process setup outside every request"),
            "match_layer": ("the match decision, a fourth operation inside a "
                            "request when a match is compiled in, and the "
                            "match sums to end_to_end with the other three"),
            "identify_layer": "the closed-set 1:N identification, an operation inside a request that replaces the score and sums to end_to_end with input_sharing and the embedding",
        },
        "security": {
            "protocol": trials[0].protocol if trials else None,
            # The options the parties were launched with.
            "protocol_runtime_args": (None if protocol_runtime_args is None
                                      else list(protocol_runtime_args)),
            "note": "weights are public; inputs and embeddings remain secret-shared",
        },
        # Present and null when no score was compiled in, so a reader never has
        # to infer from a missing key whether a run scored anything.
        "score": score,
        # The same, for the match decision: null means the servers returned the
        # two score accumulators and took no decision.
        "match": match,
        # The same, for the closed-set identification: null means the request
        # answered a declared identity rather than ranking the database.
        "identify": identify,
        "cold_end_to_end_wall_seconds": ({
            "median": statistics.median(cold_values),
            "p5": percentile(cold_values, .05),
            "p95": percentile(cold_values, .95),
        } if cold_values else None),
        "measurement_method": "each pool executes warmups+trials in one persistent process; a measured timer brackets only the post-warmup trials and its totals are normalized per trial",
        "trials": [asdict(t) for t in trials], "summary": summary,
    }
    json_path = output_dir / "report.json"
    json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "protocol", "layer", "phase", "trials",
            "metric", "median", "p5", "p95"])
        for row in summary:
            for metric in ("wall_seconds", "cpu_seconds_p0", "cpu_seconds_p1", "cpu_seconds_total",
                           "bytes_p0_to_p1", "bytes_p1_to_p0", "bytes_total", "rounds"):
                value = row[metric]
                writer.writerow([
                    row["protocol"], row["layer"], row["phase"], row["trials"],
                    metric, value["median"], value["p5"], value["p95"]])
    _write_network_csv(output_dir, metadata["network"])
    return csv_path, json_path


# Link statistics are properties of the emulated network and of whole process
# groups, not of individual trials, so they get their own flat file rather than
# a fabricated median/p5/p95 in summary.csv.
_NETWORK_CSV_KEYS = (
    "profile", "emulation", "tool", "physical_hosts",
    "payload_bytes_per_inference", "wire_bytes_per_inference",
    "wire_bytes_total", "payload_bytes_total", "wire_overhead_ratio",
    "achieved_mbit_per_second", "link_utilisation",
)


def _write_network_csv(output_dir: Path, network: dict[str, object]) -> Path | None:
    if network.get("emulation") != "simulated":
        return None
    path = output_dir / "network.csv"
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["key", "value"])
        configured = network.get("configured") or {}
        for key in ("rtt_ms", "bandwidth_mbit", "netem_limit_packets"):
            writer.writerow([f"configured_{key}", configured.get(key)])
        for key in _NETWORK_CSV_KEYS:
            if key in network:
                writer.writerow([key, network[key]])
        for pair, values in (network.get("measured_rtt_ms") or {}).items():
            writer.writerow([f"measured_rtt_min_ms_{pair}", values["min_ms"]])
        for party, count in enumerate(network.get("wire_bytes_by_party") or []):
            writer.writerow([f"wire_bytes_p{party}", count])
        for party, count in enumerate(network.get("drops_by_party") or []):
            writer.writerow([f"drops_p{party}", count])
    return path

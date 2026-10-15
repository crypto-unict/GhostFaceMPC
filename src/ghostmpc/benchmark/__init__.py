"""Three-party benchmarking of a generated MP-SPDZ program.

Three parties are launched together, on one host or in three emulated network
namespaces; each reports its own
communication counters and timers, which are then reconciled into a single
per-request figure.  A single compiled program bundles the warm-up and the
measured requests so that the two never differ in anything but position.

The modules split that work as follows:

``stats``
    parsing and aggregation of the numbers one run reports;
``process``
    launching the parties, draining their pipes, staging their input files;
``report``
    the CSV and JSON artifacts left behind;
``runner``
    the end-to-end entry point;
``network``
    the emulated LAN/WAN the parties can be launched into.
"""

from __future__ import annotations

from .process import (
    BenchmarkTimeout,
    mpspdz_commit,
    prepare_player_data,
    run_commands,
    run_group,
)
from .report import write_report
from .runner import run_benchmarks
from .stats import (
    Trial,
    parse_group_stats,
    parse_request_timer,
    percentile,
    summarize,
)

__all__ = [
    "BenchmarkTimeout",
    "Trial",
    "mpspdz_commit",
    "parse_group_stats",
    "parse_request_timer",
    "percentile",
    "prepare_player_data",
    "run_benchmarks",
    "run_commands",
    "run_group",
    "summarize",
    "write_report",
]

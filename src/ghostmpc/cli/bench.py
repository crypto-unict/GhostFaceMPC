"""Running the three-party benchmark of the deployed program."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..benchmark import run_benchmarks
from ..benchmark.network import CO_HOST, NETWORK_PROFILES, require_context
from ..codegen.fusion import Optimizations
from ..core.model import Manifest
from ..core.mpspdz_protocols import PROTOCOLS, get_protocol
from ._common import (
    _threads,
    add_score_arguments,
    identify_plan,
    match_plan,
    score_plan,
)


def _benchmark(args: argparse.Namespace) -> int:
    protocol = get_protocol(args.protocol)
    quantization = Manifest.load(Path(args.manifest)).quantization
    plan = score_plan(args, Path(args.manifest),
                      fractional_bits=quantization.fractional_bits,
                      ring_bits=quantization.ring_bits)
    decision = match_plan(args, plan)
    query = identify_plan(args, plan)
    csv_path, json_path = run_benchmarks(
        Path(args.manifest), Path(args.mpspdz), Path(args.output),
        trials=args.trials, warmups=args.warmups,
        threads=_threads(args.threads), pools=args.pools,
        port_base=args.port_base, protocol=protocol.name,
        timeout_seconds=args.timeout_seconds,
        skip_compile=args.skip_compile,
        score=plan, match=decision, identify=query,
        network=require_context(args.network_profile, parties=protocol.parties),
        optimizations=Optimizations.without(args.disable))
    print(json.dumps({"csv": str(csv_path), "json": str(json_path)}, indent=2))
    return 0


def register(commands: argparse._SubParsersAction) -> None:
    """Add this group's subcommands, in the order they are listed."""
    benchmark = commands.add_parser(
        "benchmark",
        help="compile the program and measure it on three MP-SPDZ parties")
    benchmark.add_argument("--manifest", default="artifacts/model/manifest.json")
    benchmark.add_argument("--mpspdz", default="third_party/MP-SPDZ")
    benchmark.add_argument("--output", default="results/benchmark")
    benchmark.add_argument(
        "--protocol", choices=tuple(PROTOCOLS),
        default="replicated-ring-party",
        help="replicated-ring-party is secure against one semi-honest server, "
             "ps-rep-ring-party against one actively corrupted server (with abort)")
    benchmark.add_argument("--trials", type=int, default=30,
                           help="measured requests per pool")
    benchmark.add_argument("--warmups", type=int, default=5,
                           help="discarded requests before the measured ones")
    benchmark.add_argument("--pools", type=int, default=3,
                           help="independent executions, each a fresh process group")
    benchmark.add_argument("--threads", type=int, choices=(0, 1, 2, 4, 8), default=0,
                           help="threads per party; 0 selects the largest "
                                "supported count available on the host")
    benchmark.add_argument("--port-base", type=int, default=5000)
    benchmark.add_argument(
        "--skip-compile", action="store_true",
        help="reuse the bytecode already in the MP-SPDZ tree instead of "
             "recompiling; refused unless the emitted program is unchanged")
    benchmark.add_argument("--timeout-seconds", type=float)
    benchmark.add_argument(
        "--network-profile", choices=tuple(NETWORK_PROFILES), default=CO_HOST,
        help="run each party in its own netem-shaped network namespace; "
             "anything other than co-host requires "
             "scripts/netsim/ghostmpc-netsim to have prepared the network")
    benchmark.add_argument(
        "--disable", action="append", default=[],
        choices=tuple(Optimizations.SWITCHES), metavar="OPTIMIZATION",
        help="emit the program without one circuit optimization (repeatable; "
             f"one of {', '.join(Optimizations.SWITCHES)}); for the ablation")
    add_score_arguments(benchmark)
    benchmark.set_defaults(func=_benchmark)

"""Emitting the MP-SPDZ program from a manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..codegen import generate
from ..core.identify import identify_plan_report
from ..core.match import match_plan_report
from ..core.model import Manifest
from ..core.mpspdz_protocols import PROTOCOLS
from ._common import (
    _threads,
    add_score_arguments,
    identify_plan,
    match_plan,
    score_plan,
)


def _generate(args: argparse.Namespace) -> int:
    quantization = Manifest.load(Path(args.manifest)).quantization
    plan = score_plan(args, Path(args.manifest),
                      fractional_bits=quantization.fractional_bits,
                      ring_bits=quantization.ring_bits)
    decision = match_plan(args, plan)
    query = identify_plan(args, plan)
    public_input = generate(
        Path(args.manifest), Path(args.output),
        threads=_threads(args.threads), requests=args.requests,
        score=plan, match=decision, identify=query, protocol=args.protocol,
    )
    print(json.dumps({
        "source": args.output,
        "public_input": str(public_input),
        "protocol": args.protocol,
        "score": None if plan is None else {
            "gallery_rows": plan.gallery_rows,
            "embedding_size": plan.embedding_size,
            "gallery_fractional_bits": plan.gallery_fractional_bits,
            "declared_widths": plan.declared_widths(),
        },
        "match": None if decision is None else match_plan_report(decision),
        "identify": None if query is None else identify_plan_report(query),
    }, indent=2))
    return 0


def register(commands: argparse._SubParsersAction) -> None:
    """Add this group's subcommands, in the order they are listed."""
    generate_cmd = commands.add_parser("generate", help="generate MP-SPDZ source")
    generate_cmd.add_argument("manifest")
    generate_cmd.add_argument("output")
    generate_cmd.add_argument("--threads", type=int, choices=(0, 1, 2, 4, 8), default=0,
                              help="0 selects the largest supported count available on the host")
    generate_cmd.add_argument("--requests", type=int, default=1)
    generate_cmd.add_argument(
        "--protocol", choices=tuple(PROTOCOLS), default="replicated-ring-party",
        help="replicated-ring-party is secure against one semi-honest server, "
             "ps-rep-ring-party against one actively corrupted server (with abort)")
    add_score_arguments(generate_cmd)
    generate_cmd.set_defaults(func=_generate)

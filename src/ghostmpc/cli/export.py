"""Exporting a checkpoint to a manifest, or to the Porthos baseline bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..core.model import CHECKPOINT_NAME, Quantization, deployment_quantization
from ..export.exporter import export_checkpoint


def _export(args: argparse.Namespace) -> int:
    # The float teacher exported on the ring32/f6/f_w9 profile is the starting
    # point of quantization-aware training, not a deployable model: its
    # accuracy collapses on the 32-bit ring until it is trained on it.
    # The ring64/f20 profile is the float model on the naive baseline's ring,
    # the starting point of the ablation.
    quantization = (deployment_quantization() if args.profile == "deployment"
                    else Quantization())
    manifest = export_checkpoint(
        Path(args.checkpoint), Path(args.destination),
        quantization=quantization)
    print(json.dumps(manifest.to_dict(), indent=2))
    return 0


def _export_porthos(args: argparse.Namespace) -> int:
    from ..backends.porthos import export_porthos_bundle

    result = export_porthos_bundle(
        Path(args.manifest), Path(args.destination),
        checkpoint=Path(args.checkpoint),
        input_path=(Path(args.input) if args.input else None), seed=args.seed,
        verify_float=not args.skip_float_validation,
        score=args.score, gallery_rows=args.gallery_rows,
        gallery_row=args.gallery_row)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def register(commands: argparse._SubParsersAction) -> None:
    """Add this group's subcommands, in the order they are listed."""
    export = commands.add_parser(
        "export",
        help="export a Keras checkpoint to a ring32/f6/f_w9 manifest and "
             "weights (the starting point of quantization-aware training)")
    export.add_argument("checkpoint")
    export.add_argument("destination")
    export.add_argument(
        "--profile", choices=("deployment", "ring64-f20"), default="deployment",
        help="arithmetic profile the manifest declares: the deployed "
             "ring32/f6/f_w9, or ring64/f20 for the ablation")
    export.set_defaults(func=_export)
    export_porthos = commands.add_parser(
        "export-porthos",
        help="export the ring64/f16 manifest to ONNX for the Porthos baseline")
    export_porthos.add_argument(
        "--manifest",
        default="artifacts/model-ring64-f16-b0/manifest.json")
    export_porthos.add_argument("--checkpoint", default=f"artifacts/{CHECKPOINT_NAME}")
    export_porthos.add_argument("--destination", default="artifacts/porthos")
    export_porthos.add_argument(
        "--input", help="preprocessed NHWC .npy input; deterministic synthetic if omitted")
    export_porthos.add_argument("--seed", type=int, default=0)
    export_porthos.add_argument(
        "--skip-float-validation", action="store_true",
        help="skip the check that the ONNX graph reproduces the Keras model")
    export_porthos.add_argument("--score", action="store_true",
                                help="append the 1:1 verification score")
    export_porthos.add_argument("--gallery-rows", type=int, default=1024)
    export_porthos.add_argument(
        "--gallery-row", type=int, default=0,
        help="the enrolled row the score reads. The client declares the "
             "identity, so this index is public (default %(default)s)")
    export_porthos.set_defaults(func=_export_porthos)

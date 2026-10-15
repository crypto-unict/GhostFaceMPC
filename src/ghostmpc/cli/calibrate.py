"""Calibrating the overflow bounds of a manifest on real faces.

The bounds are what the QAT starting manifest carries: they decide which
truncations the fusion planner may defer.  They are measured on a calibration
set disjoint from the five verification datasets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..core.model import Manifest


def _calibrate_bounds(args: argparse.Namespace) -> int:
    import numpy as np

    from ..eval.bounds import (
        calibration_peaks,
        fusion_audit,
        layer_bounds,
        load_images,
        validation_margins,
    )

    manifest_path = Path(args.manifest)
    output = Path(args.output) if args.output else manifest_path
    if output.parent != manifest_path.parent:
        raise SystemExit("--output must sit beside the manifest it copies, "
                         "so the relative weights_file still resolves")
    manifest = Manifest.load(manifest_path)
    weights = np.load(manifest_path.parent / manifest.weights_file, mmap_mode="r")
    fractional_bits = manifest.quantization.fractional_bits
    if args.weight_fractional_bits is not None:
        if args.weight_fractional_bits < fractional_bits:
            raise SystemExit("weight fractional bits must be >= fractional bits")
        manifest.quantization.weight_fractional_bits = args.weight_fractional_bits

    images = load_images(Path(args.images), args.count)
    if not images:
        raise SystemExit(f"no calibration images in {args.images}")
    peaks = calibration_peaks(manifest_path, images)
    total_bits = manifest.quantization.activation_bits
    layers = layer_bounds(manifest, weights, peaks, margin_bits=args.margin_bits,
                          fractional_bits=fractional_bits,
                          total_bits=total_bits)
    bounds = {name: layer.bound for name, layer in layers.items()}
    report = fusion_audit(manifest, layers, fractional_bits=fractional_bits)
    report["calibration_images"] = len(images)
    report["margin_bits"] = args.margin_bits

    if args.validation_count:
        validation = load_images(Path(args.images), args.validation_count,
                                 offset=args.count)
        margins = validation_margins(manifest_path, bounds, validation)
        thin = {name: margin for name, margin in margins.items()
                if margin < args.reserved_bits}
        report["validation_images"] = len(validation)
        report["validation_margin_bits"] = margins
        report["validation_margin_minimum"] = min(margins.values(), default=None)
        if thin and not args.force:
            raise SystemExit(
                f"held-out faces leave under {args.reserved_bits} margin bits on "
                f"{len(thin)} layer(s): {sorted(thin)}; raise --margin-bits or --count")

    if not args.dry_run:
        manifest.overflow_bounds = bounds
        manifest.save(output)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for entry in report["layers"]:
        print(f"{entry['prelu']:<28} e_static={entry['static_bits']:>3} "
              f"e_calibrated={entry['calibrated_bits']:>3} "
              f"({entry['bound_source']}) "
              f"f_a={entry['alpha_bits_calibrated']} "
              f"{'fused' if entry['fused_calibrated'] else 'excluded'}"
              f"{' (new)' if entry['fused_calibrated'] and not entry['fused_static'] else ''}")
    print(json.dumps({
        "manifest": "(dry run, unwritten)" if args.dry_run else str(output),
        "report": str(report_path),
        "prelu_fused": f"{report['fused_calibrated']}/{report['prelu_total']}",
        "was": f"{report['fused_static']}/{report['prelu_total']}",
        "newly_fused_elements": report["newly_fused_elements"],
    }, indent=2))
    return 0


def register(commands: argparse._SubParsersAction) -> None:
    """Add this group's subcommands, in the order they are listed."""
    calibrate = commands.add_parser(
        "calibrate-bounds",
        help="measure overflow bounds on real faces and write them into the manifest")
    calibrate.add_argument("--manifest", required=True)
    calibrate.add_argument("--images", required=True,
                           help="directory of aligned faces or an InsightFace .bin")
    calibrate.add_argument("--count", type=int, default=32,
                           help="calibration images taken from the head of --images")
    calibrate.add_argument("--validation-count", type=int, default=32,
                           help="held-out images taken after the calibration ones; 0 disables the check")
    calibrate.add_argument("--margin-bits", type=int, default=2)
    calibrate.add_argument("--reserved-bits", type=float, default=1.0,
                           help="margin bits the held-out set must still leave on every layer")
    calibrate.add_argument("--output", help="defaults to the manifest itself")
    calibrate.add_argument("--report", default="results/calibration/bounds-audit.json")
    calibrate.add_argument("--weight-fractional-bits", type=int,
                           help="record a finer public-kernel grid in the profile")
    calibrate.add_argument("--dry-run", action="store_true",
                           help="audit only; leave the manifest untouched")
    calibrate.add_argument("--force", action="store_true",
                           help="write the bounds even if the held-out margin is thin")
    calibrate.set_defaults(func=_calibrate_bounds)

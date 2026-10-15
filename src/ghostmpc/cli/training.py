"""Quantization-aware training of the ring32/f6/f_w9 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _prepare_ms1mv3(args: argparse.Namespace) -> int:
    from ..training import ms1mv3

    summary = ms1mv3.build_val_pairs(
        ms1mv3.iter_webdataset(Path(args.shards)), Path(args.val_pairs))
    print(json.dumps({"val_pairs": args.val_pairs, **summary}, indent=2))
    return 0


def _qat_ring(args: argparse.Namespace) -> int:
    from ..training.qat_train import train_qat_ring

    feature_ops = (args.feature_ops.split(",")
                   if args.feature_ops else None)
    summary = train_qat_ring(
        Path(args.manifest), Path(args.teacher), Path(args.shards),
        Path(args.output), fractional_bits=args.fractional_bits,
        total_bits=args.total_bits, ring_bits=args.ring_bits,
        epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps, ema_decay=args.ema_decay,
        lambda_feat=args.lambda_feat,
        lambda_range=args.lambda_range, feature_ops=feature_ops,
        val_pairs=(Path(args.val_pairs) if args.val_pairs else None),
        validation_count=args.validation_count, eps_target=args.eps_target,
        eval_every=args.eval_every, checkpoint_every=args.checkpoint_every,
        seed=args.seed,
        resume_from=(Path(args.resume_from) if args.resume_from else None),
        max_seconds=args.max_seconds,
    )
    print(json.dumps({"finished": summary["finished"],
                      "step": summary["step"], "best": summary["best"]},
                     indent=2))
    return 0


def register(commands: argparse._SubParsersAction) -> None:
    """Add this group's subcommands, in the order they are listed."""
    prepare = commands.add_parser(
        "prepare-ms1mv3",
        help="build the MS1MV3 validation-split pairs used for model selection "
             "and bound calibration")
    prepare.add_argument("--shards", required=True,
                         help="MS1MV3 WebDataset shard directory")
    prepare.add_argument("--val-pairs", required=True,
                         help="output .bin, in the layout of the benchmark files")
    prepare.set_defaults(func=_prepare_ms1mv3)
    qat_ring = commands.add_parser(
        "qat-ring",
        help="quantization-aware fine-tuning on the ring-faithful forward pass")
    qat_ring.add_argument("--manifest", required=True,
                          help="calibrated ring32/f6/f_w9 starting manifest")
    qat_ring.add_argument("--teacher", required=True,
                          help="frozen float teacher checkpoint (.h5)")
    qat_ring.add_argument("--shards", required=True,
                          help="MS1MV3 WebDataset shard directory (train split)")
    qat_ring.add_argument("--output", required=True,
                          help="run directory (ckpt/, best/, final/, history.json)")
    qat_ring.add_argument("--val-pairs", default=None,
                          help="MS1MV3 faces used to measure the selection metric")
    qat_ring.add_argument("--validation-count", type=int, default=512,
                          help="held-out faces the embedding perturbation is measured on")
    qat_ring.add_argument("--eps-target", type=float, default=0.0021,
                          help="embedding perturbation the run aims for")
    qat_ring.add_argument("--fractional-bits", type=int, default=6)
    qat_ring.add_argument("--total-bits", type=int, default=15)
    qat_ring.add_argument("--ring-bits", type=int, default=32)
    # The published model: Adam at 5e-5 with 500 warm-up steps and cosine decay,
    # batch 128, EMA 0.999, two epochs of 36,392 steps (72,784 steps).
    qat_ring.add_argument("--epochs", type=int, default=2)
    qat_ring.add_argument("--steps-per-epoch", type=int, default=36_392)
    qat_ring.add_argument("--batch-size", type=int, default=128)
    qat_ring.add_argument("--learning-rate", type=float, default=5e-5)
    qat_ring.add_argument("--warmup-steps", type=int, default=500)
    qat_ring.add_argument("--ema-decay", type=float, default=0.999)
    qat_ring.add_argument("--lambda-feat", type=float, default=0.05)
    qat_ring.add_argument("--lambda-range", type=float, default=1e-4)
    qat_ring.add_argument("--feature-ops", default=None,
                          help="comma-separated op names for L_feat (auto if unset)")
    qat_ring.add_argument("--eval-every", type=int, default=18_196)
    qat_ring.add_argument("--checkpoint-every", type=int, default=2_000)
    qat_ring.add_argument("--seed", type=int, default=0)
    qat_ring.add_argument("--resume-from", default=None)
    qat_ring.add_argument("--max-seconds", type=float, default=None,
                          help="stop, checkpoint and exit 0 after this wall time")
    qat_ring.set_defaults(func=_qat_ring)

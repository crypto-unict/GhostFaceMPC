"""Face-verification accuracy of the float teacher and the quantized model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..eval.face_accuracy import DATASETS


def _expected(manifest: Path) -> dict[str, dict[str, float]]:
    """The accuracy recorded in the bundle's ``gates/``, per dataset."""
    expected: dict[str, dict[str, float]] = {}
    for path in sorted((manifest.parent / "gates").glob("*.json")):
        for row in json.loads(path.read_text()).get("results", []):
            expected[row["dataset"]] = {"teacher": row["reference"],
                                        "quantized": row["mpc"]}
    return expected


def _evaluate_accuracy(args: argparse.Namespace) -> int:
    from ..eval.face_accuracy import (
        _embeddings,
        _verification_accuracy,
        load_dataset,
        ring_embeddings,
    )
    from ..eval.reference import FAITHFUL_ORACLE_VERSION

    manifest = Path(args.manifest)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    teacher_model = None
    if not args.skip_teacher:
        import tensorflow as tf

        teacher_model = tf.keras.models.load_model(args.checkpoint, compile=False)
    expected = _expected(manifest)
    rows = []
    for dataset in args.datasets or DATASETS:
        encoded, labels = load_dataset(Path(args.dataset_directory), dataset)
        if args.pairs:
            encoded, labels = encoded[:2 * args.pairs], labels[:args.pairs]
        started = time.time()
        row: dict[str, object] = {"dataset": dataset, "pairs": len(labels),
                                  "oracle": FAITHFUL_ORACLE_VERSION}
        if teacher_model is not None:
            row["teacher"] = _verification_accuracy(
                _embeddings(teacher_model, encoded, args.batch_size), labels)
        row["quantized"] = _verification_accuracy(
            ring_embeddings(manifest, encoded, batch_size=args.batch_size,
                            workers=args.workers), labels)
        if "teacher" in row:
            row["drop_percentage_points"] = 100 * (row["teacher"] - row["quantized"])
        row["seconds"] = time.time() - started
        reference = expected.get(dataset)
        if reference is not None and not args.pairs:
            row["expected"] = reference
            # The oracle is deterministic, so the quantized row reproduces
            # exactly; the teacher row may move in the last pair with the
            # TensorFlow build it runs on.
            row["quantized_matches_expected"] = abs(
                row["quantized"] - reference["quantized"]) < 1e-9
        (output / f"{dataset}.json").write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)
        rows.append(row)
    summary = output / "accuracy.json"
    summary.write_text(json.dumps({
        "manifest": str(manifest), "checkpoint": args.checkpoint,
        "results": rows}, indent=2, sort_keys=True) + "\n")
    print(f"wrote {summary}")
    return 0


def register(commands: argparse._SubParsersAction) -> None:
    """Add this group's subcommands, in the order they are listed."""
    evaluate = commands.add_parser(
        "evaluate-accuracy",
        help="ten-fold verification accuracy of the float teacher and of the "
             "quantized model on the five benchmark datasets")
    evaluate.add_argument("--manifest", default="artifacts/model/manifest.json",
                          help="quantized model, evaluated on the ring oracle")
    evaluate.add_argument("--checkpoint",
                          default="artifacts/GhostFaceNet_W1.3_S1_ArcFace.h5",
                          help="float teacher, evaluated with TensorFlow")
    evaluate.add_argument("--dataset-directory", default="datasets/benchmarks")
    evaluate.add_argument("--dataset", dest="datasets", action="append",
                          choices=DATASETS,
                          help="evaluate only this dataset; repeatable, "
                               "defaults to all five")
    evaluate.add_argument("--output", default="results/accuracy")
    evaluate.add_argument("--workers", type=int, default=1,
                          help="processes the ring oracle is spread over")
    evaluate.add_argument("--batch-size", type=int, default=16)
    evaluate.add_argument("--pairs", type=int,
                          help="evaluate only the first N pairs (smoke test)")
    evaluate.add_argument("--skip-teacher", action="store_true",
                          help="evaluate only the quantized model")
    evaluate.set_defaults(func=_evaluate_accuracy)

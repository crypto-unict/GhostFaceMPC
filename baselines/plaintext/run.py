#!/usr/bin/env python3
"""The plaintext baseline: one server running the float model in the clear.

This is the TensorFlow row of the comparison table: the float32 teacher on one
server, no MPC, in a warm process (model construction and weight loading are
excluded, as the secure rows exclude their one-time public-weight load).  A
request is the embedding of one aligned face followed by the 1:1 verification
score against an enrolled row of a 1,024-row database held in the clear; the
reported wall time is the median over the measured requests.

The model runs as a traced ``tf.function`` -- what a served model executes --
under a total budget of ``--threads`` threads, which defaults to the 8 threads
per party of the secure rows.  TensorFlow fixes its thread pools at the first
op, so the measurement runs in a child process that owns them.

Usage::

    .venv/bin/python baselines/plaintext/run.py --output results/plaintext
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

CHECKPOINT = REPO / "artifacts" / "GhostFaceNet_W1.3_S1_ArcFace.h5"
DATASETS = REPO / "datasets" / "benchmarks"

#: The database the score reads: seeded rows, L2-normalized, in float32.
GALLERY_SEED = 0
EMBEDDING_SIZE = 512

MARKER = "===PLAINTEXT==="


def _summary(values: list[float]) -> dict[str, float]:
    return {"median": statistics.median(values), "min": min(values),
            "max": max(values), "stdev": statistics.stdev(values)}


def _worker(threads: int, gallery_rows: int, pools: int, warmups: int,
            trials: int) -> dict:
    import time

    import numpy as np
    import tensorflow as tf

    from ghostmpc.eval.face_accuracy import _decode, load_dataset

    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(1 if threads == 1 else 2)

    encoded, _ = load_dataset(DATASETS, "lfw")
    face = np.asarray(_decode(encoded[:1])[0], dtype=np.float32)
    model = tf.keras.models.load_model(CHECKPOINT, compile=False)
    request = tf.constant(face[None, ...])
    graph = tf.function(lambda batch: model(batch, training=False),
                        reduce_retracing=True)

    generator = np.random.default_rng(GALLERY_SEED)
    gallery = generator.normal(size=(gallery_rows, EMBEDDING_SIZE)).astype(np.float32)
    gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
    enrolled = gallery[0]

    def embedding(batch):
        return np.asarray(graph(batch))[0]

    def request_with_score(batch):
        vector = embedding(batch)
        return float(np.dot(vector, enrolled)), float(np.dot(vector, vector))

    def pooled(call) -> dict:
        walls: list[float] = []
        cpus: list[float] = []
        for _ in range(pools):
            for _ in range(warmups):
                call(request)
            for _ in range(trials):
                wall0, cpu0 = time.perf_counter(), time.process_time()
                call(request)
                walls.append(time.perf_counter() - wall0)
                cpus.append(time.process_time() - cpu0)
        return {"wall_seconds": _summary(walls), "cpu_seconds": _summary(cpus),
                "samples": len(walls)}

    return {
        "threads": threads,
        "execution": "graph",
        "regions": {"embedding": pooled(embedding),
                    "total": pooled(request_with_score)},
        "tensorflow": tf.__version__,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=REPO / "results" / "plaintext")
    parser.add_argument("--threads", type=int, default=8,
                        help="total thread budget of the server (default %(default)s)")
    parser.add_argument("--gallery-rows", type=int, default=1024)
    parser.add_argument("--pools", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        print(MARKER)
        print(json.dumps(_worker(args.threads, args.gallery_rows, args.pools,
                                 args.warmups, args.trials)))
        return 0

    environment = dict(os.environ, OMP_NUM_THREADS=str(args.threads),
                       TF_NUM_INTRAOP_THREADS=str(args.threads),
                       TF_CPP_MIN_LOG_LEVEL="2", CUDA_VISIBLE_DEVICES="")
    command = [sys.executable, str(Path(__file__).resolve()), "--worker",
               "--threads", str(args.threads),
               "--gallery-rows", str(args.gallery_rows),
               "--pools", str(args.pools), "--warmups", str(args.warmups),
               "--trials", str(args.trials)]
    completed = subprocess.run(command, capture_output=True, text=True,
                               env=environment, check=False)
    if completed.returncode != 0:
        sys.stderr.write(completed.stderr[-4000:])
        raise SystemExit("the plaintext worker failed")
    row = json.loads(completed.stdout.rsplit(MARKER, 1)[-1])
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "baseline": "TensorFlow float32, one server, no MPC, warm process",
        "checkpoint": str(CHECKPOINT.relative_to(REPO)),
        "gallery_rows": args.gallery_rows,
        "pool_design": {"pools": args.pools, "warmups": args.warmups,
                        "trials": args.trials},
        "environment": {"platform": platform.platform(),
                        "processor": platform.processor(),
                        "cpu_count": os.cpu_count(),
                        "python": platform.python_version()},
        **row,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    total = row["regions"]["total"]["wall_seconds"]["median"]
    print(f"plaintext request (embedding + score): {total:.4f} s median wall, "
          f"{args.threads} threads -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

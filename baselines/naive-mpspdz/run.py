#!/usr/bin/env python3
"""The naive MP-SPDZ baseline: one verification request, scalar codegen.

The program is emitted by ``naive_ghostmpc`` -- the straightforward scalar
emitter on a 64-bit ring at f = 20, with one truncation per linear output and
no fusion -- over the 236-operation manifest ``artifacts/legacy-scalar-236``
(the float checkpoint with only batch normalization folded where it directly
follows a convolution).  A request computes the embedding and the 1:1
verification score against one row of a 1,024-row secret-shared database, the
same request the GhostFaceMPC rows measure.

The program runs on ``replicated-ring-party.x`` and is measured by the same
harness as the GhostFaceMPC rows (``ghostmpc.benchmark``), so the two differ in
the emitted program alone.  It is compiled with ``./compile.py -R 64`` and runs
with MP-SPDZ's default (online) daBit generation.

Usage::

    .venv/bin/python baselines/naive-mpspdz/run.py --output results/naive
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from ghostmpc.benchmark.process import _pool_runtime, mpspdz_commit  # noqa: E402
from ghostmpc.benchmark.report import write_report  # noqa: E402
from ghostmpc.benchmark.runner import (  # noqa: E402
    _run_protocol_group,
    _trials_from_timer,
)
from ghostmpc.benchmark.stats import Trial, _size, parse_request_timer  # noqa: E402
from ghostmpc.core.model import Manifest  # noqa: E402
from ghostmpc.core.mpspdz_protocols import get_protocol  # noqa: E402
from ghostmpc.core.score import ScorePlan  # noqa: E402

MANIFEST = REPO / "artifacts" / "legacy-scalar-236" / "manifest.json"
WEIGHTS_SHA256 = "3062c7b346b15b645f19f314c71019c8e9abe44e28ed1917f51092f57edf5a55"
RING_BITS, FRACTIONAL_BITS = 64, 20

#: A naive ``sfix`` implementation keeps probe and database on the same grid,
#: because ``sfix`` is one global fixed-point format.
GALLERY_ROWS, EMBEDDING_SIZE, GALLERY_FRACTIONAL_BITS = 1024, 512, FRACTIONAL_BITS

#: The timers the emitted program brackets, shared with ``ghostmpc.codegen``.
TIMERS = {"end_to_end": 20_000, "weight_load": 20_001, "input_sharing": 20_002,
          "embedding": 20_003, "score": 20_004, "gallery_load": 20_005}


def _check_manifest() -> None:
    if not MANIFEST.is_file():
        raise SystemExit(
            f"{MANIFEST} is missing; fetch it with "
            f"'ghostmpc fetch-bundles --bundle legacy-scalar-236'")
    digest = json.loads(MANIFEST.read_text()).get("weights_sha256")
    if digest != WEIGHTS_SHA256:
        raise SystemExit(f"{MANIFEST} carries weights {digest}, expected {WEIGHTS_SHA256}")


def _score_plan() -> ScorePlan:
    return ScorePlan(
        gallery_rows=GALLERY_ROWS, embedding_size=EMBEDDING_SIZE,
        fractional_bits=FRACTIONAL_BITS,
        gallery_fractional_bits=GALLERY_FRACTIONAL_BITS,
        ring_bits=RING_BITS).validate()


def _gallery_values(plan: ScorePlan) -> list[int]:
    """A synthetic database, encoded exactly as the dealer command does.

    The communication trace does not depend on the values, so synthetic rows
    measure the same thing real enrolments would.
    """
    import numpy as np

    from ghostmpc.eval.score import encode_gallery

    generator = np.random.default_rng(0)
    rows = generator.normal(size=(plan.gallery_rows, plan.embedding_size))
    return [int(value) for value in encode_gallery(rows, plan).reshape(-1)]


def _generate(output: Path, *, threads: int, requests: int, warmups: int) -> None:
    """Emit the program through the naive emitter, in its own interpreter."""
    score = (f"{{'gallery_rows': {GALLERY_ROWS},"
             f" 'embedding_size': {EMBEDDING_SIZE},"
             f" 'gallery_fractional_bits': {GALLERY_FRACTIONAL_BITS}}}")
    call = "\n".join([
        "import naive_ghostmpc.codegen as naive",
        "from pathlib import Path",
        f"naive.generate(Path({str(MANIFEST)!r}), Path({str(output)!r}),"
        f" fractional_bits={FRACTIONAL_BITS}, parties=3, threads={threads},"
        f" requests={requests}, ring_bits={RING_BITS}, warmups={warmups},"
        f" measured_timer=True, score={score})",
    ])
    subprocess.run([sys.executable, "-c", call], check=True, cwd=REPO,
                   env={**os.environ, "PYTHONPATH": str(HERE)})


def _compile(mpspdz: Path, program_name: str, source: Path, *,
             skip_compile: bool) -> None:
    """``./compile.py -R 64``, reused only for a byte-identical source."""
    stamp = source.with_suffix(".compiled.sha256")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if stamp.is_file() and stamp.read_text().strip() == digest:
        print(f"reusing the compilation of {program_name}")
        return
    if skip_compile:
        raise SystemExit(f"--skip-compile: {program_name} was never compiled from this source")
    subprocess.run(["./compile.py", "-R", str(RING_BITS), program_name],
                   cwd=mpspdz, check=True)
    stamp.write_text(digest + "\n")


def _run_pools(mpspdz: Path, program_name: str, results: Path, *, pools: int,
               warmups: int, trials: int, port_base: int,
               timeout_seconds: float | None) -> list[Trial]:
    spec = get_protocol("replicated-ring-party")
    manifest = Manifest.load(MANIFEST)
    plan = _score_plan()
    gallery = _gallery_values(plan)
    measured: list[Trial] = []
    for index in range(pools):
        pool_id = f"naive-{index}"
        runtime = None
        try:
            runtime = _pool_runtime(
                mpspdz, results, pool_id, _size(manifest.input_shape),
                warmups + trials, RING_BITS, protocol=spec.name,
                gallery_size=plan.gallery_size, gallery_values=gallery,
                layout="input-binary")
            _, _, _, stderr = _run_protocol_group(
                mpspdz, spec, program_name, runtime, port_base + 10 * index,
                timeout_seconds, RING_BITS)
            for label in ("end_to_end", "input_sharing", "embedding", "score"):
                measured.extend(_trials_from_timer(
                    label, parse_request_timer(stderr, timer_id=TIMERS[label]),
                    max(1, trials), spec=spec, pool_id=pool_id))
            for label in ("weight_load", "gallery_load"):
                measured.extend(_trials_from_timer(
                    label, parse_request_timer(stderr, timer_id=TIMERS[label]),
                    1, spec=spec, pool_id=pool_id, phases=("total",)))
                measured[-1].phase = "setup"
        finally:
            if runtime is not None:
                shutil.rmtree(runtime, ignore_errors=True)
    return measured


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=str(REPO / "results" / "naive"))
    parser.add_argument("--mpspdz", default=str(REPO / "third_party" / "MP-SPDZ"))
    parser.add_argument("--threads", type=int, default=8,
                        help="threads per party (default %(default)s)")
    parser.add_argument("--pools", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--port-base", type=int, default=7000)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--skip-compile", action="store_true",
                        help="reuse an earlier compilation of the same source")
    args = parser.parse_args(argv)

    _check_manifest()
    mpspdz = Path(args.mpspdz).resolve()
    results = Path(args.output).resolve()
    results.mkdir(parents=True, exist_ok=True)
    program_name = "ghostmpc_naive_score"
    source = mpspdz / "Programs" / "Source" / f"{program_name}.mpc"
    source.parent.mkdir(parents=True, exist_ok=True)
    _generate(source, threads=args.threads, requests=args.warmups + args.trials,
              warmups=args.warmups)
    _compile(mpspdz, program_name, source, skip_compile=args.skip_compile)
    measured = _run_pools(mpspdz, program_name, results, pools=args.pools,
                          warmups=args.warmups, trials=args.trials,
                          port_base=args.port_base,
                          timeout_seconds=args.timeout_seconds)
    plan = _score_plan()
    csv_path, json_path = write_report(
        measured, Manifest.load(MANIFEST), results, mpspdz_commit(mpspdz),
        threads=args.threads,
        protocol_runtime_args=get_protocol("replicated-ring-party").runtime_args_for(RING_BITS),
        score={"gallery_rows": plan.gallery_rows,
               "embedding_size": plan.embedding_size,
               "fractional_bits": plan.fractional_bits,
               "gallery_fractional_bits": plan.gallery_fractional_bits,
               "realization": "sfix elementwise product, 512 secret "
                              "multiplications and a 512-wide truncation per "
                              "inner product",
               "opened": "nothing; the servers return share components of "
                         "(<e,g>, <e,e>)"})
    print(json.dumps({"csv": str(csv_path), "json": str(json_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

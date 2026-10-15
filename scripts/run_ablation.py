#!/usr/bin/env python3
"""The ablation of the optimizations: one verification request per step.

A cumulative ladder from the naive MP-SPDZ baseline to GhostFaceMPC, adding
one optimization of the paper's Optimizations section per row, on the
semi-honest protocol and the co-host setting:

    0  naive                MP-SPDZ naive baseline, as in the comparison table
    1  vectorization        the vectorized emitter, same ring64/f20 236-op model
    2  bn-folding           20 batch normalizations folded into their convs
    3  ring                 the 32-bit ring (f = 6, f_w = 9), the deployed model
    4  prelu-fusion         + Conv -> PReLU truncation fusion
    5  se-fusion            + SE Multiply -> 1x1 conv truncation fusion
    6  add-fusion           + residual Add truncation fusion
    7  two-summand          + two-summand rounding: the deployed program

Each row writes ``<results>/ablation/<n>-<name>/report.json`` (row 0 reuses
``<results>/naive/report.json``, the comparison table's run), and a run whose
report exists is skipped.  Before its first measurement every row from 1 on
passes a correctness gate: one request on the first LFW face, with the
embedding revealed and compared against the ring oracle (ring32 rows) or the
float model (ring64 rows); the result is ``gate.json`` beside the report.

Usage::

    .venv/bin/python scripts/run_ablation.py [--quick] [--results results]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

GHOSTMPC = Path(os.environ.get("GHOSTMPC", REPO / ".venv" / "bin" / "ghostmpc"))
PYTHON = Path(os.environ.get("PYTHON", sys.executable))
CHECKPOINT = REPO / "artifacts" / "GhostFaceNet_W1.3_S1_ArcFace.h5"
LEGACY = REPO / "artifacts" / "legacy-scalar-236" / "manifest.json"
FOLDED = REPO / "artifacts" / "ablation-ring64-f20" / "manifest.json"
MODEL = REPO / "artifacts" / "model" / "manifest.json"
EVERYTHING = ("prelu-fusion", "se-fusion", "add-fusion", "two-summand")

#: Cosine an ablated ring32 row must reach against the deployed ring oracle
#: (its roundings differ by an ulp where a fusion is off), and a ring64 row
#: against the float model.  The deployed row must be bit-exact.
RING32_COSINE, RING64_COSINE = 0.99, 0.9999


@dataclass(frozen=True)
class Row:
    name: str
    label: str
    manifest: Path | None = None
    disabled: tuple[str, ...] = ()
    #: The naive row runs the naive baseline's scalar emitter, unchanged.
    naive: bool = False


ROWS = (
    Row("naive", "MP-SPDZ naive", naive=True),
    Row("vectorization", "+ Vectorization", LEGACY, EVERYTHING),
    Row("bn-folding", "+ BN folding", FOLDED, EVERYTHING),
    Row("ring", "+ Ring reduction", MODEL, EVERYTHING),
    Row("prelu-fusion", "+ Conv-PReLU fusion", MODEL,
        ("se-fusion", "add-fusion", "two-summand")),
    Row("se-fusion", "+ SE-conv fusion", MODEL, ("add-fusion", "two-summand")),
    Row("add-fusion", "+ Residual-Add fusion", MODEL, ("two-summand",)),
    Row("two-summand", "+ Two-summand rounding", MODEL, ()),
)


def row_directory(results: Path, index: int, row: Row) -> Path:
    if index == 0:
        return results / "naive"
    return results / "ablation" / f"{index}-{row.name}"


def log(message: str) -> None:
    print(f"\n==> [ablation] {message}", flush=True)


def ensure_folded_manifest() -> None:
    """The deployed graph (20 BNs folded) on the naive baseline's ring64/f20."""
    if FOLDED.is_file():
        return
    log(f"exporting the folded ring64/f20 manifest to {FOLDED.parent}")
    subprocess.run([str(GHOSTMPC), "export", str(CHECKPOINT), str(FOLDED.parent),
                    "--profile", "ring64-f20"], check=True,
                   stdout=subprocess.DEVNULL)


# --- correctness gate -------------------------------------------------------

def _face() -> object:
    from ghostmpc.eval.face_accuracy import _decode, load_dataset

    images, _ = load_dataset(REPO / "datasets" / "benchmarks", "lfw")
    return _decode(images[:1])[0]


def _expected(manifest_path: Path, face: object) -> object:
    from ghostmpc.core.model import Manifest
    from ghostmpc.eval.reference import evaluate_numpy

    quantization = Manifest.load(manifest_path).quantization
    if quantization.ring_bits == 64:
        return evaluate_numpy(manifest_path, face)
    return evaluate_numpy(
        manifest_path, face,
        fixed_point=(quantization.fractional_bits, quantization.activation_bits),
        ring_bits=quantization.ring_bits)


def _revealed(stdout: str) -> list[float]:
    match = re.search(r"GHOSTMPC_TEST_OUTPUT \[([^\]]*)\]", stdout)
    if match is None:
        raise RuntimeError("the gate program printed no embedding")
    return [float(value) for value in match.group(1).split(",")]


def gate(row: Row, mpspdz: Path, directory: Path, *, threads: int,
         port_base: int) -> dict[str, object]:
    """Reveal the embedding of one real face and compare it with the oracle."""
    import numpy as np

    from ghostmpc.benchmark.process import _compile_program, _pool_runtime
    from ghostmpc.benchmark.runner import _run_protocol_group
    from ghostmpc.codegen import generate
    from ghostmpc.codegen.fusion import Optimizations
    from ghostmpc.core.model import Manifest
    from ghostmpc.core.mpspdz_protocols import get_protocol
    from ghostmpc.core.sharing import encode_fixed

    spec = get_protocol("replicated-ring-party")
    manifest_path = row.manifest
    quantization = Manifest.load(manifest_path).quantization
    ring_bits, fractional_bits = quantization.ring_bits, quantization.fractional_bits
    program_name = f"ghostmpc_ablation_gate_{row.name.replace('-', '_')}"
    source = mpspdz / "Programs" / "Source" / f"{program_name}.mpc"
    generate(manifest_path, source, threads=threads, requests=1,
             test_reveal=True, protocol=spec.name,
             optimizations=Optimizations.without(row.disabled))
    _compile_program(mpspdz, spec, program_name, ring_bits)
    face = _face()
    values = encode_fixed(np.asarray(face).reshape(-1), fractional_bits, ring_bits)
    runtime = _pool_runtime(mpspdz, directory, "gate", len(values), 1, ring_bits,
                            protocol=spec.name, values=values)
    try:
        _, _, stdout, _ = _run_protocol_group(
            mpspdz, spec, program_name, runtime, port_base, None, ring_bits)
    finally:
        shutil.rmtree(runtime, ignore_errors=True)
    actual = np.asarray(_revealed(stdout[0]))
    # A ring32 row is held against the deployed program's oracle, so the
    # deferred roundings show up as the distance from it.  A ring64 row is held
    # against the float model, evaluated on the folded graph: folding changes
    # the parameters, not the function, and the legacy manifest records no
    # tensor shapes for the float evaluator.
    reference_manifest = MODEL if ring_bits == 32 else FOLDED
    expected = np.asarray(_expected(reference_manifest, face)).reshape(-1)
    cosine = float(actual @ expected
                   / (np.linalg.norm(actual) * np.linalg.norm(expected)))
    ulp = 2.0 ** -fractional_bits
    bit_exact = bool(ring_bits == 32 and np.all(
        np.abs(actual - expected) < ulp / 2))
    threshold = RING32_COSINE if ring_bits == 32 else RING64_COSINE
    passed = cosine >= threshold and (bit_exact or row.disabled
                                      or ring_bits == 64)
    return {"row": row.name, "reference": ("ring oracle of the deployed model"
                                           if ring_bits == 32 else "float model"),
            "cosine": cosine, "threshold": threshold, "bit_exact": bit_exact,
            "max_abs_difference": float(np.max(np.abs(actual - expected))),
            "passed": bool(passed)}


# --- measurement ------------------------------------------------------------

def measure(index: int, row: Row, args: argparse.Namespace,
            directory: Path) -> None:
    if row.naive:
        pools, warmups, trials = args.naive_design
        command = [str(PYTHON), str(REPO / "baselines" / "naive-mpspdz" / "run.py"),
                   "--mpspdz", str(args.mpspdz), "--threads", str(args.threads),
                   "--pools", str(pools), "--warmups", str(warmups),
                   "--trials", str(trials),
                   "--port-base", str(args.port_base + 100 * index),
                   "--output", str(directory)]
    else:
        pools, warmups, trials = args.design
        command = [str(GHOSTMPC), "benchmark", "--manifest", str(row.manifest),
                   "--mpspdz", str(args.mpspdz),
                   "--protocol", "replicated-ring-party",
                   "--threads", str(args.threads), "--pools", str(pools),
                   "--warmups", str(warmups), "--trials", str(trials),
                   "--port-base", str(args.port_base + 100 * index),
                   "--network-profile", "co-host", "--output", str(directory),
                   "--score", "--gallery-rows", "1024"]
        for name in row.disabled:
            command += ["--disable", name]
    log(f"row {index} ({row.name}): {pools} pool(s) x ({warmups} + {trials})")
    subprocess.run(command, check=True, cwd=REPO)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=REPO / "results")
    parser.add_argument("--mpspdz", type=Path,
                        default=REPO / "third_party" / "MP-SPDZ")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--port-base", type=int, default=14000)
    parser.add_argument("--quick", action="store_true",
                        help="one pool of one warm-up and one measured request")
    parser.add_argument("--pools", type=int)
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--trials", type=int)
    parser.add_argument("--skip-gate", action="store_true",
                        help="measure without the correctness gate")
    parser.add_argument("--rows", nargs="*", choices=[row.name for row in ROWS],
                        help="run only these rows (default: all)")
    args = parser.parse_args(argv)
    args.mpspdz = args.mpspdz.resolve()
    args.results = args.results.resolve()

    def design(pools: int, warmups: int, trials: int) -> tuple[int, int, int]:
        if args.quick:
            pools, warmups, trials = 1, 1, 1
        return (args.pools or pools, args.warmups if args.warmups is not None
                else warmups, args.trials or trials)

    # The paper's designs: the semi-honest rows' and the naive baseline's.
    args.design = design(3, 5, 30)
    args.naive_design = design(3, 2, 5)
    ensure_folded_manifest()
    for index, row in enumerate(ROWS):
        if args.rows and row.name not in args.rows:
            continue
        directory = row_directory(args.results, index, row)
        if (directory / "report.json").is_file():
            log(f"row {index} ({row.name}): kept existing {directory}/report.json")
            continue
        directory.mkdir(parents=True, exist_ok=True)
        if index > 0 and not args.skip_gate:
            gate_path = directory / "gate.json"
            if not gate_path.is_file():
                log(f"row {index} ({row.name}): correctness gate")
                result = gate(row, args.mpspdz, directory, threads=args.threads,
                              port_base=args.port_base + 100 * index + 50)
                gate_path.write_text(json.dumps(result, indent=2) + "\n")
            result = json.loads(gate_path.read_text())
            log(f"row {index} ({row.name}): gate cosine {result['cosine']:.6f}, "
                f"bit-exact {result['bit_exact']}")
            if not result["passed"]:
                raise SystemExit(f"row {row.name} failed its correctness gate: "
                                 f"{result}")
        measure(index, row, args, directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

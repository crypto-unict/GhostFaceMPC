"""The switches of the ablation, and that they leave the deployed programs alone.

No MPC runtime is involved; the ablation's own correctness gate
(``scripts/run_ablation.py``) runs each step on MP-SPDZ.  Skipped when the
model bundles have not been fetched (``ghostmpc fetch-bundles``).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ghostmpc.codegen import generate
from ghostmpc.codegen.fusion import DEPLOYED, Optimizations
from ghostmpc.core.score import deployment_score_plan

MODEL = REPO / "artifacts" / "model" / "manifest.json"
LEGACY = REPO / "artifacts" / "legacy-scalar-236" / "manifest.json"
FOLDED = REPO / "artifacts" / "ablation-ring64-f20" / "manifest.json"
NAIVE = REPO / "baselines" / "naive-mpspdz"
EVERYTHING = tuple(Optimizations.SWITCHES)

#: The naive baseline program of the comparison table (3 x (2 + 5), 8 threads).
NAIVE_SHA256 = "0e9a5def4a0306da8618ba2e006f0b2926055284504d1ecf0eb6a1fd576b974a"


def _emit(manifest: Path, work: str, *, disabled=(), score=True) -> str:
    """The program of one ladder step; the score is the deployed ring32 one."""
    target = Path(work) / "program.mpc"
    plan = deployment_score_plan(1024, 512) if score else None
    generate(manifest, target, threads=8, requests=2, warmups=1,
             measured_timer=True, score=plan,
             optimizations=Optimizations.without(disabled))
    return target.read_text()


def _naive(work: str) -> str:
    target = Path(work) / "naive.mpc"
    score = {"gallery_rows": 1024, "embedding_size": 512,
             "gallery_fractional_bits": 20}
    call = (
        "import naive_ghostmpc.codegen as naive\nfrom pathlib import Path\n"
        f"naive.generate(Path({str(LEGACY)!r}), Path({str(target)!r}),"
        " fractional_bits=20, parties=3, threads=8, requests=7, ring_bits=64,"
        f" warmups=2, measured_timer=True, score={score!r})")
    subprocess.run([sys.executable, "-c", call], check=True, cwd=REPO,
                   env={"PYTHONPATH": str(NAIVE)})
    return target.read_text()


class SwitchTests(unittest.TestCase):
    def test_the_default_is_the_deployed_set(self) -> None:
        self.assertEqual(Optimizations(), DEPLOYED)
        self.assertEqual(Optimizations.without([]), DEPLOYED)
        self.assertEqual(DEPLOYED.disabled(), [])

    def test_without_names_the_switches_it_turns_off(self) -> None:
        options = Optimizations.without(["two-summand", "se-fusion"])
        self.assertFalse(options.two_summand)
        self.assertFalse(options.fuse_multiply)
        self.assertTrue(options.fuse_prelu and options.fuse_add)
        self.assertEqual(options.disabled(), ["se-fusion", "two-summand"])

    def test_an_unknown_switch_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            Optimizations.without(["bn-folding"])


@unittest.skipUnless(MODEL.is_file(), "model bundle not fetched")
class DeployedModelTests(unittest.TestCase):
    def test_the_two_summand_switch_selects_the_rounding_kernel(self) -> None:
        with tempfile.TemporaryDirectory() as work:
            deployed = _emit(MODEL, work)
            three = _emit(MODEL, work, disabled=["two-summand"])
        self.assertIn("shrsi2", deployed)
        self.assertNotIn("shrsi2", three)
        two_summands = r"split_to_n_summands\(\w+, 2\)"
        self.assertRegex(deployed, two_summands)
        self.assertNotRegex(three, two_summands)

    def test_each_fusion_switch_empties_its_plan_alone(self) -> None:
        import numpy as np

        from ghostmpc.codegen.fusion import (
            _prelu_alpha_magnitudes,
            manifest_bounds,
            plan_fusions,
        )
        from ghostmpc.core.model import Manifest

        manifest = Manifest.load(MODEL)
        weights = np.load(MODEL.parent / manifest.weights_file, mmap_mode="r")

        def counts(*disabled: str) -> tuple[int, int, int]:
            fusion, prelu, multiply = plan_fusions(
                manifest, manifest_bounds(manifest), fractional_bits=6,
                ring_bits=32, kernel_bits=9,
                alpha_magnitudes=_prelu_alpha_magnitudes(manifest, weights),
                optimizations=Optimizations.without(disabled))
            return len(prelu), len(multiply), len(fusion.adds)

        prelu, multiply, adds = counts()
        self.assertTrue(prelu and multiply and adds)
        self.assertEqual(counts("prelu-fusion"), (0, multiply, adds))
        self.assertEqual(counts("se-fusion"), (prelu, 0, adds))
        self.assertEqual(counts("prelu-fusion", "se-fusion", "add-fusion"),
                         (0, 0, 0))

    def test_every_ladder_step_emits_a_consistent_program(self) -> None:
        # A plan the emitter and the weight stream disagree on fails to
        # resolve a weight offset, so emitting is the consistency check.
        ladder = [EVERYTHING[index:] for index in range(len(EVERYTHING) + 1)]
        with tempfile.TemporaryDirectory() as work:
            for disabled in ladder:
                with self.subTest(disabled=disabled):
                    self.assertIn("RING_BITS = 32", _emit(MODEL, work,
                                                          disabled=disabled))


@unittest.skipUnless(LEGACY.is_file(), "legacy-scalar-236 bundle not fetched")
class Ring64Tests(unittest.TestCase):
    def test_a_ring64_program_needs_the_three_summand_kernel(self) -> None:
        with tempfile.TemporaryDirectory() as work:
            with self.assertRaises(ValueError):
                _emit(LEGACY, work, disabled=EVERYTHING[:-1], score=False)
            source = _emit(LEGACY, work, disabled=EVERYTHING, score=False)
        self.assertIn("RING_BITS = 64", source)
        self.assertEqual(source.count("_affine_bias = "), 21)

    def test_the_naive_baseline_program_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as work:
            source = _naive(work)
        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(),
                         NAIVE_SHA256)
        self.assertNotIn("prefetchdabit", source)


@unittest.skipUnless(FOLDED.is_file() and MODEL.is_file() and LEGACY.is_file(),
                     "ablation manifest not exported (scripts/run_ablation.py)")
class FoldedManifestTests(unittest.TestCase):
    @staticmethod
    def _graph(path: Path) -> list[tuple]:
        return [(op["name"], op["kind"], tuple(op["inputs"]))
                for op in json.loads(path.read_text())["ops"]]

    def test_folding_removes_exactly_twenty_normalizations(self) -> None:
        folded, legacy = self._graph(FOLDED), self._graph(LEGACY)
        self.assertEqual(folded, self._graph(MODEL))
        self.assertEqual(len(legacy) - len(folded), 20)
        names = {name for name, _, _ in folded}
        self.assertEqual(
            [name for name, kind, _ in legacy if name not in names],
            [name for name, kind, _ in legacy
             if kind == "BatchNormalization" and name not in names])


if __name__ == "__main__":
    unittest.main()

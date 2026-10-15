import tempfile
import unittest
from pathlib import Path

import numpy as np

from ghostmpc.codegen.fusion import _fusion_plan, _prelu_fusion
from ghostmpc.core.model import Manifest, Op, TensorRef
from ghostmpc.eval.bounds import (
    calibrated_bounds,
    calibration_peaks,
    fusion_audit,
    layer_bounds,
    validation_margins,
)

PRODUCTION_MANIFEST = Path(
    "artifacts/model-ring64-f16-b0/manifest.json")


class CalibratedBoundTests(unittest.TestCase):
    """Bounds measured on activations, with the clamps that keep them sound."""

    FRACTIONAL_BITS = 20

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        # One 1x1 tap of 2.0 and a bias of 0.5: every bound is hand-checkable.
        self.weights = np.array([2.0, 0.5, 0.25], dtype=np.float64)
        np.save(root / "weights.npy", self.weights.astype(np.float32))
        self.manifest = Manifest(
            3, "test", "url", "c", [2, 2, 1], [2, 2, 1],
            [Op("conv", "Conv2D", ["input"], [2, 2, 1],
                {"kernel_size": [1, 1], "strides": [1, 1], "padding": "SAME",
                 "filters": 1}, ["conv:0", "conv:1"]),
             Op("act", "PReLU", ["conv"], [2, 2, 1], {}, ["act:0"])],
            [TensorRef("conv:0", [1, 1, 1, 1], 0, 1),
             TensorRef("conv:1", [1], 1, 1), TensorRef("act:0", [1, 1, 1], 2, 1)],
            "weights.npy", "w",
        )
        self.path = root / "manifest.json"
        self.manifest.save(self.path)

    def _bounds(self, peaks, margin_bits=2):
        return layer_bounds(self.manifest, self.weights, peaks,
                            margin_bits=margin_bits,
                            fractional_bits=self.FRACTIONAL_BITS, total_bits=31)

    def test_peaks_are_the_pre_activation_maximum(self):
        images = [np.array([[[-1.0], [0.0]], [[1.0], [0.5]]])]
        # The conv gives [-1.5, 0.5, 2.5, 1.5]; the PReLU must not shrink it.
        self.assertAlmostEqual(
            calibration_peaks(self.path, images)["conv"], 2.5)

    def test_bound_is_the_peak_times_the_margin(self):
        bound = self._bounds({"conv": 2.5})["conv"]
        self.assertEqual(bound.bound, 10.0)
        self.assertEqual(bound.source, "calibrated")
        # |x| < 2^11 admitted by the static rule, times the 2.0 tap, plus bias.
        self.assertEqual(bound.static, 2.0 * 2048 + 0.5)

    def test_bound_never_drops_below_the_bias_or_one(self):
        self.assertEqual(self._bounds({"conv": 0.0})["conv"].bound, 1.0)
        self.assertEqual(
            self._bounds({"conv": 0.05}, margin_bits=1)["conv"].bound, 1.0)
        self.weights[1] = 3.0
        self.assertEqual(self._bounds({"conv": 0.0})["conv"].bound, 3.0)

    def test_static_bound_wins_when_calibration_gains_under_two_bits(self):
        # Static is 4096.5 (13 bits); a peak of 512 calibrates to 2048 (11), a
        # two-bit gain, while 1024 calibrates to 4096 (12) and is not worth it.
        self.assertEqual(self._bounds({"conv": 512.0})["conv"].source, "calibrated")
        kept = self._bounds({"conv": 1024.0})["conv"]
        self.assertEqual(kept.source, "static")
        self.assertEqual(kept.bound, kept.static)

    def test_calibrated_bounds_runs_the_reference_end_to_end(self):
        images = [np.array([[[-1.0], [0.0]], [[1.0], [0.5]]])]
        self.assertEqual(calibrated_bounds(self.path, self.weights, images,
                                           fractional_bits=self.FRACTIONAL_BITS),
                         {"conv": 10.0})

    def test_validation_measures_the_margin_the_held_out_set_leaves(self):
        # The conv peaks at 2.5 here, which is two margin bits under 10.0.
        images = [np.array([[[0.0], [0.0]], [[1.0], [0.0]]])]
        self.assertAlmostEqual(
            validation_margins(self.path, {"conv": 10.0}, images)["conv"], 2.0)

    def test_validation_keeps_margin_when_the_held_out_peak_grows(self):
        # A held-out peak above the calibration one is not a failure while the
        # bound keeps room: this one is 4.5 against 10.0, still 1.15 bits clear
        # of the reserved bit the CLI gate demands.
        thin = [np.array([[[0.0], [0.0]], [[2.0], [0.0]]])]
        margin = validation_margins(self.path, {"conv": 10.0}, thin)["conv"]
        self.assertAlmostEqual(margin, np.log2(10.0 / 4.5))
        self.assertGreater(margin, 1.0)

    def test_validation_margin_goes_under_one_bit_when_the_bound_is_grazed(self):
        gone = [np.array([[[0.0], [0.0]], [[4.0], [0.0]]])]
        margin = validation_margins(self.path, {"conv": 10.0}, gone)["conv"]
        self.assertAlmostEqual(margin, np.log2(10.0 / 8.5))
        self.assertLess(margin, 1.0)

    def test_audit_reports_the_layers_the_tighter_bound_unlocks(self):
        report = fusion_audit(self.manifest, self._bounds({"conv": 2.5}),
                             fractional_bits=self.FRACTIONAL_BITS)
        # 62 - 2*20 - 13 static bits leaves 9 alpha bits; the calibrated 4 bits
        # leave 18, so the pair moves from coarse to full-precision fusion.
        self.assertEqual(report["fused_static"], 1)
        self.assertEqual(report["fused_calibrated"], 1)
        self.assertEqual(report["layers"][0]["alpha_bits_static"], 9)
        self.assertEqual(report["layers"][0]["alpha_bits_calibrated"], 18)
        self.assertEqual(report["layers"][0]["elements"], 4)


class ProductionPlanTests(unittest.TestCase):
    """The whole point of the phase: every PReLU of the shipped model fuses."""

    @unittest.skipUnless(PRODUCTION_MANIFEST.is_file(), "exported manifest absent")
    def test_realistic_bounds_fuse_every_prelu_with_full_alpha(self):
        manifest = Manifest.load(PRODUCTION_MANIFEST)
        # A five-bit accumulator is what calibration measures on real faces.
        manifest.overflow_bounds = {
            op.name: 32.0 for op in manifest.ops
            if op.kind in {"Conv2D", "DepthwiseConv2D"} and op.tensors}
        plan = _prelu_fusion(manifest, None, 20,
                                     _fusion_plan(manifest))
        prelus = [op.name for op in manifest.ops if op.kind == "PReLU"]
        self.assertEqual(set(plan), set(prelus))
        self.assertTrue(all(fusion.alpha_bits >= 16 for fusion in plan.values()))


if __name__ == "__main__":
    unittest.main()

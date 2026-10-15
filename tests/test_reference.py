import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from ghostmpc.core.model import Manifest, Op, TensorRef
from ghostmpc.eval.reference import evaluate_numpy


class ReferenceTests(unittest.TestCase):
    def test_conv_prelu_and_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "weights.npy", np.array([2.0, 0.5, 0.25], dtype=np.float32))
            manifest = Manifest(
                2, "test", "url", "c", [2, 2, 1], [1],
                [Op("conv", "Conv2D", ["input"], [2, 2, 1],
                    {"kernel_size": [1, 1], "strides": [1, 1], "padding": "SAME", "filters": 1},
                    ["conv:0", "conv:1"]),
                 Op("act", "PReLU", ["conv"], [2, 2, 1], {}, ["act:0"]),
                 Op("pool", "GlobalAveragePooling2D", ["act"], [1])],
                [TensorRef("conv:0", [1, 1, 1, 1], 0, 1),
                 TensorRef("conv:1", [1], 1, 1), TensorRef("act:0", [1, 1, 1], 2, 1)],
                "weights.npy", "w",
            )
            manifest.save(root / "manifest.json")
            value = np.array([[[-1.0], [0.0]], [[1.0], [2.0]]])
            # Conv gives [-1.5, .5, 2.5, 4.5], then alpha=.25 on the negative.
            self.assertAlmostEqual(float(evaluate_numpy(root / "manifest.json", value)[0]), 1.78125)

    def _fixture(self, root: Path) -> Path:
        rng = np.random.default_rng(4)
        np.save(root / "weights.npy", rng.uniform(-0.4, 0.4, size=27).astype(np.float64))
        manifest = Manifest(
            2, "test", "url", "c", [2, 2, 3], [2, 2, 3],
            [Op("conv", "Conv2D", ["input"], [2, 2, 3],
                {"kernel_size": [1, 1], "strides": [1, 1], "padding": "SAME",
                 "filters": 3}, ["conv:0", "conv:1"]),
             Op("act", "PReLU", ["conv"], [2, 2, 3], {"shared_axes": [1, 2]},
                ["act:0"])],
            [TensorRef("conv:0", [1, 1, 3, 3], 0, 9),
             TensorRef("conv:1", [3], 9, 3),
             TensorRef("act:0", [3], 12, 3)], "weights.npy", "w",
            overflow_bounds={"conv": 4.0})
        manifest.save(root / "manifest.json")
        return root / "manifest.json"

    def test_fixed_point_mirror_tracks_the_exact_interpreter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._fixture(root)
            value = np.random.default_rng(9).uniform(-1, 1, size=(2, 2, 3))
            exact = evaluate_numpy(path, value)
            fine = evaluate_numpy(path, value, fixed_point=(20, 31))
            # A 2^-20 grid folded through one conv->PReLU stays inside a few ulp.
            self.assertLess(float(np.max(np.abs(exact - fine))), 1e-4)

    def test_fixed_point_mirror_snaps_to_the_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._fixture(root)
            value = np.random.default_rng(1).uniform(-1, 1, size=(2, 2, 3))
            coarse = evaluate_numpy(path, value, fixed_point=(6, 15))
            scaled = coarse * (1 << 6)
            np.testing.assert_allclose(scaled, np.rint(scaled), atol=1e-9)

    def test_fixed_point_oracle_supports_an_intermediate_ring_width(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._fixture(root)
            value = np.random.default_rng(2).uniform(-1, 1, size=(2, 2, 3))
            wide = evaluate_numpy(path, value, fixed_point=(6, 15), ring_bits=64)
            intermediate = evaluate_numpy(path, value, fixed_point=(6, 15),
                                          ring_bits=40)
            np.testing.assert_array_equal(intermediate, wide)

    def test_fixed_point_oracle_quantizes_public_kernel_before_convolution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Just below half an f6 ulp: the public writer emits integer zero.
            np.save(root / "weights.npy", np.array([0.49 / 64], dtype=np.float64))
            manifest = Manifest(
                2, "test", "url", "c", [1, 1, 1], [1],
                [Op("conv", "Conv2D", ["input"], [1, 1, 1],
                    {"kernel_size": [1, 1], "strides": [1, 1],
                     "padding": "SAME", "filters": 1}, ["conv:0"]),
                 Op("embedding", "Flatten", ["conv"], [1])],
                [TensorRef("conv:0", [1, 1, 1, 1], 0, 1)],
                "weights.npy", "w")
            manifest.save(root / "manifest.json")
            value = np.ones((1, 1, 1), dtype=np.float64)
            self.assertGreater(float(evaluate_numpy(
                root / "manifest.json", value)[0]), 0)
            self.assertEqual(float(evaluate_numpy(
                root / "manifest.json", value, fixed_point=(6, 15))[0]), 0)

    def test_separate_finer_kernel_scale_recovers_sub_f6_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "weights.npy", np.array([0.49 / 64], dtype=np.float64))
            manifest = Manifest(
                2, "test", "url", "c", [1, 1, 1], [1],
                [Op("conv", "Conv2D", ["input"], [1, 1, 1],
                    {"kernel_size": [1, 1], "strides": [1, 1],
                     "padding": "SAME", "filters": 1}, ["conv:0"]),
                 Op("embedding", "Flatten", ["conv"], [1])],
                [TensorRef("conv:0", [1, 1, 1, 1], 0, 1)],
                "weights.npy", "w")
            manifest.save(root / "manifest.json")
            value = np.ones((1, 1, 1), dtype=np.float64)
            baseline = evaluate_numpy(
                root / "manifest.json", value, fixed_point=(6, 15))
            finer = evaluate_numpy(
                root / "manifest.json", value, fixed_point=(6, 15),
                weight_fractional_bits=8)
            self.assertEqual(float(baseline[0]), 0)
            self.assertEqual(float(finer[0]), 1 / 64)

    def test_separate_kernel_scale_rejects_unsafe_declared_width(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._fixture(root)
            value = np.zeros((2, 2, 3), dtype=np.float64)
            with self.assertRaises(ValueError):
                evaluate_numpy(path, value, fixed_point=(6, 15),
                               weight_fractional_bits=17)

    def test_batched_ring_oracle_is_bit_exact_with_single_image_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._fixture(root)
            images = np.random.default_rng(13).uniform(-1, 1, size=(3, 2, 2, 3))
            batched = evaluate_numpy(path, images, fixed_point=(6, 15))
            singles = np.stack([
                evaluate_numpy(path, image, fixed_point=(6, 15))
                for image in images
            ])
            np.testing.assert_array_equal(batched, singles)

    def test_ring_oracle_reuses_prepared_public_weights_between_batches(self):
        from ghostmpc.eval import reference

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._fixture(root)
            image = np.zeros((2, 2, 3), dtype=np.float64)
            reference._cached_ring_static.cache_clear()
            with mock.patch.object(
                    reference, "_build_ring_static",
                    wraps=reference._build_ring_static) as build:
                evaluate_numpy(path, image, fixed_point=(6, 15))
                evaluate_numpy(path, image, fixed_point=(6, 15))
            self.assertEqual(build.call_count, 1)

    def test_fixed_point_mirror_raises_on_overflow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._fixture(root)
            value = np.full((2, 2, 3), 100.0)
            with self.assertRaises(OverflowError):
                evaluate_numpy(path, value, fixed_point=(12, 15))


if __name__ == "__main__":
    unittest.main()

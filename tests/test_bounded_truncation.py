"""Bounded-width rounding on every linear frontier.

The exact split-``k`` truncation decomposes only the declared low bits of an
accumulator.  These tests pin which frontiers take it: the plain convolution,
the fused residual addition and the pooling all do, unless the declaration
already fills the ring.
"""

from __future__ import annotations

import unittest

from ghostmpc.codegen import render
from ghostmpc.codegen.naming import _bounded_split, _kernel_truncation
from ghostmpc.core.model import Manifest, Op, TensorRef

FRACTIONAL_BITS = 6
KERNEL_BITS = 9
TOTAL_BITS = 15


def _ring32(manifest: Manifest) -> str:
    return render(manifest, FRACTIONAL_BITS, total_bits=TOTAL_BITS,
                  kernel_bits=KERNEL_BITS, overflow_bounds=manifest.overflow_bounds)


class BoundedCoverageTests(unittest.TestCase):
    def test_the_fused_add_frontier_takes_the_bounded_split(self):
        manifest = Manifest(
            3, "test", "url", "sum", [1, 1, 2], [1, 1, 2],
            [Op("left", "Conv2D", ["input"], [1, 1, 2],
                {"kernel_size": [1, 1], "strides": [1, 1], "padding": "SAME",
                 "filters": 2}, ["left:0", "left:1"]),
             Op("right", "Conv2D", ["input"], [1, 1, 2],
                {"kernel_size": [1, 1], "strides": [1, 1], "padding": "SAME",
                 "filters": 2}, ["right:0", "right:1"]),
             Op("sum", "Add", ["left", "right"], [1, 1, 2])],
            [TensorRef("left:0", [1, 1, 2, 2], 0, 4),
             TensorRef("left:1", [2], 4, 2),
             TensorRef("right:0", [1, 1, 2, 2], 6, 4),
             TensorRef("right:1", [2], 10, 2)], "weights.npy", "w",
            overflow_bounds={"left": 8.0, "right": 8.0})
        source = _ring32(manifest)
        self.assertIn("trunc_round_nearest_bounded(v_sum_raw.get_vector()", source)
        self.assertNotIn("truncate_generic(v_sum_raw", source)

    def test_the_pooling_frontier_takes_the_bounded_split(self):
        manifest = Manifest(
            3, "test", "url", "sum", [2, 2, 3], [3],
            [Op("conv", "Conv2D", ["input"], [2, 2, 3],
                {"kernel_size": [1, 1], "strides": [1, 1], "padding": "SAME",
                 "filters": 3}, ["conv:0"]),
             Op("pool", "GlobalAveragePooling2D", ["conv"], [3], {})],
            [TensorRef("conv:0", [1, 1, 3, 3], 0, 9)], "weights.npy", "w",
            overflow_bounds={"conv": 8.0})
        self.assertIn("trunc_round_nearest_bounded(\n        v_pool_scaled",
                      _ring32(manifest))

    def test_a_truncand_filling_the_ring_keeps_the_generic_truncring(self):
        # trunc_round_nearest_bounded declares k + 1 bits, so a truncand
        # already filling the ring splits the whole ring anyway.
        self.assertTrue(_bounded_split(24, 32))
        self.assertFalse(_bounded_split(31, 32))


class DeclaredLengthTests(unittest.TestCase):
    def test_a_finer_kernel_grid_states_its_length_and_shift(self):
        self.assertEqual(_kernel_truncation(FRACTIONAL_BITS, KERNEL_BITS),
                         {"length": "sfix.k + 9", "shift": "9"})

    def test_the_activation_grid_keeps_the_symbolic_default(self):
        self.assertEqual(_kernel_truncation(FRACTIONAL_BITS, FRACTIONAL_BITS), {})


if __name__ == "__main__":
    unittest.main()

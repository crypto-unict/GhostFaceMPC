"""The two-summand rounding kernel of the semi-honest profile.

The kernel computes the same integer as the three-summand bounded truncation,
over the two-summand regrouping of the replicated sharing, so it pays two bit
injections instead of four and needs no Wallace 3:2 compressor.  These tests
hold it to that claim on both sides -- the integer it produces and the program
it emits.
"""

from __future__ import annotations

import unittest

import numpy as np
from kernel_model import (
    bounded_round_nearest_from_shares,
    exhaustive_small_ring,
    two_summand_components,
)

from ghostmpc.codegen import render
from ghostmpc.core.model import Manifest, Op, TensorRef

RING32 = {"fractional_bits": 6, "total_bits": 15, "kernel_bits": 9}


def _manifest() -> Manifest:
    return Manifest(
        3, "test", "url", "sum", [2, 2, 4], [2, 2, 3],
        [Op("conv", "Conv2D", ["input"], [2, 2, 3],
            {"kernel_size": [1, 1], "strides": [1, 1], "padding": "SAME",
             "filters": 3}, ["conv:0", "conv:1"]),
         Op("act", "PReLU", ["conv"], [2, 2, 3], {"shared_axes": [1, 2]},
            ["act:0"])],
        [TensorRef("conv:0", [1, 1, 4, 3], 0, 12),
         TensorRef("conv:1", [3], 12, 3),
         TensorRef("act:0", [3], 15, 3)], "weights.npy", "weights",
        overflow_bounds={"conv": 2.0 ** 20},
    )


class TwoSummandIdentity(unittest.TestCase):
    """The kernel computes the deployed integer, over every sharing."""

    def test_regrouping_matches_the_split2_decomposition(self):
        # Party i holds (x_i, x_{i-1}), so P0 alone knows x_0 + x_2 and x_1 is
        # the summand P1 and P2 share.
        first, second = two_summand_components((7, 11, 13), ring_bits=8)
        self.assertEqual((first, second), (20, 11))
        self.assertEqual((first + second) % 256, (7 + 11 + 13) % 256)

    def test_exhaustive_small_ring_over_every_sharing(self):
        # Every declared width, every shift, every representable value and
        # every three-share tuple that represents it.
        for ring_bits in (5, 6):
            with self.subTest(ring_bits=ring_bits):
                three = exhaustive_small_ring(ring_bits, summands=3)
                two = exhaustive_small_ring(ring_bits, summands=2)
                self.assertTrue(two["passed"])
                self.assertEqual(two["cases"], three["cases"])

    def test_agrees_with_the_three_summand_kernel_on_random_sharings(self):
        rng = np.random.default_rng(62)
        ring_bits, modulus = 32, 1 << 32
        for k, m in ((24, 9), (21, 9), (30, 15), (16, 3), (31, 9)):
            bound = 1 << (k - 1)
            for _ in range(200):
                value = int(rng.integers(-bound, bound))
                first = int(rng.integers(0, modulus))
                second = int(rng.integers(0, modulus))
                shares = (first, second, value - first - second)
                common = {"ring_bits": ring_bits, "k": k, "m": m}
                self.assertEqual(
                    bounded_round_nearest_from_shares(
                        shares, summands=2, **common),
                    bounded_round_nearest_from_shares(
                        shares, summands=3, **common),
                    msg=f"k={k} m={m} value={value}")


class TwoSummandEmitter(unittest.TestCase):
    def test_emits_the_two_summand_helpers_and_the_shrsi2_opcode(self):
        source = render(_manifest(), **RING32)
        self.assertIn("from Compiler.instructions import shrsi2", source)
        self.assertIn("def _trunc_floor_split2", source)
        self.assertIn("value.split_to_n_summands(split_k, 2)", source)
        # The three-summand definitions stay reachable; the block replaces the
        # entry points, not the identity they implement.
        self.assertIn("_generic_truncation = truncate_generic", source)
        self.assertIn("value.split_to_n_summands(split_k, 3)", source)

    def test_the_sign_follows_the_kernel(self):
        source = render(_manifest(), **RING32)
        self.assertIn("columns = value.split_to_n_summands(k, 2)", source)
        self.assertEqual(source.count("is_negative("), 2)

    def test_the_malicious_profile_keeps_three_summands(self):
        source = render(_manifest(), protocol="ps-rep-ring-party", **RING32)
        self.assertNotIn("_trunc_floor_split2", source)
        self.assertNotIn("split_to_n_summands(k, 2)", source)
        self.assertIn("value.split_to_n_summands(split_k, 3)", source)


if __name__ == "__main__":
    unittest.main()

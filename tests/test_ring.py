"""Integer ring oracle and the conv->PReLU fused-truncation length fix.

The oracle reproduces the wrap, sign and nearest rounding of MP-SPDZ's
``TruncRing`` so a declared truncation length can be checked against the real
protocol.  The frontier the plan cares about is a fused conv->PReLU whose slope
has ``|alpha| > 1``: sizing its truncation from ``alpha_bits`` alone understates
the product and wraps a large accumulator, which these tests pin down.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from ghostmpc.core import ring


def _exact_nearest(product: int, shift: int) -> int:
    """Nearest truncation with no ring wrap, for comparison."""
    return (product + (1 << (shift - 1))) >> shift


class ToSignedTests(unittest.TestCase):
    def test_recenters_the_ring(self):
        self.assertEqual(ring.to_signed(0, 32), 0)
        self.assertEqual(ring.to_signed((1 << 31) - 1, 32), (1 << 31) - 1)
        # One past the largest positive element aliases to the smallest.
        self.assertEqual(ring.to_signed(1 << 31, 32), -(1 << 31))
        self.assertEqual(ring.to_signed(-(1 << 31), 32), -(1 << 31))
        self.assertEqual(ring.to_signed((1 << 31) + 5, 32), -(1 << 31) + 5)

    def test_array_wraps_elementwise(self):
        values = np.array([0, (1 << 15), (1 << 15) - 1, -(1 << 15)], dtype=object)
        out = ring.to_signed(values, 16)
        self.assertEqual(list(out), [0, -(1 << 15), (1 << 15) - 1, -(1 << 15)])


class TruncNearestTests(unittest.TestCase):
    def test_rounds_half_toward_positive_infinity(self):
        # shift 4 => divide by 16; the emitter adds 2^(shift-1) then floors.
        self.assertEqual(ring.trunc_nearest(8, 20, 4, ring_bits=32), 1)    # +0.5
        self.assertEqual(ring.trunc_nearest(-8, 20, 4, ring_bits=32), 0)   # -0.5
        self.assertEqual(ring.trunc_nearest(7, 20, 4, ring_bits=32), 0)
        self.assertEqual(ring.trunc_nearest(-9, 20, 4, ring_bits=32), -1)

    def test_edges_within_declared_length(self):
        length, shift = 12, 4
        bound = 1 << (length - 1)
        for value in (-bound, -bound + 1, -1, 0, 1, bound - 1):
            with self.subTest(value=value):
                self.assertEqual(
                    ring.trunc_nearest(value, length, shift, ring_bits=32),
                    _exact_nearest(value, shift))

    def test_operand_past_length_wraps(self):
        # Nearest rounding gives TruncRing one carry bit (k+1).  A value that
        # fits that widened length is still exact; one beyond it aliases.
        shift = 4
        value = (1 << 11) + 3
        self.assertEqual(ring.trunc_nearest(value, 13, shift, ring_bits=32),
                         _exact_nearest(value, shift))
        self.assertNotEqual(ring.trunc_nearest(value, 11, shift, ring_bits=32),
                            _exact_nearest(value, shift))

    def test_length_cannot_exceed_ring(self):
        with self.assertRaises(ValueError):
            ring.trunc_nearest(1, 33, 4, ring_bits=32)


class FusedPreluOracleTests(unittest.TestCase):
    FRACTIONAL_BITS = 6

    def test_matches_exact_when_nothing_wraps(self):
        # Small accumulator, mixed-sign slopes within one: the oracle equals the
        # unwrapped nearest truncation of the selected product.
        acc = np.array([12, -12, 40, -40, 0], dtype=object)
        alpha = np.array([0.5, 0.5, -0.25, 0.75, 0.9])
        alpha_bits = 6
        out = ring.fused_prelu(
            acc, alpha, alpha_bits=alpha_bits, accumulator_bits=8,
            slope_mag_bits=alpha_bits, fractional_bits=self.FRACTIONAL_BITS,
            ring_bits=32)
        shift = self.FRACTIONAL_BITS + alpha_bits
        for value, a, got in zip(acc, alpha, out):
            slope = (1 << alpha_bits) if value >= 0 else int(round(a * (1 << alpha_bits)))
            self.assertEqual(got, _exact_nearest(int(value) * slope, shift))

    def test_slope_above_one_needs_the_wider_length(self):
        # |alpha| = 3 quantizes to 3 * 2^alpha_bits, three magnitude bits above
        # alpha_bits.  A large negative accumulator overflows the old length but
        # sits inside the corrected one.
        alpha_bits, f = 6, self.FRACTIONAL_BITS
        alpha = np.array([-3.0])
        q = int(round(3.0 * (1 << alpha_bits)))
        slope_mag_bits = math.ceil(math.log2(q))          # 8
        accumulator_bits = 8
        acc_int = -(1 << (accumulator_bits + 2 * f)) + 1  # near the bound
        acc = np.array([acc_int], dtype=object)
        shift = f + alpha_bits
        exact = _exact_nearest(acc_int * -q, shift)

        wide = ring.fused_prelu(
            acc, alpha, alpha_bits=alpha_bits, accumulator_bits=accumulator_bits,
            slope_mag_bits=slope_mag_bits, fractional_bits=f, ring_bits=32)
        self.assertEqual(int(wide[0]), exact)

        narrow = ring.fused_prelu(
            acc, alpha, alpha_bits=alpha_bits, accumulator_bits=accumulator_bits,
            slope_mag_bits=alpha_bits, fractional_bits=f, ring_bits=32)
        self.assertNotEqual(int(narrow[0]), exact)


if __name__ == "__main__":
    unittest.main()

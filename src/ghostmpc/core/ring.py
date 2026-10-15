"""Integer primitives of the ring the MPC program actually computes on.

MP-SPDZ evaluates the emitted circuit modulo ``2^ring_bits`` with a nearest
``TruncRing`` truncation that assumes its operand fits the declared signed
length.  This module reproduces the wrap, sign and nearest rounding exactly, so
a declared truncation length can be checked against the real protocol instead
of an idealized one.
"""

from __future__ import annotations

from typing import Any


def to_signed(value: Any, bits: int) -> Any:
    """Reduce an integer to a signed ``bits``-bit ring element.

    Works on Python ints and NumPy integer arrays alike; the ``% modulus`` keeps
    the result in ``[0, modulus)`` before the recentering makes it two's
    complement, matching how a ring value past its declared width aliases.
    """
    modulus = 1 << bits
    half = modulus >> 1
    return ((value + half) % modulus) - half


def trunc_nearest(value: Any, length: int, shift: int, *, ring_bits: int) -> Any:
    """Nearest ``TruncRing`` of ``value`` by ``shift`` bits, declared ``length``.

    MP-SPDZ implements nearest rounding as ``TruncRing(value + 2^(shift-1),
    length + 1, shift)`` (except at the full-ring frontier).  Therefore the
    rounding carry is part of the declared signed value before it wraps.  This
    distinction matters for an out-of-bound truncand just beyond ``length`` and
    is required for reveal-equivalence on synthetic inputs.
    """
    if length > ring_bits:
        raise ValueError(f"declared length {length} exceeds the ring width {ring_bits}")
    rounded = value + (1 << (shift - 1))
    if length < ring_bits:
        return to_signed(rounded, length + 1) >> shift
    # Full-width nearest rounding is lowered by MP-SPDZ to a truncation by
    # ``shift - 1`` followed by a one-bit rounding step.  No current deploy
    # candidate uses this frontier, but model it for completeness.
    first = to_signed(value, ring_bits) >> (shift - 1)
    return (first + 1) >> 1


def fused_prelu(accumulator: Any, alpha: Any, *, alpha_bits: int,
                accumulator_bits: int, slope_mag_bits: int,
                fractional_bits: int, ring_bits: int) -> Any:
    """Exact ring output of a conv->PReLU fused into one truncation.

    ``accumulator`` is the raw producer integer at scale ``2^(2f)`` and ``alpha``
    the public per-element slope in the reals.  The slope is selected on the raw
    accumulator's sign, multiplied in, then truncated once by ``f + alpha_bits``;
    the returned integer is the activation at scale ``2^f``.
    """
    import numpy as np

    accumulator = np.asarray(accumulator, dtype=object)
    quantized_slope = np.asarray(
        [int(round(a * (1 << alpha_bits))) for a in np.ravel(alpha)],
        dtype=object).reshape(np.shape(alpha))
    one = 1 << alpha_bits
    sign_bits = accumulator_bits + 2 * fractional_bits
    slope = np.where(to_signed(accumulator, sign_bits + 1) >= 0,
                     one, quantized_slope)
    product = accumulator * slope
    length = accumulator_bits + 2 * fractional_bits + slope_mag_bits + 1
    shift = fractional_bits + alpha_bits
    flat = [trunc_nearest(int(v), length, shift, ring_bits=ring_bits)
            for v in np.ravel(product)]
    return np.asarray(flat, dtype=object).reshape(np.shape(product))

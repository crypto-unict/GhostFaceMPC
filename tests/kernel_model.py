"""Clear-text model of the bounded-width rounding kernels, on shares.

The counterpart of the emitted ``trunc_round_nearest_bounded`` helpers.  It
models the additive summands of a replicated sharing explicitly, instead of
merely checking the final arithmetic shift, which is what catches a
share-dependent carry error.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import product
from typing import Any


def ring_signed(value: int, ring_bits: int) -> int:
    modulus = 1 << ring_bits
    value %= modulus
    return value - modulus if value >= modulus // 2 else value


def _validate_width(ring_bits: int, k: int, m: int) -> None:
    if ring_bits < 2:
        raise ValueError("ring_bits must be at least two")
    if not 0 <= m < k <= ring_bits:
        raise ValueError("bounded truncation requires 0 <= m < k <= ring_bits")


def two_summand_components(shares: Iterable[int], *, ring_bits: int,
                           ) -> tuple[int, int]:
    """Regroup a replicated sharing the way ``split_to_n_summands(k, 2)`` does.

    Party ``i`` holds ``(x_i, x_{i-1})``, so P0 -- and only P0 -- knows
    ``A = x_0 + x_2``, while ``B = x_1`` is held by both P1 and P2.  That is the
    decomposition the two-summand rounding kernel is built on.
    """
    values = tuple(int(value) % (1 << ring_bits) for value in shares)
    if len(values) != 3:
        raise ValueError("a replicated sharing has three additive components")
    modulus = 1 << ring_bits
    return (values[0] + values[2]) % modulus, values[1]


def bounded_floor_from_shares(
    shares: Iterable[int], *, ring_bits: int, k: int, m: int,
    signed: bool = True, validate_bound: bool = True, summands: int = 3,
) -> int:
    """Evaluate the bounded-width identity on additive ring components.

    ``summands`` selects the regrouping: three is the malicious profile's
    kernel, two the semi-honest one.  Both take the *same* replicated sharing,
    so the two results can be compared directly.
    """
    _validate_width(ring_bits, k, m)
    if summands not in (2, 3):
        raise ValueError("the rounding kernel splits into two or three summands")
    values = tuple(int(value) % (1 << ring_bits) for value in shares)
    if len(values) != 3:
        raise ValueError("a replicated sharing has three additive components")
    represented = ring_signed(sum(values), ring_bits)
    if validate_bound:
        lower = -(1 << (k - 1)) if signed else 0
        upper = (1 << (k - 1)) - 1 if signed else (1 << k) - 1
        if not lower <= represented <= upper:
            raise OverflowError(
                f"represented value {represented} is outside declared {k}-bit bound"
            )
    adjusted = [value for value in values]
    if signed:
        # A public constant is added to one component in additive sharing.
        adjusted[0] = (adjusted[0] + (1 << (k - 1))) % (1 << ring_bits)
    if summands == 2:
        adjusted = list(two_summand_components(adjusted, ring_bits=ring_bits))
    mask_k = (1 << k) - 1
    reduced = [value & mask_k for value in adjusted]
    low_mask = (1 << m) - 1
    low_carry = sum(value & low_mask for value in reduced) >> m if m else 0
    high_carry = sum(reduced) >> k
    shifted_components = sum(value >> m for value in reduced)
    result = shifted_components + low_carry - (high_carry << (k - m))
    if signed:
        result -= 1 << (k - m - 1)
    return ring_signed(result, ring_bits)


def bounded_round_nearest_from_shares(
    shares: Iterable[int], *, ring_bits: int, k: int, m: int,
    signed: bool = True, validate_bound: bool = True, summands: int = 3,
) -> int:
    """Exact MP-SPDZ half-up rounding using the bounded split identity."""
    _validate_width(ring_bits, k, m)
    values = list(int(value) % (1 << ring_bits) for value in shares)
    represented = ring_signed(sum(values), ring_bits)
    if validate_bound:
        lower = -(1 << (k - 1)) if signed else 0
        upper = (1 << (k - 1)) - 1 if signed else (1 << k) - 1
        if not lower <= represented <= upper:
            raise OverflowError(
                f"represented value {represented} is outside declared {k}-bit bound"
            )
    if m == 0:
        return represented
    values[0] = (values[0] + (1 << (m - 1))) % (1 << ring_bits)
    # The rounding term is why MP-SPDZ increments the declared width before
    # entering exact truncation.
    return bounded_floor_from_shares(
        values, ring_bits=ring_bits, k=k + 1, m=m, signed=signed,
        validate_bound=False, summands=summands,
    )


def bounded_round_nearest_with_sign_from_shares(
    shares: Iterable[int], *, ring_bits: int, k: int, m: int,
    signed: bool = True, validate_bound: bool = True, summands: int = 3,
) -> tuple[int, int]:
    rounded = bounded_round_nearest_from_shares(
        shares, ring_bits=ring_bits, k=k, m=m, signed=signed,
        validate_bound=validate_bound, summands=summands,
    )
    return rounded, int(rounded < 0)


def baseline_round_nearest(value: int, m: int) -> int:
    if m < 0:
        raise ValueError("shift must be non-negative")
    return int(value) if m == 0 else (int(value) + (1 << (m - 1))) >> m


def exhaustive_small_ring(
    ring_bits: int, *, share_values: bool = True, summands: int = 3,
) -> dict[str, Any]:
    """Exhaust all valid widths/values and optionally all three-share tuples.

    ``summands`` chooses the rounding kernel; both must reproduce the same
    clear-text ``round_nearest`` over every sharing.
    """
    if ring_bits > 8:
        raise ValueError("exhaustive helper is intentionally limited to small rings")
    modulus = 1 << ring_bits
    checked = 0
    for k in range(2, ring_bits):
        for m in range(1, k):
            for value in range(-(1 << (k - 1)), 1 << (k - 1)):
                target = baseline_round_nearest(value, m)
                if share_values:
                    sharings = (
                        (first, second, value - first - second)
                        for first, second in product(range(modulus), repeat=2)
                    )
                else:
                    sharings = ((value, 0, 0),)
                for shares in sharings:
                    actual, sign = bounded_round_nearest_with_sign_from_shares(
                        shares, ring_bits=ring_bits, k=k, m=m,
                        summands=summands,
                    )
                    if actual != target or sign != int(target < 0):
                        raise AssertionError({
                            "ring_bits": ring_bits, "summands": summands,
                            "k": k, "m": m,
                            "value": value, "shares": tuple(shares),
                            "expected": target, "actual": actual,
                            "expected_sign": int(target < 0), "sign": sign,
                        })
                    checked += 1
    return {"ring_bits": ring_bits, "summands": summands,
            "cases": checked, "passed": True}

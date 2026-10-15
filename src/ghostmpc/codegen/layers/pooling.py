"""Global average pooling and its public reciprocal."""

from __future__ import annotations

import math

from ...core.model import Op
from ..naming import _bounded_split


def _pooled_bound(op: Op, ops: dict[str, Op], bounds: dict[str, float] | None,
                  depth: int = 0) -> float | None:
    """Bound on the values a pooling op averages, or ``None`` when unknown.

    The average never exceeds the peak of its operands, so the nearest
    calibrated bound upstream sizes the pooling accumulator.  Shape-only ops
    (``Reshape``) and the pooling itself carry no bound of their own, hence the
    short walk back.
    """
    if op.kind != "GlobalAveragePooling2D" or not bounds:
        return None
    pending = list(op.inputs)
    for _ in range(6):
        following: list[str] = []
        for name in pending:
            if bounds.get(name):
                return bounds[name]
            producer = ops.get(name)
            if producer is not None:
                following.extend(producer.inputs)
        if not following:
            break
        pending = following
    return None

def pooling_reciprocal(spatial: int, bound: float | None, *,
                       fractional_bits: int, total_bits: int,
                       ring_bits: int) -> tuple[int, int, int]:
    """Public averaging constant for a pooling window, and how to truncate it.

    Averaging is a multiplication by the *public* constant ``1/spatial``.
    Quantizing it on the activation grid, as ``sfix * float`` does, is only
    sound while ``2^f / spatial`` is a well-resolved integer: at a low ``f`` and
    a wide window it collapses.  At ``f = 6`` the 7x7 windows give
    ``rint(64/49) = 1``, a systematic ``-23.4 %`` on the gate, and the 28x28 and
    14x14 windows give ``rint(64/784) = rint(64/196) = 0``, which annihilates
    those squeeze-and-excitation branches outright.

    Nothing forces a public constant onto the activation grid, so the reciprocal
    is instead carried at ``f + g`` bits and the product truncated by ``f + g``.
    ``g`` is derived, never configured: ``ceil(log2(spatial))`` keeps ``f + 1``
    significant bits in the reciprocal (relative error below ``2^-(f+1)``),
    capped by what the ring can hold, since the product occupies
    ``e + 2f + g`` bits plus the sign.

    Returns ``(reciprocal, shift, length)``: the integer constant, the
    truncation shift ``f + g``, and the declared length of that truncation.
    """
    accumulator_bits = (math.ceil(math.log2(bound)) if bound and bound > 0
                        else total_bits - fractional_bits)
    accumulator_bits = max(accumulator_bits, 0)
    ideal = math.ceil(math.log2(spatial)) if spatial > 1 else 0
    affordable = (ring_bits - 2) - (accumulator_bits + 2 * fractional_bits)
    extra = max(0, min(ideal, affordable))
    shift = fractional_bits + extra
    reciprocal = int(round((1 << shift) / spatial))
    length = accumulator_bits + fractional_bits + shift + 1
    if length > ring_bits:
        # Even the plain grid multiplication overflows here, so any emission
        # would wrap silently.  Refuse instead: this is the failure mode the
        # pooling constant already exhibited once.
        raise OverflowError(
            f"a {spatial}-element pooling of values below 2^{accumulator_bits} "
            f"needs {length} bits at f={fractional_bits}, beyond ring{ring_bits}")
    return reciprocal, shift, length

def _global_average_pool(target: str, source: str, input_shape: list[int], *,
                         fractional_bits: int, total_bits: int, ring_bits: int,
                         bound: float | None) -> str:
    height, width, channels = input_shape
    spatial = height * width
    reciprocal, shift, length = pooling_reciprocal(
        spatial, bound, fractional_bits=fractional_bits,
        total_bits=total_bits, ring_bits=ring_bits)
    if _bounded_split(length, ring_bits):
        return f'''{target}_sums = sint.Array({channels})
{target}_sums.assign({source}.get_vector(0, {channels}).v)
@for_range({spatial - 1})
def _(position):
    position += 1
    {target}_sums.assign_vector(
        {target}_sums.get_vector() +
        {source}.get_vector(position * {channels}, {channels}).v)
{target} = sfix.Array({channels})
{target}_scaled = {target}_sums.get_vector() * {reciprocal}
{target}.assign(sfix._new(trunc_round_nearest_bounded(
    {target}_scaled, {length}, {shift}, signed=True)))
'''
    return f'''{target}_sums = sint.Array({channels})
{target}_sums.assign({source}.get_vector(0, {channels}).v)
@for_range({spatial - 1})
def _(position):
    position += 1
    {target}_sums.assign_vector(
        {target}_sums.get_vector() +
        {source}.get_vector(position * {channels}, {channels}).v)
{target} = sfix.Array({channels})
{target}_scaled = {target}_sums.get_vector() * {reciprocal}
{target}.assign(sfix._new(truncate_generic(
    {target}_scaled, {length}, {shift})))
'''

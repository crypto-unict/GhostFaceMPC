"""PReLU, BatchNorm and the remaining pointwise activations."""

from __future__ import annotations

from ...core.model import Op
from ..fusion import PreluFusion
from ..naming import (
    _bounded_split,
    _conv_raw_bias_name,
    _copy,
    _offset,
    _prelu_expanded_alpha_name,
    _public_scaled_vector,
)


def _prelu_fused(op: Op, target: str, source: str, size: int,
                 offsets: dict[str, int], fusion: PreluFusion,
                 fractional_bits: int, kernel_bits: int | None = None,
                 *, ring_bits: int = 64) -> str:
    """Emit a PReLU that also truncates the accumulator that feeds it.

    ``slope`` is ``alpha`` on the negative branch and one on the positive one,
    both scaled by ``2^alpha_bits``, so multiplying the raw accumulator by it
    and shifting by ``f + alpha_bits`` yields the activation at scale ``2^f``.
    The positive branch is exactly the truncation the producer used to emit.
    """
    alpha_offset = _offset(offsets, _prelu_expanded_alpha_name(op.name))
    bias_name = _conv_raw_bias_name(fusion.producer)
    bias = (f" + public_weights.get_vector({_offset(offsets, bias_name)}, {size})"
            if bias_name in offsets else "")
    kernel_bits = fractional_bits if kernel_bits is None else kernel_bits
    one = 1 << fusion.alpha_bits
    # The accumulator carries ``f + f_w``; removing the kernel scale and the
    # slope scale lands the result back on the activation grid.
    shift = kernel_bits + fusion.alpha_bits
    # One bit above the bound of the product covers both its sign and the
    # rounding term that a nearest truncation adds before shifting.  The slope
    # magnitude, not ``alpha_bits``, sets the product width: a channel with
    # ``|alpha| > 1`` quantizes past ``2^alpha_bits`` and needs the extra bits.
    length = (fusion.accumulator_bits + fractional_bits + kernel_bits
              + fusion.slope_mag_bits + 1)
    # A comparison costs one carry chain over the declared width, so the
    # accumulator states its own instead of defaulting to the whole ring.
    sign_bits = fusion.accumulator_bits + fractional_bits + kernel_bits
    if _bounded_split(length, ring_bits):
        rounded = (
            f"trunc_round_nearest_bounded({target}_product, {length}, "
            f"{shift}, signed=True)"
        )
    else:
        rounded = (
            f"truncate_generic({target}_product, {length}, {shift})"
        )
    return f'''{target} = sfix.Array({size})
{target}_acc = {source}.get_vector(){bias}
{target}_alpha = public_weights.get_vector({alpha_offset}, {size})
{target}_positive = 1 - is_negative({target}_acc, {sign_bits})
{target}_slope = {target}_alpha + {target}_positive * ({one} - {target}_alpha)
{target}_product = {target}_acc * {target}_slope
{target}.assign(sfix._new({rounded}))
'''

def _prelu(op: Op, target: str, source: str, size: int,
           offsets: dict[str, int]) -> str:
    alpha_offset = _offset(offsets, _prelu_expanded_alpha_name(op.name))
    rounded = (
        f"trunc_round_nearest_bounded({target}_negative_raw, "
        "sfix.k + sfix.f, sfix.f, signed=True)"
    )
    return f'''{target} = sfix.Array({size})
{target}_negative_raw = {source}.get_vector().v * public_weights.get_vector({alpha_offset}, {size})
{target}_negative = sfix._new({rounded})
{target}_values = {source}.get_vector()
{target}.assign(({target}_values >= 0).if_else({target}_values, {target}_negative))
'''

def _batchnorm(op: Op, target: str, source: str, size: int,
               offsets: dict[str, int]) -> str:
    channels = op.output_shape[-1]
    scale_offset = _offset(offsets, f"{op.name}:affine_scale")
    bias_offset = _offset(offsets, f"{op.name}:affine_bias")
    scale = f"{target}_scale"
    bias = f"{target}_affine_bias"
    return (
        _public_scaled_vector(scale, size, scale_offset, channels)
        + _public_scaled_vector(bias, size, bias_offset, channels)
        + f'''{target} = sfix.Array({size})
{target}_raw = {source}.get_vector().v * {scale}.get_vector()
{target}_scaled = truncate_generic({target}_raw, sfix.k + sfix.f, sfix.f)
{target}.assign(sfix._new({target}_scaled + {bias}.get_vector()))
'''
    )

def _activation(op: Op, target: str, source: str, size: int, activation: str) -> str:
    if activation == "linear":
        return _copy(target, source, size)
    if activation in {"sigmoid", "hard_sigmoid"}:
        return f'''{target} = sfix.Array({size})
{target}_raw = 0.2 * {source}.get_vector() + 0.5
{target}.assign(({target}_raw < 0).if_else(0, ({target}_raw > 1).if_else(1, {target}_raw)))
'''
    raise ValueError(f"unsupported activation in generator: {activation}")

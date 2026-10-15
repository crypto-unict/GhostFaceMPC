"""Add, Multiply and Concatenate emission."""

from __future__ import annotations

from ...core.model import Op
from ..fusion import _raw_scale_bits
from ..naming import _bounded_split, _size, _truncate


def _concat(op: Op, target: str, sources: list[str], shapes: list[list[int]],
            *, raw: bool = False, source_units: list[int] | None = None,
            fractional_bits: int = 20) -> str:
    axis = int(op.params.get("axis", -1))
    rank = len(op.output_shape)
    if axis < 0:
        axis += rank
    if axis != rank - 1:
        raise ValueError(f"{op.name}: only channel-last concatenation is supported")
    output_size = _size(op.output_shape)
    output_channels = op.output_shape[-1]
    outer = output_size // output_channels
    name = f"{target}_raw" if raw else target
    array = "sint" if raw else "sfix"
    # Raw operands may reach the concatenation at different ``2^f`` exponents (a
    # fused SE conv is one unit higher than its neighbours), so each narrower
    # operand is lifted to the widest with a public left shift, which costs no
    # communication.  The truncated path always shares one exponent.
    target_units = max(source_units) if source_units else 0
    lines = [f"{name} = {array}.Array({output_size})\n"]
    channel_offset = 0
    for index, (source, shape) in enumerate(zip(sources, shapes)):
        channels = shape[-1]
        vector = f"{source}.get_vector(position * {channels}, {channels})"
        if source_units is not None:
            shift = fractional_bits * (target_units - source_units[index])
            if shift:
                vector = f"({vector} * {1 << shift})"
        lines.append(
            f"@for_range_multithread(THREADS, 1, {outer})\n"
            "def _(position):\n"
            f"    {name}.assign_vector({vector}, "
            f"position * {output_channels} + {channel_offset})\n"
        )
        channel_offset += channels
    return "".join(lines)

def _add(target: str, sources: list[str], size: int) -> str:
    expression = " + ".join(f"{source}.get_vector()" for source in sources)
    return f"{target} = sfix.Array({size})\n{target}.assign({expression})\n"

def _add_fused(target: str, raw_sources: list[str], size: int,
               fused_bias_offset: int, *, source_units: list[int] | None = None,
               chain_bits: int | None = None, fractional_bits: int = 20,
               total_bits: int = 31, ring_bits: int = 64,
               kernel_bits: int | None = None) -> str:
    """Sum untruncated operands and pay for a single truncation.

    When every operand shares the ``2^{2f}`` product scale the sum truncates by
    ``f`` under the default declared length.  A fused SE conv reaches the add one
    unit higher, at ``2^{3f}``; the narrower operands are lifted to it with a
    public left shift and the sum then truncates by ``2f`` under the wider signed
    length its ``e`` integer bits need.
    """
    kernel_bits = fractional_bits if kernel_bits is None else kernel_bits
    raw = f"{target}_raw"
    target_units = max(source_units) if source_units else 2
    terms: list[str] = []
    for index, source in enumerate(raw_sources):
        vector = f"{source}.get_vector()"
        if source_units is not None:
            shift = fractional_bits * (target_units - source_units[index])
            if shift:
                vector = f"({vector} * {1 << shift})"
        terms.append(vector)
    expression = " + ".join(terms)
    scale_bits = _raw_scale_bits(target_units, fractional_bits, kernel_bits)
    if target_units > 2 or kernel_bits != fractional_bits:
        # ``e`` magnitude bits over the carried scale, plus one for the sign;
        # the shift drops everything the product scale added above ``2^f``.
        length_bits = int(chain_bits) + scale_bits + 1
        shift = str(scale_bits - fractional_bits)
        truncation = _truncate(raw, target, size, fused_bias_offset,
                               length=str(length_bits), shift=shift,
                               bounded=_bounded_split(length_bits, ring_bits))
    else:
        truncation = _truncate(
            raw, target, size, fused_bias_offset,
            bounded=_bounded_split(total_bits + fractional_bits, ring_bits))
    return f"{raw} = sint.Array({size})\n{raw}.assign({expression})\n" + truncation

def _multiply(target: str, sources: list[str], sizes: list[int], size: int) -> str:
    lines = [f"{target} = sfix.Array({size})\n"]
    vectors: list[str] = []
    for index, (source, source_size) in enumerate(zip(sources, sizes)):
        if source_size == size:
            vectors.append(f"{source}.get_vector()")
        else:
            expanded = f"{target}_broadcast_{index}"
            lines.append(f"{expanded} = sfix.Array({size})\n@for_range({size})\ndef _(i):\n    {expanded}[i] = {source}[i % {source_size}]\n")
            vectors.append(f"{expanded}.get_vector()")
    lines.append(f"{target}.assign({' * '.join(vectors)})\n")
    return "".join(lines)

def _multiply_raw(target: str, feature: str, gate: str, channels: int,
                  size: int) -> str:
    """Emit the untruncated ``feature * gate`` product of a fused SE Multiply.

    Both operands are secret, so the product is one Rep3 multiplication at scale
    ``2^{2f}``; skipping ``sfix``'s truncation is exactly the byte the fusion
    saves.  The ``[1, 1, C]`` gate multiplies every ``[H, W, .]`` position, so one
    channel-wide vector per position broadcasts it without materializing a copy.
    """
    raw = f"{target}_raw"
    broadcast = f"{target}_gate"
    outer = size // channels
    # Broadcast the gate across every position with plain memory copies (no
    # communication), then pay for one vectorized secret multiplication over the
    # whole tensor.  Splitting the multiply per position instead would cost one
    # communication round each, so the full-tensor product keeps it at one round.
    # The gate is read from its global array inside the thread: a multithreaded
    # loop is a separate tape, so a register vector built in the parent tape
    # cannot cross into it, but reloading the array from memory can.
    return f'''{broadcast} = sint.Array({size})
@for_range_multithread(THREADS, 1, {outer})
def _(position):
    {broadcast}.assign_vector({gate}.get_vector(0, {channels}).v,
                              position * {channels})
{raw} = sint.Array({size})
{raw}.assign({feature}.get_vector().v * {broadcast}.get_vector())
'''

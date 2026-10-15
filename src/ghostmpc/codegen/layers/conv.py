"""Conv2D emission as tiled secret-by-public dot products."""

from __future__ import annotations

from ...core.model import Op
from ..fusion import MultiplyFusion
from ..naming import (
    CONV_ALIGNED_SPATIAL_TILE,
    CONV_CHANNEL_TILE,
    CONV_SPATIAL_TILE,
    _bounded_split,
    _conv_expanded_bias_name,
    _kernel_truncation,
    _offset,
    _size,
    _truncate,
)
from .depthwise import _axis_regions, _same_padding


def _conv2d_1x1_tile_loop(
        raw: str, source: str, *, input_c: int, output_w: int, output_c: int,
        stride_h: int, stride_w: int, input_w: int, kernel_offset: int,
        spatial_offset: int, spatial_tile: int, spatial_tiles: int,
        channel_offset: int, channel_tile: int, channel_tiles: int,
        source_is_raw: bool = False) -> str:
    """Emit one tiled 1x1 convolution loop.

    Output channels are contiguous in both the Keras kernel and NHWC output,
    so one public weight vector can update several output channels at once.
    Keeping a few spatial accumulators live also reuses that weight vector
    across neighboring pixels before loading the next input channel.
    """
    if not spatial_tiles or not channel_tiles:
        return ""
    # A raw ``Multiply SE`` accumulator is already a ring value (an ``sint``), so
    # it is read directly; a truncated ``sfix`` operand exposes its ring value as
    # ``.v``.
    ring = "" if source_is_raw else ".v"
    work_items = spatial_tiles * channel_tiles
    lines = [f'''@for_range_multithread(THREADS, 1, {work_items})
def _(tile):
    spatial_tile = tile // {channel_tiles}
    channel_base = {channel_offset} + (tile % {channel_tiles}) * {channel_tile}
''']
    for position_index in range(spatial_tile):
        lines.append(
            f"    position_{position_index} = {spatial_offset} + "
            f"spatial_tile * {spatial_tile} + {position_index}\n"
            f"    out_x_{position_index} = position_{position_index} % {output_w}\n"
            f"    out_y_{position_index} = position_{position_index} // {output_w}\n"
            f"    input_base_{position_index} = "
            f"(out_y_{position_index} * {stride_h} * {input_w} + "
            f"out_x_{position_index} * {stride_w}) * {input_c}\n"
            f"    acc_{position_index} = sint(0, size={channel_tile})\n"
        )
    for in_c in range(input_c):
        weight_base = kernel_offset + in_c * output_c
        lines.append(
            f"    weights = public_weights.get_vector("
            f"{weight_base} + channel_base, {channel_tile})\n"
        )
        for position_index in range(spatial_tile):
            lines.append(
                f"    input_{position_index} = "
                f"{source}[input_base_{position_index} + {in_c}]{ring}"
                f".expand_to_vector({channel_tile})\n"
                f"    acc_{position_index} += input_{position_index} * weights\n"
            )
    for position_index in range(spatial_tile):
        lines.append(
            f"    {raw}.assign_vector(acc_{position_index}, "
            f"position_{position_index} * {output_c} + channel_base)\n"
        )
    return "".join(lines)

def _conv2d_1x1(raw: str, source: str, *, input_c: int, output_h: int,
                 output_w: int, output_c: int, stride_h: int, stride_w: int,
                 input_w: int, kernel_offset: int,
                 source_is_raw: bool = False) -> str:
    spatial = output_h * output_w
    if spatial >= CONV_ALIGNED_SPATIAL_TILE and \
            spatial % CONV_ALIGNED_SPATIAL_TILE == 0:
        spatial_tile_size = CONV_ALIGNED_SPATIAL_TILE
    else:
        spatial_tile_size = min(spatial, CONV_SPATIAL_TILE)
    full_spatial_tiles, spatial_remainder = divmod(spatial, spatial_tile_size)
    full_channel_tiles, channel_remainder = divmod(output_c, CONV_CHANNEL_TILE)
    channel_batches = [
        (0, CONV_CHANNEL_TILE, full_channel_tiles),
        (full_channel_tiles * CONV_CHANNEL_TILE, channel_remainder,
         int(channel_remainder > 0)),
    ]
    spatial_batches = [
        (0, spatial_tile_size, full_spatial_tiles),
        (full_spatial_tiles * spatial_tile_size, 1, spatial_remainder),
    ]
    lines = [f"{raw} = sint.Array({spatial * output_c})\n"]
    for spatial_offset, spatial_tile, spatial_tiles in spatial_batches:
        for channel_offset, channel_tile, channel_tiles in channel_batches:
            lines.append(_conv2d_1x1_tile_loop(
                raw, source, input_c=input_c, output_w=output_w,
                output_c=output_c, stride_h=stride_h, stride_w=stride_w,
                input_w=input_w, kernel_offset=kernel_offset,
                spatial_offset=spatial_offset, spatial_tile=spatial_tile,
                spatial_tiles=spatial_tiles, channel_offset=channel_offset,
                channel_tile=channel_tile, channel_tiles=channel_tiles,
                source_is_raw=source_is_raw,
            ))
    return "".join(lines)

def _conv2d_spatial_tile_loop(
        raw: str, source: str, *, input_w: int, input_c: int, output_w: int,
        output_c: int, kernel_w: int, stride_h: int, stride_w: int,
        pad_h: int, pad_w: int, y_start: int, x_start: int, x_count: int,
        ky_start: int, ky_end: int, kx_start: int, kx_end: int,
        kernel_offset: int, spatial_offset: int, spatial_tile: int,
        spatial_tiles: int, channel_offset: int, channel_tile: int,
        channel_tiles: int) -> str:
    """Emit a tiled spatial convolution over contiguous output channels.

    Keras stores the output-channel dimension contiguously in both its kernel
    and NHWC output.  Accumulating a vector of output channels therefore turns
    the innermost scalar/MemValue loop into local SIMD arithmetic.  A small
    spatial tile also amortizes each public weight-vector load over several
    neighboring output pixels.

    Positions are numbered inside one ``x_count``-wide output region whose taps
    are all valid, so omitting the taps outside ``[ky_start, ky_end)`` and
    ``[kx_start, kx_end)`` is algebraically the same as multiplying by the zeros
    of a padded input.
    """
    if not spatial_tiles or not channel_tiles:
        return ""
    work_items = spatial_tiles * channel_tiles
    lines = [f'''@for_range_multithread(THREADS, 1, {work_items})
def _(tile):
    spatial_tile_index = tile // {channel_tiles}
    channel_base = {channel_offset} + (tile % {channel_tiles}) * {channel_tile}
''']
    for position_index in range(spatial_tile):
        lines.append(
            f"    position_{position_index} = {spatial_offset} + "
            f"spatial_tile_index * {spatial_tile} + {position_index}\n"
            f"    out_x_{position_index} = {x_start} + position_{position_index} % {x_count}\n"
            f"    out_y_{position_index} = {y_start} + position_{position_index} // {x_count}\n"
            f"    input_base_{position_index} = "
            f"((out_y_{position_index} * {stride_h} - {pad_h}) * {input_w} + "
            f"out_x_{position_index} * {stride_w} - {pad_w}) * {input_c}\n"
            f"    acc_{position_index} = sint(0, size={channel_tile})\n"
        )
    for ky in range(ky_start, ky_end):
        for kx in range(kx_start, kx_end):
            input_kernel_offset = (ky * input_w + kx) * input_c
            for in_c in range(input_c):
                weight_base = (
                    kernel_offset
                    + ((ky * kernel_w + kx) * input_c + in_c) * output_c
                )
                lines.append(
                    f"    weights = public_weights.get_vector("
                    f"{weight_base} + channel_base, {channel_tile})\n"
                )
                for position_index in range(spatial_tile):
                    lines.append(
                        f"    input_{position_index} = "
                        f"{source}[input_base_{position_index} + "
                        f"{input_kernel_offset + in_c}].v"
                        f".expand_to_vector({channel_tile})\n"
                        f"    acc_{position_index} += input_{position_index} * weights\n"
                    )
    for position_index in range(spatial_tile):
        lines.append(
            f"    {raw}.assign_vector(acc_{position_index}, "
            f"(out_y_{position_index} * {output_w} + out_x_{position_index}) * "
            f"{output_c} + channel_base)\n"
        )
    return "".join(lines)

def _spatial_batches(area: int) -> list[tuple[int, int, int]]:
    """Split ``area`` output positions into full tiles plus a per-pixel tail."""
    tile_size = min(area, CONV_ALIGNED_SPATIAL_TILE)
    if not tile_size:
        return []
    full_tiles, remainder = divmod(area, tile_size)
    return [(0, tile_size, full_tiles), (full_tiles * tile_size, 1, remainder)]

def _conv2d_spatial(raw: str, source: str, *, input_h: int, input_w: int,
                    input_c: int, output_h: int, output_w: int, output_c: int,
                    kernel_h: int, kernel_w: int, stride_h: int, stride_w: int,
                    pad_h: int, pad_w: int, kernel_offset: int) -> str:
    """Generate a channel-vectorized kernel for non-1x1 convolutions.

    The output is cut into rectangles that share one valid tap interval, so no
    padded copy of the secret input has to be materialized and the border MACs
    against zero disappear.
    """
    full_channel_tiles, channel_remainder = divmod(output_c, CONV_CHANNEL_TILE)
    channel_batches = [
        (0, CONV_CHANNEL_TILE, full_channel_tiles),
        (full_channel_tiles * CONV_CHANNEL_TILE, channel_remainder,
         int(channel_remainder > 0)),
    ]
    y_regions = _axis_regions(output_h, input_h, kernel_h, stride_h, pad_h)
    x_regions = _axis_regions(output_w, input_w, kernel_w, stride_w, pad_w)
    lines = [f"{raw} = sint.Array({output_h * output_w * output_c})\n"]
    for y_start, y_count, ky_start, ky_end in y_regions:
        for x_start, x_count, kx_start, kx_end in x_regions:
            for spatial_offset, spatial_tile, spatial_tiles in \
                    _spatial_batches(y_count * x_count):
                for channel_offset, channel_tile, channel_tiles in channel_batches:
                    lines.append(_conv2d_spatial_tile_loop(
                        raw, source, input_w=input_w, input_c=input_c,
                        output_w=output_w, output_c=output_c, kernel_w=kernel_w,
                        stride_h=stride_h, stride_w=stride_w, pad_h=pad_h,
                        pad_w=pad_w, y_start=y_start, x_start=x_start,
                        x_count=x_count, ky_start=ky_start, ky_end=ky_end,
                        kx_start=kx_start, kx_end=kx_end,
                        kernel_offset=kernel_offset,
                        spatial_offset=spatial_offset,
                        spatial_tile=spatial_tile, spatial_tiles=spatial_tiles,
                        channel_offset=channel_offset,
                        channel_tile=channel_tile, channel_tiles=channel_tiles,
                    ))
    return "".join(lines)

def _conv2d(op: Op, target: str, source: str, input_shape: list[int],
            offsets: dict[str, int], *, emit_truncated: bool = True,
            fused_multiply: MultiplyFusion | None = None,
            fractional_bits: int = 20,
            total_bits: int = 31,
            ring_bits: int = 64,
            kernel_bits: int | None = None) -> str:
    kernel_bits = fractional_bits if kernel_bits is None else kernel_bits
    input_h, input_w, input_c = input_shape
    output_h, output_w, output_c = op.output_shape
    kernel_h, kernel_w = op.params["kernel_size"]
    stride_h, stride_w = op.params["strides"]
    _, _, pad_h, pad_w = _same_padding(op, input_shape)
    kernel_offset = _offset(offsets, op.tensors[0])
    bias_offset = (_offset(offsets, _conv_expanded_bias_name(op.name))
                   if emit_truncated and len(op.tensors) > 1 else None)
    output_size = _size(op.output_shape)
    raw = f"{target}_raw"
    if fused_multiply is not None:
        # The raw ``feature * gate`` operand carries ``2^{2f}``, so the
        # accumulator sits at ``2^{2f + f_w}`` and the fused truncation both
        # shifts by ``f + f_w`` and declares the wider signed length that scale
        # needs.
        length_bits = (fused_multiply.accumulator_bits + 2 * fractional_bits
                       + kernel_bits + 1)
        truncation = (_truncate(raw, target, output_size, bias_offset,
                                length=str(length_bits),
                                shift=str(fractional_bits + kernel_bits),
                                bounded=_bounded_split(length_bits, ring_bits))
                      if emit_truncated else "")
    else:
        length_bits = total_bits + kernel_bits
        truncation = (_truncate(raw, target, output_size, bias_offset,
                                **_kernel_truncation(fractional_bits, kernel_bits),
                                bounded=_bounded_split(length_bits, ring_bits))
                      if emit_truncated else "")
    if kernel_h == kernel_w == 1:
        # A 1x1 kernel never needs padding: SAME keeps every tap inside the
        # input, so the 1x1 loop can index the source tensor directly.
        if pad_h or pad_w:
            raise ValueError(f"unexpected padding on the 1x1 conv {op.name}")
        code = _conv2d_1x1(
            raw, source, input_c=input_c, output_h=output_h,
            output_w=output_w, output_c=output_c, stride_h=stride_h,
            stride_w=stride_w, input_w=input_w, kernel_offset=kernel_offset,
            source_is_raw=fused_multiply is not None,
        )
    else:
        if fused_multiply is not None:
            raise ValueError(f"fused multiply feeds a non-1x1 conv {op.name}")
        code = _conv2d_spatial(
            raw, source, input_h=input_h, input_w=input_w,
            input_c=input_c, output_h=output_h, output_w=output_w,
            output_c=output_c, kernel_h=kernel_h, kernel_w=kernel_w,
            stride_h=stride_h, stride_w=stride_w, pad_h=pad_h, pad_w=pad_w,
            kernel_offset=kernel_offset,
        )
    return code + truncation

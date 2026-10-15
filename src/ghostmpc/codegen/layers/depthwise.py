"""DepthwiseConv2D emission, including the packed-lane layout."""

from __future__ import annotations

from ...core.model import Manifest, Op
from ..naming import (
    DEPTHWISE_CHANNEL_TILE,
    DEPTHWISE_LANE_CAP,
    _bounded_split,
    _conv_expanded_bias_name,
    _kernel_truncation,
    _offset,
    _size,
    _truncate,
)


def _depthwise_packed_name(op_name: str, lane: int) -> str:
    return f"{op_name}:depthwise_packed:{lane}"

def _packed_lane(x_count: int, channels: int) -> int:
    """Pixels per packed vector for a region of ``x_count`` output columns."""
    return max(1, min(x_count, DEPTHWISE_LANE_CAP // channels))

def _same_padding(op: Op, input_shape: list[int]) -> tuple[int, int, int, int]:
    """Return ``(total_h, total_w, pad_h, pad_w)`` for one conv-like op."""
    input_h, input_w = input_shape[0], input_shape[1]
    output_h, output_w = op.output_shape[0], op.output_shape[1]
    kernel_h, kernel_w = op.params["kernel_size"]
    stride_h, stride_w = op.params["strides"]
    if op.params["padding"] != "SAME":
        return 0, 0, 0, 0
    total_h = max((output_h - 1) * stride_h + kernel_h - input_h, 0)
    total_w = max((output_w - 1) * stride_w + kernel_w - input_w, 0)
    return total_h, total_w, total_h // 2, total_w // 2

def _axis_regions(output_size: int, input_size: int, kernel_size: int,
                  stride: int, pad: int) -> list[tuple[int, int, int, int]]:
    """Return contiguous output ranges with the same valid kernel interval."""
    regions: list[tuple[int, int, int, int]] = []
    for position in range(output_size):
        start = max(0, pad - position * stride)
        end = min(kernel_size, input_size + pad - position * stride)
        if regions and regions[-1][2:] == (start, end):
            previous = regions[-1]
            regions[-1] = (previous[0], previous[1] + 1, start, end)
        else:
            regions.append((position, 1, start, end))
    return regions

def _depthwise_lanes(op: Op, input_shape: list[int]) -> list[int]:
    """Lane widths a depthwise op packs with, one entry per distinct value."""
    input_h, input_w, _ = input_shape
    output_h, output_w, output_c = op.output_shape
    kernel_h, kernel_w = op.params["kernel_size"]
    stride_h, stride_w = op.params["strides"]
    _, _, pad_h, pad_w = _same_padding(op, input_shape)
    lanes: set[int] = set()
    for _, _, ky_start, ky_end in _axis_regions(
            output_h, input_h, kernel_h, stride_h, pad_h):
        if ky_start or ky_end != kernel_h:
            continue
        for _, x_count, kx_start, kx_end in _axis_regions(
                output_w, input_w, kernel_w, stride_w, pad_w):
            if kx_start or kx_end != kernel_w:
                continue
            lane = _packed_lane(x_count, output_c)
            if lane > 1:
                lanes.add(lane)
    return sorted(lanes)

def _depthwise_packed_plan(manifest: Manifest) -> dict[str, list[int]]:
    """Lane widths per depthwise op, shared by the emitter and the weight plan."""
    producer_shapes = {op.name: op.output_shape for op in manifest.ops}
    plan: dict[str, list[int]] = {}
    for op in manifest.ops:
        if op.kind != "DepthwiseConv2D" or \
                int(op.params.get("depth_multiplier", 1)) != 1:
            continue
        input_shape = manifest.input_shape
        if op.inputs:
            input_shape = producer_shapes.get(op.inputs[0], manifest.input_shape)
        plan[op.name] = _depthwise_lanes(op, input_shape)
    return plan

def _depthwise_direct_region(
        raw: str, source: str, *, input_w: int, input_c: int, output_w: int,
        output_c: int, kernel_h: int, kernel_w: int, stride_h: int,
        stride_w: int, pad_h: int, pad_w: int, y_start: int, y_count: int,
        x_start: int, x_count: int, ky_start: int, ky_end: int, kx_start: int,
        kx_end: int, channel_offset: int, channel_tile: int,
        channel_tiles: int, kernel_offset: int) -> str:
    """Emit one boundary rectangle without materializing a padded tensor."""
    if not y_count or not x_count or not channel_tiles:
        return ""
    work_items = y_count * x_count * channel_tiles
    lines = [f'''@for_range_multithread(THREADS, 1, {work_items})
def _(tile):
    channel_base = {channel_offset} + (tile % {channel_tiles}) * {channel_tile}
    position = tile // {channel_tiles}
    out_y = {y_start} + position // {x_count}
    out_x = {x_start} + position % {x_count}
    input_base = ((out_y * {stride_h} - {pad_h}) * {input_w} +
                  out_x * {stride_w} - {pad_w}) * {input_c} + channel_base
    acc = sint(0, size={channel_tile})
''']
    for ky in range(ky_start, ky_end):
        for kx in range(kx_start, kx_end):
            input_offset = (ky * input_w + kx) * input_c
            weight_offset = kernel_offset + (ky * kernel_w + kx) * input_c
            lines.append(
                f"    inputs = {source}.get_vector(input_base + {input_offset}, "
                f"{channel_tile}).v\n"
                f"    weights = public_weights.get_vector({weight_offset} + channel_base, "
                f"{channel_tile})\n"
                "    acc += inputs * weights\n"
            )
    lines.append(
        f"    {raw}.assign_vector(acc, (out_y * {output_w} + out_x) * "
        f"{output_c} + channel_base)\n"
    )
    return "".join(lines)

def _depthwise_packed_region(
        raw: str, source: str, *, input_w: int, channels: int, output_w: int,
        kernel_h: int, kernel_w: int, stride_h: int, stride_w: int,
        pad_h: int, pad_w: int, y_start: int, y_count: int, x_start: int,
        x_count: int, lane: int, packed_offset: int) -> str:
    """Emit a full-kernel horizontal tile in spatial-major SIMD layout.

    A lane covers ``lane`` adjacent output pixels over every channel, which is
    exactly one contiguous NHWC run both on the read and on the write side.
    """
    tile_count = x_count // lane
    if not y_count or not tile_count:
        return ""
    work_items = y_count * tile_count
    vector_size = lane * channels
    lines = [f'''@for_range_multithread(THREADS, 1, {work_items})
def _(tile):
    out_y = {y_start} + tile // {tile_count}
    out_x = {x_start} + (tile % {tile_count}) * {lane}
    input_base = ((out_y * {stride_h} - {pad_h}) * {input_w} +
                  out_x * {stride_w} - {pad_w}) * {channels}
    acc = sint(0, size={vector_size})
''']
    for ky in range(kernel_h):
        for kx in range(kernel_w):
            input_offset = (ky * input_w + kx) * channels
            if stride_w == 1:
                inputs = (
                    f"{source}.get_vector(input_base + {input_offset}, "
                    f"{vector_size}).v"
                )
            else:
                parts = ", ".join(
                    f"{source}.get_vector(input_base + {input_offset + position * stride_w * channels}, "
                    f"{channels}).v"
                    for position in range(lane)
                )
                inputs = f"sint.concat(({parts}))"
            packed_kernel_offset = packed_offset + (ky * kernel_w + kx) * vector_size
            lines.append(
                f"    inputs = {inputs}\n"
                f"    weights = public_weights.get_vector({packed_kernel_offset}, {vector_size})\n"
                "    acc += inputs * weights\n"
            )
    lines.append(
        f"    {raw}.assign_vector(acc, (out_y * {output_w} + out_x) * "
        f"{channels})\n"
    )
    return "".join(lines)

def _depthwise_conv2d_packed(
        op_name: str, raw: str, source: str, *, input_h: int, input_w: int,
        input_c: int, output_h: int, output_w: int, output_c: int,
        kernel_h: int, kernel_w: int, stride_h: int, stride_w: int,
        pad_h: int, pad_w: int, kernel_offset: int,
        offsets: dict[str, int]) -> str:
    """Generate exact depthwise code without a padded secret tensor.

    The interior is processed as seven adjacent NHWC pixels per vector.  The
    small edge rectangles use the same channel SIMD path but omit invalid
    kernel taps, which is algebraically identical to multiplication by zero.
    """
    full_channel_tiles, channel_remainder = divmod(output_c, DEPTHWISE_CHANNEL_TILE)
    channel_batches = [
        (0, DEPTHWISE_CHANNEL_TILE, full_channel_tiles),
        (full_channel_tiles * DEPTHWISE_CHANNEL_TILE, channel_remainder,
         int(channel_remainder > 0)),
    ]
    y_regions = _axis_regions(output_h, input_h, kernel_h, stride_h, pad_h)
    x_regions = _axis_regions(output_w, input_w, kernel_w, stride_w, pad_w)
    lines = [f"{raw} = sint.Array({output_h * output_w * output_c})\n"]
    for y_start, y_count, ky_start, ky_end in y_regions:
        for x_start, x_count, kx_start, kx_end in x_regions:
            full_kernel = ky_start == 0 and ky_end == kernel_h and \
                kx_start == 0 and kx_end == kernel_w
            lane = _packed_lane(x_count, output_c) if full_kernel else 1
            packed_width = x_count // lane * lane if lane > 1 else 0
            if packed_width:
                packed_offset = _offset(
                    offsets, _depthwise_packed_name(op_name, lane))
                lines.append(_depthwise_packed_region(
                    raw, source, input_w=input_w, channels=output_c,
                    output_w=output_w, kernel_h=kernel_h, kernel_w=kernel_w,
                    stride_h=stride_h, stride_w=stride_w, pad_h=pad_h,
                    pad_w=pad_w, y_start=y_start, y_count=y_count,
                    x_start=x_start, x_count=packed_width, lane=lane,
                    packed_offset=packed_offset,
                ))
            tail_start = x_start + packed_width
            tail_count = x_count - packed_width
            for channel_offset, channel_tile, channel_tiles in channel_batches:
                if not channel_tiles:
                    continue
                lines.append(_depthwise_direct_region(
                    raw, source, input_w=input_w, input_c=input_c,
                    output_w=output_w, output_c=output_c, kernel_h=kernel_h,
                    kernel_w=kernel_w, stride_h=stride_h, stride_w=stride_w,
                    pad_h=pad_h, pad_w=pad_w, y_start=y_start, y_count=y_count,
                    x_start=tail_start, x_count=tail_count, ky_start=ky_start,
                    ky_end=ky_end, kx_start=kx_start, kx_end=kx_end,
                    channel_offset=channel_offset, channel_tile=channel_tile,
                    channel_tiles=channel_tiles, kernel_offset=kernel_offset,
                ))
    return "".join(lines)

def _depthwise_conv2d(op: Op, target: str, source: str, input_shape: list[int],
                      offsets: dict[str, int], *,
                      emit_truncated: bool = True,
                      fractional_bits: int = 20,
                      total_bits: int = 31,
                      ring_bits: int = 64,
                      kernel_bits: int | None = None) -> str:
    kernel_bits = fractional_bits if kernel_bits is None else kernel_bits
    input_h, input_w, input_c = input_shape
    output_h, output_w, output_c = op.output_shape
    kernel_h, kernel_w = op.params["kernel_size"]
    stride_h, stride_w = op.params["strides"]
    if int(op.params.get("depth_multiplier", 1)) != 1 or output_c != input_c:
        raise ValueError(f"{op.name}: only depth multiplier 1 is supported")
    _, _, pad_h, pad_w = _same_padding(op, input_shape)
    kernel_offset = _offset(offsets, op.tensors[0])
    bias_offset = (_offset(offsets, _conv_expanded_bias_name(op.name))
                   if emit_truncated and len(op.tensors) > 1 else None)
    output_size = _size(op.output_shape)
    raw = f"{target}_raw"
    length_bits = total_bits + kernel_bits
    truncation = (_truncate(raw, target, output_size, bias_offset,
                            **_kernel_truncation(fractional_bits, kernel_bits),
                            bounded=_bounded_split(length_bits, ring_bits))
                  if emit_truncated else "")
    return (
        _depthwise_conv2d_packed(
            op.name, raw, source, input_h=input_h, input_w=input_w,
            input_c=input_c, output_h=output_h, output_w=output_w,
            output_c=output_c, kernel_h=kernel_h, kernel_w=kernel_w,
            stride_h=stride_h, stride_w=stride_w, pad_h=pad_h, pad_w=pad_w,
            kernel_offset=kernel_offset, offsets=offsets,
        )
        + truncation
    )

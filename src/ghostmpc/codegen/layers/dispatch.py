"""Dispatch from a manifest operation to its layer emitter."""

from __future__ import annotations

from ...core.model import Op
from ..fusion import FusionPlan, MultiplyFusion, PreluFusion
from ..naming import _conv_expanded_bias_name, _conv_raw_bias_name, _size
from .activation import _activation, _batchnorm, _prelu, _prelu_fused
from .conv import _conv2d
from .depthwise import _depthwise_conv2d
from .elementwise import _add, _add_fused, _concat, _multiply, _multiply_raw
from .pooling import _global_average_pool


def _layer_code(op: Op, target: str, sources: list[str], input_shapes: list[list[int]],
                offsets: dict[str, int], *, raw_sources: list[str] | None = None,
                emit_raw: bool = False, emit_truncated: bool = True,
                fused_bias_offset: int | None = None,
                prelu_fusion: PreluFusion | None = None,
                multiply_fusion: MultiplyFusion | None = None,
                fused_multiply: MultiplyFusion | None = None,
                raw_source_units: list[int] | None = None,
                raw_chain_bits: int | None = None,
                fractional_bits: int = 20,
                total_bits: int = 31,
                ring_bits: int = 64,
                kernel_bits: int | None = None,
                bound: float | None = None) -> str:
    kernel_bits = fractional_bits if kernel_bits is None else kernel_bits
    output_size = _size(op.output_shape)
    input_sizes = [_size(shape) for shape in input_shapes]
    if op.kind == "Conv2D":
        # A fused SE convolution reads the raw product, not a truncated tensor.
        source = raw_sources[0] if fused_multiply is not None else sources[0]
        return _conv2d(op, target, source, input_shapes[0], offsets,
                       emit_truncated=emit_truncated,
                       fused_multiply=fused_multiply,
                       fractional_bits=fractional_bits,
                       total_bits=total_bits, ring_bits=ring_bits,
                       kernel_bits=kernel_bits)
    if op.kind == "DepthwiseConv2D":
        return _depthwise_conv2d(op, target, sources[0], input_shapes[0],
                                 offsets, emit_truncated=emit_truncated,
                                 fractional_bits=fractional_bits,
                                 total_bits=total_bits, ring_bits=ring_bits,
                                 kernel_bits=kernel_bits)
    if op.kind == "BatchNormalization":
        return _batchnorm(op, target, sources[0], output_size, offsets)
    if op.kind == "PReLU":
        if prelu_fusion is not None:
            return _prelu_fused(op, target, raw_sources[0], output_size, offsets,
                                prelu_fusion, fractional_bits, kernel_bits,
                                ring_bits=ring_bits)
        return _prelu(op, target, sources[0], output_size, offsets)
    if op.kind == "Concatenate":
        code = ""
        if emit_raw:
            code += _concat(op, target, raw_sources, input_shapes, raw=True,
                            source_units=raw_source_units,
                            fractional_bits=fractional_bits)
        if emit_truncated:
            code += _concat(op, target, sources, input_shapes)
        return code
    if op.kind == "Add":
        if fused_bias_offset is not None:
            return _add_fused(target, raw_sources, output_size, fused_bias_offset,
                              source_units=raw_source_units,
                              chain_bits=raw_chain_bits,
                              fractional_bits=fractional_bits,
                              total_bits=total_bits, ring_bits=ring_bits,
                              kernel_bits=kernel_bits)
        return _add(target, sources, output_size)
    if op.kind == "Multiply":
        if multiply_fusion is not None:
            channels = op.output_shape[-1]
            feature_source = sources[op.inputs.index(multiply_fusion.feature)]
            gate_source = sources[op.inputs.index(multiply_fusion.gate)]
            return _multiply_raw(target, feature_source, gate_source, channels,
                                 output_size)
        return _multiply(target, sources, input_sizes, output_size)
    if op.kind == "GlobalAveragePooling2D":
        return _global_average_pool(target, sources[0], input_shapes[0],
                                    fractional_bits=fractional_bits,
                                    total_bits=total_bits, ring_bits=ring_bits,
                                    bound=bound)
    if op.kind == "Activation":
        return _activation(op, target, sources[0], output_size, op.params["activation"])
    if op.kind in {"Reshape", "Flatten"}:
        return ""
    raise ValueError(f"unsupported op in generator: {op.kind}")

def _conv_bias_name(op: Op, fusion: FusionPlan,
                    prelu_producers: dict[str, PreluFusion]) -> str | None:
    """Where a convolution bias is broadcast, or ``None`` if someone else owns it."""
    if op.kind not in {"Conv2D", "DepthwiseConv2D"} or len(op.tensors) < 2:
        return None
    if op.name in prelu_producers:
        # Added to the raw accumulator, so it carries the product scale.
        return _conv_raw_bias_name(op.name)
    if op.name in fusion.raw and op.name not in fusion.truncated:
        return None  # summed into the fused bias of the Add downstream
    return _conv_expanded_bias_name(op.name)

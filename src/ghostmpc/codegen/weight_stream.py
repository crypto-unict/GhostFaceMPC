"""Preparation of the public weight stream.

Model parameters are public, so they are laid out once as a flat ``cint``
stream that the generated program indexes by offset.  Fused biases, pre-scaled
alphas and pre-expanded channel vectors are materialised here rather than
recomputed inside the protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.model import Manifest
from .fusion import (
    DEPLOYED,
    NO_FUSION,
    FusionPlan,
    Optimizations,
    PreluFusion,
    _channel_bias,
    _prelu_alpha_magnitudes,
    plan_fusions,
)
from .layers.depthwise import _depthwise_packed_name, _depthwise_packed_plan
from .layers.dispatch import _conv_bias_name
from .naming import _add_fused_bias_name, _prelu_expanded_alpha_name, _size


@dataclass(frozen=True)
class WeightPlan:
    values: Any
    offsets: dict[str, int]
    lengths: dict[str, int]

def _default_offsets(manifest: Manifest, fusion: FusionPlan = NO_FUSION,
                     prelu_fusion: dict[str, PreluFusion] | None = None,
                     ) -> tuple[dict[str, int], dict[str, int]]:
    offsets = {tensor.name: tensor.offset for tensor in manifest.tensors}
    lengths = {tensor.name: tensor.length for tensor in manifest.tensors}
    cursor = max((tensor.offset + tensor.length for tensor in manifest.tensors), default=0)
    packed_plan = _depthwise_packed_plan(manifest)
    prelu_producers = {fused.producer: fused
                       for fused in (prelu_fusion or {}).values()}
    for op in manifest.ops:
        if op.name in fusion.adds:
            name = _add_fused_bias_name(op.name)
            length = _size(op.output_shape)
            offsets[name], lengths[name] = cursor, length
            cursor += length
        if op.kind == "PReLU":
            name = _prelu_expanded_alpha_name(op.name)
            length = _size(op.output_shape)
            offsets[name], lengths[name] = cursor, length
            cursor += length
        bias_name = _conv_bias_name(op, fusion, prelu_producers)
        if bias_name is not None:
            length = _size(op.output_shape)
            offsets[bias_name], lengths[bias_name] = cursor, length
            cursor += length
        if op.kind != "DepthwiseConv2D":
            continue
        kernel_h, kernel_w = op.params["kernel_size"]
        output_c = op.output_shape[-1]
        for lane in packed_plan.get(op.name, []):
            name = _depthwise_packed_name(op.name, lane)
            length = kernel_h * kernel_w * lane * output_c
            offsets[name], lengths[name] = cursor, length
            cursor += length
    return offsets, lengths

def _prepare_weights(manifest: Manifest, weights: Any, *,
                     fractional_bits: int = 20, ring_bits: int = 64,
                     overflow_bounds: dict[str, float] | None = None,
                     alpha_magnitudes: dict[str, float] | None = None,
                     kernel_bits: int | None = None,
                     optimizations: Optimizations = DEPLOYED) -> WeightPlan:
    import numpy as np

    kernel_bits = fractional_bits if kernel_bits is None else kernel_bits
    # One global ``rint(v * 2^f)`` writes the stream, so a tensor that belongs on
    # a different grid is pre-scaled by the exact power of two between them.
    kernel_scale = 2.0 ** (kernel_bits - fractional_bits)
    refs = {tensor.name: tensor for tensor in manifest.tensors}
    ops = {op.name: op for op in manifest.ops}
    packed_plan = _depthwise_packed_plan(manifest)
    if alpha_magnitudes is None:
        alpha_magnitudes = _prelu_alpha_magnitudes(manifest, weights)
    fusion, prelu_fusion, _ = plan_fusions(
        manifest, overflow_bounds, fractional_bits=fractional_bits,
        ring_bits=ring_bits, kernel_bits=kernel_bits,
        alpha_magnitudes=alpha_magnitudes, optimizations=optimizations)
    prelu_producers = {fused.producer: fused for fused in prelu_fusion.values()}
    chunks: list[Any] = []
    offsets: dict[str, int] = {}
    lengths: dict[str, int] = {}
    cursor = 0

    def append(name: str, values: Any) -> None:
        nonlocal cursor
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        offsets[name], lengths[name] = cursor, int(flat.size)
        chunks.append(flat)
        cursor += int(flat.size)

    for op in manifest.ops:
        if op.name in fusion.adds:
            channels = op.output_shape[-1]
            fused = sum(_channel_bias(name, ops, refs, weights)
                        for name in fusion.adds[op.name])
            if fused.size != channels:
                raise ValueError(f"{op.name}: fused bias does not match "
                                 f"{channels} output channels")
            append(_add_fused_bias_name(op.name),
                   np.tile(fused, _size(op.output_shape) // channels))
        if op.kind == "BatchNormalization" and len(op.tensors) == 4:
            arrays = [weights[refs[name].offset:refs[name].offset + refs[name].length] for name in op.tensors]
            gamma, beta, mean, variance = arrays
            if not op.params.get("scale", True):
                gamma = np.ones_like(mean)
            if not op.params.get("center", True):
                beta = np.zeros_like(mean)
            scale = gamma / np.sqrt(variance + float(op.params["epsilon"]))
            append(f"{op.name}:affine_scale", scale)
            append(f"{op.name}:affine_bias", beta - mean * scale)
            continue
        for index, name in enumerate(op.tensors):
            ref = refs[name]
            values = weights[ref.offset:ref.offset + ref.length]
            if index == 0 and op.kind in {"Conv2D", "DepthwiseConv2D"} \
                    and kernel_scale != 1.0:
                values = np.asarray(values, dtype=np.float64) * kernel_scale
            append(name, values)
        if op.kind == "PReLU":
            alpha_ref = refs[op.tensors[0]]
            alpha = weights[
                alpha_ref.offset:alpha_ref.offset + alpha_ref.length
            ]
            output_size = _size(op.output_shape)
            if output_size % alpha_ref.length:
                raise ValueError(
                    f"{op.name}: alpha length {alpha_ref.length} does not broadcast "
                    f"over {output_size} outputs"
                )
            fused = prelu_fusion.get(op.name)
            if fused is not None:
                # The fused truncation shifts by f + alpha_bits, so the slope
                # enters the ring quantized to alpha_bits while the writer still
                # applies the one global 2^f scale.
                alpha = np.rint(np.asarray(alpha, dtype=np.float64)
                                * (1 << fused.alpha_bits)) / (1 << fractional_bits)
            append(_prelu_expanded_alpha_name(op.name),
                   np.tile(alpha, output_size // alpha_ref.length))
        bias_name = _conv_bias_name(op, fusion, prelu_producers)
        if bias_name is not None:
            bias_ref = refs[op.tensors[1]]
            bias = weights[bias_ref.offset:bias_ref.offset + bias_ref.length]
            output_size = _size(op.output_shape)
            if output_size % bias_ref.length:
                raise ValueError(
                    f"{op.name}: bias length {bias_ref.length} does not broadcast "
                    f"over {output_size} outputs"
                )
            if op.name in prelu_producers:
                # Summed into an accumulator that still carries the product
                # scale, one public-kernel scale above the writer's own.
                bias = np.asarray(bias, dtype=np.float64) * (1 << kernel_bits)
            append(bias_name, np.tile(bias, output_size // bias_ref.length))
        if op.kind == "DepthwiseConv2D":
            kernel_h, kernel_w = op.params["kernel_size"]
            channels = op.output_shape[-1]
            kernel_ref = refs[op.tensors[0]]
            kernel = weights[
                kernel_ref.offset:kernel_ref.offset + kernel_ref.length
            ].reshape(kernel_h * kernel_w, channels)
            for lane in packed_plan.get(op.name, []):
                append(
                    _depthwise_packed_name(op.name, lane),
                    np.tile(kernel, (1, lane)) * kernel_scale,
                )
    values = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float64)
    return WeightPlan(values, offsets, lengths)

def _public_input_path(output: Path) -> Path:
    if output.parent.name == "Source":
        return output.parent.parent / "Public-Input" / output.stem
    return output.with_suffix(".public-input")

def _quantize_public_weights(values: Any, fractional_bits: int,
                             ring_bits: int) -> Any:
    """Return the exact signed integers ingested by ``public_input``.

    Keeping this conversion separate from the text writer gives the clear ring
    oracle the same source of truth as the generated program.  In particular,
    callers must pass the prepared ``WeightPlan.values`` rather than the raw
    manifest weights: that plan already carries raw convolution biases at
    ``2^(f + f_w)``, PReLU slopes at ``alpha_bits`` and expanded/fused parameters.
    """
    import numpy as np

    scaled = np.rint(values * (1 << fractional_bits))
    limit = 1 << (ring_bits - 1)
    if np.any(scaled < -limit) or np.any(scaled >= limit):
        raise OverflowError("a public model parameter is outside the selected ring")
    return scaled.astype(np.int64)

def _write_public_weights(path: Path, values: Any, fractional_bits: int,
                          ring_bits: int) -> None:
    import numpy as np

    scaled = _quantize_public_weights(values, fractional_bits, ring_bits)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for start in range(0, len(scaled), 65536):
            np.savetxt(handle, scaled[start:start + 65536], fmt="%d")

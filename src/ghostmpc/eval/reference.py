"""Clear NumPy interpreter of the network.

Without ``fixed_point`` it evaluates the float model.  Passing
``fixed_point=(fractional_bits, total_bits)`` switches it to the signed-integer
ring oracle: it consumes the same prepared and quantized public-weight stream
as MP-SPDZ, applies the same fusion plan, and reproduces the declared-length
nearest truncations, so its embeddings are the ones the three servers compute.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..core.model import Manifest

FAITHFUL_ORACLE_VERSION = "ring-integer-v1"


@dataclass(slots=True)
class _RingStatic:
    manifest: Any
    weights: Any
    bounds: dict[str, float]
    fusion: Any
    prelu_fusion: dict[str, Any]
    multiply_fusion: dict[str, Any]
    plan: Any
    public: Any
    refs: dict[str, Any]
    ops: dict[str, Any]


def _build_ring_static(manifest_path: str, fractional_bits: int,
                       total_bits: int, kernel_bits: int,
                       ring_bits: int,
                       overflow_bounds: dict[str, float] | None) -> _RingStatic:
    """Prepare immutable graph/weight state shared by every image batch."""
    import numpy as np

    from ..codegen.fusion import (
        _accumulator_bounds,
        _fusion_plan,
        _multiply_fusion,
        _prelu_alpha_magnitudes,
        _prelu_fusion,
        manifest_bounds,
    )
    from ..codegen.weight_stream import _prepare_weights, _quantize_public_weights

    path = Path(manifest_path)
    manifest = Manifest.load(path)
    weights_path = path.parent / manifest.weights_file
    weights = (np.load(weights_path, mmap_mode="r") if weights_path.is_file()
               else np.empty(0, dtype=np.float64))
    bounds = (overflow_bounds
              or manifest_bounds(manifest)
              or _accumulator_bounds(
                  manifest, weights, fractional_bits=fractional_bits,
                  total_bits=total_bits))
    alpha_magnitudes = _prelu_alpha_magnitudes(manifest, weights)
    # The budgets price the product at ``f + f_w`` and the prepared stream
    # writes kernels and raw biases on that grid.
    fusion = _fusion_plan(
        manifest, bounds=bounds, fractional_bits=fractional_bits,
        ring_bits=ring_bits, kernel_bits=kernel_bits)
    prelu_fusion = _prelu_fusion(
        manifest, bounds, fractional_bits, fusion, ring_bits=ring_bits,
        alpha_magnitudes=alpha_magnitudes, kernel_bits=kernel_bits)
    multiply_fusion = _multiply_fusion(
        manifest, bounds, fractional_bits, fusion, ring_bits=ring_bits,
        kernel_bits=kernel_bits)
    plan = _prepare_weights(
        manifest, weights, fractional_bits=fractional_bits,
        ring_bits=ring_bits, overflow_bounds=bounds,
        alpha_magnitudes=alpha_magnitudes, kernel_bits=kernel_bits)
    public = _quantize_public_weights(
        plan.values, fractional_bits, ring_bits)
    return _RingStatic(
        manifest=manifest, weights=weights, bounds=bounds, fusion=fusion,
        prelu_fusion=prelu_fusion, multiply_fusion=multiply_fusion,
        plan=plan, public=public,
        refs={ref.name: ref for ref in manifest.tensors},
        ops={op.name: op for op in manifest.ops})


@lru_cache(maxsize=8)
def _cached_ring_static(manifest_path: str, fractional_bits: int,
                        total_bits: int, kernel_bits: int,
                        ring_bits: int) -> _RingStatic:
    return _build_ring_static(
        manifest_path, fractional_bits, total_bits, kernel_bits, ring_bits,
        None)


def _evaluate_ring_numpy(manifest_path: Path, input_value: Any, *,
                         fractional_bits: int, total_bits: int,
                         ring_bits: int,
                         weight_fractional_bits: int | None,
                         overflow_bounds: dict[str, float] | None,
                         capture: bool) -> Any:
    """Evaluate the emitted program on signed ring integers.

    Rounding floating-point layer outputs to a grid is not the program emitted
    by :mod:`ghostmpc.codegen`: public kernels are rounded before the first
    multiplication, special biases/slopes live at different scales, and every
    ``TruncRing`` wraps at its declared length.  This interpreter consumes the
    exact prepared public-weight integers and mirrors those operations directly.
    """
    import numpy as np

    from .. import codegen
    from ..codegen.fusion import (
        _add_truncand_bits,
        _raw_scale_bits,
        _raw_scale_units,
    )
    from ..codegen.layers.pooling import _pooled_bound
    from ..codegen.naming import (
        _add_fused_bias_name,
        _conv_expanded_bias_name,
        _conv_raw_bias_name,
        _prelu_expanded_alpha_name,
    )

    if ring_bits < 2 or ring_bits > 64:
        raise ValueError("ring_bits must be in [2, 64] for the NumPy oracle")
    if weight_fractional_bits is None:
        # The profile carries the public-kernel grid, so a caller that does not
        # override it evaluates exactly the arithmetic the manifest declares.
        weight_fractional_bits = \
            Manifest.load(manifest_path).quantization.weight_fractional_bits
    kernel_bits = (fractional_bits if weight_fractional_bits is None else
                   int(weight_fractional_bits))
    if kernel_bits < fractional_bits:
        raise ValueError("weight_fractional_bits must be >= fractional_bits")
    if total_bits + kernel_bits >= ring_bits:
        raise ValueError(
            f"f_w={kernel_bits}, k={total_bits} leaves no nearest-rounding "
            f"bit in ring{ring_bits}")
    separate_kernel_scale = kernel_bits != fractional_bits
    scale = 1 << fractional_bits
    real_input = np.asarray(input_value, dtype=np.float64)
    limit = float(1 << (total_bits - fractional_bits))
    if real_input.size and float(np.max(np.abs(real_input))) >= limit:
        raise OverflowError(
            f"fixed-point value exceeds 2^{total_bits - fractional_bits} "
            f"at f={fractional_bits}, k={total_bits}")

    resolved_path = str(manifest_path.resolve())
    static = (_cached_ring_static(
        resolved_path, fractional_bits, total_bits, kernel_bits, ring_bits)
        if overflow_bounds is None else
        _build_ring_static(
            resolved_path, fractional_bits, total_bits, kernel_bits, ring_bits,
            overflow_bounds))
    manifest = static.manifest
    bounds = static.bounds
    fusion = static.fusion
    prelu_fusion = static.prelu_fusion
    multiply_fusion = static.multiply_fusion
    plan = static.plan
    public = static.public
    fused_convs = {item.conv: item for item in multiply_fusion.values()}
    raw_multiplies = {item.multiply for item in multiply_fusion.values()}
    prelu_producers = {item.producer for item in prelu_fusion.values()}

    def pub(name: str, shape: tuple[int, ...] | list[int] | None = None) -> Any:
        start = plan.offsets[name]
        data = public[start:start + plan.lengths[name]]
        return data.reshape(shape) if shape is not None else data

    def wrap(value: Any, bits: int = ring_bits) -> Any:
        array = np.asarray(value, dtype=np.int64)
        if bits == 64:
            return array
        modulus, half = 1 << bits, 1 << (bits - 1)
        return ((array + half) % modulus - half).astype(np.int64)

    def trunc(value: Any, length: int, shift: int) -> Any:
        if length > ring_bits:
            raise ValueError(
                f"declared length {length} exceeds ring width {ring_bits}")
        if length == ring_bits:
            first = wrap(value) >> (shift - 1)
            return wrap((first + 1) >> 1)
        # ``sint.round(k, m, nearest=True)`` adds the rounding term and calls
        # TruncRing at k+1, as reflected by the generated subroutine names.
        rounded = wrap(value + (1 << (shift - 1)), length + 1)
        return wrap(rounded >> shift)

    def fixed_mul(left: Any, right: Any) -> Any:
        return trunc(wrap(np.asarray(left, dtype=np.int64) *
                          np.asarray(right, dtype=np.int64)),
                     total_bits + fractional_bits, fractional_bits)

    refs = static.refs

    def kernel(op: Any) -> Any:
        # The prepared stream already carries kernels on the ``2^f_w`` grid, so
        # the oracle reads exactly the integers the program consumes instead of
        # re-deriving them.
        ref = refs[op.tensors[0]]
        return pub(op.tensors[0], ref.shape)

    def convolve(op: Any, source: Any) -> Any:
        source = np.asarray(source, dtype=np.int64)
        batched = source.ndim == 4
        sh, sw = op.params["strides"]
        kh, kw = op.params["kernel_size"]
        oh, ow, _ = op.output_shape
        weights_q = kernel(op)
        if batched and op.name not in fused_convs:
            # TensorFlow's float64 CPU kernels are substantially faster for the
            # five-dataset gate. Normal convolution accumulators are below
            # 2^53, so binary64 addition/multiplication is exact. A fused SE
            # consumer can use up to 56 declared bits; those seven 1x1
            # convolutions deliberately fall through to NumPy int64 below.
            try:  # optional ``export`` dependency; NumPy remains the fallback
                import tensorflow as tf

                tensor = tf.convert_to_tensor(source, dtype=tf.float64)
                public_kernel = tf.convert_to_tensor(weights_q, dtype=tf.float64)
                strides = [1, sh, sw, 1]
                padding = op.params["padding"].upper()
                if op.kind == "Conv2D":
                    result = tf.nn.conv2d(
                        tensor, public_kernel, strides=strides,
                        padding=padding)
                else:
                    result = tf.nn.depthwise_conv2d(
                        tensor, public_kernel, strides=strides,
                        padding=padding)
                return wrap(np.rint(result.numpy()).astype(np.int64))
            except ImportError:
                pass
        if op.params["padding"] == "SAME":
            ph = max((oh - 1) * sh + kh - source.shape[-3], 0)
            pw = max((ow - 1) * sw + kw - source.shape[-2], 0)
            spatial_pad = ((ph // 2, ph - ph // 2),
                           (pw // 2, pw - pw // 2), (0, 0))
            source = np.pad(source, (((0, 0),) + spatial_pad)
                            if batched else spatial_pad)
        result_shape = ((source.shape[0],) + tuple(op.output_shape)
                        if batched else tuple(op.output_shape))
        result = np.zeros(result_shape, dtype=np.int64)
        if op.kind == "Conv2D":
            for ky in range(kh):
                for kx in range(kw):
                    patch = (source[:, ky:ky + oh * sh:sh,
                                    kx:kx + ow * sw:sw]
                             if batched else
                             source[ky:ky + oh * sh:sh,
                                    kx:kx + ow * sw:sw])
                    result = wrap(result + np.tensordot(
                        patch, weights_q[ky, kx], axes=([-1], [0])))
        else:
            for ky in range(kh):
                for kx in range(kw):
                    patch = (source[:, ky:ky + oh * sh:sh,
                                    kx:kx + ow * sw:sw]
                             if batched else
                             source[ky:ky + oh * sh:sh,
                                    kx:kx + ow * sw:sw])
                    product = patch[..., :, None] * weights_q[ky, kx]
                    result = wrap(result + product.reshape(result_shape))
        return result

    values: dict[str, Any] = {"input": wrap(np.rint(real_input * scale))}
    if manifest.ops:
        for name in manifest.ops[0].inputs:
            values[name] = values["input"]
    raw_values: dict[str, Any] = {}
    ops = static.ops
    remaining_consumers = Counter(
        name for graph_op in manifest.ops for name in graph_op.inputs)
    captured: dict[str, Any] = {}

    for op in manifest.ops:
        inputs = [values[name] for name in op.inputs if name in values]
        emit_raw = (op.name in fusion.raw or op.name in prelu_producers
                    or op.name in raw_multiplies)
        emit_truncated = (op.name not in raw_multiplies and
                          (not emit_raw or
                           (op.name in fusion.truncated and
                            op.name not in prelu_producers)))
        y = None

        if op.kind in {"Conv2D", "DepthwiseConv2D"}:
            source = (raw_values[op.inputs[0]] if op.name in fused_convs
                      else values[op.inputs[0]])
            accumulator = convolve(op, source)
            if emit_raw:
                raw_values[op.name] = accumulator
            if emit_truncated:
                if op.name in fused_convs:
                    item = fused_convs[op.name]
                    y = trunc(accumulator,
                              item.accumulator_bits + 2 * fractional_bits
                              + kernel_bits + 1,
                              fractional_bits + kernel_bits)
                else:
                    y = trunc(accumulator, total_bits + kernel_bits,
                              kernel_bits)
                bias_name = _conv_expanded_bias_name(op.name)
                if len(op.tensors) > 1 and bias_name in plan.offsets:
                    y = wrap(y + pub(bias_name, op.output_shape))

        elif op.kind == "BatchNormalization":
            channels = op.output_shape[-1]
            repeats = int(np.prod(op.output_shape)) // channels
            affine_scale = np.tile(pub(f"{op.name}:affine_scale"), repeats).reshape(
                op.output_shape)
            affine_bias = np.tile(pub(f"{op.name}:affine_bias"), repeats).reshape(
                op.output_shape)
            y = wrap(fixed_mul(values[op.inputs[0]], affine_scale) + affine_bias)

        elif op.kind == "PReLU":
            source = values.get(op.inputs[0])
            item = prelu_fusion.get(op.name)
            if item is not None:
                accumulator = raw_values[item.producer]
                bias_name = _conv_raw_bias_name(item.producer)
                if bias_name in plan.offsets:
                    accumulator = wrap(accumulator +
                                       pub(bias_name, op.output_shape))
                alpha = pub(_prelu_expanded_alpha_name(op.name),
                            op.output_shape)
                sign_bits = (item.accumulator_bits + fractional_bits
                             + kernel_bits)
                # ``less_than(0, sign_bits)`` interprets only the declared
                # low-order signed window.  Preserve that behavior even for a
                # synthetic/OOD accumulator outside the calibrated bound.
                # MP-SPDZ's signed ``less_than(..., k)`` admits k magnitude
                # bits plus the sign bit (its LTZ subroutine operates on k+1
                # ring bits).
                positive = wrap(accumulator, sign_bits + 1) >= 0
                slope = np.where(positive, 1 << item.alpha_bits, alpha)
                product = wrap(accumulator * slope)
                if capture:
                    captured[f"{op.name}:acc"] = accumulator.copy()
                    captured[f"{op.name}:product"] = product.copy()
                y = trunc(product,
                          item.accumulator_bits + fractional_bits + kernel_bits
                          + item.slope_mag_bits + 1,
                          kernel_bits + item.alpha_bits)
            else:
                alpha = pub(_prelu_expanded_alpha_name(op.name),
                            op.output_shape)
                negative = fixed_mul(source, alpha)
                y = np.where(source >= 0, source, negative).astype(np.int64)

        elif op.kind == "Concatenate":
            axis = int(op.params.get("axis", -1))
            if emit_truncated:
                y = np.concatenate([values[name] for name in op.inputs], axis=axis)
            if emit_raw:
                units = [_raw_scale_units(name, ops, fused_convs)
                         for name in op.inputs]
                target_units = max(units)
                parts = [wrap(raw_values[name] *
                              (1 << (fractional_bits * (target_units - unit))))
                         for name, unit in zip(op.inputs, units)]
                raw_values[op.name] = np.concatenate(parts, axis=axis)

        elif op.kind == "Add":
            if op.name in fusion.adds:
                units = [_raw_scale_units(name, ops, fused_convs)
                         for name in op.inputs]
                target_units = max(units)
                terms = [wrap(raw_values[name] *
                              (1 << (fractional_bits * (target_units - unit))))
                         for name, unit in zip(op.inputs, units)]
                accumulator = terms[0]
                for term in terms[1:]:
                    accumulator = wrap(accumulator + term)
                raw_values[op.name] = accumulator
                scale_bits = _raw_scale_bits(
                    target_units, fractional_bits, kernel_bits)
                if target_units > 2 or separate_kernel_scale:
                    chain_bits = int(_add_truncand_bits(
                        op.inputs, ops, bounds))
                    y = trunc(accumulator, chain_bits + scale_bits + 1,
                              scale_bits - fractional_bits)
                else:
                    y = trunc(accumulator, total_bits + fractional_bits,
                              fractional_bits)
                y = wrap(y + pub(_add_fused_bias_name(op.name),
                                 op.output_shape))
            else:
                y = inputs[0]
                for value in inputs[1:]:
                    y = wrap(y + value)

        elif op.kind == "Multiply":
            if op.name in raw_multiplies:
                item = multiply_fusion[op.name]
                feature = values[item.feature]
                gate = values[item.gate]
                raw_values[op.name] = wrap(feature * np.broadcast_to(
                    gate, feature.shape))
            else:
                y = values[op.inputs[0]]
                for name in op.inputs[1:]:
                    y = fixed_mul(y, np.broadcast_to(values[name], y.shape))

        elif op.kind == "GlobalAveragePooling2D":
            source = values[op.inputs[0]]
            summed = wrap(np.sum(source, axis=(-3, -2), dtype=np.int64))
            reciprocal, shift, length = codegen.pooling_reciprocal(
                int(source.shape[-3]) * int(source.shape[-2]),
                _pooled_bound(op, ops, bounds),
                fractional_bits=fractional_bits, total_bits=total_bits,
                ring_bits=ring_bits)
            y = trunc(wrap(summed * reciprocal), length, shift)

        elif op.kind in {"Reshape", "Flatten"}:
            source = values[op.inputs[0]]
            batched = source.ndim == len(ops[op.inputs[0]].output_shape) + 1
            target = (tuple(op.params["target_shape"])
                      if op.kind == "Reshape" else (-1,))
            y = source.reshape(((source.shape[0],) + target)
                               if batched else target)

        elif op.kind == "Activation":
            source = values[op.inputs[0]]
            if op.params["activation"] == "linear":
                y = source.copy()
            elif op.params["activation"] in {"sigmoid", "hard_sigmoid"}:
                y = wrap(fixed_mul(int(np.rint(0.2 * scale)), source) +
                         int(np.rint(0.5 * scale)))
                y = np.clip(y, 0, scale).astype(np.int64)
            else:
                raise ValueError(f"unsupported activation {op.params['activation']}")
        else:
            raise ValueError(f"unsupported op {op.kind}")

        if y is not None:
            values[op.name] = wrap(y)
            if capture:
                captured[op.name] = values[op.name].astype(np.float64) / scale
        if capture and op.name in raw_values:
            captured[f"{op.name}:raw"] = raw_values[op.name].copy()
        # Accuracy gates evaluate large batches. Keeping all 216 intermediate
        # tensors alive scales to several GiB even though the DAG has a much
        # smaller live frontier. Drop normal/raw views after their last graph
        # consumer; captured diagnostics already own their copy.
        for name in op.inputs:
            remaining_consumers[name] -= 1
            if remaining_consumers[name] == 0:
                values.pop(name, None)
                raw_values.pop(name, None)

    output = values[manifest.ops[-1].name].astype(np.float64) / scale
    return (output, captured) if capture else output


def evaluate_numpy(manifest_path: Path, input_value: Any,
                   *, capture: bool = False,
                   fixed_point: tuple[int, int] | None = None,
                   ring_bits: int | None = None,
                   weight_fractional_bits: int | None = None,
                   overflow_bounds: dict[str, float] | None = None) -> Any:
    """Evaluate the network on one image or a batch of images.

    With ``fixed_point=(f, k)`` this is the ``ring-integer-v1`` oracle of the
    MPC program; without it, the float model the manifest describes.
    """
    import numpy as np

    if fixed_point is not None:
        fractional_bits, total_bits = fixed_point
        selected_ring_bits = (ring_bits if ring_bits is not None else
                              (64 if total_bits >= 31 else 32))
        return _evaluate_ring_numpy(
            manifest_path, input_value,
            fractional_bits=fractional_bits, total_bits=total_bits,
            ring_bits=selected_ring_bits,
            weight_fractional_bits=weight_fractional_bits,
            overflow_bounds=overflow_bounds, capture=capture)

    manifest = Manifest.load(manifest_path)
    flat = np.load(manifest_path.parent / manifest.weights_file)
    refs = {ref.name: ref for ref in manifest.tensors}

    def tensor(name: str) -> Any:
        ref = refs[name]
        shape = ref.shape if np.prod(ref.shape) == ref.length else [ref.length]
        return flat[ref.offset:ref.offset + ref.length].reshape(shape)

    values: dict[str, Any] = {"input": np.asarray(input_value, dtype=np.float64)}
    if manifest.ops:
        for name in manifest.ops[0].inputs:
            values[name] = values["input"]
    captured: dict[str, Any] = {}
    for op in manifest.ops:
        inputs = [values[name] for name in op.inputs]
        x = inputs[0]
        if op.kind in {"Conv2D", "DepthwiseConv2D"}:
            kernel = tensor(op.tensors[0]).astype(np.float64)
            sh, sw = op.params["strides"]
            kh, kw = op.params["kernel_size"]
            oh, ow, _ = op.output_shape
            if op.params["padding"] == "SAME":
                ph = max((oh - 1) * sh + kh - x.shape[0], 0)
                pw = max((ow - 1) * sw + kw - x.shape[1], 0)
                x = np.pad(x, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2), (0, 0)))
            y = np.zeros(op.output_shape, dtype=np.float64)
            for ky in range(kh):
                for kx in range(kw):
                    patch = x[ky:ky + oh * sh:sh, kx:kx + ow * sw:sw]
                    if op.kind == "Conv2D":
                        y += np.tensordot(patch, kernel[ky, kx], axes=([2], [0]))
                    else:
                        product = patch[..., :, None] * kernel[ky, kx][None, None, :, :]
                        y += product.reshape(op.output_shape)
            if len(op.tensors) > 1:
                y += tensor(op.tensors[1]).reshape((1, 1, -1))
        elif op.kind == "BatchNormalization":
            gamma, beta, mean, variance = [tensor(name) for name in op.tensors]
            if not op.params.get("scale", True):
                gamma = np.ones_like(mean)
            if not op.params.get("center", True):
                beta = np.zeros_like(mean)
            y = (x - mean) * gamma / np.sqrt(variance + op.params["epsilon"]) + beta
        elif op.kind == "PReLU":
            y = np.where(x >= 0, x, x * tensor(op.tensors[0]))
        elif op.kind == "Concatenate":
            y = np.concatenate(inputs, axis=op.params["axis"])
        elif op.kind == "Add":
            y = sum(inputs)
        elif op.kind == "Multiply":
            y = inputs[0]
            for value in inputs[1:]:
                y = y * value
        elif op.kind == "GlobalAveragePooling2D":
            y = x.mean(axis=(0, 1))
        elif op.kind == "Reshape":
            y = x.reshape(op.params["target_shape"])
        elif op.kind == "Flatten":
            y = x.reshape(-1)
        elif op.kind == "Activation":
            kind = op.params["activation"]
            if kind == "linear":
                y = x
            elif kind in {"sigmoid", "hard_sigmoid"}:
                y = np.clip(0.2 * x + 0.5, 0, 1)
            else:
                raise ValueError(f"unsupported activation {kind}")
        else:
            raise ValueError(f"unsupported op {op.kind}")
        values[op.name] = y
        if capture:
            captured[op.name] = y.copy()
    output = values[manifest.ops[-1].name]
    return (output, captured) if capture else output

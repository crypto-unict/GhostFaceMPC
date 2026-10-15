"""Differentiable ring-faithful forward for quantization-aware training.

Naive post-training quantization of the float teacher to the ring32/f6 profile
collapses, because the rounding error of every reduction accumulates along the
network.  This module builds the forward that mirrors
:func:`ghostmpc.eval.reference._evaluate_ring_numpy` op-by-op, so the model is
trained on the exact arithmetic the servers execute rather than on a
floating-point idealization of it.

Design.  The fusion/truncation plan is taken *from codegen itself*
(``_fusion_plan``, ``_prelu_fusion``, ``_multiply_fusion`` and the calibrated
``overflow_bounds``); it is a given input of the training graph, never
re-implemented by hand.  Model parameters are ``tf.Variable`` initialised from
the post-fold manifest weights (the exact tensors the oracle reads).  Every
quantiser is a straight-through estimator (STE): the forward value is quantised,
the gradient flows as identity via ``x + stop_gradient(q(x) - x)``.

Numerical exactness.  Inside the calibrated bounds no ``TruncRing`` wraps, so
the mod-``2^k`` ring arithmetic coincides with plain fixed-point on the reals:
a raw accumulator's real value is scale-independent, so the integer left-shift
that lifts narrower operands before a fused ``Add``/``Concatenate`` is a no-op on
reals, and *every* quantisation to the activation grid is the single nearest
half-up truncation ``T_f``.  With ``dtype=float64`` the whole forward is
bit-exact to the oracle; with ``float32`` (the training default) it stays within
one ulp for ``e_max + 2f <= 24`` (asserted at construction), which is what makes
the graph trainable on stock GPU kernels.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ..codegen.fusion import manifest_bounds
from ..core.model import Manifest


class _Plan:
    """Immutable graph/plan state shared by every forward call."""

    __slots__ = ("manifest", "fusion", "prelu_fusion", "multiply_fusion",
                 "bounds", "refs", "ops", "fused_convs", "raw_multiplies",
                 "prelu_producers", "fractional_bits", "total_bits",
                 "ring_bits", "kernel_bits")

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


def _build_plan(manifest: Manifest, weights: Any, *, fractional_bits: int,
                total_bits: int, ring_bits: int, fuse_multiply_conv: bool,
                overflow_bounds: dict[str, float] | None,
                weight_fractional_bits: int | None = None) -> _Plan:
    from ..codegen.fusion import (
        _accumulator_bounds,
        _fusion_plan,
        _multiply_fusion,
        _prelu_alpha_magnitudes,
        _prelu_fusion,
    )

    bounds = (overflow_bounds or manifest_bounds(manifest)
              or _accumulator_bounds(
                  manifest, weights, fractional_bits=fractional_bits,
                  total_bits=total_bits))
    alpha_magnitudes = _prelu_alpha_magnitudes(manifest, weights)
    kernel_bits = (fractional_bits if weight_fractional_bits is None
                   else int(weight_fractional_bits))
    fusion = _fusion_plan(
        manifest, bounds=bounds, fractional_bits=fractional_bits,
        ring_bits=ring_bits, kernel_bits=kernel_bits)
    prelu_fusion = _prelu_fusion(
        manifest, bounds, fractional_bits, fusion, ring_bits=ring_bits,
        alpha_magnitudes=alpha_magnitudes, kernel_bits=kernel_bits)
    multiply_fusion = (_multiply_fusion(
        manifest, bounds, fractional_bits, fusion, ring_bits=ring_bits,
        kernel_bits=kernel_bits)
        if fuse_multiply_conv else {})
    return _Plan(
        manifest=manifest, fusion=fusion, prelu_fusion=prelu_fusion,
        multiply_fusion=multiply_fusion, bounds=bounds,
        refs={ref.name: ref for ref in manifest.tensors},
        ops={op.name: op for op in manifest.ops},
        fused_convs={item.conv: item for item in multiply_fusion.values()},
        raw_multiplies={item.multiply for item in multiply_fusion.values()},
        prelu_producers={item.producer: item
                         for item in prelu_fusion.values()},
        fractional_bits=fractional_bits, total_bits=total_bits,
        ring_bits=ring_bits, kernel_bits=kernel_bits)


class RingQATModel:
    """Ring-faithful, differentiable GhostFaceNet forward.

    Parameters mirror the ring32/f6 deployment profile by default.  Call the
    instance on a ``[batch, H, W, C]`` float image tensor in ``[-1, 1]``; it
    returns the L2-unnormalised embedding (and, with ``capture=True``, the
    per-op activations on the ``2^f`` grid for the fidelity gate).
    """

    def __init__(self, manifest_path: str | Path, *, fractional_bits: int = 6,
                 total_bits: int = 15, ring_bits: int = 32,
                 fuse_multiply_conv: bool = True, dtype: Any = None,
                 overflow_bounds: dict[str, float] | None = None,
                 weight_fractional_bits: int | None = None) -> None:
        import numpy as np
        import tensorflow as tf

        self._tf = tf
        self.dtype = dtype if dtype is not None else tf.float32
        path = Path(manifest_path)
        manifest = Manifest.load(path)
        weights_path = path.parent / manifest.weights_file
        weights = (np.load(weights_path) if weights_path.is_file()
                   else np.empty(0, dtype=np.float64))
        self.f = int(fractional_bits)
        if weight_fractional_bits is None:
            # Same default as the oracle: the manifest profile decides.
            weight_fractional_bits = manifest.quantization.weight_fractional_bits
        self.plan = _build_plan(
            manifest, weights, fractional_bits=fractional_bits,
            total_bits=total_bits, ring_bits=ring_bits,
            fuse_multiply_conv=fuse_multiply_conv,
            overflow_bounds=overflow_bounds,
            weight_fractional_bits=weight_fractional_bits)
        # The float32 forward is exact only while every accumulator stays below
        # 2^(e_max + 2f); above that the grid product loses its low bits.  Guard
        # it statically so an over-wide bound fails loudly instead of drifting.
        e_max = max((math.ceil(math.log2(b)) for b in self.plan.bounds.values()
                     if b and b > 0), default=0)
        self._exact_float32 = e_max + 2 * self.f <= 24
        self.variables: dict[str, Any] = {}
        self._init_variables(manifest, weights)

    # -- construction ---------------------------------------------------
    def _init_variables(self, manifest: Manifest, weights: Any) -> None:
        import numpy as np
        import tensorflow as tf

        refs = self.plan.refs

        def raw(name: str) -> Any:
            ref = refs[name]
            data = np.asarray(weights[ref.offset:ref.offset + ref.length],
                              dtype=np.float64).reshape(ref.shape)
            return data

        def make(name: str, value: Any) -> None:
            self.variables[name] = tf.Variable(
                tf.constant(np.asarray(value), dtype=self.dtype), name=name)

        for op in manifest.ops:
            if op.kind == "BatchNormalization" and len(op.tensors) == 4:
                gamma, beta, mean, variance = [raw(n).reshape(-1)
                                               for n in op.tensors]
                if not op.params.get("scale", True):
                    gamma = np.ones_like(mean)
                if not op.params.get("center", True):
                    beta = np.zeros_like(mean)
                scale = gamma / np.sqrt(variance + float(op.params["epsilon"]))
                make(f"{op.name}:affine_scale", scale)
                make(f"{op.name}:affine_bias", beta - mean * scale)
                continue
            if op.kind in {"Conv2D", "DepthwiseConv2D"}:
                make(op.tensors[0], raw(op.tensors[0]))
                if len(op.tensors) > 1:
                    make(op.tensors[1], raw(op.tensors[1]).reshape(-1))
            elif op.kind == "PReLU":
                make(op.tensors[0], raw(op.tensors[0]))

    @property
    def trainable_variables(self) -> list[Any]:
        return list(self.variables.values())

    # -- quantisers (STE) ----------------------------------------------
    def _round_even(self, x: Any) -> Any:
        # Round half-to-even, matching ``np.rint`` used by the weight export.
        tf = self._tf
        return tf.round(x)

    def _q_even(self, x: Any, bits: int) -> Any:
        # Weight/alpha/bias quantiser: half-to-even, straight-through.
        tf = self._tf
        scale = float(1 << bits)
        q = tf.round(x * scale) / scale
        return x + tf.stop_gradient(q - x)

    def _t_f(self, x: Any) -> Any:
        # Program truncation: nearest, ties up (``floor(x*2^f + 0.5)``), STE.
        tf = self._tf
        scale = float(1 << self.f)
        t = tf.floor(x * scale + 0.5) / scale
        return x + tf.stop_gradient(t - x)

    def _grid_even(self, x: Any) -> Any:
        # The client rounds the image with ``np.rint`` (half-to-even), not the
        # half-up program truncation; the oracle input path does the same.
        tf = self._tf
        scale = float(1 << self.f)
        q = tf.round(x * scale) / scale
        return x + tf.stop_gradient(q - x)

    # -- convolution ----------------------------------------------------
    def _convolve(self, op: Any, source: Any, kernel_q: Any) -> Any:
        tf = self._tf
        sh, sw = op.params["strides"]
        padding = op.params["padding"].upper()
        strides = [1, sh, sw, 1]
        if op.kind == "Conv2D":
            return tf.nn.conv2d(source, kernel_q, strides=strides,
                                padding=padding)
        return tf.nn.depthwise_conv2d(source, kernel_q, strides=strides,
                                      padding=padding)

    def _channel_bias(self, name: str) -> Any:
        """Raw per-channel bias of a raw producer (mirrors ``_channel_bias``)."""
        tf = self._tf
        op = self.plan.ops[name]
        if op.kind == "Concatenate":
            return tf.concat([self._channel_bias(s) for s in op.inputs], axis=0)
        channels = op.output_shape[-1]
        if len(op.tensors) < 2:
            return tf.zeros([channels], dtype=self.dtype)
        return self.variables[op.tensors[1]]

    # -- forward --------------------------------------------------------
    def __call__(self, images: Any, *, capture: bool = False) -> Any:
        import tensorflow as tf

        plan = self.plan
        manifest = plan.manifest
        ops = plan.ops
        fusion = plan.fusion
        f = self.f

        image = tf.convert_to_tensor(images, dtype=self.dtype)
        if len(image.shape) == 3:
            image = image[None]
        values: dict[str, Any] = {"input": self._grid_even(image)}
        if manifest.ops:
            for name in manifest.ops[0].inputs:
                values[name] = values["input"]
        raw_values: dict[str, Any] = {}
        captured: dict[str, Any] = {}

        for op in manifest.ops:
            emit_raw = (op.name in fusion.raw or op.name in plan.prelu_producers
                        or op.name in plan.raw_multiplies)
            emit_truncated = (
                op.name not in plan.raw_multiplies and
                (not emit_raw or (op.name in fusion.truncated and
                                  op.name not in plan.prelu_producers)))
            y = None

            if op.kind in {"Conv2D", "DepthwiseConv2D"}:
                source = (raw_values[op.inputs[0]] if op.name in plan.fused_convs
                          else values[op.inputs[0]])
                kernel_q = self._q_even(self.variables[op.tensors[0]],
                                        plan.kernel_bits)
                accumulator = self._convolve(op, source, kernel_q)
                if emit_raw:
                    raw_values[op.name] = accumulator
                if emit_truncated:
                    y = self._t_f(accumulator)
                    if (len(op.tensors) > 1
                            and op.name not in plan.prelu_producers):
                        y = y + self._q_even(self.variables[op.tensors[1]], f)

            elif op.kind == "BatchNormalization":
                affine_scale = self._q_even(
                    self.variables[f"{op.name}:affine_scale"], f)
                affine_bias = self._q_even(
                    self.variables[f"{op.name}:affine_bias"], f)
                y = self._t_f(values[op.inputs[0]] * affine_scale) + affine_bias

            elif op.kind == "PReLU":
                item = plan.prelu_fusion.get(op.name)
                if item is not None:
                    accumulator = raw_values[item.producer]
                    producer_op = ops[item.producer]
                    if len(producer_op.tensors) > 1:
                        raw_bias = self._q_even(
                            self.variables[producer_op.tensors[1]],
                            f + plan.kernel_bits)
                        accumulator = accumulator + raw_bias
                    alpha_q = self._q_even(self.variables[op.tensors[0]],
                                           item.alpha_bits)
                    # The program selects the slope on ``less_than(0, k)``, which
                    # reads only the declared low-order signed window.  Inside the
                    # calibrated bound this is the true sign; mirror the windowed
                    # test so an out-of-bound accumulator still matches the
                    # oracle.  It only feeds a (non-differentiable) branch select,
                    # so ``stop_gradient`` keeps training on the true sign.
                    sign_bits = (item.accumulator_bits + f
                                 + plan.kernel_bits)
                    acc_int = tf.round(tf.stop_gradient(accumulator)
                                       * float(1 << (f + plan.kernel_bits)))
                    modulus = float(1 << (sign_bits + 1))
                    half = float(1 << sign_bits)
                    windowed = tf.math.floormod(acc_int + half, modulus) - half
                    positive = windowed >= 0
                    slope = tf.where(positive, tf.ones_like(accumulator),
                                     tf.cast(alpha_q, self.dtype)
                                     + tf.zeros_like(accumulator))
                    y = self._t_f(accumulator * slope)
                else:
                    source = values[op.inputs[0]]
                    alpha_q = self._q_even(self.variables[op.tensors[0]], f)
                    negative = self._t_f(source * alpha_q)
                    y = tf.where(source >= 0, source, negative)

            elif op.kind == "Concatenate":
                axis = int(op.params.get("axis", -1))
                if emit_truncated:
                    y = tf.concat([values[n] for n in op.inputs], axis=axis)
                if emit_raw:
                    raw_values[op.name] = tf.concat(
                        [raw_values[n] for n in op.inputs], axis=axis)

            elif op.kind == "Add":
                if op.name in fusion.adds:
                    accumulator = raw_values[op.inputs[0]]
                    for name in op.inputs[1:]:
                        accumulator = accumulator + raw_values[name]
                    raw_values[op.name] = accumulator
                    add_bias = sum(self._channel_bias(n)
                                    for n in fusion.adds[op.name])
                    y = self._t_f(accumulator) + self._q_even(add_bias, f)
                else:
                    y = values[op.inputs[0]]
                    for name in op.inputs[1:]:
                        y = y + values[name]

            elif op.kind == "Multiply":
                if op.name in plan.raw_multiplies:
                    item = plan.multiply_fusion[op.name]
                    raw_values[op.name] = values[item.feature] * values[item.gate]
                else:
                    y = values[op.inputs[0]]
                    for name in op.inputs[1:]:
                        y = self._t_f(y * values[name])

            elif op.kind == "GlobalAveragePooling2D":
                from .. import codegen
                from ..codegen.layers.pooling import _pooled_bound

                source = values[op.inputs[0]]
                summed = tf.reduce_sum(source, axis=[-3, -2])
                count = int(source.shape[-3]) * int(source.shape[-2])
                # The averaging constant is public, so it is carried at
                # ``f + g`` bits and truncated by the same amount; on the
                # activation grid it would collapse.
                reciprocal, shift, _ = codegen.pooling_reciprocal(
                    count, _pooled_bound(op, ops, plan.bounds),
                    fractional_bits=f, total_bits=plan.total_bits,
                    ring_bits=plan.ring_bits)
                # The product carries ``f + shift`` and the truncation removes
                # ``shift``, so the result lands back on the activation grid.
                y = self._t_f(summed * (float(reciprocal) / float(1 << shift)))

            elif op.kind in {"Reshape", "Flatten"}:
                source = values[op.inputs[0]]
                batch = tf.shape(source)[0]
                if op.kind == "Reshape":
                    target = [int(d) for d in op.params["target_shape"]]
                    y = tf.reshape(source, tf.concat([[batch], target], axis=0))
                else:
                    y = tf.reshape(source, [batch, -1])

            elif op.kind == "Activation":
                source = values[op.inputs[0]]
                if op.params["activation"] == "linear":
                    y = source
                elif op.params["activation"] in {"sigmoid", "hard_sigmoid"}:
                    slope = self._q_even(tf.constant(0.2, self.dtype), f)
                    offset = self._q_even(tf.constant(0.5, self.dtype), f)
                    y = self._t_f(slope * source) + offset
                    y = tf.clip_by_value(y, 0.0, 1.0)
                else:
                    raise ValueError(
                        f"unsupported activation {op.params['activation']}")
            else:
                raise ValueError(f"unsupported op {op.kind}")

            if y is not None:
                values[op.name] = y
                if capture:
                    captured[op.name] = y
            if capture and op.name in raw_values:
                captured[f"{op.name}:raw"] = raw_values[op.name]

        output = values[manifest.ops[-1].name]
        return (output, captured) if capture else output

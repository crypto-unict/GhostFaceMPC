"""Truncation fusion planning.

A linear layer normally truncates once per output.  When the consumer is
itself a truncating step the producer can stay *raw* and let the consumer
round once for the pair, provided the widened accumulator still fits the ring:
``e + 2f + 1 <= ring_bits - 1``.  The plans built here decide, per tensor,
whether that inequality holds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..core.model import Manifest, Op
from .naming import MIN_ALPHA_BITS, _fused_truncation_bits, _size


@dataclass(frozen=True)
class FusionPlan:
    """Which accumulators stay untruncated so an Add truncates once.

    ``adds`` maps an eligible Add to the raw operands it sums, ``raw`` lists the
    ops that publish an untruncated accumulator, and ``truncated`` the subset of
    those that still have a consumer needing the scaled value.
    """

    adds: dict[str, tuple[str, ...]]
    raw: frozenset[str]
    truncated: frozenset[str]

NO_FUSION = FusionPlan({}, frozenset(), frozenset())

def manifest_bounds(manifest: Manifest) -> dict[str, float]:
    """The calibrated overflow bounds a manifest carries, as a fresh dict."""
    return dict(manifest.overflow_bounds or {})

def _raw_producible(name: str, ops: dict[str, Op]) -> bool:
    """Whether ``name`` can be published as an untruncated accumulator."""
    op = ops.get(name)
    if op is None:
        return False
    if op.kind == "Conv2D":
        return True
    if op.kind == "DepthwiseConv2D":
        return int(op.params.get("depth_multiplier", 1)) == 1
    if op.kind == "Concatenate":
        return bool(op.inputs) and all(_raw_producible(i, ops) for i in op.inputs)
    return False

def _raw_chain_bits(name: str, ops: dict[str, Op],
                    bounds: dict[str, float]) -> float:
    """Declared integer bit length of the widest raw accumulator under ``name``.

    A convolution carries its own bound; a concatenation carries the widest of
    the values it stitches together.  Missing bounds read as infinite, so an
    unbounded producer is never fused by mistake.
    """
    bound = _raw_chain_bound(name, ops, bounds)
    if math.isinf(bound):
        return math.inf
    return math.ceil(math.log2(bound))

def _raw_chain_bound(name: str, ops: dict[str, Op],
                     bounds: dict[str, float]) -> float:
    """Per-element magnitude bound of the widest raw accumulator under ``name``.

    A convolution carries its own bound; a channel ``Concatenate`` carries the
    widest of the values it stitches together (each element is one of its
    parts).  Missing bounds read as infinite so an unbounded producer never fuses
    by mistake.
    """
    op = ops.get(name)
    if op is not None and op.kind == "Concatenate":
        return max((_raw_chain_bound(source, ops, bounds)
                    for source in op.inputs), default=math.inf)
    bound = bounds.get(name)
    if bound is None or bound <= 0:
        return math.inf
    return bound

def _add_truncand_bits(inputs: tuple[str, ...] | list[str], ops: dict[str, Op],
                       bounds: dict[str, float]) -> float:
    """Integer bit length a fused ``Add`` truncand needs above the sign bit.

    The add is elementwise, so its magnitude is bounded by the sum of its
    operands' per-element bounds, not the widest single one.  Sizing the declared
    length from that sum keeps the ``2^{3f}`` truncand from wrapping when both
    branches peak together.
    """
    total = sum(_raw_chain_bound(name, ops, bounds) for name in inputs)
    if math.isinf(total) or total <= 0:
        return math.inf
    return math.ceil(math.log2(total))

def _raw_scale_bits(units: int, fractional_bits: int, kernel_bits: int) -> int:
    """Fractional bits a raw accumulator of ``units`` carries.

    One unit is the public-kernel scale and every further unit an activation
    scale, so a plain conv accumulator holds ``f_w + f`` and an SE-fused one
    ``f_w + 2f``.  With ``f_w = f`` this is the historical ``units * f``, and the
    gap between consecutive units stays ``f`` either way, which is why the public
    left shift that lifts a narrower operand is unaffected.
    """
    return kernel_bits + max(units - 1, 0) * fractional_bits

def _raw_scale_units(name: str, ops: dict[str, Op],
                     fused_convs: dict[str, Any]) -> int:
    """Scale exponent, in units of ``2^f``, that a raw accumulator carries.

    A normal conv or depthwise reads a truncated ``2^f`` input and multiplies by
    the ``2^f`` weights, so its raw accumulator sits at ``2^{2f}`` (two units).  A
    conv fed by a fused SE ``Multiply`` instead reads a ``2^{2f}`` product and
    lands at ``2^{3f}`` (three units).  A ``Concatenate`` stitches raw operands,
    so it carries the widest exponent among them; the emitter lifts the narrower
    ones with a public left shift.
    """
    if name in fused_convs:
        return 3
    op = ops.get(name)
    if op is not None and op.kind == "Concatenate":
        return max((_raw_scale_units(source, ops, fused_convs)
                    for source in op.inputs), default=2)
    return 2

def _fusion_plan(manifest: Manifest, *, bounds: dict[str, float] | None = None,
                 fractional_bits: int = 20, ring_bits: int = 64,
                 kernel_bits: int | None = None) -> FusionPlan:
    """Defer truncation across Add ops whose operands are both conv outputs.

    ``round(a) + round(b)`` and ``round(a + b)`` differ by at most one ulp, so
    an Add fed only by convolution accumulators can sum the raw values and pay
    for a single truncation.  The biases of the deferred producers move into one
    fused bias added after that truncation.  A raw accumulator carries the
    product scale ``2^{2f}`` on top of its ``e`` integer bits, so the sum only
    stays inside the ring while ``e + 2f + 1 <= ring_bits - 1``; when bounds are
    supplied every add is gated on that, which at ``ring64`` excludes nothing
    (``e`` is ~13, ``2f`` is 40) but at ``ring32`` drops the chains without
    headroom instead of wrapping them.
    """
    kernel_bits = fractional_bits if kernel_bits is None else kernel_bits
    # A raw accumulator carries the product of an activation and a public
    # kernel, hence ``f + f_w`` fractional bits rather than ``2f``.
    product_bits = fractional_bits + kernel_bits
    ops = {op.name: op for op in manifest.ops}
    consumers: dict[str, list[str]] = {}
    for op in manifest.ops:
        for name in op.inputs:
            consumers.setdefault(name, []).append(op.name)

    def fits(add_inputs: tuple[str, ...]) -> bool:
        if bounds is None:
            return True
        chain_bits = max((_raw_chain_bits(name, ops, bounds)
                          for name in add_inputs), default=math.inf)
        return chain_bits + product_bits + 1 <= ring_bits - 1

    adds = {
        op.name: tuple(op.inputs)
        for op in manifest.ops
        if op.kind == "Add" and len(op.inputs) > 1
        and all(_raw_producible(name, ops) for name in op.inputs)
        and fits(tuple(op.inputs))
    }
    raw: set[str] = set()
    raw_edges: set[tuple[str, str]] = set()
    pending = [(name, add) for add, names in adds.items() for name in names]
    while pending:
        name, consumer = pending.pop()
        raw_edges.add((name, consumer))
        if name in raw:
            continue
        raw.add(name)
        if ops[name].kind == "Concatenate":
            pending.extend((source, name) for source in ops[name].inputs)
    truncated = {
        name for name in raw
        if any((name, consumer) not in raw_edges
               for consumer in consumers.get(name, []))
        or not consumers.get(name)
    }
    # A concatenation that owes someone a scaled value can only build it from
    # scaled inputs, so the truncations it defers have to come back.
    pending = sorted(truncated)
    while pending:
        op = ops[pending.pop()]
        if op.kind != "Concatenate":
            continue
        for name in op.inputs:
            if name not in truncated:
                truncated.add(name)
                pending.append(name)
    return FusionPlan(adds, frozenset(raw), frozenset(truncated))

def _channel_bias(name: str, ops: dict[str, Op], refs: dict[str, Any],
                  weights: Any) -> Any:
    """Per-channel bias of a raw producer, in output-channel order."""
    import numpy as np

    op = ops[name]
    channels = op.output_shape[-1]
    if op.kind == "Concatenate":
        return np.concatenate(
            [_channel_bias(source, ops, refs, weights) for source in op.inputs]
        )
    if len(op.tensors) < 2:
        return np.zeros(channels, dtype=np.float64)
    ref = refs[op.tensors[1]]
    bias = np.asarray(weights[ref.offset:ref.offset + ref.length],
                      dtype=np.float64)
    if bias.size != channels:
        raise ValueError(f"{name}: bias length {bias.size} does not match "
                         f"{channels} output channels")
    return bias

@dataclass(frozen=True)
class PreluFusion:
    """A PReLU folded into the truncation of the convolution that feeds it.

    ``accumulator_bits`` is the integer bit length of the producer output and
    ``alpha_bits`` the scale the negative slope is quantized to.  ``slope_mag_bits``
    is the magnitude bit length of the largest quantized slope the layer actually
    multiplies in: it equals ``alpha_bits`` when every ``|alpha| <= 1`` but grows
    when a channel has ``|alpha| > 1``, since then ``|round(alpha * 2^alpha_bits)|``
    exceeds ``2^alpha_bits`` and needs the extra integer bits inside the ring.
    Sizing the fused truncation from ``alpha_bits`` alone understates the product
    width and can push a large accumulator past the declared length, so both the
    planner (eligibility) and the emitter (truncation length) read this instead.
    """

    producer: str
    alpha_bits: int
    accumulator_bits: int
    slope_mag_bits: int

def _accumulator_bounds(manifest: Manifest, weights: Any, *,
                        fractional_bits: int, total_bits: int) -> dict[str, float]:
    """Magnitude bound of every convolution output, from the public weights.

    Any representable operand satisfies ``|x| < 2^(k - f)``, so one output
    channel is bounded by that times the sum of the taps reaching it, plus its
    bias.  Only public data enters the bound, so publishing it leaks nothing
    about an inference.
    """
    import numpy as np

    refs = {tensor.name: tensor for tensor in manifest.tensors}

    def magnitudes(name: str) -> Any:
        ref = refs[name]
        return np.abs(np.asarray(weights[ref.offset:ref.offset + ref.length],
                                 dtype=np.float64))

    input_bound = float(1 << (total_bits - fractional_bits))
    bounds: dict[str, float] = {}
    for op in manifest.ops:
        if op.kind not in {"Conv2D", "DepthwiseConv2D"} or not op.tensors:
            continue
        channels = op.output_shape[-1]
        # Both a Keras kernel and a depthwise kernel end on the dimension that
        # the NHWC output channel indexes, so one reshape covers both.
        taps = magnitudes(op.tensors[0]).reshape(-1, channels).sum(axis=0)
        bias = magnitudes(op.tensors[1]) if len(op.tensors) > 1 else 0.0
        bounds[op.name] = float(np.max(taps * input_bound + bias))
    return bounds

def _prelu_alpha_magnitudes(manifest: Manifest, weights: Any) -> dict[str, float]:
    """Largest ``|alpha|`` of every fusible PReLU, read from public weights.

    The slope is public, so its magnitude leaks nothing about an inference; the
    planner needs it to size the fused truncation for coefficients above one.
    """
    import numpy as np

    refs = {tensor.name: tensor for tensor in manifest.tensors}
    magnitudes: dict[str, float] = {}
    for op in manifest.ops:
        if op.kind != "PReLU" or not op.tensors:
            continue
        ref = refs[op.tensors[0]]
        alpha = np.abs(np.asarray(
            weights[ref.offset:ref.offset + ref.length], dtype=np.float64))
        magnitudes[op.name] = float(alpha.max()) if alpha.size else 0.0
    return magnitudes

def _slope_mag_bits(alpha_magnitude: float | None, alpha_bits: int) -> int:
    """Magnitude bit length of the widest quantized slope the layer applies.

    Without a measured ``|alpha|`` (weights absent, e.g. a source-only render)
    the caller falls back to ``alpha_bits``, which is exact whenever every slope
    stays within one and the historical, conservative choice otherwise.
    """
    if alpha_magnitude is None:
        return alpha_bits
    q_alpha_max = int(round(alpha_magnitude * (1 << alpha_bits)))
    return math.ceil(math.log2(max(1 << alpha_bits, q_alpha_max)))

def _prelu_fusion_plan(manifest: Manifest, bounds: dict[str, float],
                       fractional_bits: int,
                       deferred: frozenset[str] = frozenset(),
                       *, ring_bits: int = 64,
                       alpha_magnitudes: dict[str, float] | None = None,
                       kernel_bits: int | None = None,
                       ) -> dict[str, PreluFusion]:
    """Fold every PReLU that is the only consumer of its convolution.

    A positive scale does not move the sign of an accumulator, so the slope can
    be selected on the raw value and multiplied in before a single truncation
    that replaces both the producer's and the PReLU's.  A layer only qualifies
    while the product of accumulator and slope still fits the ring, which the
    public weight bound decides without looking at any secret.
    """
    kernel_bits = fractional_bits if kernel_bits is None else kernel_bits
    budget = _fused_truncation_bits(ring_bits)
    ops = {op.name: op for op in manifest.ops}
    consumers: dict[str, list[str]] = {}
    for op in manifest.ops:
        for name in op.inputs:
            consumers.setdefault(name, []).append(op.name)
    plan: dict[str, PreluFusion] = {}
    for op in manifest.ops:
        if op.kind != "PReLU":
            continue
        producer = ops.get(op.inputs[0]) if op.inputs else None
        if producer is None or consumers.get(producer.name) != [op.name]:
            continue
        if producer.name in deferred or \
                producer.kind not in {"Conv2D", "DepthwiseConv2D"} or \
                not _raw_producible(producer.name, ops):
            continue
        bound = bounds.get(producer.name, 0.0)
        if bound <= 0:
            continue
        accumulator_bits = math.ceil(math.log2(bound))
        alpha_mag = alpha_magnitudes.get(op.name) if alpha_magnitudes else None
        fused = _slope_scaled_fusion(
            producer.name, accumulator_bits, alpha_mag,
            fractional_bits=fractional_bits, kernel_bits=kernel_bits,
            budget=budget)
        if fused is not None:
            plan[op.name] = fused
    return plan

def _slope_scaled_fusion(producer: str, accumulator_bits: int,
                         alpha_magnitude: float | None, *,
                         fractional_bits: int, kernel_bits: int,
                         budget: int) -> PreluFusion | None:
    """The slope-scaled fusion of one pair, or ``None`` when it does not fit."""
    alpha_bits = min(
        fractional_bits,
        budget - fractional_bits - kernel_bits - accumulator_bits,
    )
    # The slope can never be finer than the activation scale, so at a low
    # fractional precision the coarseness floor is that scale, not the
    # historical high-precision floor: otherwise `f < MIN_ALPHA_BITS` would
    # reject every fusion outright.  Accuracy is still gated downstream by the
    # simulator.
    if alpha_bits < min(MIN_ALPHA_BITS, fractional_bits):
        return None
    slope_mag_bits = _slope_mag_bits(alpha_magnitude, alpha_bits)
    # A coefficient above one widens the product past ``alpha_bits``; the
    # accumulator, product scale and that widened slope together must still
    # leave the ring its sign bit and the nearest-rounding carry.
    if accumulator_bits + fractional_bits + kernel_bits + slope_mag_bits > budget:
        return None
    return PreluFusion(producer, alpha_bits, accumulator_bits, slope_mag_bits)

def _prelu_fusion(manifest: Manifest, bounds: dict[str, float] | None,
                  fractional_bits: int, fusion: FusionPlan,
                  *, ring_bits: int = 64,
                  alpha_magnitudes: dict[str, float] | None = None,
                  kernel_bits: int | None = None,
                  ) -> dict[str, PreluFusion]:
    """Single entry point, so the emitter and the weight plan decide alike."""
    bounds = manifest_bounds(manifest) if bounds is None else bounds
    return _prelu_fusion_plan(manifest, bounds, fractional_bits, fusion.raw,
                              ring_bits=ring_bits,
                              alpha_magnitudes=alpha_magnitudes,
                              kernel_bits=kernel_bits)

@dataclass(frozen=True)
class MultiplyFusion:
    """A squeeze-and-excite ``Multiply`` folded into the 1x1 conv it feeds.

    The gate scaling and the convolution share one truncation instead of two:
    the ``Multiply`` publishes the raw ``feature * gate`` product at scale
    ``2^{2f}`` and its sole consumer accumulates ``sum(raw * w)`` at ``2^{3f}``
    before a single ``2f`` shift returns the output to ``2^f``.  ``feature`` is
    the ``[H, W, C]`` operand, ``gate`` the ``[1, 1, C]`` hard-sigmoid, and
    ``accumulator_bits`` the integer bit length of the conv output, which sizes
    the fused truncation and gates eligibility against the ring.
    """

    multiply: str
    conv: str
    feature: str
    gate: str
    accumulator_bits: int

def _reachable_fused_adds(name: str, consumers: dict[str, list[str]],
                          fusion: FusionPlan) -> set[str]:
    """Fused ``Add`` ops a raw producer reaches through raw edges only.

    A raw accumulator flows to an ``Add`` either directly or through the raw
    ``Concatenate`` ops that stitch it together, so the frontier walks forward
    over the raw set until it lands on the adds that pay the truncation.
    """
    found: set[str] = set()
    seen = {name}
    stack = [name]
    while stack:
        current = stack.pop()
        for consumer in consumers.get(current, []):
            if consumer in fusion.adds:
                found.add(consumer)
            elif consumer in fusion.raw and consumer not in seen:
                seen.add(consumer)
                stack.append(consumer)
    return found

def _multiply_fusion_plan(manifest: Manifest, bounds: dict[str, float],
                          fractional_bits: int, *, ring_bits: int = 64,
                          fusion: FusionPlan = NO_FUSION,
                          kernel_bits: int | None = None,
                          ) -> dict[str, MultiplyFusion]:
    """Fold every SE ``Multiply`` whose only consumer is a 1x1 stride-1 conv.

    The gate does not move the accumulator sign, so deferring the ``Multiply``
    truncation into the convolution is exact up to the one rounding they now
    share.  A pair qualifies only while the ``2^{3f}`` accumulator still fits the
    ring, which the public conv-output bound decides without any secret.  A conv
    that also feeds a fused ``Add`` (a raw producer) lifts every accumulator in
    that add chain from ``2^{2f}`` to ``2^{3f}``: the ``Concatenate`` aligns its
    operands with a public left shift and the ``Add`` truncates once by ``2f``.
    Such a pair is admitted only while the widest bound in each reachable add
    chain still leaves the ring the two bits its ``2^{3f}`` truncand needs;
    otherwise the single pair falls back to the current ``2^{2f}`` path.
    """
    ops = {op.name: op for op in manifest.ops}
    shapes = {op.name: op.output_shape for op in manifest.ops}
    consumers: dict[str, list[str]] = {}
    for op in manifest.ops:
        for name in op.inputs:
            consumers.setdefault(name, []).append(op.name)
    kernel_bits = fractional_bits if kernel_bits is None else kernel_bits
    plan: dict[str, MultiplyFusion] = {}
    for op in manifest.ops:
        if op.kind != "Multiply" or len(op.inputs) != 2:
            continue
        channels = op.output_shape[-1]
        gate = feature = None
        for name in op.inputs:
            shape = shapes.get(name)
            if shape is None or shape[-1] != channels:
                gate = feature = None
                break
            if _size(shape) == channels and shape == op.output_shape:
                # A full-size operand whose spatial extent is one is both the
                # gate and the feature; the SE gate is the strictly smaller one.
                feature = name
            elif _size(shape) == channels:
                gate = name
            elif shape == op.output_shape:
                feature = name
        if gate is None or feature is None or gate == feature:
            continue
        downstream = consumers.get(op.name, [])
        if len(downstream) != 1:
            continue
        conv = ops.get(downstream[0])
        if conv is None or conv.kind != "Conv2D":
            continue
        if tuple(conv.params.get("kernel_size", ())) != (1, 1) or \
                tuple(conv.params.get("strides", ())) != (1, 1):
            continue
        bound = bounds.get(conv.name, 0.0)
        if bound <= 0:
            continue
        accumulator_bits = math.ceil(math.log2(bound))
        # The truncand sits at scale ``2^{3f}``; one bit above its magnitude
        # covers the sign, and the nearest-rounding carry stays below it, so the
        # declared length has to leave the ring those two bits.
        if accumulator_bits + 2 * fractional_bits + kernel_bits > \
                _fused_truncation_bits(ring_bits):
            continue
        # A conv that also defers its truncation into a fused Add carries the
        # whole add chain up to ``2^{3f}``.  Every add it reaches then truncates a
        # wider truncand, so the pair is only safe while each of those chains
        # still fits the ring one bit above its sign.
        if conv.name in fusion.raw:
            reachable = _reachable_fused_adds(conv.name, consumers, fusion)
            chain_bits = max(
                (_add_truncand_bits(ops[add].inputs, ops, bounds)
                 for add in reachable), default=0.0)
            if chain_bits + 2 * fractional_bits + kernel_bits + 1 > ring_bits - 1:
                continue
        plan[op.name] = MultiplyFusion(op.name, conv.name, feature, gate,
                                       accumulator_bits)
    return plan

def _multiply_fusion(manifest: Manifest, bounds: dict[str, float] | None,
                     fractional_bits: int, fusion: FusionPlan,
                     *, ring_bits: int = 64,
                     kernel_bits: int | None = None) -> dict[str, MultiplyFusion]:
    """Single entry point, so emitter, weight plan and mirror decide alike.

    A conv that also defers its truncation into a fused ``Add`` is admitted only
    when its whole ``2^{3f}`` add chain still fits the ring; the planner gates it
    on the same bounds as the emitter and the mirror.
    """
    bounds = manifest_bounds(manifest) if bounds is None else bounds
    return _multiply_fusion_plan(manifest, bounds, fractional_bits,
                                 ring_bits=ring_bits, fusion=fusion,
                                 kernel_bits=kernel_bits)

@dataclass(frozen=True)
class Optimizations:
    """Which of the paper's circuit optimizations a program is emitted with.

    Every switch defaults to the deployed choice, so ``Optimizations()`` emits
    the measured program byte for byte; the ablation turns them off one at a
    time.  The fusions only decide which truncations are deferred, and the
    two-summand switch only selects the rounding kernel of the semi-honest
    protocol, so none of them changes what the program computes beyond the one
    ulp a deferred rounding may move.
    """

    fuse_prelu: bool = True
    fuse_multiply: bool = True
    fuse_add: bool = True
    two_summand: bool = True

    #: The command-line names of the switches, in ``--disable`` order.
    SWITCHES = {"prelu-fusion": "fuse_prelu", "se-fusion": "fuse_multiply",
                "add-fusion": "fuse_add", "two-summand": "two_summand"}

    @classmethod
    def without(cls, disabled: "list[str] | tuple[str, ...]") -> "Optimizations":
        """The deployed set with the named switches turned off."""
        unknown = sorted(set(disabled) - set(cls.SWITCHES))
        if unknown:
            raise ValueError(f"unknown optimization(s) {', '.join(unknown)}; "
                             f"choose from {', '.join(cls.SWITCHES)}")
        return cls(**{cls.SWITCHES[name]: False for name in disabled})

    def disabled(self) -> list[str]:
        return [name for name, field in self.SWITCHES.items()
                if not getattr(self, field)]


DEPLOYED = Optimizations()


def plan_fusions(manifest: Manifest, bounds: dict[str, float] | None, *,
                 fractional_bits: int, ring_bits: int, kernel_bits: int,
                 alpha_magnitudes: dict[str, float] | None,
                 optimizations: Optimizations = DEPLOYED,
                 ) -> tuple[FusionPlan, dict[str, PreluFusion],
                            dict[str, MultiplyFusion]]:
    """The Add, PReLU and SE plans, with the disabled ones left empty.

    The emitter and the weight plan both read this, so a switch can never make
    the program index a weight the stream does not carry.
    """
    fusion = (_fusion_plan(manifest, bounds=bounds,
                           fractional_bits=fractional_bits,
                           ring_bits=ring_bits, kernel_bits=kernel_bits)
              if optimizations.fuse_add else NO_FUSION)
    prelu = (_prelu_fusion(manifest, bounds, fractional_bits, fusion,
                           ring_bits=ring_bits,
                           alpha_magnitudes=alpha_magnitudes,
                           kernel_bits=kernel_bits)
             if optimizations.fuse_prelu else {})
    multiply = (_multiply_fusion(manifest, bounds, fractional_bits, fusion,
                                 ring_bits=ring_bits, kernel_bits=kernel_bits)
                if optimizations.fuse_multiply else {})
    return fusion, prelu, multiply

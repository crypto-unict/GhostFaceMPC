"""Overflow bounds measured on real activations rather than on the weights.

The static bound of ``_accumulator_bounds`` admits any representable
operand, so it sits about nine bits above what a face actually produces and
locks most conv->PReLU pairs out of the fused truncation.  A bound calibrated
on real inputs is *distributional, not adversarial*: an out-of-distribution
input can exceed it and wrap the ring, which corrupts that one embedding but
leaks nothing, because the shares stay uniform.  Hence the margin, the clamps,
and the preference for the static bound whenever calibration buys less than
two bits.
"""

from __future__ import annotations

import math
import pickle
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..codegen.fusion import _accumulator_bounds, _fusion_plan, _prelu_fusion
from ..core.model import Manifest
from . import reference

CONV_KINDS = frozenset({"Conv2D", "DepthwiseConv2D"})
# Below this gain the calibrated bound trades soundness for nothing.
MINIMUM_BITS_GAINED = 2


@dataclass(frozen=True)
class LayerBound:
    """The bound of one convolution and where it came from."""

    name: str
    peak: float
    static: float
    calibrated: float
    bound: float
    source: str


def _bits(bound: float) -> int:
    """Declared integer bit length of a bound, as the fusion plan reads it."""
    return math.ceil(math.log2(bound)) if bound > 0 else 0


def load_images(source: Path, count: int, *, offset: int = 0) -> list[Any]:
    """Aligned faces, from a directory of images or an InsightFace ``.bin``.

    Normalization matches ``share-image``: RGB, 112x112, scaled to [-1, 1].
    """
    from io import BytesIO

    import numpy as np

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit("calibration requires `pip install -e .[export]`") from exc

    if source.is_dir():
        files = sorted(p for p in source.iterdir() if p.is_file())
        opened = [Image.open(path) for path in files[offset:offset + count]]
    else:
        with source.open("rb") as handle:
            encoded, _ = pickle.load(handle, encoding="bytes")
        opened = [Image.open(BytesIO(value))
                  for value in encoded[offset:offset + count]]

    images = []
    for image in opened:
        image = image.convert("RGB")
        if image.size != (112, 112):
            image = image.resize((112, 112), Image.BILINEAR)
        pixels = np.asarray(image, dtype=np.float64)
        images.append(pixels / 127.5 - 1.0)
    return images


def calibration_peaks(manifest_path: Path,
                      images: Sequence[Any]) -> dict[str, float]:
    """Largest pre-activation magnitude each convolution reaches on ``images``.

    ``evaluate_numpy`` keeps PReLU as its own op, so a captured convolution
    output is the accumulator the truncation has to hold.  The bound is
    **distributional, not adversarial**, and it is measured on a calibration
    set only: the five verification datasets may never be fit on.
    """
    import numpy as np

    manifest = Manifest.load(manifest_path)
    names = [op.name for op in manifest.ops
             if op.kind in CONV_KINDS and op.tensors]
    peaks = {name: 0.0 for name in names}
    for image in images:
        _, captured = reference.evaluate_numpy(manifest_path, image,
                                               capture=True)
        for name in names:
            values = np.abs(np.asarray(captured[name], dtype=np.float64))
            peaks[name] = max(peaks[name], float(values.max()) if values.size else 0.0)
    return peaks


def layer_bounds(manifest: Manifest, weights: Any, peaks: dict[str, float], *,
                 margin_bits: int, fractional_bits: int,
                 total_bits: int) -> dict[str, LayerBound]:
    """Combine the measured peaks with the static bounds and the clamps."""
    import numpy as np

    static = _accumulator_bounds(
        manifest, weights, fractional_bits=fractional_bits,
        total_bits=total_bits)
    refs = {tensor.name: tensor for tensor in manifest.tensors}

    def bias_magnitude(op: Any) -> float:
        if len(op.tensors) < 2:
            return 0.0
        ref = refs[op.tensors[1]]
        values = np.asarray(weights[ref.offset:ref.offset + ref.length],
                            dtype=np.float64)
        return float(np.max(np.abs(values))) if values.size else 0.0

    result: dict[str, LayerBound] = {}
    for op in manifest.ops:
        if op.name not in peaks:
            continue
        # The bias alone can exceed the calibrated peak of a dead channel, and
        # a bound under 1.0 would claim zero integer bits.
        calibrated = max(peaks[op.name] * float(2 ** margin_bits),
                         bias_magnitude(op), 1.0)
        reference_bound = static[op.name]
        gained = _bits(reference_bound) - _bits(calibrated)
        keep_static = gained < MINIMUM_BITS_GAINED
        result[op.name] = LayerBound(
            op.name, peaks[op.name], reference_bound, calibrated,
            reference_bound if keep_static else calibrated,
            "static" if keep_static else "calibrated")
    return result


def calibrated_bounds(manifest_path: Path, weights: Any, images: Sequence[Any],
                      margin_bits: int = 2, *, fractional_bits: int = 20,
                      total_bits: int = 31) -> dict[str, float]:
    """Overflow bound per convolution, ready for ``manifest.overflow_bounds``."""
    manifest = Manifest.load(manifest_path)
    peaks = calibration_peaks(manifest_path, images)
    bounds = layer_bounds(manifest, weights, peaks, margin_bits=margin_bits,
                          fractional_bits=fractional_bits,
                          total_bits=total_bits)
    return {name: layer.bound for name, layer in bounds.items()}


def validation_margins(manifest_path: Path, bounds: dict[str, float],
                       images: Sequence[Any]) -> dict[str, float]:
    """Margin bits each layer still has on a held-out set: ``log2(bound/peak)``.

    A finite calibration set never observes the supremum -- held-out faces do
    push past the calibration peak -- so what has to hold is that the *bound*
    keeps room to spare, not that the peak was maximal.

    Exceeding a bound is not a leak -- the shares stay uniform -- but it wraps
    the ring and corrupts that embedding, so a layer left with too little margin
    means the calibration set was not representative.
    """
    peaks = calibration_peaks(manifest_path, images)
    margins: dict[str, float] = {}
    for name, peak in peaks.items():
        bound = bounds.get(name, 0.0)
        if bound <= 0:
            continue
        margins[name] = math.inf if peak <= 0 else math.log2(bound / peak)
    return margins


def fusion_audit(manifest: Manifest, layers: dict[str, LayerBound], *,
                 fractional_bits: int) -> dict[str, Any]:
    """What the tighter bounds change in the conv->PReLU fusion plan."""
    fusion = _fusion_plan(manifest)
    static = _prelu_fusion(
        manifest, {name: layer.static for name, layer in layers.items()},
        fractional_bits, fusion)
    chosen = _prelu_fusion(
        manifest, {name: layer.bound for name, layer in layers.items()},
        fractional_bits, fusion)
    shapes = {op.name: op.output_shape for op in manifest.ops}

    def elements(name: str) -> int:
        return math.prod(shapes[name])

    entries = []
    for op in manifest.ops:
        if op.kind != "PReLU":
            continue
        producer = op.inputs[0] if op.inputs else ""
        layer = layers.get(producer)
        before, after = static.get(op.name), chosen.get(op.name)
        entries.append({
            "prelu": op.name,
            "producer": producer,
            "elements": elements(op.name),
            "peak": layer.peak if layer else None,
            "static_bits": _bits(layer.static) if layer else None,
            "calibrated_bits": _bits(layer.calibrated) if layer else None,
            "bound_source": layer.source if layer else None,
            "alpha_bits_static": before.alpha_bits if before else None,
            "alpha_bits_calibrated": after.alpha_bits if after else None,
            "fused_static": before is not None,
            "fused_calibrated": after is not None,
        })
    gained = [entry for entry in entries
              if entry["fused_calibrated"] and not entry["fused_static"]]
    return {
        "fractional_bits": fractional_bits,
        "prelu_total": len(entries),
        "fused_static": sum(entry["fused_static"] for entry in entries),
        "fused_calibrated": sum(entry["fused_calibrated"] for entry in entries),
        "newly_fused": [entry["prelu"] for entry in gained],
        "newly_fused_elements": sum(entry["elements"] for entry in gained),
        "layers": entries,
        "bounds": [asdict(layer) for layer in layers.values()],
    }

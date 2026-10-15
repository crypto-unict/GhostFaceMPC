"""Keras-to-manifest exporter.

The exporter deliberately performs only inference-safe rewrites: BatchNorm is
folded into its immediately preceding linear layer and no activation is
approximated or removed.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

from ..core.model import (
    CHECKPOINT_URL,
    MODEL_ID,
    Manifest,
    Op,
    Quantization,
    TensorRef,
    deployment_quantization,
    digest,
)


class ExportError(RuntimeError):
    pass


def _deps() -> tuple[Any, Any]:
    try:
        import numpy as np
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - exercised in user env
        raise ExportError("export requires `pip install -e .[export]`") from exc
    return np, tf


def _shape(tensor: Any) -> list[int]:
    shape = tensor.shape.as_list()
    return [int(x) for x in shape[1:]]


def _inputs(layer: Any, aliases: dict[str, str]) -> list[str]:
    raw = layer.input if isinstance(layer.input, (list, tuple)) else [layer.input]
    return [aliases.get(t.name, t.name.split(":")[0]) for t in raw]


def _op(layer: Any, aliases: dict[str, str]) -> Op | None:
    kind = layer.__class__.__name__
    if kind == "InputLayer":
        return None
    params: dict[str, Any] = {}
    if kind in {"Conv2D", "DepthwiseConv2D"}:
        params = {
            "kernel_size": list(layer.kernel_size), "strides": list(layer.strides),
            "padding": layer.padding.upper(), "use_bias": bool(layer.use_bias),
        }
        if kind == "Conv2D":
            params["filters"] = int(layer.filters)
        else:
            params["depth_multiplier"] = int(layer.depth_multiplier)
    elif kind == "PReLU":
        params = {"shared_axes": list(layer.shared_axes or [])}
    elif kind in {"AveragePooling2D", "MaxPooling2D"}:
        params = {"pool_size": list(layer.pool_size), "strides": list(layer.strides),
                  "padding": layer.padding.upper()}
    elif kind == "BatchNormalization":
        params = {"epsilon": float(layer.epsilon), "scale": bool(layer.scale), "center": bool(layer.center)}
    elif kind == "Concatenate":
        params = {"axis": int(layer.axis)}
    elif kind in {"Add", "Multiply", "Flatten", "GlobalAveragePooling2D", "Reshape", "Activation"}:
        if kind == "Reshape":
            params = {"target_shape": list(layer.target_shape)}
        elif kind == "Activation":
            activation = layer.activation.__name__
            if activation not in {"linear", "sigmoid", "hard_sigmoid"}:
                raise ExportError(f"unsupported activation {activation} in {layer.name}")
            params = {"activation": activation}
    else:
        raise ExportError(f"unsupported Keras layer {kind} ({layer.name})")
    return Op(layer.name, kind, _inputs(layer, aliases), _shape(layer.output), params)


def _fold_batchnorm(np: Any, op: Op, layer: Any, bn: Any) -> tuple[list[Any], Op]:
    weights = layer.get_weights()
    if not weights:
        raise ExportError(f"linear layer {layer.name} has no weights")
    kernel = weights[0].astype("float64")
    bias = weights[1].astype("float64") if len(weights) > 1 else np.zeros(kernel.shape[-1])
    gamma, beta, mean, variance = [x.astype("float64") for x in bn.get_weights()]
    gamma = gamma if bn.scale else np.ones_like(mean)
    beta = beta if bn.center else np.zeros_like(mean)
    multiplier = gamma / np.sqrt(variance + bn.epsilon)
    if op.kind == "Conv2D":
        kernel = kernel * multiplier.reshape((1, 1, 1, -1))
    else:
        kernel = kernel * multiplier.reshape((1, 1, -1, 1))
    bias = (bias - mean) * multiplier + beta
    op.params["use_bias"] = True
    op.fused_from.append(bn.name)
    return [kernel.astype("float32"), bias.astype("float32")], op


def export_checkpoint(
    checkpoint: Path,
    destination: Path,
    *,
    quantization: Quantization | None = None,
) -> Manifest:
    np, tf = _deps()
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = tf.keras.models.load_model(checkpoint, compile=False)
    destination.mkdir(parents=True, exist_ok=True)

    aliases: dict[str, str] = {}
    ops: list[Op] = []
    tensors: OrderedDict[str, Any] = OrderedDict()
    op_arrays: OrderedDict[str, list[Any]] = OrderedDict()
    layer_by_name: dict[str, Any] = {}
    layers = list(model.layers)
    for layer in layers:
        op = _op(layer, aliases)
        if op is None:
            continue
        op_arrays[op.name] = list(layer.get_weights())
        layer_by_name[op.name] = layer
        ops.append(op)
        aliases[layer.output.name] = op.name

    # Keras' serialization order is not a graph order for residual branches.
    # Fold a BN whenever its linear producer has no other consumer, regardless
    # of where the two layers occur in ``model.layers``.
    consumers: dict[str, int] = {}
    by_name = {op.name: op for op in ops}
    for op in ops:
        for source in op.inputs:
            consumers[source] = consumers.get(source, 0) + 1
    removed: set[str] = set()
    replacements: dict[str, str] = {}
    for bn in ops:
        if bn.kind != "BatchNormalization" or len(bn.inputs) != 1:
            continue
        parent = by_name.get(bn.inputs[0])
        if parent is None or parent.kind not in {"Conv2D", "DepthwiseConv2D"}:
            continue
        if consumers.get(parent.name, 0) != 1:
            continue
        arrays, parent = _fold_batchnorm(
            np, parent, layer_by_name[parent.name], layer_by_name[bn.name])
        op_arrays[parent.name] = arrays
        by_name[parent.name] = parent
        removed.add(bn.name)
        replacements[bn.name] = parent.name

    def resolve(name: str) -> str:
        while name in replacements:
            name = replacements[name]
        return name

    ops = [op for op in ops if op.name not in removed]
    for op in ops:
        op.inputs = [resolve(name) for name in op.inputs]
        arrays = op_arrays[op.name]
        for index, array in enumerate(arrays):
            tensor_name = f"{op.name}:{index}"
            tensors[tensor_name] = np.asarray(array, dtype=np.float32)
            op.tensors.append(tensor_name)

    if not ops:
        raise ExportError("no executable layers found")
    flat = np.concatenate([x.reshape(-1) for x in tensors.values()]) if tensors else np.empty(0, dtype=np.float32)
    weights_file = "weights.npy"
    np.save(destination / weights_file, flat)
    refs: list[TensorRef] = []
    offset = 0
    for name, values in tensors.items():
        refs.append(TensorRef(
            name=name,
            shape=[int(x) for x in values.shape],
            offset=offset,
            length=int(values.size),
            bits=32,
            minimum=float(values.min()) if values.size else None,
            maximum=float(values.max()) if values.size else None,
        ))
        offset += int(values.size)
    manifest = Manifest(
        version=2, model_id=MODEL_ID, checkpoint_url=CHECKPOINT_URL,
        checkpoint_sha256=digest(checkpoint), input_shape=_shape(model.inputs[0]),
        output_shape=ops[-1].output_shape, ops=ops, tensors=refs,
        weights_file=weights_file, weights_sha256=digest(destination / weights_file),
        quantization=(quantization if quantization is not None
                      else deployment_quantization()),
    )
    manifest.save(destination / "manifest.json")
    return manifest

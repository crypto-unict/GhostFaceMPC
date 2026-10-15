"""CrypTFlow/Porthos export and pool-report helpers.

The export starts from the frozen ring64/f16 GhostMPC manifest rather than from
the mixed-precision Keras graph, so the inference-safe rewrites the manifest
already carries (notably folded batch normalization) are identical to the ones
GhostMPC runs, while every protocol-specific lowering is left to Athos.  The
bundle written here is what ``baselines/porthos/build.sh`` compiles; the report
helpers aggregate the pool files ``baselines/porthos/run_porthos_mtls.py``
writes.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..core.model import Manifest, digest

EZPC_URL = "https://github.com/mpc-msri/EzPC.git"
EZPC_COMMIT = "f24bf3e022dde3493cf0ffe3ebd92dd73936b734"
#: Porthos carries one global fixed-point format; the comparison is frozen to
#: the ring64/f16 profile the manifest was exported at.
RING_BITS = 64
FRACTIONAL_BITS = 16
SUPPORTED_KINDS = {
    "Conv2D", "DepthwiseConv2D", "PReLU", "Concatenate", "Add",
    "GlobalAveragePooling2D", "Reshape", "Activation", "Multiply",
    "Flatten", "BatchNormalization",
}
SUMMARY_METRICS = (
    "wall_seconds", "cpu_seconds_total", "peak_rss_bytes", "bytes_total",
    "rounds", "send_calls", "recv_calls",
)
PHASE_METRICS = ("wall_seconds", "cpu_seconds_total", "bytes_total", "rounds")


class PorthosError(RuntimeError):
    pass


def _onnx() -> tuple[Any, Any, Any, Any]:
    """Import the ONNX toolchain, or explain which extra installs it."""
    try:
        import numpy as np
        import onnx
        from onnx import helper, numpy_helper
    except ImportError as exc:  # pragma: no cover - optional environment
        raise PorthosError(
            "this export requires `pip install -e '.[porthos]'`") from exc
    return np, onnx, helper, numpy_helper


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolation percentile; ``None`` for an empty sample."""
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q / 100.0
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


# --------------------------------------------------------------------------
# ONNX export of the frozen manifest
# --------------------------------------------------------------------------

def _nchw_shape(shape: Sequence[int]) -> list[int]:
    if len(shape) == 3:
        height, width, channels = (int(x) for x in shape)
        return [1, channels, height, width]
    if len(shape) == 1:
        return [1, int(shape[0])]
    return [1, *(int(x) for x in shape)]


def _same_pads(input_shape: Sequence[int], output_shape: Sequence[int],
               kernel: Sequence[int], strides: Sequence[int]) -> list[int]:
    ih, iw = (int(input_shape[0]), int(input_shape[1]))
    oh, ow = (int(output_shape[0]), int(output_shape[1]))
    kh, kw = (int(kernel[0]), int(kernel[1]))
    sh, sw = (int(strides[0]), int(strides[1]))
    total_h = max((oh - 1) * sh + kh - ih, 0)
    total_w = max((ow - 1) * sw + kw - iw, 0)
    top, left = total_h // 2, total_w // 2
    return [top, left, total_h - top, total_w - left]


def snapshot_manifest(manifest_path: Path, checkpoint: Path) -> dict[str, Any]:
    """Provenance of the exported model: which manifest, weights and checkpoint."""
    manifest_path = manifest_path.resolve()
    manifest = Manifest.load(manifest_path)
    weights = manifest_path.parent / manifest.weights_file
    return {
        "model_id": manifest.model_id,
        "profile": f"ring{manifest.quantization.ring_bits}-"
                   f"f{manifest.quantization.fractional_bits}",
        "manifest": str(manifest_path),
        "manifest_sha256": digest(manifest_path),
        "weights_sha256": digest(weights),
        "checkpoint_sha256": (digest(checkpoint) if checkpoint.is_file()
                              else manifest.checkpoint_sha256),
        "quantization": asdict(manifest.quantization),
        "operation_count": len(manifest.ops),
        "operation_kinds": dict(sorted(
            Counter(op.kind for op in manifest.ops).items())),
    }


def export_manifest_onnx(manifest_path: Path, output: Path) -> dict[str, Any]:
    """Write the manifest as an NCHW float32 ONNX graph with public initializers."""
    np, onnx, helper, numpy_helper = _onnx()
    manifest_path = manifest_path.resolve()
    manifest = Manifest.load(manifest_path)
    unsupported = sorted({op.kind for op in manifest.ops} - SUPPORTED_KINDS)
    if unsupported:
        raise PorthosError(f"unsupported manifest operations: {unsupported}")

    flat = np.load(manifest_path.parent / manifest.weights_file, mmap_mode="r")
    refs = {ref.name: ref for ref in manifest.tensors}
    values: dict[str, Any] = {}
    initializers: list[Any] = []

    def tensor(name: str) -> Any:
        if name not in values:
            ref = refs[name]
            values[name] = np.asarray(
                flat[ref.offset:ref.offset + ref.length], dtype=np.float32
            ).reshape(ref.shape)
        return values[name]

    def initializer(name: str, array: Any) -> str:
        safe = name.replace(":", "__")
        item = numpy_helper.from_array(np.asarray(array, dtype=np.float32), safe)
        item.doc_string = "MPC_PUBLIC_MODEL_WEIGHT"
        initializers.append(item)
        return safe

    graph_input_name = manifest.ops[0].inputs[0] if manifest.ops else "input"
    input_info = helper.make_tensor_value_info(
        graph_input_name, onnx.TensorProto.FLOAT, _nchw_shape(manifest.input_shape))
    input_info.doc_string = "MPC_SHARED_INPUT"
    nodes: list[Any] = []
    shapes: dict[str, list[int]] = {graph_input_name: list(manifest.input_shape)}

    for op in manifest.ops:
        inputs = list(op.inputs)
        attrs: dict[str, Any] = {}
        node_inputs = inputs.copy()
        kind = op.kind
        if kind in {"Conv2D", "DepthwiseConv2D"}:
            kernel = tensor(op.tensors[0])
            if kind == "Conv2D":
                kernel = np.transpose(kernel, (3, 2, 0, 1))
                group = 1
            else:
                kh, kw, channels, multiplier = kernel.shape
                kernel = np.transpose(kernel, (2, 3, 0, 1)).reshape(
                    channels * multiplier, 1, kh, kw)
                group = int(channels)
            node_inputs.append(initializer(op.tensors[0], kernel))
            if len(op.tensors) > 1:
                node_inputs.append(initializer(op.tensors[1], tensor(op.tensors[1])))
            attrs["kernel_shape"] = [int(x) for x in op.params["kernel_size"]]
            attrs["strides"] = [int(x) for x in op.params["strides"]]
            attrs["group"] = group
            attrs["pads"] = (
                _same_pads(shapes[inputs[0]], op.output_shape,
                           op.params["kernel_size"], op.params["strides"])
                if op.params["padding"] == "SAME" else [0, 0, 0, 0])
            onnx_kind = "Conv"
        elif kind == "PReLU":
            slope = tensor(op.tensors[0]).reshape(-1)
            if len(op.output_shape) == 3:
                # ONNX broadcasts from the trailing dimensions.  A flat Keras
                # slope therefore targets W rather than C after NHWC->NCHW.
                slope = slope.reshape(1, -1, 1, 1)
            node_inputs.append(initializer(op.tensors[0], slope))
            onnx_kind = "PRelu"
        elif kind == "Concatenate":
            attrs["axis"] = 1
            onnx_kind = "Concat"
        elif kind == "Add":
            onnx_kind = "Add"
        elif kind == "Multiply":
            onnx_kind = "Mul"
        elif kind == "GlobalAveragePooling2D":
            onnx_kind = "GlobalAveragePool"
        elif kind == "Reshape":
            target_name = f"{op.name}__shape"
            target = np.asarray(_nchw_shape(op.output_shape), dtype=np.int64)
            item = numpy_helper.from_array(target, target_name)
            item.doc_string = "MPC_PUBLIC_SHAPE"
            initializers.append(item)
            node_inputs.append(target_name)
            onnx_kind = "Reshape"
        elif kind == "Flatten":
            attrs["axis"] = 1
            onnx_kind = "Flatten"
        elif kind == "Activation":
            activation = op.params["activation"]
            if activation == "linear":
                onnx_kind = "Identity"
            elif activation in {"hard_sigmoid", "sigmoid"}:
                onnx_kind = "HardSigmoid"
                attrs.update(alpha=0.2, beta=0.5)
            else:
                raise PorthosError(f"unsupported activation {activation}")
        elif kind == "BatchNormalization":
            if len(op.tensors) != 4:
                raise PorthosError(f"{op.name}: expected four BatchNorm tensors")
            for name in op.tensors:
                node_inputs.append(initializer(name, tensor(name)))
            attrs["epsilon"] = float(op.params["epsilon"])
            onnx_kind = "BatchNormalization"
        else:  # guarded by SUPPORTED_KINDS
            raise PorthosError(f"unsupported operation {kind}")
        nodes.append(helper.make_node(
            onnx_kind, node_inputs, [op.name], name=op.name, **attrs))
        shapes[op.name] = list(op.output_shape)

    output_info = helper.make_tensor_value_info(
        manifest.ops[-1].name, onnx.TensorProto.FLOAT,
        _nchw_shape(manifest.output_shape))
    graph = helper.make_graph(
        nodes, "GhostFaceNet-B0", [input_info], [output_info],
        initializer=initializers)
    model = helper.make_model(
        graph, producer_name="ghostmpc", producer_version="0.1.0",
        opset_imports=[helper.make_opsetid("", 13)])
    model.metadata_props.add(key="ghostmpc.model_id", value=manifest.model_id)
    model.metadata_props.add(key="ghostmpc.profile", value="B0-ring64-f16")
    model.metadata_props.add(key="ghostmpc.weights_sha256", value=manifest.weights_sha256)
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)
    return {
        "onnx": str(output.resolve()),
        "onnx_sha256": digest(output),
        "onnx_nodes": len(nodes),
        "onnx_initializers": len(initializers),
        "onnx_op_types": dict(sorted(Counter(node.op_type for node in nodes).items())),
    }


def _attribute(node: Any, name: str, default: float) -> float:
    return next((float(item.f) for item in node.attribute if item.name == name),
                default)


def lower_onnx_for_porthos(source: Path, output: Path) -> dict[str, Any]:
    """Apply generic Athos lowerings while retaining NCHW and ONNX padding.

    PReLU uses one comparison and a public slope product. BatchNorm becomes a
    public affine. HardSigmoid thresholds are derived from its ONNX attributes;
    no GhostFaceNet-specific tensor name or dimension is used.
    """
    np, onnx, helper, numpy_helper = _onnx()
    model = onnx.shape_inference.infer_shapes(onnx.load(source))
    source_nodes = len(model.graph.node)
    arrays = {item.name: numpy_helper.to_array(item)
              for item in model.graph.initializer}
    shapes = {
        item.name: [int(dim.dim_value) for dim in item.type.tensor_type.shape.dim]
        for item in [*model.graph.input, *model.graph.value_info,
                     *model.graph.output]
    }
    initializers = {item.name for item in model.graph.initializer}
    for item in model.graph.initializer:
        item.doc_string = "MPC_PUBLIC_MODEL_WEIGHT"

    def add_public(name: str, value: Any, *, shape: Sequence[int] | None = None) -> str:
        array = np.asarray(value, dtype=np.float32)
        if shape is not None:
            array = array.reshape(shape)
        item = numpy_helper.from_array(array, name)
        item.doc_string = "MPC_PUBLIC_MODEL_WEIGHT"
        model.graph.initializer.append(item)
        arrays[name] = array
        initializers.add(name)
        return name

    nodes: list[Any] = []
    counts = Counter()
    for node in model.graph.node:
        base = node.name or node.output[0]
        if node.op_type == "PRelu":
            positive = f"{base}__positive"
            negative = f"{base}__negative"
            scaled = f"{base}__negative_scaled"
            nodes.extend([
                helper.make_node("Relu", [node.input[0]], [positive],
                                 name=f"{base}__comparison"),
                helper.make_node("Sub", [node.input[0], positive], [negative],
                                 name=f"{base}__negative_branch"),
                helper.make_node("Mul", [negative, node.input[1]], [scaled],
                                 name=f"{base}__public_slope"),
                helper.make_node("Add", [positive, scaled], list(node.output),
                                 name=base),
            ])
            counts["prelu"] += 1
            continue
        if node.op_type == "BatchNormalization":
            if len(node.input) != 5 or any(name not in arrays for name in node.input[1:]):
                raise PorthosError(f"{base}: BatchNorm parameters must be initializers")
            epsilon = _attribute(node, "epsilon", 1e-5)
            gamma, beta, mean, variance = [
                np.asarray(arrays[name], dtype=np.float32) for name in node.input[1:]]
            scale = gamma / np.sqrt(variance + epsilon)
            offset = beta - mean * scale
            rank = len(shapes.get(node.input[0], []))
            broadcast = [1, int(scale.size)] if rank == 2 else [1, int(scale.size), 1, 1]
            scale_name = add_public(f"{base}__scale", scale, shape=broadcast)
            offset_name = add_public(f"{base}__offset", offset, shape=broadcast)
            product = f"{base}__product"
            nodes.extend([
                helper.make_node("Mul", [node.input[0], scale_name], [product],
                                 name=f"{base}__public_scale"),
                helper.make_node("Add", [product, offset_name], list(node.output),
                                 name=base),
            ])
            counts["batchnorm"] += 1
            continue
        if node.op_type == "HardSigmoid":
            alpha = _attribute(node, "alpha", 0.2)
            beta = _attribute(node, "beta", 0.5)
            if not math.isfinite(alpha) or alpha <= 0 or not math.isfinite(beta):
                raise PorthosError(
                    f"{base}: only finite positive-alpha HardSigmoid is supported")
            lower = -beta / alpha
            upper = (1.0 - beta) / alpha
            # Athos' pinned ONNX frontend does not register rank-zero
            # initializers. A one-element tensor has identical ONNX broadcast
            # semantics and remains a generic public constant.
            lower_name = add_public(
                f"{base}__lower", np.asarray([lower], np.float32))
            upper_name = add_public(
                f"{base}__upper", np.asarray([upper], np.float32))
            alpha_name = add_public(
                f"{base}__alpha", np.asarray([alpha], np.float32))
            shifted_low = f"{base}__shifted_low"
            shifted_high = f"{base}__shifted_high"
            relu_low = f"{base}__relu_low"
            relu_high = f"{base}__relu_high"
            width = f"{base}__width"
            nodes.extend([
                helper.make_node("Sub", [node.input[0], lower_name], [shifted_low],
                                 name=f"{base}__subtract_lower"),
                helper.make_node("Sub", [node.input[0], upper_name], [shifted_high],
                                 name=f"{base}__subtract_upper"),
                helper.make_node("Relu", [shifted_low], [relu_low],
                                 name=f"{base}__relu_lower"),
                helper.make_node("Relu", [shifted_high], [relu_high],
                                 name=f"{base}__relu_upper"),
                helper.make_node("Sub", [relu_low, relu_high], [width],
                                 name=f"{base}__clipped_width"),
                helper.make_node("Mul", [width, alpha_name], list(node.output),
                                 name=base),
            ])
            counts["hard_sigmoid"] += 1
            continue
        nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(nodes)
    metadata = {
        "ghostmpc.backend": "CrypTFlow/Porthos-public",
        "ghostmpc.athos_prelu_lowering": "relu(x)+alpha*(x-relu(x))",
        "ghostmpc.athos_hardsigmoid_lowering":
            "alpha*(relu(x+beta/alpha)-relu(x-(1-beta)/alpha))",
        "ghostmpc.initializers": "public",
    }
    existing = {item.key for item in model.metadata_props}
    for key, value in metadata.items():
        if key not in existing:
            model.metadata_props.add(key=key, value=value)
    onnx.checker.check_model(model)
    model = onnx.shape_inference.infer_shapes(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)
    return {
        "source_nodes": source_nodes,
        "athos_nodes": len(nodes),
        "prelu_lowered": counts["prelu"],
        "batchnorm_lowered": counts["batchnorm"],
        "hard_sigmoid_lowered": counts["hard_sigmoid"],
        "public_initializers": len(model.graph.initializer),
        "sha256": digest(output),
    }


def verify_onnx_float(onnx_path: Path, checkpoint: Path, inputs: Any, *,
                      maximum_error: float = 1e-4,
                      mean_error: float = 1e-5) -> dict[str, Any]:
    """Compare ONNX with the checkpoint's mathematical float32 graph.

    The published H5 serializes ``mixed_float16`` compute policies.  Comparing
    a float32 ONNX graph to those fp16 intermediates measures dtype policy, not
    graph conversion.  We therefore clone the same layers and weights with
    float32 compute, and record both policies explicitly.
    """
    np, _, _, _ = _onnx()
    try:
        import onnxruntime as ort
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - optional environment
        raise PorthosError(
            "float verification requires TensorFlow and onnxruntime") from exc
    original = tf.keras.models.load_model(checkpoint, compile=False)

    def float32_layer(layer: Any) -> Any:
        config = layer.get_config()
        config["dtype"] = "float32"
        return layer.__class__.from_config(config)

    reference = tf.keras.models.clone_model(original, clone_function=float32_layer)
    reference.set_weights(original.get_weights())
    batch = np.asarray(inputs, dtype=np.float32)
    if batch.ndim == 3:
        batch = batch[None]
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feed = {session.get_inputs()[0].name: batch.transpose(0, 3, 1, 2)}
    actual = np.asarray(session.run(None, feed)[0]).reshape(batch.shape[0], -1)
    expected = np.asarray(reference(batch, training=False)).reshape(batch.shape[0], -1)
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    observed_max = float(difference.max(initial=0.0))
    observed_mean = float(difference.mean()) if difference.size else 0.0
    passed = observed_max <= maximum_error and observed_mean <= mean_error
    return {
        "status": "passed" if passed else "failed",
        "samples": int(batch.shape[0]),
        "output_elements": int(expected.shape[1]),
        "maximum_absolute_error": observed_max,
        "mean_absolute_error": observed_mean,
        "limits": {"maximum_absolute_error": maximum_error,
                   "mean_absolute_error": mean_error},
        "checkpoint_serialized_compute_policy": str(original.layers[1].compute_dtype),
        "comparison_compute_policy": "float32",
        "same_weights": True,
    }


# --------------------------------------------------------------------------
# additive ring64 inputs
# --------------------------------------------------------------------------

def share_additive_input(values: Any, *, fractional_bits: int = FRACTIONAL_BITS,
                         seed: int = 0) -> tuple[Any, Any]:
    """Encode ``values`` on the ``2^-f`` grid and split into two ring64 shares."""
    import numpy as np
    scaled = np.rint(np.asarray(values, dtype=np.float64) * (1 << fractional_bits))
    ring = scaled.astype(np.int64, casting="unsafe").view(np.uint64)
    rng = np.random.default_rng(seed)
    left = rng.integers(0, 1 << 64, size=ring.shape, dtype=np.uint64)
    right = ring - left
    return left, right


def reconstruct_additive_output(left: Any, right: Any) -> Any:
    import numpy as np
    return (np.asarray(left, dtype=np.uint64) +
            np.asarray(right, dtype=np.uint64)).view(np.int64)


def _export_gallery_row(destination: Path, *, gallery_rows: int,
                        embedding_size: int, row_index: int, seed: int,
                        error: type[Exception]) -> dict[str, Any]:
    """Write the enrolled row the score reads, as two additive shares.

    Porthos has *one* global fixed-point format (`athos_config["scale"] = 16`),
    so probe and database sit on the same `2^-16` grid.  Our own profile keeps
    the database on a finer `2^-15` grid only because our accumulator is never
    truncated; this row does truncate, so it does not get that.

    The database is secret, and in this integration P2 is a helper that receives
    no application input at all, so the row is split into exactly the two
    additive components P0 and P1 already take their image share in.
    """
    import numpy as np

    from ..core.score import deployment_score_plan
    from ..eval.score import encode_gallery

    if not 0 <= row_index < gallery_rows:
        raise error(
            f"row {row_index} is not in a database of {gallery_rows} rows")
    plan = deployment_score_plan(
        gallery_rows, embedding_size, fractional_bits=FRACTIONAL_BITS,
        gallery_fractional_bits=FRACTIONAL_BITS, ring_bits=RING_BITS)
    # Synthetic, and deliberately so: every row is the same number of ring
    # elements, so the protocol's cost does not depend on the values and a
    # measurement never needs a real enrolment.
    generator = np.random.default_rng(seed)
    rows = generator.normal(size=(plan.gallery_rows, plan.embedding_size))
    encoded = encode_gallery(rows, plan)[row_index]
    ring = encoded.astype(np.int64).view(np.uint64)
    left = generator.integers(0, 1 << 64, size=ring.shape, dtype=np.uint64)
    right = ring - left
    if not np.array_equal(reconstruct_additive_output(left, right), encoded):
        raise error("the gallery row does not reconstruct bit-exactly")
    np.save(destination / "gallery-share-p0.npy", left)
    np.save(destination / "gallery-share-p1.npy", right)
    np.save(destination / "gallery-row.npy", encoded)
    return {
        "gallery_rows": plan.gallery_rows,
        "embedding_size": plan.embedding_size,
        "row_index": row_index,
        "row_index_visibility": "public; the client declares the identity",
        "fractional_bits": plan.fractional_bits,
        "gallery_fractional_bits": plan.gallery_fractional_bits,
        "declared_widths": plan.declared_widths(),
        "grid": "probe and database both on 2^-16: Porthos has one global "
                "fixed-point format",
        "share_files": ["gallery-share-p0.npy", "gallery-share-p1.npy"],
        "p2_gallery_input": False,
        "bit_exact_reconstruction_checked": True,
        "source": f"synthetic(seed={seed})",
    }


def export_porthos_bundle(manifest_path: Path, destination: Path, *,
                          checkpoint: Path, input_path: Path | None = None,
                          seed: int = 0, verify_float: bool = True,
                          score: bool = False, gallery_rows: int = 1024,
                          gallery_row: int = 0) -> dict[str, Any]:
    """Export the frozen ring64/f16 manifest to a Porthos bundle.

    The bundle holds the source NCHW ONNX, the Athos-lowered ONNX, the Athos
    PORTHOS config, the two additive input shares, optionally the two shares
    of the enrolled gallery row, ``config.json`` and ``b0-snapshot.json``.
    """
    np, _, _, _ = _onnx()
    manifest_path = manifest_path.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest = Manifest.load(manifest_path)
    if (manifest.quantization.ring_bits != RING_BITS
            or manifest.quantization.fractional_bits != FRACTIONAL_BITS):
        raise PorthosError("the Porthos baseline is exported from the ring64/f16 manifest")
    if manifest.input_shape != [112, 112, 3] or manifest.output_shape != [512]:
        raise PorthosError(
            f"unexpected manifest shape: input={manifest.input_shape}, output={manifest.output_shape}")

    if input_path is None:
        image = np.random.default_rng(seed).uniform(
            -1.0, 1.0, size=manifest.input_shape).astype(np.float32)
        input_source = "deterministic-synthetic"
    else:
        image = np.asarray(np.load(input_path), dtype=np.float32)
        if image.shape == (1, *manifest.input_shape):
            image = image[0]
        if list(image.shape) != manifest.input_shape:
            raise PorthosError(
                f"expected input shape {manifest.input_shape}, got {list(image.shape)}")
        input_source = str(input_path.resolve())

    source_onnx = destination / "ghostfacenet-b0.onnx"
    target_onnx = destination / "ghostfacenet-b0-porthos.onnx"
    snapshot_path = destination / "b0-snapshot.json"
    snapshot = snapshot_manifest(manifest_path, checkpoint)
    snapshot.update(export_manifest_onnx(manifest_path, source_onnx))
    snapshot["ezpc"] = {"url": EZPC_URL, "commit": EZPC_COMMIT}
    snapshot_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    lowering = lower_onnx_for_porthos(source_onnx, target_onnx)
    validation = (verify_onnx_float(source_onnx, checkpoint, image)
                  if verify_float else {"status": "not-run", "development_only": True})
    if verify_float and validation["status"] != "passed":
        raise PorthosError(f"ONNX float validation failed: {validation}")

    nchw = image.transpose(2, 0, 1)[None]
    share0, share1 = share_additive_input(nchw, fractional_bits=FRACTIONAL_BITS,
                                          seed=seed)
    np.save(destination / "input.npy", nchw)
    np.save(destination / "input-share-p0.npy", share0)
    np.save(destination / "input-share-p1.npy", share1)
    reconstructed = reconstruct_additive_output(share0, share1)
    expected = np.rint(nchw.astype(np.float64) * (1 << FRACTIONAL_BITS)).astype(np.int64)
    if not np.array_equal(reconstructed, expected):
        raise PorthosError("external additive input does not reconstruct bit-exactly")

    gallery = (_export_gallery_row(
        destination, gallery_rows=gallery_rows,
        embedding_size=manifest.output_shape[0], row_index=gallery_row,
        seed=seed, error=PorthosError) if score else None)

    athos_config = {
        "model_name": str(target_onnx.resolve()),
        "input_tensors": {manifest.ops[0].inputs[0]: "1,3,112,112"},
        "target": "PORTHOS", "scale": FRACTIONAL_BITS, "bitlength": RING_BITS,
        "save_weights": True,
        "output_tensors": [manifest.ops[-1].name],
    }
    (destination / "athos-config-porthos.json").write_text(
        json.dumps(athos_config, indent=2, sort_keys=True) + "\n")
    config = {
        "schema_version": 1,
        "backend": "CrypTFlow/Porthos-public",
        "extension": "declared strengthened public-weight variant",
        "ezpc": {"url": EZPC_URL, "commit": EZPC_COMMIT},
        "model": {
            "id": manifest.model_id, "logical_operations": len(manifest.ops),
            "source_onnx": str(source_onnx.resolve()),
            "porthos_onnx": str(target_onnx.resolve()),
            "input_shape": [1, 3, 112, 112], "output_shape": [1, 512],
            "lowering": lowering,
        },
        "input": {
            "source": input_source, "representation": "two external additive shares",
            "share_files": ["input-share-p0.npy", "input-share-p1.npy"],
            "p2_application_input": False, "bit_exact_reconstruction_checked": True,
        },
        "output": {
            "representation": "P0/P1 local additive shares",
            "p2": "metadata only", "online_reconstruction": False,
        },
        "weights": {
            "visibility": "public", "sha256": manifest.weights_sha256,
            "loaded_before_online_timer": True,
            "secret_times_public_mac_traffic_bytes": 0,
        },
        "arithmetic": {
            "ring_bits": RING_BITS, "fractional_bits": FRACTIONAL_BITS,
            # The binary keeps upstream's local-share truncation behind a
            # run-time switch; run_porthos_mtls.py always selects the secure
            # exact nearest truncation (GHOSTMPC_PORTHOS_NEAREST=1).
            "truncation": "secure exact nearest (rounding half up)",
        },
        "security": {
            "parties": 3, "model": "semi-honest, at most one corrupt",
            "transport": "mTLS full mesh", "online_reveal": False,
        },
        "float_validation": validation,
        # Present and null without a score, so a reader never has to infer
        # from a missing key whether a bundle carries a database row.
        "score": gallery,
    }
    config_path = destination / "config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return {
        "onnx": str(target_onnx), "source_onnx": str(source_onnx),
        "snapshot": str(snapshot_path), "config": str(config_path),
        "athos_config": str(destination / "athos-config-porthos.json"),
        "onnx_sha256": lowering["sha256"], "float_validation": validation,
    }


# --------------------------------------------------------------------------
# pool reports written by baselines/porthos/run_porthos_mtls.py
# --------------------------------------------------------------------------

def load_pool_samples(path: Path) -> list[dict[str, Any]]:
    """The measured (non-warm-up) requests of one pool report.

    The offline phase runs once per pool process, not once per request, so it
    is a property of the file; it is carried on every row so that
    :func:`summarize_offline` can summarize it across pools.
    """
    data = json.loads(path.read_text())
    raw = data.get("samples", [])
    if not isinstance(raw, list) or not raw:
        raise PorthosError(f"{path}: no request samples")
    offline = data.get("offline")
    samples = []
    for source in raw:
        if source.get("warmup"):
            continue
        parties = source.get("parties", [])
        sent = sum(int(item["bytes_sent"]) for item in parties)
        received = sum(int(item["bytes_received"]) for item in parties)
        bytes_total = int(source["bytes_total"])
        samples.append({
            "backend": str(data.get("backend", "CrypTFlow/Porthos-public")),
            "pool": int(source.get("pool", 0)),
            "request": int(source["request"]),
            "wall_seconds": float(source["wall_seconds"]),
            "cpu_seconds_total": float(source["cpu_seconds_total"]),
            "peak_rss_bytes": (int(source["peak_rss_bytes"])
                               if source.get("peak_rss_bytes") is not None else None),
            "bytes_total": bytes_total,
            "rounds": (int(source["rounds"])
                       if source.get("rounds") is not None else None),
            "rounds_by_party": source.get("rounds_by_party"),
            "phases": source.get("phases"),
            "send_calls": source.get("send_calls"),
            "recv_calls": source.get("recv_calls"),
            "cpu_affinity": source.get("cpu_affinity"),
            "parties": parties,
            "bytes_sent": sent,
            "bytes_received": received,
            "bytes_reconciled": bool(parties) and sent == received == bytes_total,
            "causal_rounds_validated": bool(source.get("causal_rounds_validated")),
            "offline": offline,
        })
    return samples


def summarize_samples(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for metric in SUMMARY_METRICS:
        values = [float(row[metric]) for row in samples if row.get(metric) is not None]
        result[metric] = {
            "median": percentile(values, 50), "p5": percentile(values, 5),
            "p95": percentile(values, 95), "observations": len(values),
        }
    return result


def summarize_phase(samples: Sequence[Mapping[str, Any]],
                    phase: str) -> dict[str, Any] | None:
    """The distribution of one named per-request phase, or ``None``.

    ``online`` is the window `start_m()`/`end_m()` bracket and is what every
    published row quotes; ``sync_barrier`` is the 2 MB barrier
    ``StartComputation()`` runs *before* that window, once per request.  The
    barrier is excluded from the online figure; summarizing it here reports
    that cost rather than moving it.
    """
    rows = [row["phases"][phase] for row in samples
            if isinstance(row.get("phases"), Mapping)
            and phase in row["phases"]]
    if len(rows) != len(samples) or not rows:
        return None
    result: dict[str, Any] = {}
    for metric in PHASE_METRICS:
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        result[metric] = {
            "median": percentile(values, 50), "p5": percentile(values, 5),
            "p95": percentile(values, 95), "observations": len(values),
        }
    return result


def summarize_offline(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """The setup phase, one observation per pool process.

    Porthos exchanges no correlated randomness -- ``initializeMPC()`` fills
    local mod-prime tables and the PRG keys are pre-shared files -- so what is
    measured here is the connection setup, the one setup barrier, and the
    public weight and gallery streams.  A near-zero round count outside that
    barrier is the protocol's answer, not a missing measurement.
    """
    by_pool: dict[int, Mapping[str, Any]] = {}
    for row in samples:
        offline = row.get("offline")
        if not isinstance(offline, Mapping):
            return None
        by_pool.setdefault(int(row.get("pool", 0)), offline)
    if not by_pool:
        return None
    rows = list(by_pool.values())
    result: dict[str, Any] = {"observations": len(rows), "scope": "per pool process"}
    for metric in PHASE_METRICS:
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        if len(values) != len(rows):
            return None
        result[metric] = {"median": percentile(values, 50),
                          "p5": percentile(values, 5),
                          "p95": percentile(values, 95)}
    items = rows[0].get("items")
    if items is not None:
        result["items"] = list(items)
    return result


__all__ = [
    "EZPC_COMMIT", "EZPC_URL", "PorthosError",
    "export_manifest_onnx", "export_porthos_bundle", "load_pool_samples",
    "lower_onnx_for_porthos", "percentile", "reconstruct_additive_output",
    "share_additive_input", "snapshot_manifest", "summarize_offline",
    "summarize_phase", "summarize_samples", "verify_onnx_float",
]

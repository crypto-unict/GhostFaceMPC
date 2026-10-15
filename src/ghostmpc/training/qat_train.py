"""Quantization-aware fine-tuning against the ring-faithful forward.

The student is :class:`ghostmpc.training.qat_ring.RingQATModel`, whose forward is
bit-exact with the integer oracle.  We distil the frozen float teacher (the
official checkpoint) into the ring32/f6 grid: the loss is a cosine-embedding +
MSE objective, plus an optional per-stage feature term ``L_feat`` (which fights
the cumulative drift that is the root cause of the post-training-quantization
collapse) and a range hinge ``L_range`` that keeps the raw accumulators inside
the calibrated bounds so ``k`` stays narrow.

Only the model parameters move; the fusion/truncation plan is frozen for the run
(re-derived from the manifest, never hand-written).  Because the teacher and
dataset are both public, the training carries no MPC constraint and runs offline
on GPU.  The five verification datasets are never touched here — selection uses
the MS1MV3 validation-split proxy pairs; the accuracy is measured afterwards by
``ghostmpc evaluate-accuracy``.

A run can be split across jobs with a wall-time limit: weights, optimizer, EMA
and step counter are checkpointed atomically and resumed, and ``--max-seconds``
stops the run cleanly (checkpoint, exit 0).  A ``SIGTERM`` trap is the backstop.
"""

from __future__ import annotations

import json
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np

from .guard import _reject_verification_directory
from .qat_ring import RingQATModel


# --------------------------------------------------------------------------- #
# Export: QAT variables -> raw manifest weights the oracle reads
# --------------------------------------------------------------------------- #
def export_qat_manifest(model: RingQATModel, manifest_path: Path,
                        output_dir: Path) -> Path:
    """Write the trained latent weights back into a manifest the oracle reads.

    The variables carry full-precision (latent) values; the oracle quantises
    them with ``np.rint`` exactly as the QAT forward's ``Q_w`` did, so the
    exported manifest reproduces the arithmetic the student trained on.  The
    single folded BatchNorm is written as ``gamma=affine_scale``,
    ``beta=affine_bias``, ``mean=0``, ``var=1-epsilon`` so that the oracle's own
    fold recomputes the identical affine.
    """
    from ..core.model import Manifest, digest

    manifest = Manifest.load(Path(manifest_path))
    refs = {ref.name: ref for ref in manifest.tensors}
    total = max((ref.offset + ref.length for ref in manifest.tensors),
                default=0)
    stream = np.zeros(total, dtype=np.float64)
    variables = {name: var.numpy().astype(np.float64)
                 for name, var in model.variables.items()}

    bn_ops = {op.name: op for op in manifest.ops
              if op.kind == "BatchNormalization" and len(op.tensors) == 4}
    for op in bn_ops.values():
        epsilon = float(op.params["epsilon"])
        scale = variables[f"{op.name}:affine_scale"].reshape(-1)
        bias = variables[f"{op.name}:affine_bias"].reshape(-1)
        gamma = scale
        beta = bias
        mean = np.zeros_like(scale)
        variance = np.full_like(scale, 1.0 - epsilon)
        for ref_name, array in zip(op.tensors, (gamma, beta, mean, variance)):
            ref = refs[ref_name]
            stream[ref.offset:ref.offset + ref.length] = array.reshape(-1)

    bn_tensor_names = {name for op in bn_ops.values() for name in op.tensors}
    for ref in manifest.tensors:
        if ref.name in bn_tensor_names:
            continue
        if ref.name not in variables:
            continue
        stream[ref.offset:ref.offset + ref.length] = \
            variables[ref.name].reshape(-1)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / manifest.weights_file
    np.save(weights_path, stream)
    manifest.weights_sha256 = digest(weights_path)
    for ref in manifest.tensors:
        values = stream[ref.offset:ref.offset + ref.length]
        ref.minimum = float(values.min()) if values.size else None
        ref.maximum = float(values.max()) if values.size else None
    manifest.save(output_dir / "manifest.json")
    return output_dir / "manifest.json"


# --------------------------------------------------------------------------- #
# Proxy accuracy on the MS1MV3 validation split
# --------------------------------------------------------------------------- #
def _decode_pairs(encoded: list[bytes]):
    import tensorflow as tf

    images = []
    for value in encoded:
        image = tf.io.decode_image(value, channels=3, expand_animations=False)
        image = tf.image.resize(image, (112, 112))
        images.append(tf.cast(image, tf.float32) / 127.5 - 1.0)
    return tf.stack(images)


def _embed(forward, encoded: list[bytes], batch_size: int):
    """Batched, L2-normalised embeddings; decodes per batch to bound memory.

    The proxy set is ~24k images, so a single forward would exhaust GPU memory:
    decode and embed one batch at a time (both teacher and student go through
    here).
    """
    embeddings = []
    for start in range(0, len(encoded), batch_size):
        batch = _decode_pairs(encoded[start:start + batch_size])
        emb = np.asarray(forward(batch), dtype=np.float64)
        emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
        embeddings.append(emb)
    return np.concatenate(embeddings)


def proxy_accuracy(model: RingQATModel, encoded: list[bytes], labels, *,
                   batch_size: int = 64) -> float:
    """Verification accuracy on proxy pairs using the QAT forward."""
    from ..eval.face_accuracy import _verification_accuracy

    return _verification_accuracy(
        _embed(lambda x: model(x), encoded, batch_size), labels)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _stream_dataset(shards: Path, batch_size: int, seed: int):
    """Infinite, shuffled stream of train-split faces in ``[-1, 1]``."""
    import tensorflow as tf

    from . import ms1mv3

    def generator():
        for sample in ms1mv3.iter_split(
                ms1mv3.iter_webdataset(shards), "train"):
            yield sample.image

    dataset = tf.data.Dataset.from_generator(
        generator, output_signature=tf.TensorSpec([], tf.string))
    dataset = dataset.shuffle(20_000, seed=seed).repeat()

    def decode(encoded):
        image = tf.io.decode_jpeg(encoded, channels=3)
        image = tf.image.resize(image, (112, 112))
        return tf.cast(image, tf.float32) / 127.5 - 1.0

    dataset = dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size, drop_remainder=True).prefetch(
        tf.data.AUTOTUNE)


def _auto_feature_ops(manifest, count: int = 4) -> list[str]:
    """Evenly spaced residual-``Add`` boundaries as per-stage feature taps."""
    adds = [op.name for op in manifest.ops if op.kind == "Add"]
    if len(adds) <= count:
        return adds
    step = len(adds) / (count + 1)
    return [adds[int(step * (i + 1))] for i in range(count)]


def train_qat_ring(manifest_path: Path, teacher_checkpoint: Path,
                   shards: Path, output_dir: Path, *,
                   fractional_bits: int = 6, total_bits: int = 15,
                   ring_bits: int = 32, epochs: int = 2,
                   steps_per_epoch: int = 36_392, batch_size: int = 128,
                   learning_rate: float = 5e-5, warmup_steps: int = 500,
                   ema_decay: float = 0.999, lambda_mse: float = 0.1,
                   lambda_feat: float = 0.05, lambda_range: float = 1e-4,
                   feature_ops: list[str] | None = None,
                   val_pairs: Path | None = None, validation_count: int = 512,
                   eps_target: float = 0.0021, eval_every: int = 18_196,
                   checkpoint_every: int = 2_000, seed: int = 0,
                   resume_from: Path | None = None,
                   max_seconds: float | None = None) -> dict[str, Any]:
    """Fine-tune the ring-faithful student; checkpoint/resume friendly.

    Returns the run summary (also written to ``output_dir/history.json``); on the
    final job of the chain also exports ``output_dir`` as a manifest the oracle
    can gate.
    """
    import tensorflow as tf

    shards = _reject_verification_directory(Path(shards))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    total_steps = int(epochs * steps_per_epoch)

    # The manifest profile carries the public-kernel grid, so the student
    # trains on exactly the arithmetic the gate will certify.
    student = RingQATModel(manifest_path, fractional_bits=fractional_bits,
                           total_bits=total_bits, ring_bits=ring_bits,
                           dtype=tf.float32)
    print(f"profile ring{ring_bits}/f{fractional_bits} "
          f"f_w={student.plan.kernel_bits}, fused PReLU "
          f"{len(student.plan.prelu_fusion)}, fused SE "
          f"{len(student.plan.multiply_fusion)}", flush=True)
    if not student._exact_float32:
        raise ValueError("float32 forward is not exact for these bounds "
                         "(e_max + 2f > 24); refuse to train an infidel graph")
    variables = student.trainable_variables
    manifest = student.plan.manifest
    bounds = student.plan.bounds

    teacher = tf.keras.models.load_model(Path(teacher_checkpoint),
                                         compile=False)

    # Optional per-stage feature distillation.  Build a teacher sub-model that
    # emits the same op tensors the student captures; disable gracefully if the
    # names do not resolve (L_feat is an ablation, never a hard dependency).
    if feature_ops is None:
        feature_ops = _auto_feature_ops(manifest)
    feat_model = None
    if lambda_feat > 0 and feature_ops:
        try:
            outputs = [teacher.get_layer(name).output for name in feature_ops]
            feat_model = tf.keras.Model(teacher.inputs, outputs)
        except Exception as error:  # pragma: no cover - architecture dependent
            print(f"L_feat disabled: {error}", flush=True)
            feature_ops = []

    # Raw producers monitored by the range hinge, with their calibrated bound.
    raw_names = [op.name for op in manifest.ops
                 if op.name in student.plan.prelu_producers
                 or op.name in student.plan.fusion.raw]
    range_bounds = {f"{name}:raw": float(bounds[name])
                    for name in raw_names if bounds.get(name)}

    schedule = tf.keras.optimizers.schedules.CosineDecay(
        0.0, warmup_target=learning_rate, warmup_steps=warmup_steps,
        decay_steps=max(1, total_steps - warmup_steps))
    optimizer = tf.keras.optimizers.Adam(schedule)
    optimizer.build(variables)

    ema = {name: tf.Variable(var, trainable=False, name=f"ema/{name}")
           for name, var in student.variables.items()}
    step_counter = tf.Variable(0, dtype=tf.int64, trainable=False)

    # ``tf.train.Checkpoint`` tracks by keyword, so the tensor names (which carry
    # ``:`` and ``/``) must be sanitised into valid identifiers; the mapping is
    # applied identically on save and restore.
    def _key(name: str) -> str:
        return "v_" + name.replace(":", "_").replace("/", "_").replace(".", "_")

    ckpt = tf.train.Checkpoint(
        optimizer=optimizer, step=step_counter,
        model=tf.train.Checkpoint(
            **{_key(n): v for n, v in student.variables.items()}),
        ema=tf.train.Checkpoint(**{_key(n): v for n, v in ema.items()}))
    manager = tf.train.CheckpointManager(
        ckpt, directory=str(output_dir / "ckpt"), max_to_keep=2)
    if resume_from is not None and Path(resume_from).exists():
        ckpt.restore(str(resume_from)).expect_partial()
        print(f"resumed from {resume_from} at step {int(step_counter)}",
              flush=True)
    elif manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint).expect_partial()
        print(f"resumed from {manager.latest_checkpoint} at step "
              f"{int(step_counter)}", flush=True)

    def _l2(x):
        return tf.math.l2_normalize(tf.reshape(x, [tf.shape(x)[0], -1]),
                                    axis=-1)

    @tf.function(reduce_retracing=True)
    def train_step(batch):
        teacher_emb = tf.stop_gradient(
            tf.cast(teacher(batch, training=False), tf.float32))
        teacher_feats = ([tf.stop_gradient(tf.cast(t, tf.float32))
                          for t in feat_model(batch, training=False)]
                         if feat_model is not None else [])
        with tf.GradientTape() as tape:
            student_emb, captured = student(batch, capture=True)
            teacher_unit = tf.math.l2_normalize(teacher_emb, axis=-1)
            student_unit = tf.math.l2_normalize(student_emb, axis=-1)
            cosine = tf.reduce_mean(
                1.0 - tf.reduce_sum(teacher_unit * student_unit, axis=-1))
            mse = tf.reduce_mean(tf.square(teacher_emb - student_emb))
            loss = cosine + lambda_mse * mse
            if teacher_feats:
                feat = 0.0
                for name, target in zip(feature_ops, teacher_feats):
                    feat = feat + tf.reduce_mean(
                        tf.square(_l2(captured[name]) - _l2(target)))
                loss = loss + lambda_feat * feat
            if lambda_range > 0 and range_bounds:
                hinge = 0.0
                for name, bound in range_bounds.items():
                    hinge = hinge + tf.reduce_mean(
                        tf.nn.relu(tf.abs(captured[name]) - 0.9 * bound))
                loss = loss + lambda_range * hinge
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        for name, var in student.variables.items():
            ema[name].assign_sub((1.0 - ema_decay) * (ema[name] - var))
        step_counter.assign_add(1)
        return loss

    # Checkpoints are selected on the embedding perturbation
    #
    #     eps = mean(1 - cos(embedding_teacher_float, embedding_student))
    #
    # measured on held-out faces, *not* on proxy verification accuracy.  The
    # proxy is saturated: its median decision margin is eight times
    # eps, so it reports a 0.000 pp drop even where the official pose datasets
    # lose over a point, and it cannot resolve anything below ~0.5 pp.  eps is
    # dataset-independent and maps onto the gate: the calibration measured
    # drop = k * eps^alpha per dataset, with CPLFW binding at eps ~ 0.0021 for
    # 0.25 pp.
    eval_batch = 64
    validation_faces = None
    teacher_validation = None
    if val_pairs is not None and Path(val_pairs).exists():
        import pickle

        with open(val_pairs, "rb") as handle:
            encoded, _ = pickle.load(handle, encoding="bytes")
        validation_faces = encoded[:validation_count]
        teacher_validation = _embed(
            lambda x: teacher(x, training=False), validation_faces, eval_batch)
        print(f"validation faces {len(validation_faces)}", flush=True)

    def swap_in_ema():
        backup = {n: v.numpy() for n, v in student.variables.items()}
        for n, v in student.variables.items():
            v.assign(ema[n])
        return backup

    def swap_back(backup):
        for n, v in student.variables.items():
            v.assign(backup[n])

    # A resumed link starts with an empty history; reload what earlier links
    # wrote so the trajectory survives instead of being overwritten by the last
    # evaluation alone.
    history: list[dict[str, Any]] = []
    best = {"eps": float("inf"), "step": -1}
    summary_path = output_dir / "history.json"
    if summary_path.exists():
        try:
            earlier = json.loads(summary_path.read_text())
            history = list(earlier.get("history", []))
            previous = earlier.get("best") or {}
            if previous.get("step", -1) >= 0 and "eps" in previous:
                best = dict(previous)
        except (ValueError, OSError):
            pass

    def evaluate_and_maybe_save(step: int, loss_value: float) -> None:
        if validation_faces is None:
            return
        backup = swap_in_ema()
        student_embeddings = _embed(lambda x: student(x), validation_faces,
                                    eval_batch)
        swap_back(backup)
        eps = float(np.mean(1.0 - np.sum(teacher_validation
                                         * student_embeddings, axis=1)))
        record = {"step": step, "loss": float(loss_value), "eps": eps,
                  "eps_target": eps_target}
        history.append(record)
        print(f"[eval] step {step} loss {loss_value:.5f} eps {eps:.5f} "
              f"(target {eps_target:.5f})", flush=True)
        if eps < best["eps"]:
            best.update(eps=eps, step=step)
            backup = swap_in_ema()
            export_qat_manifest(student, manifest_path, output_dir / "best")
            swap_back(backup)

    stop = {"flag": False}

    def _handle_sigterm(*_a):
        stop["flag"] = True
    signal.signal(signal.SIGTERM, _handle_sigterm)

    started = time.time()
    dataset = _stream_dataset(shards, batch_size, seed)
    running = 0.0
    last_loss = 0.0
    for batch in dataset:
        step = int(step_counter)
        if step >= total_steps:
            break
        loss = train_step(batch)
        last_loss = float(loss)
        running += last_loss
        step = int(step_counter)
        if step % 100 == 0:
            print(f"step {step}/{total_steps} loss {running / 100:.5f}",
                  flush=True)
            running = 0.0
        if step % checkpoint_every == 0:
            manager.save(checkpoint_number=step)
        if step % eval_every == 0:
            evaluate_and_maybe_save(step, last_loss)
        if stop["flag"] or (max_seconds is not None
                            and time.time() - started > max_seconds):
            print(f"stopping early at step {step} "
                  f"({'sigterm' if stop['flag'] else 'max-seconds'})",
                  flush=True)
            manager.save(checkpoint_number=step)
            break

    finished = int(step_counter) >= total_steps
    manager.save(checkpoint_number=int(step_counter))
    if finished:
        evaluate_and_maybe_save(int(step_counter), last_loss)
        # Final export uses the EMA weights (evaluated alongside raw, §7.1).
        backup = swap_in_ema()
        export_qat_manifest(student, manifest_path, output_dir / "final")
        swap_back(backup)

    summary = {
        "phase": "Q3", "finished": finished,
        "step": int(step_counter), "total_steps": total_steps,
        "eps_target": eps_target,
        "best": best, "history": history,
        "feature_ops": feature_ops,
        "profile": {"fractional_bits": fractional_bits,
                    "total_bits": total_bits, "ring_bits": ring_bits},
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary

"""Face-verification accuracy on the five benchmark datasets.

The float teacher is evaluated with TensorFlow.  The quantized model is
evaluated with the ``ring-integer-v1`` oracle of :mod:`ghostmpc.eval.reference`,
which reproduces, integer for integer, the embedding the three servers compute;
the live MPC runs reveal embeddings bit-identical to it.  Both are scored by the
same ten-fold protocol.
"""

from __future__ import annotations

import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Any

DATASETS = ("lfw", "agedb_30", "calfw", "cplfw", "cfp_fp")


def load_dataset(directory: Path, name: str) -> tuple[list[bytes], list[bool]]:
    """The encoded images and the pair labels of one InsightFace ``.bin``."""
    with (directory / f"{name}.bin").open("rb") as handle:
        encoded, labels = pickle.load(handle, encoding="bytes")
    return list(encoded), [bool(label) for label in labels]


def _normalized(matrix: Any) -> Any:
    import numpy as np

    matrix = np.asarray(matrix, dtype=np.float64)
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


def _embeddings(model: Any, encoded: list[bytes], batch_size: int) -> Any:
    """L2-normalized embeddings of the float Keras model."""
    import numpy as np
    import tensorflow as tf

    output = []
    for start in range(0, len(encoded), batch_size):
        images = []
        for value in encoded[start:start + batch_size]:
            image = tf.io.decode_image(value, channels=3, expand_animations=False)
            image = tf.image.resize(image, (112, 112))
            images.append(tf.cast(image, tf.float32) / 127.5 - 1.0)
        output.append(np.asarray(model(tf.stack(images), training=False),
                                 dtype=np.float64))
    return _normalized(np.concatenate(output))


def _decode(values: list[bytes]) -> Any:
    """Aligned RGB faces scaled to ``[-1, 1]``, as the client shares them."""
    import numpy as np
    from PIL import Image

    rows = []
    for value in values:
        image = Image.open(BytesIO(value)).convert("RGB")
        if image.size != (112, 112):
            image = image.resize((112, 112), Image.BILINEAR)
        rows.append(np.asarray(image, dtype=np.float64) / 127.5 - 1.0)
    return np.stack(rows)


def _ring_batch(manifest: str, encoded: list[bytes]) -> Any:
    """Unnormalized oracle embeddings of one batch (a worker's unit of work)."""
    from ..core.model import Manifest
    from .reference import evaluate_numpy

    quantization = Manifest.load(Path(manifest)).quantization
    return evaluate_numpy(
        Path(manifest), _decode(encoded),
        fixed_point=(quantization.fractional_bits, quantization.activation_bits),
        ring_bits=quantization.ring_bits)


def ring_embeddings(manifest: Path, encoded: list[bytes], *,
                    batch_size: int = 16, workers: int = 1) -> Any:
    """L2-normalized embeddings of the quantized model, on the ring oracle.

    Batches are independent, so they are spread over ``workers`` processes;
    the result does not depend on the split.
    """
    import numpy as np

    batches = [encoded[start:start + batch_size]
               for start in range(0, len(encoded), batch_size)]
    path = str(Path(manifest).resolve())
    if workers <= 1:
        rows = [_ring_batch(path, batch) for batch in batches]
    else:
        # Spawned, not forked: the parent may already hold TensorFlow's thread
        # pools (the teacher), which a forked child would inherit locked.
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            rows = list(pool.map(_ring_batch, [path] * len(batches), batches))
    return _normalized(np.concatenate([np.asarray(row) for row in rows]))


def _verification_accuracy(embedding: Any, labels: Any, folds: int = 10) -> float:
    """Ten-fold accuracy: the threshold is chosen on nine folds, tested on one."""
    import numpy as np

    similarity = np.sum(embedding[0::2] * embedding[1::2], axis=1)
    labels = np.asarray(labels, dtype=bool)
    thresholds = np.linspace(-1.0, 1.0, 2001)
    scores = []
    indices = np.arange(len(labels))
    for fold in range(folds):
        test = indices % folds == fold
        train = ~test
        predictions = similarity[train, None] >= thresholds[None, :]
        accuracy = np.mean(predictions == labels[train, None], axis=0)
        threshold = thresholds[int(np.argmax(accuracy))]
        scores.append(float(np.mean((similarity[test] >= threshold) == labels[test])))
    return float(np.mean(scores))

"""MS1MV3 reader for quantization-aware training.

Streams the ms1m-retinaface corpus for offline QAT fine-tuning without ever
touching the verification datasets.  Two ingest paths are supported:

* WebDataset ``*.tar`` shards (``gaunernst/ms1mv3-wds`` on HuggingFace) — the
  primary path, iterated with the standard library ``tarfile`` only;
* the InsightFace IndexedRecordIO ``train.rec`` — a documented fallback, parsed
  in pure Python (magic ``0xced7230a``), never via mxnet.

Both yield :class:`Sample` records.  The 90/10 train/validation split is
deterministic **per identity** (``sha1(label) % 10 == 0`` -> validation) so no
identity leaks across the boundary.  From the validation split we build proxy
verification pairs serialized in the same pickle layout as the official
``.bin`` files, so ``bounds.load_images`` and
``face_accuracy._verification_accuracy`` read them unchanged.

Every public entry point rejects ``datasets/benchmarks`` via
``guard._reject_verification_directory`` (fail-closed).
"""

from __future__ import annotations

import hashlib
import pickle
import struct
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, NamedTuple

from .guard import _reject_verification_directory

# InsightFace / MXNet IndexedRecordIO constants.
_REC_MAGIC = 0xCED7230A
_IR_FORMAT = "<IfQQ"  # flag (uint32), label (float32), id (uint64), id2 (uint64)
_IR_SIZE = struct.calcsize(_IR_FORMAT)  # 24
_JPEG_MAGIC = b"\xff\xd8"

_IMAGE_EXTS = {"jpg", "jpeg", "png", "bmp", "webp"}
_LABEL_EXTS = {"cls", "txt", "label", "lbl"}


class Sample(NamedTuple):
    """One aligned face: identity ``label``, encoded ``image`` bytes, corpus ``index``."""

    label: int
    image: bytes
    index: int


def _validation_identity(label: int) -> bool:
    """True when identity ``label`` belongs to the validation split.

    Hashing the identity (not the image index) keeps every image of one
    identity on the same side of the split, avoiding identitarian leakage.
    """
    digest = hashlib.sha1(str(label).encode("utf-8")).hexdigest()
    return int(digest, 16) % 10 == 0


# --------------------------------------------------------------------------- #
# WebDataset (primary)
# --------------------------------------------------------------------------- #
def iter_webdataset(shard_directory: Path) -> Iterator[Sample]:
    """Yield samples from ``*.tar`` WebDataset shards, streaming, no extraction.

    A sample groups files sharing a key (the member name without extension):
    one image (``.jpg`` ...) and one label (``.cls`` ...).  Members are buffered
    per key and emitted once both parts are present.
    """
    import tarfile

    shard_directory = _reject_verification_directory(shard_directory)
    index = 0
    for shard in sorted(shard_directory.glob("*.tar")):
        with tarfile.open(shard, "r") as tar:
            pending: dict[str, dict[str, bytes]] = {}
            for member in tar:
                if not member.isfile():
                    continue
                key, _, ext = member.name.rpartition(".")
                ext = ext.lower()
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                slot = pending.setdefault(key, {})
                slot[ext] = handle.read()
                image = next((slot[e] for e in _IMAGE_EXTS if e in slot), None)
                label = next((slot[e] for e in _LABEL_EXTS if e in slot), None)
                if image is not None and label is not None:
                    del pending[key]
                    yield Sample(int(label.strip()), image, index)
                    index += 1
            if pending:
                raise ValueError(
                    f"{shard.name}: {len(pending)} incomplete sample(s), e.g. "
                    f"{next(iter(pending))!r}"
                )


# --------------------------------------------------------------------------- #
# IndexedRecordIO (fallback)
# --------------------------------------------------------------------------- #
def _read_record(handle: Any) -> bytes | None:
    """Read one MXNet record payload; ``None`` at end of stream."""
    head = handle.read(8)
    if len(head) < 8:
        return None
    magic, lrecord = struct.unpack("<II", head)
    if magic != _REC_MAGIC:
        raise ValueError(f"bad recordio magic {magic:#x}")
    length = lrecord & ((1 << 29) - 1)
    data = handle.read(length)
    pad = (((length + 3) >> 2) << 2) - length  # records align to 4 bytes
    handle.read(pad)
    return data


def _unpack(data: bytes) -> tuple[int, bytes]:
    """Split an InsightFace record into (identity, encoded image)."""
    flag, label, _id, _id2 = struct.unpack(_IR_FORMAT, data[:_IR_SIZE])
    rest = data[_IR_SIZE:]
    if flag > 0:  # header/list record: label is an array of ``flag`` floats
        labels = struct.unpack(f"<{flag}f", rest[: flag * 4])
        rest = rest[flag * 4 :]
        label = labels[0] if labels else 0.0
    return int(label), rest


def iter_recordio(rec_path: Path) -> Iterator[Sample]:
    """Yield image samples from an InsightFace ``train.rec`` (sequential).

    Non-image records (the leading identity-range header) carry no JPEG payload
    and are skipped.
    """
    rec_path = _reject_verification_directory(rec_path)
    index = 0
    with rec_path.open("rb") as handle:
        while True:
            data = _read_record(handle)
            if data is None:
                break
            label, image = _unpack(data)
            if not image[:2] == _JPEG_MAGIC:
                continue  # header/list record, no image
            yield Sample(label, image, index)
            index += 1


# --------------------------------------------------------------------------- #
# Split, statistics, proxy pairs
# --------------------------------------------------------------------------- #
def iter_split(samples: Iterable[Sample], split: str) -> Iterator[Sample]:
    """Filter ``samples`` to the ``"train"`` or ``"val"`` identity split."""
    if split not in {"train", "val"}:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    want_val = split == "val"
    for sample in samples:
        if _validation_identity(sample.label) == want_val:
            yield sample


def dataset_stats(samples: Iterable[Sample]) -> dict[str, int]:
    """One streaming pass: image and identity counts, per split."""
    train_ids: set[int] = set()
    val_ids: set[int] = set()
    train_images = val_images = 0
    for sample in samples:
        if _validation_identity(sample.label):
            val_ids.add(sample.label)
            val_images += 1
        else:
            train_ids.add(sample.label)
            train_images += 1
    return {
        "images": train_images + val_images,
        "identities": len(train_ids | val_ids),
        "train_images": train_images,
        "train_identities": len(train_ids),
        "val_images": val_images,
        "val_identities": len(val_ids),
    }


def build_val_pairs(samples: Iterable[Sample], output: Path, *,
                    identities: int = 2000, minimum_images: int = 4,
                    genuine: int = 6000, impostor: int = 6000,
                    seed: int = 0) -> dict[str, int]:
    """Build proxy verification pairs from the validation split.

    Writes ``output`` as ``pickle((encoded, labels))`` where ``encoded`` holds
    the two JPEG buffers of each pair interleaved and ``labels`` is one bool per
    pair (``True`` genuine) — the exact layout of the official ``.bin`` files.
    Selection is seeded and deterministic.
    """
    import random

    rng = random.Random(seed)
    by_identity: dict[int, list[bytes]] = {}
    for sample in iter_split(samples, "val"):
        by_identity.setdefault(sample.label, []).append(sample.image)

    eligible = {i: imgs for i, imgs in by_identity.items() if len(imgs) >= minimum_images}
    chosen = sorted(eligible)  # deterministic base order before seeded shuffle
    rng.shuffle(chosen)
    chosen = chosen[:identities]
    if len(chosen) < 2:
        raise ValueError(
            f"need >=2 eligible identities (>= {minimum_images} images), got {len(chosen)}"
        )

    encoded: list[bytes] = []
    labels: list[bool] = []
    for _ in range(genuine):
        identity = rng.choice(chosen)
        first, second = rng.sample(by_identity[identity], 2)
        encoded.extend((first, second))
        labels.append(True)
    for _ in range(impostor):
        left, right = rng.sample(chosen, 2)
        encoded.append(rng.choice(by_identity[left]))
        encoded.append(rng.choice(by_identity[right]))
        labels.append(False)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump((encoded, labels), handle)
    return {"identities": len(chosen), "genuine": genuine, "impostor": impostor,
            "pairs": genuine + impostor}


def to_tfrecords(samples: Iterable[Sample], output_directory: Path, *,
                 shard_size: int = 8192) -> list[Path]:
    """One-shot conversion to TFRecord shards (``label`` int64, ``index`` int64).

    Requires TensorFlow (lazy import); the streaming readers above do not.
    """
    import tensorflow as tf

    output_directory = _reject_verification_directory(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    shards: list[Path] = []
    writer = None
    for position, sample in enumerate(samples):
        if position % shard_size == 0:
            if writer is not None:
                writer.close()
            path = output_directory / f"ms1mv3-{len(shards):05d}.tfrecord"
            shards.append(path)
            writer = tf.io.TFRecordWriter(str(path))
        feature = {
            "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[sample.image])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[sample.label])),
            "index": tf.train.Feature(int64_list=tf.train.Int64List(value=[sample.index])),
        }
        writer.write(tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString())
    if writer is not None:
        writer.close()
    return shards

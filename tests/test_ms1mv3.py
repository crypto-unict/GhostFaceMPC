"""MS1MV3 reader: deterministic split, proxy pairs."""

import io
import pickle
import struct
from pathlib import Path

import pytest

from ghostmpc.training import ms1mv3
from ghostmpc.training.guard import verification_directory
from ghostmpc.training.ms1mv3 import Sample


def _jpeg(color: int) -> bytes:
    """Tiny real JPEG so ``load_images``/reader accept it."""
    Image = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (color % 256, 0, 0)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _pack_record(label: float, payload: bytes) -> bytes:
    """Write one MXNet IndexedRecordIO record matching ``_read_record``."""
    body = struct.pack(ms1mv3._IR_FORMAT, 0, label, 0, 0) + payload
    length = len(body)
    lrecord = length & 0x1FFFFFFF  # cflag 0
    pad = (((length + 3) >> 2) << 2) - length
    return struct.pack("<II", ms1mv3._REC_MAGIC, lrecord) + body + b"\x00" * pad


def test_recordio_roundtrip(tmp_path: Path):
    rec = tmp_path / "train.rec"
    records = [(7, _jpeg(10)), (7, _jpeg(20)), (42, _jpeg(30))]
    # A leading header/list record (flag>0, no JPEG) must be skipped.
    header = struct.pack(ms1mv3._IR_FORMAT, 2, 0.0, 0, 0) + struct.pack("<2f", 3.0, 0.0)
    length = len(header)
    lrecord = length & 0x1FFFFFFF
    pad = (((length + 3) >> 2) << 2) - length
    blob = struct.pack("<II", ms1mv3._REC_MAGIC, lrecord) + header + b"\x00" * pad
    blob += b"".join(_pack_record(float(label), payload) for label, payload in records)
    rec.write_bytes(blob)

    samples = list(ms1mv3.iter_recordio(rec))
    assert [s.label for s in samples] == [7, 7, 42]
    assert [s.index for s in samples] == [0, 1, 2]
    assert samples[0].image[:2] == b"\xff\xd8"


def test_webdataset_roundtrip(tmp_path: Path):
    import tarfile

    shard = tmp_path / "ms1mv3-0000.tar"
    with tarfile.open(shard, "w") as tar:
        for key, label, color in (("a", 7, 1), ("b", 7, 2), ("c", 42, 3)):
            for ext, payload in ((".jpg", _jpeg(color)), (".cls", str(label).encode())):
                info = tarfile.TarInfo(key + ext)
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))

    samples = list(ms1mv3.iter_webdataset(tmp_path))
    assert [s.label for s in samples] == [7, 7, 42]
    assert [s.index for s in samples] == [0, 1, 2]


def test_split_deterministic_and_by_identity():
    labels = list(range(200))
    a = {label: ms1mv3._validation_identity(label) for label in labels}
    b = {label: ms1mv3._validation_identity(label) for label in labels}
    assert a == b  # deterministic across runs
    # roughly 10% land in validation, and both splits are non-empty
    assert 0 < sum(a.values()) < len(labels)

    samples = [Sample(label, b"x", i) for i, label in enumerate(labels * 3)]
    train_ids = {s.label for s in ms1mv3.iter_split(samples, "train")}
    val_ids = {s.label for s in ms1mv3.iter_split(samples, "val")}
    assert train_ids.isdisjoint(val_ids)  # no identity leaks across the split


def test_dataset_stats():
    samples = [Sample(label, b"x", i) for i, label in enumerate([1, 1, 2, 3, 3, 3])]
    stats = ms1mv3.dataset_stats(samples)
    assert stats["images"] == 6
    assert stats["identities"] == 3
    assert stats["train_images"] + stats["val_images"] == 6
    assert stats["train_identities"] + stats["val_identities"] == 3


def test_build_val_pairs_official_format(tmp_path: Path):
    # Choose identities that all land in the validation split, 4 images each.
    val_labels = [label for label in range(1000) if ms1mv3._validation_identity(label)][:5]
    assert len(val_labels) >= 3
    samples = []
    index = 0
    for label in val_labels:
        for shade in range(4):
            samples.append(Sample(label, _jpeg(label * 4 + shade), index))
            index += 1

    output = tmp_path / "val_pairs.bin"
    summary = ms1mv3.build_val_pairs(samples, output, identities=5, genuine=10,
                                     impostor=10, seed=0)
    assert summary["pairs"] == 20

    with output.open("rb") as handle:
        encoded, pair_labels = pickle.load(handle, encoding="bytes")
    assert len(encoded) == 2 * len(pair_labels) == 40
    assert sum(pair_labels) == 10  # 10 genuine
    # Readable by the existing pipeline unchanged.
    from ghostmpc.eval.bounds import load_images
    faces = load_images(output, 4)
    assert len(faces) == 4
    assert faces[0].shape == (112, 112, 3)

    # Deterministic given the seed.
    again = tmp_path / "again.bin"
    ms1mv3.build_val_pairs(samples, again, identities=5, genuine=10, impostor=10, seed=0)
    assert again.read_bytes() == output.read_bytes()


def test_rejects_verification_directory():
    benchmarks = verification_directory()
    with pytest.raises(ValueError, match="verification datasets"):
        list(ms1mv3.iter_webdataset(benchmarks))
    with pytest.raises(ValueError, match="verification datasets"):
        list(ms1mv3.iter_recordio(benchmarks / "lfw.bin"))

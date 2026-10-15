from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

CHECKPOINT_URL = (
    "https://github.com/HamadYA/GhostFaceNets/releases/download/v1.2/"
    "GhostFaceNet_W1.3_S1_ArcFace.h5"
)
CHECKPOINT_NAME = "GhostFaceNet_W1.3_S1_ArcFace.h5"
TEACHER_CHECKPOINT_SHA256 = (
    "e4182ca2470dac3eb79974f4d5d54f2abdf66d7e231c6f3df4059a90bded1271"
)
MODEL_ID = "GhostFaceNetV1-1-W1.3-S1-ArcFace-MS1MV3"

# Deployed arithmetic profile.  Keep these values centralized: image
# sharing, export and CLI validation must move together or the parties decode a
# different fixed-point grid from the one used by the generated program.
DEPLOYMENT_PROFILE = "ring32-f6-fw9"
DEPLOYMENT_RING_BITS = 32
DEPLOYMENT_ACTIVATION_BITS = 15
DEPLOYMENT_FRACTIONAL_BITS = 6
DEPLOYMENT_WEIGHT_FRACTIONAL_BITS = 9


@dataclass(slots=True)
class TensorRef:
    name: str
    shape: list[int]
    offset: int
    length: int
    scale: float | None = None
    bits: int | None = None
    minimum: float | None = None
    maximum: float | None = None


@dataclass(slots=True)
class Op:
    name: str
    kind: str
    inputs: list[str]
    output_shape: list[int]
    params: dict[str, Any] = field(default_factory=dict)
    tensors: list[str] = field(default_factory=list)
    fused_from: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Quantization:
    """Arithmetic contract carried by manifest v2.

    ``profile=none`` denotes the accuracy baseline. Quantized exporters can
    populate the same fields without changing the graph or protocol APIs.
    """

    profile: str = "none"
    ring_bits: int = 64
    weight_bits: int = 32
    activation_bits: int = 31
    fractional_bits: int = 20
    power_of_two_scales: bool = False
    # Public kernels may sit on a finer grid than the activations: they are
    # ``cint``, so extra fractional bits cost no communication per element, only
    # the wider product scale ``f + f_w`` that the fusion budgets must absorb.
    # ``None`` keeps kernels on the activation grid (``f_w = f``).
    weight_fractional_bits: int | None = None


def deployment_quantization() -> Quantization:
    """Return a fresh copy of the deployed arithmetic contract."""
    return Quantization(
        profile=DEPLOYMENT_PROFILE,
        ring_bits=DEPLOYMENT_RING_BITS,
        activation_bits=DEPLOYMENT_ACTIVATION_BITS,
        fractional_bits=DEPLOYMENT_FRACTIONAL_BITS,
        weight_fractional_bits=DEPLOYMENT_WEIGHT_FRACTIONAL_BITS,
    )


@dataclass(slots=True)
class Manifest:
    version: int
    model_id: str
    checkpoint_url: str
    checkpoint_sha256: str
    input_shape: list[int]
    output_shape: list[int]
    ops: list[Op]
    tensors: list[TensorRef]
    weights_file: str
    weights_sha256: str
    quantization: Quantization = field(default_factory=Quantization)
    # Magnitude bound of a layer output, used to size fused truncations.
    overflow_bounds: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: Path) -> Manifest:
        data = json.loads(path.read_text())
        data["ops"] = [Op(**item) for item in data["ops"]]
        data["tensors"] = [TensorRef(**item) for item in data["tensors"]]
        data["quantization"] = Quantization(**data.get("quantization", {}))
        data.setdefault("overflow_bounds", {})
        # Optional per-channel extensions this release does not use; an empty
        # entry is the plain scalar profile and is accepted.
        for key in ("channel_gains", "channel_overflow_bounds"):
            if data.pop(key, None):
                raise ValueError(f"{path}: per-channel '{key}' is not supported")
        return cls(**data)


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

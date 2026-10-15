from __future__ import annotations

import secrets
from collections.abc import Iterable

from .model import DEPLOYMENT_RING_BITS

RING_BITS = DEPLOYMENT_RING_BITS
RING = 1 << RING_BITS
HALF_RING = 1 << (RING_BITS - 1)


def encode_fixed(values: Iterable[float], fractional_bits: int,
                 ring_bits: int = RING_BITS) -> list[int]:
    scale = 1 << fractional_bits
    ring = 1 << ring_bits
    half = 1 << (ring_bits - 1)
    encoded: list[int] = []
    for value in values:
        raw = int(round(float(value) * scale))
        if not -half <= raw < half:
            raise OverflowError(
                f"fixed-point value {value!r} is outside the {ring_bits}-bit ring")
        encoded.append(raw % ring)
    return encoded


def decode_fixed(values: Iterable[int], fractional_bits: int,
                 ring_bits: int = RING_BITS) -> list[float]:
    scale = 1 << fractional_bits
    ring = 1 << ring_bits
    half = 1 << (ring_bits - 1)
    return [((v - ring) if v >= half else v) / scale for v in values]


def share_ring_n(values: Iterable[int], parties: int, ring_bits: int = RING_BITS) -> list[list[int]]:
    """Split values into fresh additive components ``x = x_0 + x_1 + x_2``.

    The client then hands party ``i`` the pair ``(x_i, x_{i-1})``
    (:func:`replicate_components`), which *is* the replicated sharing the
    servers compute on: they load it as it is and never re-share it.
    """
    if parties != 3:
        raise ValueError("Rep3 requires exactly three parties")
    modulus = 1 << ring_bits
    result = [[] for _ in range(parties)]
    for value in values:
        random = [secrets.randbits(ring_bits) for _ in range(parties - 1)]
        for index, component in enumerate(random):
            result[index].append(component)
        result[-1].append((int(value) - sum(random)) % modulus)
    return result


def share_fixed_n(values: Iterable[float], fractional_bits: int, parties: int,
                  ring_bits: int = RING_BITS) -> list[list[int]]:
    return share_ring_n(encode_fixed(values, fractional_bits, ring_bits),
                        parties, ring_bits)


def reconstruct_ring_n(shares: Iterable[Iterable[int]], ring_bits: int = RING_BITS) -> list[int]:
    parts = [list(part) for part in shares]
    if len(parts) != 3:
        raise ValueError("Rep3 reconstruction requires exactly three components")
    size = len(parts[0])
    if any(len(part) != size for part in parts):
        raise ValueError("share lengths differ")
    modulus = 1 << ring_bits
    return [sum(part[i] for part in parts) % modulus for i in range(size)]


# The client hands each component to *both* of its holders, so what the servers
# receive is already a replicated sharing and no server re-shares anything.
# The layout is MP-SPDZ's own: party ``i`` holds ``(x_i, x_{i-1})`` as
# ``(share[0], share[1])``, so component ``j`` is held by ``P_j`` and
# ``P_{j+1}``.  With an honest client the two honest servers then hold all
# three components between them, so no single server can alter the input.


class ReplicaMismatch(ValueError):
    """Two holders of the same component returned different copies."""


def replicate_components(shares: list[list[int]]) -> list[list[list[int]]]:
    """The pair ``(x_i, x_{i-1})`` each of three parties receives."""
    if len(shares) != 3:
        raise ValueError("replicated delivery is the three-party layout")
    return [[list(shares[party]), list(shares[(party - 1) % 3])]
            for party in range(3)]


def reconstruct_replicated(pairs: Iterable[Iterable[Iterable[int]]],
                           ring_bits: int = RING_BITS) -> list[int]:
    """Check every component against its second holder, then reconstruct.

    ``pairs[i]`` is party ``i``'s ``(share[0], share[1])``.  Component ``j``
    arrives as ``pairs[j][0]`` and again as ``pairs[j+1][1]``; one server that
    alters what it returns contradicts the copy of the honest co-holder, and
    the client rejects the answer instead of reconstructing a flipped one.
    """
    parts = [[list(copy) for copy in pair] for pair in pairs]
    if len(parts) != 3 or any(len(pair) != 2 for pair in parts):
        raise ValueError("expected one (share[0], share[1]) pair per party")
    size = len(parts[0][0])
    if any(len(copy) != size for pair in parts for copy in pair):
        raise ValueError("share lengths differ")
    modulus = 1 << ring_bits
    for component in range(3):
        first = parts[component][0]
        second = parts[(component + 1) % 3][1]
        for index, (a, b) in enumerate(zip(first, second)):
            if (a - b) % modulus:
                raise ReplicaMismatch(
                    f"component {component} of value {index} differs between "
                    f"P{component} and P{(component + 1) % 3}")
    return reconstruct_ring_n([parts[party][0] for party in range(3)],
                              ring_bits)


def persistence_header(ring_bits: int = RING_BITS, *, clear: bool = False) -> bytes:
    """The signature MP-SPDZ expects at the head of a persistence file.

    ``file_signature<T>()`` is the share type's ``type_string`` (raw bytes)
    followed by its ``specification`` (the ring width as a little-endian
    32-bit integer), written as an ``octetStream``: a little-endian 64-bit
    length, then the bytes.  MP-SPDZ replaces a file whose signature does not
    match with an empty one, so these bytes are pinned by a test.  Both
    ring protocols share the type string -- ``PostSacriRepRingShare`` inherits
    the replicated one -- so one header serves either profile.  ``clear``
    gives the header of ``Transactions-clear-P<i>.data``, whose elements are
    single ring integers.
    """
    import struct

    type_string = f"Z2^{ring_bits}" if clear else f"replicated Z2^{ring_bits}"
    payload = type_string.encode() + struct.pack("<i", ring_bits)
    return struct.pack("<Q", len(payload)) + payload


def _ring_bytes(value: int, ring_bits: int) -> bytes:
    width = (ring_bits + 7) // 8
    return (int(value) % (1 << ring_bits)).to_bytes(width, "little")


def write_persistence(path, pairs: Iterable[Iterable[int]],
                      ring_bits: int = RING_BITS, *, append: bool = False) -> None:
    """Write a party's replicated shares as ``Transactions-P<i>.data``.

    ``pairs`` is ``[share[0] values, share[1] values]``, the layout
    :func:`read_persistence` returns.  A new file starts with
    :func:`persistence_header`; ``append`` adds shares after the existing ones,
    which is how per-request probes follow the enrolled database.
    """
    from pathlib import Path

    first, second = (list(part) for part in pairs)
    if len(first) != len(second):
        raise ValueError("share[0] and share[1] differ in length")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not append or not path.exists()
    with path.open("wb" if fresh else "ab") as handle:
        if fresh:
            handle.write(persistence_header(ring_bits))
        handle.write(b"".join(_ring_bytes(a, ring_bits) + _ring_bytes(b, ring_bits)
                              for a, b in zip(first, second)))


def write_clear_persistence(path, values: Iterable[int],
                            ring_bits: int = RING_BITS, *,
                            append: bool = False) -> None:
    """Write public ring integers as ``Transactions-clear-P<i>.data``."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not append or not path.exists()
    with path.open("wb" if fresh else "ab") as handle:
        if fresh:
            handle.write(persistence_header(ring_bits, clear=True))
        handle.write(b"".join(_ring_bytes(value, ring_bits) for value in values))


def read_persistence(path: str, ring_bits: int = RING_BITS, *,
                     last: int | None = None) -> list[list[int]]:
    """Read a replicated share file ``Persistence/Transactions-P<i>.data``.

    MP-SPDZ writes an ``octetStream`` signature (an eight-byte little-endian
    length, then the share type and its specification) and then every share
    as its two components, each in the minimal number of little-endian bytes.
    Returns ``[share[0] values, share[1] values]``.  The servers append what
    they deliver after the client's input shares in the same file, so
    ``last`` keeps only the final ``last`` shares.
    """
    import struct

    with open(path, "rb") as handle:
        raw = handle.read()
    if len(raw) < 8:
        raise ValueError(f"{path}: no persistence header")
    (header,) = struct.unpack_from("<Q", raw)
    start = 8 + header
    width = (ring_bits + 7) // 8
    payload = raw[start:]
    if start > len(raw) or len(payload) % (2 * width):
        raise ValueError(f"{path}: not a sequence of {ring_bits}-bit pairs")
    if last is not None:
        if not 0 < last <= len(payload) // (2 * width):
            raise ValueError(f"{path}: fewer than {last} shares")
        payload = payload[len(payload) - last * 2 * width:]
    modulus = 1 << ring_bits
    first: list[int] = []
    second: list[int] = []
    for offset in range(0, len(payload), 2 * width):
        first.append(int.from_bytes(payload[offset:offset + width], "little")
                     % modulus)
        second.append(int.from_bytes(
            payload[offset + width:offset + 2 * width], "little") % modulus)
    return [first, second]

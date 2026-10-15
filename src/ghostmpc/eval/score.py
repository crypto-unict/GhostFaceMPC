"""Clear-text oracle of the 1:1 verification score.

The emitted program computes two inner products over the ring and opens
nothing.  This module reproduces the same integers in NumPy, so a ``test_reveal``
build can be checked against them rather than against a floating-point
approximation of what it was supposed to do, and so the client's half of the
protocol -- the one division -- exists in exactly one place.

The widths come from :class:`ghostmpc.core.score.ScorePlan`; nothing here
re-derives a grid.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..core.ring import to_signed
from ..core.score import ScoreError, ScorePlan


def encode_gallery(embeddings: np.ndarray, plan: ScorePlan) -> np.ndarray:
    """L2-normalize the database rows and put them on the gallery grid.

    This is the dealer's work, done once and offline.  Normalizing here is what
    removes every square root from the protocol: with ``||g|| = 1`` the cosine
    is ``<e, g> / ||e||`` and no norm of a row is ever computed under MPC.
    """
    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != plan.embedding_size:
        raise ScoreError(
            f"the gallery must be (rows, {plan.embedding_size}), got "
            f"{values.shape}")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.all(norms > 0.0):
        raise ScoreError("a zero row has no direction and cannot be enrolled")
    unit = values / norms
    encoded = np.rint(unit * (1 << plan.gallery_fractional_bits)).astype(np.int64)
    # A unit component never reaches 1 after rounding at f_g >= 1, so this only
    # fires if the plan was built with a grid that does not fit its own ring.
    limit = 1 << (plan.ring_bits - 1)
    if np.any(np.abs(encoded) >= limit):
        raise ScoreError("the encoded gallery does not fit the ring")
    return encoded


def encode_probe(embedding: Sequence[float] | np.ndarray,
                 plan: ScorePlan) -> np.ndarray:
    """Put a clear embedding on the activation grid the MPC probe lives on."""
    values = np.asarray(embedding, dtype=np.float64).reshape(-1)
    if values.size != plan.embedding_size:
        raise ScoreError(
            f"the probe has {values.size} components and the plan expects "
            f"{plan.embedding_size}")
    return np.rint(values * (1 << plan.fractional_bits)).astype(np.int64)


def score_ring(probe: np.ndarray, gallery: np.ndarray, row: int,
               plan: ScorePlan) -> tuple[int, int]:
    """The two ring integers the servers hold at the end of a request.

    Both accumulations are taken modulo ``2^ring_bits`` and then read back
    signed, which is exactly what the protocol does: partial sums may wrap, and
    the result is correct because :meth:`ScorePlan.validate` proved the true
    value fits.
    """
    if not 0 <= row < gallery.shape[0]:
        raise ScoreError(f"row {row} is outside a gallery of {gallery.shape[0]}")
    modulus = 1 << plan.ring_bits
    probe = np.asarray(probe, dtype=object)
    similarity = int(np.dot(probe, np.asarray(gallery[row], dtype=object)))
    norm = int(np.dot(probe, probe))
    return (int(to_signed(similarity % modulus, plan.ring_bits)),
            int(to_signed(norm % modulus, plan.ring_bits)))


def cosine(similarity: int, norm: int, plan: ScorePlan) -> float:
    """The client's half: one division, in the clear, off the critical path."""
    return plan.cosine(similarity, norm)


def score(probe: Sequence[float] | np.ndarray, gallery: np.ndarray, row: int,
          plan: ScorePlan) -> float:
    """End-to-end clear-text score, from a float embedding to the cosine."""
    similarity, norm = score_ring(encode_probe(probe, plan), gallery, row, plan)
    return cosine(similarity, norm, plan)

"""Clear-text oracle of the closed-set 1:N identification query.

The emitted program computes ``N`` inner products over the ring and runs a
comparison tree over them, opening nothing.  This module reproduces the same
integers in NumPy, so a ``test_reveal`` build can be checked against them rather
than against a floating-point approximation of what it was supposed to do.

Two rules are fixed here and mirrored by the emitter:

* the tournament **folds the vector in half**: element ``i`` meets element
  ``i + W``, and an odd level carries its last element forward untouched -- the
  shape :meth:`~ghostmpc.core.identify.IdentifyPlan.level_widths` states.  The
  fold is not cosmetic: it keeps both operands contiguous, which is what the
  emitter needs, and a strided pairing of adjacent elements was measured to read
  recycled memory on a live run;
* the selection bit is ``is_negative(left - right)``, so a **tie keeps the left
  operand of the fold** -- the earlier position in the current vector, which at
  the first level is the lower row index.  A zero difference is not negative,
  and defining the rule on the emitted comparison makes the equivalence exact by
  construction rather than exact to a tolerance.

The widths come from :class:`~ghostmpc.core.identify.IdentifyPlan`; nothing here
re-derives a grid.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..core.identify import IdentifyError, IdentifyPlan
from ..core.ring import to_signed
from .score import encode_probe


def similarities_ring(probe: np.ndarray, gallery: np.ndarray,
                      plan: IdentifyPlan) -> list[int]:
    """The ``N`` signed ring integers the servers hold before the tournament.

    Each accumulation is taken modulo ``2^ring_bits`` and read back signed, for
    the reason :func:`ghostmpc.eval.score.score_ring` states: partial sums may
    wrap and the declared widths prove the result does not.
    """
    rows = np.asarray(gallery)
    if rows.ndim != 2 or rows.shape[1] != plan.score.embedding_size:
        raise IdentifyError(
            f"the gallery must be (rows, {plan.score.embedding_size}), got "
            f"{rows.shape}")
    if rows.shape[0] < plan.candidates:
        raise IdentifyError(
            f"the query ranks {plan.candidates} rows and the gallery holds "
            f"{rows.shape[0]}")
    modulus = 1 << plan.score.ring_bits
    values = np.asarray(probe, dtype=object)
    return [int(to_signed(int(np.dot(values,
                                     np.asarray(rows[row], dtype=object)))
                          % modulus, plan.score.ring_bits))
            for row in range(plan.candidates)]


def tournament(similarities: Sequence[int],
               plan: IdentifyPlan) -> tuple[int, int]:
    """The comparison tree, exactly as the emitter unrolls it.

    Returns ``(index, similarity)`` of the winner: the rank-1 row and the
    accumulator that won, both of which the servers hold as shares.
    """
    values = [int(value) for value in similarities]
    if len(values) != plan.candidates:
        raise IdentifyError(
            f"the tournament expects {plan.candidates} similarities, got "
            f"{len(values)}")
    indices = list(range(plan.candidates))
    for width in plan.level_widths():
        carry = len(values) % 2
        next_values: list[int] = []
        next_indices: list[int] = []
        for node in range(width):
            left, right = node, node + width
            # `is_negative(values[left] - values[right])`: a tie is not
            # negative, so the left operand of the fold wins it.
            takes_right = values[left] - values[right] < 0
            next_values.append(values[right] if takes_right else values[left])
            next_indices.append(indices[right] if takes_right else indices[left])
        if carry:
            next_values.append(values[-1])
            next_indices.append(indices[-1])
        values, indices = next_values, next_indices
    return int(indices[0]), int(values[0])


def identify_ring(probe: np.ndarray, gallery: np.ndarray,
                  plan: IdentifyPlan) -> tuple[int, int]:
    """The rank-1 answer a request leaves the servers holding, as integers."""
    return tournament(similarities_ring(probe, gallery, plan), plan)


def identify(probe: Sequence[float] | np.ndarray, gallery: np.ndarray,
             plan: IdentifyPlan) -> tuple[int, int]:
    """End-to-end clear-text identification, from a float probe embedding."""
    return identify_ring(encode_probe(probe, plan.score), gallery, plan)

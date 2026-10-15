"""Clear-text oracle of the match decision.

The emitted program reduces the two score accumulators, squares one of them and
compares; this module reproduces that decision *as integers*, so a
``test_reveal`` build can be checked against it rather than against a
floating-point approximation of what it was supposed to do.

Three rules are fixed here and mirrored by every emitter:

* the reduction is a **nearest** truncation, because that is what the emitted
  kernel performs -- it delegates to :func:`ghostmpc.core.ring.trunc_nearest`
  rather than to a hand-rolled shift, which would be a second model of the same
  primitive, free to drift from it;
* the comparison is ``>=`` on the squared form, i.e. a tie counts as a match,
  which is the same convention ``eval.face_accuracy`` uses on the float cosine;
* the sign that selects the branch is the sign of the **reduced** similarity,
  not of the raw one, because the emitter takes it from the truncation kernel.
  ``trunc_round_nearest_with_sign`` returns the sign of its own result, so the
  two differ only where the reduced value is zero and the raw one negative --
  ``|cos| < 3e-5`` at the deployed profile, inside the rule's own resolution --
  and defining the rule on the reduced value makes the equivalence exact by
  construction rather than exact to a tolerance.

The widths come from :class:`ghostmpc.core.match.MatchPlan`; nothing here
re-derives a grid.
"""

from __future__ import annotations

from ..core.match import MatchError, MatchPlan
from ..core.ring import to_signed, trunc_nearest


def reduce_accumulator(value: int, length: int, shift: int,
                       plan: MatchPlan) -> int:
    """One accumulator, reduced exactly as the emitter reduces it."""
    if shift == 0:
        return int(value)
    return int(trunc_nearest(int(value), length, shift,
                             ring_bits=plan.score.ring_bits))


def match_ring(similarity: int, norm: int, plan: MatchPlan) -> int:
    """The match bit the servers hold at the end of a request.

    ``similarity`` and ``norm`` are the two signed ring integers
    :func:`ghostmpc.eval.score.score_ring` produces.
    """
    if norm < 0:
        raise MatchError("a negative inner product of a vector with itself "
                         "means the norm accumulator wrapped the ring")
    reduced_similarity = reduce_accumulator(
        similarity, plan.reduction_length, plan.reduction, plan)
    reduced_norm = reduce_accumulator(
        norm, plan.reduction_length, plan.norm_reduction, plan)
    modulus = 1 << plan.score.ring_bits
    difference = to_signed(
        (reduced_similarity * reduced_similarity
         - plan.threshold_constant * reduced_norm) % modulus,
        plan.score.ring_bits)
    # The emitted kernel hands back the sign of the value it produced, so the
    # rule reads it there too.  Without a reduction the emitter compares the
    # raw accumulator, and the oracle follows.
    negative = (reduced_similarity < 0 if plan.reduction > 0
                else similarity < 0)
    below = difference < 0
    if plan.threshold >= 0.0:
        return int((not negative) and (not below))
    # A negative threshold accepts every non-negative similarity outright, and
    # a negative one only while the squared magnitude stays under the bound.
    return int((not negative) or below)


def match_float(similarity: int, norm: int, plan: MatchPlan) -> int:
    """The decision the same threshold takes on the exact cosine.

    This is *not* what the program computes -- it is the reference the reduced
    integer rule is measured against, so the cost of the reduction can be
    stated as a disagreement count rather than asserted to be zero.
    """
    return int(plan.score.cosine(similarity, norm) >= plan.threshold)


def decision_margin(similarity: int, norm: int, plan: MatchPlan) -> float:
    """``cos - T``, in the clear: the quantity the match bit is the sign of."""
    return plan.score.cosine(similarity, norm) - plan.threshold

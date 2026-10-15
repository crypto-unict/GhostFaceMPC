"""The closed-set 1:N identification query, and the widths that make it fit.

The 1:1 score and the match decision both answer a *declared* identity: the
client names the row, the index is public, and one row is read whatever the
database holds.  A probe
that declares nothing asks a different question -- *who is this?* -- and the
servers have to rank it against every enrolled row and return the best one.

Two properties collapse that to a comparison tree over raw accumulators.

**The divisor is common to every candidate.**  The dealer stores
``\\|g_i\\| = 1``, so ``cos(e, g_i) = \\langle e, g_i \\rangle / \\|e\\|``, and
``\\|e\\| > 0`` does not depend on ``i``.  Therefore

.. math::

    \\arg\\max_i \\cos(e, g_i) = \\arg\\max_i \\langle e, g_i \\rangle
                              = \\arg\\max_i s_i,

which is why a closed-set identification computes **no norm at all**: ``n =
\\langle e, e \\rangle``, the second accumulator of a 1:1 score, is never needed.
No square root and no division appear either, for the same reason they do not
appear in the 1:1 score.

**Nothing is truncated.**  The match decision had to reduce its accumulators
because it *squared* one of them; a ranking only subtracts.  The comparisons
therefore run directly on the raw ``2^{f + f_g}`` grid, and the rounding kernel
-- 93 % of the cost of an inference -- never enters the identification path.
One comparison reads the difference of two similarity accumulators, so it
declares one bit more than :attr:`~ghostmpc.core.score.ScorePlan.similarity_bits`,
and :meth:`IdentifyPlan.validate` checks that against the ring rather than
assuming it.

What the servers return is the **rank-1 index alone**, as share components:
strictly less than the ``N`` similarities, which over repeated queries would
profile the enrolled database the client never sees.

This module is the single source of truth for those widths, as
:mod:`ghostmpc.core.score` is for the score's and :mod:`ghostmpc.core.match` for
the decision's: the emitter, the CLI and the clear-text oracle all build the
same :class:`IdentifyPlan`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .score import ScorePlan


class IdentifyError(ValueError):
    """An identification configuration that would wrap the computation ring."""


@dataclass(frozen=True)
class IdentifyPlan:
    """Widths and shape of one closed-set 1:N identification query.

    ``candidates`` is the number of enrolled rows ranked per request.  Unlike
    the score's ``gallery_rows`` it *does* enter the request path: this is the
    first operation of the deployment whose cost depends on the size of the
    database, linearly in bytes and logarithmically in rounds.
    """

    score: ScorePlan
    candidates: int
    vectorized_tournament: bool = True

    @property
    def similarity_bits(self) -> int:
        """Declared width of one ``s_i``, which the score plan already proves."""
        return self.score.similarity_bits

    @property
    def difference_bits(self) -> int:
        """Declared width of ``s_j - s_i``, the value one comparison reads.

        Both operands fit signed in :attr:`similarity_bits`, so their difference
        needs one magnitude bit more and keeps its sign bit.
        """
        return self.similarity_bits + 1

    @property
    def levels(self) -> int:
        """Rounds of the tournament: ``ceil(log2 N)`` sequential stages."""
        return max(0, math.ceil(math.log2(self.candidates))
                   if self.candidates > 1 else 0)

    @property
    def comparisons(self) -> int:
        """Comparisons the tournament performs, ``N - 1`` whatever its shape."""
        return max(0, self.candidates - 1)

    @property
    def selections(self) -> int:
        """Secret products: one for the value and one for the index per node."""
        return 2 * self.comparisons

    @property
    def index_bits(self) -> int:
        """Bits the returned index occupies; it is a small public-range value."""
        return max(1, math.ceil(math.log2(max(1, self.candidates))))

    def level_widths(self) -> list[int]:
        """Elements compared at each level, in order.

        An odd level carries its last element forward untouched rather than
        padding the database with a sentinel: a padded row would have to be a
        similarity no real row can reach, which is a bound this plan would then
        have to certify.  Carrying forward certifies nothing and costs nothing.
        """
        widths: list[int] = []
        remaining = self.candidates
        while remaining > 1:
            widths.append(remaining // 2)
            remaining = remaining - remaining // 2
        return widths

    def declared_widths(self) -> dict[str, int]:
        return {"similarity": self.similarity_bits,
                "difference": self.difference_bits}

    def validate(self) -> IdentifyPlan:
        """Refuse a configuration whose comparison could wrap the ring."""
        if self.candidates < 1:
            raise IdentifyError("an identification ranks at least one row")
        if self.candidates > self.score.gallery_rows:
            raise IdentifyError(
                f"the query ranks {self.candidates} rows and the database holds "
                f"{self.score.gallery_rows}")
        self.score.validate()
        headroom = self.score.ring_bits - 1
        if self.difference_bits > headroom:
            raise IdentifyError(
                f"the comparison reads a {self.difference_bits}-bit difference "
                f"and the ring leaves {headroom}: lower the gallery grid "
                f"(f_g = {self.score.gallery_fractional_bits}) or the certified "
                f"norm bound (||e|| <= {self.score.norm_bound})")
        if sum(self.level_widths()) != self.comparisons:
            raise IdentifyError("the tournament does not perform N - 1 comparisons")
        return self


def deployment_identify_plan(score: ScorePlan, candidates: int | None = None, *,
                             vectorized_tournament: bool = True) -> IdentifyPlan:
    """The certified plan, validated before anything is emitted against it.

    ``candidates`` defaults to the whole database, which is what *closed-set*
    means: every enrolled identity is a candidate for every probe.
    """
    return IdentifyPlan(
        score=score,
        candidates=int(score.gallery_rows if candidates is None else candidates),
        vectorized_tournament=bool(vectorized_tournament)).validate()


def identify_plan_report(plan: IdentifyPlan) -> dict[str, object]:
    """What a report states about a compiled identification, in one place."""
    return {
        "candidates": plan.candidates,
        "levels": plan.levels,
        "comparisons": plan.comparisons,
        "selections": plan.selections,
        "level_widths": plan.level_widths(),
        "vectorized_tournament": plan.vectorized_tournament,
        "declared_widths": plan.declared_widths(),
        "truncations": 0,
        "opened": ("nothing; the servers return share components of the rank-1 "
                   "index and the client reconstructs it"),
        "decision": ("argmax_i <e, g_i>, which is argmax_i cos(e, g_i) because "
                     "the dealer L2-normalizes every row and ||e|| does not "
                     "depend on i: no norm, no division and no truncation"),
    }

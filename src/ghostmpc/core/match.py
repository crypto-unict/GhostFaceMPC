"""The match decision, and the algebra that removes its square root.

:mod:`ghostmpc.core.score` stops one step short of what a verification system
answers: it leaves the servers holding ``s = <e, g_id>`` and ``n = <e, e>`` and
lets the client divide.  This module adds the last step -- the comparison of the
cosine against a public threshold ``T`` -- so that a request returns

.. math::

    \\mathrm{Match}(y_1, y_2) = \\mathrm{Score}(y_1, y_2) - T

as one secret-shared bit, whose components each server hands back exactly as it
hands back the two score accumulators today.  Nothing is opened, and the client
learns strictly less than it does with the score alone: one bit instead of two
ring elements.

**No square root, and no division.**  With the accumulator scales the row uses --
``s`` at ``2^a`` and ``n`` at ``2^b`` -- the cosine is

.. math::

    \\cos(e, g) = \\frac{s / 2^{a}}{\\sqrt{n / 2^{b}}},

so for a non-negative threshold

.. math::

    \\cos \\ge T \\iff s \\ge 0 \\;\\wedge\\; s^2 \\ge T^2 2^{2a-b} n ,

and for a negative one ``cos >= T`` iff ``s >= 0`` or ``s^2 <= T^2 2^{2a-b} n``.
Squaring both sides removes the square root; ``C = T^2 2^{2a-b}`` is public, so
the right-hand side is a local secret-by-public product.  Finishing the cosine
under MPC instead would need a division, which does not compile at the deployed
ring at all (MP-SPDZ asks for ``-R 39``).

**Fitting the ring.**  A square needs twice the bits of its operand, which is
the only new constraint.  At the deployed ring32 profile ``|s| <= 2^27``, so the
two accumulators are reduced first, by ``m_s`` and ``m_n`` bits, and the
constant absorbs the difference of the two shifts:

.. math::

    (s \\gg m_s)^2 \\ge \\mathrm{round}(T^2 2^{2a-b-c}) \\cdot (n \\gg m_n),
    \\qquad c = 2 m_s - m_n .

``m_s`` is forced -- :attr:`MatchPlan.minimum_reduction` is the smallest value
whose square fits, 12 at the deployed profile.  Both reductions are issued as
**one** truncation over the two-element vector ``[s, n]``, so ``m_n = m_s`` and
the threshold constant pays the remaining ``c = m_s`` bits; the sign the branch
needs is the one that truncation already returns.  Both sides of the comparison
are non-negative, so the difference needs one bit for its sign and no extra
magnitude bit.

This module is the single source of truth for those widths, exactly as
:mod:`ghostmpc.core.score` is for the score's: the emitter and the clear-text
oracle build the same :class:`MatchPlan`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .score import ScorePlan

# A threshold constant with too few fractional bits quantizes ``T`` itself
# rather than the accumulators: at eight bits the coarsest step on ``T^2`` is
# 1/256, i.e. about 0.008 on ``T`` near the operating point, which is already
# eight times the 1e-3 grid ``eval.face_accuracy`` calibrates on.  Refuse below
# that rather than emit a threshold nobody declared.
MINIMUM_THRESHOLD_RESOLUTION_BITS = 8


class MatchError(ValueError):
    """A match configuration that would wrap the ring or quantize ``T`` away."""


@dataclass(frozen=True)
class MatchPlan:
    """Widths, threshold constant and reduction of one match decision."""

    score: ScorePlan
    threshold: float

    @property
    def similarity_scale_bits(self) -> int:
        """``a``: the power of two ``s`` is measured in (``f + f_g``)."""
        return self.score.fractional_bits + self.score.gallery_fractional_bits

    @property
    def norm_scale_bits(self) -> int:
        """``b``: the power of two ``n`` is measured in (``2f``)."""
        return 2 * self.score.fractional_bits

    @property
    def similarity_accumulator_bits(self) -> int:
        """Declared width of ``s``."""
        return self.similarity_scale_bits + self.score.norm_bits + 1

    @property
    def norm_accumulator_bits(self) -> int:
        """Declared width of ``n``."""
        return self.norm_scale_bits + 2 * self.score.norm_bits + 1

    @property
    def constant_scale_bits(self) -> int:
        """``2a - b``: the scale ``T^2`` is encoded on before any reduction."""
        return 2 * self.similarity_scale_bits - self.norm_scale_bits

    @property
    def minimum_reduction(self) -> int:
        """Smallest ``m_s`` whose squared comparison still fits the ring.

        ``|s| <= 2^(a + log2||e||)``, so ``(s >> m_s)^2`` occupies
        ``2(a + log2||e|| - m_s)`` magnitude bits.  **Both sides of the
        comparison are non-negative** -- a square, and a non-negative constant
        times an inner product of a vector with itself -- so their difference
        occupies no more magnitude than the larger of them and needs one bit
        for its sign, not two.  The right-hand side is bounded by the same
        budget through the scale identity, and :meth:`validate` checks it
        rather than assuming it.
        """
        magnitude = self.similarity_scale_bits + self.score.norm_bits
        headroom = self.score.ring_bits - 1
        # 2 * (magnitude - m) + 1 <= headroom
        needed = 2 * magnitude + 1 - headroom
        return max(0, -(-needed // 2))

    @property
    def reduction(self) -> int:
        """``m_s``, the bits dropped from ``s`` before it is squared."""
        return self.minimum_reduction

    @property
    def norm_reduction(self) -> int:
        """``m_n``, the bits dropped from ``n``.

        One truncation site carries one shift, so the norm drops as many bits
        as the similarity.  The bits it gives up are paid back by the threshold
        constant, which keeps 18 fractional bits at the deployed profile.
        """
        return self.reduction

    @property
    def reduction_length(self) -> int:
        """Declared width of the single reduction over ``[s, n]``.

        One site means one declared length for both, so it is the wider of the
        two.  Declaring the narrower one wider changes no integer -- a value
        that fits signed in its own width fits in any larger one.
        """
        return max(self.similarity_accumulator_bits,
                   self.norm_accumulator_bits)

    @property
    def constant_reduction(self) -> int:
        """``c = 2 m_s - m_n``: the bits the threshold constant gives up."""
        return 2 * self.reduction - self.norm_reduction

    def _product_magnitude_bits(self) -> int:
        constant = self.threshold_constant
        constant_bits = 0 if constant == 0 else int(constant).bit_length()
        return (self.norm_reduced_bits - 1) + constant_bits

    @property
    def threshold_constant(self) -> int:
        """``round(T^2 2^{2a-b-c})``, the public integer the emitter folds in."""
        exponent = self.constant_scale_bits - self.constant_reduction
        if exponent < 0:
            raise MatchError(
                f"a reduction of {self.reduction} bits leaves the threshold "
                f"constant no scale at all ({exponent} bits)")
        return int(round(self.threshold * self.threshold * (1 << exponent)))

    @property
    def similarity_reduced_bits(self) -> int:
        """Declared width of ``s >> m_s``."""
        return self.similarity_accumulator_bits - self.reduction

    @property
    def norm_reduced_bits(self) -> int:
        """Declared width of ``n >> m_n``."""
        return self.norm_accumulator_bits - self.norm_reduction

    @property
    def square_magnitude_bits(self) -> int:
        """Magnitude bits of ``(s >> m_s)^2``, which is never negative."""
        return 2 * (self.similarity_reduced_bits - 1)

    @property
    def square_bits(self) -> int:
        """Declared width of ``(s >> m_s)^2``, sign included."""
        return self.square_magnitude_bits + 1

    @property
    def product_bits(self) -> int:
        """Declared width of ``C * (n >> m_n)``, sign included."""
        return self._product_magnitude_bits() + 1

    @property
    def difference_bits(self) -> int:
        """Declared width of the compared difference, sign included.

        Both sides are non-negative, so the difference lies in
        ``[-max, +max]``: one sign bit above the wider magnitude, not two.
        """
        return max(self.square_magnitude_bits,
                   self._product_magnitude_bits()) + 1

    def declared_widths(self) -> dict[str, int]:
        return {"similarity": self.similarity_accumulator_bits,
                "norm": self.norm_accumulator_bits,
                "similarity_reduced": self.similarity_reduced_bits,
                "norm_reduced": self.norm_reduced_bits,
                "square": self.square_bits,
                "product": self.product_bits,
                "difference": self.difference_bits}

    def validate(self) -> MatchPlan:
        """Refuse a configuration that would wrap the ring or lose ``T``."""
        self.score.validate()
        if not -1.0 <= self.threshold <= 1.0:
            raise MatchError(
                f"a cosine threshold lives in [-1, 1], got {self.threshold}")
        if self.reduction >= self.similarity_accumulator_bits:
            raise MatchError(
                f"a reduction of {self.reduction} bits leaves the similarity "
                f"accumulator ({self.similarity_accumulator_bits} bits) nothing")
        if self.norm_reduction >= self.norm_accumulator_bits:
            raise MatchError(
                f"a reduction of {self.norm_reduction} bits leaves the norm "
                f"accumulator ({self.norm_accumulator_bits} bits) nothing")
        resolution = self.constant_scale_bits - self.constant_reduction
        if resolution < MINIMUM_THRESHOLD_RESOLUTION_BITS:
            raise MatchError(
                f"a reduction of {self.reduction} bits leaves the threshold "
                f"constant {resolution} fractional bits and at least "
                f"{MINIMUM_THRESHOLD_RESOLUTION_BITS} are needed: the decision "
                f"would quantize T rather than the accumulators")
        headroom = self.score.ring_bits - 1
        for name, width in self.declared_widths().items():
            if width > headroom:
                raise MatchError(
                    f"the {name} accumulator needs {width} bits and the ring "
                    f"leaves {headroom}")
        return self

    def decide(self, similarity: int, norm: int) -> int:
        """The match bit, from the two ring accumulators, in the clear.

        This is the *definition* the emitter realizes.  The reduction is a
        nearest truncation because that is what the emitted kernel performs;
        :mod:`ghostmpc.eval.match` models it through the shared ring oracle
        rather than reimplementing the shift here.
        """
        from ..eval.match import match_ring  # local: core stays NumPy-free
        return match_ring(similarity, norm, self)


def deployment_match_plan(score: ScorePlan, threshold: float) -> MatchPlan:
    """The certified plan, validated before anything is emitted against it."""
    return MatchPlan(score=score, threshold=float(threshold)).validate()


def match_plan_report(plan: MatchPlan) -> dict[str, object]:
    """What a report states about a compiled match, in one place."""
    return {
        "threshold": plan.threshold,
        "threshold_constant": plan.threshold_constant,
        "reduction_length": plan.reduction_length,
        "reduction_bits": plan.reduction,
        "norm_reduction_bits": plan.norm_reduction,
        "constant_reduction_bits": plan.constant_reduction,
        "declared_widths": plan.declared_widths(),
        "opened": ("nothing; the servers return share components of one match "
                   "bit and the client reconstructs it"),
        "decision": ("cos >= T, realized as a squared comparison so no square "
                     "root and no division is computed under MPC"),
    }

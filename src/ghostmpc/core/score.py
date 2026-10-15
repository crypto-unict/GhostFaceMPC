"""The 1:1 match score, and the arithmetic that makes it fit the ring.

The inference pipeline ends at a secret-shared embedding.  A deployment that
verifies an identity needs one more step: the three servers hold a database of
``(id, embedding)`` rows, secret-shared among them by a trusted dealer, and a
client that uploads a face and *declares* an identity expects the cosine
similarity between the two.

Because the claimed identity is declared, the row index is **public**: no
oblivious database access is needed and the database size does not enter the
request path.  What the servers compute is therefore two inner products of the
probe ``e`` with public-index operands,

.. math::

    s = \\langle e, g_{id} \\rangle, \\qquad n = \\langle e, e \\rangle,

and nothing is opened.  Each server hands the client its share components of
``(s, n)``; the client reconstructs and finishes in the clear:

.. math::

    \\cos(e, g) = \\frac{s\\,/\\,2^{f + f_g}}{\\sqrt{n\\,/\\,2^{2f}}}.

Three properties collapse this to two reshared ring elements.

**A normalized gallery removes every square root.**  The dealer stores
``\\|g\\| = 1``, so ``cos(e, g) = \\langle e, g \\rangle / \\|e\\|`` and no norm of a
database row is ever computed under MPC.

**Returning ``n`` is not a leak.**  The model parameters are public and the
image is the client's own, so the client can already derive ``e`` -- and hence
``\\|e\\|`` -- without the servers.  The servers learn neither, since the protocol
opens nothing.

**No truncation is needed.**  ``s`` is left on the raw ``2^{f + f_g}`` grid and
``n`` on ``2^{2f}``; the client, which is dividing anyway, absorbs both scales
into the one clear-text division.  The rounding kernel -- 93 % of the cost of an
inference -- never touches the score path.

The single constraint is that neither accumulator may wrap the ring.  With
``E = 2^f e``, ``G = 2^{f_g} g`` and ``\\|g\\| = 1``, Cauchy-Schwarz gives

.. math::

    |s| \\le 2^{f + f_g}\\|e\\|, \\qquad |n| \\le 2^{2f}\\|e\\|^2,

so the declared widths below have to stay inside ``ring_bits``.  Partial sums may
wrap; the final value is exact modulo ``2^{ring_bits}`` as long as the true value
fits signed, which is the same argument the raw convolution accumulators already
rest on.

This module is the single source of truth for those widths.  The emitter, the
dealer command and the clear-text oracle all build the same :class:`ScorePlan`,
so a grid stated in one of them cannot drift from the other two.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .model import (
    DEPLOYMENT_FRACTIONAL_BITS,
    DEPLOYMENT_RING_BITS,
)

# Database rows are unit vectors, so every component is in [-1, 1] and the grid
# only has to resolve them.  Fifteen fractional bits leave the similarity
# accumulator headroom above the largest embedding norm measured on the five
# verification datasets (41.99).
DEPLOYMENT_GALLERY_FRACTIONAL_BITS = 15

# Embedding norm the deployed widths are certified against: the measured
# maximum over the five verification datasets is 41.99, and this is the bound
# the emitter checks the ring against, kept deliberately above it.
DEPLOYMENT_NORM_BOUND = 64.0


class ScoreError(ValueError):
    """A score configuration that would wrap the computation ring."""


@dataclass(frozen=True)
class ScorePlan:
    """Widths and grids of one 1:1 verification score.

    ``gallery_rows`` sizes the database the servers ingest once at process
    setup; it does not enter the request path, because the claimed row index is
    public and a single row is read per request.
    """

    gallery_rows: int
    embedding_size: int
    fractional_bits: int = DEPLOYMENT_FRACTIONAL_BITS
    gallery_fractional_bits: int = DEPLOYMENT_GALLERY_FRACTIONAL_BITS
    ring_bits: int = DEPLOYMENT_RING_BITS
    norm_bound: float = DEPLOYMENT_NORM_BOUND

    @property
    def gallery_size(self) -> int:
        """Ring elements the dealer distributes, over the whole database."""
        return self.gallery_rows * self.embedding_size

    @property
    def norm_bits(self) -> int:
        """Integer bits of the certified embedding norm, ``ceil(log2 ||e||)``."""
        if self.norm_bound <= 0.0:
            raise ScoreError("the embedding norm bound must be positive")
        return max(0, math.ceil(math.log2(self.norm_bound)))

    @property
    def similarity_bits(self) -> int:
        """Declared width of ``s``: ``f + f_g + log2||e||`` plus its sign."""
        return (self.fractional_bits + self.gallery_fractional_bits
                + self.norm_bits + 1)

    @property
    def norm_product_bits(self) -> int:
        """Declared width of ``n``: ``2f + 2 log2||e||`` plus its sign."""
        return 2 * self.fractional_bits + 2 * self.norm_bits + 1

    def declared_widths(self) -> dict[str, int]:
        return {"similarity": self.similarity_bits,
                "norm": self.norm_product_bits}

    def validate(self) -> ScorePlan:
        """Refuse a configuration whose accumulators could wrap the ring."""
        if self.gallery_rows < 1:
            raise ScoreError("a gallery holds at least one row")
        if self.embedding_size < 1:
            raise ScoreError("an embedding has at least one component")
        if self.gallery_fractional_bits < 1:
            raise ScoreError("the gallery grid needs at least one bit")
        headroom = self.ring_bits - 1
        for name, width in self.declared_widths().items():
            if width > headroom:
                raise ScoreError(
                    f"the {name} accumulator needs {width} bits and the ring "
                    f"leaves {headroom}: lower the gallery grid "
                    f"(f_g = {self.gallery_fractional_bits}) or the certified "
                    f"norm bound (||e|| <= {self.norm_bound})")
        return self

    def cosine(self, similarity: int, norm: int) -> float:
        """Finish the score in the clear, as the client does."""
        if norm <= 0:
            raise ScoreError("a zero-norm probe has no direction")
        scaled = similarity / (1 << (self.fractional_bits
                                     + self.gallery_fractional_bits))
        magnitude = math.sqrt(norm / (1 << (2 * self.fractional_bits)))
        return scaled / magnitude


def deployment_score_plan(gallery_rows: int, embedding_size: int, *,
                          fractional_bits: int = DEPLOYMENT_FRACTIONAL_BITS,
                          gallery_fractional_bits: int | None = None,
                          ring_bits: int = DEPLOYMENT_RING_BITS,
                          norm_bound: float = DEPLOYMENT_NORM_BOUND
                          ) -> ScorePlan:
    """The certified plan, validated before anything is emitted against it."""
    return ScorePlan(
        gallery_rows=int(gallery_rows), embedding_size=int(embedding_size),
        fractional_bits=int(fractional_bits),
        gallery_fractional_bits=(DEPLOYMENT_GALLERY_FRACTIONAL_BITS
                                 if gallery_fractional_bits is None
                                 else int(gallery_fractional_bits)),
        ring_bits=int(ring_bits), norm_bound=float(norm_bound)).validate()

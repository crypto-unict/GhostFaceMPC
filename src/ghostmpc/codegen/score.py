"""Emission of the 1:1 verification score.

The arithmetic and the widths live in :mod:`ghostmpc.core.score`; this module
only turns a validated :class:`~ghostmpc.core.score.ScorePlan` into the two
source fragments the program needs -- the once-per-process database ingestion
and the two inner products a request computes -- plus the match decision and
the closed-set identification that read them.
"""

from __future__ import annotations

from ..core.identify import IdentifyPlan
from ..core.match import MatchPlan
from ..core.score import ScorePlan
from .naming import (
    GALLERY_LOAD_TIMER_ID,
    IDENTIFY_TIMER_ID,
    MATCH_TIMER_ID,
    SCORE_TIMER_ID,
)
from .templates import (
    CLAIMED_IDENTITY,
    DELIVERY,
    IDENTIFY_CARRY,
    IDENTIFY_CARRY_FIRST,
    IDENTIFY_INDEX,
    IDENTIFY_INDEX_FIRST,
    IDENTIFY_LEVEL,
    IDENTIFY_OUTPUT,
    IDENTIFY_SETUP,
    IDENTIFY_SIMILARITIES,
    MATCH_BLOCK,
    MATCH_BLOCK_SIGN_ONLY,
    MATCH_REDUCTION_FUSED,
    MATCH_SETUP,
    SCORE_BLOCK,
    SCORE_SETUP,
)

#: The row a request reads when the identity is not checked: the benchmark's
#: own public counter, which every server agrees on by construction.
COUNTER_ROW = "(request_id % GALLERY_ROWS)"
#: The row a request reads with verified input: the claimed identity, read by
#: ``claimed_identity()`` and range-checked there.
CLAIMED_ROW = "request_identity"


def delivery(output: str) -> str:
    """The verified hand-off of one request's answer (malicious profile).

    The post-sacrifice check of every pending product runs first; then every
    server writes the share pair it holds, so the client receives each
    component from its two holders.
    """
    return DELIVERY.format(output=output)


def setup(plan: ScorePlan, *, measured_timer: bool,
          match: MatchPlan | None = None,
          identify: IdentifyPlan | None = None,
          verified_io: bool = False) -> str:
    """The database ingestion, emitted once before the request loop.

    It is timed like the public-weight load and for the same reason: a trusted
    dealer distributed these components ahead of any request, so charging them
    to an inference would describe a deployment nobody runs.

    The dealer's replicated sharing sits at the head of every server's share
    file and is loaded locally.  With ``verified_io`` a 1:1 program also
    defines the identity read its requests call.
    """
    source = SCORE_SETUP.format(
        gallery_rows=plan.gallery_rows,
        embedding_size=plan.embedding_size,
        gallery_fractional_bits=plan.gallery_fractional_bits,
        gallery_load_open=(f"start_timer({GALLERY_LOAD_TIMER_ID})\n"
                           if measured_timer else ""),
        gallery_load_close=(f"stop_timer({GALLERY_LOAD_TIMER_ID})\n"
                            if measured_timer else ""),
    )
    if identify is not None:
        return (source.rstrip("\n") + "\n"
                + IDENTIFY_SETUP.format(candidates=identify.candidates))
    if verified_io:
        source = source.rstrip("\n") + "\n\n" + CLAIMED_IDENTITY
    if match is None:
        return source
    return source.rstrip("\n") + "\n" + MATCH_SETUP


def block(plan: ScorePlan, embedding: str, *, measured_timer: bool,
          test_reveal: bool = False, row: str = COUNTER_ROW) -> str:
    """The per-request score, bracketed by its own timer."""
    lines = ["marker('start', 'score')\n"]
    if measured_timer:
        lines.append(f"@if_(request_id >= WARMUPS)\ndef _():\n"
                     f"    start_timer({SCORE_TIMER_ID})\n")
    lines.append(SCORE_BLOCK.format(embedding=embedding, row=row))
    if measured_timer:
        lines.append(f"@if_(request_id >= WARMUPS)\ndef _():\n"
                     f"    stop_timer({SCORE_TIMER_ID})\n")
    lines.append("marker('stop', 'score')\n")
    if test_reveal:
        # Never on a deploy build: the servers open nothing and hand their share
        # components to the client.  The equivalence driver needs the integers.
        lines.append(
            "print_ln('GHOSTMPC_TEST_SCORE %s %s', "
            "score_output[0].reveal(), score_output[1].reveal())\n")
    return "".join(lines)


def _tournament(plan: IdentifyPlan) -> str:
    """The comparison tree, unrolled at compile time.

    The shape is :meth:`~ghostmpc.core.identify.IdentifyPlan.level_widths`, so
    the emitter and :func:`ghostmpc.eval.identify.tournament` cannot disagree
    about which elements meet.  Two properties keep the cost down:

    * a **level is one comparison site**.  The vector is folded in half, so
      element ``i`` meets element ``i + W``, both operands are contiguous
      slices, and ``is_negative`` compares the whole level at once: the tree
      costs ``ceil(log2 N)`` sequential comparisons, not ``N - 1``;
    * the **first level selects its index for free**: the two candidates are
      ``i`` and ``i + W``, both public, so the winner is ``i + bW`` and no
      product is paid.  Only from the second level on is the index secret.
    """
    values, index = "identify_level0", None
    lines: list[str] = []
    length = plan.candidates
    for level, width in enumerate(plan.level_widths()):
        carry = length % 2
        next_length = width + carry
        next_values = f"identify_level{level + 1}"
        next_index = f"identify_index{level + 1}"
        lines.append(IDENTIFY_LEVEL.format(
            values=values, width=width, next_values=next_values,
            next_length=next_length,
            difference_bits=plan.difference_bits))
        lines.append(
            IDENTIFY_INDEX_FIRST.format(
                next_index=next_index, next_length=next_length, width=width)
            if index is None else
            IDENTIFY_INDEX.format(
                index=index, width=width, next_index=next_index,
                next_length=next_length))
        if carry:
            template = IDENTIFY_CARRY_FIRST if index is None else IDENTIFY_CARRY
            lines.append(template.format(
                values=values, index=index, next_values=next_values,
                next_index=next_index, last=width,
                last_source=length - 1))
        values, index = next_values, next_index
        length = next_length
    if index is None:
        # A one-row database has nothing to compare: the answer is row zero.
        lines.append("identify_index1 = sint.Array(1)\n"
                     "identify_index1[0] = sint(0)\n")
        index = "identify_index1"
    lines.append(IDENTIFY_OUTPUT.format(index=index, values=values))
    return "".join(lines)


def identify_block(plan: IdentifyPlan, embedding: str, *, measured_timer: bool,
                   test_reveal: bool = False) -> str:
    """The per-request closed-set identification, bracketed by its own timer."""
    lines = ["marker('start', 'identify')\n"]
    if measured_timer:
        lines.append(f"@if_(request_id >= WARMUPS)\ndef _():\n"
                     f"    start_timer({IDENTIFY_TIMER_ID})\n")
    lines.append(IDENTIFY_SIMILARITIES.format(embedding=embedding))
    lines.append(_tournament(plan))
    if measured_timer:
        lines.append(f"@if_(request_id >= WARMUPS)\ndef _():\n"
                     f"    stop_timer({IDENTIFY_TIMER_ID})\n")
    lines.append("marker('stop', 'identify')\n")
    if test_reveal:
        # Never on a deploy build: the servers hand back share components of the
        # rank-1 index and the client reconstructs it.
        lines.append(
            "print_ln('GHOSTMPC_TEST_SIMILARITIES %s', "
            "identify_level0.get_vector().reveal())\n"
            "print_ln('GHOSTMPC_TEST_IDENTIFY %s %s', "
            "identify_output[0].reveal(), identify_output[1].reveal())\n")
    return "".join(lines)


def _reduction(plan: MatchPlan) -> str:
    """The accumulator reductions, or nothing when the ring needs none.

    The two reductions are **one** truncation site over a two-element vector,
    and the sign the branch needs is the one that site already computes:
    ``trunc_round_nearest_with_sign`` returns the sign of its own result, so no
    separate comparison is paid for it.
    """
    if plan.reduction == 0:
        return ("match_similarity = score_output[0]\n"
                "match_norm = score_output[1]\n")
    return MATCH_REDUCTION_FUSED.format(
        reduction_length=plan.reduction_length, reduction=plan.reduction)


def match_block(plan: MatchPlan, *, measured_timer: bool,
                test_reveal: bool = False) -> str:
    """The per-request match decision, bracketed by its own timer.

    One reduction of ``[s, n]``, one secret square, one comparison of the
    squared form, and one secret product to combine the two bits.  The public
    threshold constant multiplies a secret locally and communicates nothing.
    """
    lines = ["marker('start', 'match')\n"]
    if measured_timer:
        lines.append(f"@if_(request_id >= WARMUPS)\ndef _():\n"
                     f"    start_timer({MATCH_TIMER_ID})\n")
    if plan.threshold_constant == 0:
        lines.append(MATCH_BLOCK_SIGN_ONLY.format(
            similarity_bits=plan.similarity_accumulator_bits))
    else:
        reduction = _reduction(plan)
        sign = ""
        if plan.reduction == 0:
            sign = ("match_negative = is_negative(score_output[0], "
                    f"{plan.similarity_accumulator_bits})\n")
        combination = ("(1 - match_negative) * (1 - match_below)"
                       if plan.threshold >= 0.0
                       else "1 - match_negative * (1 - match_below)")
        lines.append(MATCH_BLOCK.format(
            reduction=reduction, sign=sign,
            threshold_constant=plan.threshold_constant,
            difference_bits=plan.difference_bits,
            combination=combination))
    if measured_timer:
        lines.append(f"@if_(request_id >= WARMUPS)\ndef _():\n"
                     f"    stop_timer({MATCH_TIMER_ID})\n")
    lines.append("marker('stop', 'match')\n")
    if test_reveal:
        # Never on a deploy build, for the same reason the score's reveal is not:
        # the servers hand back share components and the client reconstructs.
        lines.append("print_ln('GHOSTMPC_TEST_MATCH %s', "
                     "match_output[0].reveal())\n")
    return "".join(lines)

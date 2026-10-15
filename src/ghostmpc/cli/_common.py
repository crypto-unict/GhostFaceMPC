"""Helpers shared by more than one command group."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ..core.identify import IdentifyPlan, deployment_identify_plan
from ..core.match import MatchPlan, deployment_match_plan
from ..core.model import Manifest
from ..core.score import (
    DEPLOYMENT_GALLERY_FRACTIONAL_BITS,
    DEPLOYMENT_NORM_BOUND,
    ScorePlan,
    deployment_score_plan,
)


def _threads(value: int) -> int:
    if value:
        return value
    available = os.cpu_count() or 1
    return max(x for x in (1, 2, 4, 8) if x <= available)


def add_score_arguments(parser: argparse._ActionsContainer) -> None:
    """Verification-head flags, identical on every command that compiles one."""
    parser.add_argument(
        "--score", action="store_true",
        help="append the 1:1 verification score: two secret-by-secret inner "
             "products against a secret-shared database row whose index the "
             "client declares, opened by nobody and finished by the client")
    parser.add_argument(
        "--gallery-rows", type=int, default=1024,
        help="database rows the servers ingest once at process setup; the "
             "claimed row index is public, so this never enters a request")
    parser.add_argument(
        "--gallery-fractional-bits", type=int,
        default=DEPLOYMENT_GALLERY_FRACTIONAL_BITS,
        help="grid of the L2-normalized database rows (default %(default)s)")
    parser.add_argument(
        "--gallery-norm-bound", type=float, default=DEPLOYMENT_NORM_BOUND,
        help="certified maximum embedding norm the declared score widths are "
             "checked against (default %(default)s)")
    parser.add_argument(
        "--match", action="store_true",
        help="append the match decision: the servers compare the cosine "
             "against a public threshold under MPC and return share "
             "components of one bit, opening nothing.  Implies --score")
    parser.add_argument(
        "--threshold", type=float,
        help="the public decision threshold T on the cosine, required with "
             "--match; the ten-fold protocol calibrates it in the clear")
    # Identification is the query the score and the match do not answer, so
    # it is a mode of its own rather than a modifier of theirs.
    parser.add_argument(
        "--identify", action="store_true",
        help="append the closed-set 1:N identification: the probe declares no "
             "identity, the servers rank it against every enrolled row and "
             "return share components of the rank-1 index alone, opening "
             "nothing.  Ingests the gallery exactly as --score does and "
             "excludes --score/--match, which answer a declared identity")
    parser.add_argument(
        "--identify-candidates", type=int,
        help="rows ranked per request; the default is the whole database, "
             "which is what closed-set means")


def score_plan(args: argparse.Namespace, manifest_path: Path, *,
               fractional_bits: int, ring_bits: int) -> ScorePlan | None:
    """Build the validated plan the flags describe, or ``None`` for no score."""
    if not (getattr(args, "score", False) or getattr(args, "identify", False)):
        return None
    manifest = Manifest.load(manifest_path)
    embedding_size = 1
    for extent in manifest.output_shape:
        embedding_size *= int(extent)
    return deployment_score_plan(
        args.gallery_rows, embedding_size, fractional_bits=fractional_bits,
        gallery_fractional_bits=args.gallery_fractional_bits,
        ring_bits=ring_bits, norm_bound=args.gallery_norm_bound)


def match_plan(args: argparse.Namespace,
               score: ScorePlan | None) -> MatchPlan | None:
    """Build the validated match plan the flags describe, or ``None``.

    ``--match`` without a score is a mistake rather than a mode: the decision
    reads the two score accumulators, so it cannot be compiled without them.
    """
    if not getattr(args, "match", False):
        return None
    if score is None:
        raise ValueError("--match reads the score accumulators: pass --score")
    threshold = getattr(args, "threshold", None)
    if threshold is None:
        raise ValueError("--match needs the public threshold: pass --threshold")
    return deployment_match_plan(score, threshold)


def identify_plan(args: argparse.Namespace,
                  score: ScorePlan | None) -> IdentifyPlan | None:
    """Build the validated identification plan the flags describe, or ``None``.

    ``--identify`` builds its own ``ScorePlan`` through :func:`score_plan` --
    that is what ingests the database -- so a missing one here is an internal
    inconsistency rather than a user error.
    """
    if not getattr(args, "identify", False):
        return None
    if getattr(args, "match", False):
        raise ValueError(
            "--match answers a declared identity and --identify answers none: "
            "compile one or the other")
    if score is None:
        raise ValueError("--identify ranks the enrolled database: pass a manifest "
                         "whose score plan could be built")
    return deployment_identify_plan(
        score, getattr(args, "identify_candidates", None))

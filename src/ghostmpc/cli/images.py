"""Client-side and dealer-side secret sharing of the inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..core.model import DEPLOYMENT_FRACTIONAL_BITS, DEPLOYMENT_RING_BITS
from ..core.score import (
    DEPLOYMENT_GALLERY_FRACTIONAL_BITS,
    DEPLOYMENT_NORM_BOUND,
    deployment_score_plan,
)
from ..core.sharing import (
    ReplicaMismatch,
    read_persistence,
    reconstruct_replicated,
    replicate_components,
    share_fixed_n,
    share_ring_n,
    write_clear_persistence,
    write_persistence,
)


def _write_replicated(output: Path, shares: list[list[int]], ring_bits: int,
                      *, append: bool = False) -> None:
    """Hand party ``i`` its pair ``(x_i, x_{i-1})`` of the replicated sharing.

    One ``Transactions-P<i>.data`` per server, in the format MP-SPDZ's
    ``sint.read_from_file`` loads: the servers compute on this sharing as it
    is, so nothing is re-shared on their side.
    """
    for party, pair in enumerate(replicate_components(shares)):
        write_persistence(output / f"Transactions-P{party}.data", pair,
                          ring_bits, append=append)


def _share_image(args: argparse.Namespace) -> int:
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("share-image requires `pip install -e .[export]`") from exc
    image = Image.open(args.image).convert("RGB")
    if image.size != (112, 112):
        raise SystemExit(f"expected aligned 112x112 image, got {image.size}")
    values = [(channel / 127.5) - 1.0 for pixel in image.getdata() for channel in pixel]
    shares = share_fixed_n(values, args.fractional_bits, 3, args.ring_bits)
    output = Path(args.output)
    # These files are intended for each server's Persistence/ directory; with
    # --append the probe follows the enrolled database already there.
    _write_replicated(output, shares, args.ring_bits, append=args.append)
    if args.identity is not None:
        # Public, and the same for every server: the malicious 1:1 program reads
        # it as the row to score against.
        for party in range(3):
            write_clear_persistence(
                output / f"Transactions-clear-P{party}.data", [args.identity],
                args.ring_bits, append=args.append)
    print(f"wrote the replicated sharing of {len(shares[0])} ring elements "
          f"to {output}")
    return 0


def _share_gallery(args: argparse.Namespace) -> int:
    """Act as the trusted dealer of a secret-shared verification database.

    The rows are L2-normalized here, once and offline, which is what removes
    every square root from the online protocol: with ``||g|| = 1`` the cosine
    against a probe ``e`` is ``<e, g> / ||e||``, so the servers only ever compute
    two inner products and never a norm of a database row.
    """
    import numpy as np

    from ..eval.score import encode_gallery

    if args.embeddings:
        embeddings = np.load(args.embeddings)
        if embeddings.ndim != 2:
            raise SystemExit(
                f"expected a (rows, dimension) array, got {embeddings.shape}")
        rows, dimension = embeddings.shape
        identifiers = (
            [line.strip() for line in Path(args.ids).read_text().splitlines()
             if line.strip()] if args.ids else [str(i) for i in range(rows)])
        if len(identifiers) != rows:
            raise SystemExit(
                f"{len(identifiers)} identifiers for {rows} embeddings")
    else:
        # A synthetic database. The protocol's cost does not depend on the
        # values -- every row is the same number of ring elements -- so a
        # measurement never needs a real enrolment, and this keeps one out of
        # the benchmark path.
        rows, dimension = args.rows, args.dimension
        generator = np.random.default_rng(args.seed)
        embeddings = generator.normal(size=(rows, dimension))
        identifiers = [f"synthetic-{i}" for i in range(rows)]
    plan = deployment_score_plan(
        rows, dimension, fractional_bits=args.fractional_bits,
        gallery_fractional_bits=args.gallery_fractional_bits,
        ring_bits=args.ring_bits, norm_bound=args.norm_bound)
    encoded = encode_gallery(embeddings, plan)
    shares = share_ring_n(
        [int(value) % (1 << plan.ring_bits) for value in encoded.reshape(-1)],
        3, plan.ring_bits)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    # The database opens every server's share file; each request's probe is
    # appended after it (``share-image --append``).
    _write_replicated(output, shares, plan.ring_bits)
    np.save(output / "gallery-encoded.npy", encoded)
    metadata = {
        "rows": plan.gallery_rows,
        "embedding_size": plan.embedding_size,
        "fractional_bits": plan.fractional_bits,
        "gallery_fractional_bits": plan.gallery_fractional_bits,
        "ring_bits": plan.ring_bits,
        "norm_bound": plan.norm_bound,
        "declared_widths": plan.declared_widths(),
        "source": args.embeddings or f"synthetic(seed={args.seed})",
        "ids": identifiers,
    }
    (output / "gallery.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output),
        "rows": plan.gallery_rows,
        "ring_elements_per_party": plan.gallery_size,
        "declared_widths": plan.declared_widths(),
        "note": "give server i Transactions-P<i>.data as its "
                "Persistence/Transactions-P<i>.data; the program loads the "
                "database before the request loop",
    }, indent=2))
    return 0


def _open_result(args: argparse.Namespace) -> int:
    """The client's side of the verified delivery (malicious profile).

    Each server hands back ``Persistence/Transactions-P<i>.data``, the pair it
    holds of every delivered value.  Every component therefore arrives twice,
    from its two holders, and any disagreement rejects the answer.
    """
    pairs = [read_persistence(path, args.ring_bits, last=args.last)
             for path in args.shares]
    try:
        values = reconstruct_replicated(pairs, args.ring_bits)
    except ReplicaMismatch as exc:
        print(json.dumps({"status": "rejected", "reason": str(exc)}))
        return 1
    half = 1 << (args.ring_bits - 1)
    signed = [value - (1 << args.ring_bits) if value >= half else value
              for value in values]
    print(json.dumps({"status": "accepted", "values": signed}))
    return 0


def register(commands: argparse._SubParsersAction) -> None:
    """Add this group's subcommands, in the order they are listed."""
    share = commands.add_parser("share-image", help="secret-share an aligned image")
    share.add_argument("image")
    share.add_argument("output")
    share.add_argument(
        "--fractional-bits", type=int, choices=(6, 8, 12, 16, 20, 24),
        default=DEPLOYMENT_FRACTIONAL_BITS)
    share.add_argument(
        "--ring-bits", type=int, choices=(32, 64),
        default=DEPLOYMENT_RING_BITS)
    share.add_argument(
        "--append", action="store_true",
        help="append to the servers' share files (after the enrolled "
             "database or earlier requests) instead of starting them")
    share.add_argument(
        "--identity", type=int,
        help="claimed database row, written to each Transactions-clear file")
    share.set_defaults(func=_share_image)
    gallery = commands.add_parser(
        "share-gallery",
        help="act as the trusted dealer: secret-share an L2-normalized "
             "verification database among the three servers")
    gallery.add_argument("output")
    gallery.add_argument(
        "--embeddings",
        help="(rows, dimension) .npy of enrolled embeddings; omit for a "
             "synthetic database of the same size and cost")
    gallery.add_argument("--ids", help="one identifier per line, row order")
    gallery.add_argument("--rows", type=int, default=1024,
                         help="rows of the synthetic database")
    gallery.add_argument("--dimension", type=int, default=512)
    gallery.add_argument("--seed", type=int, default=0)
    gallery.add_argument(
        "--fractional-bits", type=int, choices=(6, 8, 12, 16, 20, 24),
        default=DEPLOYMENT_FRACTIONAL_BITS,
        help="activation grid of the probe the database is scored against")
    gallery.add_argument(
        "--gallery-fractional-bits", type=int,
        default=DEPLOYMENT_GALLERY_FRACTIONAL_BITS,
        help="grid of the normalized rows (default %(default)s)")
    gallery.add_argument(
        "--ring-bits", type=int, choices=(32, 64), default=DEPLOYMENT_RING_BITS)
    gallery.add_argument(
        "--norm-bound", type=float, default=DEPLOYMENT_NORM_BOUND,
        help="certified maximum embedding norm the widths are checked against")
    gallery.set_defaults(func=_share_gallery)
    opened = commands.add_parser(
        "open-result",
        help="reconstruct a verified answer from the three servers' share "
             "files, rejecting it if two holders disagree")
    opened.add_argument(
        "shares", nargs=3,
        help="Persistence/Transactions-P0.data, -P1.data and -P2.data")
    opened.add_argument(
        "--last", type=int,
        help="read only the final LAST shares: the servers append their "
             "answer after the input shares in the same file")
    opened.add_argument(
        "--ring-bits", type=int, choices=(32, 64), default=DEPLOYMENT_RING_BITS)
    opened.set_defaults(func=_open_result)

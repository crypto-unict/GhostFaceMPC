#!/usr/bin/env python3
"""Write one primary's stdin stream: public weights, gallery row, request shares.

The enrolled database row of a scored binary is a setup input like the weights,
not a per-request one, and the injected score reads it after every public
weight and before the request loop, so it is written in that position:
``weights, [gallery,] share_0, share_1, ...``.  P2 receives no stream.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def write_stream(weights: Path | None, shares: list[Path], output: Path, *,
                 gallery_share: Path | None = None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        if weights is not None:
            with weights.open() as source:
                for block in iter(lambda: source.read(1024 * 1024), ""):
                    handle.write(block)
        if gallery_share is not None:
            gallery = np.load(gallery_share)
            if gallery.dtype != np.uint64:
                raise ValueError(
                    f"{gallery_share}: expected uint64, got {gallery.dtype}")
            for value in gallery.reshape(-1):
                handle.write(f"{int(value)}\n")
        for share_path in shares:
            share = np.load(share_path)
            if share.dtype != np.uint64 or tuple(share.shape) != (1, 3, 112, 112):
                raise ValueError(
                    f"{share_path}: expected uint64 (1,3,112,112), got {share.dtype} {share.shape}")
            for value in share.reshape(-1):
                handle.write(f"{int(value)}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--gallery-share", type=Path,
                        help="this party's component of the enrolled row read "
                             "by the score; omit for a program without one")
    parser.add_argument("--share", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_stream(args.weights, args.share, args.output,
                 gallery_share=args.gallery_share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

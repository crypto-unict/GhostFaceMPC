"""Fetching the teacher checkpoint, the model bundles and the pinned MP-SPDZ source."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

from ..core.model import CHECKPOINT_NAME, CHECKPOINT_URL, digest

MPSPDZ_URL = "https://github.com/data61/MP-SPDZ.git"

#: Where the model bundles are published.  ``GHOSTMPC_RELEASE_BASE`` repoints
#: the whole set at a mirror without a code change.
RELEASE_BASE = os.environ.get(
    "GHOSTMPC_RELEASE_BASE",
    "https://github.com/crypto-unict/ghostfacempc/releases/download/v1.0")

#: The bundles, by the directory each one extracts to under ``artifacts/``.
#:
#: ``model`` is the quantized ``ring32-f6-fw9`` model every GhostFaceMPC row is
#: measured on: a QAT bundle trained for 72,784 steps on MS1MV3, with its five
#: accuracy reports under ``gates/``.  ``qat-start`` is the manifest that
#: training started from: the float teacher exported on the ring32/f6/f_w9
#: profile, with overflow bounds calibrated on the MS1MV3 validation split.
#: ``model-ring64-f16-b0`` is the ring64/f16 manifest the Porthos baseline is
#: exported from, and ``legacy-scalar-236`` the 236-operation ring64/f20
#: manifest of the naive MP-SPDZ baseline.
#:
#: The archives are built with ``tar --sort=name --mtime --owner=0 --group=0``,
#: so rebuilding one from the same directory reproduces the digest below.
BUNDLES = {
    "model": {
        "asset": "ghostmpc-model-ring32-f6-fw9.tar.gz",
        "sha256": "4b89b39d41b3f65b7936ff54aca16e0d6bdf15678f9e743bd784fa4ce9744f20",
    },
    "model-ring64-f16-b0": {
        "asset": "ghostmpc-model-ring64-f16-b0.tar.gz",
        "sha256": "ff3f3c7915e9f098947d94e6ccf87218b8d24d6e7e66abf147f0e8db3e0d80c5",
    },
    "legacy-scalar-236": {
        "asset": "ghostmpc-legacy-scalar-236.tar.gz",
        "sha256": "7ad7628a7eb3d3a9ad9d5b74ba37056d459b0e16229d6065419811a698e49d3f",
    },
    "qat-start": {
        "asset": "ghostmpc-qat-start-ring32-f6-fw9.tar.gz",
        "sha256": "36fe1d1a517b2a23c6cdf57ea9464e392a139d143c45f2e1d1a4f6945b7672e3",
    },
}

#: The bundles the experiments read; ``qat-start`` is only needed to retrain.
DEFAULT_BUNDLES = ("model", "model-ring64-f16-b0", "legacy-scalar-236")

MPSPDZ_COMMIT = "6a2256e327b507918859f605735543bb32a39d9d"

# The program does not run on stock MP-SPDZ: it targets virtual-machine
# instructions this repository adds -- the two-summand rounding shift
# ``shrsi2`` and the ``prefetchdabit``/``prefetchprep`` instructions that fill
# one request's preprocessing before it starts.  The submodule is pinned to an
# upstream commit, so those changes travel as a patch.
MPSPDZ_PATCHES = Path("third_party/patches")

# The deployed profile computes in the ring Z_{2^32}, which the ring machines
# only instantiate when built with this flag (64-bit rings stay available).
MPSPDZ_CONFIG = "MY_CFLAGS += -DRING_SIZE=32\n"


def _fetch(args: argparse.Namespace) -> int:
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not args.force:
        print(f"exists: {destination} ({digest(destination)})")
        return 0
    urlretrieve(CHECKPOINT_URL, destination)
    print(json.dumps({"file": str(destination), "url": CHECKPOINT_URL, "sha256": digest(destination)}, indent=2))
    return 0


def _fetch_bundles(args: argparse.Namespace) -> int:
    """Download and unpack the generated bundles, verifying each digest.

    A wrong bundle does not fail loudly downstream -- it measures a different
    program, or silently reproduces nothing -- so the digest is checked before
    the archive is unpacked and a mismatch leaves nothing behind.
    """
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    selected = args.bundles or list(DEFAULT_BUNDLES)
    report = []
    for name in selected:
        bundle = BUNDLES[name]
        target = destination / name
        if target.exists() and not args.force:
            report.append({"bundle": name, "status": "exists",
                           "directory": str(target)})
            continue
        url = f"{RELEASE_BASE}/{bundle['asset']}"
        with tempfile.TemporaryDirectory() as scratch:
            archive = Path(scratch) / bundle["asset"]
            urlretrieve(url, archive)
            measured = digest(archive)
            if measured != bundle["sha256"]:
                raise SystemExit(
                    f"{url} has sha256 {measured}, expected "
                    f"{bundle['sha256']}; refusing to unpack it")
            # Each archive is rooted at the bundle's own directory name, so it
            # extracts beside its siblings rather than over them.
            with tarfile.open(archive) as handle:
                _extract(handle, destination)
        report.append({"bundle": name, "status": "fetched", "url": url,
                       "sha256": measured, "directory": str(target)})
    print(json.dumps(report, indent=2))
    return 0


def _extract(handle: tarfile.TarFile, destination: Path) -> None:
    """Unpack, refusing any member that would escape ``destination``."""
    root = destination.resolve()
    for member in handle.getmembers():
        if not member.isfile() and not member.isdir():
            raise SystemExit(f"unexpected member type in the archive: {member.name}")
        if not (root / member.name).resolve().is_relative_to(root):
            raise SystemExit(f"archive member escapes {root}: {member.name}")
    handle.extractall(destination)


def _apply_patches(target: Path, patches: Path) -> list[dict[str, str]]:
    """Apply the ghostmpc virtual-machine extensions to a pinned checkout.

    Each patch is applied only if it is not already in the tree, so the command
    is idempotent and a developer who edited the submodule by hand is told
    rather than overwritten.  The digest is printed so a checkout can be
    identified from a report.
    """
    applied = []
    for patch in sorted(patches.glob("*.patch")):
        common = ["git", "apply", "--3way", str(patch.resolve())]
        already = subprocess.run(
            ["git", "apply", "--reverse", "--check", str(patch.resolve())],
            cwd=target, capture_output=True)
        if already.returncode == 0:
            applied.append({"patch": patch.name, "status": "already applied",
                            "sha256": digest(patch)})
            continue
        result = subprocess.run(common, cwd=target, capture_output=True,
                                text=True)
        if result.returncode != 0:
            raise SystemExit(
                f"cannot apply {patch.name} to {target}: "
                f"{result.stderr.strip()}")
        applied.append({"patch": patch.name, "status": "applied",
                        "sha256": digest(patch)})
    return applied


def _bootstrap(args: argparse.Namespace) -> int:
    target = Path(args.directory)
    # An uninitialized submodule is an empty directory: clone into it rather
    # than running git inside it, which would act on the enclosing repository.
    if not (target / ".git").exists():
        if target.exists() and any(target.iterdir()):
            raise SystemExit(f"{target} exists and is not a git checkout")
        subprocess.run(["git", "clone", MPSPDZ_URL, str(target)], check=True)
    subprocess.run(["git", "fetch", "--all", "--tags"], cwd=target, check=True)
    subprocess.run(["git", "checkout", "--detach", MPSPDZ_COMMIT], cwd=target, check=True)
    applied = _apply_patches(target, Path(args.patches))
    config = target / "CONFIG.mine"
    existing = config.read_text() if config.is_file() else ""
    if MPSPDZ_CONFIG.strip() not in existing:
        config.write_text(existing + MPSPDZ_CONFIG)
    print(json.dumps({"directory": str(target), "commit": MPSPDZ_COMMIT,
                      "patches": applied, "config": MPSPDZ_CONFIG.strip()},
                     indent=2))
    return 0


def register(commands: argparse._SubParsersAction) -> None:
    """Add this group's subcommands, in the order they are listed."""
    fetch = commands.add_parser(
        "fetch-checkpoint",
        help="download the official GhostFaceNetV1 float checkpoint (the teacher)")
    fetch.add_argument("--output", default=f"artifacts/{CHECKPOINT_NAME}")
    fetch.add_argument("--force", action="store_true")
    fetch.set_defaults(func=_fetch)
    bundles = commands.add_parser(
        "fetch-bundles",
        help="download the model bundles the experiments are measured on")
    bundles.add_argument("--output", default="artifacts")
    bundles.add_argument("--bundle", action="append", dest="bundles",
                         choices=sorted(BUNDLES),
                         help="fetch only these bundles; repeatable, defaults to "
                              + ", ".join(DEFAULT_BUNDLES))
    bundles.add_argument("--force", action="store_true",
                         help="re-download a bundle whose directory already exists")
    bundles.set_defaults(func=_fetch_bundles)
    bootstrap = commands.add_parser(
        "bootstrap-mpspdz",
        help="check out the pinned MP-SPDZ source, apply the ghostmpc VM "
             "extensions and enable the 32-bit ring")
    bootstrap.add_argument("--directory", default="third_party/MP-SPDZ")
    bootstrap.add_argument("--patches", default=str(MPSPDZ_PATCHES),
                           help="directory of *.patch files to apply after checkout")
    bootstrap.set_defaults(func=_bootstrap)

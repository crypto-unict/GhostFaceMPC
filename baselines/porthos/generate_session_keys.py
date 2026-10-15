#!/usr/bin/env python3
"""Generate a fresh, permission-restricted Porthos PRG key set per pool."""

from __future__ import annotations

import argparse
import json
import secrets
from hashlib import sha256
from pathlib import Path

KEY_NAMES = ("keyA", "keyAB", "keyB", "keyD")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to reuse non-empty key directory: {args.output}")
    args.output.mkdir(parents=True, mode=0o700, exist_ok=True)
    hashes = {}
    for name in KEY_NAMES:
        # Upstream reads the file into a C string and copies its first 16
        # bytes. Hex avoids embedded NUL/newline truncation while retaining
        # 128 bits of entropy per key file.
        value = secrets.token_hex(16).encode("ascii")
        path = args.output / name
        path.write_bytes(value)
        path.chmod(0o600)
        hashes[name] = sha256(value).hexdigest()
    manifest = {
        "schema_version": 1, "pool": args.pool, "fresh_per_pool": True,
        "entropy_bytes_per_key": 16, "serialized_bytes_per_key": 32,
        "key_sha256": hashes,
        "warning": "never reuse this directory for another pool or request stream",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Create a private CA and distinct mTLS identities for all Porthos parties."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run(*command: str) -> None:
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite TLS directory: {args.output}")
    args.output.mkdir(parents=True, mode=0o700, exist_ok=True)
    ca_key, ca_cert = args.output / "ca.key", args.output / "ca.crt"
    run("openssl", "req", "-x509", "-newkey", "rsa:3072", "-nodes",
        "-keyout", str(ca_key), "-out", str(ca_cert), "-days", str(args.days),
        "-subj", "/CN=ghostmpc-porthos-private-ca")
    for party in range(3):
        key = args.output / f"p{party}.key"
        request = args.output / f"p{party}.csr"
        cert = args.output / f"p{party}.crt"
        extension = args.output / f"p{party}.ext"
        extension.write_text(
            f"subjectAltName=DNS:porthos-p{party}\n"
            "extendedKeyUsage=serverAuth,clientAuth\n")
        run("openssl", "req", "-newkey", "rsa:3072", "-nodes",
            "-keyout", str(key), "-out", str(request),
            "-subj", f"/CN=porthos-p{party}")
        run("openssl", "x509", "-req", "-in", str(request),
            "-CA", str(ca_cert), "-CAkey", str(ca_key), "-CAcreateserial",
            "-out", str(cert), "-days", str(args.days),
            "-extfile", str(extension))
        key.chmod(0o600)
    ca_key.chmod(0o600)
    (args.output / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "private_ca": True, "parties": 3,
        "identities": [f"porthos-p{party}" for party in range(3)],
        "mutual_authentication": True, "validity_days": args.days,
    }, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

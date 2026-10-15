"""The ``ghostmpc`` command line.

Each module registers one contiguous group of subcommands; the order of
:data:`_GROUPS` is the order they appear in ``ghostmpc --help``.
"""

from __future__ import annotations

import argparse

from . import accuracy, bench, calibrate, export, generate, images, setup, training

_GROUPS = (setup, generate, bench, images, accuracy, export, calibrate, training)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ghostmpc")
    commands = parser.add_subparsers(dest="command", required=True)
    for group in _GROUPS:
        group.register(commands)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover - console-script entry point
    raise SystemExit(main())

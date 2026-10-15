"""The guard that keeps the verification datasets out of every training run."""

from __future__ import annotations

from pathlib import Path


def verification_directory() -> Path:
    """The face-verification corpus, which nothing may ever train on.

    Derived from the installed package rather than from a fixed number of
    parent hops, so moving this module inside the package cannot silently
    point the guard below at a directory that does not exist.
    """
    package_root = Path(__file__).resolve().parents[1]  # .../src/ghostmpc
    return (package_root.parents[1] / "datasets" / "benchmarks").resolve()


def _reject_verification_directory(training_directory: Path) -> Path:
    training_directory = training_directory.resolve()
    benchmark_directory = verification_directory()
    if training_directory == benchmark_directory or benchmark_directory in training_directory.parents:
        raise ValueError("verification datasets cannot be used for training")
    return training_directory

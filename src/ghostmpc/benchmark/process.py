"""Process orchestration for one three-party MP-SPDZ run.

Three parties are started together and their pipes drained concurrently: a
party that fills its pipe while the harness waits on the others deadlocks the
whole group.
"""

from __future__ import annotations

import os
import secrets
import struct
import subprocess
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

from ..core.model import DEPLOYMENT_RING_BITS
from ..core.mpspdz_protocols import ProtocolSpec, get_protocol


class BenchmarkTimeout(TimeoutError):
    pass

def mpspdz_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Deployment bundles can contain a populated submodule without its
        # nested .git file. Recover the gitlink from the enclosing checkout.
        for parent in root.parents:
            try:
                relative = root.resolve().relative_to(parent.resolve())
                row = subprocess.check_output(
                    ["git", "ls-tree", "HEAD", "--", str(relative)], cwd=parent,
                    text=True, stderr=subprocess.DEVNULL).strip()
            except (ValueError, subprocess.CalledProcessError, FileNotFoundError):
                continue
            fields = row.split()
            if len(fields) >= 3 and fields[0] == "160000":
                return fields[2]
        return "unknown-vendored-tree"

#: Where a GhostFaceMPC program finds its inputs: the client's replicated
#: sharing, loaded as it is.  The naive baseline keeps MP-SPDZ's private-input
#: files, from which every server re-shares its own additive component.
INPUT_LAYOUTS = ("replicated", "input-binary")


def prepare_player_data(root: Path, input_size: int,
                        ring_bits: int = DEPLOYMENT_RING_BITS,
                        *, requests: int = 1, parties: int = 3,
                        values: Sequence[int] | None = None,
                        modulus: int | None = None,
                        gallery_size: int = 0,
                        gallery_values: Sequence[int] | None = None,
                        identities: Sequence[int] | None = None,
                        layout: str = "replicated") -> None:
    """Write the inputs a run reads, as the client and the dealer would.

    The benchmark's communication trace is input-independent. Still, using
    fresh shares (rather than public zero values) keeps its input path
    identical to the client protocol and never writes a clear image on disk.

    A scored program reads its database *before* the request loop, so the
    gallery comes first and the per-request payloads follow, each shared
    afresh.  They are shared by the same routine as the image because that is
    what the trusted dealer does -- it keeps no row in the clear.

    With the ``replicated`` layout (every GhostFaceMPC program) the client
    produces the replicated sharing itself: party ``i`` receives
    ``(x_i, x_{i-1})`` in ``Persistence/Transactions-P<i>.data``, which
    ``sint.read_from_file`` loads without any interaction.  A request that
    claims an identity writes it, public and the same for every party, to
    ``Persistence/Transactions-clear-P<i>.data``.

    The ``input-binary`` layout is the naive baseline's: one additive
    component per party in ``Player-Data/Input-Binary-P<i>-0``, which the
    servers re-share with ``input_from``.
    """
    from ..core.sharing import write_clear_persistence, write_persistence

    if layout not in INPUT_LAYOUTS:
        raise ValueError(f"unknown input layout {layout!r}")
    modulus = (1 << ring_bits) if modulus is None else modulus
    if parties < 2:
        raise ValueError("secret input requires at least two input parties")
    if layout == "replicated" and parties != 3:
        raise ValueError("the replicated sharing is the three-party layout")
    if layout == "replicated" and modulus != 1 << ring_bits:
        raise ValueError("replicated shares live in the program's ring")
    clear_values = list(values) if values is not None else [0] * input_size
    if len(clear_values) != input_size:
        raise ValueError("input value count does not match the manifest")
    gallery = (list(gallery_values) if gallery_values is not None
               else [0] * gallery_size)
    if len(gallery) != gallery_size:
        raise ValueError("gallery value count does not match the score plan")
    if identities is not None:
        if layout != "replicated":
            raise ValueError("a claimed identity is read from the clear "
                             "persistence file of the replicated layout")
        if len(identities) != requests:
            raise ValueError("one claimed identity per request")

    def additive(block: Sequence[int]) -> list[list[int]]:
        components: list[list[int]] = [[] for _ in range(parties)]
        for value in block:
            random = [secrets.randbelow(modulus) for _ in range(parties - 1)]
            for party, share in enumerate(random):
                components[party].append(share)
            final = int(value) - sum(random)
            components[-1].append(final % modulus)
        return components

    blocks = ([gallery] if gallery else []) + [clear_values] * requests
    if layout == "replicated":
        persistence = root / "Persistence"
        persistence.mkdir(parents=True, exist_ok=True)
        pairs: list[list[list[int]]] = [[[], []] for _ in range(parties)]
        for block in blocks:
            components = additive(block)
            for party in range(parties):
                pairs[party][0].extend(components[party])
                pairs[party][1].extend(components[(party - 1) % parties])
        for party in range(parties):
            write_persistence(persistence / f"Transactions-P{party}.data",
                              pairs[party], ring_bits)
            clear = persistence / f"Transactions-clear-P{party}.data"
            if identities is not None:
                write_clear_persistence(clear, identities, ring_bits)
            elif clear.exists():
                clear.unlink()
        return

    player_data = root / "Player-Data"
    player_data.mkdir(parents=True, exist_ok=True)
    shares: list[list[int]] = [[] for _ in range(parties)]
    for block in blocks:
        components = additive(block)
        for party in range(parties):
            shares[party].extend(components[party])
    for party, values in enumerate(shares):
        path = player_data / f"Input-Binary-P{party}-0"
        with path.open("wb") as handle:
            for value in values:
                if modulus == 1 << 64:
                    signed = value - (1 << 64) if value >= (1 << 63) else value
                else:
                    signed = value - modulus if value > modulus // 2 else value
                if not -(1 << 63) <= signed < (1 << 63):
                    raise OverflowError("binary input component does not fit int64")
                handle.write(struct.pack("<q", signed))

def run_commands(commands: Sequence[Sequence[str]], cwd: Path, *,
                 timeout_seconds: float | None = None,
                 resource_usage: list[dict[str, float]] | None = None
                 ) -> tuple[float, list[float], list[str], list[str]]:
    """Run a heterogeneous MPC process group with exact child CPU usage."""
    started = time.monotonic()
    processes = [subprocess.Popen(
        list(command), cwd=cwd, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    ) for command in commands]

    def collect(process: subprocess.Popen[str]) -> tuple[int, float, float, str, str]:
        # Drain both pipes concurrently with the wait: a party that prints more
        # than one pipe buffer (~64 KB) of markers would otherwise block on
        # write, never exit, and hang the whole group while os.wait4 blocks on a
        # process that is itself blocked on an undrained pipe.
        buffers: dict[str, str] = {"stdout": "", "stderr": ""}
        def drain(name: str, stream: object) -> None:
            buffers[name] = stream.read() if stream else ""
        readers = [
            threading.Thread(target=drain, args=("stdout", process.stdout)),
            threading.Thread(target=drain, args=("stderr", process.stderr)),
        ]
        for reader in readers:
            reader.start()
        # wait4 returns rusage for this exact child, avoiding both GNU time and
        # the inaccurate split of aggregate RUSAGE_CHILDREN values.
        _, status, usage = os.wait4(process.pid, 0)
        for reader in readers:
            reader.join()
        process.returncode = os.waitstatus_to_exitcode(status)
        return (process.returncode, usage.ru_utime + usage.ru_stime,
                float(usage.ru_maxrss), buffers["stdout"], buffers["stderr"])

    pool = ThreadPoolExecutor(max_workers=len(processes))
    futures = [pool.submit(collect, process) for process in processes]
    _, pending = wait(futures, timeout=timeout_seconds)
    timed_out = bool(pending)
    if timed_out:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        _, pending = wait(futures, timeout=5)
        if pending:
            for process in processes:
                if process.poll() is None:
                    process.kill()
    results = [future.result() for future in futures]
    pool.shutdown(wait=True)
    elapsed = time.monotonic() - started
    if timed_out:
        raise BenchmarkTimeout(
            f"MP-SPDZ process group exceeded {timeout_seconds:g} seconds")
    if any(result[0] for result in results):
        detail = "\n".join(f"P{i}={result[0]}\n{result[4]}" for i, result in enumerate(results))
        raise RuntimeError(f"MP-SPDZ failed:\n{detail}")
    if resource_usage is not None:
        resource_usage.extend({"cpu_seconds": result[1],
                               "max_rss_kib": result[2]}
                              for result in results)
    return elapsed, [x[1] for x in results], [x[3] for x in results], [x[4] for x in results]

def run_group(command: list[str], cwd: Path, parties: int, *,
              timeout_seconds: float | None = None,
              resource_usage: list[dict[str, float]] | None = None
              ) -> tuple[float, list[float], list[str], list[str]]:
    """Run homogeneous protocol parties concurrently."""
    def party_command(party: int) -> list[str]:
        if "{party}" in command:
            return [str(party) if item == "{party}" else item for item in command]
        return command + ["-p", str(party)]

    return run_commands(
        [party_command(party) for party in range(parties)], cwd,
        timeout_seconds=timeout_seconds, resource_usage=resource_usage)

def _pool_runtime(root: Path, results: Path, pool_id: str,
                  input_size: int, requests: int, ring_bits: int,
                  *, protocol: str = "replicated-ring-party",
                  values: Sequence[int] | None = None,
                  modulus: int | None = None,
                  gallery_size: int = 0,
                  gallery_values: Sequence[int] | None = None,
                  identities: Sequence[int] | None = None,
                  layout: str = "replicated") -> Path:
    """Create an isolated cwd for one protocol execution.

    The cwd holds the run's own ``Persistence/``, so concurrent pools never
    read each other's shares or append to each other's deliveries.
    """
    spec = get_protocol(protocol)
    runtime = results / "runtime" / pool_id
    runtime.mkdir(parents=True, exist_ok=False)
    (runtime / "Programs").symlink_to((root / "Programs").resolve(), target_is_directory=True)
    prepare_player_data(
        runtime, input_size, ring_bits, requests=requests,
        parties=spec.input_parties, values=values, modulus=modulus,
        gallery_size=gallery_size, gallery_values=gallery_values,
        identities=identities, layout=layout)
    (runtime / "Player-Data").mkdir(exist_ok=True)
    source = root / "Player-Data"
    destination = runtime / "Player-Data"
    for item in source.iterdir():
        if item.name.endswith((".pem", ".key", ".0")):
            (destination / item.name).symlink_to(item.resolve())
    return runtime

def _compile_program(mpspdz: Path, spec: ProtocolSpec, program_name: str,
                     ring_bits: int = DEPLOYMENT_RING_BITS) -> None:
    subprocess.run(
        ["./compile.py", *spec.compile_args_for(ring_bits), program_name],
        cwd=mpspdz, check=True)

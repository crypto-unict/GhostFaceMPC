#!/usr/bin/env python3
"""Run one persistent three-party Porthos pool through six mTLS tunnels.

One process per party serves every request of the pool: the first ``warmups``
requests are discarded, the remaining ``trials`` are measured.  Socket setup,
the TLS handshake, keys and the public weight stream are loaded once, before
the first request, and are reported as the pool's offline phase.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path

TIME = re.compile(r"^GHOSTMPC_PORTHOS_TIME (\{.*\})$", re.MULTILINE)
COMM = re.compile(r"^GHOSTMPC_PORTHOS_COMM (\{.*\})$", re.MULTILINE)
# The setup phase, once per process, and the barrier `StartComputation()` runs
# before `start_m()`, once per request.  Both are outside the online window;
# reporting them does not move it.
OFFLINE = re.compile(r"^GHOSTMPC_PORTHOS_OFFLINE (\{.*\})$", re.MULTILINE)
SYNC = re.compile(r"^GHOSTMPC_PORTHOS_SYNC (\{.*\})$", re.MULTILINE)


def _metrics(path: Path, pattern: re.Pattern[str]) -> list[dict]:
    return [json.loads(value) for value in pattern.findall(path.read_text())]


def _phase(party_rows: Sequence[Mapping[str, object]]) -> dict:
    """One phase of one request.

    Wall is the slowest party's, because the three run concurrently and the
    request is over when the last one is done; CPU is their sum, because that
    is what the deployment spends; bytes are the payload each party sent,
    counted once at the sender; rounds are the longest serial depth among the
    three, which is the round complexity of the phase.
    """
    wall = [float(row["wall_seconds"]) for row in party_rows]
    cpu = [float(row["cpu_seconds"]) for row in party_rows]
    sent = [int(row["bytes_sent"]) for row in party_rows]
    received = [int(row["bytes_received"]) for row in party_rows]
    rounds = [int(row["rounds"]) for row in party_rows]
    return {
        "wall_seconds": max(wall), "wall_seconds_by_party": wall,
        "cpu_seconds_total": sum(cpu), "cpu_seconds_by_party": cpu,
        "bytes_total": sum(sent), "bytes_by_party": sent,
        "bytes_received_total": sum(received),
        "bytes_reconciled": sum(sent) == sum(received),
        "rounds": max(rounds), "rounds_by_party": rounds,
        "send_calls": sum(int(row["send_calls"]) for row in party_rows),
        "recv_calls": sum(int(row["recv_calls"]) for row in party_rows),
    }


def _rss(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError):
        pass
    return 0


def _tls_material(directory: Path, party: int) -> tuple[Path, Path, Path]:
    values = (directory / "ca.crt", directory / f"p{party}.crt",
              directory / f"p{party}.key")
    missing = [str(path) for path in values if not path.is_file()]
    if missing:
        raise RuntimeError("missing TLS material: " + ", ".join(missing))
    return values


def _stop(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--party0-input", type=Path, required=True)
    parser.add_argument("--party1-input", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--tls-directory", type=Path, required=True)
    parser.add_argument("--addresses", type=Path,
                        default=HERE / "addresses.local")
    parser.add_argument("--generated-source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--connect-offset", type=int, default=10000)
    parser.add_argument("--tls-offset", type=int, default=20000)
    parser.add_argument(
        "--score", action="store_true",
        help="the binary was built with the 1:1 verification score, so it "
             "retains 514 components per request instead of 512")
    parser.add_argument(
        "--backend", choices=("porthos-nearest",), default="porthos-nearest",
        help="the truncation the binary runs: secure exact nearest rounding")
    parser.add_argument(
        "--threads-per-party", type=int, default=1,
        help="the declared compute budget. Porthos exposes no protocol-level "
             "thread count -- its MPC layer is single-threaded by "
             "construction -- so this bounds Eigen and the library thread "
             "pools, and the pool report records it under that name "
             "(default %(default)s)")
    parser.add_argument(
        "--build-manifest", type=Path,
        help="the porthos-build-manifest.json this binary was built from. "
             "Rounds are published only when it records that the round "
             "counter build gate passed: a build whose gate did not run must "
             "not claim a round count")
    args = parser.parse_args()
    if args.warmups < 0 or args.trials < 1:
        raise SystemExit("warmups must be non-negative and trials positive")
    if any(offset <= 0 or 32008 + offset > 65535
           for offset in (args.connect_offset, args.tls_offset)):
        raise SystemExit("connect/tls offsets must map Porthos ports into 1..65535")
    if args.connect_offset == args.tls_offset:
        raise SystemExit("connect and TLS port offsets must differ")
    if not args.binary.is_file():
        raise SystemExit(f"missing binary: {args.binary}")
    round_counter_validated = False
    if args.build_manifest:
        if not args.build_manifest.is_file():
            raise SystemExit(f"missing build manifest: {args.build_manifest}")
        gate = json.loads(args.build_manifest.read_text()).get(
            "gates", {}).get("round_counter", {})
        round_counter_validated = gate.get("status") == "passed"
    requests = args.warmups + args.trials
    root = Path(tempfile.mkdtemp(prefix=f"ghostmpc-porthos-pool-{args.pool}-"))
    logs = root / "logs"
    logs.mkdir()
    tunnel = HERE / "mtls_tunnel.py"
    processes: list[subprocess.Popen] = []
    party_processes: list[subprocess.Popen] = []
    peak_rss = [0, 0, 0]
    started = time.perf_counter()
    common_environment = dict(os.environ)
    threads = str(args.threads_per_party)
    common_environment.update(
        OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads,
        MKL_NUM_THREADS=threads, NUMEXPR_NUM_THREADS=threads,
        EIGEN_NUM_THREADS=threads,
        GHOSTMPC_REQUESTS=str(requests),
        GHOSTMPC_CONNECT_PORT_OFFSET=str(args.connect_offset),
        GHOSTMPC_PORTHOS_NEAREST="1")
    with ExitStack() as stack:
        def launch(command: list[str], name: str) -> subprocess.Popen:
            log = stack.enter_context((logs / name).open("wb"))
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            processes.append(process)
            stack.callback(_stop, process)
            return process

        # Direction p->q uses index p*3+q and upstream receiver port 32000+index.
        for sender in range(3):
            ca, client_cert, client_key = _tls_material(args.tls_directory, sender)
            for receiver in range(3):
                if sender == receiver:
                    continue
                _, server_cert, server_key = _tls_material(args.tls_directory, receiver)
                index = sender * 3 + receiver
                plain_receiver = 32000 + index
                local_client = plain_receiver + args.connect_offset
                tls_port = plain_receiver + args.tls_offset
                launch([
                    sys.executable, str(tunnel), "server",
                    "--listen", f"127.0.0.1:{tls_port}",
                    "--connect", f"127.0.0.1:{plain_receiver}",
                    "--ca", str(ca), "--cert", str(server_cert),
                    "--key", str(server_key), "--connections", "1",
                    "--relay-direction", "right-to-left",
                    "--connect-timeout", "120",
                ], f"tls-server-{sender}-{receiver}.log")
                launch([
                    sys.executable, str(tunnel), "client",
                    "--listen", f"127.0.0.1:{local_client}",
                    "--connect", f"127.0.0.1:{tls_port}",
                    "--ca", str(ca), "--cert", str(client_cert),
                    "--key", str(client_key), "--server-name",
                    f"porthos-p{receiver}", "--connections", "1",
                    "--relay-direction", "left-to-right",
                    "--connect-timeout", "120",
                ], f"tls-client-{sender}-{receiver}.log")
        time.sleep(0.5)
        for party in range(3):
            stdout = stack.enter_context((logs / f"party-{party}.log").open("wb"))
            share_path = root / f"party-{party}.shares"
            environment = dict(common_environment)
            environment["GHOSTMPC_OUTPUT_SHARE"] = str(share_path)
            stdin_path = (args.party0_input if party == 0 else
                          args.party1_input if party == 1 else Path(os.devnull))
            stdin = stack.enter_context(stdin_path.open("rb"))
            process = subprocess.Popen([
                str(args.binary.resolve()), str(party), str(args.addresses),
                str(args.keys)], stdin=stdin, stdout=stdout,
                stderr=subprocess.STDOUT, env=environment)
            party_processes.append(process)
            stack.callback(_stop, process)
        completed_requests = 0
        next_progress_poll = time.monotonic()
        while any(process.poll() is None for process in party_processes):
            for party, process in enumerate(party_processes):
                peak_rss[party] = max(peak_rss[party], _rss(process.pid))
            if time.monotonic() >= next_progress_poll:
                next_progress_poll = time.monotonic() + 1
                observed = min(
                    len(_metrics(logs / f"party-{party}.log", TIME))
                    for party in range(3))
                if observed > completed_requests:
                    completed_requests = observed
                    print(
                        f"Porthos pool {args.pool}: "
                        f"{completed_requests}/{requests} requests completed",
                        flush=True)
            time.sleep(0.01)
        for process in party_processes:
            if process.returncode:
                raise RuntimeError(
                    f"Porthos party failed with {process.returncode}; logs in {logs}")
        pool_wall = time.perf_counter() - started
        # A scored program writes the two score components through the same
        # gate as the embedding, so it retains 514 components per request
        # rather than 512.  The count stays exact: it is the check that the run
        # produced what it was supposed to.
        components = 512 + (2 if args.score else 0)
        for party in (0, 1):
            share_path = root / f"party-{party}.shares"
            if (not share_path.is_file()
                    or len(share_path.read_text().splitlines())
                    != requests * components):
                raise RuntimeError(
                    f"party {party} did not retain "
                    f"{requests}x{components} output shares")
        if (root / "party-2.shares").exists():
            raise RuntimeError("helper must not write an output share")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    retained_shares = []
    for party in (0, 1):
        retained = args.output.parent / f"{args.output.stem}-party-{party}.shares"
        shutil.copy2(root / f"party-{party}.shares", retained)
        retained_shares.append(str(retained))

    times = [_metrics(logs / f"party-{party}.log", TIME) for party in range(3)]
    comms = [_metrics(logs / f"party-{party}.log", COMM) for party in range(3)]
    syncs = [_metrics(logs / f"party-{party}.log", SYNC) for party in range(3)]
    offlines = [_metrics(logs / f"party-{party}.log", OFFLINE) for party in range(3)]
    if any(len(value) != requests for value in [*times, *comms, *syncs]):
        raise RuntimeError(f"incomplete per-request metrics; logs in {logs}")
    if any(len(value) != 1 for value in offlines):
        raise RuntimeError(f"expected one offline phase per party; logs in {logs}")
    offline = _phase([offlines[party][0] for party in range(3)])
    rows = []
    for request in range(requests):
        party_rows = []
        for party in range(3):
            value = {**times[party][request], **comms[party][request],
                     "peak_rss_bytes": peak_rss[party]}
            party_rows.append(value)
        online = _phase(party_rows)
        barrier = _phase([syncs[party][request] for party in range(3)])
        rows.append({
            "pool": args.pool, "request": request,
            "warmup": request < args.warmups,
            "wall_seconds": online["wall_seconds"],
            "cpu_seconds_total": online["cpu_seconds_total"],
            "peak_rss_bytes": max(peak_rss),
            "bytes_total": online["bytes_total"],
            "bytes_reconciled": online["bytes_reconciled"],
            "rounds": online["rounds"],
            "rounds_by_party": online["rounds_by_party"],
            "causal_rounds_validated": round_counter_validated,
            "send_calls": online["send_calls"],
            "recv_calls": online["recv_calls"],
            # The barrier is a per-request cost that `start_m()` has always
            # excluded.  It stays excluded -- the online row above is unchanged
            # -- and is now reported instead of being invisible.
            "phases": {"online": online, "sync_barrier": barrier},
            "parties": party_rows,
            "cpu_affinity": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity") else None,
        })
    args.output.write_text(json.dumps({
        "backend": "CrypTFlow/Porthos-public-nearest",
        "truncation": "secure exact nearest (rounding half up)",
        "samples": rows,
        "pool_wall_seconds": pool_wall,
        "threads_per_party": args.threads_per_party,
        "thread_knob": "EIGEN_NUM_THREADS/OMP_NUM_THREADS; Porthos has no "
                       "protocol-level thread count",
        "score": bool(args.score),
        "transport": "mTLS full mesh (six directional tunnels)",
        "binary_bytes": args.binary.stat().st_size,
        "generated_source_bytes": (args.generated_source.stat().st_size
                                   if args.generated_source else None),
        "offline_output_shares": retained_shares,
        # Porthos exchanges no correlated randomness: its "offline module"
        # populates local mod-prime tables and its PRG keys are pre-shared
        # files, so the only offline traffic is the connection setup and the
        # one setup barrier.  The phase runs once per process, not per request.
        "offline": {**offline, "scope": "once per pool process", "items": [
            "session keys", "mod-prime tables", "socket setup",
            "setup barrier", "public weight stream",
            *(["enrolled gallery row"] if args.score else []),
        ]},
        "rounds": {
            "method": "runtime serial depth, max over parties",
            "validated": round_counter_validated,
            "gate": "round_counter" if round_counter_validated else None,
            "build_manifest": (str(args.build_manifest)
                               if args.build_manifest else None),
            "send_recv_calls_are_rounds": False,
        },
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

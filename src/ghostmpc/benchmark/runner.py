"""The end-to-end benchmark entry point.

One compiled program bundles the warm-up and the measured requests, so the
warm-ups differ from the measured requests in nothing but position.
"""

from __future__ import annotations

import hashlib
import shutil
import statistics
from collections.abc import Sequence
from pathlib import Path

from ..codegen.fusion import DEPLOYED, Optimizations
from ..core.identify import IdentifyPlan, identify_plan_report
from ..core.match import MatchPlan, match_plan_report
from ..core.model import DEPLOYMENT_RING_BITS, Manifest
from ..core.mpspdz_protocols import ProtocolSpec, get_protocol
from ..core.pools import PoolLedger
from ..core.score import ScorePlan
from .network import NetworkContext, link_counters, measure_rtt, network_report
from .process import _compile_program, _pool_runtime, mpspdz_commit, run_commands
from .report import write_report
from .stats import Trial, _size, parse_group_stats, parse_request_timer


def _protocol_commands(
        mpspdz: Path, spec: ProtocolSpec, program_name: str,
        port_base: int,
        ring_bits: int = DEPLOYMENT_RING_BITS,
        network: NetworkContext | None = None) -> list[list[str]]:
    runtime_args = spec.runtime_args_for(ring_bits)

    def coordination(port: int) -> list[str]:
        # On one host the parties find each other through party 0 on localhost.
        # On the emulated network each party owns an address, so hand MP-SPDZ
        # the hosts file instead: `Names::init` reads one address per line and
        # gives party i the port `port + i`.
        if network is None:
            return ["-pn", str(port), "-h", "localhost"]
        return ["-pn", str(port), "-ip", str(network.hosts_file)]

    def launch(party: int) -> list[str]:
        return network.party_prefix(party) if network is not None else []

    common = [program_name, *coordination(port_base), *runtime_args, "-v"]
    return [
        [*launch(party), str(mpspdz / spec.binary), str(party), *common]
        for party in range(spec.parties)
    ]

def _run_protocol_group(
        mpspdz: Path, spec: ProtocolSpec, program_name: str,
        runtime: Path, port_base: int, timeout_seconds: float | None,
        ring_bits: int = DEPLOYMENT_RING_BITS,
        network: NetworkContext | None = None,
) -> tuple[float, list[float], list[str], list[str]]:
    commands = _protocol_commands(
        mpspdz, spec, program_name, port_base, ring_bits, network)
    return run_commands(commands, runtime, timeout_seconds=timeout_seconds)

def _timer(stderr: list[str], timer_id: int) -> dict[str, object]:
    return parse_request_timer(stderr, timer_id=timer_id, require_phases=True)

def _trials_from_timer(
        label: str, timer: dict[str, object], span: int, *,
        spec: ProtocolSpec, pool_id: str,
        phases: Sequence[str] = ("total", "online", "offline"),
) -> list[Trial]:
    result: list[Trial] = []
    for phase in phases:
        stats = timer["phases"][phase]
        cpu = tuple(float(value) / span for value in stats["cpu_seconds_by_party"])
        sent = tuple(round(int(value) / span) for value in stats["bytes_by_party"])
        rounds = stats["rounds"]
        result.append(Trial(
            label, phase,
            wall_seconds=float(stats["wall_seconds"]) / span,
            cpu_seconds_p0=cpu[0] if cpu else 0.0,
            cpu_seconds_p1=cpu[1] if len(cpu) > 1 else 0.0,
            bytes_p0_to_p1=sent[0] if sent else 0,
            bytes_p1_to_p0=sent[1] if len(sent) > 1 else 0,
            rounds=(None if rounds is None else round(int(rounds) / span)),
            protocol=spec.name, pool_id=pool_id, requests=span,
            cpu_seconds_by_party=cpu, bytes_by_party=sent,
        ))
    return result

# The offline phase of the measured requests -- the daBits and, under active
# security, the binary triples generated before each of them -- reported beside
# the online one and never summed into it.  A request that refilled its buffers
# online is refused: the prefetch fell short and the online figure would carry
# preprocessing.
def _offline_trials(region, offline_region, divisor, *, spec: ProtocolSpec,
                    pool_id: str) -> list[Trial]:
    offline_leak = region["phases"]["offline"]
    if int(offline_leak["bytes_total"]) or int(offline_leak["rounds"] or 0):
        raise RuntimeError(f"the request refilled its preprocessing online: {offline_leak}")
    return _trials_from_timer("offline_dabits", offline_region, divisor,
                              spec=spec, pool_id=pool_id)


def _compilation_stamp(source: Path) -> Path:
    return source.with_suffix(".compiled.sha256")


def _record_compilation(source: Path) -> None:
    """Stamp a compiled source with its digest, for the guard below."""
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    _compilation_stamp(source).write_text(digest + "\n")


def _guard_compilation(source: Path, *, skip_compile: bool) -> None:
    """Refuse to reuse a compilation of a *different* program.

    ``--skip-compile`` is sound only when the emitted source is byte-identical
    to the one the bytecode was built from.  Nothing about a violated
    assumption is visible at run time -- the run succeeds and reports numbers
    for the wrong program -- so it is checked here and refused loudly.
    """
    if not skip_compile:
        return
    stamp = _compilation_stamp(source)
    if not stamp.is_file():
        raise FileNotFoundError(
            f"--skip-compile: no record that {source.name} was ever compiled "
            f"by this harness; compile once without the flag first")
    recorded = stamp.read_text().strip()
    current = hashlib.sha256(source.read_bytes()).hexdigest()
    if recorded != current:
        raise ValueError(
            f"--skip-compile: {source.name} differs from the source that was "
            f"compiled ({current[:12]} against {recorded[:12]}). The flag is "
            f"only valid when the emitted program is unchanged; a parameter "
            f"that enters the program -- threads, requests, warmups, the "
            f"verification head -- invalidates it.")


def _score_report(score: ScorePlan | None,
                  spec: ProtocolSpec) -> dict[str, object] | None:
    """What the report records about a compiled-in score.

    ``fused_dot_product`` is recorded because it is the one protocol property
    the score's cost turns on: with it a 512-term inner product re-shares one
    ring element, without it one per term.
    """
    if score is None:
        return None
    return {
        "gallery_rows": score.gallery_rows,
        "embedding_size": score.embedding_size,
        "fractional_bits": score.fractional_bits,
        "gallery_fractional_bits": score.gallery_fractional_bits,
        "declared_widths": score.declared_widths(),
        "norm_bound": score.norm_bound,
        "fused_dot_product": spec.fused_dot_product,
        "opened": "nothing; the servers return share components of "
                  "(<e,g>, <e,e>) and the client divides in the clear",
    }


def _match_report(match: MatchPlan | None) -> dict[str, object] | None:
    """The match rule the program compiled, widths and constant."""
    return None if match is None else match_plan_report(match)


def _identify_report(identify: IdentifyPlan | None) -> dict[str, object] | None:
    """The identification the program compiled, shape and widths."""
    return None if identify is None else identify_plan_report(identify)


def _end_to_end_medians(trials: Sequence[Trial]) -> tuple[int | None, float | None]:
    """Median per-inference payload bytes and wall over the measured pools."""
    rows = [t for t in trials if t.layer == "end_to_end" and t.phase == "total"]
    if not rows:
        return None, None
    return (round(statistics.median(t.bytes_total for t in rows)),
            statistics.median(t.wall_seconds for t in rows))


def _optimization_report(manifest_path: Path, manifest: Manifest,
                         optimizations: Optimizations,
                         spec: ProtocolSpec) -> dict[str, object]:
    """The switches a program was emitted with and what they selected."""
    from ..codegen.fusion import (
        _accumulator_bounds,
        _prelu_alpha_magnitudes,
        manifest_bounds,
        plan_fusions,
    )

    import numpy as np

    quantization = manifest.quantization
    fractional_bits = quantization.fractional_bits
    kernel_bits = quantization.weight_fractional_bits
    kernel_bits = fractional_bits if kernel_bits is None else int(kernel_bits)
    weights_path = manifest_path.parent / manifest.weights_file
    weights = np.load(weights_path, mmap_mode="r")
    bounds = manifest_bounds(manifest) or _accumulator_bounds(
        manifest, weights, fractional_bits=fractional_bits,
        total_bits=quantization.activation_bits)
    fusion, prelu, multiply = plan_fusions(
        manifest, bounds, fractional_bits=fractional_bits,
        ring_bits=quantization.ring_bits, kernel_bits=kernel_bits,
        alpha_magnitudes=_prelu_alpha_magnitudes(manifest, weights),
        optimizations=optimizations)
    return {
        "disabled": optimizations.disabled(),
        "ring_bits": quantization.ring_bits,
        "fractional_bits": fractional_bits,
        "kernel_fractional_bits": kernel_bits,
        "operations": len(manifest.ops),
        "batch_normalizations": sum(op.kind == "BatchNormalization"
                                    for op in manifest.ops),
        "fused_prelu": len(prelu),
        "fused_multiply": len(multiply),
        "fused_add": len(fusion.adds),
        "two_summand_rounding": (spec.two_summand_split
                                 and optimizations.two_summand),
    }


def run_benchmarks(manifest_path: Path, mpspdz: Path, results: Path, *, trials: int = 30,
                   warmups: int = 5, threads: int = 1, pools: int = 3,
                   port_base: int = 5000,
                   protocol: str = "replicated-ring-party",
                   timeout_seconds: float | None = None,
                   input_values: Sequence[int] | None = None,
                   input_modulus: int | None = None,
                   skip_compile: bool = False,
                   score: ScorePlan | None = None,
                   match: MatchPlan | None = None,
                   identify: IdentifyPlan | None = None,
                   network: NetworkContext | None = None,
                   optimizations: Optimizations = DEPLOYED,
                   verified_io: bool | None = None) -> tuple[Path, Path]:
    """Measure isolated executions of the deployed program under one protocol.

    ``optimizations`` turns circuit optimizations off for the ablation; the
    default is the deployed program.
    """
    from ..codegen import (
        EMBEDDING_TIMER_ID,
        GALLERY_LOAD_TIMER_ID,
        IDENTIFY_TIMER_ID,
        INPUT_SHARE_TIMER_ID,
        MATCH_TIMER_ID,
        MEASURED_TIMER_ID,
        OFFLINE_DABIT_TIMER_ID,
        SCORE_TIMER_ID,
        WEIGHT_LOAD_TIMER_ID,
        generate,
        verified_io_default,
    )

    spec = get_protocol(protocol)
    # The verified input/output is the malicious default; only a departure
    # from it earns a tag.
    verified_io = verified_io_default(spec.name, verified_io)
    manifest = Manifest.load(manifest_path)
    ring_bits = manifest.quantization.ring_bits
    # The same composition ``_protocol_commands`` builds the argv from, recorded
    # so a report states the configuration it was measured under.
    effective_runtime_args = spec.runtime_args_for(ring_bits)
    requests = warmups + trials
    mpspdz = mpspdz.resolve()
    source_dir = mpspdz / "Programs" / "Source"
    source_dir.mkdir(parents=True, exist_ok=True)
    protocol_tag = spec.name.removesuffix("-party").replace("-", "_")
    # A scored program is a different tape, so it earns its own name: reusing
    # the plain program's compilation for it would measure the wrong program.
    score_tag = "" if score is None else f"_score{score.gallery_rows}"
    # A match decision changes the tape as much as the score does, and its
    # threshold enters the emitted constant, so the name carries it too.
    if match is not None:
        score_tag += f"_match{match.threshold_constant}"
    # An identification is a different tape again, and its cost depends on how
    # many rows a request ranks, so the name carries that.
    if identify is not None:
        score_tag += f"_identify{identify.candidates}"
    # An ablated program is a different tape, and so is one on another ring.
    ablation_tag = ("" if ring_bits == DEPLOYMENT_RING_BITS
                    else f"_ring{ring_bits}")
    ablation_tag += "".join(f"_no_{name.replace('-', '_')}"
                            for name in optimizations.disabled())
    if verified_io != spec.malicious:
        ablation_tag += "_verified_io" if verified_io else "_unverified_io"
    program_name = f"ghostmpc_end_to_end_{protocol_tag}{score_tag}{ablation_tag}"
    generate(
        manifest_path, source_dir / f"{program_name}.mpc",
        threads=threads, requests=requests, warmups=warmups,
        measured_timer=True, score=score, match=match, identify=identify,
        protocol=spec.name, optimizations=optimizations,
        verified_io=verified_io,
    )
    optimization_report = _optimization_report(
        manifest_path, manifest, optimizations, spec)
    # The source is regenerated either way, so reusing a compilation is sound
    # exactly when the emitted program did not change.  The condition is
    # *checked* rather than trusted, because violating it does not fail -- it
    # silently measures the previously compiled program and normalizes by the
    # caller's parameters (``--trials`` enters ``REQUESTS``).
    _guard_compilation(source_dir / f"{program_name}.mpc",
                       skip_compile=skip_compile)
    if not skip_compile:
        _compile_program(mpspdz, spec, program_name, ring_bits)
        _record_compilation(source_dir / f"{program_name}.mpc")
    if not (mpspdz / spec.binary).is_file():
        raise FileNotFoundError(f"build MP-SPDZ binaries first: {spec.binary}")
    missing_tls = [mpspdz / "Player-Data" / f"P{i}.pem" for i in range(spec.parties)]
    if any(not path.is_file() for path in missing_tls):
        raise FileNotFoundError(
            "honest-majority protocols require TLS material; "
            "run `Scripts/setup-ssl.sh 3 Player-Data`")
    # A verified 1:1 request reads the identity it claims; the benchmark claims
    # the same rows the unverified program reads off its request counter.
    claimed_identities = (
        [identity % score.gallery_rows for identity in range(requests)]
        if verified_io and score is not None and identify is None else None)
    measured: list[Trial] = []
    ledger = PoolLedger(results / "pools")
    # The emulated link is measured before any protocol traffic, so the
    # reported round-trip time describes the link the pools then ran over.
    observed_rtt = measure_rtt(network) if network is not None else None
    # The unshaped control carries no netem qdisc and so no link statistics.
    shaped = network is not None and network.profile is not None
    wire_bytes: list[int] = [0] * spec.parties
    wire_drops: list[int] = [0] * spec.parties
    group_payload_bytes = 0
    for _ in range(pools):
        pool = ledger.create(spec.name, program_name, requests)
        runtime: Path | None = None
        try:
            runtime = _pool_runtime(
                mpspdz, results, pool.pool_id,
                _size(manifest.input_shape), requests, ring_bits,
                protocol=spec.name, values=input_values, modulus=input_modulus,
                gallery_size=0 if score is None else score.gallery_size,
                identities=claimed_identities)
            before = link_counters(network) if shaped else None
            _, _, _, stderr = _run_protocol_group(
                mpspdz, spec, program_name, runtime, port_base, timeout_seconds,
                ring_bits, network)
            if before is not None:
                # Difference the qdisc counters across this pool alone, so the
                # link statistics of one execution never absorb another's, nor
                # the ping traffic that measured the round-trip time.
                after = link_counters(network) or before
                for party in range(min(len(before), len(wire_bytes))):
                    wire_bytes[party] += after[party]["bytes"] - before[party]["bytes"]
                    wire_drops[party] += after[party]["drops"] - before[party]["drops"]
                # The framing ratio is only meaningful against the payload of
                # the very same span: the whole process group, warm-ups
                # included, not the measured timer.
                group_payload, _ = parse_group_stats(stderr, required=False)
                group_payload_bytes += sum(group_payload)
            # The measured timer brackets only the post-warm-up requests, so its
            # wall/comm/rounds/CPU exclude the WARMUPS cold ones. Normalise by the
            # trial count, not warmups+trials, so the warm-ups are discarded
            # rather than averaged in.
            region = _timer(stderr, MEASURED_TIMER_ID)
            divisor = max(1, trials)
            ledger.consume(pool.pool_id)
            measured.extend(_trials_from_timer(
                "end_to_end", region, divisor, spec=spec, pool_id=pool.pool_id))
            measured.extend(_offline_trials(
                region, _timer(stderr, OFFLINE_DABIT_TIMER_ID), divisor,
                spec=spec, pool_id=pool.pool_id))
            # Separate the once-per-process weight ingestion (span 1) from the
            # per-trial input sharing so neither hides inside the aggregate.
            measured.extend(_trials_from_timer(
                "input_sharing", _timer(stderr, INPUT_SHARE_TIMER_ID),
                divisor, spec=spec, pool_id=pool.pool_id))
            if score is not None:
                # The operations of one request, each read from its own timer
                # rather than derived from the aggregate, so that they
                # reconstruct ``end_to_end`` instead of standing in for it.  An
                # identification replaces the score.
                for label, timer_id in (("embedding", EMBEDDING_TIMER_ID),
                                        ("identify", IDENTIFY_TIMER_ID)
                                        if identify is not None else
                                        ("score", SCORE_TIMER_ID)) + (
                        (("match", MATCH_TIMER_ID),)
                        if match is not None else ()):
                    measured.extend(_trials_from_timer(
                        label, _timer(stderr, timer_id),
                        divisor, spec=spec, pool_id=pool.pool_id))
            measured.extend(_trials_from_timer(
                "weight_load", _timer(stderr, WEIGHT_LOAD_TIMER_ID), 1,
                spec=spec, pool_id=pool.pool_id, phases=("total",)))
            measured[-1].phase = "setup"
            if score is not None:
                # The database is distributed once, before any request exists,
                # so it is reported beside the public-weight load and never
                # inside an inference.
                measured.extend(_trials_from_timer(
                    "gallery_load", _timer(stderr, GALLERY_LOAD_TIMER_ID), 1,
                    spec=spec, pool_id=pool.pool_id, phases=("total",)))
                measured[-1].phase = "setup"
        except Exception as exc:
            ledger.invalidate(pool.pool_id, str(exc))
            if measured:
                write_report(
                    measured, manifest, results, mpspdz_commit(mpspdz),
                    threads=threads,
                    protocol_runtime_args=effective_runtime_args,
                    network=network_report(network, measured_rtt=observed_rtt),
                    optimizations=optimization_report)
            raise
        finally:
            if runtime is not None:
                shutil.rmtree(runtime, ignore_errors=True)
    payload_per_inference, wall_per_inference = _end_to_end_medians(measured)
    return write_report(
        measured, manifest, results, mpspdz_commit(mpspdz),
        threads=threads, score=_score_report(score, spec),
        protocol_runtime_args=effective_runtime_args,
        optimizations=optimization_report,
        match=_match_report(match), identify=_identify_report(identify),
        network=network_report(
            network, measured_rtt=observed_rtt,
            wire_bytes_by_party=wire_bytes if shaped else None,
            drops_by_party=wire_drops if shaped else None,
            payload_bytes=group_payload_bytes or None,
            per_inference_payload_bytes=payload_per_inference,
            per_inference_wall_seconds=wall_per_inference))

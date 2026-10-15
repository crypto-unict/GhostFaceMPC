"""Assembly of the complete MP-SPDZ program.

``render`` walks the manifest once, asks :mod:`~ghostmpc.codegen.fusion` which
truncations may be deferred, and concatenates the per-layer fragments around
the header from :mod:`~ghostmpc.codegen.templates`.
"""

from __future__ import annotations

from pathlib import Path

from ..core.identify import IdentifyPlan
from ..core.match import MatchPlan
from ..core.model import Manifest
from ..core.mpspdz_protocols import get_protocol
from ..core.score import ScorePlan
from . import score as score_codegen
from .fusion import (
    DEPLOYED,
    Optimizations,
    _accumulator_bounds,
    _add_truncand_bits,
    _prelu_alpha_magnitudes,
    _raw_scale_units,
    manifest_bounds,
    plan_fusions,
)
from .layers.dispatch import _layer_code
from .layers.pooling import _pooled_bound
from .naming import (
    EMBEDDING_TIMER_ID,
    INPUT_SHARE_TIMER_ID,
    MEASURED_TIMER_ID,
    OFFLINE_DABIT_TIMER_ID,
    WEIGHT_LOAD_TIMER_ID,
    _add_fused_bias_name,
    _ident,
    _offset,
    _size,
)
from .templates import (
    VERIFIED_IO_HELPERS,
    ARITHMETIC_TYPE_IMPORTS,
    BOUNDED_TRUNCATION_HELPERS,
    GENERIC_TRUNCATION_HELPER,
    HEADER,
    TEXT_WEIGHT_LOADER,
    TWO_SUMMAND_SIGN_HELPER,
    TWO_SUMMAND_TRUNCATION_HELPERS,
)
from .weight_stream import (
    _default_offsets,
    _prepare_weights,
    _public_input_path,
    _write_public_weights,
)

#: The program targets the deployed 32-bit ring: the bounded rounding kernels
#: and every declared width are proved for it.
RING_BITS = 32
#: The rings a program may be emitted for.  The 64-bit ring exists for the
#: ablation only: it takes the three-summand kernels, whose widths are generic.
SUPPORTED_RING_BITS = (32, 64)


def verified_io_default(protocol: str, verified_io: bool | None = None) -> bool:
    """Whether a program checks its inputs and verifies its delivery.

    It is the default exactly where the security claim needs it: under the
    actively secure protocol a component returned by one server alone could
    be flipped.  The post-sacrifice check of every pending product therefore
    runs before anything leaves, every server returns the pair it holds of
    the answer, and a 1:1 request reads the identity it claims.  The input
    needs no check in either profile: the client's replicated sharing is
    loaded as it is, and the honest pair fixes its value.
    """
    spec = get_protocol(protocol)
    return spec.malicious if verified_io is None else bool(verified_io)


def render(manifest: Manifest, fractional_bits: int = 6, *,
           threads: int = 1, requests: int = 1,
           total_bits: int = 15,
           kernel_bits: int | None = None,
           weight_offsets: dict[str, int] | None = None,
           weight_count: int | None = None,
           test_reveal: bool = False,
           warmups: int = 0, measured_timer: bool = False,
           overflow_bounds: dict[str, float] | None = None,
           alpha_magnitudes: dict[str, float] | None = None,
           score: ScorePlan | None = None,
           match: MatchPlan | None = None,
           identify: IdentifyPlan | None = None,
           protocol: str = "replicated-ring-party",
           ring_bits: int = RING_BITS,
           optimizations: Optimizations = DEPLOYED,
           verified_io: bool | None = None) -> str:
    protocol_spec = get_protocol(protocol)
    verified_io = verified_io_default(protocol, verified_io)
    if ring_bits not in SUPPORTED_RING_BITS:
        raise ValueError(f"unsupported ring of {ring_bits} bits")
    # The rounding kernels split only the declared low bits of an accumulator
    # (bounded truncation), over the two-summand regrouping of the replicated
    # sharing where the protocol allows it.  The regrouping hands A = x_0 + x_2
    # to P0 alone, so only the semi-honest protocol takes it; the malicious one
    # keeps the three-summand split, which survives an active adversary.
    two_summand = (protocol_spec.two_summand_split
                   and optimizations.two_summand)
    if two_summand and ring_bits != RING_BITS:
        raise ValueError(
            f"the two-summand rounding kernel is proved for the {RING_BITS}-bit "
            f"ring only; disable it to emit a {ring_bits}-bit program")
    if fractional_bits >= total_bits:
        raise ValueError("fractional precision must be smaller than total fixed-point bits")
    if total_bits + fractional_bits >= ring_bits:
        raise ValueError(
            "total plus fractional precision must leave one ring bit for "
            "nearest rounding")
    kernel_bits = fractional_bits if kernel_bits is None else kernel_bits
    fusion, prelu_fusion, multiply_fusion = plan_fusions(
        manifest, overflow_bounds, fractional_bits=fractional_bits,
        ring_bits=ring_bits, kernel_bits=kernel_bits,
        alpha_magnitudes=alpha_magnitudes, optimizations=optimizations)
    ops_by_name = {op.name: op for op in manifest.ops}
    default_offsets, default_lengths = _default_offsets(
        manifest, fusion, prelu_fusion)
    offsets = weight_offsets or default_offsets
    if weight_count is None:
        lengths = default_lengths if weight_offsets is None else {
            tensor.name: tensor.length for tensor in manifest.tensors
        }
        weight_count = max(
            (offset + lengths.get(name, 0) for name, offset in offsets.items()),
            default=0,
        )
    # Every bit injection is MP-SPDZ's daBit conversion (one binary opening per
    # bit), and the daBits a request consumes are generated before it starts:
    # the prefetch instruction at the top of every request fills the buffer the
    # request then drains, so its online phase never refills it.  The directive
    # is explicit because ``-E ring`` turns ``unsplit`` on by default.
    bit_injection_directive = "\nprogram.use_unsplit = 0"
    prefetch = protocol_spec.prefetch_instruction
    # The score is two secret-by-secret inner products.  It has to agree with
    # the profile the activations are on, or the client's single division would
    # undo the wrong scale.
    if score is not None:
        if (score.fractional_bits != fractional_bits
                or score.ring_bits != ring_bits):
            raise ValueError(
                "the score plan states a different fixed-point profile "
                f"(f = {score.fractional_bits}, ring = {score.ring_bits}) than "
                f"the program (f = {fractional_bits}, ring = {ring_bits})")
        embedding_size = _size(manifest.output_shape)
        if score.embedding_size != embedding_size:
            raise ValueError(
                f"the score plan expects a {score.embedding_size}-component "
                f"embedding and the manifest produces {embedding_size}")
        score.validate()
    # The match decision reads the score's two accumulators, so it cannot exist
    # without them, and its widths are proved against the same plan.
    if match is not None:
        if score is None:
            raise ValueError(
                "the match decision reads the score accumulators: compile a "
                "score as well")
        if match.score != score:
            raise ValueError(
                "the match plan states a different score plan than the program")
        match.validate()
    # A closed-set identification answers the question the match does not --
    # *who is this?* -- so it replaces the 1:1 score rather than reading it.
    if identify is not None:
        if score is None:
            raise ValueError(
                "the identification ranks the enrolled database: compile a "
                "score as well, which is what ingests it")
        if match is not None:
            raise ValueError(
                "the match decision answers a declared identity and the "
                "identification answers no declared identity: compile one")
        if identify.score != score:
            raise ValueError(
                "the identify plan states a different score plan than the "
                "program")
        identify.validate()
    lines = [HEADER.format(
        type_imports=ARITHMETIC_TYPE_IMPORTS,
        verified_io_helpers=VERIFIED_IO_HELPERS if verified_io else "",
        fractional_bits=fractional_bits, total_bits=total_bits,
        ring_bits=ring_bits,
        round_nearest="True", input_size=_size(manifest.input_shape),
        threads=max(1, threads), requests=max(1, requests),
        warmups=max(0, min(warmups, max(1, requests) - 1)) if measured_timer else 0,
        weight_count=weight_count, bit_injection_directive=bit_injection_directive,
        protocol_directives=protocol_spec.source_directives,
        score_setup=("" if score is None else score_codegen.setup(
            score, measured_timer=measured_timer,
            match=match,
            identify=identify, verified_io=verified_io)),
        weight_load_open=f"start_timer({WEIGHT_LOAD_TIMER_ID})\n" if measured_timer else "",
        weight_load_close=f"stop_timer({WEIGHT_LOAD_TIMER_ID})\n" if measured_timer else "",
        weight_loader=TEXT_WEIGHT_LOADER,
        instruction_imports=f"from Compiler.instructions import {prefetch}",
        truncation_helpers=(
            GENERIC_TRUNCATION_HELPER + BOUNDED_TRUNCATION_HELPERS
            + (TWO_SUMMAND_TRUNCATION_HELPERS + TWO_SUMMAND_SIGN_HELPER
               if two_summand else "")),
    )]
    prelu_producers = {fused.producer for fused in prelu_fusion.values()}
    raw_multiplies = {fused.multiply for fused in multiply_fusion.values()}
    fused_convs = {fused.conv: fused for fused in multiply_fusion.values()}
    # A fused SE conv lifts its whole raw add chain to ``2^{3f}``; the emitter
    # tags each raw operand with its exponent so the Concatenate and the Add can
    # align them, and reads the chain bound to size the wider truncation.
    scale_bounds = (manifest_bounds(manifest)
                    if overflow_bounds is None else overflow_bounds)
    selected = list(manifest.ops)
    first_inputs = manifest.ops[0].inputs if manifest.ops else []
    input_size = _size(manifest.input_shape)

    # Model aliases as names pointing at one physical compiler allocation.  A
    # reshape/flatten therefore extends its producer's lifetime without ever
    # owning (and accidentally freeing) a second array.
    value_owner: dict[str, str] = {"input": "input_value"}
    for name in first_inputs:
        value_owner[name] = "input_value"
    normal_last: dict[str, int] = {}
    raw_last: dict[str, int] = {}
    for index, candidate in enumerate(selected):
        candidate_fused_prelu = prelu_fusion.get(candidate.name)
        candidate_fused_conv = fused_convs.get(candidate.name)
        candidate_emit_raw = (candidate.name in fusion.raw
                              or candidate.name in prelu_producers
                              or candidate.name in raw_multiplies)
        uses_raw = ((candidate_emit_raw and candidate.kind == "Concatenate")
                    or candidate.name in fusion.adds
                    or candidate_fused_prelu is not None
                    or candidate_fused_conv is not None)
        for input_name in candidate.inputs:
            owner = value_owner.get(input_name, "input_value")
            normal_last[owner] = index
            if uses_raw:
                raw_last[input_name] = index
        candidate_emit_truncated = (
            candidate.name not in raw_multiplies
            and (not candidate_emit_raw
                 or (candidate.name in fusion.truncated
                     and candidate.name not in prelu_producers)))
        if candidate.kind in {"Reshape", "Flatten"}:
            value_owner[candidate.name] = value_owner.get(
                candidate.inputs[0], "input_value")
        elif candidate_emit_truncated:
            value_owner[candidate.name] = _ident(candidate.name)
    if selected and selected[-1].name in value_owner:
        normal_last[value_owner[selected[-1].name]] = len(selected)
    lines.append("@for_range(REQUESTS)\ndef _(request_id):\n")
    body: list[str] = []
    # The offline phase of the request: every daBit it will consume is
    # generated here, before any timer of the request starts, so the online
    # phase never refills the buffer.
    if measured_timer:
        body.append(
            f"@if_(request_id >= WARMUPS)\ndef _():\n"
            f"    start_timer({OFFLINE_DABIT_TIMER_ID})\n")
    body.append(f"{prefetch}(REQUESTS)\n")
    if protocol_spec.prep_prefetch:
        # An actively secure generator hands part of its work to the worker
        # threads, and a worker reports its communication and preprocessing
        # time only when it next ends a tape.  One trivial tape on every thread
        # makes that report land here, inside the offline timer, instead of
        # inside the request's.
        body.append(
            "prefetch_sync = regint.Array(THREADS)\n"
            "@for_range_multithread(THREADS, 1, THREADS)\n"
            "def _(thread):\n"
            "    prefetch_sync[thread] = thread\n")
    if measured_timer:
        body.append(
            f"@if_(request_id >= WARMUPS)\ndef _():\n"
            f"    stop_timer({OFFLINE_DABIT_TIMER_ID})\n")
    if measured_timer:
        # Time only the post-warm-up requests: the timer accumulates across the
        # measured iterations and excludes the WARMUPS cold ones entirely, so the
        # benchmark divides by the trial count rather than warmups+trials.
        body.append(
            f"@if_(request_id >= WARMUPS)\ndef _():\n"
            f"    start_timer({MEASURED_TIMER_ID})\n")
        body.append(
            f"@if_(request_id >= WARMUPS)\ndef _():\n"
            f"    start_timer({INPUT_SHARE_TIMER_ID})\n")
    # A verified 1:1 request reads the identity it claims before the probe.
    claimed_identity = (verified_io and score is not None and identify is None
                        and bool(selected))
    if claimed_identity:
        body.append("request_identity = claimed_identity(request_id)\n")
    # The client's replicated sharing of this request's probe, which follows
    # the enrolled database (if any) in every server's share file.
    position = ("GALLERY_SIZE + request_id * INPUT_SIZE" if score is not None
                else "request_id * INPUT_SIZE")
    body.append(f"input_value = secret_input({position}, {input_size})\n")
    if measured_timer:
        body.append(
            f"@if_(request_id >= WARMUPS)\ndef _():\n"
            f"    stop_timer({INPUT_SHARE_TIMER_ID})\n")
    # With a score compiled in, one request is two operations, and MP-SPDZ
    # reports every metric per timer -- so bracketing the layer loop separately
    # makes the split a measurement rather than the subtraction of two
    # aggregates.  Without a score the aggregate timer *is* the embedding.
    split_timers = measured_timer and score is not None
    if split_timers:
        body.append(
            f"@if_(request_id >= WARMUPS)\ndef _():\n"
            f"    start_timer({EMBEDDING_TIMER_ID})\n")
    values: dict[str, str] = {"input": "input_value"}
    shapes: dict[str, list[int]] = {"input": manifest.input_shape}
    producer_shapes = {op.name: op.output_shape for op in manifest.ops}
    for name in first_inputs:
        values[name] = "input_value"
        shapes[name] = manifest.input_shape
    raw_values: dict[str, str] = {}
    deleted_allocations: set[str] = set()
    for timer_id, op in enumerate(selected, start=1):
        sources = [values.get(name, "input_value") for name in op.inputs]
        input_shapes = [shapes.get(name, producer_shapes.get(name, manifest.input_shape)) for name in op.inputs]
        target = _ident(op.name)
        fused_prelu = prelu_fusion.get(op.name)
        fused_multiply = multiply_fusion.get(op.name)
        fused_multiply_conv = fused_convs.get(op.name)
        emit_raw = (op.name in fusion.raw or op.name in prelu_producers
                    or op.name in raw_multiplies)
        # A raw SE Multiply has one consumer (its fused conv), so it publishes
        # only the untruncated product.
        emit_truncated = (op.name not in raw_multiplies
                          and (not emit_raw
                               or (op.name in fusion.truncated
                                   and op.name not in prelu_producers)))
        fused_bias_offset = (_offset(offsets, _add_fused_bias_name(op.name))
                             if op.name in fusion.adds else None)
        raw_sources = ([raw_values[name] for name in op.inputs]
                       if (emit_raw and op.kind == "Concatenate")
                       or op.name in fusion.adds or fused_prelu is not None
                       or fused_multiply_conv is not None
                       else None)
        raw_source_units = raw_chain_bits = None
        if op.name in fusion.adds or (emit_raw and op.kind == "Concatenate"):
            # Without a fused SE conv every operand carries ``2^{f + f_w}``, so
            # the exponents agree and no operand is lifted; the widths are still
            # needed, because a kernel grid above the activation grid makes the
            # fused Add state its declared length explicitly.
            raw_source_units = [_raw_scale_units(name, ops_by_name, fused_convs)
                                for name in op.inputs]
            if op.name in fusion.adds:
                raw_chain_bits = _add_truncand_bits(
                    op.inputs, ops_by_name, scale_bounds)
        body.append(f"marker('start', '{op.name}')\nstart_timer({timer_id})\n")
        body.append(_layer_code(op, target, sources, input_shapes, offsets,
                                raw_sources=raw_sources, emit_raw=emit_raw,
                                emit_truncated=emit_truncated,
                                fused_bias_offset=fused_bias_offset,
                                prelu_fusion=fused_prelu,
                                multiply_fusion=fused_multiply,
                                fused_multiply=fused_multiply_conv,
                                raw_source_units=raw_source_units,
                                raw_chain_bits=raw_chain_bits,
                                fractional_bits=fractional_bits,
                                total_bits=total_bits, ring_bits=ring_bits,
                                kernel_bits=kernel_bits,
                                bound=_pooled_bound(op, ops_by_name, scale_bounds)))
        body.append(f"stop_timer({timer_id})\nmarker('stop', '{op.name}')\n")
        temporaries: list[str] = []
        # Raw accumulators that are truncated inside this layer have no
        # graph consumer. Published raw values are reclaimed by raw_last.
        if op.kind in {"Conv2D", "DepthwiseConv2D"} and not emit_raw:
            temporaries.append(f"{target}_raw")
        if op.kind == "BatchNormalization":
            temporaries.extend((f"{target}_scale",
                                f"{target}_affine_bias"))
        elif op.kind == "Multiply":
            if fused_multiply is not None:
                temporaries.append(f"{target}_gate")
            else:
                output_size = _size(op.output_shape)
                temporaries.extend(
                    f"{target}_broadcast_{index}"
                    for index, shape in enumerate(input_shapes)
                    if _size(shape) != output_size)
        elif op.kind == "GlobalAveragePooling2D":
            temporaries.append(f"{target}_sums")
        for temporary in temporaries:
            body.append(f"{temporary}.delete()\n")
        if emit_raw:
            raw_values[op.name] = f"{target}_raw"
        if op.kind in {"Reshape", "Flatten"}:
            values[op.name] = sources[0]
        elif emit_truncated:
            values[op.name] = target
        shapes[op.name] = op.output_shape
        # ``delete`` is a compile-time allocator operation, not a runtime clear
        # or a protocol action.  Emit it after the layer timer so liveness
        # cannot alter layer telemetry.
        candidates = {value_owner.get(name) for name in op.inputs}
        for owner in sorted(x for x in candidates if x):
            if normal_last.get(owner) == timer_id - 1 and owner not in deleted_allocations:
                body.append(f"{owner}.delete()\n")
                deleted_allocations.add(owner)
        for input_name in op.inputs:
            raw_name = raw_values.get(input_name)
            if (raw_name is not None and raw_last.get(input_name) == timer_id - 1
                    and raw_name not in deleted_allocations):
                body.append(f"{raw_name}.delete()\n")
                deleted_allocations.add(raw_name)
    if split_timers:
        body.append(
            f"@if_(request_id >= WARMUPS)\ndef _():\n"
            f"    stop_timer({EMBEDDING_TIMER_ID})\n")
    if test_reveal and selected:
        body.append(f"print_ln('GHOSTMPC_TEST_OUTPUT %s', {values[selected[-1].name]}.get_vector().reveal())\n")
    else:
        body.append("# The embedding remains secret-shared; no server reveals it.\n")
    if identify is not None and selected:
        # A probe that declares no identity is ranked against every enrolled
        # row, and what the servers hold at the end is the rank-1 index --
        # share components again, opened by nobody.
        body.append(score_codegen.identify_block(
            identify, values[selected[-1].name], measured_timer=measured_timer,
            test_reveal=test_reveal))
        if verified_io:
            # The rank-1 index alone: the winning accumulator stays with the
            # servers.
            body.append(score_codegen.delivery("identify_output.get_part(0, 1)"))
    elif score is not None and selected:
        # The servers open nothing here either: ``score_output`` holds two
        # secret-shared ring elements whose components each server returns to
        # the client, which reconstructs and divides in the clear.
        body.append(score_codegen.block(
            score, values[selected[-1].name], measured_timer=measured_timer,
            test_reveal=test_reveal,
            row=(score_codegen.CLAIMED_ROW if claimed_identity
                 else score_codegen.COUNTER_ROW)))
        if match is not None:
            # Still nothing is opened: ``match_output`` holds one secret-shared
            # bit whose components each server returns, and the client
            # reconstructs the match / non-match answer.
            body.append(score_codegen.match_block(
                match, measured_timer=measured_timer, test_reveal=test_reveal))
        if verified_io:
            body.append(score_codegen.delivery(
                "match_output" if match is not None else "score_output"))
    if measured_timer:
        body.append(
            f"@if_(request_id >= WARMUPS)\ndef _():\n"
            f"    stop_timer({MEASURED_TIMER_ID})\n")
    lines.extend("    " + line if line.strip() else line for chunk in body for line in chunk.splitlines(keepends=True))
    return "".join(lines)

def generate(manifest_path: Path, output: Path, *, threads: int = 1,
             requests: int = 1, test_reveal: bool = False,
             warmups: int = 0, measured_timer: bool = False,
             score: ScorePlan | None = None,
             match: MatchPlan | None = None,
             identify: IdentifyPlan | None = None,
             protocol: str = "replicated-ring-party",
             optimizations: Optimizations = DEPLOYED,
             verified_io: bool | None = None) -> Path:
    """Write the program to ``output`` and its public weights beside it.

    Returns the path of the public-weight stream the parties read through
    ``public_input``.
    """
    import numpy as np

    manifest = Manifest.load(manifest_path)
    quantization = manifest.quantization
    ring_bits = quantization.ring_bits
    if ring_bits not in SUPPORTED_RING_BITS:
        raise ValueError(
            f"the manifest declares a {ring_bits}-bit ring; the program targets "
            f"the {RING_BITS}-bit deployment profile (or {SUPPORTED_RING_BITS[-1]} "
            "bits for the ablation)")
    fractional_bits = quantization.fractional_bits
    total_bits = quantization.activation_bits
    # Public kernels may ride a finer grid than the activations; the profile
    # carries the choice.
    kernel_bits = quantization.weight_fractional_bits
    kernel_bits = fractional_bits if kernel_bits is None else int(kernel_bits)
    if kernel_bits < fractional_bits:
        raise ValueError("weight_fractional_bits must be >= fractional_bits")
    weights_path = manifest_path.parent / manifest.weights_file
    if not weights_path.is_file():
        if manifest.tensors:
            raise FileNotFoundError(weights_path)
        weights = np.empty(0, dtype=np.float64)
    else:
        weights = np.load(weights_path, mmap_mode="r")
    # A calibrated manifest carries tighter bounds than the static ones.
    bounds = manifest_bounds(manifest) or _accumulator_bounds(
        manifest, weights, fractional_bits=fractional_bits,
        total_bits=total_bits,
    )
    # The fused truncation length depends on the real slope magnitudes, so the
    # source render is handed the same measured coefficients the weight plan
    # derives, keeping the two in lock-step.
    alpha_magnitudes = _prelu_alpha_magnitudes(manifest, weights)
    plan = _prepare_weights(
        manifest, weights, fractional_bits=fractional_bits, ring_bits=ring_bits,
        overflow_bounds=bounds, alpha_magnitudes=alpha_magnitudes,
        kernel_bits=kernel_bits, optimizations=optimizations)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(
        manifest, fractional_bits, threads=threads,
        requests=requests, weight_offsets=plan.offsets,
        weight_count=len(plan.values),
        total_bits=total_bits, kernel_bits=kernel_bits,
        test_reveal=test_reveal,
        warmups=warmups, measured_timer=measured_timer,
        overflow_bounds=bounds, alpha_magnitudes=alpha_magnitudes,
        score=score, match=match, identify=identify, protocol=protocol,
        ring_bits=ring_bits, optimizations=optimizations,
        verified_io=verified_io,
    ))
    public_input_path = _public_input_path(output)
    _write_public_weights(public_input_path, plan.values, fractional_bits, ring_bits)
    return public_input_path

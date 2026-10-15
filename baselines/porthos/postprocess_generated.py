#!/usr/bin/env python3
"""Make generated Porthos C++ consume shares and retain output shares.

The transformation is intentionally fail-closed: every structural anchor and
the exact 512-element output gate must be present exactly once.  It moves the
external-input read into a persistent request loop, makes P0 and P1 each load
their own additive component, loads the public weights once on both primaries,
and replaces the reveal of the embedding by local share writes; P2 writes only
logs.  With ``--score`` it appends the 1:1 verification score, built from
Porthos' own secret-by-secret product and truncation.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

INCLUDE = '#include "porthos_runtime_support.h"\n'


def _input_block(source: str, variable: str) -> tuple[int, int, str]:
    marker = f"input variable {variable} at"
    marker_at = source.find(marker)
    if marker_at < 0:
        raise RuntimeError(f"missing generated input marker for {variable}")
    begin = source.rfind("\nvector", 0, marker_at)
    if begin < 0:
        raise RuntimeError(f"missing declaration for {variable}")
    begin += 1
    next_declaration = source.find("\nvector", marker_at)
    start_computation = source.find("\nStartComputation();", marker_at)
    candidates = [value + 1 for value in (next_declaration, start_computation)
                  if value >= 0]
    if not candidates:
        raise RuntimeError("cannot delimit generated input block")
    end = min(candidates)
    block = source[begin:end]
    if block.count("(role == CLIENT)") < 2:
        raise RuntimeError("input block is not an upstream secret CLIENT input")
    block = block.replace("(role == CLIENT)", "PRIMARY")
    return begin, end, block


def _score_gallery_read(size: int) -> str:
    """Read the enrolled row once, before the request loop.

    It is a setup input, like the public weights and unlike the per-request
    share, so it is read once and guarded by ``PRIMARY``: P2 receives no stdin
    stream at all in this integration.
    """
    return f"""std::vector<porthosSecretType> ghostmpc_gallery({size});
std::vector<porthosSecretType> ghostmpc_score(2);
for (uint32_t i = 0; i < (uint32_t){size}; i++) {{
porthosSecretType __tmp_in_ghostmpc_gallery = 0;
if (PRIMARY) {{ cin >> __tmp_in_ghostmpc_gallery; }}
ghostmpc_gallery[i] = __tmp_in_ghostmpc_gallery;
}}
"""


def transform(source: str, *, input_variable: str = "tmp0",
              output_elements: int = 512,
              score: bool = False, scale: int = 16
              ) -> tuple[str, dict[str, int]]:
    if INCLUDE.strip() not in source:
        include_anchor = '#include "ezpc.h"\n'
        if source.count(include_anchor) != 1:
            raise RuntimeError("unexpected generated include prelude")
        source = source.replace(include_anchor, include_anchor + INCLUDE)

    generated_inputs = re.findall(
        r"input variable ([A-Za-z_][A-Za-z0-9_]*) at", source)
    public_initializers_names = [
        name for name in generated_inputs if name != input_variable]

    # Public (int64_t) initializers are loaded by P0 and P1 only. P2 receives
    # neither an application input nor the public-weight stdin stream.
    temporary_pattern = (
        r"\b((?:u?int64_t)) (__tmp_in_[A-Za-z0-9_]+);")
    public_temporaries = len(re.findall(temporary_pattern, source))
    source = re.sub(temporary_pattern, r"\1 \2 = 0;", source)
    public_reads = len(re.findall(
        r"(?m)^cin >> (__tmp_in_[A-Za-z0-9_]+);$", source))
    source = re.sub(
        r"(?m)^cin >> (__tmp_in_[A-Za-z0-9_]+);$",
        r"if (PRIMARY) { cin >> \1; }", source)
    if public_temporaries == 0 or public_reads < public_temporaries:
        raise RuntimeError("no public initializer stream found")

    begin, end, input_block = _input_block(source, input_variable)
    source = source[:begin] + source[end:]
    start = source.find("StartComputation();")
    finish = source.find("EndComputation();", start)
    if start < 0 or finish < 0 or source.find("StartComputation();", start + 1) >= 0:
        raise RuntimeError("expected exactly one online computation region")
    finish += len("EndComputation();")

    separator = source.find('cout << "----------------------------------"', finish)
    if separator < 0:
        raise RuntimeError("missing generated output trailer")
    output_region = source[finish:separator]
    reveal_pattern = re.compile(
        r"cout << \(funcReconstruct2PCCons\((.+), 2\)\) << endl;",
        re.MULTILINE)
    reveals = reveal_pattern.findall(output_region)
    if len(reveals) != 1:
        raise RuntimeError(
            f"expected one vectorized output gate, found {len(reveals)}")
    embedding = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", reveals[0]).group(1)
    output_region = reveal_pattern.sub(
        r"ghostmpc_porthos::write_share(\1, ghostmpc_request);", output_region)
    if score:
        # The two score components are written through the same gate as the
        # embedding, so a run emits `output_elements + 2` components per party
        # and nothing is reconstructed online for them either.
        output_region += (
            "for (uint32_t i = 0; i < (uint32_t)2; i++){\n"
            "ghostmpc_porthos::write_share(ghostmpc_score[i], ghostmpc_request);\n"
            "}\n")
    loop_bounds = re.findall(
        r"for \(uint32_t [^;]+; [^<]+<\s*\(int32_t\)(\d+);", output_region)
    loop_elements = 1
    for bound in loop_bounds:
        loop_elements *= int(bound)
    if not loop_bounds or loop_elements != output_elements:
        raise RuntimeError(
            f"expected output loop of {output_elements} elements")

    online = source[start:finish]
    retained_initializers = 0
    for name in public_initializers_names:
        pattern = re.compile(
            rf"ClearMemPublic(?:[1-5])?\([^;]*\b{re.escape(name)}\s*\);")
        online, count = pattern.subn(
            f"/* retained public initializer {name} */", online)
        retained_initializers += count
    if retained_initializers != len(public_initializers_names):
        raise RuntimeError(
            "expected one ClearMemPublic call per public initializer; "
            f"retained {retained_initializers} of "
            f"{len(public_initializers_names)}")
    if score:
        # Computed from the finished embedding and *before* EndComputation(),
        # so its traffic falls inside the region the online timer brackets.
        call = (f"GhostScore((int32_t){output_elements}, {embedding}[0], "
                f"ghostmpc_gallery, (int32_t){scale}, ghostmpc_score);\n")
        if online.count("EndComputation();") != 1:
            raise RuntimeError("expected one EndComputation() in the online region")
        online = online.replace("EndComputation();", call + "EndComputation();", 1)
    # Everything from the first statement of main() to here -- session
    # keys, the mod-prime tables, the sockets, the setup barrier, the public
    # weight stream and the enrolled row -- is the offline phase, and it runs
    # once per process.  The request loop below is the online one.
    setup_anchor = "parseInputs(argc, argv);"
    if source.count(setup_anchor) != 1:
        raise RuntimeError("expected one generated setup entry point")
    source = source.replace(
        setup_anchor, setup_anchor + "\nGhostOfflineBegin();", 1)
    start += len("\nGhostOfflineBegin();")
    finish += len("\nGhostOfflineBegin();")
    separator += len("\nGhostOfflineBegin();")
    persistent = (
        (_score_gallery_read(output_elements) if score else "") +
        "GhostOfflineEnd();\n" +
        "for (int ghostmpc_request = 0; "
        "ghostmpc_request < ghostmpc_porthos::request_count(); "
        "++ghostmpc_request) {\n" + input_block + "\n" + online +
        output_region + "\n}\n")
    source = source[:start] + persistent + source[separator:]
    if "funcReconstruct2PCCons(" in source:
        raise RuntimeError("online output reconstruction remains after patching")
    return source, {
        "public_temporaries": public_temporaries,
        "public_reads_guarded": public_reads,
        "shared_input_blocks": 1,
        "output_elements": output_elements,
        "persistent_request_loop": 1,
        "offline_phase_bracketed": 1,
        "retained_public_initializers": retained_initializers,
        "score": 1 if score else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpp", type=Path, required=True,
                        help="the C++ file the EzPC compiler generated")
    parser.add_argument("--output", type=Path,
                        help="where to write the result (default: in place)")
    parser.add_argument("--input-variable", default="tmp0")
    parser.add_argument("--output-elements", type=int, default=512)
    parser.add_argument("--score", action="store_true",
                        help="append the 1:1 verification score, built from "
                             "Porthos' own secret-by-secret product")
    parser.add_argument("--scale", type=int, default=16)
    args = parser.parse_args()
    source = args.cpp.read_text()
    updated, result = transform(
        source, input_variable=args.input_variable,
        output_elements=args.output_elements,
        score=args.score, scale=args.scale)
    target = args.output or args.cpp
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(updated)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

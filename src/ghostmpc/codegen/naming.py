"""Identifiers, tile constants and the truncation primitive.

Every generated program refers to a tensor through one deterministic
identifier, and every communicating step is a truncation.  Both live here so
the layer emitters below stay free of string plumbing.
"""

from __future__ import annotations

import math
import re

# Timer bracketing only the measured (post-warmup) requests. It sits well above
# any per-layer timer id and above the request timer so the three never collide,
# letting the benchmark read wall/comm/rounds/CPU with the warm-up requests
# genuinely excluded instead of averaged in.
MEASURED_TIMER_ID = 20_000

# Public weight ingestion happens once, before the request loop, so its timer is
# unconditional; input sharing repeats per request and is timed only on the
# measured ones alongside the aggregate MEASURED timer.
WEIGHT_LOAD_TIMER_ID = 20_001

INPUT_SHARE_TIMER_ID = 20_002

# A scored request splits into its two operations.  MP-SPDZ reports bytes,
# rounds, wall and CPU per timer, so bracketing them separately makes the split
# a measurement rather than a subtraction of two aggregates.  They are emitted
# only when a score is compiled in.
EMBEDDING_TIMER_ID = 20_003

SCORE_TIMER_ID = 20_004

# The database is ingested once per process, like the public weights, and is
# reported outside the request path for the same reason.
GALLERY_LOAD_TIMER_ID = 20_005

# The match decision is an operation of its own, bracketed like the score and
# emitted only when one is compiled in.
MATCH_TIMER_ID = 20_006

# A closed-set identification is the operation a request performs
# *instead of* the 1:1 score, so it carries its own timer rather than reusing
# the score's: the two are never compiled into the same program, and a reader
# of a report must not have to know which one a `score` row meant.
IDENTIFY_TIMER_ID = 20_007

# The offline phase of a request: the daBits it will consume are
# generated before its MEASURED timer starts, under a timer of their own, so a
# report carries the online request and the material it consumed side by side
# without either being a subtraction.
OFFLINE_DABIT_TIMER_ID = 20_008

# Output channels are contiguous in the Keras kernel, so one weight vector and
# one input expansion cover the whole tile: the wider the tile, the fewer times
# each input scalar is re-expanded.
CONV_CHANNEL_TILE = 512

CONV_SPATIAL_TILE = 4

CONV_ALIGNED_SPATIAL_TILE = 7

# Elements per packed depthwise vector. The lane holds full-width NHWC pixels,
# so the spatial extent shrinks as the channel count grows.
DEPTHWISE_LANE_CAP = 3584

# Channels per vector on the per-pixel depthwise edge regions, which have no
# weight vector to amortize and so gain nothing from a wider tile.
DEPTHWISE_CHANNEL_TILE = 64

# A conv->PReLU truncation shifts an accumulator carrying the product scale and
# the alpha scale at once, so its declared bit length plus the rounding term has
# to stay inside the ring.  Two bits below the ring width leave room for the
# accumulator sign and the nearest-rounding carry.
def _fused_truncation_bits(ring_bits: int) -> int:
    return ring_bits - 2

# Fewer alpha bits than this quantize the negative slope too coarsely to trade
# against one saved truncation.
MIN_ALPHA_BITS = 8

def _size(shape: list[int]) -> int:
    return math.prod(shape)

def _ident(name: str) -> str:
    return "v_" + re.sub(r"[^A-Za-z0-9_]", "_", name)

def _offset(offsets: dict[str, int], name: str) -> int:
    if name not in offsets:
        raise ValueError(f"public tensor {name!r} is missing from the weight plan")
    return offsets[name]

def _copy(target: str, source: str, size: int) -> str:
    return f"{target} = sfix.Array({size})\n{target}.assign({source}.get_vector())\n"

def _public_scaled_vector(name: str, size: int, tensor_offset: int, channels: int) -> str:
    return f'''{name} = cint.Array({size})
@for_range({size})
def _(i):
    {name}[i] = public_weights[{tensor_offset} + i % {channels}]
'''

def _prelu_expanded_alpha_name(op_name: str) -> str:
    return f"{op_name}:prelu_expanded_alpha"

def _conv_expanded_bias_name(op_name: str) -> str:
    return f"{op_name}:conv_expanded_bias"

def _add_fused_bias_name(op_name: str) -> str:
    return f"{op_name}:add_fused_bias"

def _conv_raw_bias_name(op_name: str) -> str:
    return f"{op_name}:conv_raw_bias"

def _kernel_truncation(fractional_bits: int, kernel_bits: int) -> dict[str, str]:
    """Declared length and shift of a plain accumulator at ``f + f_w``.

    With kernels on the activation grid this is the symbolic ``sfix`` default;
    a wider kernel grid has to state both explicitly, since ``sfix.f`` is the
    activation scale.
    """
    if kernel_bits == fractional_bits:
        return {}
    return {"length": f"sfix.k + {kernel_bits}", "shift": str(kernel_bits)}

def _bounded_split(length_bits: int, ring_bits: int) -> bool:
    """Whether the exact bounded split is worth taking at this declared length.

    ``trunc_round_nearest_bounded`` enlarges the declaration by one bit before
    splitting, so a truncand already filling the ring splits the whole ring
    anyway and gains nothing over the generic ``TruncRing``.
    """
    return length_bits + 1 < ring_bits

def _truncate(raw: str, target: str, size: int,
              expanded_bias_offset: int | None = None,
              *, length: str = "sfix.k + sfix.f", shift: str = "sfix.f",
              bounded: bool = False) -> str:
    """Truncate one accumulator, adding an already broadcast public bias.

    ``length`` and ``shift`` default to the one-truncation-per-output contract of
    a scale ``2^f`` accumulator.  A fused ``Multiply SE -> Conv`` accumulator
    instead carries ``2^{2f}`` on its raw operand, so its consumer passes the
    wider declared length and the ``2f`` shift explicitly.
    """
    raw_value = f"{raw}.get_vector()"
    if bounded:
        expression = (
            f"trunc_round_nearest_bounded({raw_value}, "
            f"{length}, {shift}, signed=True)"
        )
    else:
        expression = f"truncate_generic({raw_value}, {length}, {shift})"
    if expanded_bias_offset is not None:
        expression += f" + public_weights.get_vector({expanded_bias_offset}, {size})"
    return (f"{target} = sfix.Array({size})\n"
            f"{target}.assign(sfix._new({expression}))\n")

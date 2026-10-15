"""Generate an MP-SPDZ inference program with public model weights.

The invariant driving the whole package is that model parameters are ``cint``
values.  Linear layers therefore perform local secret-by-public arithmetic and
truncate once per output, rather than consuming a Beaver triple per MAC.

The modules split that work as follows:

``templates``
    static source fragments of the emitted program;
``naming``
    tensor identifiers, tile constants and the truncation primitive;
``fusion``
    which truncations may be deferred into their consumer;
``weight_stream``
    layout of the flat public-weight stream;
``layers``
    one emitter per operation kind, plus the dispatch table;
``program``
    assembly of the whole program and the on-disk artifacts.

Consumers that need an internal helper import it from the owning module; only
the entry points are re-exported here.
"""

from __future__ import annotations

from .fusion import NO_FUSION, FusionPlan, MultiplyFusion, PreluFusion
from .layers.pooling import pooling_reciprocal
from .naming import (
    CONV_CHANNEL_TILE,
    DEPTHWISE_LANE_CAP,
    EMBEDDING_TIMER_ID,
    GALLERY_LOAD_TIMER_ID,
    IDENTIFY_TIMER_ID,
    INPUT_SHARE_TIMER_ID,
    MATCH_TIMER_ID,
    MEASURED_TIMER_ID,
    MIN_ALPHA_BITS,
    OFFLINE_DABIT_TIMER_ID,
    SCORE_TIMER_ID,
    WEIGHT_LOAD_TIMER_ID,
)
from .program import RING_BITS, generate, render, verified_io_default

__all__ = [
    "CONV_CHANNEL_TILE",
    "DEPTHWISE_LANE_CAP",
    "EMBEDDING_TIMER_ID",
    "FusionPlan",
    "GALLERY_LOAD_TIMER_ID",
    "IDENTIFY_TIMER_ID",
    "MATCH_TIMER_ID",
    "INPUT_SHARE_TIMER_ID",
    "MEASURED_TIMER_ID",
    "MIN_ALPHA_BITS",
    "OFFLINE_DABIT_TIMER_ID",
    "SCORE_TIMER_ID",
    "MultiplyFusion",
    "NO_FUSION",
    "PreluFusion",
    "WEIGHT_LOAD_TIMER_ID",
    "generate",
    "pooling_reciprocal",
    "RING_BITS",
    "render",
    "verified_io_default",
]

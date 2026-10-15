"""Parsing and aggregation of what one MP-SPDZ run reports.

Each party writes its own communication and timer lines, so every figure here
is reconciled across the three of them before it becomes a :class:`Trial`.
"""

from __future__ import annotations

import math
import re
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

_DATA = re.compile(r"Data sent = ([0-9.]+) MB in ~([0-9]+) rounds")

_TIMER = re.compile(
    r"^Time([0-9]+) = ([0-9.eE+-]+) seconds "
    r"\(([0-9.eE+-]+) MB, ([0-9]+) rounds, ([0-9.eE+-]+) CPU seconds\)$",
    re.MULTILINE,
)

_PHASE_TIMER = re.compile(
    r"^Time([0-9]+)\[(online|offline)\] = ([0-9.eE+-]+) seconds "
    r"\(([0-9.eE+-]+) MB, ([0-9]+) rounds, ([0-9.eE+-]+) CPU seconds\)$",
    re.MULTILINE,
)

_TIMER_BYTES = re.compile(
    r"^Time([0-9]+)(?:\[(online|offline)\])?\[bytes\] = ([0-9]+)$",
    re.MULTILINE,
)

_TOTAL_TIME = re.compile(r"^Time = ([0-9.eE+-]+) seconds", re.MULTILINE)

_REQUEST_TIMER = 10_000

def _size(shape: list[int]) -> int:
    result = 1
    for value in shape:
        result *= value
    return result

@dataclass(slots=True)
class Trial:
    layer: str
    phase: str
    wall_seconds: float
    cpu_seconds_p0: float
    cpu_seconds_p1: float
    bytes_p0_to_p1: int
    bytes_p1_to_p0: int
    rounds: int | None
    protocol: str = "rep3"
    pool_id: str = ""
    requests: int = 1
    cpu_seconds_by_party: tuple[float, ...] = ()
    bytes_by_party: tuple[int, ...] = ()

    @property
    def cpu_seconds_total(self) -> float:
        return sum(self.cpu_seconds_by_party) if self.cpu_seconds_by_party else self.cpu_seconds_p0 + self.cpu_seconds_p1

    @property
    def bytes_total(self) -> int:
        return sum(self.bytes_by_party) if self.bytes_by_party else self.bytes_p0_to_p1 + self.bytes_p1_to_p0

def parse_group_stats(stderr: list[str], *, required: bool = True) -> tuple[tuple[int, ...], int]:
    parsed = []
    for text in stderr:
        match = _DATA.search(text)
        if not match:
            if required:
                raise ValueError("MP-SPDZ statistics not found; run with a telemetry-enabled build")
            parsed.append((0, 0))
            continue
        parsed.append((round(float(match.group(1)) * 1_000_000), int(match.group(2))))
    return tuple(item[0] for item in parsed), max(item[1] for item in parsed)

def _timer_maps(
        text: str
) -> tuple[dict[int, tuple[float, int, int, float]],
           dict[tuple[int, str], tuple[float, int, int, float]]]:
    total = {
        int(timer): (float(wall), round(float(mb) * 1_000_000), int(rounds), float(cpu))
        for timer, wall, mb, rounds, cpu in _TIMER.findall(text)
    }
    phases = {
        (int(timer), phase): (
            float(wall), round(float(mb) * 1_000_000), int(rounds), float(cpu))
        for timer, phase, wall, mb, rounds, cpu in _PHASE_TIMER.findall(text)
    }
    for timer_text, phase, byte_text in _TIMER_BYTES.findall(text):
        timer = int(timer_text)
        byte_count = int(byte_text)
        if phase:
            key = (timer, phase)
            if key in phases:
                wall, _, rounds, cpu = phases[key]
                phases[key] = (wall, byte_count, rounds, cpu)
        elif timer in total:
            wall, _, rounds, cpu = total[timer]
            total[timer] = (wall, byte_count, rounds, cpu)
    return total, phases

def _aggregate_timer_values(
        values: Sequence[tuple[float, int, int, float]], *,
        divisor: int = 1) -> dict[str, object]:
    divisor = max(1, divisor)
    wall = [item[0] / divisor for item in values]
    byte_counts = [round(item[1] / divisor) for item in values]
    rounds = [round(item[2] / divisor) for item in values]
    cpu = [item[3] / divisor for item in values]
    return {
        "wall_seconds": max(wall, default=0.0),
        "wall_seconds_by_party": wall,
        "cpu_seconds_total": sum(cpu),
        "cpu_seconds_by_party": cpu,
        "bytes_total": sum(byte_counts),
        "bytes_by_party": byte_counts,
        "rounds": max(rounds, default=0),
        "rounds_by_party": rounds,
    }

def _validate_timer_partition(
        total: tuple[float, int, int, float],
        online: tuple[float, int, int, float],
        offline: tuple[float, int, int, float], *,
        label: str) -> None:
    for index, name in ((1, "bytes"), (2, "rounds")):
        if total[index] != online[index] + offline[index]:
            raise ValueError(
                f"{label}: total {name} is not online + offline")
    for index, name in ((0, "wall"), (3, "CPU")):
        if not math.isclose(
                total[index], online[index] + offline[index],
                rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError(
                f"{label}: total {name} is not online + offline")

def parse_request_timer(stderr: list[str], timer_id: int = _REQUEST_TIMER, *,
                        require_phases: bool = False) -> dict[str, object]:
    """Aggregate one MP-SPDZ inference timer without counting traffic twice."""
    values: list[tuple[float, int, int, float]] = []
    phase_values: dict[str, list[tuple[float, int, int, float]]] = {
        "online": [], "offline": []}
    for party, text in enumerate(stderr):
        timers, phases = _timer_maps(text)
        if timer_id not in timers:
            raise ValueError(f"missing request timer {timer_id} for party {party}")
        values.append(timers[timer_id])
        for phase in phase_values:
            value = phases.get((timer_id, phase))
            if value is None and require_phases:
                raise ValueError(f"missing {phase} request timer {timer_id} for party {party}")
            if value is not None:
                phase_values[phase].append(value)
        if require_phases and all(
                (timer_id, phase) in phases
                for phase in ("online", "offline")):
            _validate_timer_partition(
                timers[timer_id],
                phases[(timer_id, "online")],
                phases[(timer_id, "offline")],
                label=f"request timer {timer_id}, party {party}")
    result = {
        "wall_ms": max(item[0] for item in values) * 1000,
        "wall_ms_by_party": [item[0] * 1000 for item in values],
        "cpu_ms_total": sum(item[3] for item in values) * 1000,
        "cpu_ms_by_party": [item[3] * 1000 for item in values],
        "bytes_sent_total": sum(item[1] for item in values),
        "bytes_sent_by_party": [item[1] for item in values],
        "rounds": max(item[2] for item in values),
        "rounds_by_party": [item[2] for item in values],
    }
    result["phases"] = {"total": _aggregate_timer_values(values)}
    for phase, items in phase_values.items():
        if len(items) == len(values):
            result["phases"][phase] = _aggregate_timer_values(items)
        elif phase == "online":
            result["phases"][phase] = dict(result["phases"]["total"])
        else:
            result["phases"][phase] = _aggregate_timer_values(
                [(0.0, 0, 0, 0.0) for _ in values])
    return result

def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("empty values")
    values = sorted(values)
    index = (len(values) - 1) * q
    lo, hi = int(index), min(int(index) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)

def summarize(trials: Iterable[Trial]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[Trial]] = {}
    for trial in trials:
        groups.setdefault((trial.layer, trial.phase), []).append(trial)
    rows: list[dict[str, object]] = []
    for (layer, phase), items in sorted(groups.items()):
        def stats(attr: str) -> dict[str, float]:
            values = [float(getattr(item, attr)) for item in items]
            return {"median": statistics.median(values), "p5": percentile(values, .05), "p95": percentile(values, .95)}
        rows.append({
            "layer": layer, "phase": phase, "trials": len(items),
            "protocol": items[0].protocol,
            "wall_seconds": stats("wall_seconds"), "cpu_seconds_p0": stats("cpu_seconds_p0"),
            "cpu_seconds_p1": stats("cpu_seconds_p1"), "cpu_seconds_total": {
                "median": statistics.median(x.cpu_seconds_total for x in items),
                "p5": percentile([x.cpu_seconds_total for x in items], .05),
                "p95": percentile([x.cpu_seconds_total for x in items], .95),
            },
            "bytes_p0_to_p1": stats("bytes_p0_to_p1"), "bytes_p1_to_p0": stats("bytes_p1_to_p0"),
            "bytes_total": {
                "median": statistics.median(x.bytes_total for x in items),
                "p5": percentile([x.bytes_total for x in items], .05),
                "p95": percentile([x.bytes_total for x in items], .95),
            },
            "rounds": ({
                "median": statistics.median(
                    float(x.rounds) for x in items if x.rounds is not None),
                "p5": percentile(
                    [float(x.rounds) for x in items if x.rounds is not None], .05),
                "p95": percentile(
                    [float(x.rounds) for x in items if x.rounds is not None], .95),
            } if any(x.rounds is not None for x in items) else {
                "median": None, "p5": None, "p95": None}),
        })
    return rows

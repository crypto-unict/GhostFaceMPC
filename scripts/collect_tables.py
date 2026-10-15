#!/usr/bin/env python3
"""Print the paper's tables from the reports ``run_experiments.sh`` wrote.

Reads, under ``--results`` (default ``results/``):

* ``verification/<protocol>/<network>/report.json`` and
  ``identification/<protocol>/<network>/report.json`` -- ``ghostmpc benchmark``
  reports, one per security profile and emulated network;
* ``accuracy/<dataset>.json`` -- ``ghostmpc evaluate-accuracy``;
* ``plaintext/report.json``, ``naive/report.json`` and ``porthos/report.json``
  -- the three baselines;
* ``ablation/<n>-<step>/report.json`` -- ``scripts/run_ablation.py``, one per
  step of the ablation ladder (its step 0 is the naive baseline).

Missing reports are left out (or shown as ``--``), so the script can be run at
any point.  Bytes and rounds are deterministic properties of the programs and
reproduce exactly; wall times depend on the host.  Online and offline phases
are reported separately and never summed, as in the paper.  The tables are
printed and written to ``<results>/tables.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROTOCOLS = (("replicated-ring-party", "HbC"), ("ps-rep-ring-party", "M"))
NETWORKS = (("co-host", "Co-Host"), ("lan", "LAN"), ("fast-wan", "Fast-WAN"),
            ("wan", "WAN"))
DATASETS = (("lfw", "LFW"), ("agedb_30", "AgeDB"), ("calfw", "CALFW"),
            ("cplfw", "CPLFW"), ("cfp_fp", "CFP-FP"))


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _median(report: dict, layer: str, phase: str, metric: str) -> float | None:
    for row in report.get("summary", []):
        if row["layer"] == layer and row["phase"] == phase:
            return row[metric]["median"]
    return None


def _bandwidth(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value / 1e9:.2f} GB" if value >= 1e9 else f"{value / 1e6:.2f} MB"


def _rounds(value: float | None) -> str:
    return "--" if value is None else f"{round(value)}"


def _seconds(value: float | None) -> str:
    return "--" if value is None else f"{value:.2f}"


def _mpc_phases(report: dict) -> dict[str, tuple]:
    """Online and offline (rounds, wall, bytes) of one ghostmpc report."""
    def phase(layer: str) -> tuple:
        return (_median(report, layer, "total", "rounds"),
                _median(report, layer, "total", "wall_seconds"),
                _median(report, layer, "total", "bytes_total"))
    return {"online": phase("end_to_end"), "offline": phase("offline_dabits")}


def benchmark_table(results: Path, task: str, title: str) -> list[str]:
    lines = [f"### {title}", "",
             "| Security | Setting | Online rounds | Online wall (s) | Online bandwidth "
             "| Offline rounds | Offline wall (s) | Offline bandwidth |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    found = False
    for protocol, security in PROTOCOLS:
        for network, setting in NETWORKS:
            report = _load(results / task / protocol / network / "report.json")
            if report is None:
                continue
            found = True
            phases = _mpc_phases(report)
            cells = []
            for name in ("online", "offline"):
                rounds, wall, size = phases[name]
                cells += [_rounds(rounds), _seconds(wall), _bandwidth(size)]
            lines.append(f"| {security} | {setting} | " + " | ".join(cells) + " |")
    return lines + [""] if found else []


def _accuracy(results: Path) -> dict[str, dict]:
    rows = {}
    for dataset, _ in DATASETS:
        row = _load(results / "accuracy" / f"{dataset}.json")
        if row is not None:
            rows[dataset] = row
    return rows


def _accuracy_cells(rows: dict[str, dict], key: str) -> list[str]:
    values = [rows.get(dataset, {}).get(key) for dataset, _ in DATASETS]
    cells = ["--" if value is None else f"{100 * value:.2f}" for value in values]
    present = [value for value in values if value is not None]
    average = (f"{100 * sum(present) / len(present):.2f}"
               if len(present) == len(DATASETS) else "--")
    return cells + [average]


def summary_table(results: Path) -> list[str]:
    accuracy = _accuracy(results)
    header = ["| Method | Security | Ring | "
              + " | ".join(name for _, name in DATASETS)
              + " | Avg | Rounds | Wall (s) | Bandwidth |",
              "|---|---|---|" + "---:|" * (len(DATASETS) + 4)]
    lines = ["### Comparison with the baselines (online phase, co-host)", ""] + header
    teacher = _accuracy_cells(accuracy, "teacher")
    quantized = _accuracy_cells(accuracy, "quantized")
    # The 64-bit baselines run the float model's decisions, so they carry the
    # teacher's accuracy.
    plaintext = _load(results / "plaintext" / "report.json")
    wall = (plaintext["regions"]["total"]["wall_seconds"]["median"]
            if plaintext else None)
    lines.append("| TensorFlow | None | -- | " + " | ".join(teacher)
                 + f" | -- | {_seconds(wall)} | -- |")
    porthos = _load(results / "porthos" / "report.json")
    request = porthos.get("request", {}) if porthos else {}
    lines.append("| EzPC (Porthos) | HbC | 64 | " + " | ".join(teacher)
                 + f" | {_rounds(request.get('rounds'))}"
                 f" | {_seconds(request.get('wall_seconds'))}"
                 f" | {_bandwidth(request.get('bytes_total'))} |")
    naive = _load(results / "naive" / "report.json")
    if naive is not None:
        cells = (_median(naive, "end_to_end", "online", "rounds"),
                 _median(naive, "end_to_end", "online", "wall_seconds"),
                 _median(naive, "end_to_end", "online", "bytes_total"))
    else:
        cells = (None, None, None)
    lines.append("| MP-SPDZ (naive) | HbC | 64 | " + " | ".join(teacher)
                 + f" | {_rounds(cells[0])} | {_seconds(cells[1])}"
                 f" | {_bandwidth(cells[2])} |")
    for protocol, security in PROTOCOLS:
        report = _load(results / "verification" / protocol / "co-host" / "report.json")
        rounds, wall, size = (_mpc_phases(report)["online"] if report
                              else (None, None, None))
        lines.append(f"| GhostFaceMPC | {security} | 32 | " + " | ".join(quantized)
                     + f" | {_rounds(rounds)} | {_seconds(wall)} | {_bandwidth(size)} |")
    return lines + [""]


#: The ablation ladder, in order: (directory under ``ablation/``, label).
#: Step 0 is the naive baseline of the comparison table.
ABLATION = (
    (None, "MP-SPDZ naive"),
    ("1-vectorization", "+ Vectorization"),
    ("2-bn-folding", "+ BN folding"),
    ("3-ring", "+ Ring reduction (ring32)"),
    ("4-prelu-fusion", "+ Conv-PReLU fusion"),
    ("5-se-fusion", "+ SE-conv fusion"),
    ("6-add-fusion", "+ Residual-Add fusion"),
    ("7-two-summand", "+ Two-summand rounding"),
)


def _ablation_step(results: Path, directory: str | None) -> tuple | None:
    """Online (rounds, wall, bytes) and offline bytes of one ladder step.

    Every step is read the same way, through MP-SPDZ's own split of the
    request: its online phase is the online column, and the offline column is
    the preprocessing the request generated -- inside it (the naive baseline's
    daBits) plus, where there is one, the daBit prefetch that runs before it.
    """
    path = (results / "naive" / "report.json" if directory is None
            else results / "ablation" / directory / "report.json")
    report = _load(path)
    if report is None:
        return None
    offline = (_median(report, "end_to_end", "offline", "bytes_total") or 0.0)
    offline += _median(report, "offline_dabits", "total", "bytes_total") or 0.0
    return (_median(report, "end_to_end", "online", "rounds"),
            _median(report, "end_to_end", "online", "wall_seconds"),
            _median(report, "end_to_end", "online", "bytes_total"),
            offline)


def _change(new: float | None, old: float | None) -> str:
    if new is None or old is None or not old:
        return "--"
    return f"{100 * (new - old) / old:+.1f}%"


def ablation_table(results: Path) -> list[str]:
    steps = [(label, _ablation_step(results, directory))
             for directory, label in ABLATION]
    if all(step is None for _, step in steps):
        return []
    first = next((step for _, step in steps if step is not None), None)
    last = next((step for _, step in reversed(steps) if step is not None), None)
    total_saved = (first[2] - last[2]) if first and last else None
    lines = ["### Ablation (HbC, co-host, one verification request)", "",
             "| Step | Online rounds | Δ | Online bandwidth | Δ | Share of the "
             "byte saving | Online wall (s) | Δ | Offline bandwidth |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    previous = None
    for label, step in steps:
        if step is None:
            lines.append(f"| {label} | " + " | ".join(["--"] * 8) + " |")
            previous = None
            continue
        rounds, wall, size, offline = step
        share = ("--" if previous is None or not total_saved
                 else f"{100 * (previous[2] - size) / total_saved:.1f}%")
        lines.append(
            f"| {label} | {_rounds(rounds)} "
            f"| {_change(rounds, previous and previous[0])} "
            f"| {_bandwidth(size)} | {_change(size, previous and previous[2])} "
            f"| {share} | {_seconds(wall)} "
            f"| {_change(wall, previous and previous[1])} "
            f"| {_bandwidth(offline)} |")
        previous = step
    return lines + [""]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    lines = (summary_table(args.results)
             + benchmark_table(args.results, "verification",
                               "Face verification (one request)")
             + benchmark_table(args.results, "identification",
                               "Face identification, 1:128 closed set (one request)")
             + ablation_table(args.results))
    text = "\n".join(lines).rstrip() + "\n"
    print(text)
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "tables.md").write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

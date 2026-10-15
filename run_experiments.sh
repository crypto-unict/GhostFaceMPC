#!/usr/bin/env bash
# Reproduce the experiments of the GhostFaceMPC paper.
#
#   ./run_experiments.sh [--quick] TARGET...
#
# Targets:
#   accuracy        five-dataset verification accuracy, float teacher and
#                   quantized model (ring oracle)
#   verification    one verification request (embedding + 1:1 score),
#                   semi-honest (HbC) and malicious (M), on every network
#   identification  one 1:128 closed-set identification request, same grid
#   plaintext       TensorFlow float32 baseline
#   naive           naive MP-SPDZ baseline (ring64, f = 20)
#   porthos         EzPC/Porthos baseline (builds it first if needed)
#   baselines       plaintext + naive + porthos
#   ablation        the optimizations one at a time, from the naive baseline
#                   to GhostFaceMPC (HbC, co-host), each step checked first
#   tables          print the paper's tables from what has been measured
#   all             everything above, then the tables
#
# --quick runs one pool of one warm-up and one measured request everywhere,
# and the accuracy on the first 600 pairs of each dataset: a smoke test of the
# whole pipeline.  Bytes and rounds are exact even then; wall times are not
# medians.
#
# Environment (all optional):
#   RESULTS    output directory (default results)
#   THREADS    threads per party (default 8, as in the paper)
#   NETWORKS   networks to measure (default "co-host lan fast-wan wan")
#   WORKERS    processes for the accuracy oracle (default: number of CPUs)
#   POOLS, WARMUPS, TRIALS   override the pool design of every MPC run
#   PORT_BASE  first TCP port of the MP-SPDZ parties (default 14000)
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$REPO"

PYTHON=${PYTHON:-$REPO/.venv/bin/python}
GHOSTMPC=${GHOSTMPC:-$REPO/.venv/bin/ghostmpc}
MPSPDZ=${MPSPDZ:-third_party/MP-SPDZ}
MANIFEST=${MANIFEST:-artifacts/model/manifest.json}
RESULTS=${RESULTS:-results}
THREADS=${THREADS:-8}
NETWORKS=${NETWORKS:-co-host lan fast-wan wan}
WORKERS=${WORKERS:-$(nproc)}
PORT_BASE=${PORT_BASE:-14000}
NETSIM=$REPO/scripts/netsim/ghostmpc-netsim

QUICK=0
TARGETS=()
for argument in "$@"; do
    case $argument in
        --quick) QUICK=1 ;;
        -h|--help) sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        accuracy|verification|identification|plaintext|naive|porthos|baselines|ablation|tables|all)
            TARGETS+=("$argument") ;;
        *) echo "unknown target: $argument (see --help)" >&2; exit 2 ;;
    esac
done
[ ${#TARGETS[@]} -gt 0 ] || { sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

log() { printf '\n==> [%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[ -x "$PYTHON" ] || die "no Python at $PYTHON; run scripts/setup.sh first"

# The paper's pool designs, as (pools warm-ups trials); --quick and the
# POOLS/WARMUPS/TRIALS variables override them.
design() {
    local pools=$1 warmups=$2 trials=$3
    [ "$QUICK" = 1 ] && { pools=1; warmups=1; trials=1; }
    echo "${POOLS:-$pools} ${WARMUPS:-$warmups} ${TRIALS:-$trials}"
}

require_mpspdz() {
    for binary in replicated-ring-party.x ps-rep-ring-party.x; do
        [ -x "$MPSPDZ/$binary" ] || die "missing $MPSPDZ/$binary; run scripts/setup.sh"
    done
    [ -f "$MPSPDZ/Player-Data/P0.pem" ] || die "missing TLS material; run scripts/setup.sh"
    [ -f "$MANIFEST" ] || die "missing $MANIFEST; run 'ghostmpc fetch-bundles'"
}

# One `ghostmpc benchmark` per (task, protocol, network).  The program is
# compiled once per (task, protocol) -- on the first network -- and reused on
# the others, since the network does not enter the program.
benchmark() {
    local task=$1 protocol=$2
    local pools warmups trials
    case "$task/$protocol" in
        */replicated-ring-party)                read -r pools warmups trials <<< "$(design 3 5 30)" ;;
        verification/ps-rep-ring-party)         read -r pools warmups trials <<< "$(design 1 1 4)" ;;
        identification/ps-rep-ring-party)       read -r pools warmups trials <<< "$(design 2 1 2)" ;;
    esac
    local head=(--score --gallery-rows 1024)
    [ "$task" = identification ] && head=(--identify --gallery-rows 128)
    local compiled=0
    for network in $NETWORKS; do
        local output=$RESULTS/$task/$protocol/$network
        if [ -f "$output/report.json" ]; then
            log "$task $protocol $network: kept existing $output/report.json"
            continue
        fi
        rm -rf "$output"
        log "$task $protocol $network: $pools pool(s) x ($warmups warm-up + $trials measured)"
        local command=("$GHOSTMPC" benchmark --manifest "$MANIFEST" --mpspdz "$MPSPDZ"
                       --protocol "$protocol" --threads "$THREADS"
                       --pools "$pools" --warmups "$warmups" --trials "$trials"
                       --port-base "$PORT_BASE" --network-profile "$network"
                       --output "$output" "${head[@]}")
        [ "$compiled" = 1 ] && command+=(--skip-compile)
        if [ "$network" = co-host ]; then
            "${command[@]}"
        else
            "$NETSIM" "$network" -- "${command[@]}"
        fi
        compiled=1
    done
}

run_accuracy() {
    log "accuracy: float teacher and quantized model on the five datasets ($WORKERS workers)"
    local extra=()
    [ "$QUICK" = 1 ] && extra=(--pairs 600)
    "$GHOSTMPC" evaluate-accuracy --manifest "$MANIFEST" \
        --checkpoint artifacts/GhostFaceNet_W1.3_S1_ArcFace.h5 \
        --dataset-directory datasets/benchmarks --workers "$WORKERS" \
        --output "$RESULTS/accuracy" "${extra[@]}"
}

run_plaintext() {
    local pools warmups trials
    read -r pools warmups trials <<< "$(design 3 5 30)"
    log "plaintext baseline: $pools pool(s) x ($warmups + $trials)"
    "$PYTHON" baselines/plaintext/run.py --threads "$THREADS" \
        --pools "$pools" --warmups "$warmups" --trials "$trials" \
        --output "$RESULTS/plaintext"
}

run_naive() {
    require_mpspdz
    local pools warmups trials
    read -r pools warmups trials <<< "$(design 3 2 5)"
    log "naive MP-SPDZ baseline: $pools pool(s) x ($warmups + $trials)"
    "$PYTHON" baselines/naive-mpspdz/run.py --mpspdz "$MPSPDZ" --threads "$THREADS" \
        --pools "$pools" --warmups "$warmups" --trials "$trials" \
        --port-base "$PORT_BASE" --output "$RESULTS/naive"
}

run_porthos() {
    local pools warmups trials
    read -r pools warmups trials <<< "$(design 3 2 5)"
    if [ ! -x build/porthos/bin/ghostfacenet-porthos-score.out ]; then
        log "building the Porthos baseline"
        PYTHON=$PYTHON GHOSTMPC=$GHOSTMPC baselines/porthos/build.sh
    fi
    log "Porthos baseline: $pools pool(s) x ($warmups + $trials)"
    rm -rf "$RESULTS/porthos"
    PYTHON=$PYTHON POOLS=$pools WARMUPS=$warmups TRIALS=$trials THREADS=$THREADS \
        baselines/porthos/run.sh "$RESULTS/porthos"
}

run_ablation() {
    require_mpspdz
    log "ablation: naive baseline to GhostFaceMPC, one optimization per step"
    local extra=()
    [ "$QUICK" = 1 ] && extra+=(--quick)
    [ -n "${POOLS:-}" ] && extra+=(--pools "$POOLS")
    [ -n "${WARMUPS:-}" ] && extra+=(--warmups "$WARMUPS")
    [ -n "${TRIALS:-}" ] && extra+=(--trials "$TRIALS")
    PYTHON=$PYTHON GHOSTMPC=$GHOSTMPC "$PYTHON" scripts/run_ablation.py \
        --results "$RESULTS" --mpspdz "$MPSPDZ" --threads "$THREADS" \
        --port-base "$PORT_BASE" "${extra[@]}"
}

run_tables() {
    "$PYTHON" scripts/collect_tables.py --results "$RESULTS"
}

for target in "${TARGETS[@]}"; do
    case $target in
        accuracy) run_accuracy ;;
        verification|identification)
            require_mpspdz
            benchmark "$target" replicated-ring-party
            benchmark "$target" ps-rep-ring-party ;;
        plaintext) run_plaintext ;;
        naive) run_naive ;;
        porthos) run_porthos ;;
        baselines) run_plaintext; run_naive; run_porthos ;;
        ablation) run_ablation ;;
        tables) run_tables ;;
        all)
            run_accuracy
            require_mpspdz
            for task in verification identification; do
                benchmark "$task" replicated-ring-party
                benchmark "$task" ps-rep-ring-party
            done
            run_plaintext; run_naive; run_porthos
            run_ablation
            run_tables ;;
    esac
done

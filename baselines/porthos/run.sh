#!/usr/bin/env bash
# Measure the scored CrypTFlow/Porthos baseline: one verification request with
# the 1:1 score, nearest truncation, three parties on this host over mTLS.
#
#   baselines/porthos/run.sh [OUTPUT_DIR]
#
# For each pool: fresh PRG session keys and TLS identities, the two primaries'
# input streams (public weights, the enrolled gallery row, one input share per
# request), then one persistent three-party process per party serving every
# request of the pool.  The pools are finally summarized into OUTPUT_DIR.
#
# Environment (all optional):
#   OUTPUT     result directory (default results/porthos, or the first argument)
#   POOLS      number of pools (default 3)
#   WARMUPS    discarded warm-up requests per pool (default 2)
#   TRIALS     measured requests per pool (default 5)
#   THREADS    threads per party, bounding Eigen/OpenMP (default 8)
#   BUILD_DIR  directory build.sh wrote (default build/porthos)
#   BUNDLE     exported bundle directory (default artifacts/porthos)
#   PYTHON     Python with the package installed (default .venv/bin/python)
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
HERE=$REPO/baselines/porthos

absolute() { case $1 in /*) printf '%s\n' "$1" ;; *) printf '%s\n' "$REPO/$1" ;; esac; }
OUTPUT=$(absolute "${1:-${OUTPUT:-results/porthos}}")
POOLS=${POOLS:-3}
WARMUPS=${WARMUPS:-2}
TRIALS=${TRIALS:-5}
THREADS=${THREADS:-8}
BUILD_DIR=$(absolute "${BUILD_DIR:-build/porthos}")
BUNDLE=$(absolute "${BUNDLE:-artifacts/porthos}")
PYTHON=${PYTHON:-$REPO/.venv/bin/python}

BINARY=$BUILD_DIR/bin/ghostfacenet-porthos-score.out
SOURCE=$BUILD_DIR/ghostfacenet-porthos-score.cpp
BUILD_MANIFEST=$BUILD_DIR/porthos-build-manifest.json
WEIGHTS=$BUNDLE/ghostfacenet-b0-porthos_input_weights_fixedpt_scale_16.inp
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
for file in "$BINARY" "$BUILD_MANIFEST" "$WEIGHTS" \
        "$BUNDLE"/input-share-p{0,1}.npy "$BUNDLE"/gallery-share-p{0,1}.npy; do
    [ -f "$file" ] || die "missing $file; run baselines/porthos/build.sh first"
done
[ "$POOLS" -ge 1 ] && [ "$WARMUPS" -ge 0 ] && [ "$TRIALS" -ge 1 ] \
    || die "POOLS and TRIALS must be positive, WARMUPS non-negative"
mkdir -p "$OUTPUT"
if compgen -G "$OUTPUT/pool-*.json" >/dev/null; then
    die "$OUTPUT already holds pool reports; pick another OUTPUT or remove them"
fi

# Keys, TLS identities and the input streams live only for the pool they serve.
SESSION=$(mktemp -d "${TMPDIR:-/tmp}/ghostmpc-porthos-run.XXXXXX")
trap 'rm -rf "$SESSION"' EXIT
REQUESTS=$((WARMUPS + TRIALS))
started=$SECONDS

for ((pool = 0; pool < POOLS; pool++)); do
    printf '\n==> [%s] pool %d of %d: %d warm-up + %d measured requests\n' \
        "$(date +%H:%M:%S)" "$((pool + 1))" "$POOLS" "$WARMUPS" "$TRIALS"
    dir=$SESSION/pool-$pool
    mkdir -p "$dir"
    "$PYTHON" "$HERE/generate_session_keys.py" --pool "$pool" --output "$dir/keys" >/dev/null
    "$PYTHON" "$HERE/generate_tls.py" --output "$dir/tls" >/dev/null
    for party in 0 1; do
        shares=()
        for ((request = 0; request < REQUESTS; request++)); do
            shares+=(--share "$BUNDLE/input-share-p$party.npy")
        done
        "$PYTHON" "$HERE/prepare_party_inputs.py" \
            --weights "$WEIGHTS" \
            --gallery-share "$BUNDLE/gallery-share-p$party.npy" \
            "${shares[@]}" --output "$dir/p$party.inp"
    done
    "$PYTHON" "$HERE/run_porthos_mtls.py" \
        --binary "$BINARY" \
        --party0-input "$dir/p0.inp" --party1-input "$dir/p1.inp" \
        --keys "$dir/keys" --tls-directory "$dir/tls" \
        --backend porthos-nearest --score --threads-per-party "$THREADS" \
        --pool "$pool" --warmups "$WARMUPS" --trials "$TRIALS" \
        --build-manifest "$BUILD_MANIFEST" --generated-source "$SOURCE" \
        --output "$OUTPUT/pool-$pool.json"
    rm -rf "$dir"
done

printf '\n==> [%s] summarizing %d pool(s) after %ds\n' \
    "$(date +%H:%M:%S)" "$POOLS" "$((SECONDS - started))"
"$PYTHON" "$HERE/summarize_benchmarks.py" --pools "$OUTPUT" --output "$OUTPUT"

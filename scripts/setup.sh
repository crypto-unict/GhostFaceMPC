#!/usr/bin/env bash
# One-time setup: Python environment, MP-SPDZ, model bundles, dataset check.
#
#   scripts/setup.sh
#
# 1. creates .venv and installs the package with the export, porthos and dev
#    extras;
# 2. checks out the pinned MP-SPDZ, applies the ghostmpc VM extensions, enables
#    the 32-bit ring, builds the two party binaries and the TLS material;
# 3. downloads the float teacher checkpoint and the model bundles;
# 4. checks the five benchmark datasets in datasets/benchmarks, if present.
#
# Environment (all optional):
#   PYTHON   interpreter for the virtual environment (default python3, >= 3.10)
#   JOBS     parallel build jobs (default nproc)
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
PYTHON=${PYTHON:-python3}
JOBS=${JOBS:-$(nproc)}

log() { printf '\n==> %s\n' "$*"; }

log "Python environment (.venv)"
[ -x .venv/bin/python ] || "$PYTHON" -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -e '.[export,porthos,dev]'

log "MP-SPDZ (pinned revision + ghostmpc extensions)"
.venv/bin/ghostmpc bootstrap-mpspdz
(
    cd third_party/MP-SPDZ
    make -j"$JOBS" replicated-ring-party.x ps-rep-ring-party.x
    [ -f Player-Data/P0.pem ] || Scripts/setup-ssl.sh 3 Player-Data
)

log "Teacher checkpoint and model bundles"
.venv/bin/ghostmpc fetch-checkpoint
.venv/bin/ghostmpc fetch-bundles

log "Benchmark datasets"
missing=0
while read -r digest name; do
    path=datasets/benchmarks/$name
    if [ ! -f "$path" ]; then
        echo "    missing $path"
        missing=1
    elif echo "$digest  $path" | sha256sum --check --quiet; then
        echo "    ok      $path"
    else
        echo "    WRONG   $path (sha256 differs)"
        missing=1
    fi
done <<'EOF'
021e94cba56790be77f9813d81e13df2781b154ea9bdaf696b866936be4cd9ca lfw.bin
d9fc3c0ca98b9d98c7b7b66e6b779c4cc6775c1718dd249ff8adfdfcc0061cf5 agedb_30.bin
8a2d8d3cd5119d4d39b78b41f19ad2d1bab51d2a640a4f3743b49280f8076ed3 calfw.bin
ba76af859984c0d3329190a92ee2d8044fbac4203231095f4dd96bf176efe71a cplfw.bin
414ba112448350467dce0f0b628215cc2a43a07e551c13ae68529d79ac4de48e cfp_fp.bin
EOF
if [ "$missing" = 1 ]; then
    echo "    the accuracy and plaintext experiments need these files (see README)"
fi

log "Done. Next: ./run_experiments.sh --quick verification"

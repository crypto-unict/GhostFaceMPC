#!/usr/bin/env bash
# Build the CrypTFlow/Porthos baseline end to end.
#
#   baselines/porthos/build.sh [BUILD_DIR]
#
# Steps: fetch the pinned EzPC commit and Eigen 3.3.7, materialize and patch a
# disposable EzPC tree (running the two build gates), export the scored model
# bundle, build the EzPC compiler, run the Athos ONNX compiler for the PORTHOS
# target, post-process the generated C++ for share input/output and the 1:1
# score, build the Porthos library and link the benchmark binary.
#
# Environment (all optional):
#   BUILD_DIR      build directory (default build/porthos, or the first argument)
#   BUNDLE         exported bundle directory (default artifacts/porthos)
#   MANIFEST       ring64/f16 manifest (default artifacts/model-ring64-f16-b0/manifest.json)
#   CHECKPOINT     Keras checkpoint (default artifacts/GhostFaceNet_W1.3_S1_ArcFace.h5)
#   GALLERY_ROWS   enrolled database size (default 1024)
#   THREADS        declared threads per party, recorded in the build manifest (default 8)
#   EZPC_CACHE     bare git cache of EzPC (default third_party/EzPC-porthos-cache.git)
#   EIGEN_ARCHIVE  Eigen 3.3.7 archive (default third_party/eigen-3.3.7.tar.gz)
#   PYTHON         Python with the export and porthos extras (default .venv/bin/python)
#   GHOSTMPC       ghostmpc command (default .venv/bin/ghostmpc)
#   OPAM_SWITCH    opam switch holding OCaml 4.14 + menhir/ocamlbuild/ocamlfind/stdint
#   CXX            C++ compiler (default g++)
#   JOBS           parallel build jobs (default nproc)
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
HERE=baselines/porthos

BUILD_DIR=${1:-${BUILD_DIR:-build/porthos}}
BUNDLE=${BUNDLE:-artifacts/porthos}
MANIFEST=${MANIFEST:-artifacts/model-ring64-f16-b0/manifest.json}
CHECKPOINT=${CHECKPOINT:-artifacts/GhostFaceNet_W1.3_S1_ArcFace.h5}
GALLERY_ROWS=${GALLERY_ROWS:-1024}
THREADS=${THREADS:-8}
EZPC_CACHE=${EZPC_CACHE:-third_party/EzPC-porthos-cache.git}
EIGEN_ARCHIVE=${EIGEN_ARCHIVE:-third_party/eigen-3.3.7.tar.gz}
PYTHON=${PYTHON:-$REPO/.venv/bin/python}
GHOSTMPC=${GHOSTMPC:-$REPO/.venv/bin/ghostmpc}
CXX=${CXX:-g++}
JOBS=${JOBS:-$(nproc)}
export CXX

# Every path below is made absolute: Athos runs from inside the EzPC tree.
absolute() { case $1 in /*) printf '%s\n' "$1" ;; *) printf '%s\n' "$REPO/$1" ;; esac; }
BUILD_DIR=$(absolute "$BUILD_DIR")
BUNDLE=$(absolute "$BUNDLE")
MANIFEST=$(absolute "$MANIFEST")
CHECKPOINT=$(absolute "$CHECKPOINT")
EZPC_CACHE=$(absolute "$EZPC_CACHE")
EIGEN_ARCHIVE=$(absolute "$EIGEN_ARCHIVE")
case $PYTHON in
    /*) ;;
    */*) PYTHON=$REPO/$PYTHON ;;
    *) PYTHON=$(command -v "$PYTHON") || { echo "error: no $PYTHON on PATH" >&2; exit 1; } ;;
esac

EZPC_URL=$(sed -n 's/.*"repository": *"\([^"]*\)".*/\1/p' "$HERE/ezpc.lock.json")
EZPC_COMMIT=$(sed -n 's/.*"commit": *"\([0-9a-f]*\)".*/\1/p' "$HERE/ezpc.lock.json")
EIGEN_URL=https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz
EIGEN_SHA256=d56fbad95abf993f8af608484729e3d87ef611dd85b3380a8bad1d5cbc373a57

MODEL=ghostfacenet-b0-porthos
TREE=$BUILD_DIR/ezpc
LOGS=$BUILD_DIR/logs
GENERATED=$BUNDLE/${MODEL}_64_porthos0.cpp
WEIGHTS=$BUNDLE/${MODEL}_input_weights_fixedpt_scale_16.inp
SOURCE=$BUILD_DIR/ghostfacenet-porthos-score.cpp
BINARY=$BUILD_DIR/bin/ghostfacenet-porthos-score.out
BUILD_MANIFEST=$BUILD_DIR/porthos-build-manifest.json

# The scripts import the package from this checkout even when PYTHON is not the
# release's own virtual environment.
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

started=$SECONDS
step_started=$SECONDS
step() {
    printf '\n==> [%s] %s\n' "$(date +%H:%M:%S)" "$*"
    step_started=$SECONDS
}
done_step() { printf '    done in %ds\n' "$((SECONDS - step_started))"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing required command: $1 ($2)"; }

step "Checking prerequisites"
need git "git"
need "$CXX" "a C++17 compiler with OpenMP, e.g. apt install g++"
need cmake "apt install cmake"
need make "apt install make"
need openssl "apt install openssl"
need sha256sum "coreutils"
command -v curl >/dev/null 2>&1 || command -v wget >/dev/null 2>&1 \
    || die "missing curl or wget"
[ -x "$PYTHON" ] || command -v "$PYTHON" >/dev/null 2>&1 \
    || die "PYTHON=$PYTHON is not executable; create .venv first"
"$PYTHON" - <<'EOF' || die "PYTHON lacks a required module (pip install -e '.[export,porthos]')"
import importlib.util, sys
missing = [name for name in ("numpy", "onnx", "onnxruntime", "onnxsim", "tensorflow")
           if importlib.util.find_spec(name) is None]
if missing:
    sys.exit("missing Python modules: " + ", ".join(missing))
EOF
[ -f "$MANIFEST" ] || die "missing $MANIFEST (the ring64/f16 manifest)"
[ -f "$CHECKPOINT" ] || die "missing $CHECKPOINT (the Keras checkpoint)"
grep -qw aes /proc/cpuinfo 2>/dev/null || die "Porthos needs a CPU with AES-NI"
# The EzPC compiler is OCaml: take it from an opam switch when the tools are
# not already on PATH.
if ! { command -v ocamlbuild && command -v menhir && ocamlfind query stdint; } \
        >/dev/null 2>&1 && command -v opam >/dev/null 2>&1; then
    eval "$(opam env ${OPAM_SWITCH:+--switch="$OPAM_SWITCH" --set-switch})"
fi
need ocamlbuild "OCaml 4.14 via opam: opam install ocamlbuild"
need menhir "opam install menhir"
need ocamlfind "opam install ocamlfind"
ocamlfind query stdint >/dev/null 2>&1 || die "missing OCaml package stdint (opam install stdint)"
done_step

step "Fetching EzPC $EZPC_COMMIT into $EZPC_CACHE"
if git -C "$EZPC_CACHE" cat-file -e "$EZPC_COMMIT^{commit}" 2>/dev/null; then
    echo "    cached"
else
    [ -d "$EZPC_CACHE" ] || git init --bare --quiet "$EZPC_CACHE"
    git -C "$EZPC_CACHE" fetch --depth 1 "$EZPC_URL" "$EZPC_COMMIT"
    git -C "$EZPC_CACHE" cat-file -e "$EZPC_COMMIT^{commit}"
fi
done_step

step "Fetching Eigen 3.3.7 into $EIGEN_ARCHIVE"
if [ ! -f "$EIGEN_ARCHIVE" ]; then
    mkdir -p "$(dirname "$EIGEN_ARCHIVE")"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL -o "$EIGEN_ARCHIVE.partial" "$EIGEN_URL"
    else
        wget -q -O "$EIGEN_ARCHIVE.partial" "$EIGEN_URL"
    fi
    mv "$EIGEN_ARCHIVE.partial" "$EIGEN_ARCHIVE"
fi
echo "$EIGEN_SHA256  $EIGEN_ARCHIVE" | sha256sum --check --quiet \
    || die "$EIGEN_ARCHIVE does not match sha256 $EIGEN_SHA256"
done_step

step "Materializing and patching the EzPC tree in $TREE"
mkdir -p "$BUILD_DIR" "$LOGS"
if [ -e "$TREE" ]; then
    # Only a tree this script created is removed: prepare_build.py stamps it.
    [ -f "$TREE/porthos.lock.json" ] || die "$TREE exists and was not created by prepare_build.py"
    rm -rf "$TREE"
fi
"$PYTHON" "$HERE/prepare_build.py" \
    --cache "$EZPC_CACHE" --eigen-cache "$EIGEN_ARCHIVE" \
    --output "$TREE" --manifest "$BUILD_MANIFEST" \
    --threads-per-party "$THREADS"
done_step

step "Exporting the scored bundle to $BUNDLE"
# Remove what Athos generated last time, so every file checked below is fresh.
rm -f "$BUNDLE/astOutput.pkl" "$BUNDLE/client.zip" "$BUNDLE/optimised_$MODEL.onnx" \
    "$BUNDLE/${MODEL}_64_porthos.ezpc" "$GENERATED" "$WEIGHTS"
# GHOSTMPC is deliberately unquoted so that it may be a command with arguments.
# shellcheck disable=SC2086
$GHOSTMPC export-porthos \
    --manifest "$MANIFEST" --checkpoint "$CHECKPOINT" \
    --destination "$BUNDLE" --score --gallery-rows "$GALLERY_ROWS" \
    | tee "$LOGS/export.log"
for file in athos-config-porthos.json "$MODEL.onnx" input-share-p0.npy \
        input-share-p1.npy gallery-share-p0.npy gallery-share-p1.npy; do
    [ -f "$BUNDLE/$file" ] || die "the export did not write $BUNDLE/$file"
done
done_step

step "Building the EzPC compiler (log: $LOGS/ezpc-compiler.log)"
make -C "$TREE/EzPC/EzPC" > "$LOGS/ezpc-compiler.log" 2>&1 \
    || { tail -30 "$LOGS/ezpc-compiler.log"; die "the EzPC compiler did not build"; }
[ -x "$TREE/EzPC/EzPC/ezpc" ] || die "the EzPC compiler did not produce ezpc"
done_step

step "Compiling the ONNX graph with Athos for PORTHOS (log: $LOGS/athos-*.log)"
# Two Athos passes over the same config.  The server pass simplifies the graph,
# dumps the fixed-point weight stream and writes the weight-stripped model, in
# which every weight is a graph input tagged MPC_MODEL_WEIGHTS.  The client pass
# compiles that stripped model; the patched frontend reads those inputs as
# public (int64_pl), which the server pass cannot do because onnxsim drops the
# initializers' doc strings.  Athos runs SeeDot through whatever `python3` is
# on PATH, so this PYTHON goes first; PYTHONHASHSEED fixes the order in which
# its liveness pass emits the ClearMem* calls.  The Porthos library is not
# built yet, so Athos stops after generating C++; the binary is linked below
# from the post-processed source.
SHIMS=$BUILD_DIR/shims
mkdir -p "$SHIMS"
printf '#!/bin/sh\nexec "%s" "$@"\n' "$PYTHON" > "$SHIMS/python3"
chmod +x "$SHIMS/python3"
for role in server client; do
    (
        cd "$TREE/Athos"
        PATH="$SHIMS:$PATH" PYTHONHASHSEED=0 "$PYTHON" CompileONNXGraph.py \
            --role "$role" --config "$BUNDLE/athos-config-porthos.json"
    ) > "$LOGS/athos-$role.log" 2>&1 \
        || { tail -30 "$LOGS/athos-$role.log"; die "Athos ($role) failed"; }
    # CompileONNXGraph.py does not propagate the exit status of its sub-steps.
    [ -s "$GENERATED" ] || { tail -30 "$LOGS/athos-$role.log"; die "Athos ($role) did not generate $GENERATED"; }
    if [ "$role" = server ]; then
        [ -s "$WEIGHTS" ] || die "Athos did not write $WEIGHTS"
        [ -s "$BUNDLE/optimised_$MODEL.onnx" ] || die "Athos did not write the stripped model"
        rm -f "$GENERATED" "$BUNDLE/${MODEL}_64_porthos.ezpc"
    fi
done
grep -q 'ClearMemPublic' "$GENERATED" \
    || die "$GENERATED reads the weights as secret inputs; the public-weight patch did not apply"
done_step

step "Post-processing the generated C++ into $SOURCE"
"$PYTHON" "$HERE/postprocess_generated.py" \
    --cpp "$GENERATED" --score --output "$SOURCE"
done_step

step "Building the Porthos library (log: $LOGS/porthos-library.log)"
{
    cmake -S "$TREE/Porthos/src" -B "$TREE/Porthos/src/build" \
        -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    cmake --build "$TREE/Porthos/src/build" --target Porthos-Protocols \
        --parallel "$JOBS"
} > "$LOGS/porthos-library.log" 2>&1 \
    || { tail -30 "$LOGS/porthos-library.log"; die "the Porthos library did not build"; }
done_step

step "Linking $BINARY (log: $LOGS/binary.log)"
mkdir -p "$(dirname "$BINARY")"
# The flags are the ones Athos/CompileONNXGraph.py uses for the PORTHOS target.
LINK=("$CXX" -O3 -fopenmp -pthread -w -march=native -msse4.1 -maes -mpclmul
      -mrdseed -fpermissive -fpic -std=c++17
      -L "$TREE/Porthos/src/build/lib" -I "$TREE/Porthos/src" "$SOURCE"
      -lPorthos-Protocols -lssl -lcrypto -lrt -lboost_system -o "$BINARY")
"${LINK[@]}" > "$LOGS/binary.log" 2>&1 \
    || { tail -30 "$LOGS/binary.log"; die "the Porthos binary did not link"; }
done_step

step "Recording the build in $BUILD_MANIFEST"
"$PYTHON" - "$BUILD_MANIFEST" "$BINARY" "$SOURCE" "$GENERATED" "$WEIGHTS" \
        "$BUNDLE/config.json" "${LINK[*]}" <<'EOF'
import json, sys
from hashlib import sha256
from pathlib import Path

manifest, binary, source, generated, weights, config, link = sys.argv[1:]

def digest(path):
    value = sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()

data = json.loads(Path(manifest).read_text())
data["build"] = {
    "athos": "cd <tree>/Athos && for role in server client; do "
             "PYTHONHASHSEED=0 python3 CompileONNXGraph.py --role $role "
             "--config <bundle>/athos-config-porthos.json; done",
    "postprocess": "python baselines/porthos/postprocess_generated.py "
                   "--cpp <generated> --score --output <source>",
    "library": "cmake --build <tree>/Porthos/src/build --target Porthos-Protocols",
    "link": link,
    "artifacts": {
        name: {"path": path, "sha256": digest(path)}
        for name, path in (("binary", binary), ("source", source),
                           ("generated", generated), ("weights", weights),
                           ("bundle_config", config))
    },
    "score": True,
}
Path(manifest).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
EOF
done_step

printf '\nBuilt %s in %ds.\n' "$BINARY" "$((SECONDS - started))"
printf 'Build manifest: %s\nNext: baselines/porthos/run.sh\n' "$BUILD_MANIFEST"

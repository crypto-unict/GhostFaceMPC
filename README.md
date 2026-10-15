# GhostFaceMPC: Privacy-Preserving Face Recognition with Multi-Party Computation

Official implementation of *GhostFaceMPC: Privacy-Preserving Face Recognition
with Multi-Party Computation* (Davide Carnemolla, Dario Catalano, Mario Di
Raimondo — University of Catania). Paper: *ePrint link coming soon*.

GhostFaceMPC runs face recognition with **GhostFaceNetV1** (width 1.3,
stride 1, ArcFace head, trained on MS1MV3) among three servers with
**replicated secret sharing** in [MP-SPDZ](https://github.com/data61/MP-SPDZ).

- The client secret-shares the aligned 112×112 face: it produces the
  replicated sharing itself, so the servers load their pairs without
  re-sharing anything. No server ever sees the image, an activation, the
  embedding or the enrolled database.
- The model parameters are public, so every linear layer is local
  secret-by-public arithmetic with one fixed-point reduction per output.
- The model is quantized to a 32-bit ring (6 fractional bits, 9 for the
  kernels) by quantization-aware training on the exact arithmetic the servers
  execute.
- It supports two security profiles: semi-honest (`replicated-ring-party`) and
  malicious with abort (`ps-rep-ring-party`).
- Each request is split into an **offline** phase and an **online** phase. The
  offline phase generates the input-independent preprocessing. The online phase
  is the inference itself, followed by a 1:1 verification score or a 1:N
  closed-set identification. Neither phase opens anything.

This repository contains exactly the code needed to reproduce the experiments
of the paper.

```bibtex
@misc{ghostfacempc,
  title  = {{GhostFaceMPC}: Privacy-Preserving Face Recognition with Multi-Party Computation},
  author = {Carnemolla, Davide and Catalano, Dario and Di Raimondo, Mario},
  note   = {Cryptology ePrint Archive, to appear},
  year   = {2026}
}
```

## Contents

| Path | What it is |
|------|------------|
| `src/ghostmpc/` | Python package and `ghostmpc` command line |
| `src/ghostmpc/codegen/` | compiler from the model manifest to an MP-SPDZ program |
| `src/ghostmpc/eval/` | clear-text ring oracle (bit-exact with the MPC program) and accuracy evaluation |
| `src/ghostmpc/benchmark/` | three-party benchmark harness and emulated networks |
| `src/ghostmpc/training/` | quantization-aware training on the ring-faithful forward pass |
| `third_party/patches/` | the virtual-machine instructions this work adds to MP-SPDZ |
| `baselines/` | the baselines: TensorFlow plaintext, naive MP-SPDZ, EzPC/Porthos |
| `run_experiments.sh` | one script that runs every experiment of the paper |
| `scripts/` | setup, rootless network emulation, table collection |
| `tests/` | unit tests (`pytest`) |

## 1. Requirements

- **Machine.** Linux on x86-64. The paper used a 32-core VM (Intel Xeon Gold
  6238R) and runs 8 threads per party. All three servers run on one host.
- **Python.** Python ≥ 3.10 (tested with 3.11).
- **MP-SPDZ build dependencies**
  ([upstream list](https://github.com/data61/MP-SPDZ#requirements)):

  ```bash
  sudo apt-get install automake build-essential clang cmake git libboost-dev \
      libboost-filesystem-dev libboost-iostreams-dev libboost-thread-dev \
      libgmp-dev libntl-dev libsodium-dev libssl-dev libtool python3 python3-venv
  ```

- **LAN/WAN emulation.** Needs `iproute2` (`ip`, `tc`), `iputils-ping`,
  `unshare` (util-linux) and unprivileged user namespaces. The script needs no
  root. On Debian/Ubuntu: `sudo sysctl kernel.unprivileged_userns_clone=1`.
- **Porthos baseline only.**
  - `libboost-system-dev`
  - a CPU with AES-NI
  - OCaml 4.14 via [opam](https://opam.ocaml.org/), with
    `opam install ocamlbuild menhir ocamlfind stdint`

## 2. Setup

```bash
git clone --recursive https://github.com/crypto-unict/ghostfacempc.git
cd ghostfacempc
scripts/setup.sh
```

`scripts/setup.sh` does four things:

1. Creates `.venv` and installs the package. This pulls `tensorflow-cpu`,
   `Pillow`, `onnx`, `onnxruntime` and `onnxsim`.
2. Prepares MP-SPDZ in `third_party/MP-SPDZ`:
   - checks out the pinned revision;
   - applies `third_party/patches/`;
   - enables the 32-bit ring (`MY_CFLAGS += -DRING_SIZE=32` in `CONFIG.mine`);
   - builds `replicated-ring-party.x` and `ps-rep-ring-party.x`;
   - generates the TLS certificates of the three parties.
3. Downloads the model files (see the table below).
4. Checks the benchmark datasets.

The model files land in `artifacts/`:

| File | Content |
|------|---------|
| `GhostFaceNet_W1.3_S1_ArcFace.h5` | float teacher, from the [GhostFaceNets](https://github.com/HamadYA/GhostFaceNets) release |
| `model/` | the quantized ring32/f6/f_w9 model, with its accuracy reports |
| `model-ring64-f16-b0/` | ring64 manifest the Porthos baseline is exported from |
| `legacy-scalar-236/` | ring64/f20 manifest of the naive MP-SPDZ baseline |

The bundles are release assets of this repository, and every archive is
checked against its SHA-256. `ghostmpc fetch-bundles --bundle qat-start` also
downloads the manifest QAT starts from (see §5).

### Datasets

The accuracy experiment and the plaintext baseline read five InsightFace
evaluation sets: LFW, AgeDB-30, CALFW, CPLFW and CFP-FP. Use the `.bin` files
distributed with the MS1MV3 training data of the
[InsightFace dataset zoo](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_)
and place them as:

```
datasets/benchmarks/{lfw,agedb_30,calfw,cplfw,cfp_fp}.bin
```

`scripts/setup.sh` verifies them against the SHA-256 digests we used. The MPC
benchmarks do not need the datasets: their communication does not depend on
the input values, so they secret-share synthetic inputs.

## 3. Reproducing the experiments

```bash
./run_experiments.sh --quick verification   # smoke test, about one hour
./run_experiments.sh all                    # every table of the paper
./run_experiments.sh tables                 # print the tables measured so far
```

Results go to `results/` (one `report.json` and `summary.csv` per run), and
`tables` writes `results/tables.md` in the layout of the paper. Targets can be
run separately and in any order. A run whose report already exists is skipped.

| Target | Paper | Runtime on our machine (approx.) |
|--------|-------|----------------------------------|
| `accuracy` | comparison table, accuracy columns | 3 h per dataset per core; split with `WORKERS` |
| `verification` | Face Verification Benchmarks: HbC and M, Co-Host / LAN / Fast-WAN / WAN | ~6 h |
| `identification` | Face Identification Benchmarks: 1:128 closed set, same grid | ~6 h |
| `plaintext` | comparison table, TensorFlow row | 1 min |
| `naive` | comparison table, MP-SPDZ (naive) row | ~1 h including compilation |
| `porthos` | comparison table, EzPC (Porthos) row | ~5 min build + ~1.5 h |
| `baselines` | the three baselines | |
| `ablation` | ablation table: the optimizations one at a time, from the naive baseline to GhostFaceMPC (HbC, co-host) | ~4 h |

**Environment variables:**

| Variable | Meaning | Default |
|----------|---------|---------|
| `THREADS` | threads per party | 8 |
| `NETWORKS` | networks to measure | `co-host lan fast-wan wan` |
| `WORKERS` | processes for the accuracy oracle | number of CPUs |
| `POOLS`, `WARMUPS`, `TRIALS` | override the pool design | the paper's |
| `RESULTS` | output directory | `results` |

**`--quick`** runs one pool of one warm-up and one measured request, and the
accuracy on the first 600 pairs of each dataset.

### What is measured

- **Pool designs**, as in the paper:
  - semi-honest rows: three pools of five discarded warm-ups and thirty
    measured requests;
  - malicious rows: one pool of 1 + 4 for verification, two pools of 1 + 2 for
    identification;
  - naive and Porthos baselines: three pools of 2 + 5.
- **Bandwidth** is the total sent by the three servers. **Rounds** are
  MP-SPDZ's (for Porthos, the runtime serial depth). **Wall time** is the
  median over the measured requests.
- **Online and offline** phases are reported separately and never summed.
- **Networks** (`lan`, `fast-wan`, `wan`) are emulated with `tc netem`, with
  each party in its own network namespace:

  | Network | RTT | Bandwidth |
  |---------|-----|-----------|
  | LAN | 0.1 ms | 1 Gbit/s |
  | Fast-WAN | 10 ms | 1 Gbit/s |
  | WAN | 100 ms | 100 Mbit/s |

  `scripts/netsim/ghostmpc-netsim <profile> -- <command>` runs any command on
  them.
- **Verification** is one embedding plus the 1:1 score against one row of a
  1,024-row secret-shared database. **Identification** ranks the probe against
  all 128 rows and returns the share of the rank-1 index.

### Expected results

Bytes and rounds are deterministic properties of the compiled programs. The
programs this repository emits are byte-identical to the ones measured in the
paper (`tests/test_emitted_program.py`), so with the paper's pool designs
bandwidth and rounds reproduce exactly.

**Client-side replicated input.** The client (and, for the database, the
dealer) splits each value into `x = x_0 + x_1 + x_2` and hands server `i` the
pair `(x_i, x_{i-1})` in `Persistence/Transactions-P<i>.data`, which the
program loads with `sint.read_from_file`: the servers exchange nothing to
ingest an input. **Malicious profile: verified output.** The input needs no check: with an
honest client the two honest servers hold all three components between them,
so a corrupt server that alters its copies only deviates in what it sends,
which `ps-rep-ring` catches (post-sacrifice check of the products, hashed
openings). The claimed identity is read from
`Persistence/Transactions-clear-P<i>.data`. Before the answer leaves, the
programs force the post-sacrifice check and every server appends its share
pair to `Persistence/`, which `ghostmpc open-result --last N` checks component
by component; this is the default of `ps-rep-ring-party`
(`tests/test_emitted_program.py` pins the default programs). The client side
is `ghostmpc share-gallery <dir>` followed by
`ghostmpc share-image face.png <dir> --append [--identity <row>]`. The pool design enters the program (it
fixes how much preprocessing each request generates ahead), so `--quick` runs
deviate slightly: by a few hundred offline bytes in the semi-honest profile, and
by about 2% of the online bytes in the malicious one. Wall times depend on the
host.

**Online phase of one request:**

| Request | Security | Online rounds | Online bandwidth |
|---------|----------|--------------:|-----------------:|
| Verification | HbC | 1,433 | 312.35 MB |
| Verification | M | 1,391 | 883.27 MB |
| Identification | HbC | 1,496 | 312.35 MB |
| Identification | M | 1,454 | 888.54 MB |

**Offline phase of one request:**

| Security | Offline rounds | Offline bandwidth |
|----------|---------------:|------------------:|
| HbC | 14 | 168.76 MB |
| M | 380 | 5.50 GB |

**Wall time of one verification request, in seconds (paper):**

| Security | Co-Host | LAN | Fast-WAN | WAN |
|----------|--------:|----:|---------:|----:|
| HbC | 2.43 | 3.82 | 10.76 | 87.93 |
| M | 5.53 | 8.14 | 16.72 | 124.68 |

**Accuracy, in percent:**

| Model | LFW | AgeDB-30 | CALFW | CPLFW | CFP-FP |
|-------|----:|---------:|------:|------:|-------:|
| Float teacher | 99.72 | 97.80 | 95.80 | 91.92 | 96.39 |
| GhostFaceMPC (ring32) | 99.68 | 97.67 | 95.85 | 91.60 | 96.37 |

The quantized accuracy is computed on the ring oracle
(`ghostmpc evaluate-accuracy`). The oracle reproduces, integer for integer,
the embedding the three servers compute. It is deterministic, so it matches
the reports shipped in `artifacts/model/gates/` exactly. The float teacher runs
in TensorFlow, whose numerics vary slightly across builds and CPUs, so its row
can move by a pair or two (e.g. 99.70 instead of 99.72 on LFW).

**Baselines (online phase of one verification request, co-host):**

| Baseline | Rounds | Wall | Bandwidth |
|----------|-------:|-----:|----------:|
| TensorFlow (plaintext) | -- | 0.16 s | -- |
| MP-SPDZ naive (ring64) | 42,784 | 44.45 s | 1.05 GB |
| EzPC/Porthos (ring64) | 2,152 | 259 s | 17.97 GB |

## 4. Using the command line directly

```bash
.venv/bin/ghostmpc --help
# emit the MPC program of one verification request
.venv/bin/ghostmpc generate artifacts/model/manifest.json verify.mpc --score --gallery-rows 1024
# measure it (semi-honest, co-host)
.venv/bin/ghostmpc benchmark --protocol replicated-ring-party --score --gallery-rows 1024
# ... or on the emulated WAN, malicious profile
scripts/netsim/ghostmpc-netsim wan -- .venv/bin/ghostmpc benchmark \
    --protocol ps-rep-ring-party --score --gallery-rows 1024 --network-profile wan
# the match decision instead of the score: the client receives one bit
.venv/bin/ghostmpc benchmark --score --match --threshold 0.28
# client and dealer sides: secret-share an aligned face and an enrolled database
.venv/bin/ghostmpc share-gallery shares/ --embeddings enrolled.npy
.venv/bin/ghostmpc share-image face.png shares/ --append --identity 0
# shares/Transactions-P<i>.data goes to server i's Persistence/; the client
# reconstructs the answer the servers append to it
.venv/bin/ghostmpc open-result P0.data P1.data P2.data --last 1
```

## 5. Training the quantized model (optional)

The quantized model in `artifacts/model` was obtained by quantization-aware
fine-tuning. The teacher is distilled into a student whose forward pass
reproduces the ring arithmetic of the MPC program operation by operation, with
straight-through estimators. Retraining needs the MS1MV3 training set as
WebDataset shards
([`gaunernst/ms1mv3-wds`](https://huggingface.co/datasets/gaunernst/ms1mv3-wds))
and a GPU. The published run took 9.6 h on one V100.

```bash
# 1. validation pairs from the MS1MV3 validation split (identity-disjoint)
.venv/bin/ghostmpc prepare-ms1mv3 --shards datasets/ms1mv3/shards \
    --val-pairs datasets/ms1mv3/val_pairs.bin
# 2. starting manifest: the teacher on the ring32/f6/f_w9 profile, with overflow
#    bounds calibrated on 256 + 256 validation faces
#    (or: ghostmpc fetch-bundles --bundle qat-start)
.venv/bin/ghostmpc export artifacts/GhostFaceNet_W1.3_S1_ArcFace.h5 artifacts/qat-start
.venv/bin/ghostmpc calibrate-bounds --manifest artifacts/qat-start/manifest.json \
    --images datasets/ms1mv3/val_pairs.bin --count 256 --validation-count 256 \
    --margin-bits 2
# 3. quantization-aware training (2 epochs, 72,784 steps, batch 128)
.venv/bin/ghostmpc qat-ring --manifest artifacts/qat-start/manifest.json \
    --teacher artifacts/GhostFaceNet_W1.3_S1_ArcFace.h5 \
    --shards datasets/ms1mv3/shards --val-pairs datasets/ms1mv3/val_pairs.bin \
    --output artifacts/qat-run
# 4. accuracy of the trained model
.venv/bin/ghostmpc evaluate-accuracy --manifest artifacts/qat-run/final/manifest.json
```

The training and calibration tools refuse to read `datasets/benchmarks`, so the
model is never fitted on the data that evaluates it.

## 6. Tests

```bash
.venv/bin/pytest
```

The tests are pure Python and need no MPC runtime. Some of them read the model
bundle and skip when it has not been downloaded. Among them:

- `test_emitted_program.py` checks that the emitted programs are the ones the
  paper measured;
- `test_qat_ring.py` checks that the training forward pass is bit-exact with
  the ring oracle on the deployed model.

## Acknowledgements

This work builds on:

- [MP-SPDZ](https://github.com/data61/MP-SPDZ) (the MPC framework);
- [GhostFaceNets](https://github.com/HamadYA/GhostFaceNets) (the model and its
  checkpoint);
- [EzPC](https://github.com/mpc-msri/EzPC) (the Porthos baseline);
- [InsightFace](https://github.com/deepinsight/insightface) and
  [AdaFace](https://github.com/mk-minchul/AdaFace) (datasets and evaluation
  protocol).

This work was funded in part by the PIAno di inCEntivi per la Ricerca di
Ateneo 2024/2026, linea di intervento 1, University of Catania.

## License

MIT, see [LICENSE](LICENSE). MP-SPDZ, EzPC and the model checkpoint are
distributed under their own licenses.

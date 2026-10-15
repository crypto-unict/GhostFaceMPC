"""The emitted programs are the ones the paper measured.

Bytes and rounds are deterministic properties of the compiled program, so a
release that emits byte-identical sources reproduces them exactly.  The digests
below are those of the MP-SPDZ sources the paper's tables were measured on, at
the paper's pool designs (8 threads per party).  No MPC runtime is involved.

Every program loads the client's replicated sharing as it is (PLAN 23), and
the malicious ones are pinned with their default verified delivery.

Skipped when the model bundle has not been fetched (``ghostmpc fetch-bundles``).
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ghostmpc.codegen import generate
from ghostmpc.core.identify import deployment_identify_plan
from ghostmpc.core.match import deployment_match_plan
from ghostmpc.core.score import deployment_score_plan

MANIFEST = REPO / "artifacts" / "model" / "manifest.json"

#: (label, protocol, warm-ups, measured requests, head, gallery rows, sha256)
PROGRAMS = (
    ('hbc-score1024-5+30', 'replicated-ring-party', 5, 30, 'score', 1024,
     '7055d288e0314c7eef333e14fac6f2423ecd2eaaf961fe3984e3c5c5be6d674b'),
    ('mal-score1024-1+4', 'ps-rep-ring-party', 1, 4, 'score', 1024,
     'd3eee2fada7d10d804e375ec78cf6b493f4a21d00f08536822665aae875925c5'),
    ('hbc-identify128-5+30', 'replicated-ring-party', 5, 30, 'identify', 128,
     '5b33aa1160e46257da2d7b445be9d10174852c1f0aaeba9a2b78183fe0bfa1ae'),
    ('mal-identify128-1+2', 'ps-rep-ring-party', 1, 2, 'identify', 128,
     '80a546e9fca22419325dde3d85069f5c81dbe0322316f41e10db2fddee914e90'),
    ('hbc-match1024-5+30', 'replicated-ring-party', 5, 30, 'match', 1024,
     '7dd1808dc8ddd0803116c0183b4f375854f0487975de68ef45c0b5f21cc2545d'),
    ('mal-match1024-1+4', 'ps-rep-ring-party', 1, 4, 'match', 1024,
     '7c8f4ba87377a4ed19f38665c273f708e22964ff104f755098646626932ad4d5'),
)

#: The public-weight stream every program above reads.
WEIGHTS_SHA256 = '397b1eb2237032d98979f7c87f73d841f026edffc8cb168816a9841b5aa9e4ef'


@unittest.skipUnless(MANIFEST.is_file(), "model bundle not fetched")
class EmittedProgramTests(unittest.TestCase):
    def test_programs_are_byte_identical_to_the_measured_ones(self) -> None:
        with tempfile.TemporaryDirectory() as work:
            for label, protocol, warmups, trials, head, rows, expected in PROGRAMS:
                with self.subTest(label=label):
                    score = deployment_score_plan(rows, 512)
                    match = (deployment_match_plan(score, 0.28)
                             if head == "match" else None)
                    identify = (deployment_identify_plan(score)
                                if head == "identify" else None)
                    target = Path(work) / f"{label}.mpc"
                    weights = generate(
                        MANIFEST, target, threads=8, requests=warmups + trials,
                        warmups=warmups, measured_timer=True, score=score,
                        match=match, identify=identify, protocol=protocol)
                    self.assertEqual(
                        hashlib.sha256(target.read_bytes()).hexdigest(), expected)
                    self.assertEqual(
                        hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
                        WEIGHTS_SHA256)


if __name__ == "__main__":
    unittest.main()

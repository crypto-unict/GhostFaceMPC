"""The 1:1 verification score.

The score is two inner products and no truncation, which is exactly why it needs
tests: everything that could go wrong is silent. A gallery grid stated in the
dealer and read differently in the emitter still runs; an accumulator one bit
too wide still runs and returns a wrapped integer; a gallery written after the
per-request payloads still runs and scores the wrong values. None of it raises.

So these tests hold the properties nothing else would catch: the declared
widths fit the ring, the emitter is invisible when no score is compiled in, and
the file the parties read is ordered the way the program reads it.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from ghostmpc.benchmark.process import prepare_player_data
from ghostmpc.codegen import score as score_codegen
from ghostmpc.codegen.templates import HEADER
from ghostmpc.core.score import ScoreError, ScorePlan, deployment_score_plan
from ghostmpc.core.sharing import (
    persistence_header,
    read_persistence,
    reconstruct_replicated,
    reconstruct_ring_n,
    share_ring_n,
)
from ghostmpc.eval.score import cosine, encode_gallery, encode_probe, score_ring

REPO = Path(__file__).resolve().parents[1]


class DeclaredWidths(unittest.TestCase):
    """The one thing that can wrap the ring, and the one thing that must not."""

    def test_the_deployed_profile_fits_ring32_with_room_to_spare(self):
        plan = deployment_score_plan(1024, 512)
        # f + f_g + log2||e|| + 1 = 6 + 15 + 6 + 1, and 2f + 2log2||e|| + 1.
        self.assertEqual(plan.declared_widths(),
                         {"similarity": 28, "norm": 25})
        for width in plan.declared_widths().values():
            self.assertLessEqual(width, plan.ring_bits - 1)

    def test_a_gallery_grid_that_would_wrap_is_refused_rather_than_measured(self):
        # An accumulator one bit too wide does not fail at run time: it returns
        # a wrapped integer and the client divides it into a plausible cosine.
        with self.assertRaises(ScoreError):
            deployment_score_plan(1, 512, gallery_fractional_bits=26)

    def test_a_norm_bound_above_the_certified_one_is_refused(self):
        deployment_score_plan(1, 512, norm_bound=64.0)
        with self.assertRaises(ScoreError):
            deployment_score_plan(1, 512, norm_bound=4096.0)

    def test_the_widths_track_the_two_grids_independently(self):
        plan = ScorePlan(gallery_rows=1, embedding_size=512,
                         fractional_bits=6, gallery_fractional_bits=9,
                         ring_bits=32, norm_bound=64.0).validate()
        # Only the similarity carries f_g; the norm is the probe against itself.
        self.assertEqual(plan.similarity_bits, 6 + 9 + 6 + 1)
        self.assertEqual(plan.norm_product_bits, 2 * 6 + 2 * 6 + 1)


class EmitterIsInvisibleWithoutAScore(unittest.TestCase):
    """Every committed baseline was measured on the unscored program."""

    def test_the_header_placeholder_leaves_no_trace_when_empty(self):
        # A placeholder on its own line would add a blank one, and every
        # emitted program -- and so every recorded digest -- would move.
        self.assertIn("{score_setup}def marker(", HEADER)

    def test_the_score_setup_ends_where_the_next_definition_begins(self):
        plan = deployment_score_plan(4, 512)
        setup = score_codegen.setup(plan, measured_timer=False)
        self.assertTrue(setup.endswith("\n\n"))


class DealerShares(unittest.TestCase):
    """The database must survive the round trip the parties will make."""

    def test_the_encoded_gallery_reconstructs_from_its_components(self):
        plan = deployment_score_plan(4, 8, gallery_fractional_bits=9)
        generator = np.random.default_rng(1)
        encoded = encode_gallery(generator.normal(size=(4, 8)), plan)
        flat = [int(v) % (1 << plan.ring_bits) for v in encoded.reshape(-1)]
        shares = share_ring_n(flat, 3, plan.ring_bits)
        self.assertEqual(reconstruct_ring_n(shares, plan.ring_bits), flat)

    def test_every_row_is_a_unit_vector_before_it_is_quantized(self):
        plan = deployment_score_plan(4, 512)
        generator = np.random.default_rng(2)
        rows = generator.normal(size=(4, 512)) * np.array([[1.0], [10.0],
                                                           [0.1], [100.0]])
        encoded = encode_gallery(rows, plan)
        norms = np.linalg.norm(
            encoded / (1 << plan.gallery_fractional_bits), axis=1)
        # Normalization is what removes the square root from the protocol, so a
        # scale surviving into the database would silently change every score.
        for norm in norms:
            self.assertAlmostEqual(norm, 1.0, places=3)

    def test_a_zero_row_is_refused_rather_than_enrolled(self):
        plan = deployment_score_plan(2, 4)
        rows = np.zeros((2, 4))
        rows[0, 0] = 1.0
        with self.assertRaises(ScoreError):
            encode_gallery(rows, plan)


class PlayerDataOrder(unittest.TestCase):
    """The program reads the database first; the file has to agree."""

    def test_the_gallery_components_precede_the_request_payloads(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_player_data(root, 4, 32, requests=3, parties=3,
                                values=[1, 2, 3, 4], gallery_size=8)
            pairs = []
            for party in range(3):
                path = root / "Persistence" / f"Transactions-P{party}.data"
                # The signature, then 4-byte share[0] and share[1] per element.
                self.assertEqual(len(path.read_bytes()),
                                 len(persistence_header(32)) + (8 + 4 * 3) * 8)
                pairs.append(read_persistence(path, 32))
            reconstructed = reconstruct_replicated(pairs, 32)
            # Eight gallery elements, then the image repeated per request.
            self.assertEqual(reconstructed[:8], [0] * 8)
            self.assertEqual(reconstructed[8:], [1, 2, 3, 4] * 3)

    def test_a_gallery_length_that_disagrees_with_the_plan_is_refused(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                prepare_player_data(Path(directory), 4, 32, values=[1, 2, 3, 4],
                                    gallery_size=8, gallery_values=[1, 2])


class ClearTextOracle(unittest.TestCase):
    """The client's half of the protocol exists in exactly one place."""

    def test_the_ring_score_reproduces_the_float_cosine(self):
        plan = deployment_score_plan(4, 512)
        generator = np.random.default_rng(5)
        gallery = encode_gallery(generator.normal(size=(4, 512)), plan)
        probe = generator.normal(scale=0.5, size=512)
        similarity, norm = score_ring(encode_probe(probe, plan), gallery, 2,
                                      plan)
        unit = gallery[2] / np.linalg.norm(gallery[2])
        expected = float(probe @ unit / np.linalg.norm(probe))
        # The residual is the f = 6 probe grid, which the MPC probe is already
        # on: it is quantization, not a modelling error.
        self.assertAlmostEqual(cosine(similarity, norm, plan), expected,
                               places=2)

    def test_the_accumulators_are_read_back_signed(self):
        plan = deployment_score_plan(2, 4, gallery_fractional_bits=9)
        gallery = encode_gallery(np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]]),
                                 plan)
        probe = encode_probe(np.array([-2.0, 0, 0, 0]), plan)
        similarity, norm = score_ring(probe, gallery, 0, plan)
        self.assertLess(similarity, 0)
        self.assertGreater(norm, 0)
        self.assertAlmostEqual(cosine(similarity, norm, plan), -1.0, places=3)

    def test_a_row_outside_the_gallery_is_refused(self):
        plan = deployment_score_plan(2, 4)
        gallery = encode_gallery(np.ones((2, 4)), plan)
        with self.assertRaises(ScoreError):
            score_ring(encode_probe(np.ones(4), plan), gallery, 5, plan)


if __name__ == "__main__":
    unittest.main()

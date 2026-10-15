"""The closed-set 1:N identification under MPC.

Every test here is pure Python: the declared widths, the shape of the
tournament, the clear-text rank-1 rule and the source the emitter writes.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from ghostmpc.codegen import score as score_codegen
from ghostmpc.core.identify import (
    IdentifyError,
    IdentifyPlan,
    deployment_identify_plan,
    identify_plan_report,
)
from ghostmpc.core.model import Manifest
from ghostmpc.core.score import deployment_score_plan
from ghostmpc.eval.identify import identify_ring, similarities_ring, tournament
from ghostmpc.eval.score import encode_gallery, encode_probe

DEPLOYED = deployment_score_plan(128, 512)


class DeclaredWidths(unittest.TestCase):
    def test_the_deployed_profile_fits_ring32(self):
        plan = deployment_identify_plan(DEPLOYED)
        widths = plan.declared_widths()
        self.assertEqual(widths["similarity"], 28)
        # One comparison reads a difference of two similarities: one magnitude
        # bit more, and the sign bit it already had.
        self.assertEqual(widths["difference"], 29)
        for name, width in widths.items():
            self.assertLessEqual(width, DEPLOYED.ring_bits - 1,
                                 f"{name} does not fit the ring")

    def test_nothing_is_truncated(self):
        """The ranking only subtracts, so no accumulator has to be reduced."""
        self.assertEqual(identify_plan_report(
            deployment_identify_plan(DEPLOYED))["truncations"], 0)

    def test_a_gallery_grid_that_does_not_fit_is_refused(self):
        score = deployment_score_plan(128, 512, gallery_fractional_bits=18)
        with self.assertRaises(IdentifyError):
            IdentifyPlan(score=score, candidates=128).validate()

    def test_a_query_wider_than_the_database_is_refused(self):
        with self.assertRaises(IdentifyError):
            IdentifyPlan(score=DEPLOYED, candidates=129).validate()


class TournamentShape(unittest.TestCase):
    def test_a_power_of_two_halves_every_level(self):
        plan = deployment_identify_plan(DEPLOYED)
        self.assertEqual(plan.level_widths(), [64, 32, 16, 8, 4, 2, 1])
        self.assertEqual(plan.levels, 7)
        self.assertEqual(plan.comparisons, 127)

    def test_an_odd_level_carries_its_last_element_forward(self):
        score = deployment_score_plan(5, 512)
        plan = deployment_identify_plan(score)
        self.assertEqual(plan.level_widths(), [2, 1, 1])
        # N - 1 comparisons whatever the shape: nobody is compared twice and
        # nobody is dropped.
        self.assertEqual(sum(plan.level_widths()), 4)

    def test_every_size_performs_n_minus_one_comparisons(self):
        for candidates in range(1, 130):
            plan = IdentifyPlan(
                score=deployment_score_plan(max(1, candidates), 512),
                candidates=candidates).validate()
            self.assertEqual(sum(plan.level_widths()), candidates - 1)


class ClearTextRule(unittest.TestCase):
    """The oracle ranks what the float cosine ranks, and breaks ties downwards."""

    def _plan(self, candidates: int) -> IdentifyPlan:
        return deployment_identify_plan(deployment_score_plan(candidates, 8))

    def test_the_integer_rank_one_is_the_float_rank_one(self):
        generator = np.random.default_rng(18)
        for _ in range(64):
            plan = self._plan(16)
            rows = generator.normal(size=(16, 8))
            probe = generator.normal(size=8)
            gallery = encode_gallery(rows, plan.score)
            index, _ = identify_ring(encode_probe(probe, plan.score), gallery,
                                     plan)
            unit = rows / np.linalg.norm(rows, axis=1, keepdims=True)
            expected = int(np.argmax(unit @ probe))
            self.assertEqual(index, expected)

    def test_a_tie_keeps_the_left_operand_of_the_fold(self):
        plan = self._plan(4)
        index, value = tournament([7, 7, 7, 7], plan)
        self.assertEqual((index, value), (0, 7))
        # Folded in half, row 1 meets row 3 and row 0 meets row 2: the tie
        # between rows 1 and 2 is never taken, and the winner of the second
        # level is the earlier position of that level, which is row 2.
        self.assertEqual(tournament([1, 9, 9, 2], plan)[0], 2)

    def test_the_carried_element_can_win(self):
        plan = self._plan(5)
        self.assertEqual(tournament([1, 2, 3, 4, 5], plan), (4, 5))
        self.assertEqual(tournament([9, 2, 3, 4, 5], plan), (0, 9))

    def test_the_oracle_ranks_what_the_maximum_is(self):
        """Whatever the fold, the winner is a maximum of the whole vector."""
        generator = np.random.default_rng(9)
        for candidates in (2, 3, 5, 8, 9, 16, 33):
            plan = self._plan(candidates)
            for _ in range(16):
                values = [int(v) for v in
                          generator.integers(-1 << 20, 1 << 20, candidates)]
                index, value = tournament(values, plan)
                self.assertEqual(value, max(values))
                self.assertEqual(values[index], value)

    def test_the_probe_itself_wins_its_own_row(self):
        plan = self._plan(8)
        generator = np.random.default_rng(1818)
        rows = generator.normal(size=(8, 8))
        probe = generator.normal(size=8)
        rows[5] = probe
        gallery = encode_gallery(rows, plan.score)
        index, _ = identify_ring(encode_probe(probe, plan.score), gallery, plan)
        self.assertEqual(index, 5)

    def test_the_similarities_are_the_scores_of_every_row(self):
        """The ranking reads the same accumulator a 1:1 score would return."""
        from ghostmpc.eval.score import score_ring

        plan = self._plan(4)
        generator = np.random.default_rng(3)
        gallery = encode_gallery(generator.normal(size=(4, 8)), plan.score)
        probe = encode_probe(generator.normal(size=8), plan.score)
        self.assertEqual(
            similarities_ring(probe, gallery, plan),
            [score_ring(probe, gallery, row, plan.score)[0] for row in range(4)])


class EmittedSource(unittest.TestCase):
    def _body(self, plan: IdentifyPlan) -> str:
        return score_codegen.identify_block(plan, "v_embedding",
                                            measured_timer=False)

    def test_nothing_is_truncated(self):
        body = self._body(deployment_identify_plan(DEPLOYED))
        self.assertNotIn("trunc", body)

    def test_a_level_is_one_vector_comparison(self):
        plan = deployment_identify_plan(DEPLOYED)
        body = self._body(plan)
        self.assertEqual(body.count("is_negative("), plan.levels)
        self.assertEqual(body.count(f", {plan.difference_bits})"), plan.levels)

    def test_the_first_level_selects_its_index_for_free(self):
        """``i`` and ``i + W`` are public, so the winner is ``i + bW``."""
        body = self._body(deployment_identify_plan(DEPLOYED))
        self.assertIn("identify_bit * 64 + regint.inc(64, 0, 1)", body)

    def test_one_dot_product_loop_reads_every_row(self):
        body = self._body(deployment_identify_plan(DEPLOYED))
        self.assertEqual(body.count("sint.dot_product("), 1)
        self.assertIn("for _row in range(CANDIDATES):", body)


class ProgramContract(unittest.TestCase):
    def _manifest(self):
        path = (Path(__file__).resolve().parents[1] / "artifacts" / "model"
                / "manifest.json")
        if not path.is_file():
            self.skipTest("no exported manifest in this checkout")
        return Manifest.load(path)

    def test_an_identification_without_a_database_is_refused(self):
        from ghostmpc.codegen.program import render

        manifest = self._manifest()
        with self.assertRaises(ValueError):
            render(manifest, fractional_bits=6, total_bits=15,
                   kernel_bits=9,
                   identify=deployment_identify_plan(DEPLOYED))

    def test_a_match_and_an_identification_exclude_each_other(self):
        from ghostmpc.codegen.program import render
        from ghostmpc.core.match import deployment_match_plan

        manifest = self._manifest()
        with self.assertRaises(ValueError):
            render(manifest, fractional_bits=6, total_bits=15,
                   kernel_bits=9, score=DEPLOYED,
                   match=deployment_match_plan(DEPLOYED, 0.28),
                   identify=deployment_identify_plan(DEPLOYED))

    def test_the_identification_replaces_the_one_to_one_score(self):
        import tempfile

        from ghostmpc.codegen.program import generate

        path = (Path(__file__).resolve().parents[1] / "artifacts" / "model"
                / "manifest.json")
        if not path.is_file():
            self.skipTest("no exported manifest in this checkout")
        with tempfile.TemporaryDirectory() as work:
            target = Path(work) / "identify.mpc"
            generate(path, target, requests=1, score=DEPLOYED,
                     identify=deployment_identify_plan(DEPLOYED))
            source = target.read_text()
        self.assertIn("identify_output[0]", source)
        self.assertNotIn("score_output[0] = sint.dot_product", source)
        # The database ingestion is the score's, unchanged.
        self.assertIn("gallery.assign(gallery_input(GALLERY_SIZE))", source)


if __name__ == "__main__":
    unittest.main()

"""The match decision under MPC.

Every test here is pure Python: the widths, the threshold constant and the
clear-text decision rule, plus the source the emitter writes for them.
"""

from __future__ import annotations

import unittest

from ghostmpc.codegen import score as score_codegen
from ghostmpc.core.match import (
    MINIMUM_THRESHOLD_RESOLUTION_BITS,
    MatchError,
    MatchPlan,
    deployment_match_plan,
    match_plan_report,
)
from ghostmpc.core.ring import to_signed
from ghostmpc.core.score import deployment_score_plan
from ghostmpc.eval.match import match_float, match_ring

DEPLOYED = deployment_score_plan(1024, 512)


class DeclaredWidths(unittest.TestCase):
    def test_deployed_profile_fits_ring32(self):
        plan = deployment_match_plan(DEPLOYED, 0.28)
        widths = plan.declared_widths()
        self.assertEqual(widths["similarity"], 28)
        self.assertEqual(widths["norm"], 25)
        for name, width in widths.items():
            self.assertLessEqual(width, DEPLOYED.ring_bits - 1,
                                 f"{name} does not fit the ring")

    def test_the_reduction_is_the_smallest_legal_one_and_shared(self):
        plan = deployment_match_plan(DEPLOYED, 0.28)
        self.assertEqual(plan.reduction, 12)
        self.assertEqual(plan.minimum_reduction, 12)
        # One truncation site over [s, n] carries one shift and one length.
        self.assertEqual(plan.norm_reduction, plan.reduction)
        self.assertEqual(plan.reduction_length,
                         max(plan.similarity_accumulator_bits,
                             plan.norm_accumulator_bits))

    def test_the_difference_pays_one_sign_bit_and_no_magnitude_bit(self):
        """Both sides are non-negative, so the difference is not wider."""
        plan = deployment_match_plan(DEPLOYED, 0.28)
        self.assertEqual(plan.difference_bits,
                         max(plan.square_bits, plan.product_bits))
        self.assertLessEqual(plan.difference_bits, DEPLOYED.ring_bits - 1)

    def test_the_accumulator_scales_are_the_two_grids(self):
        plan = deployment_match_plan(DEPLOYED, 0.28)
        self.assertEqual(plan.similarity_scale_bits,
                         DEPLOYED.fractional_bits + DEPLOYED.gallery_fractional_bits)
        self.assertEqual(plan.norm_scale_bits, 2 * DEPLOYED.fractional_bits)
        self.assertEqual(plan.similarity_accumulator_bits, DEPLOYED.similarity_bits)
        self.assertEqual(plan.norm_accumulator_bits, DEPLOYED.norm_product_bits)

    def test_the_threshold_keeps_its_resolution(self):
        for threshold in (-0.9, -0.35, 0.0, 0.28, 0.9):
            plan = deployment_match_plan(DEPLOYED, threshold)
            self.assertEqual(plan.constant_reduction,
                             2 * plan.reduction - plan.norm_reduction)
            self.assertGreaterEqual(
                plan.constant_scale_bits - plan.constant_reduction,
                MINIMUM_THRESHOLD_RESOLUTION_BITS)

    def test_a_threshold_outside_the_cosine_range_is_refused(self):
        with self.assertRaises(MatchError):
            deployment_match_plan(DEPLOYED, 1.5)


class ClearTextDecision(unittest.TestCase):
    def _accumulators(self, cosine: float, norm: float, plan: MatchPlan):
        """Two ring integers with a stated cosine and embedding norm."""
        scale = 1 << plan.similarity_scale_bits
        similarity = int(round(cosine * norm * scale))
        squared = int(round(norm * norm * (1 << plan.norm_scale_bits)))
        return similarity, squared

    def test_the_bit_is_the_sign_of_the_margin(self):
        plan = deployment_match_plan(DEPLOYED, 0.28)
        for cosine in (-0.9, -0.3, 0.0, 0.1, 0.27, 0.35, 0.8):
            similarity, norm = self._accumulators(cosine, 32.0, plan)
            self.assertEqual(match_ring(similarity, norm, plan),
                             int(cosine >= 0.28), f"cos = {cosine}")

    def test_a_negative_threshold_takes_the_other_branch(self):
        plan = deployment_match_plan(DEPLOYED, -0.35)
        for cosine in (-0.9, -0.4, -0.3, 0.0, 0.5):
            similarity, norm = self._accumulators(cosine, 32.0, plan)
            self.assertEqual(match_ring(similarity, norm, plan),
                             int(cosine >= -0.35), f"cos = {cosine}")

    def test_the_decision_agrees_with_the_exact_cosine_off_the_boundary(self):
        plan = deployment_match_plan(DEPLOYED, 0.28)
        for step in range(-40, 41):
            cosine = 0.28 + step * 0.005
            if abs(cosine - 0.28) < 0.002:
                continue
            similarity, norm = self._accumulators(cosine, 32.0, plan)
            self.assertEqual(match_ring(similarity, norm, plan),
                             match_float(similarity, norm, plan),
                             f"cos = {cosine}")

    def test_a_wrapped_norm_is_refused_rather_than_decided(self):
        plan = deployment_match_plan(DEPLOYED, 0.28)
        with self.assertRaises(MatchError):
            match_ring(1, -1, plan)

    def test_the_reduction_is_the_shared_nearest_truncation(self):
        """No second model of a primitive the repository already owns."""
        from ghostmpc.core.ring import trunc_nearest
        from ghostmpc.eval.match import reduce_accumulator
        plan = deployment_match_plan(DEPLOYED, 0.28)
        for value in (-(1 << 20) - 3, -1, 0, 1, (1 << 20) + 5):
            self.assertEqual(
                reduce_accumulator(value, plan.reduction_length,
                                   plan.reduction, plan),
                trunc_nearest(value, plan.reduction_length,
                              plan.reduction, ring_bits=plan.score.ring_bits))

    def test_the_comparison_is_taken_modulo_the_ring(self):
        """Both sides fit, so the signed difference is exact -- check it is."""
        plan = deployment_match_plan(DEPLOYED, 0.9)
        similarity, norm = self._accumulators(0.95, 41.9, plan)
        modulus = 1 << plan.score.ring_bits
        left = (similarity >> plan.reduction) ** 2
        right = plan.threshold_constant * (norm >> plan.norm_reduction)
        difference = to_signed((left - right) % modulus, plan.score.ring_bits)
        self.assertEqual(difference, left - right)
        self.assertEqual(match_ring(similarity, norm, plan), 1)

    def test_the_report_states_what_is_opened(self):
        plan = deployment_match_plan(DEPLOYED, 0.28)
        report = match_plan_report(plan)
        self.assertEqual(report["threshold"], 0.28)
        self.assertEqual(report["reduction_bits"], plan.reduction)
        self.assertIn("nothing", str(report["opened"]))


class Emitter(unittest.TestCase):
    def _emit(self, plan, **kwargs):
        return score_codegen.match_block(plan, measured_timer=False, **kwargs)

    def test_the_declared_widths_reach_the_emitted_source(self):
        plan = deployment_match_plan(DEPLOYED, 0.28)
        source = self._emit(plan)
        self.assertIn(
            f"match_operands.get_vector(), {plan.reduction_length}, "
            f"{plan.reduction}, bounded=True", source)
        self.assertIn(f"{plan.threshold_constant} * match_norm", source)
        self.assertIn(f", {plan.difference_bits})", source)
        # The sign is taken from the reduction, not computed again.
        self.assertIn("match_negative = match_signs[0]", source)
        self.assertNotIn("is_negative(score_output[0]", source)
        self.assertEqual(source.count("trunc_round_nearest_with_sign("), 1)

    def test_a_zero_threshold_emits_the_sign_alone(self):
        source = self._emit(deployment_match_plan(DEPLOYED, 0.0))
        self.assertNotIn("match_square", source)
        self.assertIn("match_output[0] = 1 - match_negative", source)

    def test_a_negative_threshold_takes_the_other_combination(self):
        positive = self._emit(deployment_match_plan(DEPLOYED, 0.28))
        negative = self._emit(deployment_match_plan(DEPLOYED, -0.35))
        self.assertIn("(1 - match_negative) * (1 - match_below)", positive)
        self.assertIn("1 - match_negative * (1 - match_below)", negative)

    def test_nothing_is_opened_unless_the_build_is_a_test_one(self):
        plan = deployment_match_plan(DEPLOYED, 0.28)
        self.assertNotIn("reveal", self._emit(plan))
        test = self._emit(plan, test_reveal=True)
        self.assertIn("match_output[0].reveal()", test)

    def test_the_match_output_exists_only_with_a_match(self):
        scored = score_codegen.setup(DEPLOYED, measured_timer=False)
        matched = score_codegen.setup(DEPLOYED, measured_timer=False,
                                      match=deployment_match_plan(DEPLOYED, 0.28))
        self.assertNotIn("match_output", scored)
        self.assertIn("match_output = sint.Array(1)", matched)
        # Both keep the blank line the header's placeholder relies on.
        self.assertTrue(scored.endswith("\n\n"))
        self.assertTrue(matched.endswith("\n\n"))


if __name__ == "__main__":
    unittest.main()

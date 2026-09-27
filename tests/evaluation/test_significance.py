"""Unit tests for SignificanceTester paired difference estimation."""

import unittest

from evaluation.refinement import create_refined_prism_evaluator
from evaluation.refinement_protocol import RefinementConfig, get_development_scenario_partition
from evaluation.runner import create_prism_evaluator_system
from evaluation.significance import SignificanceTester


class TestSignificance(unittest.TestCase):
    """Test suite for statistical significance hypothesis testing."""

    def test_paired_difference_calculation(self) -> None:
        baseline = create_prism_evaluator_system()
        refined = create_refined_prism_evaluator(RefinementConfig(), random_seed=42)
        dev_sc = get_development_scenario_partition()

        res = SignificanceTester.evaluate_paired_difference(
            baseline_evaluator=baseline,
            refined_evaluator=refined,
            scenarios=dev_sc,
            seeds=(42, 101),
        )

        self.assertEqual(res.sample_size, 2)
        self.assertIsInstance(res.mean_difference, float)
        self.assertIsInstance(res.is_statistically_significant, bool)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for RefinementConfig and create_refined_prism_evaluator."""

import unittest

from evaluation.refinement import create_refined_prism_evaluator
from evaluation.refinement_protocol import (
    RefinementConfig,
    get_development_scenario_partition,
    get_holdout_scenario_partition,
)
from evaluation.runner import EvaluationRunner


class TestRefinement(unittest.TestCase):
    """Test suite for refinement protocol and evaluator construction."""

    def test_scenario_partition_disjointness(self) -> None:
        dev_sc = get_development_scenario_partition()
        holdout_sc = get_holdout_scenario_partition()

        dev_keys = set(dev_sc.keys())
        holdout_keys = set(holdout_sc.keys())

        # Crucial scientific check: Dev and Holdout partitions must have 0 overlap!
        self.assertEqual(len(dev_keys.intersection(holdout_keys)), 0)
        self.assertGreater(len(dev_keys), 0)
        self.assertGreater(len(holdout_keys), 0)

    def test_refined_prism_execution(self) -> None:
        cfg = RefinementConfig(horizon=3.5, alpha_smooth=0.30)
        evaluator = create_refined_prism_evaluator(cfg, random_seed=42)
        runner = EvaluationRunner(random_seed=42)

        dev_sc = get_development_scenario_partition()
        samples = runner.run_scenario(dev_sc["HEAD_ON"], evaluator)
        self.assertGreater(len(samples), 0)


if __name__ == "__main__":
    unittest.main()

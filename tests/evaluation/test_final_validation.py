"""Unit tests for final holdout validation partition."""

import unittest

from evaluation.comparison import SystemComparator
from evaluation.refinement_protocol import get_holdout_scenario_partition


class TestFinalValidation(unittest.TestCase):
    """Test suite for holdout dataset validation."""

    def test_holdout_partition_evaluation(self) -> None:
        comparator = SystemComparator(random_seed=42)
        holdout_sc = get_holdout_scenario_partition()

        results = comparator.compare_systems(holdout_sc)
        self.assertIn("Baseline PRISM", results)
        self.assertIn("Refined PRISM", results)
        self.assertIn("TTC Baseline", results)
        self.assertIn("Distance Baseline", results)


if __name__ == "__main__":
    unittest.main()

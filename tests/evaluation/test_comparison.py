"""Unit tests for SystemComparator."""

import unittest

from evaluation.comparison import SystemComparator
from evaluation.refinement_protocol import get_development_scenario_partition


class TestComparison(unittest.TestCase):
    """Test suite for SystemComparator."""

    def test_compare_systems_development_partition(self) -> None:
        comparator = SystemComparator(random_seed=42)
        dev_sc = get_development_scenario_partition()

        results = comparator.compare_systems(dev_sc)
        self.assertIn("Distance Baseline", results)
        self.assertIn("TTC Baseline", results)
        self.assertIn("Baseline PRISM", results)
        self.assertIn("Refined PRISM", results)


if __name__ == "__main__":
    unittest.main()

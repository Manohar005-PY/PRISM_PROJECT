"""Unit tests for StatisticalSummary and aggregate_values."""

import unittest

from evaluation.aggregation import aggregate_values


class TestAggregation(unittest.TestCase):
    """Test suite for statistical aggregation."""

    def test_aggregate_values(self) -> None:
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = aggregate_values(data)

        self.assertEqual(stats.count, 5)
        self.assertAlmostEqual(stats.mean, 30.0)
        self.assertAlmostEqual(stats.median, 30.0)
        self.assertEqual(stats.min_val, 10.0)
        self.assertEqual(stats.max_val, 50.0)
        self.assertLess(stats.ci_95_lower, stats.mean)
        self.assertGreater(stats.ci_95_upper, stats.mean)


if __name__ == "__main__":
    unittest.main()

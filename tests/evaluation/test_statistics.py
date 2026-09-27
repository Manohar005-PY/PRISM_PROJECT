"""Unit tests for compute_statistics, standard errors, and confidence intervals."""

import unittest

from evaluation.statistics import compute_statistics


class TestStatistics(unittest.TestCase):
    """Test suite for statistical estimators."""

    def test_compute_statistics_metrics(self) -> None:
        data = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
        stats = compute_statistics(data)

        self.assertEqual(stats.count, 6)
        self.assertAlmostEqual(stats.mean, 15.0)
        self.assertAlmostEqual(stats.median, 15.0)
        self.assertGreater(stats.std_dev, 0.0)
        self.assertGreater(stats.std_err, 0.0)
        self.assertLess(stats.ci_95_lower, stats.mean)
        self.assertGreater(stats.ci_95_upper, stats.mean)

    def test_empty_and_single_sample_safety(self) -> None:
        empty_stats = compute_statistics([])
        self.assertEqual(empty_stats.count, 0)
        self.assertEqual(empty_stats.mean, 0.0)

        single_stats = compute_statistics([42.0])
        self.assertEqual(single_stats.count, 1)
        self.assertEqual(single_stats.mean, 42.0)
        self.assertEqual(single_stats.std_dev, 0.0)


if __name__ == "__main__":
    unittest.main()

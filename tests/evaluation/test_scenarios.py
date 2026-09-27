"""Unit tests for benchmark scenario generators."""

import unittest

from evaluation.scenarios import get_canonical_benchmark_suite


class TestEvaluationScenarios(unittest.TestCase):
    """Test suite for evaluation scenario generators."""

    def test_benchmark_suite_completeness(self) -> None:
        suite = get_canonical_benchmark_suite()
        required_keys = [
            "HEAD_ON",
            "HEAD_ON_FAST",
            "HEAD_ON_SLOW",
            "CROSSING",
            "PARALLEL",
            "DIVERGING",
            "STATIONARY_OBSTACLE",
            "NEAR_MISS",
            "DEPARTING_OBJECT",
            "MULTI_OBJECT",
        ]
        for k in required_keys:
            self.assertIn(k, suite)
            sc = suite[k]
            self.assertGreater(sc.duration, 0.0)
            self.assertGreater(len(sc.objects_initial), 0)


if __name__ == "__main__":
    unittest.main()

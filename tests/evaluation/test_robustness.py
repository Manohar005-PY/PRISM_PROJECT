"""Unit tests for RobustnessAnalyzer perturbation testing."""

import unittest

from evaluation.robustness import RobustnessAnalyzer
from evaluation.scenarios import create_head_on_scenario


class TestRobustness(unittest.TestCase):
    """Test suite for RobustnessAnalyzer."""

    def test_detection_drop_robustness(self) -> None:
        analyzer = RobustnessAnalyzer(random_seed=42)
        sc = create_head_on_scenario(duration=2.0, timestep=0.1)

        res = analyzer.evaluate_detection_drop_robustness(
            drop_rates=(0.0, 0.20),
            scenarios={"HEAD_ON": sc},
        )

        self.assertEqual(res.perturbation_type, "detection_drop_rate")
        self.assertEqual(len(res.perturbation_levels), 2)
        self.assertIn(0.0, res.metrics_by_level)
        self.assertIn(0.20, res.metrics_by_level)
        self.assertIn("position_rmse", res.metrics_by_level[0.0])


if __name__ == "__main__":
    unittest.main()

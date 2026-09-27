"""Unit tests for SensitivityAnalyzer parameter sweeps."""

import unittest

from evaluation.scenarios import create_head_on_scenario
from evaluation.sensitivity import SensitivityAnalyzer


class TestSensitivity(unittest.TestCase):
    """Test suite for SensitivityAnalyzer."""

    def test_horizon_sensitivity_sweep(self) -> None:
        analyzer = SensitivityAnalyzer(random_seed=42)
        sc = create_head_on_scenario(duration=2.0, timestep=0.1)

        res = analyzer.evaluate_horizon_sensitivity(
            horizons=(1.0, 2.0),
            scenarios={"HEAD_ON": sc},
        )

        self.assertEqual(res.parameter_name, "prediction_horizon")
        self.assertEqual(len(res.parameter_values), 2)
        self.assertIn(1.0, res.metrics_by_value)
        self.assertIn(2.0, res.metrics_by_value)
        self.assertIn("hazard_detection_rate", res.metrics_by_value[1.0])


if __name__ == "__main__":
    unittest.main()

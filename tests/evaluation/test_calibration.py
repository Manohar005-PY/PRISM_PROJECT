"""Unit tests for PerceptionCalibrator depth-bin analysis, coverage, and calibration status."""

import unittest

from app.core.types import Vector2
from app.perception.model import PhysicalObservation
from evaluation.calibration import PerceptionCalibrator


class TestCalibration(unittest.TestCase):
    """Test suite for PerceptionCalibrator."""

    def test_depth_binning_and_calibration_ratio(self) -> None:
        calibrator = PerceptionCalibrator()

        # Add samples in 2-4m range with known noise
        for _ in range(10):
            gt = Vector2(0.0, 3.0)
            obs = PhysicalObservation(
                position=Vector2(0.05, 3.10),
                measurement_covariance=((0.01, 0.0), (0.0, 0.04)),  # sigma_z = 0.20
                confidence=0.9,
                timestamp=0.1,
                detection_id="d1",
                class_name="person",
            )
            calibrator.record_sample(gt, obs)

        bins = calibrator.evaluate_calibration()
        bin_2_4 = next(b for b in bins if b.depth_range == "2-4m")

        self.assertEqual(bin_2_4.sample_count, 10)
        self.assertAlmostEqual(bin_2_4.bias_z, 0.10, places=2)
        self.assertAlmostEqual(bin_2_4.predicted_sigma_z, 0.20, places=2)
        self.assertAlmostEqual(bin_2_4.calibration_ratio_z, 0.50, places=2)
        self.assertEqual(bin_2_4.calibration_status, "CONSERVATIVE")
        self.assertEqual(bin_2_4.coverage_1sigma_z, 100.0)


if __name__ == "__main__":
    unittest.main()

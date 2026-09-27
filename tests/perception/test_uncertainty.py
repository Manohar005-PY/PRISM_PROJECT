"""Unit tests for MeasurementUncertaintyModel distance and confidence scaling."""

import unittest

from app.perception.uncertainty import (
    MeasurementUncertaintyConfig,
    MeasurementUncertaintyModel,
)


class TestMeasurementUncertainty(unittest.TestCase):
    """Test suite for observation uncertainty calculation."""

    def setUp(self) -> None:
        self.model = MeasurementUncertaintyModel(
            MeasurementUncertaintyConfig(
                sigma_x0=0.10,
                sigma_z0=0.20,
                beta_x=0.01,
                beta_z=0.04,
                confidence_noise_gain=1.5,
                min_confidence=0.20,
            )
        )

    def test_distance_dependent_uncertainty_growth(self) -> None:
        """Uncertainty covariance R must grow monotonically with depth Z."""
        r_near = self.model.compute_covariance(depth_z=2.0, confidence=1.0)
        r_far = self.model.compute_covariance(depth_z=6.0, confidence=1.0)

        # Consistent Part 4 formulation:
        # Near: Z=2m => R_xx = 0.10^2 * (1 + 0.01*4) = 0.01 * 1.04 = 0.0104
        # Far:  Z=6m => R_xx = 0.10^2 * (1 + 0.01*36) = 0.01 * 1.36 = 0.0136
        self.assertGreater(r_far[0][0], r_near[0][0])
        self.assertGreater(r_far[1][1], r_near[1][1])

    def test_confidence_scaling(self) -> None:
        """Lower confidence must scale uncertainty up monotonically."""
        r_high_conf = self.model.compute_covariance(depth_z=3.0, confidence=1.0)
        r_low_conf = self.model.compute_covariance(depth_z=3.0, confidence=0.4)

        self.assertGreater(r_low_conf[0][0], r_high_conf[0][0])
        self.assertGreater(r_low_conf[1][1], r_high_conf[1][1])

    def test_invalid_depth_raises(self) -> None:
        """Non-positive or non-finite depth must raise ValueError."""
        with self.assertRaises(ValueError):
            self.model.compute_covariance(depth_z=0.0)
        with self.assertRaises(ValueError):
            self.model.compute_covariance(depth_z=-3.0)


if __name__ == "__main__":
    unittest.main()

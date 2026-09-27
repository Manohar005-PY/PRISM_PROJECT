"""Unit tests for spatial uncertainty expansion and collision probability."""

import math
import unittest

from app.core.types import Covariance4x4
from app.risk.uncertainty import (
    compute_collision_probability,
    compute_uncertainty_expanded_radius,
    extract_position_std,
)


class TestRiskUncertainty(unittest.TestCase):
    """Test suite for spatial uncertainty operations."""

    def test_extract_position_std_none_and_finite(self) -> None:
        self.assertEqual(extract_position_std(None), 0.0)

        cov: Covariance4x4 = (
            (0.09, 0.0, 0.0, 0.0),
            (0.0, 0.16, 0.0, 0.0),
            (0.0, 0.0, 0.25, 0.0),
            (0.0, 0.0, 0.0, 0.25),
        )
        # sqrt(0.09 + 0.16) = sqrt(0.25) = 0.5
        self.assertAlmostEqual(extract_position_std(cov), 0.5)

    def test_expanded_radius(self) -> None:
        cov: Covariance4x4 = (
            (0.09, 0.0, 0.0, 0.0),
            (0.0, 0.16, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
        )
        base_r = 1.0
        # R_eff = 1.0 + 1.5 * 0.5 = 1.75m
        r_eff = compute_uncertainty_expanded_radius(base_r, cov, k_sigma=1.5)
        self.assertAlmostEqual(r_eff, 1.75)

    def test_collision_probability_deterministic_and_probabilistic(self) -> None:
        # 1. d_cpa <= radius => P = 1.0
        self.assertEqual(compute_collision_probability(d_cpa=0.8, collision_radius=1.0, covariance=None), 1.0)

        # 2. d_cpa > radius with zero covariance => P = 0.0
        self.assertEqual(compute_collision_probability(d_cpa=1.5, collision_radius=1.0, covariance=None), 0.0)

        # 3. d_cpa > radius with positive covariance
        cov: Covariance4x4 = (
            (0.125, 0.0, 0.0, 0.0),
            (0.0, 0.125, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
        )
        # var_pos = 0.25, sigma = 0.5, margin = 1.5 - 1.0 = 0.5
        # exponent = -(0.5^2) / (2 * 0.25) = -0.25 / 0.5 = -0.5
        # P = exp(-0.5) ~ 0.6065
        p = compute_collision_probability(d_cpa=1.5, collision_radius=1.0, covariance=cov)
        self.assertAlmostEqual(p, math.exp(-0.5), delta=0.01)


if __name__ == "__main__":
    unittest.main()

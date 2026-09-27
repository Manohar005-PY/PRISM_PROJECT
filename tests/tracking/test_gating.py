"""Unit tests for Mahalanobis innovation distance and spatial gating."""

import math
import unittest

from app.core.types import Vector2
from app.tracking.gating import (
    compute_innovation_covariance,
    compute_mahalanobis_distance,
    evaluate_gate,
    invert_2x2,
)


class TestTrackingGating(unittest.TestCase):
    """Test suite for Mahalanobis gating."""

    def test_innovation_covariance_calculation(self) -> None:
        P_pred = (
            (0.16, 0.0, 0.0, 0.0),
            (0.0, 0.25, 0.0, 0.0),
            (0.0, 0.0, 0.5, 0.0),
            (0.0, 0.0, 0.0, 0.5),
        )
        R_meas = ((0.04, 0.0), (0.0, 0.09))

        S = compute_innovation_covariance(P_pred, R_meas)
        # S00 = 0.16 + 0.04 = 0.20
        # S11 = 0.25 + 0.09 = 0.34
        self.assertAlmostEqual(S[0][0], 0.20)
        self.assertAlmostEqual(S[1][1], 0.34)
        self.assertAlmostEqual(S[0][1], 0.0)

    def test_invert_2x2_diagonal(self) -> None:
        S = ((0.20, 0.0), (0.0, 0.50))
        S_inv = invert_2x2(S)
        self.assertIsNotNone(S_inv)
        self.assertAlmostEqual(S_inv[0][0], 5.0)  # type: ignore[index]
        self.assertAlmostEqual(S_inv[1][1], 2.0)  # type: ignore[index]

    def test_mahalanobis_distance_computation(self) -> None:
        meas = Vector2(0.2, 3.4)
        pred = Vector2(0.0, 3.0)
        # dx = 0.2, dz = 0.4
        S_inv = ((25.0, 0.0), (0.0, 6.25))  # S = diag(0.04, 0.16) => S_inv = diag(25.0, 6.25)

        # d_M^2 = (0.2)^2 * 25.0 + (0.4)^2 * 6.25 = 0.04 * 25.0 + 0.16 * 6.25 = 1.0 + 1.0 = 2.0
        d_m2 = compute_mahalanobis_distance(meas, pred, S_inv)
        self.assertAlmostEqual(d_m2, 2.0)

    def test_evaluate_gate_acceptance_and_rejection(self) -> None:
        meas_near = Vector2(0.1, 3.1)
        meas_far = Vector2(3.0, 9.0)
        pred = Vector2(0.0, 3.0)
        S = ((0.10, 0.0), (0.0, 0.20))

        is_acc, d_m2 = evaluate_gate(meas_near, pred, S, gating_threshold=9.21)
        self.assertTrue(is_acc)
        self.assertLess(d_m2, 9.21)

        is_acc_far, d_m2_far = evaluate_gate(meas_far, pred, S, gating_threshold=9.21)
        self.assertFalse(is_acc_far)
        self.assertGreater(d_m2_far, 9.21)


if __name__ == "__main__":
    unittest.main()

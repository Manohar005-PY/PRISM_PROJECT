"""Unit tests for spatial association, Mahalanobis gating, and Hungarian global assignment."""

import unittest

from app.core.types import Vector2
from app.tracking.association import (
    compute_innovation_covariance,
    invert_2x2,
    linear_sum_assignment,
    mahalanobis_distance_squared,
    solve_association,
)


class TestTrackingAssociation(unittest.TestCase):
    """Test suite for association mathematics and Hungarian solver."""

    def test_innovation_covariance(self) -> None:
        P_pred = (
            (0.16, 0.0, 0.0, 0.0),
            (0.0, 0.25, 0.0, 0.0),
            (0.0, 0.0, 0.5, 0.0),
            (0.0, 0.0, 0.0, 0.5),
        )
        R_meas = ((0.04, 0.0), (0.0, 0.09))
        S = compute_innovation_covariance(P_pred, R_meas)
        self.assertAlmostEqual(S[0][0], 0.20)
        self.assertAlmostEqual(S[1][1], 0.34)

    def test_invert_2x2(self) -> None:
        S = ((0.20, 0.0), (0.0, 0.50))
        S_inv = invert_2x2(S)
        self.assertIsNotNone(S_inv)
        self.assertAlmostEqual(S_inv[0][0], 5.0)  # type: ignore[index]
        self.assertAlmostEqual(S_inv[1][1], 2.0)  # type: ignore[index]

    def test_mahalanobis_distance_squared(self) -> None:
        z = Vector2(0.2, 3.4)
        z_hat = Vector2(0.0, 3.0)
        S = ((0.04, 0.0), (0.0, 0.16))
        d_m2 = mahalanobis_distance_squared(z, z_hat, S)
        # (0.2)^2 / 0.04 + (0.4)^2 / 0.16 = 1.0 + 1.0 = 2.0
        self.assertAlmostEqual(d_m2, 2.0)

    def test_hungarian_assignment(self) -> None:
        cost = [
            [4.0, 1.0, 3.0],
            [2.0, 0.5, 5.0],
            [3.0, 2.0, 2.0],
        ]
        matches = linear_sum_assignment(cost)
        match_dict = dict(matches)
        self.assertEqual(match_dict[0], 1)
        self.assertEqual(match_dict[1], 0)
        self.assertEqual(match_dict[2], 2)

    def test_solve_association_with_gating(self) -> None:
        cost_3x2 = [
            [0.5, 10.0],
            [10.0, 0.8],
            [10.0, 10.0],
        ]
        matches, un_trks, un_obs = solve_association(cost_3x2, num_tracks=3, num_observations=2, max_cost_threshold=5.0)
        self.assertEqual(len(matches), 2)
        self.assertEqual(dict(matches)[0], 0)
        self.assertEqual(dict(matches)[1], 1)
        self.assertEqual(un_trks, [2])
        self.assertEqual(un_obs, [])

    def test_empty_association(self) -> None:
        matches, un_trks, un_obs = solve_association([], num_tracks=0, num_observations=3, max_cost_threshold=5.0)
        self.assertEqual(matches, [])
        self.assertEqual(un_trks, [])
        self.assertEqual(un_obs, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()

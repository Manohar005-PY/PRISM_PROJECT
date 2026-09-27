"""Unit tests for analytical CPA, quadratic TTC, and collision boundary geometry."""

import math
import unittest

from app.core.types import Vector2
from app.prediction.collision_geometry import (
    CollisionModel,
    analyze_collision,
    closest_point_of_approach,
    time_to_collision,
)


class TestCollisionGeometry(unittest.TestCase):
    """Test suite for analytical CPA and TTC equations."""

    def test_head_on_cpa_and_ttc(self) -> None:
        """Head-on approach: p_rel=(0, 5), v_rel=(0, -2), R=1.0m."""
        p_rel = Vector2(0.0, 5.0)
        v_rel = Vector2(0.0, -2.0)
        radius = 1.0

        cpa = closest_point_of_approach(p_rel, v_rel)
        self.assertEqual(cpa.time_to_cpa, 2.5)
        self.assertEqual(cpa.distance_at_cpa, 0.0)
        self.assertEqual(cpa.relative_position_at_cpa, Vector2(0.0, 0.0))
        self.assertTrue(cpa.is_approaching)

        # Boundary intersection: entering radius R=1.0m at t=2.0s
        ttc = time_to_collision(p_rel, v_rel, collision_radius=radius)
        self.assertIsNotNone(ttc)
        self.assertAlmostEqual(ttc, 2.0)  # type: ignore[arg-type]

        # Prediction struct
        model = CollisionModel(object_radius=0.4, ego_radius=0.3, additional_margin=0.3)  # R=1.0m
        pred = analyze_collision(p_rel, v_rel, model, horizon=3.0)
        self.assertTrue(pred.collision_predicted)
        self.assertEqual(pred.ttc, 2.0)
        self.assertEqual(pred.closest_distance, 0.0)

    def test_near_miss_no_collision(self) -> None:
        """Near-miss path: object passes 3m to the side at CPA, safety radius is 1m."""
        p_rel = Vector2(3.0, 4.0)
        v_rel = Vector2(0.0, -1.0)
        radius = 1.0

        cpa = closest_point_of_approach(p_rel, v_rel)
        self.assertEqual(cpa.time_to_cpa, 4.0)
        self.assertAlmostEqual(cpa.distance_at_cpa, 3.0)

        # Since CPA distance 3.0m > radius 1.0m, no boundary collision occurs
        ttc = time_to_collision(p_rel, v_rel, collision_radius=radius)
        self.assertIsNone(ttc)

        pred = analyze_collision(p_rel, v_rel, CollisionModel(object_radius=0.3, ego_radius=0.3, additional_margin=0.4), horizon=5.0)
        self.assertFalse(pred.collision_predicted)
        self.assertIsNone(pred.ttc)

    def test_tangent_collision(self) -> None:
        """Trajectory grazes the boundary exactly at CPA (d_CPA = R)."""
        p_rel = Vector2(1.0, 4.0)
        v_rel = Vector2(0.0, -1.0)
        radius = 1.0

        cpa = closest_point_of_approach(p_rel, v_rel)
        self.assertEqual(cpa.time_to_cpa, 4.0)
        self.assertAlmostEqual(cpa.distance_at_cpa, 1.0)

        ttc = time_to_collision(p_rel, v_rel, collision_radius=radius)
        self.assertIsNotNone(ttc)
        self.assertAlmostEqual(ttc, 4.0)  # type: ignore[arg-type]

    def test_already_inside_collision_radius(self) -> None:
        """When initial separation <= radius, TTC must be 0.0 immediately."""
        p_rel = Vector2(0.2, 0.4)  # Distance = sqrt(0.2^2 + 0.4^2) = ~0.447m < 1.0m
        v_rel = Vector2(0.0, -1.0)
        radius = 1.0

        ttc = time_to_collision(p_rel, v_rel, collision_radius=radius)
        self.assertEqual(ttc, 0.0)

    def test_diverging_and_past_collisions(self) -> None:
        """When trajectories are moving apart, TTC must be None."""
        p_rel = Vector2(0.0, 5.0)
        v_rel = Vector2(0.0, 2.0)  # Moving away at 2 m/s
        radius = 1.0

        cpa = closest_point_of_approach(p_rel, v_rel)
        self.assertEqual(cpa.time_to_cpa, 0.0)
        self.assertEqual(cpa.distance_at_cpa, 5.0)
        self.assertFalse(cpa.is_approaching)

        ttc = time_to_collision(p_rel, v_rel, collision_radius=radius)
        self.assertIsNone(ttc)

    def test_zero_relative_velocity(self) -> None:
        """Stationary relative motion outside safety boundary produces TTC=None."""
        p_rel = Vector2(0.0, 5.0)
        v_rel = Vector2(0.0, 0.0)
        radius = 1.0

        cpa = closest_point_of_approach(p_rel, v_rel)
        self.assertEqual(cpa.time_to_cpa, 0.0)
        self.assertEqual(cpa.distance_at_cpa, 5.0)
        self.assertFalse(cpa.is_approaching)

        ttc = time_to_collision(p_rel, v_rel, collision_radius=radius)
        self.assertIsNone(ttc)

    def test_collision_outside_horizon(self) -> None:
        """Collision occurring beyond the evaluation horizon is marked collision_predicted=False."""
        p_rel = Vector2(0.0, 10.0)
        v_rel = Vector2(0.0, -1.0)  # Reaches R=1m at t=9.0s
        model = CollisionModel(object_radius=0.4, ego_radius=0.3, additional_margin=0.3)  # R=1.0m

        pred_short_horizon = analyze_collision(p_rel, v_rel, model, horizon=5.0)
        self.assertFalse(pred_short_horizon.collision_predicted)  # TTC=9.0s > horizon=5.0s
        self.assertEqual(pred_short_horizon.ttc, 9.0)

        pred_long_horizon = analyze_collision(p_rel, v_rel, model, horizon=10.0)
        self.assertTrue(pred_long_horizon.collision_predicted)


if __name__ == "__main__":
    unittest.main()

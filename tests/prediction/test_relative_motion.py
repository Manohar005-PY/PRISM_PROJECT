"""Unit tests for RelativeMotion and relative kinematic computations."""

import unittest

from app.core.types import ObjectState, Vector2
from app.prediction.relative_motion import (
    EgoMotionState,
    RelativeMotion,
    compute_relative_motion,
    compute_relative_motion_from_states,
)


class TestRelativeMotion(unittest.TestCase):
    """Test suite for relative motion formulations."""

    def test_stationary_object_moving_ego(self) -> None:
        """Moving ego toward stationary obstacle: relative velocity opposes ego velocity."""
        ego_pos = Vector2(0.0, 0.0)
        ego_vel = Vector2(0.0, 1.2)  # User walking forward at 1.2 m/s
        obj_pos = Vector2(0.0, 6.0)  # Obstacle 6m ahead
        obj_vel = Vector2(0.0, 0.0)

        rel = compute_relative_motion(ego_pos, ego_vel, obj_pos, obj_vel)
        self.assertEqual(rel.relative_position, Vector2(0.0, 6.0))
        self.assertEqual(rel.relative_velocity, Vector2(0.0, -1.2))
        self.assertEqual(rel.distance, 6.0)
        self.assertAlmostEqual(rel.relative_speed, 1.2)
        self.assertTrue(rel.is_approaching)
        self.assertAlmostEqual(rel.closing_speed, 1.2)

    def test_equal_velocity_parallel_motion(self) -> None:
        """When user and object move at identical velocity, relative velocity is zero."""
        ego_pos = Vector2(0.0, 0.0)
        ego_vel = Vector2(0.0, 1.0)
        obj_pos = Vector2(2.0, 5.0)
        obj_vel = Vector2(0.0, 1.0)

        rel = compute_relative_motion(ego_pos, ego_vel, obj_pos, obj_vel)
        self.assertEqual(rel.relative_velocity, Vector2(0.0, 0.0))
        self.assertFalse(rel.is_approaching)
        self.assertAlmostEqual(rel.closing_speed, 0.0)

    def test_head_on_closing_speed(self) -> None:
        """Head-on approach: closing speed is the sum of both speeds."""
        ego_pos = Vector2(0.0, 0.0)
        ego_vel = Vector2(0.0, 1.0)
        obj_pos = Vector2(0.0, 5.0)
        obj_vel = Vector2(0.0, -1.0)

        rel = compute_relative_motion(ego_pos, ego_vel, obj_pos, obj_vel)
        self.assertEqual(rel.relative_velocity, Vector2(0.0, -2.0))
        self.assertTrue(rel.is_approaching)
        self.assertAlmostEqual(rel.closing_speed, 2.0)

    def test_diverging_motion(self) -> None:
        """Diverging trajectories: is_approaching is False and closing speed is negative."""
        ego_pos = Vector2(0.0, 2.0)
        ego_vel = Vector2(0.0, -1.0)  # Moving backward
        obj_pos = Vector2(0.0, 5.0)
        obj_vel = Vector2(0.0, 1.0)   # Moving forward

        rel = compute_relative_motion(ego_pos, ego_vel, obj_pos, obj_vel)
        self.assertEqual(rel.relative_position, Vector2(0.0, 3.0))
        self.assertEqual(rel.relative_velocity, Vector2(0.0, 2.0))
        self.assertFalse(rel.is_approaching)
        self.assertAlmostEqual(rel.closing_speed, -2.0)

    def test_compute_from_states(self) -> None:
        """Verify helper extracts positions and velocities from state objects."""
        ego = EgoMotionState(position=Vector2(0.0, 0.0), velocity=Vector2(1.0, 0.0))
        obj = ObjectState(1, "person", Vector2(0.0, 5.0), Vector2(0.0, -1.0), 0.0)

        rel = compute_relative_motion_from_states(ego, obj)
        self.assertEqual(rel.relative_position, Vector2(0.0, 5.0))
        self.assertEqual(rel.relative_velocity, Vector2(-1.0, -1.0))


if __name__ == "__main__":
    unittest.main()

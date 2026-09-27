"""Unit tests for predefined PRISM simulation scenarios."""

import math
import unittest

from app.core.types import Vector2
from simulation.motion import ConstantVelocityModel
from simulation.scenarios import (
    create_crossing_scenario,
    create_diverging_scenario,
    create_head_on_scenario,
    create_parallel_scenario,
    create_stationary_obstacle_scenario,
)


class TestScenarios(unittest.TestCase):
    """Test suite verifying geometric invariants of canonical scenarios."""

    def test_head_on_scenario_kinematics(self) -> None:
        """Verify head-on approach: relative distance decreases toward convergence."""
        scenario = create_head_on_scenario(duration=3.0, timestep=0.5)
        ego = scenario.ego_initial
        obj = scenario.objects_initial[0]

        # Initial relative distance
        init_dist = (obj.position - ego.position).magnitude
        self.assertEqual(init_dist, 5.0)

        # Propagate to t=2.0s
        t = 2.0
        ego_t = ConstantVelocityModel.advance_ego(ego, t)
        obj_t = ConstantVelocityModel.advance_entity(obj, t)

        dist_t = (obj_t.position - ego_t.position).magnitude
        self.assertAlmostEqual(dist_t, 1.0)
        self.assertTrue(dist_t < init_dist)

        # Theoretical convergence at t=2.5s
        t_coll = 2.5
        ego_coll = ConstantVelocityModel.advance_ego(ego, t_coll)
        obj_coll = ConstantVelocityModel.advance_entity(obj, t_coll)
        self.assertEqual(ego_coll.position, Vector2(0.0, 2.5))
        self.assertEqual(obj_coll.position, Vector2(0.0, 2.5))
        self.assertEqual(ego_coll.position, obj_coll.position)

    def test_crossing_scenario_exact_intersection(self) -> None:
        """Verify crossing scenario: paths intersect at known time and position.

        Expected Ground Truth:
            expected_intersection_time = 5.0 s
            expected_intersection_position = (0.0, 0.0)
        """
        scenario = create_crossing_scenario(duration=6.0, timestep=0.1)
        ego = scenario.ego_initial
        obj = scenario.objects_initial[0]

        expected_intersection_time = 5.0
        expected_intersection_position = Vector2(0.0, 0.0)

        # At t=0s, ego is at (-5, 0) and obj is at (0, 5)
        self.assertEqual(ego.position, Vector2(-5.0, 0.0))
        self.assertEqual(obj.position, Vector2(0.0, 5.0))

        # At t = expected_intersection_time, both must occupy expected_intersection_position
        ego_cross = ConstantVelocityModel.advance_ego(ego, expected_intersection_time)
        obj_cross = ConstantVelocityModel.advance_entity(obj, expected_intersection_time)

        self.assertEqual(ego_cross.position, expected_intersection_position)
        self.assertEqual(obj_cross.position, expected_intersection_position)

    def test_parallel_scenario_constant_separation(self) -> None:
        """Verify parallel motion: separation remains perfectly constant over time."""
        scenario = create_parallel_scenario(duration=5.0, timestep=1.0)
        ego = scenario.ego_initial
        obj = scenario.objects_initial[0]

        # Initial separation vector and Euclidean distance
        init_sep = obj.position - ego.position
        init_dist = init_sep.magnitude

        for t in [1.0, 2.0, 3.0, 4.0, 5.0]:
            ego_t = ConstantVelocityModel.advance_ego(ego, t)
            obj_t = ConstantVelocityModel.advance_entity(obj, t)
            sep_t = obj_t.position - ego_t.position

            self.assertAlmostEqual(sep_t.x, 2.0)
            self.assertAlmostEqual(sep_t.z, 5.0)
            self.assertAlmostEqual(sep_t.magnitude, init_dist)

    def test_diverging_scenario_increasing_separation(self) -> None:
        """Verify diverging motion: no collision occurs and trajectories diverge.

        Ego: (0, 0), v=(0, 1) -> (0, t)
        Obj: (2, 5), v=(1, 0) -> (2 + t, 5)
        Separation vector: (2 + t, 5 - t). Minimum distance at t=1.5s is ~4.95m (safe).
        For t >= 1.5s, Euclidean distance increases strictly.
        """
        scenario = create_diverging_scenario(duration=5.0, timestep=1.0)
        ego = scenario.ego_initial
        obj = scenario.objects_initial[0]

        # Minimum distance at t=1.5s is > 4.9m (never collides)
        ego_min = ConstantVelocityModel.advance_ego(ego, 1.5)
        obj_min = ConstantVelocityModel.advance_entity(obj, 1.5)
        min_dist = (obj_min.position - ego_min.position).magnitude
        self.assertAlmostEqual(min_dist, math.sqrt(3.5**2 + 3.5**2))
        self.assertTrue(min_dist > 4.9)

        # Distance strictly increases for t >= 2.0s
        prev_dist = min_dist
        for t in [2.0, 3.0, 4.0, 5.0]:
            ego_t = ConstantVelocityModel.advance_ego(ego, t)
            obj_t = ConstantVelocityModel.advance_entity(obj, t)
            dist_t = (obj_t.position - ego_t.position).magnitude

            self.assertTrue(dist_t > prev_dist)
            prev_dist = dist_t

        # At t=5.0s, distance is 7.0m > initial distance (~5.385m)
        ego_end = ConstantVelocityModel.advance_ego(ego, 5.0)
        obj_end = ConstantVelocityModel.advance_entity(obj, 5.0)
        final_dist = (obj_end.position - ego_end.position).magnitude
        self.assertAlmostEqual(final_dist, 7.0)

    def test_stationary_obstacle_scenario(self) -> None:
        """Verify stationary obstacle: obstacle remains immobile while user approaches."""
        scenario = create_stationary_obstacle_scenario(duration=4.0, timestep=1.0)
        ego = scenario.ego_initial
        obj = scenario.objects_initial[0]

        self.assertEqual(obj.velocity, Vector2(0.0, 0.0))

        for t in [1.0, 2.0, 3.0, 4.0]:
            ego_t = ConstantVelocityModel.advance_ego(ego, t)
            obj_t = ConstantVelocityModel.advance_entity(obj, t)

            # Obstacle position does not change
            self.assertEqual(obj_t.position, Vector2(0.0, 5.0))
            # Distance decreases linearly
            expected_dist = 5.0 - t
            actual_dist = (obj_t.position - ego_t.position).magnitude
            self.assertAlmostEqual(actual_dist, expected_dist)


if __name__ == "__main__":
    unittest.main()

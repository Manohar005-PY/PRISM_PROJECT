"""Unit tests for trajectory prediction error and CPA metrics."""

import math
import unittest

from app.core.types import Trajectory, TrajectoryPoint, Vector2
from app.prediction.collision_geometry import CPAResult
from app.prediction.metrics import (
    cpa_error,
    prediction_error_at_horizon,
    trajectory_position_rmse,
)
from simulation.entities import SimulatedEntity


class TestPredictionMetrics(unittest.TestCase):
    """Test suite for timestamp-aligned prediction metrics."""

    def test_perfect_prediction_rmse(self) -> None:
        """Zero error when predicted points match ground truth exactly."""
        pts = [
            TrajectoryPoint(timestamp=0.0, position=Vector2(0.0, 5.0), velocity=Vector2(0.0, -1.0)),
            TrajectoryPoint(timestamp=0.5, position=Vector2(0.0, 4.5), velocity=Vector2(0.0, -1.0)),
            TrajectoryPoint(timestamp=1.0, position=Vector2(0.0, 4.0), velocity=Vector2(0.0, -1.0)),
        ]
        traj = Trajectory(object_id=1, points=tuple(pts), prediction_horizon=1.0)

        gt = [
            SimulatedEntity(object_id=1, class_id="person", position=Vector2(0.0, 5.0), velocity=Vector2(0.0, -1.0), timestamp=0.0),
            SimulatedEntity(object_id=1, class_id="person", position=Vector2(0.0, 4.5), velocity=Vector2(0.0, -1.0), timestamp=0.5),
            SimulatedEntity(object_id=1, class_id="person", position=Vector2(0.0, 4.0), velocity=Vector2(0.0, -1.0), timestamp=1.0),
        ]

        rmse = trajectory_position_rmse(traj, gt)
        self.assertAlmostEqual(rmse, 0.0)

    def test_known_offset_rmse(self) -> None:
        """Constant 0.3m lateral error produces exact 0.3m RMSE."""
        pts = [
            TrajectoryPoint(timestamp=0.0, position=Vector2(0.3, 5.0), velocity=Vector2(0.0, -1.0)),
            TrajectoryPoint(timestamp=0.5, position=Vector2(0.3, 4.5), velocity=Vector2(0.0, -1.0)),
        ]
        traj = Trajectory(object_id=1, points=tuple(pts), prediction_horizon=0.5)

        gt = [
            SimulatedEntity(object_id=1, class_id="person", position=Vector2(0.0, 5.0), velocity=Vector2(0.0, -1.0), timestamp=0.0),
            SimulatedEntity(object_id=1, class_id="person", position=Vector2(0.0, 4.5), velocity=Vector2(0.0, -1.0), timestamp=0.5),
        ]

        rmse = trajectory_position_rmse(traj, gt)
        self.assertAlmostEqual(rmse, 0.3)

    def test_prediction_error_at_horizon(self) -> None:
        """Evaluate error at a specific future horizon."""
        pts = [
            TrajectoryPoint(timestamp=1.0, position=Vector2(0.0, 5.0), velocity=Vector2(0.0, -1.0)),
            TrajectoryPoint(timestamp=1.5, position=Vector2(0.1, 4.5), velocity=Vector2(0.0, -1.0)),
            TrajectoryPoint(timestamp=2.0, position=Vector2(0.4, 4.0), velocity=Vector2(0.0, -1.0)),
        ]
        traj = Trajectory(object_id=1, points=tuple(pts), prediction_horizon=1.0)

        gt = [
            SimulatedEntity(object_id=1, class_id="person", position=Vector2(0.0, 5.0), velocity=Vector2(0.0, -1.0), timestamp=1.0),
            SimulatedEntity(object_id=1, class_id="person", position=Vector2(0.0, 4.5), velocity=Vector2(0.0, -1.0), timestamp=1.5),
            SimulatedEntity(object_id=1, class_id="person", position=Vector2(0.0, 4.0), velocity=Vector2(0.0, -1.0), timestamp=2.0),
        ]

        err_05 = prediction_error_at_horizon(traj, gt, target_time_offset=0.5)
        err_10 = prediction_error_at_horizon(traj, gt, target_time_offset=1.0)
        self.assertAlmostEqual(err_05, 0.1)
        self.assertAlmostEqual(err_10, 0.4)

    def test_cpa_error(self) -> None:
        """Evaluate CPA time and distance discrepancies."""
        cpa_pred = CPAResult(time_to_cpa=2.2, distance_at_cpa=0.15, relative_position_at_cpa=Vector2(0.1, 0.11), is_approaching=True)
        cpa_gt = CPAResult(time_to_cpa=2.0, distance_at_cpa=0.0, relative_position_at_cpa=Vector2(0.0, 0.0), is_approaching=True)

        dt_err, dist_err = cpa_error(cpa_pred, cpa_gt)
        self.assertAlmostEqual(dt_err, 0.2)
        self.assertAlmostEqual(dist_err, 0.15)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for TrajectoryPoint and Trajectory domain types."""

from dataclasses import FrozenInstanceError
import unittest

from app.core.types import Covariance4x4, Trajectory, TrajectoryPoint, Vector2


class TestTrajectory(unittest.TestCase):
    """Test suite for TrajectoryPoint and Trajectory."""

    def setUp(self) -> None:
        self.cov: Covariance4x4 = (
            (0.2, 0.0, 0.0, 0.0),
            (0.0, 0.2, 0.0, 0.0),
            (0.0, 0.0, 0.5, 0.0),
            (0.0, 0.0, 0.0, 0.5),
        )
        self.point1 = TrajectoryPoint(
            timestamp=1.5,
            position=Vector2(1.0, 4.0),
            velocity=Vector2(0.0, -1.0),
            covariance=self.cov,
        )
        self.point2 = TrajectoryPoint(
            timestamp=2.0,
            position=Vector2(1.0, 3.5),
            velocity=Vector2(0.0, -1.0),
        )

    def test_trajectory_point_construction(self) -> None:
        """Test valid creation of TrajectoryPoint."""
        pt = TrajectoryPoint(
            timestamp=1.0,
            position=Vector2(0.0, 5.0),
            velocity=Vector2(0.0, -1.0),
        )
        self.assertEqual(pt.timestamp, 1.0)
        self.assertEqual(pt.position, Vector2(0.0, 5.0))
        self.assertEqual(pt.velocity, Vector2(0.0, -1.0))
        self.assertIsNone(pt.covariance)

    def test_trajectory_point_immutability(self) -> None:
        """Test that TrajectoryPoint is immutable."""
        with self.assertRaises(FrozenInstanceError):
            self.point1.timestamp = 2.0  # type: ignore[misc]

    def test_trajectory_construction_with_points_sequence(self) -> None:
        """Test Trajectory creation from a list/tuple of points and verify tuple conversion."""
        traj = Trajectory(
            object_id=12,
            points=[self.point1, self.point2],
            prediction_horizon=2.0,
        )
        self.assertEqual(traj.object_id, 12)
        self.assertEqual(len(traj.points), 2)
        self.assertIsInstance(traj.points, tuple)
        self.assertEqual(traj.prediction_horizon, 2.0)

    def test_empty_trajectory(self) -> None:
        """Test that empty point sequences are permitted."""
        empty_traj = Trajectory(
            object_id=99,
            points=[],
            prediction_horizon=0.0,
        )
        self.assertEqual(len(empty_traj.points), 0)
        self.assertEqual(empty_traj.prediction_horizon, 0.0)

    def test_invalid_horizon_raises(self) -> None:
        """Test that negative or non-finite prediction horizon raises ValueError."""
        with self.assertRaises(ValueError):
            Trajectory(object_id=1, points=[], prediction_horizon=-0.5)

        with self.assertRaises(ValueError):
            Trajectory(object_id=1, points=[], prediction_horizon=float("nan"))

    def test_invalid_point_type_raises(self) -> None:
        """Test that non-TrajectoryPoint elements in points raise TypeError."""
        with self.assertRaises(TypeError):
            Trajectory(
                object_id=1,
                points=[self.point1, "invalid_point"],  # type: ignore[list-item]
                prediction_horizon=1.0,
            )

    def test_trajectory_immutability(self) -> None:
        """Test that Trajectory attributes cannot be mutated."""
        traj = Trajectory(
            object_id=1,
            points=[self.point1],
            prediction_horizon=1.0,
        )
        with self.assertRaises(FrozenInstanceError):
            traj.prediction_horizon = 3.0  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()

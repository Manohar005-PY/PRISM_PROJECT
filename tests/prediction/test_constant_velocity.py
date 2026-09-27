"""Unit tests for ConstantVelocityPredictor and uncertainty covariance propagation."""

import math
import unittest

from app.core.interfaces import Predictor
from app.core.types import Covariance4x4, ObjectState, Trajectory, Vector2
from app.prediction.constant_velocity import (
    ConstantVelocityPredictor,
    PredictionConfig,
)


class TestConstantVelocityPredictor(unittest.TestCase):
    """Test suite for kinematic trajectory predictor."""

    def setUp(self) -> None:
        self.config = PredictionConfig(timestep=0.5, max_horizon=5.0)
        self.predictor = ConstantVelocityPredictor(self.config)

    def test_protocol_conformance(self) -> None:
        """Verify predictor conforms to Predictor protocol."""
        self.assertIsInstance(self.predictor, Predictor)

    def test_zero_velocity_prediction(self) -> None:
        """Stationary object remains at same position across horizon."""
        state = ObjectState(
            object_id=1,
            class_id="chair",
            position=Vector2(2.0, 5.0),
            velocity=Vector2(0.0, 0.0),
            timestamp=10.0,
        )
        traj = self.predictor.predict(state, horizon=2.0)
        self.assertEqual(traj.object_id, 1)
        self.assertEqual(traj.prediction_horizon, 2.0)
        self.assertEqual(len(traj.points), 5)  # t=10.0, 10.5, 11.0, 11.5, 12.0

        for pt in traj.points:
            self.assertEqual(pt.position, Vector2(2.0, 5.0))
            self.assertEqual(pt.velocity, Vector2(0.0, 0.0))

    def test_linear_motion_prediction(self) -> None:
        """Moving object propagates linearly according to p0 + v * dt."""
        state = ObjectState(
            object_id=2,
            class_id="person",
            position=Vector2(0.0, 5.0),
            velocity=Vector2(1.0, -1.0),
            timestamp=0.0,
        )
        traj = self.predictor.predict(state, horizon=2.0)
        self.assertEqual(len(traj.points), 5)

        # Expected positions: t=0: (0,5), t=0.5: (0.5, 4.5), t=1.0: (1.0, 4.0), t=1.5: (1.5, 3.5), t=2.0: (2.0, 3.0)
        expected_positions = [
            Vector2(0.0, 5.0),
            Vector2(0.5, 4.5),
            Vector2(1.0, 4.0),
            Vector2(1.5, 3.5),
            Vector2(2.0, 3.0),
        ]
        for i, pt in enumerate(traj.points):
            self.assertAlmostEqual(pt.timestamp, i * 0.5)
            self.assertAlmostEqual(pt.position.x, expected_positions[i].x)
            self.assertAlmostEqual(pt.position.z, expected_positions[i].z)
            self.assertEqual(pt.velocity, Vector2(1.0, -1.0))

    def test_horizon_validation(self) -> None:
        """Negative horizon or exceeding max_horizon must raise ValueError."""
        state = ObjectState(1, "p", Vector2(0, 0), Vector2(0, 1), 0.0)
        with self.assertRaises(ValueError):
            self.predictor.predict(state, horizon=-0.5)
        with self.assertRaises(ValueError):
            self.predictor.predict(state, horizon=10.0)  # max_horizon is 5.0

    def test_covariance_propagation_none(self) -> None:
        """When state covariance is None, predicted points have covariance None."""
        state = ObjectState(1, "p", Vector2(0, 0), Vector2(0, 1), 0.0, covariance=None)
        traj = self.predictor.predict(state, horizon=1.0)
        for pt in traj.points:
            self.assertIsNone(pt.covariance)

    def test_covariance_propagation_with_process_noise(self) -> None:
        """Covariance propagates and grows monotonically when process noise is active."""
        P0: Covariance4x4 = (
            (0.1, 0.0, 0.0, 0.0),
            (0.0, 0.1, 0.0, 0.0),
            (0.0, 0.0, 0.2, 0.0),
            (0.0, 0.0, 0.0, 0.2),
        )
        state = ObjectState(1, "p", Vector2(0, 5), Vector2(0, -1), 0.0, covariance=P0)

        cfg_noise = PredictionConfig(timestep=0.5, max_horizon=3.0, process_noise_ax=0.5, process_noise_az=0.5)
        pred_noise = ConstantVelocityPredictor(cfg_noise)

        traj = pred_noise.predict(state, horizon=2.0)
        self.assertEqual(len(traj.points), 5)

        prev_pos_var = 0.0
        for i, pt in enumerate(traj.points):
            cov = pt.covariance
            self.assertIsNotNone(cov)
            # Check symmetry
            for r in range(4):
                for c in range(4):
                    self.assertAlmostEqual(cov[r][c], cov[c][r], delta=1e-7)  # type: ignore[index]

            pos_var = cov[0][0] + cov[1][1]  # type: ignore[index]
            if i > 0:
                self.assertGreater(pos_var, prev_pos_var)  # Positional uncertainty expands into the future
            prev_pos_var = pos_var


if __name__ == "__main__":
    unittest.main()

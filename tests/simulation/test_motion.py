"""Unit tests for ConstantVelocityModel."""

import unittest

from app.core.types import Vector2
from simulation.entities import EgoEntity, SimulatedEntity
from simulation.motion import ConstantVelocityModel


class TestConstantVelocityModel(unittest.TestCase):
    """Test suite for deterministic constant velocity motion propagation."""

    def test_zero_velocity(self) -> None:
        """Position must remain unchanged when velocity is zero."""
        p0 = Vector2(1.5, 3.0)
        v0 = Vector2(0.0, 0.0)
        p1 = ConstantVelocityModel.advance(p0, v0, dt=10.0)
        self.assertEqual(p1, p0)

    def test_positive_and_negative_velocity(self) -> None:
        """Test forward and backward propagation along X and Z."""
        p0 = Vector2(0.0, 5.0)
        v = Vector2(1.0, -2.0)
        p1 = ConstantVelocityModel.advance(p0, v, dt=2.0)
        self.assertEqual(p1, Vector2(2.0, 1.0))

    def test_zero_dt(self) -> None:
        """Advancing with dt=0 returns exact identical position."""
        p0 = Vector2(3.0, 4.0)
        v = Vector2(2.0, -1.0)
        p1 = ConstantVelocityModel.advance(p0, v, dt=0.0)
        self.assertEqual(p1, p0)

    def test_fractional_dt(self) -> None:
        """Test fractional timestamps and sub-second steps."""
        p0 = Vector2(0.0, 0.0)
        v = Vector2(1.0, 1.0)
        p1 = ConstantVelocityModel.advance(p0, v, dt=0.03333333333333333)
        self.assertAlmostEqual(p1.x, 1.0 / 30.0)
        self.assertAlmostEqual(p1.z, 1.0 / 30.0)

    def test_negative_dt_raises(self) -> None:
        """Negative time step must raise ValueError."""
        p0 = Vector2(0.0, 0.0)
        v = Vector2(1.0, 1.0)
        with self.assertRaises(ValueError):
            ConstantVelocityModel.advance(p0, v, dt=-0.1)

    def test_non_finite_dt_raises(self) -> None:
        """NaN or Inf dt must raise ValueError."""
        p0 = Vector2(0.0, 0.0)
        v = Vector2(1.0, 1.0)
        with self.assertRaises(ValueError):
            ConstantVelocityModel.advance(p0, v, dt=float("nan"))
        with self.assertRaises(ValueError):
            ConstantVelocityModel.advance(p0, v, dt=float("inf"))

    def test_advance_entity(self) -> None:
        """Test advancing a SimulatedEntity advances position and timestamp while keeping velocity & id."""
        entity = SimulatedEntity(
            object_id=1,
            class_id="person",
            position=Vector2(0.0, 5.0),
            velocity=Vector2(0.0, -1.0),
            timestamp=10.0,
        )
        advanced = ConstantVelocityModel.advance_entity(entity, dt=1.5)
        self.assertEqual(advanced.object_id, 1)
        self.assertEqual(advanced.class_id, "person")
        self.assertEqual(advanced.position, Vector2(0.0, 3.5))
        self.assertEqual(advanced.velocity, Vector2(0.0, -1.0))
        self.assertEqual(advanced.timestamp, 11.5)

    def test_advance_ego(self) -> None:
        """Test advancing an EgoEntity."""
        ego = EgoEntity(
            position=Vector2(0.0, 0.0),
            velocity=Vector2(0.0, 1.0),
            timestamp=5.0,
        )
        advanced = ConstantVelocityModel.advance_ego(ego, dt=2.0)
        self.assertEqual(advanced.position, Vector2(0.0, 2.0))
        self.assertEqual(advanced.velocity, Vector2(0.0, 1.0))
        self.assertEqual(advanced.timestamp, 7.0)


if __name__ == "__main__":
    unittest.main()

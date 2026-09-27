"""Unit tests for ObjectState domain type and Covariance4x4 representation."""

from dataclasses import FrozenInstanceError
import unittest

from app.core.types import (
    Covariance4x4,
    ObjectState,
    Vector2,
    validate_covariance_4x4,
)


class TestObjectState(unittest.TestCase):
    """Test suite for ObjectState."""

    def setUp(self) -> None:
        self.pos = Vector2(x=1.2, z=4.5)
        self.vel = Vector2(x=-0.3, z=-0.8)
        self.valid_cov: Covariance4x4 = (
            (0.1, 0.0, 0.0, 0.0),
            (0.0, 0.1, 0.0, 0.0),
            (0.0, 0.0, 0.5, 0.0),
            (0.0, 0.0, 0.0, 0.5),
        )

    def test_valid_construction_with_defaults(self) -> None:
        """Test valid creation of ground-plane ObjectState."""
        state = ObjectState(
            object_id=12,
            class_id="person",
            position=self.pos,
            velocity=self.vel,
            timestamp=123.45,
        )
        self.assertEqual(state.object_id, 12)
        self.assertEqual(state.class_id, "person")
        self.assertEqual(state.position, self.pos)
        self.assertEqual(state.velocity, self.vel)
        self.assertEqual(state.timestamp, 123.45)
        self.assertIsNone(state.covariance)
        self.assertEqual(state.confidence, 1.0)
        self.assertEqual(state.age, 1)
        self.assertEqual(state.time_since_update, 0.0)

    def test_valid_construction_with_covariance(self) -> None:
        """Test ObjectState with explicit 4x4 state covariance."""
        state = ObjectState(
            object_id=1,
            class_id=0,
            position=self.pos,
            velocity=self.vel,
            timestamp=10.0,
            covariance=self.valid_cov,
            confidence=0.85,
            age=15,
            time_since_update=0.033,
        )
        self.assertEqual(state.covariance, self.valid_cov)
        self.assertEqual(state.confidence, 0.85)
        self.assertEqual(state.age, 15)
        self.assertEqual(state.time_since_update, 0.033)

    def test_state_represents_ground_plane_coordinates(self) -> None:
        """Verify that ObjectState uses physical ground-plane coordinates (Vector2 x/z) not image coordinates."""
        state = ObjectState(
            object_id=5,
            class_id="obstacle",
            position=Vector2(x=0.5, z=2.0),  # 0.5m right, 2.0m forward
            velocity=Vector2(x=0.0, z=-1.2),  # 1.2 m/s approaching
            timestamp=1.0,
        )
        self.assertEqual(state.position.x, 0.5)
        self.assertEqual(state.position.z, 2.0)
        self.assertEqual(state.velocity.x, 0.0)
        self.assertEqual(state.velocity.z, -1.2)

    def test_covariance_shape_validation(self) -> None:
        """Test that invalid covariance structures raise ValueError."""
        # Non 4x4 shape (3 rows)
        bad_cov_3rows = (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
        )
        with self.assertRaises(ValueError):
            validate_covariance_4x4(bad_cov_3rows)  # type: ignore[arg-type]

        # Non 4-length row
        bad_cov_ragged = (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        with self.assertRaises(ValueError):
            validate_covariance_4x4(bad_cov_ragged)  # type: ignore[arg-type]

        # Non-finite entry
        bad_cov_nan = (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, float("nan"), 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        with self.assertRaises(ValueError):
            validate_covariance_4x4(bad_cov_nan)

    def test_invalid_attributes_raise(self) -> None:
        """Test that invalid confidence, age, and types raise errors."""
        with self.assertRaises(ValueError):
            ObjectState(
                object_id=1,
                class_id="person",
                position=self.pos,
                velocity=self.vel,
                timestamp=1.0,
                confidence=-0.1,
            )

        with self.assertRaises(ValueError):
            ObjectState(
                object_id=1,
                class_id="person",
                position=self.pos,
                velocity=self.vel,
                timestamp=1.0,
                age=-1,
            )

        with self.assertRaises(TypeError):
            ObjectState(
                object_id=1,
                class_id="person",
                position=(1.0, 2.0),  # type: ignore[arg-type]
                velocity=self.vel,
                timestamp=1.0,
            )

    def test_immutability(self) -> None:
        """Test that ObjectState is immutable."""
        state = ObjectState(
            object_id=1,
            class_id="person",
            position=self.pos,
            velocity=self.vel,
            timestamp=1.0,
        )
        with self.assertRaises(FrozenInstanceError):
            state.position = Vector2(0.0, 0.0)  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()

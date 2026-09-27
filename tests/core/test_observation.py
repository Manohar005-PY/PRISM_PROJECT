"""Unit tests for ObjectObservation domain type."""

from dataclasses import FrozenInstanceError
import unittest

from app.core.types import BoundingBox, ObjectObservation


class TestObjectObservation(unittest.TestCase):
    """Test suite for ObjectObservation."""

    def setUp(self) -> None:
        self.bbox = BoundingBox(x_min=100.0, y_min=150.0, x_max=200.0, y_max=350.0)

    def test_valid_construction_with_defaults(self) -> None:
        """Test valid construction of perception observation without object_id."""
        obs = ObjectObservation(
            class_id="person",
            confidence=0.92,
            bbox=self.bbox,
            timestamp=1000.5,
        )
        self.assertEqual(obs.class_id, "person")
        self.assertAlmostEqual(obs.confidence, 0.92)
        self.assertEqual(obs.bbox, self.bbox)
        self.assertEqual(obs.timestamp, 1000.5)
        self.assertIsNone(obs.object_id)

    def test_valid_construction_with_object_id_and_int_class(self) -> None:
        """Test observation when tracked or using integer class_id."""
        obs = ObjectObservation(
            object_id=42,
            class_id=0,
            confidence=1.0,
            bbox=self.bbox,
            timestamp=1000.5,
        )
        self.assertEqual(obs.object_id, 42)
        self.assertEqual(obs.class_id, 0)
        self.assertEqual(obs.confidence, 1.0)

    def test_confidence_boundary_values(self) -> None:
        """Test valid confidence boundaries 0.0 and 1.0."""
        obs_zero = ObjectObservation(
            class_id="obstacle",
            confidence=0.0,
            bbox=self.bbox,
            timestamp=10.0,
        )
        self.assertEqual(obs_zero.confidence, 0.0)

        obs_one = ObjectObservation(
            class_id="obstacle",
            confidence=1.0,
            bbox=self.bbox,
            timestamp=10.0,
        )
        self.assertEqual(obs_one.confidence, 1.0)

    def test_invalid_confidence_raises(self) -> None:
        """Test that confidence < 0.0 or > 1.0 raises ValueError."""
        with self.assertRaises(ValueError):
            ObjectObservation(
                class_id="person",
                confidence=-0.01,
                bbox=self.bbox,
                timestamp=10.0,
            )

        with self.assertRaises(ValueError):
            ObjectObservation(
                class_id="person",
                confidence=1.01,
                bbox=self.bbox,
                timestamp=10.0,
            )

        with self.assertRaises(ValueError):
            ObjectObservation(
                class_id="person",
                confidence=float("nan"),
                bbox=self.bbox,
                timestamp=10.0,
            )

    def test_invalid_types_raise(self) -> None:
        """Test that invalid types for bbox, class_id, and object_id raise TypeError."""
        with self.assertRaises(TypeError):
            ObjectObservation(
                class_id="person",
                confidence=0.9,
                bbox="not_a_bbox",  # type: ignore[arg-type]
                timestamp=10.0,
            )

        with self.assertRaises(TypeError):
            ObjectObservation(
                class_id=[1, 2],  # type: ignore[arg-type]
                confidence=0.9,
                bbox=self.bbox,
                timestamp=10.0,
            )

        with self.assertRaises(TypeError):
            ObjectObservation(
                object_id="not_an_int",  # type: ignore[arg-type]
                class_id="person",
                confidence=0.9,
                bbox=self.bbox,
                timestamp=10.0,
            )

    def test_immutability(self) -> None:
        """Test that ObjectObservation cannot be modified."""
        obs = ObjectObservation(
            class_id="person",
            confidence=0.9,
            bbox=self.bbox,
            timestamp=10.0,
        )
        with self.assertRaises(FrozenInstanceError):
            obs.confidence = 0.5  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()

"""Unit tests for BoundingBox domain type."""

from dataclasses import FrozenInstanceError
import unittest

from app.core.types import BoundingBox


class TestBoundingBox(unittest.TestCase):
    """Test suite for BoundingBox."""

    def test_construction_and_properties(self) -> None:
        """Test valid bounding box creation and computed geometric properties."""
        box = BoundingBox(x_min=10.0, y_min=20.0, x_max=50.0, y_max=100.0)
        self.assertEqual(box.x_min, 10.0)
        self.assertEqual(box.y_min, 20.0)
        self.assertEqual(box.x_max, 50.0)
        self.assertEqual(box.y_max, 100.0)
        self.assertEqual(box.width, 40.0)
        self.assertEqual(box.height, 80.0)
        self.assertEqual(box.center, (30.0, 60.0))
        self.assertEqual(box.bottom_center, (30.0, 100.0))

    def test_zero_area_box(self) -> None:
        """Test that single-point bounding boxes are valid (zero width and height)."""
        box = BoundingBox(x_min=15.0, y_min=25.0, x_max=15.0, y_max=25.0)
        self.assertEqual(box.width, 0.0)
        self.assertEqual(box.height, 0.0)
        self.assertEqual(box.center, (15.0, 25.0))
        self.assertEqual(box.bottom_center, (15.0, 25.0))

    def test_invalid_coordinates_raise(self) -> None:
        """Test that inverted coordinates (x_max < x_min or y_max < y_min) raise ValueError."""
        with self.assertRaises(ValueError):
            BoundingBox(x_min=50.0, y_min=20.0, x_max=10.0, y_max=100.0)

        with self.assertRaises(ValueError):
            BoundingBox(x_min=10.0, y_min=100.0, x_max=50.0, y_max=20.0)

    def test_non_finite_coordinates_raise(self) -> None:
        """Test that NaN and Inf coordinates raise ValueError."""
        with self.assertRaises(ValueError):
            BoundingBox(x_min=float("nan"), y_min=0.0, x_max=10.0, y_max=10.0)
        with self.assertRaises(ValueError):
            BoundingBox(x_min=0.0, y_min=0.0, x_max=float("inf"), y_max=10.0)

    def test_immutability(self) -> None:
        """Test that BoundingBox attributes cannot be reassigned."""
        box = BoundingBox(x_min=0.0, y_min=0.0, x_max=10.0, y_max=10.0)
        with self.assertRaises(FrozenInstanceError):
            box.x_min = 5.0  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()

"""Unit tests for Vector2 domain type."""

from dataclasses import FrozenInstanceError
import math
import unittest

from app.core.types import Vector2


class TestVector2(unittest.TestCase):
    """Test suite for Vector2."""

    def test_construction_and_properties(self) -> None:
        """Test valid construction and property access."""
        v = Vector2(x=3.0, z=4.0)
        self.assertEqual(v.x, 3.0)
        self.assertEqual(v.z, 4.0)
        self.assertAlmostEqual(v.magnitude, 5.0)

    def test_integer_conversion_to_float(self) -> None:
        """Test that integer arguments are stored as floats."""
        v = Vector2(x=1, z=2)
        self.assertIsInstance(v.x, float)
        self.assertIsInstance(v.z, float)
        self.assertEqual(v.x, 1.0)
        self.assertEqual(v.z, 2.0)

    def test_addition(self) -> None:
        """Test vector addition."""
        v1 = Vector2(1.5, -2.0)
        v2 = Vector2(2.5, 5.0)
        v3 = v1 + v2
        self.assertEqual(v3, Vector2(4.0, 3.0))

    def test_subtraction(self) -> None:
        """Test vector subtraction."""
        v1 = Vector2(5.0, 10.0)
        v2 = Vector2(2.0, 3.0)
        v3 = v1 - v2
        self.assertEqual(v3, Vector2(3.0, 7.0))

    def test_scalar_multiplication(self) -> None:
        """Test vector scalar multiplication and reverse scalar multiplication."""
        v = Vector2(2.0, -3.5)
        self.assertEqual(v * 2, Vector2(4.0, -7.0))
        self.assertEqual(v * 2.5, Vector2(5.0, -8.75))
        self.assertEqual(3 * v, Vector2(6.0, -10.5))

    def test_magnitude(self) -> None:
        """Test magnitude calculation for zero, positive, and negative components."""
        zero = Vector2(0.0, 0.0)
        self.assertEqual(zero.magnitude, 0.0)

        v_neg = Vector2(-3.0, -4.0)
        self.assertAlmostEqual(v_neg.magnitude, 5.0)

    def test_equality_and_hashability(self) -> None:
        """Test value equality and set/dict membership."""
        v1 = Vector2(1.0, 2.0)
        v2 = Vector2(1.0, 2.0)
        v3 = Vector2(1.0, 2.1)
        self.assertEqual(v1, v2)
        self.assertNotEqual(v1, v3)
        self.assertEqual(hash(v1), hash(v2))

        vector_set = {v1, v2, v3}
        self.assertEqual(len(vector_set), 2)

    def test_immutability(self) -> None:
        """Test that Vector2 cannot be mutated after creation."""
        v = Vector2(1.0, 2.0)
        with self.assertRaises(FrozenInstanceError):
            v.x = 5.0  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            v.z = 5.0  # type: ignore[misc]

    def test_invalid_values_raise(self) -> None:
        """Test that NaN and Inf raise ValueError."""
        with self.assertRaises(ValueError):
            Vector2(float("nan"), 1.0)
        with self.assertRaises(ValueError):
            Vector2(1.0, float("inf"))
        with self.assertRaises(ValueError):
            Vector2(float("-inf"), 1.0)
        with self.assertRaises(ValueError):
            Vector2("1.0", 2.0)  # type: ignore[arg-type]

    def test_invalid_operands_raise(self) -> None:
        """Test that invalid operator types raise TypeError."""
        v = Vector2(1.0, 2.0)
        with self.assertRaises(TypeError):
            _ = v + "invalid"  # type: ignore[operator]
        with self.assertRaises(TypeError):
            _ = v - 5  # type: ignore[operator]
        with self.assertRaises(TypeError):
            _ = v * "invalid"  # type: ignore[operator]


if __name__ == "__main__":
    unittest.main()

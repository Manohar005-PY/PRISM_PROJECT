"""Unit tests for Decision domain type."""

from dataclasses import FrozenInstanceError
import unittest

from app.core.enums import DecisionType
from app.core.types import Decision


class TestDecision(unittest.TestCase):
    """Test suite for Decision."""

    def test_valid_construction_minimal(self) -> None:
        """Test minimal Decision creation."""
        decision = Decision(decision_type=DecisionType.NO_ACTION)
        self.assertEqual(decision.decision_type, DecisionType.NO_ACTION)
        self.assertIsNone(decision.object_id)
        self.assertEqual(decision.timestamp, 0.0)
        self.assertIsNone(decision.message_id)

    def test_valid_construction_full(self) -> None:
        """Test fully populated Decision."""
        decision = Decision(
            decision_type=DecisionType.WARNING,
            object_id=7,
            timestamp=12.5,
            message_id="approaching_pedestrian_ahead",
        )
        self.assertEqual(decision.decision_type, DecisionType.WARNING)
        self.assertEqual(decision.object_id, 7)
        self.assertEqual(decision.timestamp, 12.5)
        self.assertEqual(decision.message_id, "approaching_pedestrian_ahead")

    def test_invalid_types_raise(self) -> None:
        """Test that non-DecisionType decision_type raises TypeError."""
        with self.assertRaises(TypeError):
            Decision(decision_type="WARNING")  # type: ignore[arg-type]

        with self.assertRaises(TypeError):
            Decision(
                decision_type=DecisionType.WARNING,
                object_id="not_an_int",  # type: ignore[arg-type]
            )

        with self.assertRaises(TypeError):
            Decision(
                decision_type=DecisionType.WARNING,
                message_id=123,  # type: ignore[arg-type]
            )

    def test_immutability(self) -> None:
        """Test that Decision is immutable."""
        decision = Decision(decision_type=DecisionType.STOP, object_id=1)
        with self.assertRaises(FrozenInstanceError):
            decision.decision_type = DecisionType.NO_ACTION  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()

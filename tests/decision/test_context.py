"""Unit tests for DecisionContext validation and properties."""

import math
import unittest

from app.core.enums import RiskLevel
from app.core.types import ObjectState, RiskAssessment, Vector2
from app.decision.context import DecisionContext


class TestDecisionContext(unittest.TestCase):
    """Test suite for DecisionContext."""

    def test_default_context(self) -> None:
        ctx = DecisionContext()
        self.assertEqual(ctx.ego_position, Vector2(0.0, 0.0))
        self.assertEqual(ctx.current_heading, 0.0)
        self.assertAlmostEqual(ctx.max_turn_angle, math.pi / 2.0)
        self.assertEqual(ctx.minimum_clearance, 1.0)
        self.assertEqual(ctx.current_speed, 1.0)
        self.assertEqual(ctx.maximum_speed, 1.5)

    def test_valid_custom_context(self) -> None:
        obs = ObjectState(1, "person", Vector2(0, 3), Vector2(0, -1), 0.0)
        risk = RiskAssessment(1, RiskLevel.HIGH, 75.0, ttc=1.5, cpa_distance=0.0)
        ctx = DecisionContext(
            ego_position=Vector2(1.0, 2.0),
            current_heading=0.2,
            preferred_heading=0.0,
            max_turn_angle=1.2,
            minimum_clearance=1.2,
            current_speed=1.1,
            maximum_speed=1.8,
            risk_assessment=risk,
            predicted_obstacles=(obs,),
            timestamp=5.0,
        )
        self.assertEqual(len(ctx.predicted_obstacles), 1)
        self.assertEqual(ctx.risk_assessment.risk_level, RiskLevel.HIGH)  # type: ignore[union-attr]

    def test_invalid_parameters_raise(self) -> None:
        with self.assertRaises(ValueError):
            DecisionContext(max_turn_angle=-0.5)
        with self.assertRaises(ValueError):
            DecisionContext(minimum_clearance=0.0)
        with self.assertRaises(ValueError):
            DecisionContext(current_speed=2.0, maximum_speed=1.0)


if __name__ == "__main__":
    unittest.main()

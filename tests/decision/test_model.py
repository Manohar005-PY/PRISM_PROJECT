"""Unit tests for decision domain types, enum priorities, and immutability."""

import unittest

from app.core.enums import RiskLevel
from app.decision.model import (
    DecisionReason,
    NavigationAction,
    NavigationDecision,
    get_action_priority,
)


class TestDecisionModel(unittest.TestCase):
    """Test suite for decision model types and priority rules."""

    def test_action_priorities(self) -> None:
        """Verify explicit integer priority hierarchy."""
        self.assertGreater(get_action_priority(NavigationAction.EMERGENCY_STOP), get_action_priority(NavigationAction.STOP))
        self.assertGreater(get_action_priority(NavigationAction.STOP), get_action_priority(NavigationAction.TURN_LEFT))
        self.assertEqual(get_action_priority(NavigationAction.TURN_LEFT), get_action_priority(NavigationAction.TURN_RIGHT))
        self.assertGreater(get_action_priority(NavigationAction.TURN_LEFT), get_action_priority(NavigationAction.SLOW_DOWN))
        self.assertGreater(get_action_priority(NavigationAction.SLOW_DOWN), get_action_priority(NavigationAction.CAUTION))
        self.assertGreater(get_action_priority(NavigationAction.CAUTION), get_action_priority(NavigationAction.CONTINUE))

    def test_navigation_decision_valid_and_immutable(self) -> None:
        decision = NavigationDecision(
            action=NavigationAction.TURN_LEFT,
            reason=DecisionReason.SAFE_PATH_AVAILABLE,
            confidence=0.95,
            primary_threat_id=1,
            recommended_heading=-0.35,
            risk_level=RiskLevel.HIGH,
            timestamp=12.5,
        )
        self.assertEqual(decision.action, NavigationAction.TURN_LEFT)
        self.assertEqual(decision.reason, DecisionReason.SAFE_PATH_AVAILABLE)
        self.assertEqual(decision.confidence, 0.95)
        self.assertEqual(decision.primary_threat_id, 1)
        self.assertEqual(decision.recommended_heading, -0.35)
        self.assertEqual(decision.risk_level, RiskLevel.HIGH)
        self.assertEqual(decision.timestamp, 12.5)

        # Immutability
        with self.assertRaises(Exception):
            decision.action = NavigationAction.STOP  # type: ignore[misc]

    def test_invalid_confidence_raises(self) -> None:
        with self.assertRaises(ValueError):
            NavigationDecision(
                action=NavigationAction.CONTINUE,
                reason=DecisionReason.NO_HAZARD,
                confidence=1.5,
                primary_threat_id=None,
                recommended_heading=None,
                risk_level=RiskLevel.SAFE,
                timestamp=0.0,
            )


if __name__ == "__main__":
    unittest.main()

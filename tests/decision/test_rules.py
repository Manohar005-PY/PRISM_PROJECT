"""Unit tests for deterministic safety rules."""

import unittest

from app.core.enums import RiskLevel
from app.core.types import RiskAssessment
from app.decision.context import DecisionContext
from app.decision.heading import SafeHeading
from app.decision.model import DecisionReason, NavigationAction
from app.decision.rules import evaluate_safety_rules


class TestSafetyRules(unittest.TestCase):
    """Test suite for safety rule evaluations."""

    def test_critical_risk_emergency_stop(self) -> None:
        risk = RiskAssessment(1, RiskLevel.CRITICAL, 95.0, ttc=0.5)
        ctx = DecisionContext(risk_assessment=risk)
        act, reason = evaluate_safety_rules(RiskLevel.CRITICAL, None, ctx)
        self.assertEqual(act, NavigationAction.EMERGENCY_STOP)
        self.assertEqual(reason, DecisionReason.EMERGENCY_COLLISION_RISK)

    def test_critical_risk_standard_stop(self) -> None:
        risk = RiskAssessment(1, RiskLevel.CRITICAL, 90.0, ttc=1.5)
        ctx = DecisionContext(risk_assessment=risk)
        act, reason = evaluate_safety_rules(RiskLevel.CRITICAL, None, ctx)
        self.assertEqual(act, NavigationAction.STOP)
        self.assertEqual(reason, DecisionReason.CRITICAL_COLLISION_RISK)

    def test_high_risk_with_safe_left_turn(self) -> None:
        best_heading = SafeHeading(angle=-0.35, clearance=3.5, deviation_from_current=0.35, is_valid=True)
        ctx = DecisionContext()
        act, reason = evaluate_safety_rules(RiskLevel.HIGH, best_heading, ctx)
        self.assertEqual(act, NavigationAction.TURN_LEFT)
        self.assertEqual(reason, DecisionReason.SAFE_PATH_AVAILABLE)

    def test_high_risk_with_safe_right_turn(self) -> None:
        best_heading = SafeHeading(angle=0.40, clearance=3.5, deviation_from_current=0.40, is_valid=True)
        ctx = DecisionContext()
        act, reason = evaluate_safety_rules(RiskLevel.HIGH, best_heading, ctx)
        self.assertEqual(act, NavigationAction.TURN_RIGHT)
        self.assertEqual(reason, DecisionReason.SAFE_PATH_AVAILABLE)

    def test_high_risk_no_safe_heading_stops(self) -> None:
        ctx = DecisionContext()
        act, reason = evaluate_safety_rules(RiskLevel.HIGH, None, ctx)
        self.assertEqual(act, NavigationAction.STOP)
        self.assertEqual(reason, DecisionReason.NO_SAFE_PATH)

    def test_medium_risk_slow_down(self) -> None:
        best_heading = SafeHeading(angle=0.0, clearance=3.0, deviation_from_current=0.0, is_valid=True)
        ctx = DecisionContext()
        act, reason = evaluate_safety_rules(RiskLevel.MEDIUM, best_heading, ctx)
        self.assertEqual(act, NavigationAction.SLOW_DOWN)
        self.assertEqual(reason, DecisionReason.MEDIUM_COLLISION_RISK)

    def test_safe_level_continue(self) -> None:
        ctx = DecisionContext()
        act, reason = evaluate_safety_rules(RiskLevel.SAFE, None, ctx)
        self.assertEqual(act, NavigationAction.CONTINUE)
        self.assertEqual(reason, DecisionReason.NO_HAZARD)


if __name__ == "__main__":
    unittest.main()

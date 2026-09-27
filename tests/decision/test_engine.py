"""Unit tests for DecisionEngine pipeline orchestration and protocol adapter."""

import unittest

from app.core.enums import DecisionType, RiskLevel
from app.core.interfaces import DecisionEngine as DecisionEngineProtocol
from app.core.types import ObjectState, RiskAssessment, Vector2
from app.decision.context import DecisionContext
from app.decision.engine import DecisionEngine
from app.decision.model import NavigationAction


class TestDecisionEngine(unittest.TestCase):
    """Test suite for DecisionEngine."""

    def setUp(self) -> None:
        self.engine = DecisionEngine()

    def test_protocol_conformance(self) -> None:
        self.assertIsInstance(self.engine, DecisionEngineProtocol)

    def test_adapter_decide_safe(self) -> None:
        risk = RiskAssessment(1, RiskLevel.SAFE, 10.0, timestamp=1.0)
        decision = self.engine.decide(risk)
        self.assertEqual(decision.decision_type, DecisionType.NO_ACTION)
        self.assertEqual(decision.object_id, 1)

    def test_adapter_decide_critical(self) -> None:
        risk = RiskAssessment(1, RiskLevel.CRITICAL, 95.0, ttc=0.4, timestamp=2.0)
        decision = self.engine.decide(risk)
        self.assertEqual(decision.decision_type, DecisionType.STOP)
        self.assertEqual(decision.object_id, 1)

    def test_full_navigation_pipeline_evasive_turn(self) -> None:
        """Obstacle directly ahead with open sides triggers safe turn recommendation."""
        obs = ObjectState(1, "person", Vector2(0.0, 2.0), Vector2(0.0, -1.0), 0.0)
        risk = RiskAssessment(1, RiskLevel.HIGH, 75.0, ttc=1.5, cpa_distance=0.0)
        ctx = DecisionContext(
            predicted_obstacles=(obs,),
            risk_assessment=risk,
            minimum_clearance=1.2,
        )

        nav_dec = self.engine.decide_navigation(ctx)
        self.assertIn(nav_dec.action, (NavigationAction.TURN_LEFT, NavigationAction.TURN_RIGHT))
        self.assertIsNotNone(nav_dec.recommended_heading)


if __name__ == "__main__":
    unittest.main()

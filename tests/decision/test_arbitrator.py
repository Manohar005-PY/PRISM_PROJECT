"""Unit tests for DecisionArbitrator, lateral hysteresis, and emergency escalation."""

import unittest

from app.core.enums import RiskLevel
from app.decision.arbitrator import DecisionArbitrator, DecisionMemory
from app.decision.context import DecisionContext
from app.decision.model import DecisionReason, NavigationAction, NavigationDecision
from app.decision.planner import CandidateAction


class TestDecisionArbitrator(unittest.TestCase):
    """Test suite for DecisionArbitrator."""

    def setUp(self) -> None:
        self.arbitrator = DecisionArbitrator(hysteresis_margin=0.20)
        self.memory = DecisionMemory()

    def test_emergency_stop_escalation_bypasses_hysteresis(self) -> None:
        """EMERGENCY_STOP immediately overrides previous state without delay."""
        self.memory.previous_action = NavigationAction.TURN_LEFT
        self.memory.previous_risk_level = RiskLevel.MEDIUM

        ctx = DecisionContext()
        cand = CandidateAction(NavigationAction.EMERGENCY_STOP, 0.05, None, DecisionReason.EMERGENCY_COLLISION_RISK, True)
        decision = self.arbitrator.arbitrate(
            candidates=[cand],
            rules_action=NavigationAction.EMERGENCY_STOP,
            rules_reason=DecisionReason.EMERGENCY_COLLISION_RISK,
            context=ctx,
            memory=self.memory,
        )
        self.assertEqual(decision.action, NavigationAction.EMERGENCY_STOP)

    def test_lateral_hysteresis_suppresses_small_cost_jitter(self) -> None:
        """When turning left, a tiny right turn advantage (< margin) does not flip the decision."""
        self.memory.previous_action = NavigationAction.TURN_LEFT
        self.memory.previous_risk_level = RiskLevel.HIGH

        ctx = DecisionContext()
        # Left turn cost = 0.35, Right turn cost = 0.30 (difference 0.05 < margin 0.20)
        cand_right = CandidateAction(NavigationAction.TURN_RIGHT, 0.30, 0.40, DecisionReason.SAFE_PATH_AVAILABLE, True)
        cand_left = CandidateAction(NavigationAction.TURN_LEFT, 0.35, -0.40, DecisionReason.SAFE_PATH_AVAILABLE, True)

        decision = self.arbitrator.arbitrate(
            candidates=[cand_right, cand_left],
            rules_action=NavigationAction.TURN_RIGHT,
            rules_reason=DecisionReason.SAFE_PATH_AVAILABLE,
            context=ctx,
            memory=self.memory,
        )
        # Hysteresis retains TURN_LEFT
        self.assertEqual(decision.action, NavigationAction.TURN_LEFT)
        self.assertEqual(decision.reason, DecisionReason.HYSTERESIS_HOLD)

    def test_lateral_hysteresis_allows_large_advantage_flip(self) -> None:
        """When turning left, a large right turn advantage (> margin) flips the decision."""
        self.memory.previous_action = NavigationAction.TURN_LEFT
        self.memory.previous_risk_level = RiskLevel.HIGH

        ctx = DecisionContext()
        # Left turn cost = 0.60, Right turn cost = 0.20 (difference 0.40 > margin 0.20)
        cand_right = CandidateAction(NavigationAction.TURN_RIGHT, 0.20, 0.40, DecisionReason.SAFE_PATH_AVAILABLE, True)
        cand_left = CandidateAction(NavigationAction.TURN_LEFT, 0.60, -0.40, DecisionReason.SAFE_PATH_AVAILABLE, True)

        decision = self.arbitrator.arbitrate(
            candidates=[cand_right, cand_left],
            rules_action=NavigationAction.TURN_RIGHT,
            rules_reason=DecisionReason.SAFE_PATH_AVAILABLE,
            context=ctx,
            memory=self.memory,
        )
        self.assertEqual(decision.action, NavigationAction.TURN_RIGHT)


if __name__ == "__main__":
    unittest.main()

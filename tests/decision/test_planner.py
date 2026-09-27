"""Unit tests for LocalMotionPlanner candidate evaluation and hard constraint filtering."""

import unittest

from app.core.enums import RiskLevel
from app.core.types import ObjectState, RiskAssessment, Vector2
from app.decision.context import DecisionContext
from app.decision.heading import SafeHeading, compute_safe_headings
from app.decision.model import NavigationAction
from app.decision.planner import LocalMotionPlanner


class TestLocalMotionPlanner(unittest.TestCase):
    """Test suite for LocalMotionPlanner."""

    def setUp(self) -> None:
        self.planner = LocalMotionPlanner()

    def test_open_environment_continue_lowest_cost(self) -> None:
        """In open space under SAFE risk, CONTINUE is valid and has lowest cost."""
        ctx = DecisionContext()
        safe_headings = compute_safe_headings(ctx, num_samples=11)
        candidates = self.planner.plan_candidates(ctx, safe_headings)

        self.assertGreater(len(candidates), 0)
        best = candidates[0]
        self.assertEqual(best.action, NavigationAction.CONTINUE)
        self.assertTrue(best.is_safe)

    def test_blocked_forward_path_rejects_continue(self) -> None:
        """When forward is blocked under HIGH risk, CONTINUE is rejected from safe candidates."""
        obs = ObjectState(1, "person", Vector2(0.0, 2.0), Vector2(0.0, -1.0), 0.0)
        risk = RiskAssessment(1, RiskLevel.HIGH, 80.0, ttc=1.2)
        ctx = DecisionContext(predicted_obstacles=(obs,), risk_assessment=risk, minimum_clearance=1.5)

        safe_headings = compute_safe_headings(ctx, num_samples=37)
        candidates = self.planner.plan_candidates(ctx, safe_headings)

        actions = [c.action for c in candidates]
        self.assertNotIn(NavigationAction.CONTINUE, actions)
        # Should contain evasive turns or stop
        self.assertTrue(any(a in (NavigationAction.TURN_LEFT, NavigationAction.TURN_RIGHT, NavigationAction.STOP) for a in actions))


if __name__ == "__main__":
    unittest.main()

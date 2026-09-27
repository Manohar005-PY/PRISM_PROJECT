"""Unit tests for decision evaluation metrics."""

import math
import unittest

from app.core.enums import RiskLevel
from app.decision.metrics import (
    collision_avoidance_rate,
    decision_accuracy,
    decision_switch_rate,
    mean_heading_deviation,
    unnecessary_stop_rate,
)
from app.decision.model import (
    DecisionReason,
    NavigationAction,
    NavigationDecision,
)


class TestDecisionMetrics(unittest.TestCase):
    """Test suite for decision performance metrics."""

    def test_decision_accuracy_and_stop_rate(self) -> None:
        preds = [NavigationAction.CONTINUE, NavigationAction.TURN_LEFT, NavigationAction.STOP, NavigationAction.STOP]
        refs = [NavigationAction.CONTINUE, NavigationAction.TURN_LEFT, NavigationAction.CONTINUE, NavigationAction.STOP]

        acc = decision_accuracy(preds, refs)
        self.assertEqual(acc, 75.0)  # 3 of 4 match

        # Item 3: pred=STOP but ref=CONTINUE => 1 false stop out of 2 safe references (50%)
        stop_rate = unnecessary_stop_rate(preds, refs)
        self.assertAlmostEqual(stop_rate, 50.0)

    def test_avoidance_rate_and_switch_rate(self) -> None:
        d1 = NavigationDecision(NavigationAction.CONTINUE, DecisionReason.NO_HAZARD, 1.0, None, None, RiskLevel.SAFE, 0.0)
        d2 = NavigationDecision(NavigationAction.TURN_LEFT, DecisionReason.SAFE_PATH_AVAILABLE, 1.0, 1, -0.4, RiskLevel.HIGH, 0.5)
        d3 = NavigationDecision(NavigationAction.TURN_LEFT, DecisionReason.SAFE_PATH_AVAILABLE, 1.0, 1, -0.4, RiskLevel.HIGH, 1.0)
        d4 = NavigationDecision(NavigationAction.STOP, DecisionReason.CRITICAL_COLLISION_RISK, 1.0, 1, None, RiskLevel.CRITICAL, 1.5)

        decisions = [d1, d2, d3, d4]

        avoid_rate = collision_avoidance_rate(decisions, ground_truth_hazard_occurred=True)
        self.assertEqual(avoid_rate, 75.0)  # 3 of 4 are intervention actions

        # Switches: d1->d2 (switch), d2->d3 (same), d3->d4 (switch) => 2 switches over 1.5s = 1.33 switches/sec
        switch_rate = decision_switch_rate(decisions, duration=1.5)
        self.assertAlmostEqual(switch_rate, 2.0 / 1.5)

        mean_dev = mean_heading_deviation(decisions)
        self.assertAlmostEqual(mean_dev, 0.4)


if __name__ == "__main__":
    unittest.main()

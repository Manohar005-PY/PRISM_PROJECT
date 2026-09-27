"""Unit tests for scientific evaluation metrics."""

import unittest

from app.core.enums import RiskLevel
from app.core.types import Vector2
from app.decision.model import NavigationAction
from evaluation.models import DecisionRecord, EvaluationSample, GroundTruthState, RiskRecord
from evaluation.metrics import (
    compute_decision_switch_rate,
    compute_false_alarm_rate,
    compute_hazard_detection_rate,
    compute_reaction_time,
    compute_unnecessary_stop_rate,
)


class TestEvaluationMetrics(unittest.TestCase):
    """Test suite for evaluation metric calculations."""

    def test_hazard_detection_and_false_alarm_rates(self) -> None:
        gt_haz = GroundTruthState(1.0, 1, "person", Vector2(0.0, 2.0), Vector2(0.0, -1.0), True)
        gt_safe = GroundTruthState(2.0, 1, "person", Vector2(0.0, 5.0), Vector2(0.0, 0.0), False)

        risk_haz = RiskRecord(1.0, 1, 90.0, RiskLevel.CRITICAL, 1.0, 0.2)
        dec_haz = DecisionRecord(1.0, NavigationAction.STOP, 1, 0.0, 0.95)

        risk_safe = RiskRecord(2.0, 1, 10.0, RiskLevel.SAFE, None, 5.0)
        dec_safe = DecisionRecord(2.0, NavigationAction.CONTINUE, None, 0.0, 1.0)

        s_haz = EvaluationSample(1.0, 1, {1: gt_haz}, True, {}, risk_haz, dec_haz, 1.0)
        s_safe = EvaluationSample(2.0, 2, {1: gt_safe}, False, {}, risk_safe, dec_safe, 1.0)

        samples = [s_haz, s_safe]

        det_rate = compute_hazard_detection_rate(samples)
        fa_rate = compute_false_alarm_rate(samples)
        unnecessary_stops = compute_unnecessary_stop_rate(samples)

        self.assertEqual(det_rate, 100.0)
        self.assertEqual(fa_rate, 0.0)
        self.assertEqual(unnecessary_stops, 0.0)

    def test_decision_switch_rate(self) -> None:
        gt = GroundTruthState(1.0, 1, "person", Vector2(0.0, 3.0), Vector2(0.0, 0.0), False)
        risk = RiskRecord(1.0, 1, 10.0, RiskLevel.SAFE, None, None)

        # 3 frames across 2 seconds with 2 action switches
        s1 = EvaluationSample(0.0, 1, {1: gt}, False, {}, risk, DecisionRecord(0.0, NavigationAction.CONTINUE, None, 0.0, 1.0), 1.0)
        s2 = EvaluationSample(1.0, 2, {1: gt}, False, {}, risk, DecisionRecord(1.0, NavigationAction.CAUTION, None, 0.0, 1.0), 1.0)
        s3 = EvaluationSample(2.0, 3, {1: gt}, False, {}, risk, DecisionRecord(2.0, NavigationAction.CONTINUE, None, 0.0, 1.0), 1.0)

        rate = compute_decision_switch_rate([s1, s2, s3])
        self.assertAlmostEqual(rate, 1.0)  # 2 switches / 2 seconds = 1.0 switch/sec


if __name__ == "__main__":
    unittest.main()

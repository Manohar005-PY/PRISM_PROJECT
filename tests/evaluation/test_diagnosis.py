"""Unit tests for CausalFailureDiagnoser."""

import unittest

from app.core.enums import RiskLevel
from app.core.types import Vector2
from app.decision.model import NavigationAction
from evaluation.diagnosis import CausalFailureDiagnoser
from evaluation.models import DecisionRecord, EvaluationSample, GroundTruthState, PredictionRecord, RiskRecord
from evaluation.validation import FailureCategory


class TestDiagnosis(unittest.TestCase):
    """Test suite for causal failure diagnosis."""

    def test_safe_scene_no_failure(self) -> None:
        sample = EvaluationSample(
            timestamp=1.0,
            frame_id=1,
            ground_truth={},
            is_scene_hazard=False,
            predictions={},
            risk=RiskRecord(1.0, None, 0.0, RiskLevel.SAFE, None, None),
            decision=DecisionRecord(1.0, NavigationAction.CONTINUE, None, 0.0, 1.0),
            processing_latency_ms=0.5,
        )
        diag = CausalFailureDiagnoser.diagnose_frame(sample)
        self.assertIsNone(diag.failure_category)
        self.assertFalse(diag.is_scene_hazard)

    def test_diagnose_risk_score_failure(self) -> None:
        gt = GroundTruthState(1.0, 1, "person", Vector2(0.0, 3.0), Vector2(0.0, -1.0), True)
        pred = PredictionRecord(1.0, 1, Vector2(0.0, 3.0), Vector2(0.0, -1.0), 1.0)
        sample = EvaluationSample(
            timestamp=1.0,
            frame_id=1,
            ground_truth={1: gt},
            is_scene_hazard=True,
            predictions={1: pred},
            risk=RiskRecord(1.0, 1, 30.0, RiskLevel.LOW, 2.0, 0.5),  # Risk score 30 < 45 => LOW
            decision=DecisionRecord(1.0, NavigationAction.CONTINUE, None, 0.0, 1.0),
            processing_latency_ms=0.5,
        )
        diag = CausalFailureDiagnoser.diagnose_frame(sample)
        self.assertEqual(diag.failure_category, FailureCategory.RISK_CLASSIFICATION_FAILURE)


if __name__ == "__main__":
    unittest.main()

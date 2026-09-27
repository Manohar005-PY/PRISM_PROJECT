"""Unit tests for FailureClassifier and failure category attribution."""

import unittest

from app.core.enums import RiskLevel
from app.core.types import Vector2
from app.decision.model import NavigationAction
from evaluation.models import DecisionRecord, EvaluationSample, GroundTruthState, RiskRecord
from evaluation.validation import FailureCategory, FailureClassifier


class TestValidation(unittest.TestCase):
    """Test suite for failure taxonomy and classification."""

    def test_classify_perception_failure(self) -> None:
        gt = GroundTruthState(1.0, 1, "person", Vector2(0.0, 2.0), Vector2(0.0, -1.0), True)
        sample = EvaluationSample(
            timestamp=1.0,
            frame_id=1,
            ground_truth={1: gt},
            is_scene_hazard=True,
            predictions={},  # Empty predictions => perception failure
            risk=RiskRecord(1.0, None, 0.0, RiskLevel.SAFE, None, None),
            decision=DecisionRecord(1.0, NavigationAction.CONTINUE, None, 0.0, 1.0),
            processing_latency_ms=1.0,
        )

        failure = FailureClassifier.classify_missed_hazard(sample)
        self.assertEqual(failure.category, FailureCategory.PERCEPTION_FAILURE)


if __name__ == "__main__":
    unittest.main()

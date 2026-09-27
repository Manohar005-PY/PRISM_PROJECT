"""Unit tests for evaluation domain models and immutability."""

import unittest

from app.core.enums import RiskLevel
from app.core.types import Vector2
from app.decision.model import NavigationAction
from evaluation.models import (
    DecisionRecord,
    EvaluationSample,
    GroundTruthState,
    PredictionRecord,
    RiskRecord,
)


class TestEvaluationModels(unittest.TestCase):
    """Test suite for evaluation data models."""

    def test_ground_truth_state_invariants(self) -> None:
        gt = GroundTruthState(
            timestamp=1.0,
            object_id=1,
            class_name="person",
            position=Vector2(0.0, 3.0),
            velocity=Vector2(0.0, -1.0),
            is_hazard=True,
        )
        self.assertEqual(gt.object_id, 1)
        self.assertTrue(gt.is_hazard)

    def test_evaluation_sample_structure(self) -> None:
        gt = GroundTruthState(1.0, 1, "person", Vector2(0.0, 3.0), Vector2(0.0, -1.0), True)
        pred = PredictionRecord(1.0, 1, Vector2(0.0, 2.0), Vector2(0.0, -1.0), 1.0)
        risk = RiskRecord(1.0, 1, 80.0, RiskLevel.HIGH, 2.0, 0.5)
        dec = DecisionRecord(1.0, NavigationAction.STOP, 1, 0.0, 0.95)

        sample = EvaluationSample(
            timestamp=1.0,
            frame_id=1,
            ground_truth={1: gt},
            is_scene_hazard=True,
            predictions={1: pred},
            risk=risk,
            decision=dec,
            processing_latency_ms=1.2,
        )
        self.assertEqual(sample.frame_id, 1)
        self.assertTrue(sample.is_scene_hazard)


if __name__ == "__main__":
    unittest.main()

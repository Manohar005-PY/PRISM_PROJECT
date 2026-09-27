"""Unit tests for multi-factor risk assessment and RiskEvaluator protocol conformance."""

import unittest

from app.core.enums import RiskLevel
from app.core.interfaces import RiskEvaluator
from app.core.types import ObjectState, Trajectory, TrajectoryPoint, Vector2, WorldState
from app.prediction.relative_motion import EgoMotionState
from app.risk.evaluator import PredictiveRiskEvaluator
from app.risk.model import RiskConfig


class TestPredictiveRiskEvaluator(unittest.TestCase):
    """Test suite for PredictiveRiskEvaluator."""

    def setUp(self) -> None:
        self.evaluator = PredictiveRiskEvaluator(RiskConfig())

    def test_protocol_conformance(self) -> None:
        self.assertIsInstance(self.evaluator, RiskEvaluator)

    def test_empty_world_evaluation(self) -> None:
        world = WorldState(timestamp=0.0, objects={})
        assessments = self.evaluator.evaluate_world(world)
        self.assertEqual(len(assessments), 0)
        primary = self.evaluator.get_primary_threat(world)
        self.assertIsNone(primary)

    def test_imminent_head_on_threat(self) -> None:
        """Obstacle directly ahead approaching at high speed produces HIGH or CRITICAL risk."""
        obj = ObjectState(
            object_id=1,
            class_id="person",
            position=Vector2(0.0, 2.0),
            velocity=Vector2(0.0, -1.5),
            timestamp=0.0,
        )
        ego = EgoMotionState(position=Vector2(0.0, 0.0), velocity=Vector2(0.0, 1.0))
        world = WorldState(timestamp=0.0, objects={1: obj})

        primary = self.evaluator.get_primary_threat(world, ego=ego)
        self.assertIsNotNone(primary)
        self.assertIn(primary.risk_level, (RiskLevel.HIGH, RiskLevel.CRITICAL))  # type: ignore[union-attr]
        self.assertGreater(primary.risk_score, 70.0)  # type: ignore[union-attr]
        self.assertEqual(primary.object_id, 1)  # type: ignore[union-attr]

    def test_diverging_safe_object(self) -> None:
        """Obstacle moving away produces SAFE or LOW risk."""
        obj = ObjectState(
            object_id=2,
            class_id="person",
            position=Vector2(0.0, 4.0),
            velocity=Vector2(0.0, 1.5),
            timestamp=0.0,
        )
        ego = EgoMotionState(position=Vector2(0.0, 0.0), velocity=Vector2(0.0, 0.0))
        world = WorldState(timestamp=0.0, objects={2: obj})

        primary = self.evaluator.get_primary_threat(world, ego=ego)
        self.assertIsNotNone(primary)
        self.assertIn(primary.risk_level, (RiskLevel.SAFE, RiskLevel.LOW))  # type: ignore[union-attr]
        self.assertLess(primary.risk_score, 30.0)  # type: ignore[union-attr]

    def test_multi_object_primary_threat_selection(self) -> None:
        """Evaluator correctly identifies the highest risk entity among multiple obstacles."""
        safe_obj = ObjectState(1, "person", Vector2(3.0, 5.0), Vector2(1.0, 0.0), 0.0)
        dangerous_obj = ObjectState(2, "person", Vector2(0.0, 2.5), Vector2(0.0, -1.2), 0.0)

        world = WorldState(timestamp=0.0, objects={1: safe_obj, 2: dangerous_obj})
        primary = self.evaluator.get_primary_threat(world)

        self.assertIsNotNone(primary)
        self.assertEqual(primary.object_id, 2)  # type: ignore[union-attr]
        self.assertGreater(primary.risk_score, 50.0)  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()

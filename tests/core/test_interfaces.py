"""Tests for core protocol contracts and interface structural conformance."""

import unittest

from app.core.enums import DecisionType, RiskLevel
from app.core.interfaces import (
    DecisionEngine,
    Detector,
    EgoMotionProvider,
    GroundProjector,
    Predictor,
    RiskEvaluator,
    StateEstimator,
    Tracker,
)
from app.core.types import (
    BoundingBox,
    Decision,
    ObjectObservation,
    ObjectState,
    RiskAssessment,
    Trajectory,
    TrajectoryPoint,
    Vector2,
    WorldState,
)


# Dummy conformant implementations for structural protocol verification
class MockDetector:
    def detect(self, frame: object) -> list[ObjectObservation]:
        return [
            ObjectObservation(
                class_id="person",
                confidence=0.9,
                bbox=BoundingBox(0, 0, 10, 10),
                timestamp=0.0,
            )
        ]


class MockTracker:
    def update(
        self,
        observations: list[ObjectObservation],
        timestamp: float,
    ) -> list[ObjectObservation]:
        return observations


class MockGroundProjector:
    def project(self, point: tuple[float, float]) -> Vector2 | None:
        return Vector2(x=point[0] * 0.01, z=point[1] * 0.05)


class MockEgoMotionProvider:
    def estimate_motion(self, timestamp: float) -> Vector2 | None:
        return Vector2(x=0.0, z=1.0)


class MockStateEstimator:
    def update(
        self,
        observations: list[ObjectObservation],
        timestamp: float,
    ) -> WorldState:
        return WorldState(timestamp=timestamp)


class MockPredictor:
    def predict(self, state: ObjectState, horizon: float) -> Trajectory:
        point = TrajectoryPoint(
            timestamp=state.timestamp + horizon,
            position=state.position + state.velocity * horizon,
            velocity=state.velocity,
        )
        return Trajectory(
            object_id=state.object_id,
            points=[point],
            prediction_horizon=horizon,
        )


class MockRiskEvaluator:
    def evaluate(
        self,
        state: ObjectState,
        trajectory: Trajectory,
        timestamp: float,
    ) -> RiskAssessment:
        return RiskAssessment(
            object_id=state.object_id,
            risk_level=RiskLevel.SAFE,
            risk_score=0.0,
            timestamp=timestamp,
        )


class MockDecisionEngine:
    def decide(self, risk_assessment: RiskAssessment) -> Decision:
        return Decision(
            decision_type=DecisionType.NO_ACTION,
            object_id=risk_assessment.object_id,
            timestamp=risk_assessment.timestamp,
        )


class NonConformingClass:
    """Class lacking required methods for protocol conformance."""
    pass


class TestInterfaces(unittest.TestCase):
    """Test suite verifying protocol contracts."""

    def test_detector_protocol_conformance(self) -> None:
        detector = MockDetector()
        self.assertIsInstance(detector, Detector)
        self.assertNotIsInstance(NonConformingClass(), Detector)

    def test_tracker_protocol_conformance(self) -> None:
        tracker = MockTracker()
        self.assertIsInstance(tracker, Tracker)
        self.assertNotIsInstance(NonConformingClass(), Tracker)

    def test_ground_projector_protocol_conformance(self) -> None:
        projector = MockGroundProjector()
        self.assertIsInstance(projector, GroundProjector)
        self.assertNotIsInstance(NonConformingClass(), GroundProjector)

    def test_ego_motion_provider_protocol_conformance(self) -> None:
        provider = MockEgoMotionProvider()
        self.assertIsInstance(provider, EgoMotionProvider)
        self.assertNotIsInstance(NonConformingClass(), EgoMotionProvider)

    def test_state_estimator_protocol_conformance(self) -> None:
        estimator = MockStateEstimator()
        self.assertIsInstance(estimator, StateEstimator)
        self.assertNotIsInstance(NonConformingClass(), StateEstimator)

    def test_predictor_protocol_conformance(self) -> None:
        predictor = MockPredictor()
        self.assertIsInstance(predictor, Predictor)
        self.assertNotIsInstance(NonConformingClass(), Predictor)

    def test_risk_evaluator_protocol_conformance(self) -> None:
        evaluator = MockRiskEvaluator()
        self.assertIsInstance(evaluator, RiskEvaluator)
        self.assertNotIsInstance(NonConformingClass(), RiskEvaluator)

    def test_decision_engine_protocol_conformance(self) -> None:
        engine = MockDecisionEngine()
        self.assertIsInstance(engine, DecisionEngine)
        self.assertNotIsInstance(NonConformingClass(), DecisionEngine)


if __name__ == "__main__":
    unittest.main()

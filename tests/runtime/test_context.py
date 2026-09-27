"""Unit tests for ContextBuilder data accumulation."""

import unittest

from app.core.enums import RiskLevel
from app.core.types import RiskAssessment, Vector2, WorldState
from app.decision.model import DecisionReason, NavigationAction, NavigationDecision
from app.perception.model import PhysicalObservation, TrackedObservation
from app.runtime.context import ContextBuilder
from app.runtime.timing import StageTiming


class TestRuntimeContextBuilder(unittest.TestCase):
    """Test suite for ContextBuilder."""

    def test_context_builder_accumulation(self) -> None:
        builder = ContextBuilder(frame_id=1, timestamp=0.5)

        obs = PhysicalObservation(
            position=Vector2(0.0, 3.0),
            measurement_covariance=((0.04, 0.0), (0.0, 0.04)),
            confidence=0.9,
            timestamp=0.5,
            detection_id="d1",
            class_name="person",
        )
        tracked_obs = TrackedObservation(track_id=1, observation=obs)
        world_st = WorldState(timestamp=0.5, objects={})
        risk = RiskAssessment(
            object_id=0,
            risk_level=RiskLevel.SAFE,
            risk_score=0.0,
            confidence=1.0,
            timestamp=0.5,
            ttc=None,
            cpa_distance=None,
        )
        dec = NavigationDecision(
            action=NavigationAction.CONTINUE,
            reason=DecisionReason.NO_HAZARD,
            confidence=1.0,
            primary_threat_id=None,
            recommended_heading=0.0,
            risk_level=RiskLevel.SAFE,
            timestamp=0.5,
        )
        timing = StageTiming(perception_ms=1.0, total_ms=5.0, effective_fps=200.0)

        ctx = (
            builder.set_perception_output([], [obs], [])
            .set_tracking_output([tracked_obs])
            .set_world_state(world_st)
            .set_risk(risk)
            .set_decision(dec)
            .set_timing(timing)
            .build()
        )

        self.assertEqual(ctx.frame_id, 1)
        self.assertEqual(len(ctx.physical_observations), 1)
        self.assertEqual(len(ctx.tracked_observations), 1)
        self.assertEqual(ctx.decision.action, NavigationAction.CONTINUE)
        self.assertEqual(ctx.timing.perception_ms, 1.0)


if __name__ == "__main__":
    unittest.main()

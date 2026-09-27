"""Unit tests for runtime domain models (FrameContext, RuntimeResult)."""

import unittest

from app.core.enums import RiskLevel
from app.core.types import RiskAssessment
from app.decision.model import DecisionReason, NavigationAction, NavigationDecision
from app.runtime.health import RuntimeHealth, RuntimeStatus
from app.runtime.model import FrameContext, RuntimeResult
from app.runtime.timing import StageTiming


class TestRuntimeModel(unittest.TestCase):
    """Test suite for FrameContext and RuntimeResult."""

    def test_frame_context_immutability(self) -> None:
        ctx = FrameContext(
            frame_id=1,
            timestamp=0.1,
        )
        self.assertEqual(ctx.frame_id, 1)
        self.assertEqual(ctx.timestamp, 0.1)
        self.assertEqual(len(ctx.physical_observations), 0)

    def test_runtime_result_fields(self) -> None:
        ctx = FrameContext(frame_id=1, timestamp=0.1)
        dec = NavigationDecision(
            action=NavigationAction.CONTINUE,
            reason=DecisionReason.NO_HAZARD,
            confidence=1.0,
            primary_threat_id=None,
            recommended_heading=0.0,
            risk_level=RiskLevel.SAFE,
            timestamp=0.1,
        )
        risk = RiskAssessment(
            object_id=0,
            risk_level=RiskLevel.SAFE,
            risk_score=0.0,
            confidence=1.0,
            timestamp=0.1,
            ttc=None,
            cpa_distance=None,
        )
        health = RuntimeHealth(status=RuntimeStatus.HEALTHY)
        timing = StageTiming(total_ms=1.5, effective_fps=666.0)

        res = RuntimeResult(
            frame_id=1,
            timestamp=0.1,
            processing_duration_ms=1.5,
            decision=dec,
            risk_assessment=risk,
            active_track_count=0,
            detection_count=0,
            observation_count=0,
            runtime_status=RuntimeStatus.HEALTHY,
            timing=timing,
            health=health,
            context=ctx,
        )
        self.assertEqual(res.frame_id, 1)
        self.assertEqual(res.decision.action, NavigationAction.CONTINUE)


if __name__ == "__main__":
    unittest.main()

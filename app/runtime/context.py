"""Frame context builder facilitating structured data accumulation across pipeline stages."""

from __future__ import annotations

from typing import Any, Mapping

from app.core.types import RiskAssessment, Trajectory, WorldState
from app.decision.model import NavigationDecision
from app.perception.model import Detection, DetectionRejection, PhysicalObservation, TrackedObservation
from app.runtime.model import FrameContext
from app.runtime.timing import StageTiming


class ContextBuilder:
    """Builder for constructing immutable FrameContext objects."""

    def __init__(self, frame_id: int, timestamp: float, raw_frame: Any | None = None) -> None:
        self.frame_id = frame_id
        self.timestamp = timestamp
        self.raw_frame = raw_frame
        self.detections: list[Detection] = []
        self.physical_observations: list[PhysicalObservation] = []
        self.tracked_observations: list[TrackedObservation] = []
        self.world_state: WorldState | None = None
        self.trajectories: dict[int, Trajectory] = {}
        self.risk_assessment: RiskAssessment | None = None
        self.decision: NavigationDecision | None = None
        self.timing: StageTiming = StageTiming()
        self.rejection_diagnostics: list[DetectionRejection] = []

    def set_perception_output(
        self,
        detections: list[Detection],
        observations: list[PhysicalObservation],
        rejections: list[DetectionRejection],
    ) -> ContextBuilder:
        self.detections = list(detections)
        self.physical_observations = list(observations)
        self.rejection_diagnostics = list(rejections)
        return self

    def set_tracking_output(self, tracked_observations: list[TrackedObservation]) -> ContextBuilder:
        self.tracked_observations = list(tracked_observations)
        return self

    def set_world_state(self, world_state: WorldState) -> ContextBuilder:
        self.world_state = world_state
        return self

    def set_trajectories(self, trajectories: Mapping[int, Trajectory]) -> ContextBuilder:
        self.trajectories = dict(trajectories)
        return self

    def set_risk(self, risk_assessment: RiskAssessment) -> ContextBuilder:
        self.risk_assessment = risk_assessment
        return self

    def set_decision(self, decision: NavigationDecision) -> ContextBuilder:
        self.decision = decision
        return self

    def set_timing(self, timing: StageTiming) -> ContextBuilder:
        self.timing = timing
        return self

    def build(self) -> FrameContext:
        return FrameContext(
            frame_id=self.frame_id,
            timestamp=self.timestamp,
            raw_frame=self.raw_frame,
            detections=tuple(self.detections),
            physical_observations=tuple(self.physical_observations),
            tracked_observations=tuple(self.tracked_observations),
            world_state=self.world_state,
            trajectories=dict(self.trajectories) if self.trajectories else None,
            risk_assessment=self.risk_assessment,
            decision=self.decision,
            timing=self.timing,
            rejection_diagnostics=tuple(self.rejection_diagnostics),
        )

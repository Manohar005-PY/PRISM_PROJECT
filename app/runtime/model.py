"""Runtime domain models: FrameContext, RuntimeResult, and per-frame tracking diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from app.core.types import RiskAssessment, Trajectory, WorldState
from app.decision.model import NavigationDecision
from app.perception.model import Detection, DetectionRejection, PhysicalObservation, TrackedObservation
from app.runtime.health import RuntimeHealth, RuntimeStatus
from app.runtime.timing import StageTiming


@dataclass(frozen=True)
class FrameContext:
    """Immutable data context representing the full lifecycle of data flowing through one frame."""

    frame_id: int
    timestamp: float
    raw_frame: Any | None = None
    detections: tuple[Detection, ...] = ()
    physical_observations: tuple[PhysicalObservation, ...] = ()
    tracked_observations: tuple[TrackedObservation, ...] = ()
    world_state: WorldState | None = None
    trajectories: Mapping[int, Trajectory] | None = None
    risk_assessment: RiskAssessment | None = None
    decision: NavigationDecision | None = None
    timing: StageTiming = StageTiming()
    rejection_diagnostics: tuple[DetectionRejection, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.frame_id, int) or self.frame_id < 0:
            raise ValueError(f"frame_id must be non-negative integer, got {self.frame_id!r}")
        if not isinstance(self.timestamp, (int, float)) or not math.isfinite(self.timestamp):
            raise ValueError(f"timestamp must be finite float, got {self.timestamp!r}")

        object.__setattr__(self, "timestamp", float(self.timestamp))
        object.__setattr__(self, "detections", tuple(self.detections))
        object.__setattr__(self, "physical_observations", tuple(self.physical_observations))
        object.__setattr__(self, "tracked_observations", tuple(self.tracked_observations))
        object.__setattr__(self, "rejection_diagnostics", tuple(self.rejection_diagnostics))


@dataclass(frozen=True)
class RuntimeResult:
    """Structured result returned at the completion of processing a single frame."""

    frame_id: int
    timestamp: float
    processing_duration_ms: float
    decision: NavigationDecision
    risk_assessment: RiskAssessment
    active_track_count: int
    detection_count: int
    observation_count: int
    runtime_status: RuntimeStatus
    timing: StageTiming
    health: RuntimeHealth
    context: FrameContext

    def __post_init__(self) -> None:
        if not isinstance(self.frame_id, int):
            raise TypeError(f"frame_id must be int, got {type(self.frame_id)}")
        if not isinstance(self.timestamp, (int, float)) or not math.isfinite(self.timestamp):
            raise ValueError(f"timestamp must be a finite float, got {self.timestamp!r}")
        if not isinstance(self.processing_duration_ms, (int, float)) or not math.isfinite(self.processing_duration_ms):
            raise ValueError(f"processing_duration_ms must be a finite float, got {self.processing_duration_ms!r}")

        object.__setattr__(self, "timestamp", float(self.timestamp))
        object.__setattr__(self, "processing_duration_ms", float(self.processing_duration_ms))

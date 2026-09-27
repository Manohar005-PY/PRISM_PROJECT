"""Data models, configuration contracts, and result structures for PRISM Part 20 Real-World Benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class RealWorldCameraConfig:
    """Camera intrinsic and extrinsic parameters for monocular ground-plane projection."""

    mounting_height_m: float = 1.25
    pitch_rad: float = -0.05
    resolution: tuple[int, int] = (720, 1280)  # (width, height)
    nominal_fps: float = 30.0
    hfov_deg: float = 65.0
    vfov_deg: float = 80.0

    def __post_init__(self) -> None:
        if self.mounting_height_m <= 0.0:
            raise ValueError(f"mounting_height_m must be positive, got {self.mounting_height_m}")


@dataclass(frozen=True)
class RealWorldConfig:
    """Configuration container for Part 20 Real-World Benchmark."""

    dataset_dir: str = "experiments/datasets/real_world"
    output_dir: str = "experiments/real_world/part20"
    camera_config: RealWorldCameraConfig = field(default_factory=RealWorldCameraConfig)
    detector_confidence_thresh: float = 0.35
    prediction_horizon: float = 3.0
    min_safe_clearance_m: float = 0.50
    lead_time_late_threshold_s: float = 0.50
    detector_type: str = "auto"  # 'yolo', 'hog', or 'auto'


@dataclass
class RealWorldDetectionBox:
    """2D bounding box detection on an image frame."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float
    class_id: int | str = 0
    class_name: str = "person"

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Returns the (u, v) pixel coordinate of the ground-contact point."""
        return ((self.x_min + self.x_max) / 2.0, self.y_max)


@dataclass
class RealWorldRunMetadata:
    """Metadata specification for a single real-world video run."""

    run_id: str
    scenario_family: str
    video_path: str
    has_collision: bool
    collision_timestamp_s: float | None
    num_frames: int
    duration_s: float
    fps: float
    resolution: tuple[int, int]
    description: str
    environment: str


@dataclass
class RealWorldFrameResult:
    """Frame-level record of detection, tracking, prediction, and decision outputs."""

    timestamp: float
    frame_index: int
    raw_detections_count: int
    projected_observations_count: int
    active_tracks_count: int
    decision_action: str
    decision_reason: str
    risk_level: str
    risk_score: float
    predicted_ttc: float | None
    predicted_cpa: float | None
    primary_threat_id: int | None
    recommended_heading_deg: float | None
    latency_ms: float
    track_positions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RealWorldRunResult:
    """Aggregated evaluation metrics for a single real-world video run."""

    run_id: str
    scenario_family: str
    video_path: str
    has_collision: bool
    collision_timestamp_s: float | None = None
    first_warning_timestamp_s: float | None = None
    first_caution_timestamp_s: float | None = None
    first_emergency_timestamp_s: float | None = None
    warning_lead_time_s: float | None = None
    warned: bool = False
    collision_recalled: bool = False
    collision_missed: bool = False
    caution_alert_issued: bool = False
    emergency_warning_issued: bool = False
    proximity_caution_alert: bool = False
    false_collision_warning: bool = False
    false_warning: bool = False
    total_frames: int = 0
    duration_s: float = 0.0
    fps: float = 30.0
    mean_detection_count: float = 0.0
    tracking_continuity_ratio: float | None = None
    track_coverage_ratio: float | None = None
    detection_association_coverage_ratio: float | None = None
    zero_detection_frames: int = 0
    coasting_frames: int = 0
    gap_bridging_rate: float | None = None
    mean_latency_ms: float = 0.0
    decision_changes: int = 0
    steering_reversals: int = 0
    frame_results: list[RealWorldFrameResult] = field(default_factory=list)

    @property
    def detector_dropout_frames(self) -> int:
        """Alias for zero_detection_frames."""
        return self.zero_detection_frames

    @property
    def dropout_bridging_rate(self) -> float | None:
        """Alias for gap_bridging_rate."""
        return self.gap_bridging_rate

    def to_dict(self, include_frames: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "run_id": self.run_id,
            "scenario_family": self.scenario_family,
            "video_path": self.video_path,
            "has_collision": self.has_collision,
            "collision_timestamp_s": self.collision_timestamp_s,
            "first_warning_timestamp_s": self.first_warning_timestamp_s,
            "first_caution_timestamp_s": self.first_caution_timestamp_s,
            "first_emergency_timestamp_s": self.first_emergency_timestamp_s,
            "warning_lead_time_s": self.warning_lead_time_s,
            "warned": self.warned,
            "collision_recalled": self.collision_recalled,
            "collision_missed": self.collision_missed,
            "caution_alert_issued": self.caution_alert_issued,
            "emergency_warning_issued": self.emergency_warning_issued,
            "proximity_caution_alert": self.proximity_caution_alert,
            "false_collision_warning": self.false_collision_warning,
            "false_warning": self.false_warning,
            "total_frames": self.total_frames,
            "duration_s": round(self.duration_s, 2),
            "fps": round(self.fps, 1),
            "mean_detection_count": round(self.mean_detection_count, 2),
            "tracking_continuity_ratio": round(self.tracking_continuity_ratio, 4) if self.tracking_continuity_ratio is not None else None,
            "track_coverage_ratio": round(self.track_coverage_ratio, 4) if self.track_coverage_ratio is not None else None,
            "detection_association_coverage_ratio": round(self.detection_association_coverage_ratio, 4) if self.detection_association_coverage_ratio is not None else None,
            "zero_detection_frames": self.zero_detection_frames,
            "detector_dropout_frames": self.zero_detection_frames,
            "coasting_frames": self.coasting_frames,
            "gap_bridging_rate": round(self.gap_bridging_rate, 4) if self.gap_bridging_rate is not None else None,
            "dropout_bridging_rate": round(self.gap_bridging_rate, 4) if self.gap_bridging_rate is not None else None,
            "mean_latency_ms": round(self.mean_latency_ms, 3),
            "decision_changes": self.decision_changes,
            "steering_reversals": self.steering_reversals,
        }
        if include_frames:
            d["frame_results"] = [
                {
                    "timestamp": f.timestamp,
                    "frame_index": f.frame_index,
                    "raw_detections": f.raw_detections_count,
                    "active_tracks": f.active_tracks_count,
                    "action": f.decision_action,
                    "reason": f.decision_reason,
                    "risk_level": f.risk_level,
                    "risk_score": f.risk_score,
                    "predicted_ttc": f.predicted_ttc,
                    "predicted_cpa": f.predicted_cpa,
                    "recommended_heading_deg": f.recommended_heading_deg,
                    "latency_ms": f.latency_ms,
                    "tracks": f.track_positions,
                }
                for f in self.frame_results
            ]
        return d


@dataclass
class RealWorldFailureCase:
    """Isolated diagnostic discrepancy or failure instance in real-world testing."""

    run_id: str
    scenario_family: str
    failure_type: str  # MISSED_WARNING, LATE_WARNING, FALSE_WARNING, TRACK_FRAGMENTATION, STEERING_FLUTTER
    description: str
    timestamp_s: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "scenario_family": self.scenario_family,
            "failure_type": self.failure_type,
            "description": self.description,
            "timestamp_s": self.timestamp_s,
            "details": self.details,
        }

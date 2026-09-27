"""Structured runtime telemetry recording per-frame metrics and aggregate performance statistics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from typing import Sequence

from app.runtime.model import RuntimeResult


@dataclass(frozen=True)
class FrameTelemetryRecord:
    """Telemetry captured for a single pipeline execution frame."""

    timestamp: float
    frame_id: int
    frame_interval_ms: float
    processing_latency_ms: float
    detection_count: int
    valid_observation_count: int
    active_track_count: int
    primary_threat_id: int | None
    risk_level: str
    risk_score: float
    ttc: float | None
    cpa_distance: float | None
    decision: str
    decision_heading: float | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AggregateTelemetry:
    """Summary statistics across a completed runtime session."""

    total_frames: int
    mean_fps: float
    min_fps: float
    max_fps: float
    mean_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    dropped_frames: int
    detection_failures: int
    projection_rejections: int
    decision_switch_count: int

    def to_dict(self) -> dict:
        return asdict(self)


class TelemetryLogger:
    """Collects per-frame telemetry and exports structured JSONL and CSV logs."""

    def __init__(self, log_filepath: str | None = None) -> None:
        self.log_filepath = log_filepath
        self._records: list[FrameTelemetryRecord] = []
        self._last_timestamp: float | None = None
        self._last_decision: str | None = None
        self._decision_switch_count: int = 0
        self._dropped_frames_count: int = 0
        self._detection_failures_count: int = 0
        self._projection_rejections_count: int = 0

        if self.log_filepath:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_filepath)), exist_ok=True)

    def record_frame(
        self,
        result: RuntimeResult,
        dropped: bool = False,
        detection_failure: bool = False,
        rejection_count: int = 0,
    ) -> FrameTelemetryRecord:
        """Capture and record telemetry for one processed runtime frame."""
        now = result.timestamp
        interval_ms = (now - self._last_timestamp) * 1000.0 if self._last_timestamp is not None else 33.3
        self._last_timestamp = now

        if dropped:
            self._dropped_frames_count += 1
        if detection_failure:
            self._detection_failures_count += 1
        self._projection_rejections_count += rejection_count

        dec_str = result.decision.action.value
        if self._last_decision is not None and dec_str != self._last_decision:
            self._decision_switch_count += 1
        self._last_decision = dec_str

        risk_obj = result.risk_assessment
        threat_id = risk_obj.object_id if risk_obj else None
        risk_lvl = risk_obj.risk_level.value if risk_obj else "SAFE"
        risk_sc = risk_obj.risk_score if risk_obj else 0.0
        ttc_val = risk_obj.ttc if risk_obj else None
        cpa_val = risk_obj.cpa_distance if risk_obj else None

        record = FrameTelemetryRecord(
            timestamp=now,
            frame_id=result.frame_id,
            frame_interval_ms=interval_ms,
            processing_latency_ms=result.processing_duration_ms,
            detection_count=result.detection_count,
            valid_observation_count=result.observation_count,
            active_track_count=result.active_track_count,
            primary_threat_id=threat_id,
            risk_level=risk_lvl,
            risk_score=risk_sc,
            ttc=ttc_val,
            cpa_distance=cpa_val,
            decision=dec_str,
            decision_heading=result.decision.recommended_heading,
        )

        self._records.append(record)

        if self.log_filepath:
            with open(self.log_filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict()) + "\n")

        return record

    def compute_aggregates(self) -> AggregateTelemetry:
        """Compute summary telemetry statistics across all recorded frames."""
        if not self._records:
            return AggregateTelemetry(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0)

        latencies = sorted(r.processing_latency_ms for r in self._records)
        n = len(latencies)
        mean_lat = sum(latencies) / n
        p95_lat = latencies[int(0.95 * n)] if n > 0 else 0.0
        p99_lat = latencies[int(0.99 * n)] if n > 0 else 0.0

        fps_values = [1000.0 / max(0.001, r.processing_latency_ms) for r in self._records]
        mean_fps = sum(fps_values) / len(fps_values)
        min_fps = min(fps_values)
        max_fps = max(fps_values)

        return AggregateTelemetry(
            total_frames=n,
            mean_fps=mean_fps,
            min_fps=min_fps,
            max_fps=max_fps,
            mean_latency_ms=mean_lat,
            p95_latency_ms=p95_lat,
            p99_latency_ms=p99_lat,
            dropped_frames=self._dropped_frames_count,
            detection_failures=self._detection_failures_count,
            projection_rejections=self._projection_rejections_count,
            decision_switch_count=self._decision_switch_count,
        )

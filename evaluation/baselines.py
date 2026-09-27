"""Reactive baselines (Distance & TTC) and PRISM pipeline wrapper for comparative evaluation."""

from __future__ import annotations

import math
from typing import Protocol, Sequence, runtime_checkable

from app.core.enums import RiskLevel
from app.core.types import Vector2
from app.decision.model import NavigationAction
from app.perception.model import PhysicalObservation
from app.prediction.relative_motion import EgoMotionState
from app.runtime.model import RuntimeResult
from app.runtime.pipeline import PRISMPipeline
from evaluation.models import DecisionRecord, RiskRecord


@runtime_checkable
class SystemEvaluator(Protocol):
    """Protocol for systems evaluated within the benchmarking framework."""

    def process(
        self,
        observations: list[PhysicalObservation],
        timestamp: float,
        ego_state: EgoMotionState | None = None,
    ) -> tuple[RiskRecord, DecisionRecord]: ...

    def reset(self) -> None: ...


class DistanceBaseline:
    """Reactive distance thresholding baseline.

    Rule: If Euclidean distance to nearest obstacle < distance_threshold, emit ALERT / STOP.
    Lacks relative velocity estimation, trajectory projection, and heading evasion.
    """

    def __init__(
        self,
        stop_threshold: float = 1.8,
        caution_threshold: float = 3.0,
    ) -> None:
        self.stop_threshold = stop_threshold
        self.caution_threshold = caution_threshold

    def reset(self) -> None:
        pass

    def process(
        self,
        observations: list[PhysicalObservation],
        timestamp: float,
        ego_state: EgoMotionState | None = None,
    ) -> tuple[RiskRecord, DecisionRecord]:
        if not observations:
            return (
                RiskRecord(timestamp, None, 0.0, RiskLevel.SAFE, None, None),
                DecisionRecord(timestamp, NavigationAction.CONTINUE, None, 0.0, 1.0),
            )

        min_dist = min(math.hypot(obs.position.x, obs.position.z) for obs in observations)

        if min_dist < self.stop_threshold:
            risk = RiskRecord(timestamp, 1, 90.0, RiskLevel.CRITICAL, None, min_dist)
            dec = DecisionRecord(timestamp, NavigationAction.STOP, 1, 0.0, 0.9)
        elif min_dist < self.caution_threshold:
            risk = RiskRecord(timestamp, 1, 50.0, RiskLevel.MEDIUM, None, min_dist)
            dec = DecisionRecord(timestamp, NavigationAction.CAUTION, 1, 0.0, 0.7)
        else:
            risk = RiskRecord(timestamp, 1, 10.0, RiskLevel.SAFE, None, min_dist)
            dec = DecisionRecord(timestamp, NavigationAction.CONTINUE, None, 0.0, 1.0)

        return risk, dec


class TTCBaseline:
    """Reactive instantaneous TTC baseline.

    Calculates raw instantaneous TTC using consecutive frame positions without Kalman filtering,
    without temporal EMA risk smoothing, and without uncertainty-aware collision likelihood.
    """

    def __init__(self, ttc_threshold: float = 2.5) -> None:
        self.ttc_threshold = ttc_threshold
        self._prev_positions: dict[str, tuple[float, float, float]] = {}  # id -> (x, z, t)

    def reset(self) -> None:
        self._prev_positions.clear()

    def process(
        self,
        observations: list[PhysicalObservation],
        timestamp: float,
        ego_state: EgoMotionState | None = None,
    ) -> tuple[RiskRecord, DecisionRecord]:
        if not observations:
            return (
                RiskRecord(timestamp, None, 0.0, RiskLevel.SAFE, None, None),
                DecisionRecord(timestamp, NavigationAction.CONTINUE, None, 0.0, 1.0),
            )

        min_ttc: float | None = None
        min_dist = float("inf")

        for obs in observations:
            x, z = obs.position.x, obs.position.z
            dist = math.hypot(x, z)
            if dist < min_dist:
                min_dist = dist

            obs_id = obs.detection_id
            if obs_id in self._prev_positions:
                px, pz, pt = self._prev_positions[obs_id]
                dt = timestamp - pt
                if dt > 1e-4:
                    vx = (x - px) / dt
                    vz = (z - pz) / dt
                    v_sq = vx**2 + vz**2
                    if v_sq > 1e-4:
                        raw_ttc = -(x * vx + z * vz) / v_sq
                        if raw_ttc > 0.0:
                            if min_ttc is None or raw_ttc < min_ttc:
                                min_ttc = raw_ttc
            self._prev_positions[obs_id] = (x, z, timestamp)

        if min_ttc is not None and min_ttc <= self.ttc_threshold:
            risk = RiskRecord(timestamp, 1, 85.0, RiskLevel.HIGH, min_ttc, min_dist)
            dec = DecisionRecord(timestamp, NavigationAction.STOP, 1, 0.0, 0.85)
        else:
            risk = RiskRecord(timestamp, 1, 10.0, RiskLevel.SAFE, min_ttc, min_dist)
            dec = DecisionRecord(timestamp, NavigationAction.CONTINUE, None, 0.0, 1.0)

        return risk, dec


class PRISMEvaluator:
    """Evaluates the full PRISMPipeline (Parts 1–10)."""

    def __init__(self, pipeline: PRISMPipeline) -> None:
        self.pipeline = pipeline

    def reset(self) -> None:
        self.pipeline.reset()

    def process(
        self,
        observations: list[PhysicalObservation],
        timestamp: float,
        ego_state: EgoMotionState | None = None,
    ) -> tuple[RiskRecord, DecisionRecord]:
        res: RuntimeResult = self.pipeline.process_frame(
            raw_frame=None,
            timestamp=timestamp,
            ego_state=ego_state,
            explicit_observations=observations,
        )

        if res.risk_assessment is not None:
            risk = RiskRecord(
                timestamp=timestamp,
                object_id=res.risk_assessment.object_id,
                risk_score=res.risk_assessment.risk_score,
                risk_level=res.risk_assessment.risk_level,
                ttc=res.risk_assessment.ttc,
                cpa_distance=res.risk_assessment.cpa_distance,
            )
        else:
            risk = RiskRecord(
                timestamp=timestamp,
                object_id=None,
                risk_score=0.0,
                risk_level=RiskLevel.SAFE,
                ttc=None,
                cpa_distance=None,
            )

        dec = DecisionRecord(
            timestamp=timestamp,
            action=res.decision.action,
            primary_threat_id=res.decision.primary_threat_id,
            recommended_heading=res.decision.recommended_heading,
            confidence=res.decision.confidence,
        )

        return risk, dec

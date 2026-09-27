"""Purely proximity-based reactive baseline for controlled scientific comparison against PRISM."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from app.core.enums import DecisionType, RiskLevel
from app.core.types import Decision, ObjectObservation, RiskAssessment, Vector2
from app.decision.model import DecisionReason, NavigationAction, NavigationDecision
from evaluation.baselines import SystemEvaluator
from evaluation.models import DecisionRecord, EvaluationSample, GroundTruthState, PredictionRecord, RiskRecord


class ProximityReactiveBaseline(SystemEvaluator):
    """Reactive proximity baseline triggering warnings solely based on instantaneous distance."""

    def __init__(self, stop_threshold: float = 1.8, caution_threshold: float = 3.0) -> None:
        self.stop_threshold = stop_threshold
        self.caution_threshold = caution_threshold

    def evaluate_frame(
        self,
        frame_id: int,
        timestamp: float,
        ground_truth: dict[int, GroundTruthState],
        observations: Sequence[ObjectObservation],
        is_scene_hazard: bool,
    ) -> EvaluationSample:
        """Process frame solely using instantaneous detected obstacle distance."""
        min_dist = float("inf")
        closest_id = None

        for obs in observations:
            # Estimate distance from bounding box bottom-center heuristic or direct range
            # Bottom center v -> distance proxy
            u, v = obs.bbox.bottom_center
            # Synthetic camera depth proxy
            depth = max(0.5, (480.0 - v) * 0.03)
            if depth < min_dist:
                min_dist = depth
                closest_id = obs.object_id if obs.object_id is not None else 1

        if min_dist <= self.stop_threshold:
            action = NavigationAction.STOP
            risk_lvl = RiskLevel.CRITICAL
            score = 90.0
            reason = DecisionReason.EMERGENCY_COLLISION_RISK
        elif min_dist <= self.caution_threshold:
            action = NavigationAction.CAUTION
            risk_lvl = RiskLevel.MEDIUM
            score = 50.0
            reason = DecisionReason.CLOSE_PROXIMITY
        else:
            action = NavigationAction.CONTINUE
            risk_lvl = RiskLevel.SAFE
            score = 0.0
            reason = DecisionReason.NO_HAZARD

        risk_rec = RiskRecord(
            timestamp=timestamp,
            object_id=closest_id,
            risk_score=score,
            risk_level=risk_lvl,
            ttc=None,
            cpa_distance=min_dist if math.isfinite(min_dist) else None,
        )

        dec_rec = DecisionRecord(
            timestamp=timestamp,
            action=action,
            primary_threat_id=closest_id,
            recommended_heading=0.0,
            confidence=1.0,
        )

        return EvaluationSample(
            timestamp=timestamp,
            frame_id=frame_id,
            ground_truth=ground_truth,
            is_scene_hazard=is_scene_hazard,
            predictions={},
            risk=risk_rec,
            decision=dec_rec,
            processing_latency_ms=0.05,
        )

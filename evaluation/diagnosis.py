"""Failure diagnosis: Frame-level causal tracing and subsystem attribution for missed hazards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from app.core.enums import RiskLevel
from app.decision.model import NavigationAction
from evaluation.models import EvaluationSample
from evaluation.validation import FailureCategory


@dataclass(frozen=True)
class DiagnosisRecord:
    """Frame-level causal diagnostic record attributing the earliest failing subsystem."""

    timestamp: float
    frame_id: int
    is_scene_hazard: bool
    perception_valid: bool
    track_valid: bool
    state_valid: bool
    prediction_valid: bool
    ttc_valid: bool
    risk_score: float
    risk_level: RiskLevel
    action: NavigationAction
    failure_category: FailureCategory | None
    causal_chain: str


class CausalFailureDiagnoser:
    """Performs step-by-step causal tracing across the pipeline to diagnose missed hazards."""

    @staticmethod
    def diagnose_frame(sample: EvaluationSample) -> DiagnosisRecord:
        """Trace pipeline dataflow for a single frame and determine earliest failing stage."""
        if not sample.is_scene_hazard:
            return DiagnosisRecord(
                timestamp=sample.timestamp,
                frame_id=sample.frame_id,
                is_scene_hazard=False,
                perception_valid=True,
                track_valid=True,
                state_valid=True,
                prediction_valid=True,
                ttc_valid=True,
                risk_score=sample.risk.risk_score,
                risk_level=sample.risk.risk_level,
                action=sample.decision.action,
                failure_category=None,
                causal_chain="SCENE_SAFE -> NO_FAILURE",
            )

        # Active scene hazard: check if appropriately mitigated
        is_high_risk = sample.risk.risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL)
        is_action_taken = sample.decision.action in (
            NavigationAction.STOP,
            NavigationAction.EMERGENCY_STOP,
            NavigationAction.TURN_LEFT,
            NavigationAction.TURN_RIGHT,
            NavigationAction.SLOW_DOWN,
            NavigationAction.CAUTION,
        )

        if is_high_risk or is_action_taken:
            return DiagnosisRecord(
                timestamp=sample.timestamp,
                frame_id=sample.frame_id,
                is_scene_hazard=True,
                perception_valid=True,
                track_valid=True,
                state_valid=True,
                prediction_valid=True,
                ttc_valid=True,
                risk_score=sample.risk.risk_score,
                risk_level=sample.risk.risk_level,
                action=sample.decision.action,
                failure_category=None,
                causal_chain="HAZARD_ACTIVE -> DETECTED_AND_MITIGATED",
            )

        # Causal tracing for unmitigated hazard
        # Stage 1: Perception
        if not sample.predictions:
            return DiagnosisRecord(
                timestamp=sample.timestamp,
                frame_id=sample.frame_id,
                is_scene_hazard=True,
                perception_valid=False,
                track_valid=False,
                state_valid=False,
                prediction_valid=False,
                ttc_valid=False,
                risk_score=sample.risk.risk_score,
                risk_level=sample.risk.risk_level,
                action=sample.decision.action,
                failure_category=FailureCategory.PERCEPTION_FAILURE,
                causal_chain="HAZARD_ACTIVE -> NO_PERCEPTION_OBSERVATION -> PERCEPTION_FAILURE",
            )

        # Stage 2: Tracking & State Estimation
        if sample.risk.object_id is None:
            return DiagnosisRecord(
                timestamp=sample.timestamp,
                frame_id=sample.frame_id,
                is_scene_hazard=True,
                perception_valid=True,
                track_valid=False,
                state_valid=False,
                prediction_valid=False,
                ttc_valid=False,
                risk_score=sample.risk.risk_score,
                risk_level=sample.risk.risk_level,
                action=sample.decision.action,
                failure_category=FailureCategory.TRACKING_FAILURE,
                causal_chain="OBSERVATION_PRESENT -> TRACK_UNCONFIRMED -> TRACKING_FAILURE",
            )

        # Stage 3: TTC / Motion Prediction
        if sample.risk.ttc is None or sample.risk.ttc > 5.0:
            return DiagnosisRecord(
                timestamp=sample.timestamp,
                frame_id=sample.frame_id,
                is_scene_hazard=True,
                perception_valid=True,
                track_valid=True,
                state_valid=True,
                prediction_valid=True,
                ttc_valid=False,
                risk_score=sample.risk.risk_score,
                risk_level=sample.risk.risk_level,
                action=sample.decision.action,
                failure_category=FailureCategory.TTC_FAILURE,
                causal_chain="TRACK_VALID -> TTC_ABSENT_OR_OVERESTIMATED -> TTC_FAILURE",
            )

        # Stage 4: Risk Scoring / Weighting
        if sample.risk.risk_score < 45.0:  # Below medium threshold
            return DiagnosisRecord(
                timestamp=sample.timestamp,
                frame_id=sample.frame_id,
                is_scene_hazard=True,
                perception_valid=True,
                track_valid=True,
                state_valid=True,
                prediction_valid=True,
                ttc_valid=True,
                risk_score=sample.risk.risk_score,
                risk_level=sample.risk.risk_level,
                action=sample.decision.action,
                failure_category=FailureCategory.RISK_CLASSIFICATION_FAILURE,
                causal_chain="TTC_VALID -> MULTI_FACTOR_SCORE_UNDER_THRESHOLD -> RISK_SCORE_FAILURE",
            )

        # Stage 5: Decision Engine Arbitration
        if sample.decision.action == NavigationAction.CONTINUE:
            return DiagnosisRecord(
                timestamp=sample.timestamp,
                frame_id=sample.frame_id,
                is_scene_hazard=True,
                perception_valid=True,
                track_valid=True,
                state_valid=True,
                prediction_valid=True,
                ttc_valid=True,
                risk_score=sample.risk.risk_score,
                risk_level=sample.risk.risk_level,
                action=sample.decision.action,
                failure_category=FailureCategory.DECISION_FAILURE,
                causal_chain="RISK_HIGH -> DECISION_COMMANDED_CONTINUE -> DECISION_FAILURE",
            )

        return DiagnosisRecord(
            timestamp=sample.timestamp,
            frame_id=sample.frame_id,
            is_scene_hazard=True,
            perception_valid=True,
            track_valid=True,
            state_valid=True,
            prediction_valid=True,
            ttc_valid=True,
            risk_score=sample.risk.risk_score,
            risk_level=sample.risk.risk_level,
            action=sample.decision.action,
            failure_category=FailureCategory.UNKNOWN,
            causal_chain="UNCLASSIFIED_FAILURE",
        )

    @classmethod
    def diagnose_samples(cls, samples: Sequence[EvaluationSample]) -> list[DiagnosisRecord]:
        """Diagnose a complete sequence of evaluation samples."""
        return [cls.diagnose_frame(s) for s in samples]

    @classmethod
    def summarize_failure_counts(cls, diagnoses: Sequence[DiagnosisRecord]) -> dict[str, int]:
        """Aggregate total counts of each failure category across diagnosed records."""
        counts: dict[str, int] = {}
        for d in diagnoses:
            if d.failure_category is not None:
                cat_name = d.failure_category.value
                counts[cat_name] = counts.get(cat_name, 0) + 1
        return counts

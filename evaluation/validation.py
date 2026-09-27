"""Validation & failure analysis: Categorizes and isolates subsystem root causes of missed hazards."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence

from app.core.enums import RiskLevel
from app.decision.model import NavigationAction
from evaluation.models import EvaluationSample


class FailureCategory(str, Enum):
    """Subsystem failure taxonomy for unmitigated hazards and navigation collisions."""

    PERCEPTION_FAILURE = "PERCEPTION_FAILURE"
    PROJECTION_FAILURE = "PROJECTION_FAILURE"
    TRACKING_FAILURE = "TRACKING_FAILURE"
    STATE_ESTIMATION_FAILURE = "STATE_ESTIMATION_FAILURE"
    MOTION_MODEL_FAILURE = "MOTION_MODEL_FAILURE"
    TTC_FAILURE = "TTC_FAILURE"
    RISK_CLASSIFICATION_FAILURE = "RISK_CLASSIFICATION_FAILURE"
    DECISION_FAILURE = "DECISION_FAILURE"
    LATENCY_FAILURE = "LATENCY_FAILURE"
    NO_SAFE_PATH = "NO_SAFE_PATH"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class FailureCase:
    """Detailed diagnostic report for an individual failure instance."""

    scenario_id: str
    trial_index: int
    timestamp: float
    category: FailureCategory
    explanation: str
    sample: EvaluationSample


class FailureClassifier:
    """Analyzes failed evaluation samples to attribute root-cause subsystem failures."""

    @staticmethod
    def classify_missed_hazard(
        sample: EvaluationSample,
        scenario_id: str = "UNKNOWN",
        trial_index: int = 0,
    ) -> FailureCase:
        """Diagnose why an active hazard was not mitigated."""
        # 1. Check if ground-truth objects were observed
        if not sample.predictions:
            return FailureCase(
                scenario_id=scenario_id,
                trial_index=trial_index,
                timestamp=sample.timestamp,
                category=FailureCategory.PERCEPTION_FAILURE,
                explanation="No observations or tracks were extracted for active ground-truth obstacles.",
                sample=sample,
            )

        # 2. Check risk classification
        if sample.risk.risk_level in (RiskLevel.SAFE, RiskLevel.LOW):
            if sample.risk.ttc is not None and sample.risk.ttc > 4.0:
                return FailureCase(
                    scenario_id=scenario_id,
                    trial_index=trial_index,
                    timestamp=sample.timestamp,
                    category=FailureCategory.TTC_FAILURE,
                    explanation=f"Overestimated TTC ({sample.risk.ttc:.2f}s) suppressed risk classification.",
                    sample=sample,
                )
            return FailureCase(
                scenario_id=scenario_id,
                trial_index=trial_index,
                timestamp=sample.timestamp,
                category=FailureCategory.RISK_CLASSIFICATION_FAILURE,
                explanation="Active collision course was evaluated as SAFE or LOW risk by the risk engine.",
                sample=sample,
            )

        # 3. Check decision engine action
        if sample.decision.action in (NavigationAction.CONTINUE,):
            return FailureCase(
                scenario_id=scenario_id,
                trial_index=trial_index,
                timestamp=sample.timestamp,
                category=FailureCategory.DECISION_FAILURE,
                explanation="High risk was detected but decision engine commanded CONTINUE.",
                sample=sample,
            )

        # 4. Check latency
        if sample.processing_latency_ms > 100.0:
            return FailureCase(
                scenario_id=scenario_id,
                trial_index=trial_index,
                timestamp=sample.timestamp,
                category=FailureCategory.LATENCY_FAILURE,
                explanation=f"Processing latency ({sample.processing_latency_ms:.1f}ms) exceeded deadline.",
                sample=sample,
            )

        return FailureCase(
            scenario_id=scenario_id,
            trial_index=trial_index,
            timestamp=sample.timestamp,
            category=FailureCategory.UNKNOWN,
            explanation="Unclassified failure mode.",
            sample=sample,
        )

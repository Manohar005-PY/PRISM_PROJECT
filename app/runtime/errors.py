"""Runtime exceptions, failure isolation categories, and fail-safe decision generators."""

from __future__ import annotations

from app.core.enums import RiskLevel
from app.core.types import RiskAssessment
from app.decision.model import DecisionReason, NavigationAction, NavigationDecision


class PRISMRuntimeError(Exception):
    """Base exception for PRISM runtime failures."""


class StageExecutionError(PRISMRuntimeError):
    """Raised when a specific pipeline stage fails to execute."""

    def __init__(self, stage_name: str, message: str, original_exception: Exception | None = None) -> None:
        super().__init__(f"Stage [{stage_name}] failed: {message}")
        self.stage_name = stage_name
        self.original_exception = original_exception


class StaleDataError(PRISMRuntimeError):
    """Raised when pipeline input data exceeds allowable staleness timeout."""


def create_fail_safe_risk_assessment(timestamp: float) -> RiskAssessment:
    """Create a conservative fail-safe RiskAssessment."""
    return RiskAssessment(
        object_id=0,
        risk_level=RiskLevel.HIGH,
        risk_score=75.0,
        confidence=1.0,
        timestamp=timestamp,
        ttc=0.0,
        cpa_distance=0.0,
    )


def create_fail_safe_decision(
    timestamp: float,
    reason: DecisionReason = DecisionReason.EMERGENCY_COLLISION_RISK,
    action: NavigationAction = NavigationAction.STOP,
) -> NavigationDecision:
    """Create an emergency or conservative fail-safe NavigationDecision."""
    return NavigationDecision(
        action=action,
        reason=reason,
        confidence=1.0,
        primary_threat_id=None,
        recommended_heading=0.0,
        risk_level=RiskLevel.HIGH,
        timestamp=timestamp,
    )

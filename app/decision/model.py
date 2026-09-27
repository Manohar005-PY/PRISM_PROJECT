"""Domain models and semantic action types for PRISM Decision Engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from app.core.enums import RiskLevel


class NavigationAction(str, Enum):
    """Discrete navigation recommendation issued to the user or presentation runtime."""

    EMERGENCY_STOP = "emergency_stop"
    STOP = "stop"
    SLOW_DOWN = "slow_down"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    CAUTION = "caution"
    CONTINUE = "continue"


def get_action_priority(action: NavigationAction) -> int:
    """Return explicit integer priority for a navigation action (higher = greater safety precedence).

    Precedence: EMERGENCY_STOP (6) > STOP (5) > TURN (4) > SLOW_DOWN (3) > CAUTION (2) > CONTINUE (1).
    """
    if action == NavigationAction.EMERGENCY_STOP:
        return 6
    elif action == NavigationAction.STOP:
        return 5
    elif action in (NavigationAction.TURN_LEFT, NavigationAction.TURN_RIGHT):
        return 4
    elif action == NavigationAction.SLOW_DOWN:
        return 3
    elif action == NavigationAction.CAUTION:
        return 2
    elif action == NavigationAction.CONTINUE:
        return 1
    raise ValueError(f"Unknown action {action!r}")


class DecisionReason(str, Enum):
    """Semantic reason justifying a chosen navigation action."""

    EMERGENCY_COLLISION_RISK = "emergency_collision_risk"
    CRITICAL_COLLISION_RISK = "critical_collision_risk"
    HIGH_COLLISION_RISK = "high_collision_risk"
    MEDIUM_COLLISION_RISK = "medium_collision_risk"
    LOW_COLLISION_RISK = "low_collision_risk"
    PATH_BLOCKED = "path_blocked"
    SAFE_PATH_AVAILABLE = "safe_path_available"
    NO_SAFE_PATH = "no_safe_path"
    NO_HAZARD = "no_hazard"
    HYSTERESIS_HOLD = "hysteresis_hold"


@dataclass(frozen=True)
class NavigationDecision:
    """Immutable, machine-readable navigation recommendation produced by DecisionEngine.

    Attributes:
        action: Selected discrete NavigationAction.
        reason: Justifying DecisionReason.
        confidence: Confidence score in [0.0, 1.0].
        primary_threat_id: ID of the primary obstacle prompting the action (or None).
        recommended_heading: Recommended ground-plane evasion angle in radians (or None).
        risk_level: Evaluated RiskLevel from Part 6.
        timestamp: Timestamp of the decision in seconds.
    """

    action: NavigationAction
    reason: DecisionReason
    confidence: float
    primary_threat_id: int | None
    recommended_heading: float | None
    risk_level: RiskLevel
    timestamp: float

    def __post_init__(self) -> None:
        if not isinstance(self.action, NavigationAction):
            raise TypeError(f"action must be NavigationAction, got {type(self.action)}")
        if not isinstance(self.reason, DecisionReason):
            raise TypeError(f"reason must be DecisionReason, got {type(self.reason)}")
        if not isinstance(self.confidence, (int, float)) or not math.isfinite(self.confidence):
            raise ValueError(f"confidence must be a finite float, got {self.confidence!r}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")
        if self.primary_threat_id is not None and not isinstance(self.primary_threat_id, int):
            raise TypeError(f"primary_threat_id must be int or None, got {type(self.primary_threat_id)}")
        if self.recommended_heading is not None:
            if not isinstance(self.recommended_heading, (int, float)) or not math.isfinite(self.recommended_heading):
                raise ValueError(f"recommended_heading must be a finite float or None, got {self.recommended_heading!r}")
            object.__setattr__(self, "recommended_heading", float(self.recommended_heading))
        if not isinstance(self.risk_level, RiskLevel):
            raise TypeError(f"risk_level must be RiskLevel, got {type(self.risk_level)}")
        if not isinstance(self.timestamp, (int, float)) or not math.isfinite(self.timestamp):
            raise ValueError(f"timestamp must be a finite float, got {self.timestamp!r}")

        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "timestamp", float(self.timestamp))

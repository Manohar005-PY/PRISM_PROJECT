"""Semantic enums for PRISM core domain."""

from enum import Enum


class RiskLevel(str, Enum):
    """Semantic risk category representing the severity of an estimated hazard."""

    SAFE = "SAFE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class DecisionType(str, Enum):
    """Actionable alert decision issued by the decision engine."""

    NO_ACTION = "NO_ACTION"
    CAUTION = "CAUTION"
    WARNING = "WARNING"
    STOP = "STOP"

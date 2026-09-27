"""Evaluation domain models: Ground truth states, prediction records, risk records, and evaluation samples."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from app.core.enums import RiskLevel
from app.core.types import Vector2
from app.decision.model import NavigationAction


@dataclass(frozen=True)
class GroundTruthState:
    """Immutable ground-truth physical state of an obstacle at a given timestamp."""

    timestamp: float
    object_id: int
    class_name: str
    position: Vector2
    velocity: Vector2
    is_hazard: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, (int, float)) or not math.isfinite(self.timestamp):
            raise ValueError(f"timestamp must be a finite float, got {self.timestamp!r}")
        if not isinstance(self.object_id, int):
            raise TypeError(f"object_id must be int, got {type(self.object_id)}")
        if not isinstance(self.class_name, str):
            raise TypeError(f"class_name must be str, got {type(self.class_name)}")
        if not isinstance(self.position, Vector2):
            raise TypeError(f"position must be Vector2, got {type(self.position)}")
        if not isinstance(self.velocity, Vector2):
            raise TypeError(f"velocity must be Vector2, got {type(self.velocity)}")
        if not isinstance(self.is_hazard, bool):
            raise TypeError(f"is_hazard must be bool, got {type(self.is_hazard)}")

        object.__setattr__(self, "timestamp", float(self.timestamp))


@dataclass(frozen=True)
class PredictionRecord:
    """Predicted future state of an obstacle evaluated at a specific forward horizon."""

    timestamp: float
    object_id: int
    predicted_position: Vector2
    predicted_velocity: Vector2
    prediction_horizon: float

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, (int, float)) or not math.isfinite(self.timestamp):
            raise ValueError(f"timestamp must be a finite float, got {self.timestamp!r}")
        if not isinstance(self.object_id, int):
            raise TypeError(f"object_id must be int, got {type(self.object_id)}")
        if not isinstance(self.predicted_position, Vector2):
            raise TypeError(f"predicted_position must be Vector2, got {type(self.predicted_position)}")
        if not isinstance(self.predicted_velocity, Vector2):
            raise TypeError(f"predicted_velocity must be Vector2, got {type(self.predicted_velocity)}")
        if not isinstance(self.prediction_horizon, (int, float)) or not math.isfinite(self.prediction_horizon) or self.prediction_horizon < 0.0:
            raise ValueError(f"prediction_horizon must be non-negative float, got {self.prediction_horizon!r}")

        object.__setattr__(self, "timestamp", float(self.timestamp))
        object.__setattr__(self, "prediction_horizon", float(self.prediction_horizon))


@dataclass(frozen=True)
class RiskRecord:
    """Extracted risk assessment evaluated for a frame."""

    timestamp: float
    object_id: int | None
    risk_score: float
    risk_level: RiskLevel
    ttc: float | None
    cpa_distance: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, (int, float)) or not math.isfinite(self.timestamp):
            raise ValueError(f"timestamp must be finite float, got {self.timestamp!r}")
        if self.object_id is not None and not isinstance(self.object_id, int):
            raise TypeError(f"object_id must be int or None, got {type(self.object_id)}")
        if not isinstance(self.risk_score, (int, float)) or not math.isfinite(self.risk_score):
            raise ValueError(f"risk_score must be finite float, got {self.risk_score!r}")
        if not isinstance(self.risk_level, RiskLevel):
            raise TypeError(f"risk_level must be RiskLevel enum, got {type(self.risk_level)}")
        if self.ttc is not None and (not isinstance(self.ttc, (int, float)) or not math.isfinite(self.ttc)):
            raise ValueError(f"ttc must be finite float or None, got {self.ttc!r}")
        if self.cpa_distance is not None and (not isinstance(self.cpa_distance, (int, float)) or not math.isfinite(self.cpa_distance)):
            raise ValueError(f"cpa_distance must be finite float or None, got {self.cpa_distance!r}")

        object.__setattr__(self, "timestamp", float(self.timestamp))
        object.__setattr__(self, "risk_score", float(self.risk_score))
        if self.ttc is not None:
            object.__setattr__(self, "ttc", float(self.ttc))
        if self.cpa_distance is not None:
            object.__setattr__(self, "cpa_distance", float(self.cpa_distance))


@dataclass(frozen=True)
class DecisionRecord:
    """Extracted navigation decision for a frame."""

    timestamp: float
    action: NavigationAction
    primary_threat_id: int | None
    recommended_heading: float | None
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, (int, float)) or not math.isfinite(self.timestamp):
            raise ValueError(f"timestamp must be finite float, got {self.timestamp!r}")
        if not isinstance(self.action, NavigationAction):
            raise TypeError(f"action must be NavigationAction, got {type(self.action)}")
        if self.primary_threat_id is not None and not isinstance(self.primary_threat_id, int):
            raise TypeError(f"primary_threat_id must be int or None, got {type(self.primary_threat_id)}")
        if self.recommended_heading is not None and (not isinstance(self.recommended_heading, (int, float)) or not math.isfinite(self.recommended_heading)):
            raise ValueError(f"recommended_heading must be finite float or None, got {self.recommended_heading!r}")
        if not isinstance(self.confidence, (int, float)) or not math.isfinite(self.confidence) or not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence!r}")

        object.__setattr__(self, "timestamp", float(self.timestamp))
        object.__setattr__(self, "confidence", float(self.confidence))
        if self.recommended_heading is not None:
            object.__setattr__(self, "recommended_heading", float(self.recommended_heading))


@dataclass(frozen=True)
class EvaluationSample:
    """Synchronized ground truth and system outputs for a single evaluation timestep."""

    timestamp: float
    frame_id: int
    ground_truth: Mapping[int, GroundTruthState]
    is_scene_hazard: bool
    predictions: Mapping[int, PredictionRecord]
    risk: RiskRecord
    decision: DecisionRecord
    processing_latency_ms: float

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, (int, float)) or not math.isfinite(self.timestamp):
            raise ValueError(f"timestamp must be finite float, got {self.timestamp!r}")
        if not isinstance(self.frame_id, int) or self.frame_id < 0:
            raise ValueError(f"frame_id must be non-negative int, got {self.frame_id!r}")
        if not isinstance(self.ground_truth, Mapping):
            raise TypeError(f"ground_truth must be Mapping[int, GroundTruthState], got {type(self.ground_truth)}")
        if not isinstance(self.is_scene_hazard, bool):
            raise TypeError(f"is_scene_hazard must be bool, got {type(self.is_scene_hazard)}")
        if not isinstance(self.risk, RiskRecord):
            raise TypeError(f"risk must be RiskRecord, got {type(self.risk)}")
        if not isinstance(self.decision, DecisionRecord):
            raise TypeError(f"decision must be DecisionRecord, got {type(self.decision)}")
        if not isinstance(self.processing_latency_ms, (int, float)) or not math.isfinite(self.processing_latency_ms) or self.processing_latency_ms < 0.0:
            raise ValueError(f"processing_latency_ms must be non-negative float, got {self.processing_latency_ms!r}")

        object.__setattr__(self, "timestamp", float(self.timestamp))
        object.__setattr__(self, "processing_latency_ms", float(self.processing_latency_ms))

"""Core domain types for PRISM.

All core types are immutable dataclasses with explicit validation.
These types represent the foundational contracts between PRISM subsystems:
Perception -> State Estimation -> Prediction -> Risk Assessment -> Decision -> Presentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Mapping, Sequence

from app.core.enums import DecisionType, RiskLevel

# 4x4 state covariance representing uncertainty over [X, Z, Vx, Vz]
Covariance4x4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]


def validate_covariance_4x4(cov: Covariance4x4 | None) -> None:
    """Validate that covariance is either None or a valid 4x4 matrix of finite floats.

    Args:
        cov: The covariance matrix to validate.

    Raises:
        TypeError: If cov is not a tuple of tuples.
        ValueError: If cov is not 4x4 or contains non-finite values.
    """
    if cov is None:
        return
    if not isinstance(cov, tuple) or len(cov) != 4:
        raise ValueError(f"Covariance must be a 4-tuple of 4-tuples, got {type(cov)} of length {len(cov) if isinstance(cov, tuple) else 'N/A'}")
    for i, row in enumerate(cov):
        if not isinstance(row, tuple) or len(row) != 4:
            raise ValueError(f"Covariance row {i} must be a 4-tuple, got {type(row)} of length {len(row) if isinstance(row, tuple) else 'N/A'}")
        for j, val in enumerate(row):
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"Covariance entry ({i}, {j}) must be a finite float, got {val!r}")


@dataclass(frozen=True)
class Vector2:
    """Immutable 2D vector on the ground plane.

    PRISM coordinates:
        x: Lateral position or velocity (meters or m/s; positive = right).
        z: Forward / depth position or velocity (meters or m/s; positive = forward).
    """

    x: float
    z: float

    def __post_init__(self) -> None:
        if not isinstance(self.x, (int, float)) or not math.isfinite(self.x):
            raise ValueError(f"Vector2.x must be a finite float, got {self.x!r}")
        if not isinstance(self.z, (int, float)) or not math.isfinite(self.z):
            raise ValueError(f"Vector2.z must be a finite float, got {self.z!r}")
        object.__setattr__(self, "x", float(self.x))
        object.__setattr__(self, "z", float(self.z))

    def __add__(self, other: Vector2) -> Vector2:
        if not isinstance(other, Vector2):
            return NotImplemented
        return Vector2(self.x + other.x, self.z + other.z)

    def __sub__(self, other: Vector2) -> Vector2:
        if not isinstance(other, Vector2):
            return NotImplemented
        return Vector2(self.x - other.x, self.z - other.z)

    def __mul__(self, scalar: float | int) -> Vector2:
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return Vector2(self.x * scalar, self.z * scalar)

    def __rmul__(self, scalar: float | int) -> Vector2:
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return Vector2(self.x * scalar, self.z * scalar)

    @property
    def magnitude(self) -> float:
        """Euclidean magnitude / norm of the vector in meters or m/s."""
        return math.hypot(self.x, self.z)


@dataclass(frozen=True)
class BoundingBox:
    """2D detection bounding box in pixel coordinates (image space).

    Coordinates:
        x_min: Left edge.
        y_min: Top edge.
        x_max: Right edge.
        y_max: Bottom edge.
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        for name, val in [("x_min", self.x_min), ("y_min", self.y_min), ("x_max", self.x_max), ("y_max", self.y_max)]:
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"BoundingBox.{name} must be a finite float, got {val!r}")
        if self.x_max < self.x_min:
            raise ValueError(f"x_max ({self.x_max}) cannot be less than x_min ({self.x_min})")
        if self.y_max < self.y_min:
            raise ValueError(f"y_max ({self.y_max}) cannot be less than y_min ({self.y_min})")
        object.__setattr__(self, "x_min", float(self.x_min))
        object.__setattr__(self, "y_min", float(self.y_min))
        object.__setattr__(self, "x_max", float(self.x_max))
        object.__setattr__(self, "y_max", float(self.y_max))

    @property
    def width(self) -> float:
        """Width of the bounding box in pixels."""
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        """Height of the bounding box in pixels."""
        return self.y_max - self.y_min

    @property
    def center(self) -> tuple[float, float]:
        """Geometric center (cx, cy) of the bounding box in pixels."""
        return ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0)

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Bottom-center (cx, y_max) in pixels, used as the contact point for ground-plane projection."""
        return ((self.x_min + self.x_max) / 2.0, self.y_max)


@dataclass(frozen=True)
class ObjectObservation:
    """Perception output representing what the detector saw in a single image frame.

    Note:
        Contains NO ground-plane position, velocity, or risk estimation.
        Observation answers: "What did perception detect in this frame?"
    """

    class_id: int | str
    confidence: float
    bbox: BoundingBox
    timestamp: float
    object_id: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.confidence, (int, float)) or not math.isfinite(self.confidence):
            raise ValueError(f"ObjectObservation.confidence must be a finite float, got {self.confidence!r}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"ObjectObservation.confidence must be in [0.0, 1.0], got {self.confidence}")
        if not isinstance(self.timestamp, (int, float)) or not math.isfinite(self.timestamp):
            raise ValueError(f"ObjectObservation.timestamp must be a finite float, got {self.timestamp!r}")
        if not isinstance(self.bbox, BoundingBox):
            raise TypeError(f"ObjectObservation.bbox must be a BoundingBox instance, got {type(self.bbox)}")
        if self.object_id is not None and not isinstance(self.object_id, int):
            raise TypeError(f"ObjectObservation.object_id must be int or None, got {type(self.object_id)}")
        if not isinstance(self.class_id, (int, str)):
            raise TypeError(f"ObjectObservation.class_id must be int or str, got {type(self.class_id)}")
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "timestamp", float(self.timestamp))


@dataclass(frozen=True)
class ObjectState:
    """Estimated physical state of a tracked object on the ground plane.

    State representation: [X, Z, Vx, Vz]
    Answers: "Where do we estimate the object actually is, and how is it moving in the physical world?"
    """

    object_id: int
    class_id: int | str
    position: Vector2
    velocity: Vector2
    timestamp: float
    covariance: Covariance4x4 | None = None
    confidence: float = 1.0
    age: int = 1
    time_since_update: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.object_id, int):
            raise TypeError(f"ObjectState.object_id must be int, got {type(self.object_id)}")
        if not isinstance(self.class_id, (int, str)):
            raise TypeError(f"ObjectState.class_id must be int or str, got {type(self.class_id)}")
        if not isinstance(self.position, Vector2):
            raise TypeError(f"ObjectState.position must be a Vector2, got {type(self.position)}")
        if not isinstance(self.velocity, Vector2):
            raise TypeError(f"ObjectState.velocity must be a Vector2, got {type(self.velocity)}")
        if not isinstance(self.timestamp, (int, float)) or not math.isfinite(self.timestamp):
            raise ValueError(f"ObjectState.timestamp must be a finite float, got {self.timestamp!r}")
        if not isinstance(self.confidence, (int, float)) or not math.isfinite(self.confidence):
            raise ValueError(f"ObjectState.confidence must be a finite float, got {self.confidence!r}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"ObjectState.confidence must be in [0.0, 1.0], got {self.confidence}")
        if not isinstance(self.age, int) or self.age < 0:
            raise ValueError(f"ObjectState.age must be a non-negative integer, got {self.age!r}")
        if not isinstance(self.time_since_update, (int, float)) or not math.isfinite(self.time_since_update) or self.time_since_update < 0.0:
            raise ValueError(f"ObjectState.time_since_update must be a non-negative finite float, got {self.time_since_update!r}")
        validate_covariance_4x4(self.covariance)
        object.__setattr__(self, "timestamp", float(self.timestamp))
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "time_since_update", float(self.time_since_update))


@dataclass(frozen=True)
class TrajectoryPoint:
    """Predicted future physical state at a specific timestamp.

    Answers: "Where do we expect the object to be at this future moment?"
    """

    timestamp: float
    position: Vector2
    velocity: Vector2
    covariance: Covariance4x4 | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, (int, float)) or not math.isfinite(self.timestamp):
            raise ValueError(f"TrajectoryPoint.timestamp must be a finite float, got {self.timestamp!r}")
        if not isinstance(self.position, Vector2):
            raise TypeError(f"TrajectoryPoint.position must be a Vector2, got {type(self.position)}")
        if not isinstance(self.velocity, Vector2):
            raise TypeError(f"TrajectoryPoint.velocity must be a Vector2, got {type(self.velocity)}")
        validate_covariance_4x4(self.covariance)
        object.__setattr__(self, "timestamp", float(self.timestamp))


@dataclass(frozen=True)
class Trajectory:
    """Sequence of future predicted trajectory points over a prediction horizon.

    Answers: "What is the projected physical path of the object over time?"
    """

    object_id: int
    points: tuple[TrajectoryPoint, ...]
    prediction_horizon: float

    def __post_init__(self) -> None:
        if not isinstance(self.object_id, int):
            raise TypeError(f"Trajectory.object_id must be int, got {type(self.object_id)}")
        if not isinstance(self.prediction_horizon, (int, float)) or not math.isfinite(self.prediction_horizon):
            raise ValueError(f"Trajectory.prediction_horizon must be a finite float, got {self.prediction_horizon!r}")
        if self.prediction_horizon < 0.0:
            raise ValueError(f"Trajectory.prediction_horizon cannot be negative, got {self.prediction_horizon}")
        
        # Ensure points is an immutable tuple of TrajectoryPoint instances
        if not isinstance(self.points, (tuple, list)):
            raise TypeError(f"Trajectory.points must be a Sequence of TrajectoryPoint, got {type(self.points)}")
        tuple_points = tuple(self.points)
        for i, pt in enumerate(tuple_points):
            if not isinstance(pt, TrajectoryPoint):
                raise TypeError(f"Trajectory point {i} must be a TrajectoryPoint, got {type(pt)}")
        object.__setattr__(self, "points", tuple_points)
        object.__setattr__(self, "prediction_horizon", float(self.prediction_horizon))


@dataclass(frozen=True)
class RiskAssessment:
    """Evaluated collision threat for a tracked object based on its predicted motion.

    Answers: "Does the predicted interaction represent danger to the user?"
    """

    object_id: int
    risk_level: RiskLevel
    risk_score: float
    ttc: float | None = None
    cpa_distance: float | None = None
    confidence: float = 1.0
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.object_id, int):
            raise TypeError(f"RiskAssessment.object_id must be int, got {type(self.object_id)}")
        if not isinstance(self.risk_level, RiskLevel):
            raise TypeError(f"RiskAssessment.risk_level must be a RiskLevel enum, got {type(self.risk_level)}")
        if not isinstance(self.risk_score, (int, float)) or not math.isfinite(self.risk_score):
            raise ValueError(f"RiskAssessment.risk_score must be a finite float, got {self.risk_score!r}")
        if not (0.0 <= self.risk_score <= 100.0):
            raise ValueError(f"RiskAssessment.risk_score must be in [0.0, 100.0], got {self.risk_score}")
        if self.ttc is not None:
            if not isinstance(self.ttc, (int, float)) or not math.isfinite(self.ttc):
                raise ValueError(f"RiskAssessment.ttc must be a finite float or None, got {self.ttc!r}")
            object.__setattr__(self, "ttc", float(self.ttc))
        if self.cpa_distance is not None:
            if not isinstance(self.cpa_distance, (int, float)) or not math.isfinite(self.cpa_distance):
                raise ValueError(f"RiskAssessment.cpa_distance must be a finite float or None, got {self.cpa_distance!r}")
            if self.cpa_distance < 0.0:
                raise ValueError(f"RiskAssessment.cpa_distance cannot be negative, got {self.cpa_distance}")
            object.__setattr__(self, "cpa_distance", float(self.cpa_distance))
        if not isinstance(self.confidence, (int, float)) or not math.isfinite(self.confidence):
            raise ValueError(f"RiskAssessment.confidence must be a finite float, got {self.confidence!r}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"RiskAssessment.confidence must be in [0.0, 1.0], got {self.confidence}")
        if not isinstance(self.timestamp, (int, float)) or not math.isfinite(self.timestamp):
            raise ValueError(f"RiskAssessment.timestamp must be a finite float, got {self.timestamp!r}")
        object.__setattr__(self, "risk_score", float(self.risk_score))
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "timestamp", float(self.timestamp))


@dataclass(frozen=True)
class Decision:
    """Action chosen by the decision engine based on risk assessments.

    Answers: "What action should the system communicate or execute?"
    """

    decision_type: DecisionType
    object_id: int | None = None
    timestamp: float = 0.0
    message_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.decision_type, DecisionType):
            raise TypeError(f"Decision.decision_type must be a DecisionType enum, got {type(self.decision_type)}")
        if self.object_id is not None and not isinstance(self.object_id, int):
            raise TypeError(f"Decision.object_id must be int or None, got {type(self.object_id)}")
        if not isinstance(self.timestamp, (int, float)) or not math.isfinite(self.timestamp):
            raise ValueError(f"Decision.timestamp must be a finite float, got {self.timestamp!r}")
        if self.message_id is not None and not isinstance(self.message_id, str):
            raise TypeError(f"Decision.message_id must be str or None, got {type(self.message_id)}")
        object.__setattr__(self, "timestamp", float(self.timestamp))


@dataclass(frozen=True)
class WorldState:
    """Snapshot of estimated physical states across all tracked objects at a given timestamp.

    Note:
        Objects are stored in a read-only Mapping to preserve immutability.
    """

    timestamp: float
    objects: Mapping[int, ObjectState] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, (int, float)) or not math.isfinite(self.timestamp):
            raise ValueError(f"WorldState.timestamp must be a finite float, got {self.timestamp!r}")
        if not isinstance(self.objects, Mapping):
            raise TypeError(f"WorldState.objects must be a Mapping[int, ObjectState], got {type(self.objects)}")
        # Validate elements and freeze the mapping
        validated_objects: dict[int, ObjectState] = {}
        for k, v in self.objects.items():
            if not isinstance(k, int):
                raise TypeError(f"WorldState object key must be int, got {type(k)}")
            if not isinstance(v, ObjectState):
                raise TypeError(f"WorldState object value for key {k} must be ObjectState, got {type(v)}")
            validated_objects[k] = v
        object.__setattr__(self, "objects", MappingProxyType(validated_objects))
        object.__setattr__(self, "timestamp", float(self.timestamp))

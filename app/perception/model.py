"""Perception domain models: Detection, PhysicalObservation, TrackedObservation, and Rejection diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from app.core.types import BoundingBox, Vector2


class DetectionRejectionReason(str, Enum):
    """Explicit reason explaining why an image-space detection could not become a physical observation."""

    OUT_OF_BOUNDS = "out_of_bounds"
    ABOVE_HORIZON = "above_horizon"
    INVALID_BBOX = "invalid_bbox"
    PROJECTION_FAILED = "projection_failed"
    LOW_CONFIDENCE = "low_confidence"
    INVALID_DEPTH = "invalid_depth"


@dataclass(frozen=True)
class DetectionRejection:
    """Diagnostic record of a rejected detection."""

    detection: Detection
    reason: DetectionRejectionReason
    detail: str


@dataclass(frozen=True)
class Detection:
    """Raw detector output in 2D image space.

    Answers: "What object did the 2D visual detector propose in this frame?"

    Important Architectural Boundary:
        detection_id identifies this specific proposal in the current image frame.
        A 2D detector does NOT perform tracking. detection_id is ephemeral and must
        NEVER be assumed to be a persistent temporal tracking identity (track_id).
    """

    detection_id: int | str
    class_id: int | str
    class_name: str
    confidence: float
    bbox: BoundingBox
    timestamp: float

    def __post_init__(self) -> None:
        if not isinstance(self.detection_id, (int, str)):
            raise TypeError(f"detection_id must be int or str, got {type(self.detection_id)}")
        if not isinstance(self.class_id, (int, str)):
            raise TypeError(f"class_id must be int or str, got {type(self.class_id)}")
        if not isinstance(self.class_name, str):
            raise TypeError(f"class_name must be str, got {type(self.class_name)}")
        if not isinstance(self.confidence, (int, float)) or not math.isfinite(self.confidence):
            raise ValueError(f"confidence must be a finite float, got {self.confidence!r}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")
        if not isinstance(self.bbox, BoundingBox):
            raise TypeError(f"bbox must be BoundingBox, got {type(self.bbox)}")
        if not isinstance(self.timestamp, (int, float)) or not math.isfinite(self.timestamp):
            raise ValueError(f"timestamp must be a finite float, got {self.timestamp!r}")

        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "timestamp", float(self.timestamp))


@dataclass(frozen=True)
class PhysicalObservation:
    """Projected ground-plane observation with 2x2 measurement covariance R.

    Answers: "Where is the object measured on the ground plane, and what is its measurement noise?"

    Note:
        measurement_covariance is the 2x2 measurement noise matrix R:
            [[R_xx, 0],
             [0, R_zz]]
        This is distinct from the 4x4 Kalman state covariance matrix P.
    """

    position: Vector2
    measurement_covariance: tuple[tuple[float, float], tuple[float, float]]
    confidence: float
    timestamp: float
    detection_id: int | str
    class_name: str = "unknown"
    bbox: BoundingBox | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.position, Vector2):
            raise TypeError(f"position must be Vector2, got {type(self.position)}")
        if not isinstance(self.confidence, (int, float)) or not math.isfinite(self.confidence):
            raise ValueError(f"confidence must be a finite float, got {self.confidence!r}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")
        if not isinstance(self.timestamp, (int, float)) or not math.isfinite(self.timestamp):
            raise ValueError(f"timestamp must be a finite float, got {self.timestamp!r}")

        # Validate 2x2 measurement covariance matrix R
        cov = self.measurement_covariance
        if not isinstance(cov, (tuple, list)) or len(cov) != 2:
            raise ValueError(f"measurement_covariance must be a 2x2 matrix, got {cov}")
        for row in cov:
            if not isinstance(row, (tuple, list)) or len(row) != 2:
                raise ValueError(f"measurement_covariance rows must have length 2, got {row}")
            for elem in row:
                if not isinstance(elem, (int, float)) or not math.isfinite(elem):
                    raise ValueError(f"measurement_covariance elements must be finite floats, got {elem}")

        r00, r01 = float(cov[0][0]), float(cov[0][1])
        r10, r11 = float(cov[1][0]), float(cov[1][1])

        if r00 <= 0.0 or r11 <= 0.0:
            raise ValueError(f"measurement_covariance diagonal elements must be strictly positive, got R00={r00}, R11={r11}")
        if abs(r01 - r10) > 1e-6:
            raise ValueError(f"measurement_covariance must be symmetric, got R01={r01} != R10={r10}")

        object.__setattr__(self, "measurement_covariance", ((r00, r01), (r10, r11)))
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "timestamp", float(self.timestamp))


@dataclass(frozen=True)
class TrackedObservation:
    """An observation explicitly linked to a persistent temporal tracking identity (track_id).

    Answers: "This physical observation belongs to active track track_id."

    Architectural Boundary:
        TrackedObservation is the bridge produced by data association/tracking
        and consumed by the Part 4 Kalman State Estimator.
    """

    track_id: int
    observation: PhysicalObservation

    def __post_init__(self) -> None:
        if not isinstance(self.track_id, int):
            raise TypeError(f"track_id must be int, got {type(self.track_id)}")
        if not isinstance(self.observation, PhysicalObservation):
            raise TypeError(f"observation must be PhysicalObservation, got {type(self.observation)}")

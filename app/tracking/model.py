"""Tracking domain models: TrackStatus, TrackRecord, AssociationPair, and TrackingResult."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence

from app.core.types import Vector2
from app.perception.model import PhysicalObservation, TrackedObservation


class TrackStatus(str, Enum):
    """Lifecycle status of a multi-object track."""

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    DELETED = "deleted"


@dataclass(frozen=True)
class TrackRecord:
    """Represents a persistent tracking identity and temporal lifecycle metadata.

    Answers: "What is the identity, continuity, and lifecycle status of this tracked physical entity?"

    Important Architectural Principle:
        TrackRecord manages identity and lifecycle metadata only.
        Kinematic state estimation [X, Z, Vx, Vz] and covariance P remain owned
        by the Part 4 Kalman State Estimator.
    """

    track_id: int
    class_name: str
    status: TrackStatus
    first_timestamp: float
    last_timestamp: float
    latest_observation: PhysicalObservation | None = None
    age: int = 1
    hits: int = 1
    consecutive_hits: int = 1
    missed_frames: int = 0
    time_since_update: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.track_id, int):
            raise TypeError(f"track_id must be int, got {type(self.track_id)}")
        if not isinstance(self.class_name, str):
            raise TypeError(f"class_name must be str, got {type(self.class_name)}")
        if not isinstance(self.status, TrackStatus):
            raise TypeError(f"status must be TrackStatus enum, got {type(self.status)}")
        for name, val in [
            ("first_timestamp", self.first_timestamp),
            ("last_timestamp", self.last_timestamp),
            ("time_since_update", self.time_since_update),
        ]:
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"{name} must be a finite float, got {val!r}")
        for name, val in [
            ("age", self.age),
            ("hits", self.hits),
            ("consecutive_hits", self.consecutive_hits),
            ("missed_frames", self.missed_frames),
        ]:
            if not isinstance(val, int) or val < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {val!r}")

        object.__setattr__(self, "first_timestamp", float(self.first_timestamp))
        object.__setattr__(self, "last_timestamp", float(self.last_timestamp))
        object.__setattr__(self, "time_since_update", float(self.time_since_update))

    # Compatibility properties
    @property
    def created_at(self) -> float:
        return self.first_timestamp

    @property
    def last_update(self) -> float:
        return self.last_timestamp

    @property
    def last_observation(self) -> PhysicalObservation | None:
        return self.latest_observation


# Alias Track to TrackRecord for complete compatibility
Track = TrackRecord


@dataclass(frozen=True)
class AssociationPair:
    """Represents a matched observation-to-track pair."""

    track_id: int
    observation_id: int | str
    cost: float
    distance: float
    class_compatible: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.track_id, int):
            raise TypeError(f"track_id must be int, got {type(self.track_id)}")
        if not isinstance(self.observation_id, (int, str)):
            raise TypeError(f"observation_id must be int or str, got {type(self.observation_id)}")
        for name, val in [("cost", self.cost), ("distance", self.distance)]:
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"{name} must be a finite float, got {val!r}")
        if not isinstance(self.class_compatible, bool):
            raise TypeError(f"class_compatible must be bool, got {type(self.class_compatible)}")

        object.__setattr__(self, "cost", float(self.cost))
        object.__setattr__(self, "distance", float(self.distance))


# Alias Association to AssociationPair
Association = AssociationPair


@dataclass(frozen=True)
class TrackingResult:
    """Frame tracking output containing matched associations, active tracks, and lifecycle diagnostics."""

    confirmed_tracks: tuple[TrackRecord, ...]
    new_tracks: tuple[TrackRecord, ...]
    lost_tracks: tuple[TrackRecord, ...]
    deleted_tracks: tuple[TrackRecord, ...]
    associations: tuple[AssociationPair, ...]
    unmatched_observations: tuple[PhysicalObservation, ...]
    tracked_observations: tuple[TrackedObservation, ...]
    active_tracks: tuple[TrackRecord, ...]
    timestamp: float

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, (int, float)) or not math.isfinite(self.timestamp):
            raise ValueError(f"timestamp must be a finite float, got {self.timestamp!r}")
        object.__setattr__(self, "confirmed_tracks", tuple(self.confirmed_tracks))
        object.__setattr__(self, "new_tracks", tuple(self.new_tracks))
        object.__setattr__(self, "lost_tracks", tuple(self.lost_tracks))
        object.__setattr__(self, "deleted_tracks", tuple(self.deleted_tracks))
        object.__setattr__(self, "associations", tuple(self.associations))
        object.__setattr__(self, "unmatched_observations", tuple(self.unmatched_observations))
        object.__setattr__(self, "tracked_observations", tuple(self.tracked_observations))
        object.__setattr__(self, "active_tracks", tuple(self.active_tracks))
        object.__setattr__(self, "timestamp", float(self.timestamp))

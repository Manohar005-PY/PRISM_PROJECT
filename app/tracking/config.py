"""Configuration parameters for multi-object tracking and association."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TrackingConfig:
    """Configuration parameters for multi-object tracking and observation gating.

    Attributes:
        gating_threshold: Maximum Mahalanobis distance squared (d_M^2) for spatial gating.
                          Default 9.21 corresponds to Chi-squared 2-DOF at p=0.01 (99% confidence region).
        max_missed_frames: Maximum number of consecutive missed frames before deleting a confirmed track.
        confirmation_hits: Number of consecutive successful associations to promote TENTATIVE to CONFIRMED.
        tentative_max_missed: Maximum missed frames for a TENTATIVE track before early deletion.
        max_assignment_distance: Maximum physical Euclidean distance (meters) sanity gate.
        class_aware: If True, restricts association to matching object classes.
        allow_class_change: If True, allows confirmed tracks to update class on high-confidence re-association.
    """

    gating_threshold: float = 9.21
    max_missed_frames: int = 5
    confirmation_hits: int = 3
    tentative_max_missed: int = 1
    max_assignment_distance: float = 8.0
    class_aware: bool = True
    allow_class_change: bool = False

    def __post_init__(self) -> None:
        for name, val in [
            ("gating_threshold", self.gating_threshold),
            ("max_assignment_distance", self.max_assignment_distance),
        ]:
            if not isinstance(val, (int, float)) or not math.isfinite(val) or val <= 0.0:
                raise ValueError(f"{name} must be a positive finite float, got {val!r}")

        for name, val in [
            ("max_missed_frames", self.max_missed_frames),
            ("confirmation_hits", self.confirmation_hits),
            ("tentative_max_missed", self.tentative_max_missed),
        ]:
            if not isinstance(val, int) or val <= 0:
                raise ValueError(f"{name} must be a positive integer, got {val!r}")

        if not isinstance(self.class_aware, bool):
            raise TypeError(f"class_aware must be bool, got {type(self.class_aware)}")
        if not isinstance(self.allow_class_change, bool):
            raise TypeError(f"allow_class_change must be bool, got {type(self.allow_class_change)}")

        object.__setattr__(self, "gating_threshold", float(self.gating_threshold))
        object.__setattr__(self, "max_assignment_distance", float(self.max_assignment_distance))

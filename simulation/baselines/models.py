"""Data contracts and configuration models for baseline navigation strategies."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any


@dataclass(frozen=True)
class ReactiveBaselineConfig:
    """Centralized, immutable configuration for the reactive distance baseline."""

    name: str = "reactive_distance_baseline"
    version: str = "18.0.0"
    warning_distance_m: float = 3.0
    avoidance_distance_m: float = 1.5
    stop_distance_m: float = 0.8
    lateral_turn_angle_rad: float = 0.35  # ~20 degrees
    hysteresis_frames: int = 2
    description: str = (
        "Pure reactive Euclidean proximity baseline mapping instantaneous obstacle distance "
        "to deterministic navigation decisions without trajectory prediction, TTC, or Kalman filtering."
    )

    def __post_init__(self) -> None:
        if not (self.stop_distance_m < self.avoidance_distance_m < self.warning_distance_m):
            raise ValueError(
                f"Threshold hierarchy must be stop < avoid < warn, got "
                f"{self.stop_distance_m} < {self.avoidance_distance_m} < {self.warning_distance_m}"
            )
        if self.hysteresis_frames < 0:
            raise ValueError(f"hysteresis_frames must be >= 0, got {self.hysteresis_frames}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReactiveBaselineDecision:
    """Deterministic output emitted by the reactive distance baseline at timestep t."""

    timestamp: float
    frame_index: int
    action: str  # "CONTINUE", "SLOW_DOWN", "TURN_LEFT", "TURN_RIGHT", "STOP"
    warning_level: str  # "SAFE", "HIGH", "CRITICAL"
    recommended_heading_rad: float
    closest_obstacle_id: int | str | None
    min_observed_distance_m: float | None
    is_warning: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

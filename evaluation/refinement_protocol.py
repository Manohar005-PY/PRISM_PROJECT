"""Refinement protocol: Explicit separation of development/tuning datasets and holdout test partitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

from evaluation.scenarios import (
    create_crossing_scenario,
    create_departing_scenario,
    create_diverging_scenario,
    create_head_on_fast_scenario,
    create_head_on_scenario,
    create_head_on_slow_scenario,
    create_multi_object_scenario,
    create_near_miss_scenario,
    create_parallel_scenario,
    create_stationary_obstacle_scenario,
)
from simulation.scenarios import Scenario


@dataclass(frozen=True)
class RefinementConfig:
    """Explicit parameters for refined PRISM risk assessment and decision arbitration."""

    horizon: float = 3.5
    max_risk_distance: float = 5.0
    ref_closing_speed: float = 2.0
    k_sigma: float = 1.0

    weight_time: float = 0.45
    weight_dist: float = 0.25
    weight_speed: float = 0.15
    weight_dir: float = 0.15

    thresh_low: float = 0.15
    thresh_medium: float = 0.35
    thresh_high: float = 0.65
    thresh_critical: float = 0.85

    alpha_smooth: float = 0.30

    def __post_init__(self) -> None:
        total_w = self.weight_time + self.weight_dist + self.weight_speed + self.weight_dir
        if abs(total_w - 1.0) > 1e-4:
            raise ValueError(f"Weights must sum to 1.0, got {total_w}")
        if not (0.0 < self.thresh_low < self.thresh_medium < self.thresh_high < self.thresh_critical <= 1.0):
            raise ValueError("Thresholds must be strictly monotonic in (0.0, 1.0]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_development_scenario_partition() -> dict[str, Scenario]:
    """Return development scenarios used strictly for failure analysis and parameter tuning."""
    return {
        "HEAD_ON": create_head_on_scenario(),
        "CROSSING": create_crossing_scenario(),
        "STATIONARY_OBSTACLE": create_stationary_obstacle_scenario(),
    }


def get_holdout_scenario_partition() -> dict[str, Scenario]:
    """Return strictly held-out evaluation scenarios evaluated only once for validation."""
    return {
        "HEAD_ON_FAST": create_head_on_fast_scenario(),
        "HEAD_ON_SLOW": create_head_on_slow_scenario(),
        "NEAR_MISS": create_near_miss_scenario(),
        "DEPARTING_OBJECT": create_departing_scenario(),
        "MULTI_OBJECT": create_multi_object_scenario(),
        "PARALLEL": create_parallel_scenario(),
        "DIVERGING": create_diverging_scenario(),
    }

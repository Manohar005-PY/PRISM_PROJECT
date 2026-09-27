"""Data models, configuration contracts, and result structures for PRISM Part 19 Ablation & Sensitivity Analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


class AblationVariant(str, Enum):
    """Architectural ablation variant identifiers for controlled comparison."""

    FULL_PRISM = "FULL_PRISM"
    NO_KALMAN = "NO_KALMAN"
    POSITION_ONLY = "POSITION_ONLY"
    TTC_ONLY = "TTC_ONLY"
    DCPA_ONLY = "DCPA_ONLY"
    NO_PREDICTIVE_RISK = "NO_PREDICTIVE_RISK"


@dataclass(frozen=True)
class AblationConfig:
    """Configuration container for Part 19 Ablation & Sensitivity Analysis."""

    dataset_dir: str = "experiments/datasets/synthetic"
    output_dir: str = "experiments/ablation/part19"
    prediction_horizon: float = 3.0
    min_safe_clearance_m: float = 0.50
    lead_time_late_threshold_s: float = 0.50
    decision_instability_threshold: int = 5
    steering_reversal_threshold: int = 2

    def __post_init__(self) -> None:
        if self.prediction_horizon <= 0.0:
            raise ValueError(f"prediction_horizon must be positive, got {self.prediction_horizon}")
        if self.min_safe_clearance_m <= 0.0:
            raise ValueError(f"min_safe_clearance_m must be positive, got {self.min_safe_clearance_m}")


@dataclass(frozen=True)
class SensitivityParamSpec:
    """Specification of a single parameter sweep for sensitivity analysis on FULL_PRISM."""

    param_name: str
    target_component: str
    nominal_value: float
    sweep_values: tuple[float, ...]
    unit: str
    description: str


# Pre-declared, frozen sensitivity parameter specifications
FROZEN_SENSITIVITY_SPECS: tuple[SensitivityParamSpec, ...] = (
    SensitivityParamSpec(
        param_name="horizon",
        target_component="RiskConfig",
        nominal_value=3.0,
        sweep_values=(2.0, 3.0, 4.0),
        unit="s",
        description="Predictive risk evaluation lookahead time horizon.",
    ),
    SensitivityParamSpec(
        param_name="thresh_high",
        target_component="RiskConfig",
        nominal_value=0.70,
        sweep_values=(0.60, 0.70, 0.80),
        unit="score",
        description="Threshold for escalating to HIGH risk level and initiating active warning/avoidance.",
    ),
    SensitivityParamSpec(
        param_name="alpha_smooth",
        target_component="RiskConfig",
        nominal_value=0.60,
        sweep_values=(0.30, 0.60, 0.90),
        unit="coefficient",
        description="Exponential moving average smoothing factor for continuous risk score.",
    ),
    SensitivityParamSpec(
        param_name="k_sigma",
        target_component="RiskConfig",
        nominal_value=1.0,
        sweep_values=(0.50, 1.00, 2.00),
        unit="multiplier",
        description="Positional uncertainty covariance expansion multiplier for physical safety boundary.",
    ),
    SensitivityParamSpec(
        param_name="sigma_a",
        target_component="KalmanFilterConfig",
        nominal_value=0.50,
        sweep_values=(0.25, 0.50, 1.00),
        unit="m/s^2",
        description="Standard deviation of process acceleration noise in Kalman state filter.",
    ),
)


@dataclass
class VariantFrameOutput:
    """Lightweight frame-level output recorded during variant execution."""

    timestamp: float
    frame_index: int
    action: str
    reason: str
    risk_level: str
    risk_score: float
    predicted_ttc: float | None
    predicted_cpa: float | None
    recommended_heading_rad: float | None
    active_track_count: int
    latency_ms: float


@dataclass
class VariantRunResult:
    """Run-level summary metrics for a single variant on a single experimental scenario run."""

    scenario_id: str
    scenario_family: str
    run_id: str
    variant_id: str
    num_frames: int
    duration: float
    noise_condition: str
    has_collision: bool
    collision_time_s: float | None
    first_warning_time_s: float | None
    warning_lead_time_s: float | None
    warned: bool
    false_warning: bool
    warning_maintained_ratio: float | None
    decision_changes: int
    steering_reversals: int
    min_clearance_m: float
    safe_clearance_ratio: float
    mean_latency_ms: float
    frame_outputs: list[VariantFrameOutput] = field(default_factory=list)

    def to_dict(self, include_frames: bool = False) -> dict[str, Any]:
        """Convert run result to JSON-serializable dictionary."""
        d: dict[str, Any] = {
            "scenario_id": self.scenario_id,
            "scenario_family": self.scenario_family,
            "run_id": self.run_id,
            "variant_id": self.variant_id,
            "num_frames": self.num_frames,
            "duration": self.duration,
            "noise_condition": self.noise_condition,
            "has_collision": self.has_collision,
            "collision_time_s": self.collision_time_s,
            "first_warning_time_s": self.first_warning_time_s,
            "warning_lead_time_s": self.warning_lead_time_s,
            "warned": self.warned,
            "false_warning": self.false_warning,
            "warning_maintained_ratio": self.warning_maintained_ratio,
            "decision_changes": self.decision_changes,
            "steering_reversals": self.steering_reversals,
            "min_clearance_m": self.min_clearance_m,
            "safe_clearance_ratio": self.safe_clearance_ratio,
            "mean_latency_ms": self.mean_latency_ms,
        }
        if include_frames:
            d["frame_outputs"] = [
                {
                    "timestamp": f.timestamp,
                    "frame_index": f.frame_index,
                    "action": f.action,
                    "reason": f.reason,
                    "risk_level": f.risk_level,
                    "risk_score": f.risk_score,
                    "predicted_ttc": f.predicted_ttc,
                    "predicted_cpa": f.predicted_cpa,
                    "recommended_heading_rad": f.recommended_heading_rad,
                    "active_track_count": f.active_track_count,
                    "latency_ms": f.latency_ms,
                }
                for f in self.frame_outputs
            ]
        return d


@dataclass
class PairwiseVariantComparison:
    """Paired comparative metric differences between FULL_PRISM and an ablated variant for one run."""

    scenario_id: str
    scenario_family: str
    run_id: str
    variant_id: str
    noise_condition: str
    has_collision: bool
    full_lead_time_s: float | None
    variant_lead_time_s: float | None
    delta_lead_time_s: float | None  # Full - Variant (positive means Full warns earlier)
    full_min_clearance_m: float
    variant_min_clearance_m: float
    delta_min_clearance_m: float  # Full - Variant
    full_decision_changes: int
    variant_decision_changes: int
    delta_decision_changes: int  # Variant - Full (positive means Variant has more chatter)
    full_steering_reversals: int
    variant_steering_reversals: int
    delta_steering_reversals: int  # Variant - Full
    full_warned: bool
    variant_warned: bool
    full_false_warning: bool
    variant_false_warning: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "scenario_family": self.scenario_family,
            "run_id": self.run_id,
            "variant_id": self.variant_id,
            "noise_condition": self.noise_condition,
            "has_collision": self.has_collision,
            "full_lead_time_s": self.full_lead_time_s,
            "variant_lead_time_s": self.variant_lead_time_s,
            "delta_lead_time_s": self.delta_lead_time_s,
            "full_min_clearance_m": self.full_min_clearance_m,
            "variant_min_clearance_m": self.variant_min_clearance_m,
            "delta_min_clearance_m": self.delta_min_clearance_m,
            "full_decision_changes": self.full_decision_changes,
            "variant_decision_changes": self.variant_decision_changes,
            "delta_decision_changes": self.delta_decision_changes,
            "full_steering_reversals": self.full_steering_reversals,
            "variant_steering_reversals": self.variant_steering_reversals,
            "delta_steering_reversals": self.delta_steering_reversals,
            "full_warned": self.full_warned,
            "variant_warned": self.variant_warned,
            "full_false_warning": self.full_false_warning,
            "variant_false_warning": self.variant_false_warning,
        }


@dataclass
class AblationFailureCase:
    """Identified comparative failure instance for an ablated variant."""

    scenario_id: str
    scenario_family: str
    run_id: str
    variant_id: str
    failure_type: str  # LATE_WARNING, MISSED_WARNING, FALSE_WARNING, UNSTABLE_STEERING, CLEARANCE_VIOLATION
    description: str
    timestamp_s: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "scenario_family": self.scenario_family,
            "run_id": self.run_id,
            "variant_id": self.variant_id,
            "failure_type": self.failure_type,
            "description": self.description,
            "timestamp_s": self.timestamp_s,
            "details": self.details,
        }


@dataclass
class SensitivitySettingResult:
    """Aggregated evaluation metrics for one parameter setting on FULL_PRISM."""

    param_name: str
    setting_value: float
    total_runs: int
    collision_runs: int
    safe_runs: int
    mean_lead_time_s: float
    median_lead_time_s: float
    warning_rate: float
    false_warning_rate: float
    mean_decision_changes: float
    mean_steering_reversals: float
    mean_min_clearance_m: float
    safe_clearance_rate: float
    mean_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "param_name": self.param_name,
            "setting_value": self.setting_value,
            "total_runs": self.total_runs,
            "collision_runs": self.collision_runs,
            "safe_runs": self.safe_runs,
            "mean_lead_time_s": round(self.mean_lead_time_s, 4),
            "median_lead_time_s": round(self.median_lead_time_s, 4),
            "warning_rate": round(self.warning_rate, 4),
            "false_warning_rate": round(self.false_warning_rate, 4),
            "mean_decision_changes": round(self.mean_decision_changes, 2),
            "mean_steering_reversals": round(self.mean_steering_reversals, 2),
            "mean_min_clearance_m": round(self.mean_min_clearance_m, 4),
            "safe_clearance_rate": round(self.safe_clearance_rate, 4),
            "mean_latency_ms": round(self.mean_latency_ms, 3),
        }

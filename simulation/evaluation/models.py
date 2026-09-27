"""Data models and evaluation contracts for PRISM Part 17."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Sequence


@dataclass(frozen=True)
class MetricStats:
    """Statistical summary for continuous evaluation metrics."""

    n: int
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    p25: float | None = None
    p75: float | None = None
    p95: float | None = None
    mae: float | None = None
    rmse: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClassificationStats:
    """Binary or multi-class confusion matrix and detection metrics."""

    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    accuracy: float | None = None
    confusion_matrix: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RiskCalibrationBin:
    """Empirical event frequency evaluated within a risk score interval."""

    bin_label: str
    min_score: float
    max_score: float
    sample_count: int
    collision_event_count: int
    empirical_collision_frequency: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration governing evaluation horizons, analysis thresholds, and tolerances."""

    dataset_dir: str = "experiments/datasets/synthetic"
    output_dir: str = "experiments/evaluation/results"
    spatial_gating_threshold_m: float = 1.5
    prediction_horizons: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0)
    event_horizons: tuple[float, ...] = (1.0, 2.0, 3.0)
    risk_bins: tuple[tuple[float, float], ...] = (
        (0.0, 20.0),
        (20.0, 40.0),
        (40.0, 60.0),
        (60.0, 80.0),
        (80.0, 100.0),
    )
    diagnostic_pred_error_threshold_m: float = 1.5
    diagnostic_ttc_error_threshold_s: float = 2.0
    min_safe_clearance_m: float = 0.50
    version: str = "17.0.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FailureCase:
    """Structured diagnostic record of a detected anomaly or failure."""

    failure_type: str
    scenario_id: str
    scenario_family: str
    run_id: str
    timestamp: float
    frame_index: int
    object_id: int | str | None
    track_id: int | None
    metric_name: str
    metric_value: float | str | None
    ground_truth_summary: dict[str, Any]
    prism_output_summary: dict[str, Any]
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunEvaluationResult:
    """Complete evaluation metrics and diagnostics for a single experimental run."""

    scenario_id: str
    scenario_family: str
    run_id: str
    num_frames: int
    duration: float
    noise_condition: str
    state_estimation: dict[str, Any]
    tracking: dict[str, Any]
    prediction: dict[str, Any]
    ttc_dcpa: dict[str, Any]
    risk: dict[str, Any]
    warning: dict[str, Any]
    avoidance: dict[str, Any]
    failures: list[FailureCase] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["failures"] = [f.to_dict() for f in self.failures]
        return d


@dataclass(frozen=True)
class AggregateEvaluationReport:
    """Dataset-level aggregated metrics, macro/micro summaries, and noise degradation report."""

    dataset_name: str
    evaluation_version: str
    created_at: str
    total_runs_evaluated: int
    total_frames_evaluated: int
    scenario_families_evaluated: list[str]
    noise_conditions_evaluated: list[str]
    overall_state_estimation: dict[str, Any]
    overall_tracking: dict[str, Any]
    overall_prediction: dict[str, Any]
    overall_ttc_dcpa: dict[str, Any]
    overall_risk: dict[str, Any]
    overall_warning: dict[str, Any]
    overall_avoidance: dict[str, Any]
    scenario_breakdown: dict[str, Any]
    noise_breakdown: dict[str, Any]
    risk_calibration_bins: list[dict[str, Any]]
    total_failures_identified: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

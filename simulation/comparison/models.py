"""Data contracts and schemas for PRISM vs. Baseline comparative evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Sequence

from simulation.baselines.models import ReactiveBaselineConfig
from simulation.evaluation.models import MetricStats


@dataclass(frozen=True)
class ComparisonConfig:
    """Configuration governing the Part 18 comparative experiment."""

    dataset_dir: str = "experiments/datasets/synthetic"
    output_dir: str = "experiments/comparison/part18"
    baseline_config: ReactiveBaselineConfig = field(default_factory=ReactiveBaselineConfig)
    version: str = "18.0.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_dir": self.dataset_dir,
            "output_dir": self.output_dir,
            "baseline_config": self.baseline_config.to_dict(),
            "version": self.version,
        }


@dataclass(frozen=True)
class ComparativeWarningStats:
    """Warning detection and lead time metrics compared between PRISM and Baseline."""

    has_collision: bool
    prism_warned: bool
    baseline_warned: bool
    prism_first_warning_time_s: float | None
    baseline_first_warning_time_s: float | None
    collision_time_s: float | None
    prism_lead_time_s: float | None
    baseline_lead_time_s: float | None
    delta_lead_time_s: float | None  # PRISM lead time - Baseline lead time
    earlier_warning_system: str | None  # "PRISM", "BASELINE", "EQUAL", "NONE"
    prism_maintained_ratio: float | None
    baseline_maintained_ratio: float | None
    prism_false_warning: bool
    baseline_false_warning: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComparativeAvoidanceStats:
    """Action safety, stability, and clearance comparison for a run."""

    prism_decision_changes: int
    baseline_decision_changes: int
    prism_steering_reversals: int
    baseline_steering_reversals: int
    prism_min_clearance_m: float
    baseline_min_clearance_m: float
    prism_safe_clearance_ratio: float
    baseline_safe_clearance_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComparativeFailureCase:
    """Structured diagnostic record of a comparative discrepancy or system failure."""

    failure_type: str  # "BASELINE_MISSED_WARNING", "BASELINE_LATE_WARNING", "PRISM_FALSE_WARNING", "BASELINE_FALSE_WARNING"
    scenario_id: str
    scenario_family: str
    run_id: str
    timestamp: float
    frame_index: int
    metric_name: str
    prism_value: Any
    baseline_value: Any
    ground_truth_summary: dict[str, Any]
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PairwiseRunComparison:
    """Complete comparative evaluation results for a single experimental run."""

    scenario_id: str
    scenario_family: str
    run_id: str
    num_frames: int
    duration: float
    noise_condition: str
    warning_comparison: ComparativeWarningStats
    avoidance_comparison: ComparativeAvoidanceStats
    failures: list[ComparativeFailureCase] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["warning_comparison"] = self.warning_comparison.to_dict()
        d["avoidance_comparison"] = self.avoidance_comparison.to_dict()
        d["failures"] = [f.to_dict() for f in self.failures]
        return d


@dataclass(frozen=True)
class PredictiveAdvantageStats:
    """Dataset-level statistical summary quantifying PRISM's predictive advantage."""

    total_collision_runs: int
    paired_lead_time_samples: int  # Runs where both PRISM and Baseline warned
    mean_delta_lead_time_s: float | None
    median_delta_lead_time_s: float | None
    std_delta_lead_time_s: float | None
    min_delta_lead_time_s: float | None
    max_delta_lead_time_s: float | None
    earlier_warning_count: int  # Count of collision runs where PRISM warned earlier than Baseline
    earlier_warning_rate: float | None  # Earlier warnings / total collision runs
    prism_missed_events: int
    baseline_missed_events: int
    prism_mean_lead_time_s: float | None
    baseline_mean_lead_time_s: float | None
    relative_lead_time_improvement_pct: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComparativeAggregateReport:
    """Master dataset-level aggregate report comparing PRISM against the Baseline."""

    dataset_name: str
    comparison_version: str
    created_at: str
    total_runs_evaluated: int
    total_collision_runs: int
    total_safe_runs: int
    predictive_advantage: PredictiveAdvantageStats
    overall_warning_summary: dict[str, Any]
    overall_avoidance_summary: dict[str, Any]
    scenario_breakdown: dict[str, Any]
    noise_breakdown: dict[str, Any]
    total_comparative_failures: int

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["predictive_advantage"] = self.predictive_advantage.to_dict()
        return d

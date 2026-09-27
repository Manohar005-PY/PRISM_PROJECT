"""Experiment data model: Preserves raw trial-level measurements, explicit numerators/denominators, and aggregated statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from evaluation.models import EvaluationSample
from evaluation.protocol import ExperimentProtocol
from evaluation.statistics import StatisticalSummary


@dataclass(frozen=True)
class MetricResult:
    """Explicit metric measurement preserving mathematical numerator, denominator, and units."""

    metric_name: str
    numerator: float
    denominator: float
    value: float
    unit: str

    def __post_init__(self) -> None:
        if not isinstance(self.metric_name, str):
            raise TypeError(f"metric_name must be str, got {type(self.metric_name)}")
        if not isinstance(self.value, (int, float)):
            raise TypeError(f"value must be float, got {type(self.value)}")


@dataclass(frozen=True)
class TrialResult:
    """Individual trial execution record within a scenario."""

    scenario_id: str
    trial_index: int
    seed: int
    duration_seconds: float
    sample_count: int
    failure_count: int
    metrics: Mapping[str, MetricResult]
    samples: tuple[EvaluationSample, ...]


@dataclass(frozen=True)
class ExperimentRun:
    """Complete experimental benchmark run containing all trial results and aggregate statistics."""

    protocol: ExperimentProtocol
    system_name: str
    trial_results: tuple[TrialResult, ...]
    aggregated_metrics: Mapping[str, StatisticalSummary]

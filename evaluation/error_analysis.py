"""Error analysis: Multi-objective Pareto trade-off evaluation across safety and stability metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from evaluation.metrics import (
    compute_decision_switch_rate,
    compute_false_alarm_rate,
    compute_hazard_detection_rate,
    compute_position_rmse,
    compute_unnecessary_stop_rate,
)
from evaluation.models import EvaluationSample


@dataclass(frozen=True)
class ParetoTradeoffProfile:
    """Multi-objective performance profile balancing safety, false-alarms, and stability."""

    system_name: str
    hazard_detection_rate: float
    false_alarm_rate: float
    unnecessary_stop_rate: float
    decision_switch_rate: float
    position_rmse: float
    mean_latency_ms: float


class MultiObjectiveErrorAnalyzer:
    """Evaluates multi-objective performance trade-offs without reducing to a single arbitrary scalar."""

    @staticmethod
    def evaluate_profile(
        system_name: str,
        samples: Sequence[EvaluationSample],
    ) -> ParetoTradeoffProfile:
        """Calculate comprehensive Pareto metric profile for an evaluation dataset."""
        if not samples:
            return ParetoTradeoffProfile(system_name, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        det = compute_hazard_detection_rate(samples)
        fa = compute_false_alarm_rate(samples)
        unnec_stop = compute_unnecessary_stop_rate(samples)
        sw = compute_decision_switch_rate(samples)
        rmse = compute_position_rmse(samples)
        mean_lat = sum(s.processing_latency_ms for s in samples) / len(samples)

        return ParetoTradeoffProfile(
            system_name=system_name,
            hazard_detection_rate=det,
            false_alarm_rate=fa,
            unnecessary_stop_rate=unnec_stop,
            decision_switch_rate=sw,
            position_rmse=rmse,
            mean_latency_ms=mean_lat,
        )

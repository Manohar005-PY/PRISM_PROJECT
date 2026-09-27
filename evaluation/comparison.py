"""System comparison: Side-by-side quantitative benchmarking comparing Baseline PRISM, Refined PRISM, and Baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from evaluation.baselines import DistanceBaseline, PRISMEvaluator, SystemEvaluator, TTCBaseline
from evaluation.metrics import (
    compute_decision_switch_rate,
    compute_false_alarm_rate,
    compute_hazard_detection_rate,
    compute_position_rmse,
    compute_unnecessary_stop_rate,
)
from evaluation.refinement import create_refined_prism_evaluator
from evaluation.refinement_protocol import RefinementConfig
from evaluation.runner import EvaluationRunner, create_prism_evaluator_system
from simulation.scenarios import Scenario


@dataclass(frozen=True)
class MetricComparison:
    """Pairwise metric difference comparison."""

    metric_name: str
    baseline_val: float
    refined_val: float
    absolute_change: float
    relative_change_pct: float


class SystemComparator:
    """Computes rigorous pairwise and multi-system comparative benchmark profiles."""

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed
        self.runner = EvaluationRunner(random_seed=random_seed)

    def compare_systems(
        self,
        scenarios: Mapping[str, Scenario],
        systems: Mapping[str, SystemEvaluator] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Run all systems over target scenarios and report normalized aggregate metrics."""
        sys_dict = systems or {
            "Distance Baseline": DistanceBaseline(stop_threshold=1.8, caution_threshold=3.0),
            "TTC Baseline": TTCBaseline(ttc_threshold=2.5),
            "Baseline PRISM": create_prism_evaluator_system(),
            "Refined PRISM": create_refined_prism_evaluator(RefinementConfig(), self.random_seed),
        }

        results: dict[str, dict[str, float]] = {}
        for name, evaluator in sys_dict.items():
            all_samples = []
            for sc in scenarios.values():
                samples = self.runner.run_scenario(sc, evaluator)
                all_samples.extend(samples)

            results[name] = {
                "hazard_detection_rate": compute_hazard_detection_rate(all_samples),
                "false_alarm_rate": compute_false_alarm_rate(all_samples),
                "unnecessary_stop_rate": compute_unnecessary_stop_rate(all_samples),
                "switch_rate": compute_decision_switch_rate(all_samples),
                "position_rmse": compute_position_rmse(all_samples),
            }

        return results

    @staticmethod
    def compute_pairwise_differences(
        baseline_metrics: Mapping[str, float],
        refined_metrics: Mapping[str, float],
    ) -> list[MetricComparison]:
        """Compute absolute and percentage improvements between baseline and refined configurations."""
        comparisons: list[MetricComparison] = []
        for k, b_val in baseline_metrics.items():
            r_val = refined_metrics.get(k, 0.0)
            diff = r_val - b_val
            rel_pct = (diff / max(1e-4, abs(b_val))) * 100.0 if abs(b_val) > 1e-4 else 0.0
            comparisons.append(
                MetricComparison(
                    metric_name=k,
                    baseline_val=b_val,
                    refined_val=r_val,
                    absolute_change=diff,
                    relative_change_pct=rel_pct,
                )
            )
        return comparisons

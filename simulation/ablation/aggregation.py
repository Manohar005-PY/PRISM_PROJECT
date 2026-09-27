"""Statistical aggregation and summary reporting for PRISM Part 19 Ablation & Sensitivity Analysis."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from simulation.ablation.models import (
    PairwiseVariantComparison,
    VariantRunResult,
)


def compute_mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def compute_median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 1:
        return sorted_vals[mid]
    return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0


def compute_std(values: Sequence[float], ddof: int = 1) -> float:
    n = len(values)
    if n <= ddof:
        return 0.0
    mean = compute_mean(values)
    var = sum((x - mean) ** 2 for x in values) / (n - ddof)
    return math.sqrt(max(0.0, var))


def compute_cohens_d(paired_diffs: Sequence[float]) -> float:
    """Calculate Cohen's d effect size for paired samples: d = mean(diff) / std(diff)."""
    if len(paired_diffs) < 2:
        return 0.0
    m = compute_mean(paired_diffs)
    s = compute_std(paired_diffs, ddof=1)
    if s < 1e-9:
        return 0.0
    return m / s


class AblationAggregator:
    """Aggregates ablation and comparative experimental results across scenarios, noise, and variants."""

    def __init__(self) -> None:
        pass

    def compile_variant_summary(self, results: Sequence[VariantRunResult]) -> dict[str, Any]:
        """Compile aggregate performance summary for a single variant across multiple runs."""
        total_runs = len(results)
        collision_runs = [r for r in results if r.has_collision]
        safe_runs = [r for r in results if not r.has_collision]

        n_col = len(collision_runs)
        n_safe = len(safe_runs)

        lead_times = [r.warning_lead_time_s for r in collision_runs if r.warning_lead_time_s is not None]
        maint_ratios = [r.warning_maintained_ratio for r in collision_runs if r.warning_maintained_ratio is not None]
        warned_col_count = sum(1 for r in collision_runs if r.warned)
        false_warn_count = sum(1 for r in safe_runs if r.false_warning)

        decision_changes = [float(r.decision_changes) for r in results]
        steering_reversals = [float(r.steering_reversals) for r in results]
        min_clearances = [r.min_clearance_m for r in results]
        safe_clearance_rates = [r.safe_clearance_ratio for r in results]
        latencies = [r.mean_latency_ms for r in results]

        return {
            "total_runs": total_runs,
            "collision_runs": n_col,
            "safe_runs": n_safe,
            "warning_detection_rate": round(warned_col_count / max(1, n_col), 4),
            "false_warning_rate": round(false_warn_count / max(1, n_safe), 4),
            "mean_warning_lead_time_s": round(compute_mean(lead_times), 4),
            "median_warning_lead_time_s": round(compute_median(lead_times), 4),
            "std_warning_lead_time_s": round(compute_std(lead_times), 4),
            "mean_maintained_ratio": round(compute_mean(maint_ratios), 4),
            "mean_decision_changes": round(compute_mean(decision_changes), 2),
            "mean_steering_reversals": round(compute_mean(steering_reversals), 2),
            "mean_min_clearance_m": round(compute_mean(min_clearances), 4),
            "mean_safe_clearance_ratio": round(compute_mean(safe_clearance_rates), 4),
            "mean_latency_ms": round(compute_mean(latencies), 3),
        }

    def compile_pairwise_summary(self, pairwise_list: Sequence[PairwiseVariantComparison]) -> dict[str, Any]:
        """Compile statistical difference summary between FULL_PRISM and an ablated variant."""
        col_pairs = [p for p in pairwise_list if p.has_collision]
        delta_leads = [p.delta_lead_time_s for p in col_pairs if p.delta_lead_time_s is not None]
        delta_clrs = [p.delta_min_clearance_m for p in pairwise_list]
        delta_changes = [float(p.delta_decision_changes) for p in pairwise_list]
        delta_reversals = [float(p.delta_steering_reversals) for p in pairwise_list]

        return {
            "total_pairs": len(pairwise_list),
            "collision_pairs": len(col_pairs),
            "delta_lead_time": {
                "mean_s": round(compute_mean(delta_leads), 4),
                "median_s": round(compute_median(delta_leads), 4),
                "std_s": round(compute_std(delta_leads), 4),
                "cohens_d": round(compute_cohens_d(delta_leads), 4),
                "count": len(delta_leads),
            },
            "delta_min_clearance": {
                "mean_m": round(compute_mean(delta_clrs), 4),
                "median_m": round(compute_median(delta_clrs), 4),
                "std_m": round(compute_std(delta_clrs), 4),
                "cohens_d": round(compute_cohens_d(delta_clrs), 4),
            },
            "delta_decision_changes": {
                "mean": round(compute_mean(delta_changes), 2),
                "median": round(compute_median(delta_changes), 2),
                "std": round(compute_std(delta_changes), 2),
            },
            "delta_steering_reversals": {
                "mean": round(compute_mean(delta_reversals), 2),
                "median": round(compute_median(delta_reversals), 2),
                "std": round(compute_std(delta_reversals), 2),
            },
        }

    def aggregate_by_scenario(
        self,
        results_by_variant: Mapping[str, Sequence[VariantRunResult]],
        pairwise_by_variant: Mapping[str, Sequence[PairwiseVariantComparison]],
    ) -> dict[str, Any]:
        """Break down results by scenario family (S01 through S08)."""
        families = sorted(
            list(
                {
                    r.scenario_family
                    for var_results in results_by_variant.values()
                    for r in var_results
                }
            )
        )
        scenario_report: dict[str, Any] = {}

        for fam in families:
            fam_data: dict[str, Any] = {"variants": {}, "pairwise": {}}
            for var_name, var_results in results_by_variant.items():
                matching = [r for r in var_results if r.scenario_family == fam]
                fam_data["variants"][var_name] = self.compile_variant_summary(matching)

            for var_name, pairs in pairwise_by_variant.items():
                matching_p = [p for p in pairs if p.scenario_family == fam]
                fam_data["pairwise"][var_name] = self.compile_pairwise_summary(matching_p)

            scenario_report[fam] = fam_data

        return scenario_report

    def aggregate_by_noise(
        self,
        results_by_variant: Mapping[str, Sequence[VariantRunResult]],
        pairwise_by_variant: Mapping[str, Sequence[PairwiseVariantComparison]],
    ) -> dict[str, Any]:
        """Break down results by noise condition."""
        noise_conds = sorted(
            list(
                {
                    r.noise_condition
                    for var_results in results_by_variant.values()
                    for r in var_results
                }
            )
        )
        noise_report: dict[str, Any] = {}

        for n_cond in noise_conds:
            n_data: dict[str, Any] = {"variants": {}, "pairwise": {}}
            for var_name, var_results in results_by_variant.items():
                matching = [r for r in var_results if r.noise_condition == n_cond]
                n_data["variants"][var_name] = self.compile_variant_summary(matching)

            for var_name, pairs in pairwise_by_variant.items():
                matching_p = [p for p in pairs if p.noise_condition == n_cond]
                n_data["pairwise"][var_name] = self.compile_pairwise_summary(matching_p)

            noise_report[n_cond] = n_data

        return noise_report

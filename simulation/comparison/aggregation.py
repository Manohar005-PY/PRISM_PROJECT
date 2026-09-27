"""Aggregation engine for PRISM vs. Baseline comparative benchmarks."""

from __future__ import annotations

import datetime
import math
from typing import Any, Sequence

from simulation.comparison.models import (
    ComparativeAggregateReport,
    PairwiseRunComparison,
    PredictiveAdvantageStats,
)
from simulation.evaluation.state_metrics import compute_metric_stats


class ComparisonAggregator:
    """Aggregates run-level paired comparisons into dataset summaries and predictive advantage metrics."""

    def aggregate_comparisons(
        self,
        dataset_name: str,
        run_comparisons: Sequence[PairwiseRunComparison],
        comparison_version: str = "18.0.0",
    ) -> ComparativeAggregateReport:
        """Compute aggregate comparative metrics and predictive advantage statistics."""
        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
        total_runs = len(run_comparisons)

        if total_runs == 0:
            empty_adv = PredictiveAdvantageStats(
                total_collision_runs=0,
                paired_lead_time_samples=0,
                mean_delta_lead_time_s=None,
                median_delta_lead_time_s=None,
                std_delta_lead_time_s=None,
                min_delta_lead_time_s=None,
                max_delta_lead_time_s=None,
                earlier_warning_count=0,
                earlier_warning_rate=None,
                prism_missed_events=0,
                baseline_missed_events=0,
                prism_mean_lead_time_s=None,
                baseline_mean_lead_time_s=None,
                relative_lead_time_improvement_pct=None,
            )
            return ComparativeAggregateReport(
                dataset_name=dataset_name,
                comparison_version=comparison_version,
                created_at=now_str,
                total_runs_evaluated=0,
                total_collision_runs=0,
                total_safe_runs=0,
                predictive_advantage=empty_adv,
                overall_warning_summary={},
                overall_avoidance_summary={},
                scenario_breakdown={},
                noise_breakdown={},
                total_comparative_failures=0,
            )

        collision_runs = [r for r in run_comparisons if r.warning_comparison.has_collision]
        safe_runs = [r for r in run_comparisons if not r.warning_comparison.has_collision]

        total_collision_count = len(collision_runs)
        total_safe_count = len(safe_runs)

        # 1. Predictive Advantage Computation
        deltas: list[float] = [
            r.warning_comparison.delta_lead_time_s
            for r in collision_runs
            if r.warning_comparison.delta_lead_time_s is not None
        ]
        prism_leads: list[float] = [
            r.warning_comparison.prism_lead_time_s
            for r in collision_runs
            if r.warning_comparison.prism_lead_time_s is not None
        ]
        baseline_leads: list[float] = [
            r.warning_comparison.baseline_lead_time_s
            for r in collision_runs
            if r.warning_comparison.baseline_lead_time_s is not None
        ]

        prism_missed = sum(1 for r in collision_runs if r.warning_comparison.prism_lead_time_s is None)
        baseline_missed = sum(1 for r in collision_runs if r.warning_comparison.baseline_lead_time_s is None)
        earlier_count = sum(1 for r in collision_runs if r.warning_comparison.earlier_warning_system == "PRISM")

        delta_stats = compute_metric_stats(deltas)
        pr_mean_lead = round(sum(prism_leads) / len(prism_leads), 4) if prism_leads else None
        bl_mean_lead = round(sum(baseline_leads) / len(baseline_leads), 4) if baseline_leads else None

        rel_improvement: float | None = None
        if pr_mean_lead is not None and bl_mean_lead is not None and bl_mean_lead > 1e-4:
            rel_improvement = round(((pr_mean_lead - bl_mean_lead) / bl_mean_lead) * 100.0, 2)

        pred_advantage = PredictiveAdvantageStats(
            total_collision_runs=total_collision_count,
            paired_lead_time_samples=len(deltas),
            mean_delta_lead_time_s=delta_stats.mean,
            median_delta_lead_time_s=delta_stats.median,
            std_delta_lead_time_s=delta_stats.std,
            min_delta_lead_time_s=delta_stats.min,
            max_delta_lead_time_s=delta_stats.max,
            earlier_warning_count=earlier_count,
            earlier_warning_rate=round(earlier_count / max(1, total_collision_count), 4),
            prism_missed_events=prism_missed,
            baseline_missed_events=baseline_missed,
            prism_mean_lead_time_s=pr_mean_lead,
            baseline_mean_lead_time_s=bl_mean_lead,
            relative_lead_time_improvement_pct=rel_improvement,
        )

        # 2. Overall Warning Summary
        pr_warn_count = sum(1 for r in collision_runs if r.warning_comparison.prism_warned)
        bl_warn_count = sum(1 for r in collision_runs if r.warning_comparison.baseline_warned)
        pr_false_count = sum(1 for r in safe_runs if r.warning_comparison.prism_false_warning)
        bl_false_count = sum(1 for r in safe_runs if r.warning_comparison.baseline_false_warning)

        overall_warning = {
            "prism_collision_warning_detection_rate": round(pr_warn_count / max(1, total_collision_count), 4),
            "baseline_collision_warning_detection_rate": round(bl_warn_count / max(1, total_collision_count), 4),
            "prism_mean_lead_time_s": pr_mean_lead,
            "baseline_mean_lead_time_s": bl_mean_lead,
            "mean_lead_time_advantage_s": delta_stats.mean,
            "prism_false_warning_rate_on_safe_runs": round(pr_false_count / max(1, total_safe_count), 4),
            "baseline_false_warning_rate_on_safe_runs": round(bl_false_count / max(1, total_safe_count), 4),
        }

        # 3. Overall Avoidance Summary
        pr_changes = [r.avoidance_comparison.prism_decision_changes for r in run_comparisons]
        bl_changes = [r.avoidance_comparison.baseline_decision_changes for r in run_comparisons]
        pr_reversals = sum(r.avoidance_comparison.prism_steering_reversals for r in run_comparisons)
        bl_reversals = sum(r.avoidance_comparison.baseline_steering_reversals for r in run_comparisons)

        overall_avoidance = {
            "prism_mean_decision_changes": round(sum(pr_changes) / len(pr_changes), 2),
            "baseline_mean_decision_changes": round(sum(bl_changes) / len(bl_changes), 2),
            "prism_total_steering_reversals": pr_reversals,
            "baseline_total_steering_reversals": bl_reversals,
        }

        # 4. Scenario Breakdown
        scenario_families = sorted(list({r.scenario_family for r in run_comparisons}))
        scenario_breakdown: dict[str, Any] = {}

        for fam in scenario_families:
            fam_runs = [r for r in run_comparisons if r.scenario_family == fam]
            f_col_runs = [r for r in fam_runs if r.warning_comparison.has_collision]
            f_deltas = [r.warning_comparison.delta_lead_time_s for r in f_col_runs if r.warning_comparison.delta_lead_time_s is not None]
            f_pr_leads = [r.warning_comparison.prism_lead_time_s for r in f_col_runs if r.warning_comparison.prism_lead_time_s is not None]
            f_bl_leads = [r.warning_comparison.baseline_lead_time_s for r in f_col_runs if r.warning_comparison.baseline_lead_time_s is not None]
            f_earlier = sum(1 for r in f_col_runs if r.warning_comparison.earlier_warning_system == "PRISM")

            scenario_breakdown[fam] = {
                "total_runs": len(fam_runs),
                "collision_runs": len(f_col_runs),
                "prism_mean_lead_time_s": round(sum(f_pr_leads) / len(f_pr_leads), 4) if f_pr_leads else None,
                "baseline_mean_lead_time_s": round(sum(f_bl_leads) / len(f_bl_leads), 4) if f_bl_leads else None,
                "mean_delta_lead_time_s": round(sum(f_deltas) / len(f_deltas), 4) if f_deltas else None,
                "prism_earlier_warning_count": f_earlier,
                "comparative_failures": sum(len(r.failures) for r in fam_runs),
            }

        # 5. Noise Breakdown
        noise_conditions = sorted(list({r.noise_condition for r in run_comparisons}))
        noise_breakdown: dict[str, Any] = {}

        for cond in noise_conditions:
            cond_runs = [r for r in run_comparisons if r.noise_condition == cond]
            c_col_runs = [r for r in cond_runs if r.warning_comparison.has_collision]
            c_deltas = [r.warning_comparison.delta_lead_time_s for r in c_col_runs if r.warning_comparison.delta_lead_time_s is not None]
            c_pr_leads = [r.warning_comparison.prism_lead_time_s for r in c_col_runs if r.warning_comparison.prism_lead_time_s is not None]
            c_bl_leads = [r.warning_comparison.baseline_lead_time_s for r in c_col_runs if r.warning_comparison.baseline_lead_time_s is not None]

            noise_breakdown[cond] = {
                "total_runs": len(cond_runs),
                "collision_runs": len(c_col_runs),
                "prism_mean_lead_time_s": round(sum(c_pr_leads) / len(c_pr_leads), 4) if c_pr_leads else None,
                "baseline_mean_lead_time_s": round(sum(c_bl_leads) / len(c_bl_leads), 4) if c_bl_leads else None,
                "mean_delta_lead_time_s": round(sum(c_deltas) / len(c_deltas), 4) if c_deltas else None,
            }

        total_failures = sum(len(r.failures) for r in run_comparisons)

        return ComparativeAggregateReport(
            dataset_name=dataset_name,
            comparison_version=comparison_version,
            created_at=now_str,
            total_runs_evaluated=total_runs,
            total_collision_runs=total_collision_count,
            total_safe_runs=total_safe_count,
            predictive_advantage=pred_advantage,
            overall_warning_summary=overall_warning,
            overall_avoidance_summary=overall_avoidance,
            scenario_breakdown=scenario_breakdown,
            noise_breakdown=noise_breakdown,
            total_comparative_failures=total_failures,
        )

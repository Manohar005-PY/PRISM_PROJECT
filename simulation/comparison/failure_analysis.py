"""Comparative failure and anomaly extraction engine for PRISM Part 18."""

from __future__ import annotations

from typing import Any, Sequence

from simulation.baselines.models import ReactiveBaselineDecision
from simulation.comparison.models import ComparativeFailureCase, ComparativeWarningStats


class ComparativeFailureAnalyzer:
    """Detects and extracts structured comparative failure cases between PRISM and Baseline."""

    def extract_failures(
        self,
        scenario_id: str,
        scenario_family: str,
        run_id: str,
        gt_records: Sequence[dict[str, Any]],
        prism_records: Sequence[dict[str, Any]],
        baseline_decisions: Sequence[ReactiveBaselineDecision],
        warning_stats: ComparativeWarningStats,
    ) -> list[ComparativeFailureCase]:
        """Inspect paired execution and return structured comparative failure records."""
        failures: list[ComparativeFailureCase] = []

        # 1. Baseline Missed Warning on Collision Run
        if warning_stats.has_collision and warning_stats.prism_lead_time_s is not None and warning_stats.baseline_lead_time_s is None:
            failures.append(
                ComparativeFailureCase(
                    failure_type="BASELINE_MISSED_WARNING",
                    scenario_id=scenario_id,
                    scenario_family=scenario_family,
                    run_id=run_id,
                    timestamp=warning_stats.collision_time_s or 0.0,
                    frame_index=len(gt_records) - 1,
                    metric_name="warning_lead_time_s",
                    prism_value=warning_stats.prism_lead_time_s,
                    baseline_value=None,
                    ground_truth_summary={"collision_time_s": warning_stats.collision_time_s},
                    description=f"PRISM issued warning {warning_stats.prism_lead_time_s:.2f}s before collision, but Baseline failed to warn before impact.",
                )
            )

        # 2. Baseline Late Warning (< 1.0s Lead Time)
        if warning_stats.has_collision and warning_stats.baseline_lead_time_s is not None:
            if warning_stats.baseline_lead_time_s < 1.00:
                failures.append(
                    ComparativeFailureCase(
                        failure_type="BASELINE_LATE_WARNING",
                        scenario_id=scenario_id,
                        scenario_family=scenario_family,
                        run_id=run_id,
                        timestamp=warning_stats.baseline_first_warning_time_s or 0.0,
                        frame_index=int(round((warning_stats.baseline_first_warning_time_s or 0.0) / 0.05)),
                        metric_name="baseline_lead_time_s",
                        prism_value=warning_stats.prism_lead_time_s,
                        baseline_value=warning_stats.baseline_lead_time_s,
                        ground_truth_summary={"collision_time_s": warning_stats.collision_time_s},
                        description=f"Baseline warned late with only {warning_stats.baseline_lead_time_s:.2f}s lead time (PRISM: {warning_stats.prism_lead_time_s or 0.0:.2f}s).",
                    )
                )

        # 3. PRISM False Warning on Safe Run
        if not warning_stats.has_collision and warning_stats.prism_false_warning and not warning_stats.baseline_false_warning:
            failures.append(
                ComparativeFailureCase(
                    failure_type="PRISM_FALSE_WARNING",
                    scenario_id=scenario_id,
                    scenario_family=scenario_family,
                    run_id=run_id,
                    timestamp=warning_stats.prism_first_warning_time_s or 0.0,
                    frame_index=0,
                    metric_name="false_warning",
                    prism_value=True,
                    baseline_value=False,
                    ground_truth_summary={"has_collision": False},
                    description="PRISM triggered warning on a safe run while Baseline correctly remained in SAFE state.",
                )
            )

        # 4. Baseline False Warning on Safe Run
        if not warning_stats.has_collision and warning_stats.baseline_false_warning and not warning_stats.prism_false_warning:
            failures.append(
                ComparativeFailureCase(
                    failure_type="BASELINE_FALSE_WARNING",
                    scenario_id=scenario_id,
                    scenario_family=scenario_family,
                    run_id=run_id,
                    timestamp=warning_stats.baseline_first_warning_time_s or 0.0,
                    frame_index=0,
                    metric_name="false_warning",
                    prism_value=False,
                    baseline_value=True,
                    ground_truth_summary={"has_collision": False},
                    description="Baseline triggered proximity warning on a safe run while PRISM correctly evaluated low collision risk.",
                )
            )

        return failures

"""Central orchestrator for PRISM Part 18: Baseline Comparison & Predictive Advantage."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

from simulation.baselines.models import ReactiveBaselineConfig
from simulation.comparison.aggregation import ComparisonAggregator
from simulation.comparison.failure_analysis import ComparativeFailureAnalyzer
from simulation.comparison.metrics import ComparativeMetricsEvaluator
from simulation.comparison.models import (
    ComparativeAggregateReport,
    ComparativeFailureCase,
    ComparisonConfig,
    PairwiseRunComparison,
)
from simulation.comparison.runner import ComparisonRunner
from simulation.evaluation.loader import DatasetLoader


class PRISMBaselineComparator:
    """Master orchestrator conducting paired evaluation of PRISM against the Reactive Baseline."""

    def __init__(self, config: ComparisonConfig | None = None) -> None:
        self.config: ComparisonConfig = config if config is not None else ComparisonConfig()

        self.loader = DatasetLoader(self.config.dataset_dir)
        self.runner = ComparisonRunner(self.config.baseline_config)
        self.metrics_evaluator = ComparativeMetricsEvaluator()
        self.failure_analyzer = ComparativeFailureAnalyzer()
        self.aggregator = ComparisonAggregator()

    def compare_run(
        self,
        metadata: dict[str, Any],
        gt_records: Sequence[dict[str, Any]],
        obs_records: Sequence[dict[str, Any]],
        prism_records: Sequence[dict[str, Any]],
    ) -> PairwiseRunComparison:
        """Run paired comparison on a single experimental run."""
        scenario_id = metadata.get("scenario_id", "unknown_scenario")
        scenario_family = metadata.get("scenario_family", "unknown_family")
        run_id = metadata.get("run_id", scenario_id)
        num_frames = len(gt_records)
        duration = metadata.get("duration", 0.0)
        noise_cond = metadata.get("parameters", {}).get("noise_level", "NO_NOISE")

        # 1. Execute Reactive Baseline over raw observations
        baseline_decisions = self.runner.run_baseline_on_stream(obs_records, gt_records=gt_records)

        # 2. Compute Paired Metrics
        warn_stats, avoid_stats = self.metrics_evaluator.evaluate_run(
            gt_records=gt_records,
            prism_records=prism_records,
            baseline_decisions=baseline_decisions,
        )

        # 3. Extract Comparative Failures
        failures: list[ComparativeFailureCase] = self.failure_analyzer.extract_failures(
            scenario_id=scenario_id,
            scenario_family=scenario_family,
            run_id=run_id,
            gt_records=gt_records,
            prism_records=prism_records,
            baseline_decisions=baseline_decisions,
            warning_stats=warn_stats,
        )

        return PairwiseRunComparison(
            scenario_id=scenario_id,
            scenario_family=scenario_family,
            run_id=run_id,
            num_frames=num_frames,
            duration=duration,
            noise_condition=noise_cond,
            warning_comparison=warn_stats,
            avoidance_comparison=avoid_stats,
            failures=failures,
        )

    def compare_dataset(
        self,
        scenario_filter: str | None = None,
        run_filter: str | None = None,
        noise_filter: str | None = None,
        output_dir: Path | str | None = None,
        export_csv: bool = True,
    ) -> tuple[ComparativeAggregateReport, list[PairwiseRunComparison]]:
        """Evaluate entire benchmark dataset, persist comparison outputs, and generate summaries."""
        manifest = self.loader.load_manifest()
        dataset_name = manifest.get("dataset_name", "synthetic_prism_dataset")

        target_out = Path(output_dir) if output_dir else Path(self.config.output_dir)
        config_dir = target_out / "config"
        runs_dir = target_out / "runs"
        aggregate_dir = target_out / "aggregate"
        failures_dir = target_out / "failures"

        config_dir.mkdir(parents=True, exist_ok=True)
        runs_dir.mkdir(parents=True, exist_ok=True)
        aggregate_dir.mkdir(parents=True, exist_ok=True)
        failures_dir.mkdir(parents=True, exist_ok=True)

        # Persist baseline configuration
        with open(config_dir / "baseline_config.json", "w", encoding="utf-8") as f:
            json.dump(self.config.baseline_config.to_dict(), f, indent=2)

        run_comparisons: list[PairwiseRunComparison] = []
        all_failures: list[ComparativeFailureCase] = []

        # Iterate and compare matching runs
        for metadata, gt_recs, obs_recs, pr_recs in self.loader.iter_runs(
            scenario_family_filter=scenario_filter,
            run_id_filter=run_filter,
            noise_filter=noise_filter,
        ):
            run_comp = self.compare_run(metadata, gt_recs, obs_recs, pr_recs)
            run_comparisons.append(run_comp)
            all_failures.extend(run_comp.failures)

            # Persist per-run result JSON
            run_out_path = runs_dir / f"{run_comp.run_id}.json"
            with open(run_out_path, "w", encoding="utf-8") as f:
                json.dump(run_comp.to_dict(), f, indent=2)

        # Aggregate metrics
        aggregate_report = self.aggregator.aggregate_comparisons(
            dataset_name=dataset_name,
            run_comparisons=run_comparisons,
            comparison_version=self.config.version,
        )

        # Persist aggregate report & predictive advantage report
        agg_out_path = aggregate_dir / "overall_comparison.json"
        with open(agg_out_path, "w", encoding="utf-8") as f:
            json.dump(aggregate_report.to_dict(), f, indent=2)

        pred_adv_path = aggregate_dir / "predictive_advantage.json"
        with open(pred_adv_path, "w", encoding="utf-8") as f:
            json.dump(aggregate_report.predictive_advantage.to_dict(), f, indent=2)

        # Persist comparative failures JSONL
        failures_out_path = failures_dir / "comparative_failures.jsonl"
        with open(failures_out_path, "w", encoding="utf-8") as f:
            for fail in all_failures:
                f.write(json.dumps(fail.to_dict()) + "\n")

        # Optionally export CSV summary tables
        if export_csv:
            self._export_csv_reports(target_out, run_comparisons, aggregate_report)

        return aggregate_report, run_comparisons

    def _export_csv_reports(
        self,
        output_dir: Path,
        run_comparisons: list[PairwiseRunComparison],
        report: ComparativeAggregateReport,
    ) -> None:
        """Export tabular summaries for downstream spreadsheet analysis."""
        csv_dir = output_dir / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)

        # 1. Run-Level Comparison CSV
        run_csv = csv_dir / "run_comparison.csv"
        with open(run_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "run_id",
                "scenario_family",
                "noise_condition",
                "has_collision",
                "prism_warned",
                "baseline_warned",
                "collision_time_s",
                "prism_lead_time_s",
                "baseline_lead_time_s",
                "delta_lead_time_s",
                "earlier_warning_system",
                "prism_decision_changes",
                "baseline_decision_changes",
                "prism_steering_reversals",
                "baseline_steering_reversals",
                "failures_count",
            ])
            for r in run_comparisons:
                w = r.warning_comparison
                a = r.avoidance_comparison
                writer.writerow([
                    r.run_id,
                    r.scenario_family,
                    r.noise_condition,
                    w.has_collision,
                    w.prism_warned,
                    w.baseline_warned,
                    w.collision_time_s,
                    w.prism_lead_time_s,
                    w.baseline_lead_time_s,
                    w.delta_lead_time_s,
                    w.earlier_warning_system,
                    a.prism_decision_changes,
                    a.baseline_decision_changes,
                    a.prism_steering_reversals,
                    a.baseline_steering_reversals,
                    len(r.failures),
                ])

        # 2. Scenario-Level Breakdown CSV
        sc_csv = csv_dir / "scenario_comparison.csv"
        with open(sc_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "scenario_family",
                "total_runs",
                "collision_runs",
                "prism_mean_lead_time_s",
                "baseline_mean_lead_time_s",
                "mean_delta_lead_time_s",
                "prism_earlier_count",
                "comparative_failures",
            ])
            for fam, s_stat in report.scenario_breakdown.items():
                writer.writerow([
                    fam,
                    s_stat.get("total_runs"),
                    s_stat.get("collision_runs"),
                    s_stat.get("prism_mean_lead_time_s"),
                    s_stat.get("baseline_mean_lead_time_s"),
                    s_stat.get("mean_delta_lead_time_s"),
                    s_stat.get("prism_earlier_warning_count"),
                    s_stat.get("comparative_failures"),
                ])

        # 3. Noise Breakdown CSV
        noise_csv = csv_dir / "noise_comparison.csv"
        with open(noise_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "noise_condition",
                "total_runs",
                "collision_runs",
                "prism_mean_lead_time_s",
                "baseline_mean_lead_time_s",
                "mean_delta_lead_time_s",
            ])
            for cond, n_stat in report.noise_breakdown.items():
                writer.writerow([
                    cond,
                    n_stat.get("total_runs"),
                    n_stat.get("collision_runs"),
                    n_stat.get("prism_mean_lead_time_s"),
                    n_stat.get("baseline_mean_lead_time_s"),
                    n_stat.get("mean_delta_lead_time_s"),
                ])

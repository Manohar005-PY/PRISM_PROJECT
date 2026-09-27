"""Central evaluation orchestrator for PRISM Part 17."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

from simulation.evaluation.aggregation import MetricsAggregator
from simulation.evaluation.alignment import AlignedFrame, SpatialTemporalAligner
from simulation.evaluation.avoidance_metrics import AvoidanceEvaluator
from simulation.evaluation.dcpa_metrics import DCPAEvaluator
from simulation.evaluation.failure_analysis import FailureAnalyzer
from simulation.evaluation.loader import DatasetLoader
from simulation.evaluation.models import (
    AggregateEvaluationReport,
    EvaluationConfig,
    FailureCase,
    RunEvaluationResult,
)
from simulation.evaluation.prediction_metrics import PredictionEvaluator
from simulation.evaluation.risk_metrics import RiskEvaluator
from simulation.evaluation.robustness import RobustnessEvaluator
from simulation.evaluation.state_metrics import StateEstimationEvaluator
from simulation.evaluation.tracking_metrics import TrackingEvaluator
from simulation.evaluation.ttc_metrics import TTCEvaluator
from simulation.evaluation.warning_metrics import WarningEvaluator


class PRISMEvaluator:
    """Master quantitative evaluation orchestrator for synthetic PRISM experimental benchmarks."""

    def __init__(self, config: EvaluationConfig | None = None) -> None:
        self.config: EvaluationConfig = config if config is not None else EvaluationConfig()

        self.loader = DatasetLoader(self.config.dataset_dir)
        self.aligner = SpatialTemporalAligner(self.config.spatial_gating_threshold_m)
        self.state_evaluator = StateEstimationEvaluator()
        self.tracking_evaluator = TrackingEvaluator()
        self.prediction_evaluator = PredictionEvaluator(self.config.prediction_horizons)
        self.ttc_evaluator = TTCEvaluator()
        self.dcpa_evaluator = DCPAEvaluator()
        self.risk_evaluator = RiskEvaluator(
            event_horizons=self.config.event_horizons,
            risk_bins=self.config.risk_bins,
        )
        self.warning_evaluator = WarningEvaluator()
        self.avoidance_evaluator = AvoidanceEvaluator(min_safe_clearance_m=self.config.min_safe_clearance_m)
        self.failure_analyzer = FailureAnalyzer(config=self.config)
        self.robustness_evaluator = RobustnessEvaluator()
        self.aggregator = MetricsAggregator()

    def evaluate_run(
        self,
        metadata: dict[str, Any],
        gt_records: Sequence[dict[str, Any]],
        obs_records: Sequence[dict[str, Any]],
        prism_records: Sequence[dict[str, Any]],
    ) -> RunEvaluationResult:
        """Evaluate a single experimental run across all quantitative dimensions."""
        scenario_id = metadata.get("scenario_id", "unknown_scenario")
        scenario_family = metadata.get("scenario_family", "unknown_family")
        run_id = metadata.get("run_id", scenario_id)
        num_frames = len(gt_records)
        duration = metadata.get("duration", 0.0)
        noise_cond = metadata.get("parameters", {}).get("noise_level", "NO_NOISE")

        # 1. Spatio-temporal matching & alignment
        aligned_frames: list[AlignedFrame] = self.aligner.align_run(gt_records, obs_records, prism_records)

        # 2. State estimation metrics
        state_res = self.state_evaluator.evaluate_run(aligned_frames)

        # 3. Multi-object tracking metrics
        tracking_res = self.tracking_evaluator.evaluate_run(aligned_frames)

        # 4. Trajectory prediction metrics
        pred_res = self.prediction_evaluator.evaluate_run(aligned_frames)

        # 5. TTC & dCPA metrics
        ttc_res = self.ttc_evaluator.evaluate_run(aligned_frames)
        dcpa_res = self.dcpa_evaluator.evaluate_run(aligned_frames)
        ttc_dcpa_res = {
            "error_stats": ttc_res["error_stats"],
            "bias_s": ttc_res["bias_s"],
            "finite_vs_finite_count": ttc_res["finite_vs_finite_count"],
            "false_predicted_collision_count": ttc_res["false_predicted_collision_count"],
            "missed_predicted_collision_count": ttc_res["missed_predicted_collision_count"],
            "concordant_safe_count": ttc_res["concordant_safe_count"],
            "ratio_within_0_25s": ttc_res["ratio_within_0_25s"],
            "ratio_within_0_50s": ttc_res["ratio_within_0_50s"],
            "ratio_within_1_00s": ttc_res["ratio_within_1_00s"],
            "dcpa_stats": dcpa_res["error_stats"],
            "dcpa_bias_m": dcpa_res["bias_m"],
        }

        # 6. Risk estimation & calibration
        risk_res = self.risk_evaluator.evaluate_run(aligned_frames)

        # 7. Warning lead time & timeliness
        warning_res = self.warning_evaluator.evaluate_run(aligned_frames)

        # 8. Avoidance safety & decision stability
        avoidance_res = self.avoidance_evaluator.evaluate_run(aligned_frames)

        # 9. Automated failure analysis
        failures: list[FailureCase] = self.failure_analyzer.extract_failures(
            scenario_id=scenario_id,
            scenario_family=scenario_family,
            run_id=run_id,
            aligned_frames=aligned_frames,
        )

        return RunEvaluationResult(
            scenario_id=scenario_id,
            scenario_family=scenario_family,
            run_id=run_id,
            num_frames=num_frames,
            duration=duration,
            noise_condition=noise_cond,
            state_estimation=state_res,
            tracking=tracking_res,
            prediction=pred_res,
            ttc_dcpa=ttc_dcpa_res,
            risk=risk_res,
            warning=warning_res,
            avoidance=avoidance_res,
            failures=failures,
        )

    def evaluate_dataset(
        self,
        scenario_filter: str | None = None,
        run_filter: str | None = None,
        noise_filter: str | None = None,
        output_dir: Path | str | None = None,
        export_csv: bool = True,
    ) -> tuple[AggregateEvaluationReport, list[RunEvaluationResult]]:
        """Evaluate entire benchmark dataset, persist outputs, and generate aggregate summaries."""
        manifest = self.loader.load_manifest()
        dataset_name = manifest.get("dataset_name", "synthetic_prism_dataset")

        target_out = Path(output_dir) if output_dir else Path(self.config.output_dir)
        run_metrics_dir = target_out / "run_metrics"
        aggregate_dir = target_out / "aggregate"
        failures_dir = target_out / "failures"

        run_metrics_dir.mkdir(parents=True, exist_ok=True)
        aggregate_dir.mkdir(parents=True, exist_ok=True)
        failures_dir.mkdir(parents=True, exist_ok=True)

        run_results: list[RunEvaluationResult] = []
        all_failures: list[FailureCase] = []

        # Stream and evaluate runs
        for metadata, gt_recs, obs_recs, pr_recs in self.loader.iter_runs(
            scenario_family_filter=scenario_filter,
            run_id_filter=run_filter,
            noise_filter=noise_filter,
        ):
            run_res = self.evaluate_run(metadata, gt_recs, obs_recs, pr_recs)
            run_results.append(run_res)
            all_failures.extend(run_res.failures)

            # Persist per-run result JSON
            run_out_path = run_metrics_dir / f"{run_res.run_id}.json"
            with open(run_out_path, "w", encoding="utf-8") as f:
                json.dump(run_res.to_dict(), f, indent=2)

        # Aggregate metrics
        aggregate_report = self.aggregator.aggregate_dataset(
            dataset_name=dataset_name,
            run_results=run_results,
            evaluation_version=self.config.version,
        )

        # Compute noise breakdown
        noise_breakdown = self.robustness_evaluator.evaluate_noise_conditions(run_results)
        aggregate_dict = aggregate_report.to_dict()
        aggregate_dict["noise_breakdown"] = noise_breakdown

        # Persist aggregate report and summary.json
        agg_out_path = aggregate_dir / "aggregate_report.json"
        with open(agg_out_path, "w", encoding="utf-8") as f:
            json.dump(aggregate_dict, f, indent=2)

        summary_json_path = target_out / "summary.json"
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(aggregate_dict, f, indent=2)

        # Persist failures JSONL
        failures_out_path = failures_dir / "failure_cases.jsonl"
        with open(failures_out_path, "w", encoding="utf-8") as f:
            for fail in all_failures:
                f.write(json.dumps(fail.to_dict()) + "\n")

        # Optionally export CSV summary tables
        if export_csv:
            self._export_csv_reports(target_out, run_results, aggregate_report)

        return aggregate_report, run_results

    def _export_csv_reports(
        self,
        output_dir: Path,
        run_results: list[RunEvaluationResult],
        aggregate_report: AggregateEvaluationReport | None = None,
    ) -> None:
        """Export tabular summaries for downstream spreadsheet analysis."""
        csv_dir = output_dir / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)

        # 1. Per-Run CSV
        headers = [
            "run_id",
            "scenario_family",
            "noise_condition",
            "frames",
            "pos_mae_m",
            "pos_rmse_m",
            "vel_mae_mps",
            "confirmation_rate",
            "id_switches",
            "mota",
            "idf1",
            "pred_mae_1s_m",
            "pred_mae_2s_m",
            "ttc_mae_s",
            "has_collision",
            "lead_time_s",
            "decision_changes",
            "min_clearance_m",
            "failures_count",
        ]
        rows = []
        for r in run_results:
            rows.append([
                r.run_id,
                r.scenario_family,
                r.noise_condition,
                r.num_frames,
                r.state_estimation.get("position_2d", {}).get("mae"),
                r.state_estimation.get("position_2d", {}).get("rmse"),
                r.state_estimation.get("velocity_2d", {}).get("mae"),
                r.tracking.get("confirmation_rate"),
                r.tracking.get("id_switch_count"),
                r.tracking.get("mota"),
                r.tracking.get("idf1"),
                r.prediction.get("horizon_1.0s", {}).get("mae"),
                r.prediction.get("horizon_2.0s", {}).get("mae"),
                r.ttc_dcpa.get("error_stats", {}).get("mae"),
                r.warning.get("has_collision_event"),
                r.warning.get("lead_time_s"),
                r.avoidance.get("decision_change_count"),
                r.avoidance.get("min_ground_clearance_m"),
                len(r.failures),
            ])

        for fname in ("runs_summary.csv", "per_run.csv"):
            with open(csv_dir / fname, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(rows)

        # 2. Per-Scenario CSV
        if aggregate_report and aggregate_report.scenario_breakdown:
            scenario_csv = csv_dir / "per_scenario.csv"
            with open(scenario_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "scenario_family",
                    "runs_count",
                    "pos_mae_m",
                    "confirmation_rate",
                    "id_switches",
                    "mean_lead_time_s",
                    "failures_count",
                ])
                for fam, data in sorted(aggregate_report.scenario_breakdown.items()):
                    writer.writerow([
                        fam,
                        data.get("run_count"),
                        data.get("position_mae_m"),
                        data.get("mean_confirmation_rate"),
                        data.get("total_id_switches"),
                        data.get("mean_lead_time_s"),
                        data.get("failures_count"),
                    ])

        # 3. Summary CSV
        if aggregate_report:
            summary_csv = csv_dir / "summary.csv"
            with open(summary_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "value"])
                writer.writerow(["total_runs_evaluated", aggregate_report.total_runs_evaluated])
                writer.writerow(["total_frames_evaluated", aggregate_report.total_frames_evaluated])
                writer.writerow(["position_2d_mae_m", aggregate_report.overall_state_estimation.get("position_2d_macro", {}).get("mean")])
                writer.writerow(["position_2d_rmse_m", aggregate_report.overall_state_estimation.get("position_rmse_macro", {}).get("mean")])
                writer.writerow(["mean_mota", aggregate_report.overall_tracking.get("mean_mota")])
                writer.writerow(["mean_idf1", aggregate_report.overall_tracking.get("mean_idf1")])
                writer.writerow(["pred_1s_mae_m", aggregate_report.overall_prediction.get("horizon_1.0s", {}).get("mean")])
                writer.writerow(["pred_2s_mae_m", aggregate_report.overall_prediction.get("horizon_2.0s", {}).get("mean")])
                writer.writerow(["pred_3s_mae_m", aggregate_report.overall_prediction.get("horizon_3.0s", {}).get("mean")])
                writer.writerow(["ttc_mae_s", aggregate_report.overall_ttc_dcpa.get("ttc_mae_macro", {}).get("mean")])
                writer.writerow(["dcpa_mae_m", aggregate_report.overall_ttc_dcpa.get("dcpa_mae_macro", {}).get("mean")])
                writer.writerow(["mean_warning_lead_time_s", aggregate_report.overall_warning.get("lead_time_stats", {}).get("mean")])
                writer.writerow(["false_warning_rate_safe_runs", aggregate_report.overall_warning.get("false_warning_rate_on_safe_runs")])
                writer.writerow(["total_failures_identified", aggregate_report.total_failures_identified])

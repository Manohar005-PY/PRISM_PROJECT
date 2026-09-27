"""Master orchestrator for PRISM Part 20 Real-World Benchmark."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

from simulation.real_world.aggregation import RealWorldAggregator
from simulation.real_world.failure_analysis import RealWorldFailureAnalyzer
from simulation.real_world.loader import RealWorldDatasetLoader
from simulation.real_world.metrics import RealWorldMetricsEvaluator
from simulation.real_world.models import (
    RealWorldConfig,
    RealWorldFailureCase,
    RealWorldRunMetadata,
    RealWorldRunResult,
)
from simulation.real_world.runner import RealWorldRunner


class PRISMRealWorldAnalyzer:
    """Master orchestrator executing PRISM benchmark on real-world video datasets."""

    def __init__(self, config: RealWorldConfig | None = None) -> None:
        self.config: RealWorldConfig = config if config is not None else RealWorldConfig()
        self.loader = RealWorldDatasetLoader(self.config.dataset_dir)
        self.runner = RealWorldRunner(self.config)
        self.metrics_evaluator = RealWorldMetricsEvaluator()
        self.failure_analyzer = RealWorldFailureAnalyzer(self.config)
        self.aggregator = RealWorldAggregator()

    def evaluate_dataset(
        self,
        scenario_filter: str | None = None,
        run_filter: str | None = None,
        output_dir: Path | str | None = None,
        export_csv: bool = True,
    ) -> dict[str, Any]:
        """Execute real-world evaluation across all matching video runs."""
        out_dir = Path(output_dir) if output_dir is not None else Path(self.config.output_dir)

        # 1. Load video runs
        run_metas = list(self.loader.iter_runs(scenario_filter=scenario_filter, run_filter=run_filter))
        if not run_metas:
            raise ValueError("No matching real-world video runs found.")

        # 2. Execute PRISM on each video
        run_results: list[RealWorldRunResult] = []
        all_failures: list[RealWorldFailureCase] = []

        for meta in run_metas:
            frame_results = self.runner.run_video(meta)
            run_res = self.metrics_evaluator.evaluate_run(meta, frame_results)
            run_results.append(run_res)

            # Analyze failure modes
            fails = self.failure_analyzer.analyze_run(run_res)
            all_failures.extend(fails)

        # 3. Aggregate statistics
        overall_summary = self.aggregator.compile_summary(run_results)
        scenario_breakdown = self.aggregator.aggregate_by_scenario(run_results)

        # 4. Save results to disk
        if out_dir is not None:
            self._save_reports(
                out_dir=out_dir,
                run_results=run_results,
                overall_summary=overall_summary,
                scenario_breakdown=scenario_breakdown,
                all_failures=all_failures,
                export_csv=export_csv,
            )

        return {
            "overall_summary": overall_summary,
            "scenario_breakdown": scenario_breakdown,
            "failures_count": len(all_failures),
            "total_runs_evaluated": len(run_results),
            "run_results": run_results,
            "all_failures": all_failures,
        }

    def _save_reports(
        self,
        out_dir: Path,
        run_results: list[RealWorldRunResult],
        overall_summary: dict[str, Any],
        scenario_breakdown: dict[str, Any],
        all_failures: list[RealWorldFailureCase],
        export_csv: bool,
    ) -> None:
        """Persist structured benchmark files, per-run JSONs, and CSV reports."""
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config").mkdir(parents=True, exist_ok=True)
        (out_dir / "runs").mkdir(parents=True, exist_ok=True)
        (out_dir / "aggregate").mkdir(parents=True, exist_ok=True)
        (out_dir / "failures").mkdir(parents=True, exist_ok=True)
        (out_dir / "csv").mkdir(parents=True, exist_ok=True)

        # Config
        with open(out_dir / "config" / "evaluation_config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset_dir": self.config.dataset_dir,
                    "detector_type": self.runner.detector.active_backend,
                    "confidence_thresh": self.config.detector_confidence_thresh,
                    "prediction_horizon": self.config.prediction_horizon,
                    "min_safe_clearance_m": self.config.min_safe_clearance_m,
                    "camera_config": {
                        "height_m": self.config.camera_config.mounting_height_m,
                        "pitch_rad": self.config.camera_config.pitch_rad,
                        "resolution": list(self.config.camera_config.resolution),
                        "nominal_fps": self.config.camera_config.nominal_fps,
                    },
                },
                f,
                indent=2,
            )

        # Runs
        for r in run_results:
            sc_dir = out_dir / "runs" / r.scenario_family
            sc_dir.mkdir(parents=True, exist_ok=True)
            with open(sc_dir / f"{r.run_id}.json", "w", encoding="utf-8") as f:
                json.dump(r.to_dict(include_frames=True), f, indent=2)

        # Aggregates
        with open(out_dir / "aggregate" / "overall_report.json", "w", encoding="utf-8") as f:
            json.dump(overall_summary, f, indent=2)

        with open(out_dir / "aggregate" / "scenario_report.json", "w", encoding="utf-8") as f:
            json.dump(scenario_breakdown, f, indent=2)

        # Failures
        with open(out_dir / "failures" / "failure_cases.jsonl", "w", encoding="utf-8") as f:
            for fail in all_failures:
                f.write(json.dumps(fail.to_dict()) + "\n")

        # CSVs
        if export_csv:
            csv_dir = out_dir / "csv"

            def _fmt(val: Any) -> Any:
                return "N/A" if val is None else val

            # Overall CSV
            with open(csv_dir / "overall.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "total_runs", "col_runs", "safe_runs", "collision_event_recall", "collision_event_miss_rate",
                    "proximity_caution_alert_rate", "false_collision_warning_rate",
                    "mean_lead_time_s", "median_lead_time_s", "std_lead_time_s",
                    "tracking_continuity", "mean_track_coverage", "mean_detection_association_coverage",
                    "total_zero_detection_frames", "total_coasting_frames", "mean_gap_bridging_rate",
                    "mean_detections", "mean_decision_changes", "mean_steering_reversals", "mean_latency_ms"
                ])
                writer.writerow([
                    _fmt(overall_summary["total_runs"]), _fmt(overall_summary["collision_runs"]),
                    _fmt(overall_summary["safe_runs"]), _fmt(overall_summary["collision_event_recall"]),
                    _fmt(overall_summary["collision_event_miss_rate"]), _fmt(overall_summary["proximity_caution_alert_rate"]),
                    _fmt(overall_summary["false_collision_warning_rate"]),
                    _fmt(overall_summary["mean_warning_lead_time_s"]), _fmt(overall_summary["median_warning_lead_time_s"]),
                    _fmt(overall_summary["std_warning_lead_time_s"]),
                    _fmt(overall_summary["mean_tracking_continuity"]), _fmt(overall_summary["mean_track_coverage"]),
                    _fmt(overall_summary["mean_detection_association_coverage"]),
                    _fmt(overall_summary["total_zero_detection_frames"]), _fmt(overall_summary["total_coasting_frames"]),
                    _fmt(overall_summary["mean_gap_bridging_rate"]),
                    _fmt(overall_summary["mean_detection_count"]), _fmt(overall_summary["mean_decision_changes"]),
                    _fmt(overall_summary["mean_steering_reversals"]), _fmt(overall_summary["mean_latency_ms"])
                ])

            # Scenario CSV
            with open(csv_dir / "scenario.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "scenario_family", "total_runs", "col_runs", "safe_runs",
                    "collision_event_recall", "proximity_caution_alert_rate", "false_collision_warning_rate",
                    "mean_lead_time_s", "median_lead_time_s",
                    "tracking_continuity", "mean_track_coverage", "mean_detection_association_coverage",
                    "total_zero_detections", "total_coasting_frames", "mean_gap_bridging_rate",
                    "mean_decision_changes", "mean_steering_reversals", "mean_latency_ms"
                ])
                for fam, stats in scenario_breakdown.items():
                    writer.writerow([
                        fam, _fmt(stats["total_runs"]), _fmt(stats["collision_runs"]), _fmt(stats["safe_runs"]),
                        _fmt(stats["collision_event_recall"]), _fmt(stats["proximity_caution_alert_rate"]),
                        _fmt(stats["false_collision_warning_rate"]),
                        _fmt(stats["mean_warning_lead_time_s"]), _fmt(stats["median_warning_lead_time_s"]),
                        _fmt(stats["mean_tracking_continuity"]), _fmt(stats["mean_track_coverage"]),
                        _fmt(stats["mean_detection_association_coverage"]),
                        _fmt(stats["total_zero_detection_frames"]), _fmt(stats["total_coasting_frames"]),
                        _fmt(stats["mean_gap_bridging_rate"]),
                        _fmt(stats["mean_decision_changes"]), _fmt(stats["mean_steering_reversals"]),
                        _fmt(stats["mean_latency_ms"])
                    ])

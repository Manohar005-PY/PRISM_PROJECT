"""Master orchestrator for PRISM Part 19 Ablation & Sensitivity Analysis."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

from simulation.ablation.aggregation import AblationAggregator
from simulation.ablation.failure_analysis import AblationFailureAnalyzer
from simulation.ablation.metrics import AblationMetricsEvaluator
from simulation.ablation.models import (
    AblationConfig,
    AblationFailureCase,
    AblationVariant,
    FROZEN_SENSITIVITY_SPECS,
    PairwiseVariantComparison,
    SensitivitySettingResult,
    VariantRunResult,
)
from simulation.ablation.runner import AblationRunner
from simulation.ablation.sensitivity import SensitivityAnalyzer
from simulation.ablation.variants import create_variant_pipeline
from simulation.evaluation.loader import DatasetLoader


class PRISMAblationAnalyzer:
    """Master orchestrator executing complete ablation and sensitivity benchmarks."""

    def __init__(self, config: AblationConfig | None = None) -> None:
        self.config: AblationConfig = config if config is not None else AblationConfig()
        self.loader = DatasetLoader(self.config.dataset_dir)
        self.runner = AblationRunner()
        self.metrics_evaluator = AblationMetricsEvaluator(
            min_safe_clearance_m=self.config.min_safe_clearance_m
        )
        self.failure_analyzer = AblationFailureAnalyzer(self.config)
        self.aggregator = AblationAggregator()
        self.sensitivity_analyzer = SensitivityAnalyzer(self.metrics_evaluator)

    def run_ablation_analysis(
        self,
        variants: Sequence[AblationVariant | str] | None = None,
        scenario_filter: str | None = None,
        run_filter: str | None = None,
        noise_filter: str | None = None,
        run_sensitivity: bool = True,
        output_dir: Path | str | None = None,
        export_csv: bool = True,
    ) -> dict[str, Any]:
        """Execute full ablation and sensitivity benchmark across specified runs and variants."""
        target_variants = (
            [AblationVariant(v) for v in variants]
            if variants is not None
            else list(AblationVariant)
        )
        out_dir = Path(output_dir) if output_dir is not None else Path(self.config.output_dir)

        # 1. Load dataset runs into memory for consistent replay
        raw_runs = list(
            self.loader.iter_runs(
                scenario_family_filter=scenario_filter,
                run_id_filter=run_filter,
                noise_filter=noise_filter,
            )
        )
        # Drop prism_recs to guarantee clean evaluation from observations and gt only
        dataset_runs: list[tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]] = [
            (meta, gt_recs, obs_recs) for meta, gt_recs, obs_recs, _ in raw_runs
        ]

        if not dataset_runs:
            raise ValueError("No matching dataset runs found for evaluation.")

        # 2. Run all variants
        results_by_variant: dict[str, list[VariantRunResult]] = {}
        for var_enum in target_variants:
            var_name = var_enum.value
            pipe = create_variant_pipeline(
                variant=var_enum,
                prediction_horizon=self.config.prediction_horizon,
            )
            var_results: list[VariantRunResult] = []
            for meta, gt_recs, obs_recs in dataset_runs:
                frame_outs = self.runner.run_pipeline_on_stream(pipe, obs_recs, gt_recs)
                run_res = self.metrics_evaluator.evaluate_variant_run(
                    metadata=meta,
                    gt_records=gt_recs,
                    frame_outputs=frame_outs,
                    variant_id=var_name,
                )
                var_results.append(run_res)
            results_by_variant[var_name] = var_results

        # 3. Compute pairwise comparisons against FULL_PRISM
        pairwise_by_variant: dict[str, list[PairwiseVariantComparison]] = {}
        all_failures: list[AblationFailureCase] = []

        if AblationVariant.FULL_PRISM.value in results_by_variant:
            full_results = results_by_variant[AblationVariant.FULL_PRISM.value]
            for var_name, var_results in results_by_variant.items():
                if var_name == AblationVariant.FULL_PRISM.value:
                    continue
                pairs: list[PairwiseVariantComparison] = []
                for full_r, var_r in zip(full_results, var_results):
                    pair = self.metrics_evaluator.compare_runs(full_r, var_r)
                    pairs.append(pair)
                    # Extract failures
                    fails = self.failure_analyzer.analyze_comparison(full_r, var_r, pair)
                    all_failures.extend(fails)
                pairwise_by_variant[var_name] = pairs

        # 4. Aggregations
        overall_variants_summary: dict[str, Any] = {}
        overall_pairwise_summary: dict[str, Any] = {}
        for var_name, res_list in results_by_variant.items():
            overall_variants_summary[var_name] = self.aggregator.compile_variant_summary(res_list)
        for var_name, p_list in pairwise_by_variant.items():
            overall_pairwise_summary[var_name] = self.aggregator.compile_pairwise_summary(p_list)

        scenario_breakdown = self.aggregator.aggregate_by_scenario(results_by_variant, pairwise_by_variant)
        noise_breakdown = self.aggregator.aggregate_by_noise(results_by_variant, pairwise_by_variant)

        # 5. Sensitivity Analysis
        sensitivity_report: dict[str, list[SensitivitySettingResult]] = {}
        if run_sensitivity:
            sensitivity_report = self.sensitivity_analyzer.run_sensitivity_grid(dataset_runs)

        # 6. Save Artifacts & Reports
        if out_dir is not None:
            self._save_reports(
                out_dir=out_dir,
                results_by_variant=results_by_variant,
                pairwise_by_variant=pairwise_by_variant,
                overall_variants=overall_variants_summary,
                overall_pairwise=overall_pairwise_summary,
                scenario_breakdown=scenario_breakdown,
                noise_breakdown=noise_breakdown,
                sensitivity_report=sensitivity_report,
                all_failures=all_failures,
                export_csv=export_csv,
            )

        return {
            "overall_variants": overall_variants_summary,
            "overall_pairwise": overall_pairwise_summary,
            "scenario_breakdown": scenario_breakdown,
            "noise_breakdown": noise_breakdown,
            "sensitivity_report": {
                k: [s.to_dict() for s in v] for k, v in sensitivity_report.items()
            },
            "failures_count": len(all_failures),
            "results_by_variant": results_by_variant,
            "pairwise_by_variant": pairwise_by_variant,
            "all_failures": all_failures,
        }

    def _save_reports(
        self,
        out_dir: Path,
        results_by_variant: dict[str, list[VariantRunResult]],
        pairwise_by_variant: dict[str, list[PairwiseVariantComparison]],
        overall_variants: dict[str, Any],
        overall_pairwise: dict[str, Any],
        scenario_breakdown: dict[str, Any],
        noise_breakdown: dict[str, Any],
        sensitivity_report: dict[str, list[SensitivitySettingResult]],
        all_failures: list[AblationFailureCase],
        export_csv: bool,
    ) -> None:
        """Persist structured benchmark data, reports, configs, and CSV summaries to disk."""
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config").mkdir(parents=True, exist_ok=True)
        (out_dir / "runs").mkdir(parents=True, exist_ok=True)
        (out_dir / "aggregate").mkdir(parents=True, exist_ok=True)
        (out_dir / "sensitivity").mkdir(parents=True, exist_ok=True)
        (out_dir / "failures").mkdir(parents=True, exist_ok=True)
        (out_dir / "csv").mkdir(parents=True, exist_ok=True)

        # Configs
        with open(out_dir / "config" / "experiment_config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "prediction_horizon": self.config.prediction_horizon,
                    "min_safe_clearance_m": self.config.min_safe_clearance_m,
                    "lead_time_late_threshold_s": self.config.lead_time_late_threshold_s,
                    "decision_instability_threshold": self.config.decision_instability_threshold,
                    "steering_reversal_threshold": self.config.steering_reversal_threshold,
                    "variants": [v.value for v in AblationVariant],
                },
                f,
                indent=2,
            )

        with open(out_dir / "config" / "sensitivity_config.json", "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "param_name": s.param_name,
                        "target_component": s.target_component,
                        "nominal_value": s.nominal_value,
                        "sweep_values": list(s.sweep_values),
                        "unit": s.unit,
                        "description": s.description,
                    }
                    for s in FROZEN_SENSITIVITY_SPECS
                ],
                f,
                indent=2,
            )

        # Runs
        for var_name, res_list in results_by_variant.items():
            var_dir = out_dir / "runs" / var_name
            var_dir.mkdir(parents=True, exist_ok=True)
            for r in res_list:
                with open(var_dir / f"{r.run_id}.json", "w", encoding="utf-8") as f:
                    json.dump(r.to_dict(include_frames=True), f, indent=2)

        # Aggregates
        with open(out_dir / "aggregate" / "overall_report.json", "w", encoding="utf-8") as f:
            json.dump(
                {"variants": overall_variants, "pairwise": overall_pairwise},
                f,
                indent=2,
            )

        with open(out_dir / "aggregate" / "scenario_report.json", "w", encoding="utf-8") as f:
            json.dump(scenario_breakdown, f, indent=2)

        with open(out_dir / "aggregate" / "noise_report.json", "w", encoding="utf-8") as f:
            json.dump(noise_breakdown, f, indent=2)

        # Sensitivity
        with open(out_dir / "sensitivity" / "sensitivity_report.json", "w", encoding="utf-8") as f:
            json.dump(
                {k: [s.to_dict() for s in v] for k, v in sensitivity_report.items()},
                f,
                indent=2,
            )

        # Failures
        with open(out_dir / "failures" / "failure_cases.jsonl", "w", encoding="utf-8") as f:
            for fail in all_failures:
                f.write(json.dumps(fail.to_dict()) + "\n")

        # CSVs
        if export_csv:
            csv_dir = out_dir / "csv"
            # Overall CSV
            with open(csv_dir / "overall.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "variant", "total_runs", "col_runs", "warning_rate", "false_warning_rate",
                    "mean_lead_time_s", "median_lead_time_s", "std_lead_time_s", "maintained_ratio",
                    "mean_decision_changes", "mean_steering_reversals", "mean_min_clearance_m",
                    "safe_clearance_ratio", "mean_latency_ms"
                ])
                for v_name, stats in overall_variants.items():
                    writer.writerow([
                        v_name, stats["total_runs"], stats["collision_runs"], stats["warning_detection_rate"],
                        stats["false_warning_rate"], stats["mean_warning_lead_time_s"], stats["median_warning_lead_time_s"],
                        stats["std_warning_lead_time_s"], stats["mean_maintained_ratio"], stats["mean_decision_changes"],
                        stats["mean_steering_reversals"], stats["mean_min_clearance_m"], stats["mean_safe_clearance_ratio"],
                        stats["mean_latency_ms"]
                    ])

            # Pairwise CSV
            with open(csv_dir / "pairwise.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "variant", "total_pairs", "col_pairs", "delta_lead_mean_s", "delta_lead_median_s",
                    "delta_lead_cohens_d", "delta_clearance_mean_m", "delta_decision_changes_mean",
                    "delta_steering_reversals_mean"
                ])
                for v_name, stats in overall_pairwise.items():
                    writer.writerow([
                        v_name, stats["total_pairs"], stats["collision_pairs"],
                        stats["delta_lead_time"]["mean_s"], stats["delta_lead_time"]["median_s"],
                        stats["delta_lead_time"]["cohens_d"], stats["delta_min_clearance"]["mean_m"],
                        stats["delta_decision_changes"]["mean"], stats["delta_steering_reversals"]["mean"]
                    ])

            # Scenario CSV
            with open(csv_dir / "scenario.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "scenario_family", "variant", "total_runs", "col_runs", "warning_rate",
                    "mean_lead_time_s", "median_lead_time_s", "mean_decision_changes", "mean_steering_reversals", "mean_min_clearance_m"
                ])
                for fam, fam_dict in scenario_breakdown.items():
                    for v_name, stats in fam_dict.get("variants", {}).items():
                        writer.writerow([
                            fam, v_name, stats["total_runs"], stats["collision_runs"], stats["warning_detection_rate"],
                            stats["mean_warning_lead_time_s"], stats["median_warning_lead_time_s"],
                            stats["mean_decision_changes"], stats["mean_steering_reversals"], stats["mean_min_clearance_m"]
                        ])

            # Noise CSV
            with open(csv_dir / "noise.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "noise_condition", "variant", "total_runs", "col_runs", "warning_rate",
                    "mean_lead_time_s", "median_lead_time_s", "mean_decision_changes", "mean_steering_reversals", "mean_min_clearance_m"
                ])
                for n_cond, n_dict in noise_breakdown.items():
                    for v_name, stats in n_dict.get("variants", {}).items():
                        writer.writerow([
                            n_cond, v_name, stats["total_runs"], stats["collision_runs"], stats["warning_detection_rate"],
                            stats["mean_warning_lead_time_s"], stats["median_warning_lead_time_s"],
                            stats["mean_decision_changes"], stats["mean_steering_reversals"], stats["mean_min_clearance_m"]
                        ])

            # Sensitivity CSV
            with open(csv_dir / "sensitivity.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "param_name", "setting_value", "total_runs", "mean_lead_time_s", "median_lead_time_s",
                    "warning_rate", "false_warning_rate", "mean_decision_changes", "mean_steering_reversals",
                    "mean_min_clearance_m", "safe_clearance_rate", "mean_latency_ms"
                ])
                for param, s_list in sensitivity_report.items():
                    for s in s_list:
                        writer.writerow([
                            s.param_name, s.setting_value, s.total_runs, s.mean_lead_time_s,
                            s.median_lead_time_s, s.warning_rate, s.false_warning_rate,
                            s.mean_decision_changes, s.mean_steering_reversals, s.mean_min_clearance_m,
                            s.safe_clearance_rate, s.mean_latency_ms
                        ])

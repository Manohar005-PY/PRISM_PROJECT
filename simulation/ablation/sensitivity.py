"""Parameter sensitivity analysis engine for PRISM Part 19."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from app.risk.model import RiskConfig
from app.state.kalman import KalmanFilterConfig
from simulation.ablation.aggregation import compute_mean, compute_median
from simulation.ablation.metrics import AblationMetricsEvaluator
from simulation.ablation.models import (
    FROZEN_SENSITIVITY_SPECS,
    SensitivityParamSpec,
    SensitivitySettingResult,
    VariantRunResult,
)
from simulation.ablation.runner import AblationRunner
from simulation.ablation.variants import create_variant_pipeline


class SensitivityAnalyzer:
    """Evaluates parameter sensitivity sweeps on the production FULL_PRISM architecture."""

    def __init__(self, metrics_evaluator: AblationMetricsEvaluator | None = None) -> None:
        self.runner = AblationRunner()
        self.metrics_evaluator = (
            metrics_evaluator if metrics_evaluator is not None else AblationMetricsEvaluator()
        )

    def build_pipeline_for_setting(
        self,
        param_name: str,
        value: float,
    ) -> Any:
        """Create a FULL_PRISM pipeline with a specific parameter variation."""
        r_kwargs: dict[str, Any] = {"alpha_smooth": 0.50}
        k_kwargs: dict[str, Any] = {}
        pred_horizon = 3.0

        if param_name == "horizon":
            r_kwargs["horizon"] = value
            pred_horizon = value
        elif param_name == "thresh_high":
            r_kwargs["thresh_high"] = value
        elif param_name == "alpha_smooth":
            r_kwargs["alpha_smooth"] = value
        elif param_name == "k_sigma":
            r_kwargs["k_sigma"] = value
        elif param_name == "sigma_a":
            k_kwargs["sigma_ax"] = value
            k_kwargs["sigma_az"] = value
        else:
            raise ValueError(f"Unknown sensitivity parameter: {param_name}")

        r_cfg = RiskConfig(**r_kwargs)
        k_cfg = KalmanFilterConfig(**k_kwargs)

        return create_variant_pipeline(
            variant="FULL_PRISM",
            custom_risk_config=r_cfg,
            custom_kalman_config=k_cfg,
            prediction_horizon=pred_horizon,
        )

    def evaluate_setting(
        self,
        param_spec: SensitivityParamSpec,
        value: float,
        dataset_runs: Sequence[tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]],
    ) -> tuple[SensitivitySettingResult, list[VariantRunResult]]:
        """Run all dataset scenarios under a single parameter configuration."""
        pipeline = self.build_pipeline_for_setting(param_spec.param_name, value)
        run_results: list[VariantRunResult] = []

        for meta, gt_recs, obs_recs in dataset_runs:
            frame_outs = self.runner.run_pipeline_on_stream(pipeline, obs_recs, gt_recs)
            res = self.metrics_evaluator.evaluate_variant_run(
                metadata=meta,
                gt_records=gt_recs,
                frame_outputs=frame_outs,
                variant_id=f"SENS_{param_spec.param_name}_{value}",
            )
            run_results.append(res)

        total_runs = len(run_results)
        col_runs = [r for r in run_results if r.has_collision]
        safe_runs = [r for r in run_results if not r.has_collision]

        n_col = len(col_runs)
        n_safe = len(safe_runs)

        lead_times = [r.warning_lead_time_s for r in col_runs if r.warning_lead_time_s is not None]
        warned_count = sum(1 for r in col_runs if r.warned)
        false_warn_count = sum(1 for r in safe_runs if r.false_warning)

        dec_changes = [float(r.decision_changes) for r in run_results]
        steer_revs = [float(r.steering_reversals) for r in run_results]
        min_clrs = [r.min_clearance_m for r in run_results]
        safe_clr_ratios = [r.safe_clearance_ratio for r in run_results]
        latencies = [r.mean_latency_ms for r in run_results]

        summary = SensitivitySettingResult(
            param_name=param_spec.param_name,
            setting_value=value,
            total_runs=total_runs,
            collision_runs=n_col,
            safe_runs=n_safe,
            mean_lead_time_s=compute_mean(lead_times),
            median_lead_time_s=compute_median(lead_times),
            warning_rate=warned_count / max(1, n_col),
            false_warning_rate=false_warn_count / max(1, n_safe),
            mean_decision_changes=compute_mean(dec_changes),
            mean_steering_reversals=compute_mean(steer_revs),
            mean_min_clearance_m=compute_mean(min_clrs),
            safe_clearance_rate=compute_mean(safe_clr_ratios),
            mean_latency_ms=compute_mean(latencies),
        )

        return summary, run_results

    def run_sensitivity_grid(
        self,
        dataset_runs: Sequence[tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]],
        specs: Sequence[SensitivityParamSpec] | None = None,
    ) -> dict[str, list[SensitivitySettingResult]]:
        """Execute the full frozen sensitivity parameter grid."""
        target_specs = specs if specs is not None else FROZEN_SENSITIVITY_SPECS
        sensitivity_report: dict[str, list[SensitivitySettingResult]] = {}

        for spec in target_specs:
            results_for_param: list[SensitivitySettingResult] = []
            for val in spec.sweep_values:
                summary, _ = self.evaluate_setting(spec, val, dataset_runs)
                results_for_param.append(summary)
            sensitivity_report[spec.param_name] = results_for_param

        return sensitivity_report

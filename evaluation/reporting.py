"""Evaluation reporting: Formats scientifically rigorous tables, statistical bounds, and validation findings."""

from __future__ import annotations

from typing import Mapping, Sequence

from evaluation.calibration import DepthBinMetrics
from evaluation.robustness import RobustnessResult
from evaluation.sensitivity import SensitivityResult


def format_baseline_comparison_table(results: Mapping[str, Mapping[str, float]]) -> str:
    """Format comparative baseline table with explicit percentage and rate columns."""
    lines = [
        f"{'SYSTEM':<26} | {'HAZARD DET.':<12} | {'FALSE ALARM':<12} | {'UNNECESSARY STOP':<16} | {'SWITCH RATE':<12}",
        "-" * 88,
    ]
    for sys_name, m in results.items():
        det_str = f"{m.get('hazard_detection_rate', 0.0):.1f}%"
        fa_str = f"{m.get('false_alarm_rate', 0.0):.1f}%"
        stop_str = f"{m.get('unnecessary_stop_rate', 0.0):.1f}%"
        sw_str = f"{m.get('switch_rate', 0.0):.2f}/s"
        lines.append(f"{sys_name:<26} | {det_str:<12} | {fa_str:<12} | {stop_str:<16} | {sw_str:<12}")
    return "\n".join(lines)


def format_calibration_table(bins: Sequence[DepthBinMetrics]) -> str:
    """Format perception calibration table with empirical coverage and conservative status."""
    lines = [
        f"{'DEPTH BIN':<10} | {'SAMPLES':<8} | {'RMSE_Z':<8} | {'PRED_SIG_Z':<10} | {'RATIO_Z':<8} | {'COV_1SIG':<9} | {'COV_2SIG':<9} | {'STATUS':<16}",
        "-" * 95,
    ]
    for b in bins:
        lines.append(
            f"{b.depth_range:<10} | {b.sample_count:<8} | {b.rmse_z:<8.3f} | {b.predicted_sigma_z:<10.3f} | {b.calibration_ratio_z:<8.2f} | {b.coverage_1sigma_z:<8.1f}% | {b.coverage_2sigma_z:<8.1f}% | {b.calibration_status:<16}"
        )
    return "\n".join(lines)


def format_sensitivity_table(result: SensitivityResult) -> str:
    """Format parameter sensitivity sweep table."""
    lines = [
        f"Sensitivity Sweep: {result.parameter_name}",
        f"{'PARAM VALUE':<14} | {'HAZARD DET.':<12} | {'FALSE ALARM':<12} | {'UNNECESSARY STOP':<16} | {'SWITCH RATE':<12}",
        "-" * 78,
    ]
    for val in result.parameter_values:
        m = result.metrics_by_value.get(val, {})
        det_str = f"{m.get('hazard_detection_rate', 0.0):.1f}%"
        fa_str = f"{m.get('false_alarm_rate', 0.0):.1f}%"
        stop_str = f"{m.get('unnecessary_stop_rate', 0.0):.1f}%"
        sw_str = f"{m.get('switch_rate', 0.0):.2f}/s"
        lines.append(f"{val:<14.2f} | {det_str:<12} | {fa_str:<12} | {stop_str:<16} | {sw_str:<12}")
    return "\n".join(lines)


def format_robustness_table(result: RobustnessResult) -> str:
    """Format perturbation robustness degradation table."""
    lines = [
        f"Robustness Perturbation: {result.perturbation_type}",
        f"{'LEVEL':<10} | {'HAZARD DET.':<12} | {'FALSE ALARM':<12} | {'UNNECESSARY STOP':<16} | {'POS RMSE':<10}",
        "-" * 72,
    ]
    for lvl in result.perturbation_levels:
        m = result.metrics_by_level.get(lvl, {})
        det_str = f"{m.get('hazard_detection_rate', 0.0):.1f}%"
        fa_str = f"{m.get('false_alarm_rate', 0.0):.1f}%"
        stop_str = f"{m.get('unnecessary_stop_rate', 0.0):.1f}%"
        rmse_str = f"{m.get('position_rmse', 0.0):.3f}m"
        lines.append(f"{lvl:<10.2f} | {det_str:<12} | {fa_str:<12} | {stop_str:<16} | {rmse_str:<10}")
    return "\n".join(lines)

"""CLI executable script to run Part 17 quantitative evaluation on synthetic PRISM datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from simulation.evaluation.evaluator import PRISMEvaluator
from simulation.evaluation.models import EvaluationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PRISM Part 17: Quantitative Evaluation & Experimental Analysis Framework"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="experiments/datasets/synthetic",
        help="Path to synthetic dataset directory (default: experiments/datasets/synthetic)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/evaluation/results",
        help="Path to output directory (default: experiments/evaluation/results)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Optional filter for scenario family (e.g. S01_stationary, S02_head_on, S05_multi_agent)",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Optional filter for specific run ID (e.g. S02_head_on_run_0001)",
    )
    parser.add_argument(
        "--noise",
        type=str,
        default=None,
        help="Optional filter for noise condition (NO_NOISE, LOW, MED, HIGH, DROPOUT)",
    )
    parser.add_argument(
        "--gating-threshold",
        type=float,
        default=1.5,
        help="Hungarian spatial distance gating threshold in meters (default: 1.5)",
    )
    parser.add_argument(
        "--pred-threshold",
        type=float,
        default=1.5,
        help="Diagnostic prediction displacement error threshold in meters (default: 1.5)",
    )
    parser.add_argument(
        "--ttc-threshold",
        type=float,
        default=2.0,
        help="Diagnostic TTC absolute error threshold in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Disable tabular CSV export",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = EvaluationConfig(
        dataset_dir=args.dataset,
        output_dir=args.output,
        spatial_gating_threshold_m=args.gating_threshold,
        diagnostic_pred_error_threshold_m=args.pred_threshold,
        diagnostic_ttc_error_threshold_s=args.ttc_threshold,
    )

    print("================================================================================")
    print("                PRISM Part 17: Quantitative Evaluation Engine                   ")
    print("================================================================================")
    print(f" Dataset Directory   : {config.dataset_dir}")
    print(f" Output Directory    : {config.output_dir}")
    print(f" Scenario Filter     : {args.scenario or 'ALL'}")
    print(f" Run ID Filter       : {args.run or 'ALL'}")
    print(f" Noise Filter        : {args.noise or 'ALL'}")
    print(f" Gating Threshold    : {config.spatial_gating_threshold_m:.2f} m")
    print(f" Diagnostic Pred Th. : {config.diagnostic_pred_error_threshold_m:.2f} m")
    print(f" Diagnostic TTC Th.  : {config.diagnostic_ttc_error_threshold_s:.2f} s")
    print("================================================================================\n")

    evaluator = PRISMEvaluator(config)
    report, runs = evaluator.evaluate_dataset(
        scenario_filter=args.scenario,
        run_filter=args.run,
        noise_filter=args.noise,
        output_dir=args.output,
        export_csv=not args.no_csv,
    )

    print(f"[OK] Evaluated {len(runs)} experimental runs across {report.total_frames_evaluated} frames.")
    print(f"[OK] Run Composition: {report.overall_warning.get('collision_run_count', 0)} collision-event runs / {report.overall_warning.get('safe_run_count', 0)} non-collision runs")
    print(f"[OK] Total Diagnostic Failure Events: {report.total_failures_identified}")
    print("\n--- Summary Performance Across Evaluated Runs ---")

    # State Estimation Summary
    pos_macro = report.overall_state_estimation.get("position_2d_macro", {})
    vel_macro = report.overall_state_estimation.get("velocity_2d_macro", {})
    print(f" - State Estimation 2D Pos MAE   : {pos_macro.get('mean', 0.0):.4f} m (RMSE macro: {report.overall_state_estimation.get('position_rmse_macro', {}).get('mean', 0.0):.4f} m)")
    print(f" - State Estimation 2D Vel MAE   : {vel_macro.get('mean', 0.0):.4f} m/s")

    # Tracking Summary
    trk = report.overall_tracking
    print(f" - Tracking Confirmation Rate    : {trk.get('mean_confirmation_rate', 0.0) * 100:.1f}%")
    print(f" - Tracking MOTA / IDF1          : {trk.get('mean_mota', 0.0):.4f} / {trk.get('mean_idf1', 0.0):.4f}")
    print(f" - ID Switches / Fragmentations  : {trk.get('total_id_switches', 0)} / {trk.get('total_fragmentations', 0)}")

    # Prediction Summary (All 4 supported horizons)
    pred = report.overall_prediction
    print(f" - Trajectory Pred MAE (0.5s)    : {pred.get('horizon_0.5s', {}).get('mean', 0.0):.4f} m (RMSE: {pred.get('horizon_0.5s', {}).get('rmse', 0.0):.4f} m)")
    print(f" - Trajectory Pred MAE (1.0s)    : {pred.get('horizon_1.0s', {}).get('mean', 0.0):.4f} m (RMSE: {pred.get('horizon_1.0s', {}).get('rmse', 0.0):.4f} m)")
    print(f" - Trajectory Pred MAE (2.0s)    : {pred.get('horizon_2.0s', {}).get('mean', 0.0):.4f} m (RMSE: {pred.get('horizon_2.0s', {}).get('rmse', 0.0):.4f} m)")
    print(f" - Trajectory Pred MAE (3.0s)    : {pred.get('horizon_3.0s', {}).get('mean', 0.0):.4f} m (RMSE: {pred.get('horizon_3.0s', {}).get('rmse', 0.0):.4f} m)")

    # TTC Summary
    ttc_macro = report.overall_ttc_dcpa.get("ttc_mae_macro", {})
    dcpa_macro = report.overall_ttc_dcpa.get("dcpa_mae_macro", {})
    print(f" - TTC Estimation MAE (macro)    : {ttc_macro.get('mean', 0.0):.4f} s")
    print(f" - dCPA Estimation MAE (macro)   : {dcpa_macro.get('mean', 0.0):.4f} m")

    # Warning Summary
    warn = report.overall_warning
    warn_lead = warn.get("lead_time_stats", {})
    print(f" - Mean Warning Lead Time        : {warn_lead.get('mean', 0.0):.4f} s ({warn.get('collision_run_count', 0)} collision runs)")
    print(f" - False Warning Rate (Safe Runs): {warn.get('false_warning_rate_on_safe_runs', 0.0) * 100:.1f}% ({warn.get('safe_run_count', 0)} safe runs)")

    print("\n--- Scenario Family Breakdown ---")
    print(f"{'Scenario Family':<25} | {'Runs':<5} | {'Pos MAE (m)':<12} | {'Conf Rate':<10} | {'Lead Time (s)':<14} | {'Failures':<8}")
    print("-" * 88)
    for fam, s_stat in report.scenario_breakdown.items():
        pos_str = f"{s_stat['position_mae_m']:.4f}" if s_stat['position_mae_m'] is not None else "N/A"
        conf_str = f"{s_stat['mean_confirmation_rate'] * 100:.1f}%"
        lead_str = f"{s_stat['mean_lead_time_s']:.2f}" if s_stat['mean_lead_time_s'] is not None else "N/A"
        print(f"{fam:<25} | {s_stat['run_count']:<5} | {pos_str:<12} | {conf_str:<10} | {lead_str:<14} | {s_stat['failures_count']:<8}")

    print("\n--- Empirical Risk Stratification: Risk-Outcome Monotonicity (H=3.0s) ---")
    print(f"{'Risk Score Bin':<18} | {'Total Samples':<15} | {'Collision Events':<18} | {'Empirical Collision Freq'}")
    print("-" * 75)
    for b in report.risk_calibration_bins:
        freq_str = f"{b['empirical_collision_frequency'] * 100:.1f}%" if b['empirical_collision_frequency'] is not None else "N/A"
        print(f"{b['bin_label']:<18} | {b['total_samples']:<15} | {b['collision_events']:<18} | {freq_str}")

    print("\n================================================================================")
    print(f"[SUCCESS] Results exported to: {config.output_dir}")
    print("================================================================================")


if __name__ == "__main__":
    main()

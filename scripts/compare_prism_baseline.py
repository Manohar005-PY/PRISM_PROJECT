"""CLI executable script to run Part 18 baseline comparison against PRISM."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from simulation.baselines.models import ReactiveBaselineConfig
from simulation.comparison.comparator import PRISMBaselineComparator
from simulation.comparison.models import ComparisonConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PRISM Part 18: Baseline Comparison & Predictive Advantage Engine"
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
        default="experiments/comparison/part18",
        help="Path to comparison output directory (default: experiments/comparison/part18)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Optional filter for scenario family (e.g. S01_stationary, S02_head_on)",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Optional filter for specific run ID",
    )
    parser.add_argument(
        "--noise",
        type=str,
        default=None,
        help="Optional filter for noise condition (NO_NOISE, LOW_NOISE, MEDIUM_NOISE, HIGH_NOISE, DROPOUT_ONLY)",
    )
    parser.add_argument(
        "--warning-dist",
        type=float,
        default=3.0,
        help="Baseline warning distance threshold in meters (default: 3.0)",
    )
    parser.add_argument(
        "--avoid-dist",
        type=float,
        default=1.5,
        help="Baseline avoidance distance threshold in meters (default: 1.5)",
    )
    parser.add_argument(
        "--stop-dist",
        type=float,
        default=0.8,
        help="Baseline stop distance threshold in meters (default: 0.8)",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Disable tabular CSV export",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    baseline_cfg = ReactiveBaselineConfig(
        warning_distance_m=args.warning_dist,
        avoidance_distance_m=args.avoid_dist,
        stop_distance_m=args.stop_dist,
    )

    comp_cfg = ComparisonConfig(
        dataset_dir=args.dataset,
        output_dir=args.output,
        baseline_config=baseline_cfg,
    )

    print("================================================================================")
    print("        PRISM Part 18: Baseline Comparison & Predictive Advantage Engine        ")
    print("================================================================================")
    print(f" Dataset Directory       : {comp_cfg.dataset_dir}")
    print(f" Output Directory        : {comp_cfg.output_dir}")
    print(f" Baseline Name           : {baseline_cfg.name} (v{baseline_cfg.version})")
    print(f" Baseline Thresholds     : Stop <= {baseline_cfg.stop_distance_m:.1f}m | Avoid <= {baseline_cfg.avoidance_distance_m:.1f}m | Warn <= {baseline_cfg.warning_distance_m:.1f}m")
    print(f" Scenario Filter         : {args.scenario or 'ALL'}")
    print(f" Run ID Filter           : {args.run or 'ALL'}")
    print(f" Noise Filter            : {args.noise or 'ALL'}")
    print("================================================================================\n")

    comparator = PRISMBaselineComparator(comp_cfg)
    report, runs = comparator.compare_dataset(
        scenario_filter=args.scenario,
        run_filter=args.run,
        noise_filter=args.noise,
        output_dir=args.output,
        export_csv=not args.no_csv,
    )

    adv = report.predictive_advantage
    warn = report.overall_warning_summary
    avoid = report.overall_avoidance_summary

    print(f"[OK] Evaluated {len(runs)} experimental runs across {report.total_collision_runs} collision runs and {report.total_safe_runs} safe runs.")
    print(f"[OK] Total Comparative Discrepancies / Diagnostic Failures: {report.total_comparative_failures}")
    print("\n--- Predictive Advantage & Warning Lead-Time Comparison ---")
    print(f" - Collision Warning Detection   : PRISM {warn.get('prism_collision_warning_detection_rate', 0.0) * 100:.1f}% vs Baseline {warn.get('baseline_collision_warning_detection_rate', 0.0) * 100:.1f}%")
    print(f" - PRISM Mean Lead Time          : {adv.prism_mean_lead_time_s or 0.0:.4f} s (Median delta: {adv.median_delta_lead_time_s or 0.0:.4f} s)")
    print(f" - Baseline Mean Lead Time       : {adv.baseline_mean_lead_time_s or 0.0:.4f} s")
    print(f" - Mean Lead-Time Advantage (Delta) : +{adv.mean_delta_lead_time_s or 0.0:.4f} s (PRISM lead time - Baseline lead time)")
    if adv.relative_lead_time_improvement_pct is not None:
        print(f" - Relative Improvement          : +{adv.relative_lead_time_improvement_pct:.1f}% earlier warning buffer")
    print(f" - Earlier Warning Rate          : {(adv.earlier_warning_rate or 0.0) * 100:.1f}% of collision runs ({adv.earlier_warning_count}/{adv.total_collision_runs})")
    print(f" - Missed Collision Events       : PRISM {adv.prism_missed_events} vs Baseline {adv.baseline_missed_events}")
    print(f" - False Warning Rate (Safe Runs): PRISM {warn.get('prism_false_warning_rate_on_safe_runs', 0.0) * 100:.1f}% vs Baseline {warn.get('baseline_false_warning_rate_on_safe_runs', 0.0) * 100:.1f}%")

    print("\n--- Action Stability & Control Comparison ---")
    print(f" - Mean Decision Changes per Run : PRISM {avoid.get('prism_mean_decision_changes', 0.0):.2f} vs Baseline {avoid.get('baseline_mean_decision_changes', 0.0):.2f}")
    print(f" - Steering Reversals (Flapping) : PRISM {avoid.get('prism_total_steering_reversals', 0)} vs Baseline {avoid.get('baseline_total_steering_reversals', 0)}")

    print("\n--- Scenario Family Breakdown ---")
    print(f"{'Scenario Family':<25} | {'Runs':<5} | {'Col':<4} | {'PRISM Lead':<12} | {'Base Lead':<11} | {'Delta Lead':<12} | {'Earlier':<8}")
    print("-" * 88)
    for fam, s_stat in report.scenario_breakdown.items():
        pr_str = f"{s_stat['prism_mean_lead_time_s']:.2f}s" if s_stat['prism_mean_lead_time_s'] is not None else "N/A"
        bl_str = f"{s_stat['baseline_mean_lead_time_s']:.2f}s" if s_stat['baseline_mean_lead_time_s'] is not None else "N/A"
        del_str = f"+{s_stat['mean_delta_lead_time_s']:.2f}s" if s_stat['mean_delta_lead_time_s'] is not None else "N/A"
        print(f"{fam:<25} | {s_stat['total_runs']:<5} | {s_stat['collision_runs']:<4} | {pr_str:<12} | {bl_str:<11} | {del_str:<12} | {s_stat['prism_earlier_warning_count']:<8}")

    print("\n--- Noise Degradation Breakdown ---")
    print(f"{'Noise Condition':<20} | {'Runs':<5} | {'Col':<4} | {'PRISM Lead (s)':<16} | {'Baseline Lead (s)':<18} | {'Delta Lead (s)':<14}")
    print("-" * 85)
    for cond, n_stat in report.noise_breakdown.items():
        pr_str = f"{n_stat['prism_mean_lead_time_s']:.2f}s" if n_stat['prism_mean_lead_time_s'] is not None else "N/A"
        bl_str = f"{n_stat['baseline_mean_lead_time_s']:.2f}s" if n_stat['baseline_mean_lead_time_s'] is not None else "N/A"
        del_str = f"+{n_stat['mean_delta_lead_time_s']:.2f}s" if n_stat['mean_delta_lead_time_s'] is not None else "N/A"
        print(f"{cond:<20} | {n_stat['total_runs']:<5} | {n_stat['collision_runs']:<4} | {pr_str:<16} | {bl_str:<18} | {del_str:<14}")

    print("\n================================================================================")
    print(f"[SUCCESS] Part 18 Comparative Results exported to: {comp_cfg.output_dir}")
    print("================================================================================")


if __name__ == "__main__":
    main()

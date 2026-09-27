"""CLI executable script to run PRISM Part 19 Ablation & Sensitivity Analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from simulation.ablation.analyzer import PRISMAblationAnalyzer
from simulation.ablation.models import AblationConfig, AblationVariant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PRISM Part 19: Ablation & Sensitivity Analysis Engine"
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
        default="experiments/ablation/part19",
        help="Path to ablation output directory (default: experiments/ablation/part19)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Optional comma-separated filter for ablation variants (e.g. FULL_PRISM,NO_KALMAN,POSITION_ONLY)",
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
        "--sensitivity",
        action="store_true",
        default=True,
        help="Run parameter sensitivity sweeps on FULL_PRISM (default: True)",
    )
    parser.add_argument(
        "--no-sensitivity",
        action="store_false",
        dest="sensitivity",
        help="Skip parameter sensitivity sweeps",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        default=True,
        help="Export summary tabular CSV files (default: True)",
    )
    parser.add_argument(
        "--no-csv",
        action="store_false",
        dest="export_csv",
        help="Disable tabular CSV export",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    selected_variants = None
    if args.variant:
        selected_variants = [v.strip() for v in args.variant.split(",") if v.strip()]

    config = AblationConfig(
        dataset_dir=args.dataset,
        output_dir=args.output,
    )

    print("================================================================================")
    print("           PRISM Part 19: Ablation & Sensitivity Analysis Engine                ")
    print("================================================================================")
    print(f" Dataset Directory       : {config.dataset_dir}")
    print(f" Output Directory        : {config.output_dir}")
    print(f" Target Variants         : {selected_variants or [v.value for v in AblationVariant]}")
    print(f" Run Sensitivity Sweeps  : {args.sensitivity}")
    print(f" Scenario Filter         : {args.scenario or 'ALL'}")
    print(f" Run ID Filter           : {args.run or 'ALL'}")
    print(f" Noise Filter            : {args.noise or 'ALL'}")
    print("================================================================================\n")

    analyzer = PRISMAblationAnalyzer(config)
    results = analyzer.run_ablation_analysis(
        variants=selected_variants,
        scenario_filter=args.scenario,
        run_filter=args.run,
        noise_filter=args.noise,
        run_sensitivity=args.sensitivity,
        output_dir=args.output,
        export_csv=args.export_csv,
    )

    overall_v = results["overall_variants"]
    overall_p = results["overall_pairwise"]

    print(f"[OK] Evaluated {len(overall_v)} architectural variants.")
    print(f"[OK] Total Comparative Diagnostic Failure Cases: {results['failures_count']}")

    print("\n--- Summary Performance by Architectural Variant ---")
    print(f"{'Variant':<20} | {'Warn Rate':<10} | {'False Warn':<11} | {'Mean Lead (s)':<14} | {'Med Lead (s)':<13} | {'Changes':<8} | {'Reversals':<10} | {'Min Clr (m)':<12}")
    print("-" * 115)
    for v_name, stats in overall_v.items():
        w_rate = f"{stats['warning_detection_rate'] * 100:.1f}%"
        f_rate = f"{stats['false_warning_rate'] * 100:.1f}%"
        mean_l = f"{stats['mean_warning_lead_time_s']:.3f}s"
        med_l = f"{stats['median_warning_lead_time_s']:.3f}s"
        changes = f"{stats['mean_decision_changes']:.1f}"
        revs = f"{stats['mean_steering_reversals']:.2f}"
        clr = f"{stats['mean_min_clearance_m']:.3f}m"
        print(f"{v_name:<20} | {w_rate:<10} | {f_rate:<11} | {mean_l:<14} | {med_l:<13} | {changes:<8} | {revs:<10} | {clr:<12}")

    print("\n--- Pairwise Comparison vs FULL_PRISM (Full - Variant) ---")
    print(f"{'Ablated Variant':<20} | {'Mean Delta Lead (s)':<20} | {'Med Delta Lead (s)':<19} | {'Cohen d':<9} | {'Delta Changes':<14} | {'Delta Revs':<11}")
    print("-" * 105)
    for v_name, p_stat in overall_p.items():
        m_d = f"{p_stat['delta_lead_time']['mean_s']:+.3f}s"
        med_d = f"{p_stat['delta_lead_time']['median_s']:+.3f}s"
        cd = f"{p_stat['delta_lead_time']['cohens_d']:.2f}"
        d_ch = f"{p_stat['delta_decision_changes']['mean']:+.1f}"
        d_rev = f"{p_stat['delta_steering_reversals']['mean']:+.2f}"
        print(f"{v_name:<20} | {m_d:<20} | {med_d:<19} | {cd:<9} | {d_ch:<14} | {d_rev:<11}")

    if args.sensitivity and "sensitivity_report" in results:
        print("\n--- Sensitivity Grid Sweeps on FULL_PRISM ---")
        sens = results["sensitivity_report"]
        for p_name, s_list in sens.items():
            print(f"  Parameter: {p_name}")
            for s in s_list:
                print(f"    - Value {s['setting_value']:<5} -> Mean Lead: {s['mean_lead_time_s']:.3f}s | Warn Rate: {s['warning_rate']*100:.1f}% | False Warn: {s['false_warning_rate']*100:.1f}% | Chatter: {s['mean_decision_changes']:.1f}")

    print("\n================================================================================")
    print(f"[SUCCESS] Part 19 Ablation & Sensitivity Results exported to: {config.output_dir}")
    print("================================================================================")


if __name__ == "__main__":
    main()

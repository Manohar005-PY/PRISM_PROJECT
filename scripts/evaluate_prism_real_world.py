"""CLI executable script to run PRISM Part 20 Real-World Benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from simulation.real_world.analyzer import PRISMRealWorldAnalyzer
from simulation.real_world.models import RealWorldConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PRISM Part 20: Real-World Benchmark & Validation Engine"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="experiments/datasets/real_world",
        help="Path to real-world dataset directory (default: experiments/datasets/real_world)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/real_world/part20",
        help="Path to real-world benchmark output directory (default: experiments/real_world/part20)",
    )
    parser.add_argument(
        "--detector",
        type=str,
        choices=["auto", "yolo", "hog"],
        default="auto",
        help="Detection backend to use: 'auto' (YOLO if available else HOG), 'yolo', 'hog' (default: auto)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="Detector confidence threshold (default: 0.35)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Optional filter for scenario family (e.g. R01_stationary, R02_head_on)",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Optional filter for specific run ID",
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

    config = RealWorldConfig(
        dataset_dir=args.dataset,
        output_dir=args.output,
        detector_type=args.detector,
        detector_confidence_thresh=args.confidence,
    )

    print("================================================================================")
    print("             PRISM Part 20: Real-World Benchmark & Validation Engine            ")
    print("================================================================================")
    print(f" Dataset Directory       : {config.dataset_dir}")
    print(f" Output Directory        : {config.output_dir}")
    print(f" Detector Backend Mode   : {config.detector_type}")
    print(f" Confidence Threshold    : {config.detector_confidence_thresh:.2f}")
    print(f" Scenario Filter         : {args.scenario or 'ALL'}")
    print(f" Run ID Filter           : {args.run or 'ALL'}")
    print("================================================================================\n")

    analyzer = PRISMRealWorldAnalyzer(config)
    print(f"[INIT] Active Detector Backend: {analyzer.runner.detector.active_backend.upper()}")

    results = analyzer.evaluate_dataset(
        scenario_filter=args.scenario,
        run_filter=args.run,
        output_dir=args.output,
        export_csv=args.export_csv,
    )

    summary = results["overall_summary"]
    scenarios = results["scenario_breakdown"]

    print(f"\n[OK] Evaluated {results['total_runs_evaluated']} real-world video runs.")
    print(f"[OK] Total Comparative Diagnostic Failure Cases: {results['failures_count']}")

    def _fmt_pct(val: float | None) -> str:
        return f"{val * 100:.1f}%" if val is not None else "N/A"

    def _fmt_sec(val: float | None) -> str:
        return f"{val:.3f} s" if val is not None else "N/A"

    def _fmt_ms(val: float | None) -> str:
        return f"{val:.2f} ms" if val is not None else "N/A"

    print("\n--- Real-World Predictive Safety & Warning Metrics ---")
    print(f" - Collision-Event Recall (TP)      : {_fmt_pct(summary['collision_event_recall'])} ({summary['collision_runs']} collision runs)")
    print(f" - Collision-Event Miss Rate (FN)   : {_fmt_pct(summary['collision_event_miss_rate'])}")
    print(f" - Proximity / Caution Alert Rate   : {_fmt_pct(summary['proximity_caution_alert_rate'])} ({summary['safe_runs']} safe runs with proximity alerts)")
    print(f" - False Collision-Warning Rate     : {_fmt_pct(summary['false_collision_warning_rate'])} (Emergency stops in safe encounters)")
    print(f" - Mean Warning Lead Time           : {_fmt_sec(summary['mean_warning_lead_time_s'])} (Median: {_fmt_sec(summary['median_warning_lead_time_s'])}, Std: {_fmt_sec(summary['std_warning_lead_time_s'])})")
    print(f" - Detection Association Coverage   : {_fmt_pct(summary['mean_detection_association_coverage'])} (Track Coverage: {_fmt_pct(summary['mean_track_coverage'])})")
    print(f" - Zero-Detection Gaps & Coasting   : {summary['total_zero_detection_frames']} gap frames, {summary['total_coasting_frames']} coasted frames (Gap Bridging Rate: {_fmt_pct(summary['mean_gap_bridging_rate'])})")
    print(f" - Mean Processing Latency per Frame: {_fmt_ms(summary['mean_latency_ms'])}")
    print(f" - Mean Decision Changes per Run    : {summary['mean_decision_changes']:.2f}")
    print(f" - Mean Steering Reversals per Run  : {summary['mean_steering_reversals']:.2f}")

    print("\n--- Scenario Family Breakdown ---")
    print(f"{'Scenario Family':<23} | {'Runs':<5} | {'Col':<4} | {'Recall':<8} | {'Caution':<8} | {'FalseCol':<9} | {'Lead Time':<10} | {'Assoc Cov':<10} | {'Coasting':<9} | {'Latency':<9}")
    print("-" * 125)
    for fam, s_stat in scenarios.items():
        rec_str = _fmt_pct(s_stat["collision_event_recall"])
        caut_str = _fmt_pct(s_stat["proximity_caution_alert_rate"])
        fcol_str = _fmt_pct(s_stat["false_collision_warning_rate"])
        lead_str = f"{s_stat['mean_warning_lead_time_s']:.2f}s" if s_stat["mean_warning_lead_time_s"] is not None else "N/A"
        cont_str = _fmt_pct(s_stat["mean_detection_association_coverage"])
        coast_str = f"{s_stat['total_coasting_frames']} f"
        lat_str = _fmt_ms(s_stat["mean_latency_ms"])
        print(f"{fam:<23} | {s_stat['total_runs']:<5} | {s_stat['collision_runs']:<4} | {rec_str:<8} | {caut_str:<8} | {fcol_str:<9} | {lead_str:<10} | {cont_str:<10} | {coast_str:<9} | {lat_str:<9}")

    print("\n================================================================================")
    print(f"[SUCCESS] Part 20 Real-World Benchmark Results exported to: {config.output_dir}")
    print("================================================================================")


if __name__ == "__main__":
    main()

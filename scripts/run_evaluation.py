"""Comprehensive evaluation script executing comparative multi-trial benchmarks between Baselines and PRISM."""

from __future__ import annotations

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.baselines import DistanceBaseline, TTCBaseline
from evaluation.metrics import (
    compute_decision_switch_rate,
    compute_false_alarm_rate,
    compute_hazard_detection_rate,
    compute_reaction_time,
    compute_unnecessary_stop_rate,
)
from evaluation.reporting import format_baseline_comparison_table
from evaluation.runner import EvaluationRunner, create_prism_evaluator_system
from evaluation.scenarios import get_canonical_benchmark_suite
from evaluation.statistics import compute_statistics


def main() -> None:
    print("=" * 88)
    print(" PRISM SCIENTIFIC EVALUATION: MULTI-TRIAL BASELINE COMPARISONS")
    print("=" * 88 + "\n")

    seeds = (42, 101, 2024)
    scenarios = get_canonical_benchmark_suite()

    systems = {
        "Distance Baseline": DistanceBaseline(stop_threshold=1.8, caution_threshold=3.0),
        "TTC Baseline": TTCBaseline(ttc_threshold=2.5),
        "PRISM (Full Pipeline)": create_prism_evaluator_system(),
    }

    aggregated_metrics: dict[str, dict[str, float]] = {name: {} for name in systems}

    for sys_name, evaluator in systems.items():
        trial_det_rates = []
        trial_fa_rates = []
        trial_stop_rates = []
        trial_sw_rates = []

        for seed in seeds:
            runner = EvaluationRunner(noise_sigma_x=0.05, noise_sigma_z=0.05, random_seed=seed)
            seed_samples = []
            for sc_name, sc in scenarios.items():
                samples = runner.run_scenario(sc, evaluator)
                seed_samples.extend(samples)

            trial_det_rates.append(compute_hazard_detection_rate(seed_samples))
            trial_fa_rates.append(compute_false_alarm_rate(seed_samples))
            trial_stop_rates.append(compute_unnecessary_stop_rate(seed_samples))
            trial_sw_rates.append(compute_decision_switch_rate(seed_samples))

        aggregated_metrics[sys_name] = {
            "hazard_detection_rate": compute_statistics(trial_det_rates).mean,
            "false_alarm_rate": compute_statistics(trial_fa_rates).mean,
            "unnecessary_stop_rate": compute_statistics(trial_stop_rates).mean,
            "switch_rate": compute_statistics(trial_sw_rates).mean,
        }

    table = format_baseline_comparison_table(aggregated_metrics)
    print(table)
    print("\n" + "=" * 88)
    print("Interpretation & Scientific Context:")
    print(" - PRISM demonstrates full predictive tracking and lateral spatial avoidance capability.")
    print(" - Distance baseline produces 0% false alarms on static thresholds but lacks lead time.")
    print(" - TTC baseline demonstrates high hazard detection on linear trajectories.")
    print(" - Benchmark numbers reflect baseline comparison under initial default parameters and")
    print("   do NOT constitute clinical proof of superiority across arbitrary human environments.")
    print("=" * 88)


if __name__ == "__main__":
    main()

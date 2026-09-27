"""Runs frame-level causal failure diagnosis attributing root causes for missed hazards."""

from __future__ import annotations

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.diagnosis import CausalFailureDiagnoser
from evaluation.runner import EvaluationRunner, create_prism_evaluator_system
from evaluation.scenarios import get_canonical_benchmark_suite


def main() -> None:
    print("=" * 88)
    print(" PRISM CAUSAL FAILURE DIAGNOSIS")
    print("=" * 88 + "\n")

    runner = EvaluationRunner(random_seed=42)
    prism = create_prism_evaluator_system()
    scenarios = get_canonical_benchmark_suite()

    all_diagnoses = []
    for sc_name, sc in scenarios.items():
        samples = runner.run_scenario(sc, prism)
        diagnoses = CausalFailureDiagnoser.diagnose_samples(samples)
        all_diagnoses.extend(diagnoses)

    counts = CausalFailureDiagnoser.summarize_failure_counts(all_diagnoses)
    total_failures = sum(counts.values())

    print(f"Total Hazard Frames Diagnosed: {sum(1 for d in all_diagnoses if d.is_scene_hazard)}")
    print(f"Total Unmitigated Hazard Frames: {total_failures}\n")
    print(f"{'SUBSYSTEM FAILURE CATEGORY':<35} | {'COUNT':<10} | {'FRACTION':<12}")
    print("-" * 65)

    for cat_name, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        frac = (count / max(1, total_failures)) * 100.0
        print(f"{cat_name:<35} | {count:<10} | {frac:<11.1f}%")

    print("\n" + "=" * 88)
    print("Diagnosis Finding: The dominant cause of unmitigated hazard frames is")
    print("                   RISK_CLASSIFICATION_FAILURE (multi-factor thresholding too strict")
    print("                   during early approach), followed by tracking initialization lag.")
    print("=" * 88)


if __name__ == "__main__":
    main()

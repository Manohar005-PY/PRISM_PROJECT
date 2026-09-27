"""Runs parameter sensitivity sweeps over prediction horizon and risk smoothing alpha."""

from __future__ import annotations

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.reporting import format_sensitivity_table
from evaluation.sensitivity import SensitivityAnalyzer


def main() -> None:
    print("=" * 88)
    print(" PRISM PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 88 + "\n")

    analyzer = SensitivityAnalyzer(random_seed=42)

    # 1. Prediction Horizon Sweep
    print("1. EVALUATING PREDICTION HORIZON SENSITIVITY")
    res_horizon = analyzer.evaluate_horizon_sensitivity(horizons=(1.0, 1.5, 2.0, 3.0, 4.0))
    print(format_sensitivity_table(res_horizon))
    print()

    # 2. Risk Smoothing Alpha Sweep
    print("2. EVALUATING RISK SMOOTHING ALPHA SENSITIVITY")
    res_alpha = analyzer.evaluate_alpha_smoothing_sensitivity(alphas=(0.2, 0.4, 0.6, 0.8, 1.0))
    print(format_sensitivity_table(res_alpha))
    print("\n" + "=" * 88)
    print("Sensitivity Insight: Higher prediction horizon extends early warning lead time,")
    print("                     while lower smoothing alpha reduces decision switch frequency.")
    print("=" * 88)


if __name__ == "__main__":
    main()

"""Runtime throughput and latency percentile benchmark across scenarios."""

from __future__ import annotations

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.aggregation import aggregate_values
from evaluation.runner import EvaluationRunner, create_prism_evaluator_system
from evaluation.scenarios import get_canonical_benchmark_suite


def main() -> None:
    print("=" * 88)
    print(" PRISM RUNTIME LATENCY & TAIL BENCHMARK")
    print("=" * 88 + "\n")

    runner = EvaluationRunner(random_seed=42)
    scenarios = get_canonical_benchmark_suite()
    prism = create_prism_evaluator_system()

    latencies: list[float] = []

    for name, sc in scenarios.items():
        samples = runner.run_scenario(sc, prism)
        latencies.extend(s.processing_latency_ms for s in samples)

    stats = aggregate_values(latencies)
    sorted_lats = sorted(latencies)
    p95 = sorted_lats[int(len(sorted_lats) * 0.95)] if sorted_lats else 0.0
    p99 = sorted_lats[int(len(sorted_lats) * 0.99)] if sorted_lats else 0.0

    print(f"Total Frames Evaluated: {stats.count}")
    print(f"Mean Latency:           {stats.mean:.3f} ms")
    print(f"Median (P50) Latency:   {stats.median:.3f} ms")
    print(f"P95 Latency:            {p95:.3f} ms")
    print(f"P99 Latency:            {p99:.3f} ms")
    print(f"Min / Max Latency:      {stats.min_val:.3f} ms / {stats.max_val:.3f} ms")
    print(f"95% Confidence Interval: [{stats.ci_95_lower:.3f} ms, {stats.ci_95_upper:.3f} ms]")
    print(f"Effective Throughput:   {1000.0 / max(0.001, stats.mean):.1f} FPS")
    print("=" * 88)


if __name__ == "__main__":
    main()

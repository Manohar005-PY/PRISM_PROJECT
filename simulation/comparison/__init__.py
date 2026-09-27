"""PRISM Part 18: Baseline Comparison & Predictive Advantage Package."""

from simulation.comparison.comparator import PRISMBaselineComparator
from simulation.comparison.models import (
    ComparativeAggregateReport,
    ComparativeAvoidanceStats,
    ComparativeFailureCase,
    ComparativeWarningStats,
    ComparisonConfig,
    PairwiseRunComparison,
    PredictiveAdvantageStats,
)

__all__ = [
    "ComparativeAggregateReport",
    "ComparativeAvoidanceStats",
    "ComparativeFailureCase",
    "ComparativeWarningStats",
    "ComparisonConfig",
    "PRISMBaselineComparator",
    "PairwiseRunComparison",
    "PredictiveAdvantageStats",
]

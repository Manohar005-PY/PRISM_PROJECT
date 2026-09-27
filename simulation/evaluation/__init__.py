"""PRISM Part 17: Quantitative Evaluation & Experimental Analysis Framework.

Provides modular quantitative evaluation of state estimation, multi-object tracking,
motion prediction, collision geometry (TTC/dCPA), risk assessment, warning lead times,
avoidance decisions, and noise robustness against known synthetic ground truth.
"""

from simulation.evaluation.models import (
    AggregateEvaluationReport,
    ClassificationStats,
    EvaluationConfig,
    FailureCase,
    MetricStats,
    RunEvaluationResult,
)
from simulation.evaluation.evaluator import PRISMEvaluator

__all__ = [
    "AggregateEvaluationReport",
    "ClassificationStats",
    "EvaluationConfig",
    "FailureCase",
    "MetricStats",
    "PRISMEvaluator",
    "RunEvaluationResult",
]

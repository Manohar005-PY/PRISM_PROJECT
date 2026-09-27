"""Statistical estimators: Mean, median, standard error, and 95% confidence intervals."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class StatisticalSummary:
    """Rigorous statistical summary of an experimental metric distribution."""

    count: int
    mean: float
    median: float
    std_dev: float
    std_err: float
    min_val: float
    max_val: float
    ci_95_lower: float
    ci_95_upper: float
    is_gaussian_assumed: bool = True


def compute_statistics(values: Sequence[float]) -> StatisticalSummary:
    """Calculate descriptive statistics and 95% Student-t / asymptotic confidence intervals."""
    if not values:
        return StatisticalSummary(
            count=0,
            mean=0.0,
            median=0.0,
            std_dev=0.0,
            std_err=0.0,
            min_val=0.0,
            max_val=0.0,
            ci_95_lower=0.0,
            ci_95_upper=0.0,
            is_gaussian_assumed=True,
        )

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mean_val = sum(sorted_vals) / n

    if n % 2 == 1:
        median_val = sorted_vals[n // 2]
    else:
        median_val = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0

    if n > 1:
        variance = sum((x - mean_val) ** 2 for x in sorted_vals) / (n - 1)
        std_dev = math.sqrt(variance)
        std_err = std_dev / math.sqrt(n)
        # 1.96 critical value for large N; for small N, standard error margin
        margin = 1.96 * std_err
    else:
        std_dev = 0.0
        std_err = 0.0
        margin = 0.0

    ci_lower = mean_val - margin
    ci_upper = mean_val + margin

    return StatisticalSummary(
        count=n,
        mean=mean_val,
        median=median_val,
        std_dev=std_dev,
        std_err=std_err,
        min_val=sorted_vals[0],
        max_val=sorted_vals[-1],
        ci_95_lower=ci_lower,
        ci_95_upper=ci_upper,
        is_gaussian_assumed=(n >= 10),
    )

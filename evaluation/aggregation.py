"""Statistical aggregation: Mean, median, standard deviation, and 95% confidence intervals."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class StatisticalSummary:
    """Summary statistics for an evaluation metric distribution."""

    count: int
    mean: float
    median: float
    std_dev: float
    min_val: float
    max_val: float
    ci_95_lower: float
    ci_95_upper: float


def aggregate_values(values: Sequence[float]) -> StatisticalSummary:
    """Compute complete statistical summary across a sequence of values."""
    if not values:
        return StatisticalSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mean_val = sum(sorted_vals) / n

    if n % 2 == 1:
        median_val = sorted_vals[n // 2]
    else:
        median_val = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0

    variance = sum((x - mean_val) ** 2 for x in sorted_vals) / max(1, n - 1)
    std_dev = math.sqrt(variance)

    # 95% confidence interval for the mean: mean +/- 1.96 * (std / sqrt(n))
    margin = 1.96 * (std_dev / math.sqrt(n)) if n > 1 else 0.0
    ci_lower = mean_val - margin
    ci_upper = mean_val + margin

    return StatisticalSummary(
        count=n,
        mean=mean_val,
        median=median_val,
        std_dev=std_dev,
        min_val=sorted_vals[0],
        max_val=sorted_vals[-1],
        ci_95_lower=ci_lower,
        ci_95_upper=ci_upper,
    )

"""Quantitative state estimation evaluation metrics for PRISM.

Provides standard-library functions to compute Root Mean Squared Error (RMSE)
for position and velocity trajectories against ground truth.
"""

from __future__ import annotations

import math
from typing import Sequence

from app.core.types import Vector2


def position_rmse(
    estimated: Sequence[Vector2],
    ground_truth: Sequence[Vector2],
) -> float:
    """Compute Root Mean Squared Error (RMSE) in 2D ground position (meters).

    Args:
        estimated: Sequence of estimated position Vector2 instances.
        ground_truth: Sequence of true position Vector2 instances.

    Returns:
        Position RMSE in meters.

    Raises:
        ValueError: If sequences are empty or have unequal lengths.
    """
    n = len(estimated)
    if n == 0:
        raise ValueError("Cannot compute RMSE on empty sequences.")
    if n != len(ground_truth):
        raise ValueError(f"Sequence lengths do not match: {n} estimated vs {len(ground_truth)} ground truth.")

    sq_error_sum = 0.0
    for est, true in zip(estimated, ground_truth):
        dx = est.x - true.x
        dz = est.z - true.z
        sq_error_sum += dx * dx + dz * dz

    return math.sqrt(sq_error_sum / n)


def velocity_rmse(
    estimated: Sequence[Vector2],
    ground_truth: Sequence[Vector2],
) -> float:
    """Compute Root Mean Squared Error (RMSE) in 2D ground velocity (m/s).

    Args:
        estimated: Sequence of estimated velocity Vector2 instances.
        ground_truth: Sequence of true velocity Vector2 instances.

    Returns:
        Velocity RMSE in m/s.

    Raises:
        ValueError: If sequences are empty or have unequal lengths.
    """
    n = len(estimated)
    if n == 0:
        raise ValueError("Cannot compute RMSE on empty sequences.")
    if n != len(ground_truth):
        raise ValueError(f"Sequence lengths do not match: {n} estimated vs {len(ground_truth)} ground truth.")

    sq_error_sum = 0.0
    for est, true in zip(estimated, ground_truth):
        dvx = est.x - true.x
        dvz = est.z - true.z
        sq_error_sum += dvx * dvx + dvz * dvz

    return math.sqrt(sq_error_sum / n)


def measurement_position_rmse(
    noisy_measurements: Sequence[Vector2],
    ground_truth: Sequence[Vector2],
) -> float:
    """Compute Root Mean Squared Error (RMSE) of raw noisy position measurements.

    Args:
        noisy_measurements: Sequence of raw measured position Vector2 instances.
        ground_truth: Sequence of true position Vector2 instances.

    Returns:
        Measurement position RMSE in meters.
    """
    return position_rmse(noisy_measurements, ground_truth)


def estimation_improvement(
    noisy_rmse: float,
    kalman_rmse: float,
) -> float:
    """Compute the percentage error reduction achieved by Kalman filtering over raw measurements.

    Args:
        noisy_rmse: Baseline RMSE from raw noisy measurements.
        kalman_rmse: Filtered RMSE from Kalman state estimator.

    Returns:
        Percentage improvement: ((noisy_rmse - kalman_rmse) / noisy_rmse) * 100%.
    """
    if noisy_rmse <= 0.0:
        return 0.0
    return ((noisy_rmse - kalman_rmse) / noisy_rmse) * 100.0

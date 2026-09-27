"""Quantitative perception and ground-plane observation evaluation metrics."""

from __future__ import annotations

import math
from typing import Sequence

from app.core.types import Vector2


def contact_pixel_error(
    measured_contact: tuple[float, float],
    ground_truth_contact: tuple[float, float],
) -> float:
    """Compute Euclidean error in pixels between measured contact point and ground truth contact point."""
    du = measured_contact[0] - ground_truth_contact[0]
    dv = measured_contact[1] - ground_truth_contact[1]
    return math.hypot(du, dv)


def valid_projection_rate(valid_count: int, total_detections: int) -> float:
    """Compute the percentage of detections that successfully projected to physical ground coordinates.

    Returns:
        Percentage in [0.0, 100.0].
    """
    if total_detections <= 0:
        return 0.0
    return (valid_count / total_detections) * 100.0


def position_rmse(
    measured_positions: Sequence[Vector2],
    ground_truth_positions: Sequence[Vector2],
) -> float:
    """Compute 2D Euclidean Root Mean Square Error between physical observations and ground truth (meters)."""
    n = len(measured_positions)
    if n == 0 or n != len(ground_truth_positions):
        raise ValueError(f"Mismatched or empty sequences: {n} measured vs {len(ground_truth_positions)} ground truth.")

    sum_sq = sum(
        (m.x - g.x) ** 2 + (m.z - g.z) ** 2
        for m, g in zip(measured_positions, ground_truth_positions)
    )
    return math.sqrt(sum_sq / n)


def x_rmse(
    measured_positions: Sequence[Vector2],
    ground_truth_positions: Sequence[Vector2],
) -> float:
    """Compute lateral (X) RMSE between physical observations and ground truth (meters)."""
    n = len(measured_positions)
    if n == 0 or n != len(ground_truth_positions):
        raise ValueError(f"Mismatched or empty sequences: {n} measured vs {len(ground_truth_positions)} ground truth.")

    sum_sq = sum((m.x - g.x) ** 2 for m, g in zip(measured_positions, ground_truth_positions))
    return math.sqrt(sum_sq / n)


def z_rmse(
    measured_positions: Sequence[Vector2],
    ground_truth_positions: Sequence[Vector2],
) -> float:
    """Compute longitudinal / depth (Z) RMSE between physical observations and ground truth (meters)."""
    n = len(measured_positions)
    if n == 0 or n != len(ground_truth_positions):
        raise ValueError(f"Mismatched or empty sequences: {n} measured vs {len(ground_truth_positions)} ground truth.")

    sum_sq = sum((m.z - g.z) ** 2 for m, g in zip(measured_positions, ground_truth_positions))
    return math.sqrt(sum_sq / n)


def normalized_position_error(
    measured: Vector2,
    ground_truth: Vector2,
    measurement_covariance: tuple[tuple[float, float], tuple[float, float]],
) -> float:
    """Compute normalized Mahalanobis residual error: e^T * R^-1 * e = (dx^2 / R00) + (dz^2 / R11).

    Used as an uncertainty consistency diagnostic: expected value is approximately 2.0 for a calibrated 2D Gaussian.
    """
    dx = measured.x - ground_truth.x
    dz = measured.z - ground_truth.z
    var_x = measurement_covariance[0][0]
    var_z = measurement_covariance[1][1]

    if var_x <= 0.0 or var_z <= 0.0:
        raise ValueError("Measurement covariance diagonal elements must be strictly positive.")

    return (dx * dx) / var_x + (dz * dz) / var_z

"""Quantitative evaluation metrics for trajectory prediction and collision geometry.

Implements timestamp-aligned error calculations between predicted trajectories and
ground-truth simulated entities.
"""

from __future__ import annotations

import math
from typing import Sequence

from app.core.types import Trajectory, Vector2
from app.prediction.collision_geometry import CPAResult
from simulation.entities import SimulatedEntity


def trajectory_position_rmse(
    predicted: Trajectory,
    ground_truth: Sequence[SimulatedEntity],
    time_tolerance: float = 1e-4,
) -> float:
    """Compute Root Mean Squared Error (RMSE) between predicted trajectory and ground truth.

    Matches points strictly by timestamp: e(t_i) = ||p_pred(t_i) - p_true(t_i)||.

    Args:
        predicted: Predicted Trajectory instance.
        ground_truth: Sequence of true SimulatedEntity instances over time.
        time_tolerance: Max timestamp difference to consider a match.

    Returns:
        Position prediction RMSE in meters.

    Raises:
        ValueError: If no timestamp-matched points are found.
    """
    if not predicted.points or not ground_truth:
        raise ValueError("Cannot compute trajectory RMSE on empty trajectory or ground truth.")

    # Build ground truth timestamp lookup
    gt_by_time: dict[float, Vector2] = {gt.timestamp: gt.position for gt in ground_truth}

    sq_error_sum = 0.0
    matched_count = 0

    for pt in predicted.points:
        # Find matching ground-truth point within tolerance
        matched_gt_pos: Vector2 | None = None
        for gt_t, gt_pos in gt_by_time.items():
            if abs(pt.timestamp - gt_t) <= time_tolerance:
                matched_gt_pos = gt_pos
                break

        if matched_gt_pos is not None:
            err = (pt.position - matched_gt_pos).magnitude
            sq_error_sum += err * err
            matched_count += 1

    if matched_count == 0:
        raise ValueError("No matching timestamps found between predicted trajectory and ground truth.")

    return math.sqrt(sq_error_sum / matched_count)


def prediction_error_at_horizon(
    predicted: Trajectory,
    ground_truth: Sequence[SimulatedEntity],
    target_time_offset: float,
    time_tolerance: float = 1e-4,
) -> float:
    """Compute Euclidean position error ||p_pred - p_true|| at a specific future horizon offset.

    Args:
        predicted: Predicted Trajectory.
        ground_truth: Sequence of true SimulatedEntity instances.
        target_time_offset: Relative horizon time offset (e.g. 0.5s, 1.0s, 2.0s).
        time_tolerance: Max timestamp difference.

    Returns:
        Euclidean position error in meters at the target horizon.

    Raises:
        ValueError: If target horizon is not found in predicted points or ground truth.
    """
    if not predicted.points:
        raise ValueError("Predicted trajectory is empty.")

    t0 = predicted.points[0].timestamp
    target_t = t0 + target_time_offset

    # Find predicted point
    pred_pt: Vector2 | None = None
    for pt in predicted.points:
        if abs(pt.timestamp - target_t) <= time_tolerance:
            pred_pt = pt.position
            break

    if pred_pt is None:
        raise ValueError(f"Target timestamp {target_t} (offset {target_time_offset}s) not found in trajectory points.")

    # Find ground truth point
    gt_pt: Vector2 | None = None
    for gt in ground_truth:
        if abs(gt.timestamp - target_t) <= time_tolerance:
            gt_pt = gt.position
            break

    if gt_pt is None:
        raise ValueError(f"Target timestamp {target_t} (offset {target_time_offset}s) not found in ground truth.")

    return (pred_pt - gt_pt).magnitude


def cpa_error(
    predicted_cpa: CPAResult,
    ground_truth_cpa: CPAResult,
) -> tuple[float, float]:
    """Compute absolute error in CPA time and CPA distance against ground truth.

    Args:
        predicted_cpa: CPA result calculated from estimated state.
        ground_truth_cpa: Exact CPA result calculated from ground truth.

    Returns:
        (time_error_seconds, distance_error_meters).
    """
    time_err = abs(predicted_cpa.time_to_cpa - ground_truth_cpa.time_to_cpa)
    dist_err = abs(predicted_cpa.distance_at_cpa - ground_truth_cpa.distance_at_cpa)
    return (time_err, dist_err)

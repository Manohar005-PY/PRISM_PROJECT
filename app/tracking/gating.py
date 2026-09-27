"""Mahalanobis innovation distance and spatial gating calculations for data association."""

from __future__ import annotations

import math

from app.core.types import Covariance4x4, Vector2


def compute_innovation_covariance(
    P_pred: Covariance4x4 | None,
    R_meas: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute 2x2 innovation covariance matrix S = H * P_pred * H^T + R_meas.

    Args:
        P_pred: 4x4 predicted state covariance matrix (or None).
        R_meas: 2x2 measurement noise covariance matrix.

    Returns:
        2x2 innovation covariance matrix tuple ((S00, S01), (S10, S11)).
    """
    p00 = P_pred[0][0] if P_pred is not None else 0.0
    p01 = P_pred[0][1] if P_pred is not None else 0.0
    p10 = P_pred[1][0] if P_pred is not None else 0.0
    p11 = P_pred[1][1] if P_pred is not None else 0.0

    r00 = R_meas[0][0]
    r01 = R_meas[0][1]
    r10 = R_meas[1][0]
    r11 = R_meas[1][1]

    return (
        (p00 + r00, p01 + r01),
        (p10 + r10, p11 + r11),
    )


def invert_2x2(
    S: tuple[tuple[float, float], tuple[float, float]],
    epsilon: float = 1e-9,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Compute analytical inverse of 2x2 matrix S.

    Returns:
        2x2 inverse matrix tuple, or None if matrix is singular (|det| <= epsilon).
    """
    a, b = S[0]
    c, d = S[1]
    det = a * d - b * c

    if abs(det) <= epsilon or not math.isfinite(det):
        return None

    inv_det = 1.0 / det
    return (
        (d * inv_det, -b * inv_det),
        (-c * inv_det, a * inv_det),
    )


def compute_mahalanobis_distance(
    measurement: Vector2,
    predicted_position: Vector2,
    S_inv: tuple[tuple[float, float], tuple[float, float]],
) -> float:
    """Compute squared Mahalanobis distance: d_M^2 = nu^T * S^-1 * nu.

    Args:
        measurement: Measured position Vector2(X, Z).
        predicted_position: Predicted position Vector2(X, Z).
        S_inv: Inverted 2x2 innovation covariance matrix.

    Returns:
        Squared Mahalanobis distance d_M^2.
    """
    dx = measurement.x - predicted_position.x
    dz = measurement.z - predicted_position.z

    # v = [dx, dz]
    # S_inv * v = [S00*dx + S01*dz, S10*dx + S11*dz]
    sv_x = S_inv[0][0] * dx + S_inv[0][1] * dz
    sv_z = S_inv[1][0] * dx + S_inv[1][1] * dz

    d_m2 = dx * sv_x + dz * sv_z
    return max(0.0, float(d_m2)) if math.isfinite(d_m2) else float("inf")


def evaluate_gate(
    measurement: Vector2,
    predicted_position: Vector2,
    S: tuple[tuple[float, float], tuple[float, float]],
    gating_threshold: float,
) -> tuple[bool, float]:
    """Check if measurement lies within the spatial validation gate S.

    Returns:
        (is_accepted, d_M^2) tuple.
    """
    S_inv = invert_2x2(S)
    if S_inv is None:
        return False, float("inf")

    d_m2 = compute_mahalanobis_distance(measurement, predicted_position, S_inv)
    return (d_m2 <= gating_threshold, d_m2)

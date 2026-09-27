"""Spatial association, Mahalanobis innovation gating, and deterministic global assignment."""

from __future__ import annotations

import math
from typing import Sequence

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


def mahalanobis_distance_squared(
    z: Vector2,
    z_hat: Vector2,
    S: tuple[tuple[float, float], tuple[float, float]],
) -> float:
    """Compute squared Mahalanobis distance: d_M^2 = (z - z_hat)^T * S^-1 * (z - z_hat).

    Args:
        z: Measured ground position Vector2(X, Z).
        z_hat: Predicted ground position Vector2(X, Z).
        S: 2x2 innovation covariance matrix.

    Returns:
        Squared Mahalanobis distance d_M^2, or float('inf') if S is singular.
    """
    S_inv = invert_2x2(S)
    if S_inv is None:
        return float("inf")

    dx = z.x - z_hat.x
    dz = z.z - z_hat.z

    sv_x = S_inv[0][0] * dx + S_inv[0][1] * dz
    sv_z = S_inv[1][0] * dx + S_inv[1][1] * dz

    d_m2 = dx * sv_x + dz * sv_z
    return max(0.0, float(d_m2)) if math.isfinite(d_m2) else float("inf")


def linear_sum_assignment(cost_matrix: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    """Solve linear sum assignment min sum C[i, j] using Hungarian (Munkres) algorithm.

    Pure standard-library implementation supporting arbitrary rectangular matrices.

    Args:
        cost_matrix: M x N matrix of assignment costs.

    Returns:
        List of (row_idx, col_idx) matched pairs.
    """
    m = len(cost_matrix)
    if m == 0:
        return []
    n = len(cost_matrix[0])
    if n == 0:
        return []

    # Handle rectangular matrix by padding to square K x K
    k = max(m, n)
    pad_val = 1e9

    cost: list[list[float]] = [[pad_val for _ in range(k)] for _ in range(k)]
    for r in range(m):
        for c in range(n):
            cost[r][c] = float(cost_matrix[r][c])

    # 1-indexed Jonker-Volgenant / Hungarian implementation
    u = [0.0] * (k + 1)
    v = [0.0] * (k + 1)
    p = [0] * (k + 1)
    way = [0] * (k + 1)

    for i in range(1, k + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (k + 1)
        used = [False] * (k + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0

            for j in range(1, k + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            for j in range(k + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    # Extract original row -> col matches
    matches: list[tuple[int, int]] = []
    for j in range(1, k + 1):
        if p[j] > 0 and p[j] <= m and (j - 1) < n:
            r = p[j] - 1
            c = j - 1
            matches.append((r, c))

    matches.sort(key=lambda x: x[0])
    return matches


def solve_association(
    cost_matrix: Sequence[Sequence[float]],
    num_tracks: int,
    num_observations: int,
    max_cost_threshold: float = 1e6,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Perform global minimum-cost assignment and filter out gated/infeasible pairings.

    Args:
        cost_matrix: M x N matrix where rows are tracks and columns are observations.
        num_tracks: Count of candidate tracks (M).
        num_observations: Count of incoming observations (N).
        max_cost_threshold: Threshold above which an assigned pairing is discarded (gated).

    Returns:
        (valid_matches, unmatched_track_indices, unmatched_obs_indices) tuple.
    """
    m = num_tracks
    n = num_observations

    if m == 0:
        return [], [], list(range(n))
    if n == 0:
        return [], list(range(m)), []

    raw_matches = linear_sum_assignment(cost_matrix)

    matched_tracks: set[int] = set()
    matched_obs: set[int] = set()
    valid_matches: list[tuple[int, int]] = []

    for r, c in raw_matches:
        if cost_matrix[r][c] < max_cost_threshold:
            valid_matches.append((r, c))
            matched_tracks.add(r)
            matched_obs.add(c)

    unmatched_tracks = [r for r in range(m) if r not in matched_tracks]
    unmatched_obs = [c for c in range(n) if c not in matched_obs]

    return valid_matches, unmatched_tracks, unmatched_obs

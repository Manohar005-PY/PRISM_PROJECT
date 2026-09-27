"""Deterministic global minimum-cost assignment using pure standard-library Hungarian algorithm."""

from __future__ import annotations

from typing import Sequence

from app.tracking.cost import INF_COST


def linear_sum_assignment(cost_matrix: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    """Solve the linear sum assignment problem min sum C[i, j] using the Hungarian (Munkres) algorithm.

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
    # Find a safe finite pad value (or 0.0)
    pad_val = 1e9

    cost: list[list[float]] = [[pad_val for _ in range(k)] for _ in range(k)]
    for r in range(m):
        for c in range(n):
            cost[r][c] = float(cost_matrix[r][c])

    # 1-indexed Jonker-Volgenant / Hungarian implementation
    # Dual variables u (rows), v (cols)
    u = [0.0] * (k + 1)
    v = [0.0] * (k + 1)
    p = [0] * (k + 1)  # p[j] is row assigned to column j
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


def assign_tracks_to_observations(
    cost_matrix: Sequence[Sequence[float]],
    num_tracks: int | None = None,
    num_observations: int | None = None,
    max_cost_threshold: float = 1e6,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Perform global minimum-cost assignment and filter out gated/infeasible pairings.

    Args:
        cost_matrix: M x N matrix where rows are tracks and columns are observations.
        num_tracks: Explicit count of candidate tracks (M).
        num_observations: Explicit count of incoming observations (N).
        max_cost_threshold: Threshold above which an assigned pairing is discarded (gated).

    Returns:
        (valid_matches, unmatched_track_indices, unmatched_obs_indices) tuple.
    """
    m = num_tracks if num_tracks is not None else len(cost_matrix)
    n = num_observations if num_observations is not None else (len(cost_matrix[0]) if len(cost_matrix) > 0 else 0)

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

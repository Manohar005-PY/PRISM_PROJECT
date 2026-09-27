"""Association cost matrix formulation combining normalized Mahalanobis distance, class compatibility, and track age."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from app.core.types import ObjectState
from app.perception.model import PhysicalObservation
from app.tracking.association import compute_innovation_covariance, mahalanobis_distance_squared
from app.tracking.config import TrackingConfig
from app.tracking.model import TrackRecord

INF_COST: float = 1e9


def compute_pair_cost(
    track: TrackRecord,
    predicted_state: ObjectState | None,
    observation: PhysicalObservation,
    config: TrackingConfig,
    w_distance: float = 0.85,
    w_age: float = 0.15,
) -> tuple[float, float, bool]:
    """Compute association cost for a single (track, observation) pair.

    Returns:
        (cost, d_m2, is_gated) tuple.
    """
    if predicted_state is None:
        return INF_COST, float("inf"), False

    # 1. Class-aware gating
    class_match = (track.class_name == observation.class_name)
    if config.class_aware and not class_match:
        return INF_COST, float("inf"), False

    # 2. Euclidean distance sanity gate
    pred_pos = predicted_state.position
    euclid_dist = math.hypot(observation.position.x - pred_pos.x, observation.position.z - pred_pos.z)
    if euclid_dist > config.max_assignment_distance:
        return INF_COST, float("inf"), False

    # 3. Mahalanobis innovation distance
    S = compute_innovation_covariance(predicted_state.covariance, observation.measurement_covariance)
    d_m2 = mahalanobis_distance_squared(observation.position, pred_pos, S)

    if d_m2 > config.gating_threshold or not math.isfinite(d_m2):
        return INF_COST, d_m2, False

    # 4. Normalized cost components
    # Normalized Mahalanobis distance in [0.0, 1.0]
    c_dist = min(1.0, d_m2 / config.gating_threshold)

    # Track age bonus (slight preference for mature confirmed tracks)
    c_age = max(0.0, 1.0 - min(track.age, 10) / 10.0)

    total_cost = w_distance * c_dist + w_age * c_age
    return total_cost, d_m2, True


def build_cost_matrix(
    tracks: Sequence[TrackRecord],
    predicted_states: Mapping[int, ObjectState],
    observations: Sequence[PhysicalObservation],
    config: TrackingConfig,
) -> list[list[float]]:
    """Build M x N cost matrix for global assignment.

    Args:
        tracks: List of M candidate tracks.
        predicted_states: Mapping from track_id to predicted ObjectState.
        observations: List of N incoming PhysicalObservation proposals.
        config: Tracking configuration.

    Returns:
        M x N list of lists containing association costs or INF_COST if gated out.
    """
    m = len(tracks)
    n = len(observations)

    cost_matrix: list[list[float]] = [[INF_COST for _ in range(n)] for _ in range(m)]

    for i, trk in enumerate(tracks):
        pred_st = predicted_states.get(trk.track_id)
        for j, obs in enumerate(observations):
            cost, _, is_gated = compute_pair_cost(trk, pred_st, obs, config)
            cost_matrix[i][j] = cost if is_gated else INF_COST

    return cost_matrix

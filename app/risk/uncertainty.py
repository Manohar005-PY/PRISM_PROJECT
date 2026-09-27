"""Uncertainty-aware spatial risk formulation and collision probability estimation.

Integrates Kalman state covariance P into geometric collision boundaries to compute
probabilistic risk metrics under measurement uncertainty.
"""

from __future__ import annotations

import math

from app.core.types import Covariance4x4


def extract_position_std(covariance: Covariance4x4 | None) -> float:
    """Extract composite 2D positional standard deviation sqrt(Var(X) + Var(Z)) from 4x4 covariance."""
    if covariance is None:
        return 0.0
    var_x = max(0.0, covariance[0][0])
    var_z = max(0.0, covariance[1][1])
    return math.sqrt(var_x + var_z)


def compute_uncertainty_expanded_radius(
    base_radius: float,
    covariance: Covariance4x4 | None,
    k_sigma: float = 1.0,
) -> float:
    """Expand geometric safety boundary by positional uncertainty standard deviation.

    R_effective = R_base + k_sigma * sigma_pos

    Args:
        base_radius: Deterministic safety radius in meters.
        covariance: 4x4 state covariance matrix (or None).
        k_sigma: Multiplier for positional uncertainty standard deviation.

    Returns:
        Uncertainty-expanded radius in meters.
    """
    if base_radius < 0.0:
        raise ValueError(f"base_radius must be non-negative, got {base_radius}")
    if k_sigma < 0.0:
        raise ValueError(f"k_sigma must be non-negative, got {k_sigma}")

    sigma_pos = extract_position_std(covariance)
    return base_radius + k_sigma * sigma_pos


def compute_collision_probability(
    d_cpa: float,
    collision_radius: float,
    covariance: Covariance4x4 | None,
    epsilon: float = 1e-4,
) -> float:
    """Estimate probability of safety boundary penetration based on CPA distance and spatial uncertainty.

    Uses a normalized Gaussian tail model:
    - If d_cpa <= collision_radius: P = 1.0 (deterministic boundary violation).
    - If d_cpa > collision_radius:
        If covariance is None or zero: P = 0.0 (deterministic miss).
        Else: P = exp( - (d_cpa - collision_radius)^2 / (2 * (sigma_pos^2 + epsilon)) )

    Args:
        d_cpa: Distance at closest point of approach in meters.
        collision_radius: Effective collision boundary radius in meters.
        covariance: 4x4 state covariance matrix.
        epsilon: Numerical stability safeguard.

    Returns:
        Probability in [0.0, 1.0].
    """
    if d_cpa < 0.0:
        raise ValueError(f"d_cpa must be non-negative, got {d_cpa}")
    if collision_radius < 0.0:
        raise ValueError(f"collision_radius must be non-negative, got {collision_radius}")

    # Case 1: Predicted trajectory enters safety boundary deterministically
    if d_cpa <= collision_radius:
        return 1.0

    # Case 2: Near-miss with uncertainty
    sigma_pos = extract_position_std(covariance)
    if sigma_pos <= epsilon:
        return 0.0  # Zero uncertainty implies zero probability of intrusion for d_cpa > radius

    margin = d_cpa - collision_radius
    var_pos = sigma_pos * sigma_pos
    exponent = -(margin * margin) / (2.0 * (var_pos + epsilon))

    return max(0.0, min(1.0, math.exp(exponent)))

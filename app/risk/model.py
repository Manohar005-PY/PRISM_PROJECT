"""Domain models and configuration containers for PRISM Risk Assessment."""

from __future__ import annotations

from dataclasses import dataclass
import math

from app.core.enums import RiskLevel


@dataclass(frozen=True)
class RiskFactors:
    """Individual normalized physical risk factor contributions in [0.0, 1.0].

    Attributes:
        time_factor: Urgency derived from Time-to-Collision (TTC) or Time-to-CPA.
        distance_factor: Threat derived from separation distance at CPA (d_CPA).
        speed_factor: Threat derived from relative closing speed.
        direction_factor: Threat derived from forward spatial alignment with user travel path.
        collision_probability: Probability of safety boundary penetration given covariance P.
    """

    time_factor: float
    distance_factor: float
    speed_factor: float
    direction_factor: float
    collision_probability: float

    def __post_init__(self) -> None:
        for name, val in [
            ("time_factor", self.time_factor),
            ("distance_factor", self.distance_factor),
            ("speed_factor", self.speed_factor),
            ("direction_factor", self.direction_factor),
            ("collision_probability", self.collision_probability),
        ]:
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"{name} must be a finite float, got {val!r}")
            clamped = max(0.0, min(1.0, float(val)))
            object.__setattr__(self, name, clamped)

    def to_dict(self) -> dict[str, float]:
        """Export factors as a string-to-float dictionary."""
        return {
            "time_factor": self.time_factor,
            "distance_factor": self.distance_factor,
            "speed_factor": self.speed_factor,
            "direction_factor": self.direction_factor,
            "collision_probability": self.collision_probability,
        }


@dataclass(frozen=True)
class RiskConfig:
    """Configuration parameters for multi-factor risk assessment.

    Attributes:
        horizon: Evaluation time horizon in seconds.
        max_risk_distance: Distance threshold (meters) beyond which static proximity risk is 0.
        ref_closing_speed: Reference fast closing speed (m/s) for normalizing speed risk.
        k_sigma: Uncertainty multiplier for spatial boundary expansion.
        weight_time: Multi-factor weight for TTC urgency.
        weight_dist: Multi-factor weight for CPA miss distance.
        weight_speed: Multi-factor weight for closing speed.
        weight_dir: Multi-factor weight for forward directional alignment.
        thresh_low: Score threshold for RiskLevel.LOW (in [0.0, 1.0]).
        thresh_medium: Score threshold for RiskLevel.MEDIUM (in [0.0, 1.0]).
        thresh_high: Score threshold for RiskLevel.HIGH (in [0.0, 1.0]).
        thresh_critical: Score threshold for RiskLevel.CRITICAL (in [0.0, 1.0]).
        alpha_smooth: Exponential moving average coefficient in (0.0, 1.0] (1.0 = no smoothing).
    """

    horizon: float = 3.0
    max_risk_distance: float = 4.0
    ref_closing_speed: float = 2.0
    k_sigma: float = 1.0

    weight_time: float = 0.40
    weight_dist: float = 0.30
    weight_speed: float = 0.15
    weight_dir: float = 0.15

    thresh_low: float = 0.20
    thresh_medium: float = 0.45
    thresh_high: float = 0.70
    thresh_critical: float = 0.88

    alpha_smooth: float = 0.60

    def __post_init__(self) -> None:
        for name, val in [
            ("horizon", self.horizon),
            ("max_risk_distance", self.max_risk_distance),
            ("ref_closing_speed", self.ref_closing_speed),
            ("k_sigma", self.k_sigma),
        ]:
            if not isinstance(val, (int, float)) or not math.isfinite(val) or val <= 0.0:
                raise ValueError(f"{name} must be a positive finite float, got {val!r}")

        total_weight = self.weight_time + self.weight_dist + self.weight_speed + self.weight_dir
        if abs(total_weight - 1.0) > 1e-4:
            raise ValueError(f"Risk factor weights must sum to 1.0, got {total_weight}")

        if not (0.0 < self.thresh_low < self.thresh_medium < self.thresh_high < self.thresh_critical <= 1.0):
            raise ValueError("Risk thresholds must be strictly monotonic: 0 < low < medium < high < critical <= 1")

        if not (0.0 < self.alpha_smooth <= 1.0):
            raise ValueError(f"alpha_smooth must be in (0.0, 1.0], got {self.alpha_smooth}")


@dataclass(frozen=True)
class ObjectRiskProfile:
    """Individual threat profile for a single tracked obstacle.

    Attributes:
        object_id: Unique identifier of the tracked entity.
        score: Continuous composite risk score in [0.0, 1.0].
        risk_level: Semantic discrete RiskLevel.
        factors: Breakdown of individual physical factor scores.
        ttc: Time-to-collision in seconds (or None).
        d_cpa: Separation distance at CPA in meters.
        is_approaching: True iff object is currently closing distance toward the user.
    """

    object_id: int
    score: float
    risk_level: RiskLevel
    factors: RiskFactors
    ttc: float | None
    d_cpa: float
    is_approaching: bool

    def __post_init__(self) -> None:
        if not isinstance(self.object_id, int):
            raise TypeError(f"object_id must be int, got {type(self.object_id)}")
        if not isinstance(self.score, (int, float)) or not math.isfinite(self.score):
            raise ValueError(f"score must be a finite float, got {self.score!r}")
        if not isinstance(self.risk_level, RiskLevel):
            raise TypeError(f"risk_level must be a RiskLevel enum, got {type(self.risk_level)}")
        if not isinstance(self.factors, RiskFactors):
            raise TypeError(f"factors must be RiskFactors, got {type(self.factors)}")
        object.__setattr__(self, "score", max(0.0, min(1.0, float(self.score))))

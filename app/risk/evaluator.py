"""Predictive Risk Evaluator implementing PRISM's core RiskEvaluator protocol."""

from __future__ import annotations

from typing import Mapping

from app.core.enums import RiskLevel
from app.core.interfaces import RiskEvaluator
from app.core.types import (
    ObjectState,
    RiskAssessment,
    Trajectory,
    Vector2,
    WorldState,
)
from app.prediction.collision_geometry import (
    CollisionModel,
    closest_point_of_approach,
    time_to_collision,
)
from app.prediction.relative_motion import (
    EgoMotionState,
    compute_relative_motion,
)
from app.risk.model import (
    ObjectRiskProfile,
    RiskConfig,
    RiskFactors,
)
from app.risk.temporal import TemporalRiskTracker
from app.risk.uncertainty import (
    compute_collision_probability,
    compute_uncertainty_expanded_radius,
)


class PredictiveRiskEvaluator:
    """Evaluates collision risk from kinematic trajectories, relative geometry, and state uncertainty.

    Implements the app.core.interfaces.RiskEvaluator protocol.
    """

    def __init__(
        self,
        config: RiskConfig | None = None,
        tracker: TemporalRiskTracker | None = None,
    ) -> None:
        self.config: RiskConfig = config if config is not None else RiskConfig()
        self.tracker: TemporalRiskTracker = (
            tracker if tracker is not None else TemporalRiskTracker(default_alpha=self.config.alpha_smooth)
        )

    def _score_to_risk_level(self, score: float) -> RiskLevel:
        """Map continuous risk score in [0.0, 1.0] to discrete RiskLevel."""
        if score >= self.config.thresh_critical:
            return RiskLevel.CRITICAL
        elif score >= self.config.thresh_high:
            return RiskLevel.HIGH
        elif score >= self.config.thresh_medium:
            return RiskLevel.MEDIUM
        elif score >= self.config.thresh_low:
            return RiskLevel.LOW
        return RiskLevel.SAFE

    def evaluate_object(
        self,
        obj_state: ObjectState,
        ego: EgoMotionState | None = None,
    ) -> ObjectRiskProfile:
        """Evaluate continuous risk score and factors for a single obstacle.

        Args:
            obj_state: Physical state of the obstacle.
            ego: Ego kinematic motion state (defaults to stationary at origin).

        Returns:
            ObjectRiskProfile struct detailing score, risk level, and factor breakdown.
        """
        ego_state = ego if ego is not None else EgoMotionState(Vector2(0.0, 0.0), Vector2(0.0, 0.0))

        rel = compute_relative_motion(
            ego_position=ego_state.position,
            ego_velocity=ego_state.velocity,
            object_position=obj_state.position,
            object_velocity=obj_state.velocity,
        )

        # 1. Uncertainty-expanded safety boundary
        base_model = CollisionModel(object_radius=0.35, ego_radius=0.30, additional_margin=0.35)
        r_eff = compute_uncertainty_expanded_radius(
            base_radius=base_model.effective_radius,
            covariance=obj_state.covariance,
            k_sigma=self.config.k_sigma,
        )

        # 2. Geometric Collision Analysis
        cpa = closest_point_of_approach(rel.relative_position, rel.relative_velocity)
        ttc = time_to_collision(rel.relative_position, rel.relative_velocity, collision_radius=r_eff)
        p_coll = compute_collision_probability(cpa.distance_at_cpa, r_eff, obj_state.covariance)

        # 3. Individual Physical Factor Calculations
        # Factor A: Time Urgency
        if ttc is not None and ttc <= self.config.horizon:
            f_time = max(0.0, 1.0 - (ttc / self.config.horizon))
        elif rel.is_approaching and cpa.time_to_cpa <= self.config.horizon:
            f_time = max(0.0, 1.0 - (cpa.time_to_cpa / self.config.horizon)) * p_coll
        else:
            f_time = 0.0

        # Factor B: Distance Proximity at CPA
        if cpa.distance_at_cpa < self.config.max_risk_distance:
            dist_threat = max(0.0, 1.0 - (cpa.distance_at_cpa / self.config.max_risk_distance))
            # Approaching objects pose higher distance threat than departing
            approach_factor = 1.0 if rel.is_approaching else 0.2
            f_dist = dist_threat * approach_factor
        else:
            f_dist = 0.0

        # Factor C: Closing Speed
        f_speed = min(1.0, max(0.0, rel.closing_speed) / self.config.ref_closing_speed)

        # Factor D: Direction / Spatial Alignment in Forward Travel Sector
        if rel.distance > 0.0:
            # Positive Z_rel means obstacle is ahead in user's forward hemisphere
            cos_theta = rel.relative_position.z / rel.distance
            f_dir = max(0.0, cos_theta)
        else:
            f_dir = 1.0

        # 4. Composite Weighted Score
        raw_score = (
            self.config.weight_time * f_time
            + self.config.weight_dist * f_dist
            + self.config.weight_speed * f_speed
            + self.config.weight_dir * f_dir
        )
        clamped_raw = max(0.0, min(1.0, raw_score))

        # 5. Temporal Smoothing
        smooth_score = self.tracker.update_score(
            object_id=obj_state.object_id,
            raw_score=clamped_raw,
            alpha=self.config.alpha_smooth,
        )

        level = self._score_to_risk_level(smooth_score)

        factors = RiskFactors(
            time_factor=f_time,
            distance_factor=f_dist,
            speed_factor=f_speed,
            direction_factor=f_dir,
            collision_probability=p_coll,
        )

        return ObjectRiskProfile(
            object_id=obj_state.object_id,
            score=smooth_score,
            risk_level=level,
            factors=factors,
            ttc=ttc,
            d_cpa=cpa.distance_at_cpa,
            is_approaching=rel.is_approaching,
        )

    def evaluate(
        self,
        state: ObjectState,
        trajectory: Trajectory,
        timestamp: float,
        ego: EgoMotionState | None = None,
    ) -> RiskAssessment:
        """Assess the collision risk for an object given its state and predicted trajectory.

        Implements the app.core.interfaces.RiskEvaluator Protocol.
        """
        profile = self.evaluate_object(state, ego)
        return RiskAssessment(
            object_id=state.object_id,
            risk_level=profile.risk_level,
            risk_score=profile.score * 100.0,  # Scaled to [0.0, 100.0] domain contract
            ttc=profile.ttc,
            cpa_distance=profile.d_cpa,
            confidence=state.confidence,
            timestamp=timestamp,
        )

    def evaluate_world(
        self,
        world: WorldState,
        ego: EgoMotionState | None = None,
    ) -> list[RiskAssessment]:
        """Evaluate collision risk assessments across all active obstacles in WorldState."""
        active_ids = set(world.objects.keys())
        self.tracker.prune_inactive(active_ids)

        assessments: list[RiskAssessment] = []
        for obj in world.objects.values():
            profile = self.evaluate_object(obj, ego)
            assessments.append(
                RiskAssessment(
                    object_id=obj.object_id,
                    risk_level=profile.risk_level,
                    risk_score=profile.score * 100.0,
                    ttc=profile.ttc,
                    cpa_distance=profile.d_cpa,
                    confidence=obj.confidence,
                    timestamp=world.timestamp,
                )
            )
        return assessments

    def get_primary_threat(
        self,
        world: WorldState,
        ego: EgoMotionState | None = None,
    ) -> RiskAssessment | None:
        """Find and return the highest risk threat in the current WorldState (or None if empty)."""
        assessments = self.evaluate_world(world, ego)
        if not assessments:
            return None
        return max(assessments, key=lambda a: a.risk_score)

"""Factory and pipeline builder for PRISM Part 19 architectural ablation variants."""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

from app.core.enums import RiskLevel
from app.core.interfaces import StateEstimator
from app.core.types import (
    Covariance4x4,
    ObjectState,
    RiskAssessment,
    Trajectory,
    Vector2,
    WorldState,
)
from app.decision.engine import DecisionEngine
from app.perception.adapters.synthetic import SyntheticDetector
from app.perception.pipeline import PerceptionPipeline
from app.prediction.collision_geometry import (
    CollisionModel,
    closest_point_of_approach,
    time_to_collision,
)
from app.prediction.constant_velocity import ConstantVelocityPredictor
from app.prediction.relative_motion import EgoMotionState, compute_relative_motion
from app.risk.evaluator import PredictiveRiskEvaluator
from app.risk.model import ObjectRiskProfile, RiskConfig, RiskFactors
from app.risk.temporal import TemporalRiskTracker
from app.risk.uncertainty import (
    compute_collision_probability,
    compute_uncertainty_expanded_radius,
)
from app.runtime.config import RuntimeConfig
from app.runtime.pipeline import PRISMPipeline
from app.state.kalman import KalmanFilterConfig, KalmanStateEstimator, SingleObjectKalmanFilter
from app.state.noise import ConstantMeasurementNoise, Covariance2x2, MeasurementNoiseModel
from app.state.projection import (
    CameraIntrinsics,
    CameraPose,
    GroundPlaneProjector,
    ProjectionConfig,
)
from app.tracking.config import TrackingConfig
from app.tracking.tracker import MultiObjectTracker
from simulation.ablation.models import AblationVariant


# ============================================================================
# SPECIALIZED ABLATED STATE ESTIMATORS
# ============================================================================

class InstantaneousRawStateEstimator(KalmanStateEstimator):
    """NO_KALMAN State Estimator: Bypasses Kalman temporal dynamics.
    
    Directly reflects instantaneous raw sensor measurements without temporal filtering
    or dynamic velocity propagation (v = (0, 0), P = R).
    """

    def __init__(self) -> None:
        super().__init__(config=KalmanFilterConfig())
        self._raw_states: dict[int, ObjectState] = {}

    def reset(self) -> None:
        super().reset()
        self._raw_states.clear()

    def update_measurement(
        self,
        object_id: int,
        class_id: int | str,
        measurement: Vector2,
        timestamp: float,
        measurement_covariance: Covariance2x2 | None = None,
    ) -> ObjectState:
        cov2 = measurement_covariance if measurement_covariance is not None else ((0.01, 0.0), (0.0, 0.01))
        # Embed 2x2 measurement covariance into 4x4 matrix with large velocity uncertainty
        cov4: Covariance4x4 = (
            (cov2[0][0], cov2[0][1], 0.0, 0.0),
            (cov2[1][0], cov2[1][1], 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        pos_var = cov2[0][0] + cov2[1][1]
        conf = max(0.0, min(1.0, 1.0 / (1.0 + 0.5 * pos_var)))

        st = ObjectState(
            object_id=object_id,
            class_id=class_id,
            position=measurement,
            velocity=Vector2(0.0, 0.0),  # Zero dynamic velocity estimation
            timestamp=timestamp,
            covariance=cov4,
            confidence=conf,
            age=1,
            time_since_update=0.0,
        )
        self._raw_states[object_id] = st
        return st

    def get_state(self, object_id: int) -> ObjectState | None:
        return self._raw_states.get(object_id)

    def prune_stale(self, max_staleness_seconds: float) -> list[int]:
        # No temporal persistence without new measurements
        return []


class PositionOnlyStateEstimator(KalmanStateEstimator):
    """POSITION_ONLY State Estimator: Retains Kalman position filtering but suppresses velocity."""

    def get_state(self, object_id: int) -> ObjectState | None:
        raw_st = super().get_state(object_id)
        if raw_st is None:
            return None
        # Explicitly suppress dynamic velocity to isolate position-only tracking
        return ObjectState(
            object_id=raw_st.object_id,
            class_id=raw_st.class_id,
            position=raw_st.position,
            velocity=Vector2(0.0, 0.0),
            timestamp=raw_st.timestamp,
            covariance=raw_st.covariance,
            confidence=raw_st.confidence,
            age=raw_st.age,
            time_since_update=raw_st.time_since_update,
        )


# ============================================================================
# SPECIALIZED ABLATED RISK EVALUATORS
# ============================================================================

class TTCOnlyRiskEvaluator(PredictiveRiskEvaluator):
    """TTC_ONLY Risk Evaluator: Completely isolates TTC urgency with zero dCPA influence."""

    def evaluate_object(
        self,
        obj_state: ObjectState,
        ego: EgoMotionState | None = None,
    ) -> ObjectRiskProfile:
        ego_state = ego if ego is not None else EgoMotionState(Vector2(0.0, 0.0), Vector2(0.0, 0.0))

        rel = compute_relative_motion(
            ego_position=ego_state.position,
            ego_velocity=ego_state.velocity,
            object_position=obj_state.position,
            object_velocity=obj_state.velocity,
        )

        base_model = CollisionModel(object_radius=0.35, ego_radius=0.30, additional_margin=0.35)
        r_eff = compute_uncertainty_expanded_radius(
            base_radius=base_model.effective_radius,
            covariance=obj_state.covariance,
            k_sigma=self.config.k_sigma,
        )

        cpa = closest_point_of_approach(rel.relative_position, rel.relative_velocity)
        ttc = time_to_collision(rel.relative_position, rel.relative_velocity, collision_radius=r_eff)
        p_coll = compute_collision_probability(cpa.distance_at_cpa, r_eff, obj_state.covariance)

        # Factor A: Time Urgency only
        if ttc is not None and ttc <= self.config.horizon:
            f_time = max(0.0, 1.0 - (ttc / self.config.horizon))
        elif rel.is_approaching and cpa.time_to_cpa <= self.config.horizon:
            f_time = max(0.0, 1.0 - (cpa.time_to_cpa / self.config.horizon)) * p_coll
        else:
            f_time = 0.0

        # Zero out distance proximity factor completely
        f_dist = 0.0

        # Closing Speed & Directional alignment
        f_speed = min(1.0, max(0.0, rel.closing_speed) / self.config.ref_closing_speed)
        f_dir = max(0.0, rel.relative_position.z / rel.distance) if rel.distance > 0.0 else 1.0

        # Re-weighted score without distance factor (0.70 time, 0.15 speed, 0.15 dir)
        raw_score = 0.70 * f_time + 0.15 * f_speed + 0.15 * f_dir
        clamped_raw = max(0.0, min(1.0, raw_score))

        smooth_score = self.tracker.update_score(
            object_id=obj_state.object_id,
            raw_score=clamped_raw,
            alpha=self.config.alpha_smooth,
        )

        level = self._score_to_risk_level(smooth_score)

        factors = RiskFactors(
            time_factor=f_time,
            distance_factor=0.0,
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


class DCPAOnlyRiskEvaluator(PredictiveRiskEvaluator):
    """DCPA_ONLY Risk Evaluator: Completely isolates dCPA miss-distance with zero TTC urgency."""

    def evaluate_object(
        self,
        obj_state: ObjectState,
        ego: EgoMotionState | None = None,
    ) -> ObjectRiskProfile:
        ego_state = ego if ego is not None else EgoMotionState(Vector2(0.0, 0.0), Vector2(0.0, 0.0))

        rel = compute_relative_motion(
            ego_position=ego_state.position,
            ego_velocity=ego_state.velocity,
            object_position=obj_state.position,
            object_velocity=obj_state.velocity,
        )

        base_model = CollisionModel(object_radius=0.35, ego_radius=0.30, additional_margin=0.35)
        r_eff = compute_uncertainty_expanded_radius(
            base_radius=base_model.effective_radius,
            covariance=obj_state.covariance,
            k_sigma=self.config.k_sigma,
        )

        cpa = closest_point_of_approach(rel.relative_position, rel.relative_velocity)
        ttc = time_to_collision(rel.relative_position, rel.relative_velocity, collision_radius=r_eff)
        p_coll = compute_collision_probability(cpa.distance_at_cpa, r_eff, obj_state.covariance)

        # Zero out time urgency factor completely
        f_time = 0.0

        # Factor B: Distance Proximity at CPA
        if cpa.distance_at_cpa < self.config.max_risk_distance:
            dist_threat = max(0.0, 1.0 - (cpa.distance_at_cpa / self.config.max_risk_distance))
            approach_factor = 1.0 if rel.is_approaching else 0.2
            f_dist = dist_threat * approach_factor
        else:
            f_dist = 0.0

        # Closing Speed & Directional alignment
        f_speed = min(1.0, max(0.0, rel.closing_speed) / self.config.ref_closing_speed)
        f_dir = max(0.0, rel.relative_position.z / rel.distance) if rel.distance > 0.0 else 1.0

        # Re-weighted score without time factor (0.70 dist, 0.15 speed, 0.15 dir)
        raw_score = 0.70 * f_dist + 0.15 * f_speed + 0.15 * f_dir
        clamped_raw = max(0.0, min(1.0, raw_score))

        smooth_score = self.tracker.update_score(
            object_id=obj_state.object_id,
            raw_score=clamped_raw,
            alpha=self.config.alpha_smooth,
        )

        level = self._score_to_risk_level(smooth_score)

        factors = RiskFactors(
            time_factor=0.0,
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


class DirectGeometricRiskEvaluator(PredictiveRiskEvaluator):
    """NO_PREDICTIVE_RISK: Direct geometric thresholding on predicted position without continuous multi-factor risk engine."""

    def evaluate_object(
        self,
        obj_state: ObjectState,
        ego: EgoMotionState | None = None,
    ) -> ObjectRiskProfile:
        ego_state = ego if ego is not None else EgoMotionState(Vector2(0.0, 0.0), Vector2(0.0, 0.0))

        rel = compute_relative_motion(
            ego_position=ego_state.position,
            ego_velocity=ego_state.velocity,
            object_position=obj_state.position,
            object_velocity=obj_state.velocity,
        )

        cpa = closest_point_of_approach(rel.relative_position, rel.relative_velocity)
        dist = rel.distance

        # Direct geometric thresholding: If current separation <= 1.5m or dCPA <= 0.8m within 3s -> HIGH risk
        is_threat = (dist <= 1.5) or (cpa.distance_at_cpa <= 0.8 and 0.0 < cpa.time_to_cpa <= self.config.horizon)

        score = 0.85 if is_threat else (0.40 if dist <= 3.0 else 0.0)
        level = RiskLevel.HIGH if is_threat else (RiskLevel.MEDIUM if dist <= 3.0 else RiskLevel.SAFE)

        factors = RiskFactors(
            time_factor=1.0 if is_threat else 0.0,
            distance_factor=1.0 if is_threat else 0.0,
            speed_factor=0.0,
            direction_factor=0.0,
            collision_probability=1.0 if is_threat else 0.0,
        )

        return ObjectRiskProfile(
            object_id=obj_state.object_id,
            score=score,
            risk_level=level,
            factors=factors,
            ttc=cpa.time_to_cpa if is_threat else None,
            d_cpa=cpa.distance_at_cpa,
            is_approaching=rel.is_approaching,
        )


# ============================================================================
# VARIANT PIPELINE FACTORY
# ============================================================================

def create_variant_pipeline(
    variant: AblationVariant | str,
    custom_risk_config: RiskConfig | None = None,
    custom_kalman_config: KalmanFilterConfig | None = None,
    prediction_horizon: float = 3.0,
) -> PRISMPipeline:
    """Instantiate a fully configured PRISMPipeline for the specified ablation variant.
    
    Args:
        variant: AblationVariant identifier.
        custom_risk_config: Optional custom RiskConfig for sensitivity sweeps.
        custom_kalman_config: Optional custom KalmanFilterConfig for sensitivity sweeps.
        prediction_horizon: Prediction lookahead window in seconds.
        
    Returns:
        Configured PRISMPipeline ready for step-by-step stream processing.
    """
    v_enum = AblationVariant(variant)

    intrinsics = CameraIntrinsics(640, 480, 500.0, 500.0, 320.0, 240.0)
    pose = CameraPose(1.2, 0.15)
    proj_cfg = ProjectionConfig(intrinsics=intrinsics, camera_pose=pose)
    projector = GroundPlaneProjector(proj_cfg)
    detector = SyntheticDetector(config=proj_cfg, pixel_noise_sigma=0.0, random_seed=42)
    perc_pipe = PerceptionPipeline(detector, projector)

    k_cfg = custom_kalman_config if custom_kalman_config is not None else KalmanFilterConfig()
    r_cfg = custom_risk_config if custom_risk_config is not None else RiskConfig(alpha_smooth=0.50)

    # 1. State Estimator selection
    if v_enum == AblationVariant.NO_KALMAN:
        state_estimator: KalmanStateEstimator = InstantaneousRawStateEstimator()
    elif v_enum == AblationVariant.POSITION_ONLY:
        state_estimator = PositionOnlyStateEstimator(config=k_cfg)
    else:
        state_estimator = KalmanStateEstimator(config=k_cfg)

    # 2. Multi-Object Tracker
    tracker = MultiObjectTracker(
        TrackingConfig(confirmation_hits=2, gating_threshold=9.21),
        state_estimator=state_estimator,
    )

    # 3. Trajectory Predictor
    predictor = ConstantVelocityPredictor()

    # 4. Risk Evaluator selection
    if v_enum == AblationVariant.TTC_ONLY:
        risk_evaluator: PredictiveRiskEvaluator = TTCOnlyRiskEvaluator(config=r_cfg)
    elif v_enum == AblationVariant.DCPA_ONLY:
        risk_evaluator = DCPAOnlyRiskEvaluator(config=r_cfg)
    elif v_enum == AblationVariant.NO_PREDICTIVE_RISK:
        risk_evaluator = DirectGeometricRiskEvaluator(config=r_cfg)
    else:
        risk_evaluator = PredictiveRiskEvaluator(config=r_cfg)

    # 5. Decision Engine
    decision_engine = DecisionEngine()

    return PRISMPipeline(
        perception_pipeline=perc_pipe,
        tracker=tracker,
        state_estimator=state_estimator,
        predictor=predictor,
        risk_evaluator=risk_evaluator,
        decision_engine=decision_engine,
        config=RuntimeConfig(prediction_horizon=prediction_horizon),
    )

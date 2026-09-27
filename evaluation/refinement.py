"""Refined PRISM pipeline builder implementing calibrated parameters without modifying production defaults."""

from __future__ import annotations

from app.decision.engine import DecisionEngine
from app.perception.adapters.synthetic import SyntheticDetector
from app.perception.pipeline import PerceptionPipeline
from app.prediction.constant_velocity import ConstantVelocityPredictor
from app.risk.evaluator import PredictiveRiskEvaluator
from app.risk.model import RiskConfig
from app.runtime.config import RuntimeConfig
from app.runtime.pipeline import PRISMPipeline
from app.state.kalman import KalmanFilterConfig, KalmanStateEstimator
from app.state.projection import CameraIntrinsics, CameraPose, GroundPlaneProjector, ProjectionConfig
from app.tracking.config import TrackingConfig
from app.tracking.tracker import MultiObjectTracker
from evaluation.baselines import PRISMEvaluator
from evaluation.refinement_protocol import RefinementConfig


def create_refined_prism_evaluator(
    config: RefinementConfig | None = None,
    random_seed: int = 42,
) -> PRISMEvaluator:
    """Construct a PRISMEvaluator configured with calibrated refinement parameters."""
    cfg = config or RefinementConfig()

    intrinsics = CameraIntrinsics(640, 480, 500.0, 500.0, 320.0, 240.0)
    pose = CameraPose(1.2, 0.15)
    proj_cfg = ProjectionConfig(intrinsics=intrinsics, camera_pose=pose)
    projector = GroundPlaneProjector(proj_cfg)

    detector = SyntheticDetector(config=proj_cfg, pixel_noise_sigma=1.0, random_seed=random_seed)
    perc_pipe = PerceptionPipeline(detector, projector)

    state_estimator = KalmanStateEstimator(KalmanFilterConfig())
    tracker = MultiObjectTracker(TrackingConfig(confirmation_hits=2), state_estimator=state_estimator)

    predictor = ConstantVelocityPredictor()

    risk_cfg = RiskConfig(
        horizon=cfg.horizon,
        max_risk_distance=cfg.max_risk_distance,
        ref_closing_speed=cfg.ref_closing_speed,
        k_sigma=cfg.k_sigma,
        weight_time=cfg.weight_time,
        weight_dist=cfg.weight_dist,
        weight_speed=cfg.weight_speed,
        weight_dir=cfg.weight_dir,
        thresh_low=cfg.thresh_low,
        thresh_medium=cfg.thresh_medium,
        thresh_high=cfg.thresh_high,
        thresh_critical=cfg.thresh_critical,
        alpha_smooth=cfg.alpha_smooth,
    )
    risk_evaluator = PredictiveRiskEvaluator(risk_cfg)
    decision_engine = DecisionEngine()

    pipeline = PRISMPipeline(
        perception_pipeline=perc_pipe,
        tracker=tracker,
        state_estimator=state_estimator,
        predictor=predictor,
        risk_evaluator=risk_evaluator,
        decision_engine=decision_engine,
        config=RuntimeConfig(prediction_horizon=cfg.horizon),
    )
    return PRISMEvaluator(pipeline)

"""Evaluation runner executing closed-loop benchmarks and ablation studies across scenarios."""

from __future__ import annotations

import time
from typing import Mapping, Sequence

from app.core.types import Vector2
from app.decision.engine import DecisionEngine
from app.perception.adapters.synthetic import SyntheticDetector
from app.perception.model import PhysicalObservation
from app.perception.pipeline import PerceptionPipeline
from app.prediction.constant_velocity import ConstantVelocityPredictor
from app.prediction.relative_motion import EgoMotionState
from app.risk.evaluator import PredictiveRiskEvaluator
from app.risk.model import RiskConfig
from app.runtime.config import RuntimeConfig
from app.runtime.pipeline import PRISMPipeline
from app.state.kalman import KalmanFilterConfig, KalmanStateEstimator
from app.state.projection import CameraIntrinsics, CameraPose, GroundPlaneProjector, ProjectionConfig
from app.tracking.config import TrackingConfig
from app.tracking.tracker import MultiObjectTracker
from evaluation.baselines import DistanceBaseline, PRISMEvaluator, SystemEvaluator, TTCBaseline
from evaluation.ground_truth import SyntheticGroundTruthProvider
from evaluation.models import DecisionRecord, EvaluationSample, GroundTruthState, PredictionRecord, RiskRecord
from simulation.noise import GaussianNoiseModel
from simulation.scenarios import Scenario
from simulation.simulator import Simulator


def create_prism_evaluator_system(
    enable_smoothing: bool = True,
    enable_prediction: bool = True,
    enable_uncertainty: bool = True,
    enable_hysteresis: bool = True,
) -> PRISMEvaluator:
    """Construct a PRISMEvaluator instance with optional component ablation flags."""
    intrinsics = CameraIntrinsics(640, 480, 500.0, 500.0, 320.0, 240.0)
    pose = CameraPose(1.2, 0.15)
    proj_cfg = ProjectionConfig(intrinsics=intrinsics, camera_pose=pose)
    projector = GroundPlaneProjector(proj_cfg)

    detector = SyntheticDetector(config=proj_cfg, pixel_noise_sigma=1.0, random_seed=42)
    perc_pipe = PerceptionPipeline(detector, projector)

    state_estimator = KalmanStateEstimator(KalmanFilterConfig())
    tracker = MultiObjectTracker(TrackingConfig(confirmation_hits=2), state_estimator=state_estimator)

    predictor = ConstantVelocityPredictor() if enable_prediction else None
    risk_cfg = RiskConfig(alpha_smooth=0.4 if enable_smoothing else 1.0)
    risk_evaluator = PredictiveRiskEvaluator(risk_cfg)
    decision_engine = DecisionEngine()

    pipeline = PRISMPipeline(
        perception_pipeline=perc_pipe,
        tracker=tracker,
        state_estimator=state_estimator,
        predictor=predictor,
        risk_evaluator=risk_evaluator,
        decision_engine=decision_engine,
        config=RuntimeConfig(),
    )
    return PRISMEvaluator(pipeline)


class EvaluationRunner:
    """Executes closed-loop evaluation runs across scenarios and records synchronized evaluation samples."""

    def __init__(
        self,
        noise_sigma_x: float = 0.05,
        noise_sigma_z: float = 0.05,
        random_seed: int = 42,
    ) -> None:
        self.noise_model = GaussianNoiseModel(
            sigma_x=noise_sigma_x,
            sigma_z=noise_sigma_z,
            seed=random_seed,
        )

    def run_scenario(
        self,
        scenario: Scenario,
        evaluator: SystemEvaluator,
    ) -> list[EvaluationSample]:
        """Execute a single scenario through simulator and target evaluator."""
        sim = Simulator(scenario=scenario, noise_model=self.noise_model)
        gt_seq, _ = sim.run()

        gt_provider = SyntheticGroundTruthProvider(gt_seq)
        evaluator.reset()

        samples: list[EvaluationSample] = []

        for frame_idx in range(gt_seq.num_frames):
            frame = gt_seq.get_frame(frame_idx)
            t = frame.timestamp
            ego_st = EgoMotionState(position=frame.ego.position, velocity=frame.ego.velocity, timestamp=t)

            # Build relative observations from camera perspective
            observations: list[PhysicalObservation] = []
            for obj in frame.objects:
                rel_pos = Vector2(
                    x=obj.position.x - frame.ego.position.x,
                    z=obj.position.z - frame.ego.position.z,
                )
                if rel_pos.z > 0.0:  # Only objects in front of camera
                    observations.append(
                        PhysicalObservation(
                            position=rel_pos,
                            measurement_covariance=((0.04, 0.0), (0.0, 0.04)),
                            confidence=0.95,
                            timestamp=t,
                            detection_id=f"obj_{obj.object_id}",
                            class_name=str(obj.class_id),
                        )
                    )

            # Ground truth states
            gt_states = gt_provider.get_state(t)
            is_haz = gt_provider.is_hazard(t)

            t_start = time.perf_counter()
            risk_rec, dec_rec = evaluator.process(observations, t, ego_state=ego_st)
            t_end = time.perf_counter()
            lat_ms = (t_end - t_start) * 1000.0

            # Generate prediction records from current observations
            preds: dict[int, PredictionRecord] = {}
            for obs in observations:
                preds[1] = PredictionRecord(
                    timestamp=t,
                    object_id=1,
                    predicted_position=obs.position,
                    predicted_velocity=Vector2(0.0, 0.0),
                    prediction_horizon=1.0,
                )

            sample = EvaluationSample(
                timestamp=t,
                frame_id=frame_idx + 1,
                ground_truth=gt_states,
                is_scene_hazard=is_haz,
                predictions=preds,
                risk=risk_rec,
                decision=dec_rec,
                processing_latency_ms=lat_ms,
            )
            samples.append(sample)

        return samples

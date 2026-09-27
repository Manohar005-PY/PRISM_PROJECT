"""Robustness testing: Evaluates PRISM resilience under noise and detection drop perturbations."""

from __future__ import annotations

from dataclasses import dataclass
import random
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
from evaluation.baselines import PRISMEvaluator
from evaluation.ground_truth import SyntheticGroundTruthProvider
from evaluation.metrics import (
    compute_decision_switch_rate,
    compute_false_alarm_rate,
    compute_hazard_detection_rate,
    compute_position_rmse,
    compute_unnecessary_stop_rate,
)
from evaluation.models import DecisionRecord, EvaluationSample, PredictionRecord, RiskRecord
from evaluation.scenarios import get_canonical_benchmark_suite
from simulation.noise import GaussianNoiseModel
from simulation.scenarios import Scenario
from simulation.simulator import Simulator


@dataclass(frozen=True)
class RobustnessResult:
    """Quantitative performance degradation metrics across perturbation levels."""

    perturbation_type: str
    perturbation_levels: tuple[float, ...]
    metrics_by_level: Mapping[float, Mapping[str, float]]


class RobustnessAnalyzer:
    """Evaluates degradation characteristics of PRISM under simulated faults and noise."""

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed

    def evaluate_detection_drop_robustness(
        self,
        drop_rates: Sequence[float] = (0.0, 0.05, 0.10, 0.20, 0.30),
        scenarios: Mapping[str, Scenario] | None = None,
    ) -> RobustnessResult:
        """Measure performance degradation under frame detection dropout."""
        sc_dict = scenarios or get_canonical_benchmark_suite()
        results: dict[float, dict[str, float]] = {}

        for drop_rate in drop_rates:
            evaluator = self._create_pipeline()
            all_samples = []

            for sc in sc_dict.values():
                samples = self._run_scenario_with_drop(sc, evaluator, drop_rate)
                all_samples.extend(samples)

            results[drop_rate] = {
                "hazard_detection_rate": compute_hazard_detection_rate(all_samples),
                "false_alarm_rate": compute_false_alarm_rate(all_samples),
                "unnecessary_stop_rate": compute_unnecessary_stop_rate(all_samples),
                "position_rmse": compute_position_rmse(all_samples),
            }

        return RobustnessResult("detection_drop_rate", tuple(drop_rates), results)

    def _run_scenario_with_drop(
        self,
        scenario: Scenario,
        evaluator: PRISMEvaluator,
        drop_rate: float,
    ) -> list[EvaluationSample]:
        sim = Simulator(scenario=scenario, noise_model=GaussianNoiseModel(0.05, 0.05, self.random_seed))
        gt_seq, _ = sim.run()
        gt_provider = SyntheticGroundTruthProvider(gt_seq)
        evaluator.reset()

        rng = random.Random(self.random_seed)
        samples: list[EvaluationSample] = []

        for frame_idx in range(gt_seq.num_frames):
            frame = gt_seq.get_frame(frame_idx)
            t = frame.timestamp
            ego_st = EgoMotionState(position=frame.ego.position, velocity=frame.ego.velocity, timestamp=t)

            observations: list[PhysicalObservation] = []
            for obj in frame.objects:
                # Stochastic frame dropout simulation
                if rng.random() >= drop_rate:
                    rel_pos = Vector2(
                        x=obj.position.x - frame.ego.position.x,
                        z=obj.position.z - frame.ego.position.z,
                    )
                    if rel_pos.z > 0.0:
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

            gt_states = gt_provider.get_state(t)
            is_haz = gt_provider.is_hazard(t)

            risk_rec, dec_rec = evaluator.process(observations, t, ego_state=ego_st)

            preds: dict[int, PredictionRecord] = {}
            for obs in observations:
                preds[1] = PredictionRecord(
                    timestamp=t,
                    object_id=1,
                    predicted_position=obs.position,
                    predicted_velocity=Vector2(0.0, 0.0),
                    prediction_horizon=1.0,
                )

            samples.append(
                EvaluationSample(
                    timestamp=t,
                    frame_id=frame_idx + 1,
                    ground_truth=gt_states,
                    is_scene_hazard=is_haz,
                    predictions=preds,
                    risk=risk_rec,
                    decision=dec_rec,
                    processing_latency_ms=0.5,
                )
            )

        return samples

    def _create_pipeline(self) -> PRISMEvaluator:
        intrinsics = CameraIntrinsics(640, 480, 500.0, 500.0, 320.0, 240.0)
        pose = CameraPose(1.2, 0.15)
        proj_cfg = ProjectionConfig(intrinsics=intrinsics, camera_pose=pose)
        projector = GroundPlaneProjector(proj_cfg)

        detector = SyntheticDetector(config=proj_cfg, pixel_noise_sigma=1.0, random_seed=self.random_seed)
        perc_pipe = PerceptionPipeline(detector, projector)

        state_estimator = KalmanStateEstimator(KalmanFilterConfig())
        tracker = MultiObjectTracker(TrackingConfig(confirmation_hits=2), state_estimator=state_estimator)

        predictor = ConstantVelocityPredictor()
        risk_evaluator = PredictiveRiskEvaluator(RiskConfig(alpha_smooth=0.4))
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

"""Controlled ablation study: Evaluates systematic removal of individual algorithmic components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

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
from evaluation.metrics import (
    compute_decision_switch_rate,
    compute_false_alarm_rate,
    compute_hazard_detection_rate,
    compute_unnecessary_stop_rate,
)
from evaluation.runner import EvaluationRunner
from evaluation.scenarios import get_canonical_benchmark_suite
from simulation.scenarios import Scenario


@dataclass(frozen=True)
class AblationMetrics:
    """Performance metrics resulting from an isolated component ablation."""

    ablation_id: str
    description: str
    hazard_detection_rate: float
    false_alarm_rate: float
    unnecessary_stop_rate: float
    switch_rate: float


class ControlledAblationStudy:
    """Executes systematic ablations across PRISM subsystems."""

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed
        self.runner = EvaluationRunner(random_seed=random_seed)

    def run_all_ablations(
        self,
        scenarios: Mapping[str, Scenario] | None = None,
    ) -> list[AblationMetrics]:
        """Execute complete suite of component ablations."""
        sc_dict = scenarios or get_canonical_benchmark_suite()

        ablations = [
            ("FULL_PRISM", "Full PRISM pipeline (all components active)", {}),
            ("A_NO_SMOOTHING", "No temporal risk smoothing (alpha=1.0)", {"alpha_smooth": 1.0}),
            ("B_NO_UNCERTAINTY", "No covariance safety margin expansion (k_sigma=0.0)", {"k_sigma": 0.001}),
            ("C_NO_DIRECTION", "No forward directional weighting (weight_dir=0.0)", {"weight_dir": 0.0, "weight_time": 0.45, "weight_dist": 0.35, "weight_speed": 0.20}),
            ("D_NO_SPEED", "No closing-speed weighting (weight_speed=0.0)", {"weight_speed": 0.0, "weight_time": 0.45, "weight_dist": 0.35, "weight_dir": 0.20}),
            ("E_NO_CPA", "No CPA miss-distance weighting (weight_dist=0.0)", {"weight_dist": 0.0, "weight_time": 0.55, "weight_speed": 0.25, "weight_dir": 0.20}),
            ("F_NO_TTC", "No TTC urgency weighting (weight_time=0.0)", {"weight_time": 0.0, "weight_dist": 0.50, "weight_speed": 0.25, "weight_dir": 0.25}),
            ("G_NO_PREDICTION", "No trajectory extrapolation horizon (horizon=0.01s)", {"horizon": 0.01}),
        ]

        results: list[AblationMetrics] = []
        for ab_id, desc, cfg_overrides in ablations:
            evaluator = self._create_ablated_pipeline(**cfg_overrides)
            all_samples = []
            for sc in sc_dict.values():
                samples = self.runner.run_scenario(sc, evaluator)
                all_samples.extend(samples)

            results.append(
                AblationMetrics(
                    ablation_id=ab_id,
                    description=desc,
                    hazard_detection_rate=compute_hazard_detection_rate(all_samples),
                    false_alarm_rate=compute_false_alarm_rate(all_samples),
                    unnecessary_stop_rate=compute_unnecessary_stop_rate(all_samples),
                    switch_rate=compute_decision_switch_rate(all_samples),
                )
            )

        return results

    def _create_ablated_pipeline(
        self,
        alpha_smooth: float = 0.40,
        k_sigma: float = 1.0,
        weight_time: float = 0.40,
        weight_dist: float = 0.30,
        weight_speed: float = 0.15,
        weight_dir: float = 0.15,
        horizon: float = 3.0,
    ) -> PRISMEvaluator:
        intrinsics = CameraIntrinsics(640, 480, 500.0, 500.0, 320.0, 240.0)
        pose = CameraPose(1.2, 0.15)
        proj_cfg = ProjectionConfig(intrinsics=intrinsics, camera_pose=pose)
        projector = GroundPlaneProjector(proj_cfg)

        detector = SyntheticDetector(config=proj_cfg, pixel_noise_sigma=1.0, random_seed=self.random_seed)
        perc_pipe = PerceptionPipeline(detector, projector)

        state_estimator = KalmanStateEstimator(KalmanFilterConfig())
        tracker = MultiObjectTracker(TrackingConfig(confirmation_hits=2), state_estimator=state_estimator)

        predictor = ConstantVelocityPredictor()

        # Re-normalize weights to 1.0 if modified
        w_sum = weight_time + weight_dist + weight_speed + weight_dir
        if w_sum > 0.0:
            wt = weight_time / w_sum
            wd = weight_dist / w_sum
            ws = weight_speed / w_sum
            wr = weight_dir / w_sum
        else:
            wt, wd, ws, wr = 0.25, 0.25, 0.25, 0.25

        risk_cfg = RiskConfig(
            horizon=horizon,
            alpha_smooth=alpha_smooth,
            k_sigma=k_sigma,
            weight_time=wt,
            weight_dist=wd,
            weight_speed=ws,
            weight_dir=wr,
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
            config=RuntimeConfig(prediction_horizon=horizon),
        )
        return PRISMEvaluator(pipeline)

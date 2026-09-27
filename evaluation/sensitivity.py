"""Sensitivity analysis: Evaluates PRISM performance under systematic parameter variations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from app.core.types import Vector2
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
class SensitivityResult:
    """Quantitative performance metrics across parameter variation levels."""

    parameter_name: str
    parameter_values: tuple[float, ...]
    metrics_by_value: Mapping[float, Mapping[str, float]]


class SensitivityAnalyzer:
    """Performs controlled sensitivity sweeps over PRISM configuration parameters."""

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed
        self.runner = EvaluationRunner(random_seed=random_seed)

    def evaluate_horizon_sensitivity(
        self,
        horizons: Sequence[float] = (1.0, 1.5, 2.0, 3.0, 4.0),
        scenarios: Mapping[str, Scenario] | None = None,
    ) -> SensitivityResult:
        """Evaluate performance impact of varying trajectory prediction horizons."""
        sc_dict = scenarios or get_canonical_benchmark_suite()
        results: dict[float, dict[str, float]] = {}

        for h in horizons:
            evaluator = self._create_pipeline(prediction_horizon=h)
            all_samples = []
            for sc in sc_dict.values():
                samples = self.runner.run_scenario(sc, evaluator)
                all_samples.extend(samples)

            results[h] = {
                "hazard_detection_rate": compute_hazard_detection_rate(all_samples),
                "false_alarm_rate": compute_false_alarm_rate(all_samples),
                "unnecessary_stop_rate": compute_unnecessary_stop_rate(all_samples),
                "switch_rate": compute_decision_switch_rate(all_samples),
            }

        return SensitivityResult("prediction_horizon", tuple(horizons), results)

    def evaluate_alpha_smoothing_sensitivity(
        self,
        alphas: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 1.0),
        scenarios: Mapping[str, Scenario] | None = None,
    ) -> SensitivityResult:
        """Evaluate performance impact of varying temporal risk smoothing alpha."""
        sc_dict = scenarios or get_canonical_benchmark_suite()
        results: dict[float, dict[str, float]] = {}

        for a in alphas:
            evaluator = self._create_pipeline(alpha_smooth=a)
            all_samples = []
            for sc in sc_dict.values():
                samples = self.runner.run_scenario(sc, evaluator)
                all_samples.extend(samples)

            results[a] = {
                "hazard_detection_rate": compute_hazard_detection_rate(all_samples),
                "false_alarm_rate": compute_false_alarm_rate(all_samples),
                "unnecessary_stop_rate": compute_unnecessary_stop_rate(all_samples),
                "switch_rate": compute_decision_switch_rate(all_samples),
            }

        return SensitivityResult("temporal_alpha_smooth", tuple(alphas), results)

    def _create_pipeline(
        self,
        prediction_horizon: float = 3.0,
        alpha_smooth: float = 0.4,
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
        risk_evaluator = PredictiveRiskEvaluator(RiskConfig(alpha_smooth=alpha_smooth))
        decision_engine = DecisionEngine()

        pipeline = PRISMPipeline(
            perception_pipeline=perc_pipe,
            tracker=tracker,
            state_estimator=state_estimator,
            predictor=predictor,
            risk_evaluator=risk_evaluator,
            decision_engine=decision_engine,
            config=RuntimeConfig(prediction_horizon=prediction_horizon),
        )
        return PRISMEvaluator(pipeline)

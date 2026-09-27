"""End-to-end runtime integration tests verifying the full chain across multi-frame scenarios."""

import unittest

from app.core.types import Vector2
from app.decision.engine import DecisionEngine
from app.decision.model import NavigationAction
from app.perception.adapters.synthetic import SyntheticDetector
from app.perception.pipeline import PerceptionPipeline
from app.prediction.constant_velocity import ConstantVelocityPredictor
from app.prediction.relative_motion import EgoMotionState
from app.risk.evaluator import PredictiveRiskEvaluator
from app.risk.model import RiskConfig
from app.runtime.config import RuntimeConfig
from app.runtime.health import RuntimeStatus
from app.runtime.pipeline import PRISMPipeline
from app.runtime.runner import PRISMRunner
from app.state.kalman import KalmanFilterConfig, KalmanStateEstimator
from app.state.projection import CameraIntrinsics, CameraPose, GroundPlaneProjector, ProjectionConfig
from app.tracking.config import TrackingConfig
from app.tracking.tracker import MultiObjectTracker
from simulation.noise import GaussianNoiseModel
from simulation.scenarios import create_head_on_scenario
from simulation.simulator import Simulator


class TestRuntimeIntegration(unittest.TestCase):
    """Integration test suite executing closed-loop runtime pipeline on simulation scenarios."""

    def setUp(self) -> None:
        intrinsics = CameraIntrinsics(640, 480, 500.0, 500.0, 320.0, 240.0)
        pose = CameraPose(1.2, 0.15)
        self.proj_cfg = ProjectionConfig(intrinsics=intrinsics, camera_pose=pose)
        self.projector = GroundPlaneProjector(self.proj_cfg)
        self.detector = SyntheticDetector(config=self.proj_cfg, pixel_noise_sigma=1.0, random_seed=42)
        self.perc_pipe = PerceptionPipeline(self.detector, self.projector)

        self.state_estimator = KalmanStateEstimator(KalmanFilterConfig())
        self.tracker = MultiObjectTracker(
            config=TrackingConfig(confirmation_hits=2),
            state_estimator=self.state_estimator,
        )
        self.pipeline = PRISMPipeline(
            perception_pipeline=self.perc_pipe,
            tracker=self.tracker,
            state_estimator=self.state_estimator,
            predictor=ConstantVelocityPredictor(),
            risk_evaluator=PredictiveRiskEvaluator(RiskConfig()),
            decision_engine=DecisionEngine(),
            config=RuntimeConfig(),
        )

    def test_approaching_obstacle_head_on_escalation(self) -> None:
        """Verify that a head-on approaching collision escalates decision appropriately."""
        scenario = create_head_on_scenario(duration=3.5, timestep=0.1)
        sim = Simulator(scenario=scenario, noise_model=GaussianNoiseModel(0.0, 0.0))
        gt_seq, _ = sim.run()

        runner = PRISMRunner(self.pipeline)

        frames = []
        for i in range(gt_seq.num_frames):
            fr = gt_seq.get_frame(i)
            sim_entities = [(obj.object_id, obj.class_id, obj.position) for obj in fr.objects]
            frames.append((sim_entities, fr.timestamp))

        results = runner.run_sequence(frames)
        self.assertEqual(len(results), gt_seq.num_frames)

        initial_dec = results[0].decision.action
        final_dec = results[-1].decision.action

        self.assertIn(initial_dec, (NavigationAction.CONTINUE, NavigationAction.CAUTION))
        self.assertIn(
            final_dec,
            (
                NavigationAction.STOP,
                NavigationAction.EMERGENCY_STOP,
                NavigationAction.SLOW_DOWN,
                NavigationAction.TURN_LEFT,
                NavigationAction.TURN_RIGHT,
            ),
        )
        self.assertEqual(results[-1].runtime_status, RuntimeStatus.HEALTHY)


if __name__ == "__main__":
    unittest.main()

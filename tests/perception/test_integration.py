"""End-to-end integration tests: Perception -> Ground Projection -> Association (TrackedObservation) -> Kalman -> Prediction -> Risk -> Decision."""

import unittest

from app.core.enums import RiskLevel
from app.core.types import ObjectObservation, Vector2, WorldState
from app.decision.context import DecisionContext
from app.decision.engine import DecisionEngine
from app.decision.model import NavigationAction
from app.perception.adapters.synthetic import SyntheticDetector
from app.perception.metrics import position_rmse
from app.perception.model import TrackedObservation
from app.perception.pipeline import PerceptionPipeline
from app.prediction.relative_motion import EgoMotionState
from app.risk.evaluator import PredictiveRiskEvaluator
from app.risk.model import RiskConfig
from app.state.kalman import KalmanFilterConfig, KalmanStateEstimator
from app.state.noise import ConstantMeasurementNoise
from app.state.projection import (
    CameraIntrinsics,
    CameraPose,
    GroundPlaneProjector,
    ProjectionConfig,
)
from simulation.noise import GaussianNoiseModel
from simulation.scenarios import create_head_on_scenario
from simulation.simulator import Simulator


class TestEndToEndPerceptionPipeline(unittest.TestCase):
    """Integration test verifying closed-loop data flow from perception to decision."""

    def setUp(self) -> None:
        self.intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, image_width=640, image_height=480)
        self.pose = CameraPose(height=1.2, pitch=0.15)
        self.config = ProjectionConfig(intrinsics=self.intrinsics, camera_pose=self.pose)
        self.projector = GroundPlaneProjector(self.config)
        self.detector = SyntheticDetector(config=self.config, pixel_noise_sigma=2.0, random_seed=42)
        self.pipeline = PerceptionPipeline(self.detector, self.projector)

        self.kalman_estimator = KalmanStateEstimator(
            config=KalmanFilterConfig(),
            noise_model=ConstantMeasurementNoise(0.15, 0.25),
        )
        self.risk_evaluator = PredictiveRiskEvaluator(RiskConfig(alpha_smooth=1.0))
        self.decision_engine = DecisionEngine()

    def test_perception_to_decision_pipeline_head_on(self) -> None:
        """Full pipeline execution: Noisy 2D detections -> Projected PhysicalObservation (X,Z,R) -> Explicit Association (TrackedObservation) -> Kalman -> Prediction -> Risk -> Decision."""
        scenario = create_head_on_scenario(duration=3.0, timestep=0.1)
        gt_seq, _ = Simulator(scenario=scenario, noise_model=GaussianNoiseModel(0.0, 0.0)).run()

        raw_positions: list[Vector2] = []
        kalman_positions: list[Vector2] = []
        ground_truth_positions: list[Vector2] = []
        decisions = []

        for i in range(gt_seq.num_frames):
            frame = gt_seq.get_frame(i)
            gt_obj = frame.objects[0]
            gt_pos = gt_obj.position
            ground_truth_positions.append(gt_pos)

            # Step 1: Perception generates 2D detections with injected pixel noise (Detection with ephemeral detection_id)
            synthetic_frame = [(gt_obj.object_id, gt_obj.class_id, gt_pos)]
            raw_detections = self.detector.detect(synthetic_frame, frame.timestamp)
            self.assertEqual(len(raw_detections), 1)

            # Step 2: Perception pipeline projects detection to physical ground observation (X, Z, R)
            obs_list, rejections = self.pipeline.process_frame(synthetic_frame, frame.timestamp)
            self.assertEqual(len(obs_list), 1)
            raw_positions.append(obs_list[0].position)

            # Step 3: Explicit Downstream Association layer creates TrackedObservation (bridging perception to tracking)
            tracked_obs = TrackedObservation(track_id=gt_obj.object_id, observation=obs_list[0])

            # Step 4: Part 4 Kalman State Estimator consumes TrackedObservation
            estimated_obj = self.kalman_estimator.update_measurement(
                object_id=tracked_obs.track_id,
                class_id=tracked_obs.observation.class_name,
                measurement=tracked_obs.observation.position,
                timestamp=frame.timestamp,
                measurement_covariance=tracked_obs.observation.measurement_covariance,
            )
            kalman_positions.append(estimated_obj.position)
            world_state = WorldState(timestamp=frame.timestamp, objects={gt_obj.object_id: estimated_obj})

            # Step 5: Part 6 Risk Assessment
            ego = EgoMotionState(frame.ego.position, frame.ego.velocity, frame.timestamp)
            risk = self.risk_evaluator.get_primary_threat(world_state, ego=ego)

            # Step 6: Part 7 Decision Engine
            ctx = DecisionContext(
                ego_position=ego.position,
                risk_assessment=risk,
                predicted_obstacles=(estimated_obj,),
                timestamp=frame.timestamp,
            )
            dec = self.decision_engine.decide_navigation(ctx)
            decisions.append(dec)

        # Verify Kalman filtering improves/smoothes raw noisy observation RMSE
        raw_rmse = position_rmse(raw_positions, ground_truth_positions)
        kalman_rmse = position_rmse(kalman_positions, ground_truth_positions)
        self.assertLess(kalman_rmse, raw_rmse)

        # Verify decision engine intervened appropriately during approach
        approach_decisions = [d for frame_idx, d in enumerate(decisions) if gt_seq.get_frame(frame_idx).timestamp <= 2.0]
        critical_dec = approach_decisions[-1]
        self.assertIn(
            critical_dec.action,
            (NavigationAction.STOP, NavigationAction.EMERGENCY_STOP, NavigationAction.TURN_LEFT, NavigationAction.TURN_RIGHT),
        )


if __name__ == "__main__":
    unittest.main()

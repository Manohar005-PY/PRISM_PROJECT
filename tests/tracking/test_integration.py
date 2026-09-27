"""End-to-end integration tests: Perception -> Tracking (Association) -> Kalman -> Prediction -> Risk -> Decision."""

import unittest

from app.core.types import Vector2, WorldState
from app.decision.context import DecisionContext
from app.decision.engine import DecisionEngine
from app.perception.adapters.synthetic import SyntheticDetector
from app.perception.pipeline import PerceptionPipeline
from app.prediction.relative_motion import EgoMotionState
from app.risk.evaluator import PredictiveRiskEvaluator
from app.risk.model import RiskConfig
from app.state.kalman import KalmanFilterConfig, KalmanStateEstimator
from app.state.projection import (
    CameraIntrinsics,
    CameraPose,
    GroundPlaneProjector,
    ProjectionConfig,
)
from app.tracking.config import TrackingConfig
from app.tracking.metrics import count_id_switches
from app.tracking.tracker import MultiObjectTracker
from simulation.noise import GaussianNoiseModel
from simulation.scenarios import create_crossing_scenario
from simulation.simulator import Simulator


class TestTrackingFullPipelineIntegration(unittest.TestCase):
    """Integration test verifying closed-loop data flow with autonomous multi-object data association."""

    def setUp(self) -> None:
        self.intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, image_width=640, image_height=480)
        self.pose = CameraPose(height=1.2, pitch=0.15)
        self.config = ProjectionConfig(intrinsics=self.intrinsics, camera_pose=self.pose)
        self.projector = GroundPlaneProjector(self.config)
        self.detector = SyntheticDetector(config=self.config, pixel_noise_sigma=1.5, random_seed=42)
        self.pipeline = PerceptionPipeline(self.detector, self.projector)

        self.kalman_estimator = KalmanStateEstimator(config=KalmanFilterConfig())
        self.tracker = MultiObjectTracker(
            config=TrackingConfig(confirmation_hits=2, gating_threshold=9.21),
            state_estimator=self.kalman_estimator,
        )
        self.risk_evaluator = PredictiveRiskEvaluator(RiskConfig(alpha_smooth=1.0))
        self.decision_engine = DecisionEngine()

    def test_crossing_objects_tracking_and_decision(self) -> None:
        """Verify autonomous tracking of crossing scenario preserves identity and drives decisions."""
        scenario = create_crossing_scenario(duration=4.0, timestep=0.1)
        gt_seq, _ = Simulator(scenario=scenario, noise_model=GaussianNoiseModel(0.0, 0.0)).run()

        tracked_ids_seq_obj1: list[int | None] = []
        tracked_ids_seq_obj2: list[int | None] = []

        for i in range(gt_seq.num_frames):
            frame = gt_seq.get_frame(i)
            # Create synthetic 2D observations from frame ground truth
            sim_entities = [(obj.object_id, obj.class_id, obj.position) for obj in frame.objects]
            obs_list, _ = self.pipeline.process_frame(sim_entities, frame.timestamp)

            # Part 9 MultiObjectTracker associates observations to persistent track_ids
            tracking_result = self.tracker.update(obs_list, frame.timestamp)

            # Build world state from tracked Kalman estimates
            active_objects = {}
            for to in tracking_result.tracked_observations:
                st = self.kalman_estimator.get_state(to.track_id)
                if st is not None:
                    active_objects[to.track_id] = st

            world_state = WorldState(timestamp=frame.timestamp, objects=active_objects)

            # Record tracking ID for crossing entity
            if tracking_result.tracked_observations:
                tracked_ids_seq_obj1.append(tracking_result.tracked_observations[0].track_id)

            # Run downstream Risk and Decision
            ego = EgoMotionState(frame.ego.position, frame.ego.velocity, frame.timestamp)
            risk = self.risk_evaluator.get_primary_threat(world_state, ego=ego)

            ctx = DecisionContext(
                ego_position=ego.position,
                risk_assessment=risk,
                predicted_obstacles=tuple(world_state.objects.values()),
                timestamp=frame.timestamp,
            )
            dec = self.decision_engine.decide_navigation(ctx)
            self.assertIsNotNone(dec)

        # Verify zero ID switches during the crossing sequence
        switches = count_id_switches(tracked_ids_seq_obj1)
        self.assertEqual(switches, 0)


if __name__ == "__main__":
    unittest.main()

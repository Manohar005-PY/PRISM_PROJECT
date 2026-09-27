"""End-to-end integration tests: Simulation -> Projection -> Kalman -> Prediction -> Collision Geometry."""

import unittest

from app.core.types import Vector2
from app.prediction.collision_geometry import (
    CollisionModel,
    analyze_collision,
    closest_point_of_approach,
    time_to_collision,
)
from app.prediction.constant_velocity import (
    ConstantVelocityPredictor,
    PredictionConfig,
)
from app.prediction.metrics import (
    prediction_error_at_horizon,
    trajectory_position_rmse,
)
from app.prediction.relative_motion import compute_relative_motion
from app.state.kalman import KalmanFilterConfig, KalmanStateEstimator
from app.state.noise import ConstantMeasurementNoise
from simulation.noise import GaussianNoiseModel
from simulation.scenarios import (
    create_crossing_scenario,
    create_diverging_scenario,
    create_head_on_scenario,
    create_parallel_scenario,
    create_stationary_obstacle_scenario,
)
from simulation.simulator import Simulator


class TestEndToEndPredictionPipeline(unittest.TestCase):
    """Scientific integration test suite connecting Parts 2, 3, 4, and 5."""

    def test_head_on_prediction_and_collision(self) -> None:
        """HEAD_ON scenario: verify trajectory prediction and collision analysis against future ground truth."""
        scenario = create_head_on_scenario(duration=3.0, timestep=0.1)
        sim_noise = GaussianNoiseModel(sigma_x=0.08, sigma_z=0.15, seed=42)
        simulator = Simulator(scenario=scenario, noise_model=sim_noise)
        gt_seq, obs_history = simulator.run()

        # Kalman Filter state estimator
        estimator = KalmanStateEstimator(
            config=KalmanFilterConfig(sigma_ax=0.5, sigma_az=0.5),
            noise_model=ConstantMeasurementNoise(sigma_x=0.08, sigma_z=0.15),
        )
        predictor = ConstantVelocityPredictor(PredictionConfig(timestep=0.1, max_horizon=3.0))

        # Feed frames 0..10 (t=0.0 to t=1.0)
        estimated_state = None
        for i in range(11):
            frame = gt_seq.get_frame(i)
            obs = obs_history[i][0]
            meas_pos = Vector2(
                x=(obs.bbox.center[0] - 320.0) / 50.0,
                z=(obs.bbox.bottom_center[1] - 240.0) / 50.0,
            )
            estimated_state = estimator.update_measurement(
                object_id=obs.object_id,
                class_id=obs.class_id,
                measurement=meas_pos,
                timestamp=frame.timestamp,
            )

        self.assertIsNotNone(estimated_state)
        # At t=1.0s, predict forward 2.0s into the future (t=1.0 to t=3.0)
        traj = predictor.predict(estimated_state, horizon=2.0)  # type: ignore[arg-type]
        self.assertEqual(len(traj.points), 21)

        # Extract future ground-truth entity points for t in [1.0, 3.0]
        future_gt_entities = [
            gt_seq.get_frame(i).objects[0]
            for i in range(10, gt_seq.num_frames)
        ]

        # Calculate prediction RMSE against future ground truth
        rmse = trajectory_position_rmse(traj, future_gt_entities)
        self.assertLess(rmse, 0.25)  # Position prediction matches future trajectory within 25cm

        # Check prediction error at 1.0s and 2.0s horizons
        err_10 = prediction_error_at_horizon(traj, future_gt_entities, target_time_offset=1.0)
        err_20 = prediction_error_at_horizon(traj, future_gt_entities, target_time_offset=2.0)
        self.assertLess(err_10, 0.25)
        self.assertLess(err_20, 0.40)

        # Collision Analysis at t=1.0s:
        # Ego is at (0, 0) stationary (or moving), object estimated position ~ (0, 4.0), velocity ~ (0, -1.0)
        ego_pos = Vector2(0.0, 0.0)
        ego_vel = Vector2(0.0, 0.0)
        rel = compute_relative_motion(ego_pos, ego_vel, estimated_state.position, estimated_state.velocity)  # type: ignore[union-attr]

        model = CollisionModel(object_radius=0.4, ego_radius=0.3, additional_margin=0.3)  # R=1.0m
        collision_pred = analyze_collision(rel.relative_position, rel.relative_velocity, model, horizon=3.0)

        self.assertTrue(collision_pred.collision_predicted)
        self.assertIsNotNone(collision_pred.ttc)
        # With object at Z=4m and Vz=-1m/s, reaches R=1m in ~3s
        self.assertAlmostEqual(collision_pred.ttc, 3.0, delta=0.5)  # type: ignore[arg-type]

    def test_crossing_scenario_cpa(self) -> None:
        """CROSSING scenario: path intersection at (0, 0) at t=5.0s."""
        scenario = create_crossing_scenario(duration=6.0, timestep=0.1)
        sim_noise = GaussianNoiseModel(sigma_x=0.05, sigma_z=0.05, seed=123)
        simulator = Simulator(scenario=scenario, noise_model=sim_noise)
        gt_seq, _ = simulator.run()

        # At t=0: Ego at (-5, 0) with v=(1, 0); Object at (0, 5) with v=(0, -1)
        frame0 = gt_seq.get_frame(0)
        ego0 = frame0.ego
        obj0 = frame0.objects[0]

        rel0 = compute_relative_motion(ego0.position, ego0.velocity, obj0.position, obj0.velocity)
        # p_rel = (0 - (-5), 5 - 0) = (5, 5)
        # v_rel = (0 - 1, -1 - 0) = (-1, -1)
        # t_cpa = -(5*-1 + 5*-1) / (1 + 1) = -(-10)/2 = 5.0s
        cpa0 = closest_point_of_approach(rel0.relative_position, rel0.relative_velocity)
        self.assertAlmostEqual(cpa0.time_to_cpa, 5.0)
        self.assertAlmostEqual(cpa0.distance_at_cpa, 0.0)

    def test_diverging_scenario_no_collision(self) -> None:
        """DIVERGING scenario: lateral separation creates a safe bypass (d_CPA=4.95m >> R=1m) -> no collision."""
        scenario = create_diverging_scenario(duration=4.0, timestep=0.1)
        gt_seq, _ = Simulator(scenario=scenario, noise_model=GaussianNoiseModel(0.0, 0.0)).run()

        frame0 = gt_seq.get_frame(0)
        ego0 = frame0.ego
        obj0 = frame0.objects[0]

        rel0 = compute_relative_motion(ego0.position, ego0.velocity, obj0.position, obj0.velocity)
        cpa0 = closest_point_of_approach(rel0.relative_position, rel0.relative_velocity)
        # CPA occurs at t=1.5s with minimum separation distance sqrt(3.5^2 + 3.5^2) = 4.95m
        self.assertAlmostEqual(cpa0.time_to_cpa, 1.5)
        self.assertAlmostEqual(cpa0.distance_at_cpa, (3.5**2 + 3.5**2)**0.5, delta=0.01)

        model = CollisionModel(object_radius=0.4, ego_radius=0.3, additional_margin=0.3)  # R=1.0m
        pred = analyze_collision(rel0.relative_position, rel0.relative_velocity, model, horizon=5.0)
        self.assertFalse(pred.collision_predicted)
        self.assertIsNone(pred.ttc)

        # At t=2.0s (after CPA), objects are strictly diverging (is_approaching is False)
        frame20 = gt_seq.get_frame(20)
        rel20 = compute_relative_motion(frame20.ego.position, frame20.ego.velocity, frame20.objects[0].position, frame20.objects[0].velocity)
        self.assertFalse(rel20.is_approaching)

    def test_reactive_vs_predictive_ablation(self) -> None:
        """SCIENTIFIC ABLATION: Compare reactive distance thresholding vs predictive trajectory collision analysis."""
        # Both objects are at current distance = 3.0 meters:
        # Case A: Object at (0, 3) moving AWAY at Vz = +1.5 m/s
        # Case B: Object at (0, 3) moving TOWARD at Vz = -1.5 m/s
        ego_pos = Vector2(0.0, 0.0)
        ego_vel = Vector2(0.0, 0.0)

        obj_a_pos = Vector2(0.0, 3.0)
        obj_a_vel = Vector2(0.0, 1.5)

        obj_b_pos = Vector2(0.0, 3.0)
        obj_b_vel = Vector2(0.0, -1.5)

        rel_a = compute_relative_motion(ego_pos, ego_vel, obj_a_pos, obj_a_vel)
        rel_b = compute_relative_motion(ego_pos, ego_vel, obj_b_pos, obj_b_vel)

        # 1. Reactive baseline (Distance threshold = 3.5m)
        reactive_threshold = 3.5
        reactive_alert_a = rel_a.distance < reactive_threshold
        reactive_alert_b = rel_b.distance < reactive_threshold

        # Reactive system alerts on BOTH identically because current distance is identical
        self.assertTrue(reactive_alert_a)
        self.assertTrue(reactive_alert_b)

        # 2. Predictive PRISM reasoning (Safety radius = 1.0m, Horizon = 3.0s)
        model = CollisionModel(object_radius=0.35, ego_radius=0.30, additional_margin=0.35)  # R=1.0m
        pred_a = analyze_collision(rel_a.relative_position, rel_a.relative_velocity, model, horizon=3.0)
        pred_b = analyze_collision(rel_b.relative_position, rel_b.relative_velocity, model, horizon=3.0)

        # Predictive system selectively alerts ONLY on Case B (approaching obstacle)
        self.assertFalse(pred_a.collision_predicted)
        self.assertIsNone(pred_a.ttc)

        self.assertTrue(pred_b.collision_predicted)
        self.assertIsNotNone(pred_b.ttc)
        # Reaches R=1m in (3.0 - 1.0) / 1.5 = 1.33s
        self.assertAlmostEqual(pred_b.ttc, 4.0 / 3.0, delta=0.05)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()

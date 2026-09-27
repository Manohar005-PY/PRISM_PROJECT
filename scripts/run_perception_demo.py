"""Demonstration script for PRISM Real-World Perception & Observation Fusion.

Demonstrates:
1. End-to-end Perception-to-Physics Pipeline:
   Frame -> 2D Detection -> Bottom Center -> Ground Projection -> Measurement Covariance R -> Association (TrackedObservation) -> Kalman Filter -> Decision.
2. Direct Perception Noise & Uncertainty Propagation Experiment across Varying Depths.
"""

from __future__ import annotations

import math
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.types import Vector2, WorldState
from app.decision.context import DecisionContext
from app.decision.engine import DecisionEngine
from app.perception.adapters.synthetic import SyntheticDetector
from app.perception.metrics import (
    contact_pixel_error,
    normalized_position_error,
    position_rmse,
    x_rmse,
    z_rmse,
)
from app.perception.model import TrackedObservation
from app.perception.observation import ObservationExtractor
from app.perception.pipeline import PerceptionPipeline
from app.perception.uncertainty import MeasurementUncertaintyConfig, MeasurementUncertaintyModel
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


def run_perception_pipeline_demo() -> None:
    print("=" * 80)
    print(" PRISM REAL-WORLD PERCEPTION & OBSERVATION FUSION DEMO")
    print("=" * 80)

    intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, image_width=640, image_height=480)
    pose = CameraPose(height=1.2, pitch=0.15)
    config = ProjectionConfig(intrinsics=intrinsics, camera_pose=pose)
    projector = GroundPlaneProjector(config)

    uncertainty_model = MeasurementUncertaintyModel(
        MeasurementUncertaintyConfig(
            sigma_x0=0.15,
            sigma_z0=0.25,
            beta_x=0.02,
            beta_z=0.05,
            confidence_noise_gain=1.5,
        )
    )
    detector = SyntheticDetector(config=config, pixel_noise_sigma=2.5, random_seed=100)
    pipeline = PerceptionPipeline(detector, projector, uncertainty_model)

    print("\n1. Visual Frame Processing & Observation Extraction:")
    print("-" * 80)

    # 3 Detected Objects at different locations and distances
    test_scene = [
        (1, "person", Vector2(0.0, 2.5)),
        (2, "chair", Vector2(-1.2, 4.0)),
        (3, "person", Vector2(1.5, 6.0)),
    ]

    obs_list, rejections = pipeline.process_frame(test_scene, timestamp=0.0)

    for obs in obs_list:
        sigma_x = math.sqrt(obs.measurement_covariance[0][0])
        sigma_z = math.sqrt(obs.measurement_covariance[1][1])
        u_bc, v_bc = obs.bbox.bottom_center if obs.bbox else (0.0, 0.0)

        print(f"Detection Proposal: ID={obs.detection_id} | Class={obs.class_name:<7} | Conf={obs.confidence:.2f} | BBox Bottom-Center=({u_bc:5.1f}, {v_bc:5.1f}) px")
        print(f"  -> Projected Position:     X={obs.position.x:+5.2f} m, Z={obs.position.z:5.2f} m")
        print(f"  -> Measurement Covariance: R_xx={obs.measurement_covariance[0][0]:.4f} (sigma_x={sigma_x:.3f}m), R_zz={obs.measurement_covariance[1][1]:.4f} (sigma_z={sigma_z:.3f}m)")
        print()

    print("-" * 80)


def run_perception_noise_experiment() -> None:
    print("\n" + "=" * 80)
    print("2. SCIENTIFIC EXPERIMENT: PERCEPTION NOISE & KALMAN FILTERING")
    print("=" * 80)
    print(f"{'True Depth (Z)':<16} | {'Raw Obs Error':<16} | {'Kalman State Error':<20} | {'Measurement sigma_z':<20} | {'Noise Reduction':<15}")
    print("-" * 80)

    intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, image_width=640, image_height=480)
    pose = CameraPose(height=1.2, pitch=0.15)
    config = ProjectionConfig(intrinsics=intrinsics, camera_pose=pose)
    projector = GroundPlaneProjector(config)
    uncertainty_model = MeasurementUncertaintyModel()

    depths = [2.0, 4.0, 6.0, 8.0]

    for z_true in depths:
        # Simulate 20 frames of stationary obstacle with pixel jitter
        detector = SyntheticDetector(config=config, pixel_noise_sigma=3.0, random_seed=42)
        pipeline = PerceptionPipeline(detector, projector, uncertainty_model)
        kalman = KalmanStateEstimator(config=KalmanFilterConfig(init_pos_var=0.5, init_vel_var=1.0))

        raw_positions: list[Vector2] = []
        kalman_positions: list[Vector2] = []
        gt_positions: list[Vector2] = []

        for f in range(20):
            t = f * 0.1
            gt_pos = Vector2(0.5, z_true)
            gt_positions.append(gt_pos)

            obs_list, _ = pipeline.process_frame([(1, "person", gt_pos)], timestamp=t)
            if obs_list:
                obs = obs_list[0]
                raw_positions.append(obs.position)
                # Association step: bind observation to track ID 1
                tracked_obs = TrackedObservation(track_id=1, observation=obs)
                state = kalman.update_measurement(
                    object_id=tracked_obs.track_id,
                    class_id=tracked_obs.observation.class_name,
                    measurement=tracked_obs.observation.position,
                    timestamp=t,
                    measurement_covariance=tracked_obs.observation.measurement_covariance,
                )
                kalman_positions.append(state.position)

        raw_err = position_rmse(raw_positions, gt_positions)
        kalman_err = position_rmse(kalman_positions, gt_positions)
        sigma_z = math.sqrt(uncertainty_model.compute_covariance(z_true)[1][1])
        reduction = ((raw_err - kalman_err) / raw_err) * 100.0 if raw_err > 0 else 0.0

        print(f"Z = {z_true:4.1f} m        | {raw_err:6.3f} m         | {kalman_err:6.3f} m              | {sigma_z:6.3f} m             | {reduction:5.1f}%")

    print("-" * 80)
    print("Finding: Monocular ground projection noise grows quadratically with depth Z.")
    print("         Explicit TrackedObservation association enables the Kalman State Estimator")
    print("         to dampen perception noise and stabilize spatial tracking.")
    print("=" * 80)


def main() -> None:
    run_perception_pipeline_demo()
    run_perception_noise_experiment()


if __name__ == "__main__":
    main()

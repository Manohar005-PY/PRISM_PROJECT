"""Demonstration script for PRISM Kalman state estimation.

Runs a canonical simulation scenario (HEAD_ON) with noisy measurements, passes them through
the KalmanStateEstimator, and evaluates estimated positions and velocities against ground truth.
"""

from __future__ import annotations

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.types import Vector2
from app.state.kalman import KalmanFilterConfig, KalmanStateEstimator
from app.state.metrics import (
    estimation_improvement,
    measurement_position_rmse,
    position_rmse,
    velocity_rmse,
)
from app.state.noise import ConstantMeasurementNoise
from simulation.noise import GaussianNoiseModel
from simulation.scenarios import create_head_on_scenario
from simulation.simulator import Simulator


def main() -> None:
    print("=" * 86)
    print(" PRISM KALMAN STATE ESTIMATION & UNCERTAINTY DEMONSTRATION")
    print("=" * 86)
    print("Scenario:              HEAD_ON (Obstacle at Z=5.0m moving at Vz=-1.0m/s toward user)")
    print("Measurement Noise:     Zero-mean Gaussian (sigma_x=0.10m, sigma_z=0.20m, seed=42)")
    print("State Vector:          [X, Z, Vx, Vz]")
    print("-" * 86)

    # 1. Run simulation to obtain Ground Truth and Noisy Observations
    scenario = create_head_on_scenario(duration=3.0, timestep=0.1)
    sim_noise = GaussianNoiseModel(sigma_x=0.10, sigma_z=0.20, seed=42)
    simulator = Simulator(scenario=scenario, noise_model=sim_noise)
    gt_seq, obs_history = simulator.run()

    # 2. Run Kalman Filter on physical measurements
    kalman_config = KalmanFilterConfig(
        sigma_ax=0.5,
        sigma_az=0.5,
        init_pos_var=0.5,
        init_vel_var=4.0,
    )
    estimator = KalmanStateEstimator(
        config=kalman_config,
        noise_model=ConstantMeasurementNoise(sigma_x=0.10, sigma_z=0.20),
    )

    true_positions: list[Vector2] = []
    true_velocities: list[Vector2] = []
    noisy_measurements: list[Vector2] = []
    kalman_positions: list[Vector2] = []
    kalman_velocities: list[Vector2] = []
    kalman_cov_traces: list[float] = []

    print(f"{'Time(s)':<8} | {'True Pos (X, Z)':<18} | {'Noisy Meas (X, Z)':<20} | {'Kalman State [X, Z, Vx, Vz]':<32}")
    print("-" * 86)

    for i in range(gt_seq.num_frames):
        frame = gt_seq.get_frame(i)
        obj_true = frame.objects[0]
        t = frame.timestamp

        # Extract noisy measurement from synthetic generator coordinates
        obs = obs_history[i][0]
        meas_pos = Vector2(
            x=(obs.bbox.center[0] - 320.0) / 50.0,
            z=(obs.bbox.bottom_center[1] - 240.0) / 50.0,
        )

        state = estimator.update_measurement(
            object_id=obj_true.object_id,
            class_id=obj_true.class_id,
            measurement=meas_pos,
            timestamp=t,
        )

        true_positions.append(obj_true.position)
        true_velocities.append(obj_true.velocity)
        noisy_measurements.append(meas_pos)
        kalman_positions.append(state.position)
        kalman_velocities.append(state.velocity)

        cov = state.covariance
        trace_p = cov[0][0] + cov[1][1] + cov[2][2] + cov[3][3] if cov else 0.0
        kalman_cov_traces.append(trace_p)

        if i % 3 == 0 or i == gt_seq.num_frames - 1:
            true_str = f"({obj_true.position.x:4.2f}, {obj_true.position.z:4.2f})"
            noisy_str = f"({meas_pos.x:4.2f}, {meas_pos.z:4.2f})"
            state_str = f"[{state.position.x:4.2f}, {state.position.z:4.2f}, {state.velocity.x:5.2f}, {state.velocity.z:5.2f}]"
            print(f"{t:<8.2f} | {true_str:<18} | {noisy_str:<20} | {state_str:<32}")

    print("-" * 86)
    # Compute quantitative metrics
    meas_rmse = measurement_position_rmse(noisy_measurements[5:], true_positions[5:])
    kalman_pos_rmse = position_rmse(kalman_positions[5:], true_positions[5:])
    kalman_vel_rmse = velocity_rmse(kalman_velocities[5:], true_velocities[5:])
    improvement = estimation_improvement(meas_rmse, kalman_pos_rmse)

    print("QUANTITATIVE PERFORMANCE EVALUATION (Steady-State):")
    print(f"  Raw Measurement Position RMSE:  {meas_rmse:.4f} meters")
    print(f"  Kalman Estimated Position RMSE: {kalman_pos_rmse:.4f} meters")
    print(f"  Position Error Reduction:      {improvement:.1f}% improvement over raw noise")
    print(f"  Kalman Estimated Velocity RMSE: {kalman_vel_rmse:.4f} m/s (recovered from position-only observations)")
    print(f"  Final State Covariance Trace:   {kalman_cov_traces[-1]:.4f}")
    print("=" * 86)


if __name__ == "__main__":
    main()

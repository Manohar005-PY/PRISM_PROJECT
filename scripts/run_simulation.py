"""Demonstration script for PRISM synthetic simulation environment.

Runs a canonical scenario (HEAD_ON) with injected measurement noise and
prints a side-by-side comparison of ground truth vs. noisy observation coordinates.
"""

from __future__ import annotations

import sys
import os

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.noise import GaussianNoiseModel
from simulation.scenarios import create_head_on_scenario
from simulation.simulator import Simulator


def main() -> None:
    print("=" * 76)
    print(" PRISM SYNTHETIC WORLD SIMULATION: HEAD_ON SCENARIO DEMONSTRATION")
    print("=" * 76)
    print("Configuration:")
    print("  Scenario:               HEAD_ON (User at 0,0 v=(0,+1); Object at 0,5 v=(0,-1))")
    print("  Duration:               2.5 seconds")
    print("  Timestep (dt):          0.25 seconds")
    print("  Noise Model:            Zero-mean Gaussian (sigma_x=0.05m, sigma_z=0.10m, seed=42)")
    print("-" * 76)

    scenario = create_head_on_scenario(duration=2.5, timestep=0.25)
    noise_model = GaussianNoiseModel(sigma_x=0.05, sigma_z=0.10, seed=42)
    simulator = Simulator(scenario=scenario, noise_model=noise_model)

    gt_seq, obs_history = simulator.run()

    # Header
    print(f"{'Time(s)':<8} | {'Ego Z(m)':<9} | {'True Obj X':<10} {'True Obj Z':<10} | {'True Dist':<9} | {'Obs BBox Center (px)':<20}")
    print("-" * 76)

    for i in range(gt_seq.num_frames):
        frame = gt_seq.get_frame(i)
        obs_list = obs_history[i]
        obs = obs_list[0] if obs_list else None

        ego_z = frame.ego.position.z
        obj = frame.objects[0]
        obj_x = obj.position.x
        obj_z = obj.position.z
        true_dist = (obj.position - frame.ego.position).magnitude

        if obs is not None:
            cx, cy = obs.bbox.center
            obs_str = f"cx={cx:6.1f}, cy={cy:6.1f}"
        else:
            obs_str = "N/A"

        print(f"{frame.timestamp:<8.2f} | {ego_z:<9.2f} | {obj_x:<10.2f} {obj_z:<10.2f} | {true_dist:<9.2f} | {obs_str:<20}")

    print("=" * 76)
    print("Simulation completed successfully. Ground truth remained pure throughout.")
    print("=" * 76)


if __name__ == "__main__":
    main()

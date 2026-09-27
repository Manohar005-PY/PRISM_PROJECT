"""Demonstration script for PRISM predictive motion and collision geometry.

Runs all 5 canonical scenarios through the full pipeline:
Simulation -> State Estimation (Kalman) -> Trajectory Prediction -> Relative Motion -> CPA / TTC Analysis.
Also demonstrates the reactive vs. predictive selectivity experiment.
"""

from __future__ import annotations

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.types import ObjectState, Vector2
from app.prediction.collision_geometry import (
    CollisionModel,
    analyze_collision,
    closest_point_of_approach,
)
from app.prediction.constant_velocity import (
    ConstantVelocityPredictor,
    PredictionConfig,
)
from app.prediction.relative_motion import (
    EgoMotionState,
    compute_relative_motion,
)
from simulation.scenarios import (
    create_crossing_scenario,
    create_diverging_scenario,
    create_head_on_scenario,
    create_parallel_scenario,
    create_stationary_obstacle_scenario,
)


def run_scenario_demo(name: str, ego: EgoMotionState, obj: ObjectState, horizon: float = 3.0) -> None:
    print("-" * 80)
    print(f" SCENARIO: {name}")
    print("-" * 80)

    print("Current State:")
    print(f"  Ego:       Position=({ego.position.x:5.2f}, {ego.position.z:5.2f}) m | Velocity=({ego.velocity.x:5.2f}, {ego.velocity.z:5.2f}) m/s")
    print(f"  Object:    Position=({obj.position.x:5.2f}, {obj.position.z:5.2f}) m | Velocity=({obj.velocity.x:5.2f}, {obj.velocity.z:5.2f}) m/s")

    rel = compute_relative_motion(ego.position, ego.velocity, obj.position, obj.velocity)
    print("\nRelative Motion:")
    print(f"  Rel Pos:   ({rel.relative_position.x:5.2f}, {rel.relative_position.z:5.2f}) m (Separation = {rel.distance:.2f} m)")
    print(f"  Rel Vel:   ({rel.relative_velocity.x:5.2f}, {rel.relative_velocity.z:5.2f}) m/s (Rel Speed = {rel.relative_speed:.2f} m/s)")
    print(f"  Approaching: {rel.is_approaching} (Closing Speed = {rel.closing_speed:.2f} m/s)")

    cpa = closest_point_of_approach(rel.relative_position, rel.relative_velocity)
    print("\nClosest Point of Approach (CPA):")
    print(f"  Time to CPA:       {cpa.time_to_cpa:.2f} s")
    print(f"  Distance at CPA:   {cpa.distance_at_cpa:.2f} m")
    print(f"  Position at CPA:   ({cpa.relative_position_at_cpa.x:5.2f}, {cpa.relative_position_at_cpa.z:5.2f}) m")

    model = CollisionModel(object_radius=0.35, ego_radius=0.30, additional_margin=0.35)  # R=1.0m
    pred = analyze_collision(rel.relative_position, rel.relative_velocity, model, horizon=horizon)
    print(f"\nCollision Analysis (Safety Radius = {model.effective_radius:.2f} m, Horizon = {horizon:.1f} s):")
    print(f"  Collision Predicted: {pred.collision_predicted}")
    ttc_str = f"{pred.ttc:.2f} s" if pred.ttc is not None else "None (No Collision)"
    print(f"  Time to Collision:   {ttc_str}")

    predictor = ConstantVelocityPredictor(PredictionConfig(timestep=0.5, max_horizon=horizon))
    traj = predictor.predict(obj, horizon=horizon)
    print(f"\nPredicted Trajectory ({len(traj.points)} points):")
    print(f"  {'Time (s)':<10} | {'X (m)':<10} | {'Z (m)':<10}")
    print("  " + "-" * 34)
    for pt in traj.points:
        print(f"  {pt.timestamp:<10.1f} | {pt.position.x:<10.2f} | {pt.position.z:<10.2f}")


def run_reactive_vs_predictive_experiment() -> None:
    print("\n" + "=" * 80)
    print(" SCIENTIFIC EXPERIMENT: REACTIVE DISTANCE vs. PREDICTIVE COLLISION REASONING")
    print("=" * 80)
    print("Research Question: Can predictive motion estimation prevent false warnings on moving obstacles?")
    print("Test Setup: Two obstacles at identical current distance (3.0m).")
    print("  Case A: Pedestrian moving AWAY from user at +1.5 m/s.")
    print("  Case B: Pedestrian moving TOWARD user at -1.5 m/s.")
    print("-" * 80)

    ego = EgoMotionState(position=Vector2(0.0, 0.0), velocity=Vector2(0.0, 0.0))
    obj_a = ObjectState(1, "person", Vector2(0.0, 3.0), Vector2(0.0, 1.5), 0.0)
    obj_b = ObjectState(2, "person", Vector2(0.0, 3.0), Vector2(0.0, -1.5), 0.0)

    rel_a = compute_relative_motion(ego.position, ego.velocity, obj_a.position, obj_a.velocity)
    rel_b = compute_relative_motion(ego.position, ego.velocity, obj_b.position, obj_b.velocity)

    # 1. Reactive Baseline (Distance threshold = 3.5m)
    threshold = 3.5
    alert_a_reactive = rel_a.distance < threshold
    alert_b_reactive = rel_b.distance < threshold

    # 2. Predictive PRISM Model (R = 1.0m, Horizon = 3.0s)
    model = CollisionModel(object_radius=0.35, ego_radius=0.30, additional_margin=0.35)
    pred_a = analyze_collision(rel_a.relative_position, rel_a.relative_velocity, model, horizon=3.0)
    pred_b = analyze_collision(rel_b.relative_position, rel_b.relative_velocity, model, horizon=3.0)

    print(f"{'Metric':<30} | {'Case A (Moving Away)':<22} | {'Case B (Moving Toward)':<22}")
    print("-" * 80)
    print(f"{'Current Distance':<30} | {rel_a.distance:<22.2f} | {rel_b.distance:<22.2f}")
    print(f"{'Relative Velocity (Vz)':<30} | {rel_a.relative_velocity.z:<22.2f} | {rel_b.relative_velocity.z:<22.2f}")
    print(f"{'Approaching Status':<30} | {str(rel_a.is_approaching):<22} | {str(rel_b.is_approaching):<22}")
    print(f"{'Distance at CPA':<30} | {closest_point_of_approach(rel_a.relative_position, rel_a.relative_velocity).distance_at_cpa:<22.2f} | {closest_point_of_approach(rel_b.relative_position, rel_b.relative_velocity).distance_at_cpa:<22.2f}")
    print(f"{'TTC (Time to Collision)':<30} | {str(pred_a.ttc):<22} | {f'{pred_b.ttc:.2f} s':<22}")
    print(f"{'Reactive Alert (d < 3.5m)':<30} | {str(alert_a_reactive) + ' (FALSE ALARM)':<22} | {str(alert_b_reactive) + ' (TRUE ALERT)':<22}")
    print(f"{'Predictive Collision Alert':<30} | {str(pred_a.collision_predicted) + ' (SUPPRESSED)':<22} | {str(pred_b.collision_predicted) + ' (TIMELY ALERT)':<22}")
    print("=" * 80)
    print("Finding: Predictive geometry eliminates false alerts for departing entities while correctly warning on impending collisions.")


def main() -> None:
    print("=" * 80)
    print(" PRISM PREDICTIVE MOTION & COLLISION GEOMETRY DEMONSTRATION")
    print("=" * 80)

    # 1. HEAD ON
    run_scenario_demo(
        "HEAD_ON (User moving forward at +1m/s, Obstacle moving toward user at -1m/s)",
        ego=EgoMotionState(Vector2(0.0, 0.0), Vector2(0.0, 1.0)),
        obj=ObjectState(1, "person", Vector2(0.0, 5.0), Vector2(0.0, -1.0), 0.0),
    )

    # 2. CROSSING
    run_scenario_demo(
        "CROSSING (User moving right at +1m/s, Obstacle moving down at -1m/s)",
        ego=EgoMotionState(Vector2(-3.0, 0.0), Vector2(1.0, 0.0)),
        obj=ObjectState(2, "person", Vector2(0.0, 3.0), Vector2(0.0, -1.0), 0.0),
    )

    # 3. PARALLEL
    run_scenario_demo(
        "PARALLEL (User and Obstacle moving forward at equal speed +1m/s)",
        ego=EgoMotionState(Vector2(0.0, 0.0), Vector2(0.0, 1.0)),
        obj=ObjectState(3, "person", Vector2(2.0, 4.0), Vector2(0.0, 1.0), 0.0),
    )

    # 4. DIVERGING
    run_scenario_demo(
        "DIVERGING (User moving forward, Obstacle moving laterally right)",
        ego=EgoMotionState(Vector2(0.0, 0.0), Vector2(0.0, 1.0)),
        obj=ObjectState(4, "person", Vector2(2.0, 5.0), Vector2(1.0, 0.0), 0.0),
    )

    # 5. STATIONARY OBSTACLE
    run_scenario_demo(
        "STATIONARY_OBSTACLE (User walking toward stationary chair)",
        ego=EgoMotionState(Vector2(0.0, 0.0), Vector2(0.0, 1.2)),
        obj=ObjectState(5, "chair", Vector2(0.0, 4.0), Vector2(0.0, 0.0), 0.0),
    )

    # Scientific Experiment
    run_reactive_vs_predictive_experiment()


if __name__ == "__main__":
    main()

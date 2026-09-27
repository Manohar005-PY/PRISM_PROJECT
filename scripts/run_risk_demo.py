"""Demonstration script for PRISM Risk Assessment Engine.

Demonstrates continuous multi-factor risk scoring, discrete RiskLevel classification,
multi-obstacle ranking, and uncertainty-aware boundary expansion.
"""

from __future__ import annotations

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.types import Covariance4x4, ObjectState, Vector2, WorldState
from app.prediction.relative_motion import EgoMotionState
from app.risk.evaluator import PredictiveRiskEvaluator
from app.risk.model import RiskConfig


def main() -> None:
    print("=" * 80)
    print(" PRISM RISK ASSESSMENT ENGINE DEMONSTRATION")
    print("=" * 80)

    evaluator = PredictiveRiskEvaluator(RiskConfig())

    # Demo 1: Multi-Obstacle Scene Evaluation & Threat Ranking
    print("\nDEMO 1: Multi-Obstacle Threat Ranking in a Complex Hallway Scene")
    print("-" * 80)
    ego = EgoMotionState(position=Vector2(0.0, 0.0), velocity=Vector2(0.0, 1.0))

    # Obstacle 1: Approaching Head-On at 2.5m (Urgent)
    obj1 = ObjectState(1, "person", Vector2(0.0, 2.5), Vector2(0.0, -1.2), 0.0)
    # Obstacle 2: Crossing from right at 3.0m
    obj2 = ObjectState(2, "person", Vector2(1.5, 3.0), Vector2(-0.8, 0.0), 0.0)
    # Obstacle 3: Moving away ahead at 4.0m (Safe)
    obj3 = ObjectState(3, "person", Vector2(0.0, 4.0), Vector2(0.0, 1.0), 0.0)
    # Obstacle 4: Stationary chair off to side at (2.0, 3.0)
    obj4 = ObjectState(4, "chair", Vector2(2.0, 3.0), Vector2(0.0, 0.0), 0.0)

    world = WorldState(timestamp=0.0, objects={1: obj1, 2: obj2, 3: obj3, 4: obj4})
    assessments = evaluator.evaluate_world(world, ego=ego)
    primary = evaluator.get_primary_threat(world, ego=ego)

    print(f"{'Obj ID':<8} | {'Class':<8} | {'Position (m)':<15} | {'Velocity (m/s)':<16} | {'TTC':<10} | {'Risk Score':<12} | {'Risk Level':<10}")
    print("-" * 80)
    for a in assessments:
        obj = world.objects[a.object_id]
        pos_str = f"({obj.position.x:4.1f}, {obj.position.z:4.1f})"
        vel_str = f"({obj.velocity.x:4.1f}, {obj.velocity.z:4.1f})"
        ttc_str = f"{a.ttc:4.2f} s" if a.ttc is not None else "None"
        print(f"{a.object_id:<8} | {obj.class_id:<8} | {pos_str:<15} | {vel_str:<16} | {ttc_str:<10} | {a.risk_score:<12.1f}% | {a.risk_level.value:<10}")

    print("-" * 80)
    if primary is not None:
        print(f"Primary Selected Threat: Object #{primary.object_id} -> {primary.risk_level.value} Risk ({primary.risk_score:.1f}%)")

    # Demo 2: Uncertainty-Aware Spatial Ablation on Borderline Near-Miss
    print("\n" + "=" * 80)
    print("DEMO 2: SCIENTIFIC ABLATION - DETERMINISTIC vs. UNCERTAINTY-AWARE RISK")
    print("=" * 80)
    print("Scenario: Obstacle passing on near-miss path at d_CPA = 1.10m (Base Radius = 1.00m).")
    print("-" * 80)

    # 1. Deterministic Case
    det_obj = ObjectState(1, "person", Vector2(1.10, 3.0), Vector2(0.0, -1.0), 0.0, covariance=None)
    profile_det = evaluator.evaluate_object(det_obj, ego=EgoMotionState(Vector2(0, 0), Vector2(0, 0)))

    # 2. Uncertainty-Aware Case (Measurement noise covariance sigma_pos = 0.35m)
    cov: Covariance4x4 = (
        (0.06, 0.0, 0.0, 0.0),
        (0.0, 0.06, 0.0, 0.0),
        (0.0, 0.0, 0.1, 0.0),
        (0.0, 0.0, 0.0, 0.1),
    )
    unc_obj = ObjectState(1, "person", Vector2(1.10, 3.0), Vector2(0.0, -1.0), 0.0, covariance=cov)
    profile_unc = evaluator.evaluate_object(unc_obj, ego=EgoMotionState(Vector2(0, 0), Vector2(0, 0)))

    print(f"{'Condition':<30} | {'Collision Prob (P_coll)':<25} | {'Risk Score':<12} | {'Risk Level':<10}")
    print("-" * 80)
    print(f"{'Deterministic (No Covariance)':<30} | {profile_det.factors.collision_probability:<25.2f} | {profile_det.score * 100.0:<12.1f}% | {profile_det.risk_level.value:<10}")
    print(f"{'Uncertainty-Aware (Kalman P)':<30} | {profile_unc.factors.collision_probability:<25.2f} | {profile_unc.score * 100.0:<12.1f}% | {profile_unc.risk_level.value:<10}")
    print("-" * 80)
    print("Finding: Spatial covariance integration enables PRISM to maintain appropriate vigilance on borderline near-misses.")
    print("=" * 80)


if __name__ == "__main__":
    main()

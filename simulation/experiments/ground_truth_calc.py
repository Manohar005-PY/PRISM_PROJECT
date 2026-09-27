"""Independent ground-truth collision, TTC, and dCPA reference calculation.

CRITICAL SCIENTIFIC PRINCIPLE:
This module provides an independent reference ground-truth solver implemented strictly
from first-principles kinematics and analytical geometry. It does NOT invoke or depend on
PRISM's production collision geometry, prediction, or risk engines, preventing circular validation.
"""

from __future__ import annotations

import math
from typing import Sequence

from app.core.types import Vector2
from simulation.experiments.models import (
    AgentSpecification,
    GroundTruthCollisionState,
    GroundTruthObjectState,
)


def compute_relative_kinematics(
    ego_pos: Vector2,
    ego_vel: Vector2,
    target_pos: Vector2,
    target_vel: Vector2,
) -> tuple[Vector2, Vector2]:
    """Compute relative position and velocity vectors (target relative to ego)."""
    rel_pos = Vector2(x=target_pos.x - ego_pos.x, z=target_pos.z - ego_pos.z)
    rel_vel = Vector2(x=target_vel.x - ego_vel.x, z=target_vel.z - ego_vel.z)
    return rel_pos, rel_vel


def compute_constant_velocity_dcpa(
    rel_pos: Vector2,
    rel_vel: Vector2,
) -> tuple[float, float]:
    """Compute analytical closest point of approach (t_CPA, d_CPA).

    Args:
        rel_pos: Relative position vector (target - ego).
        rel_vel: Relative velocity vector (target - ego).

    Returns:
        tuple (t_cpa, d_cpa):
            t_cpa: Future time to closest approach in seconds (>= 0.0).
            d_cpa: Distance at closest point of approach in meters.
    """
    v_sq = rel_vel.x * rel_vel.x + rel_vel.z * rel_vel.z
    current_dist = math.sqrt(rel_pos.x * rel_pos.x + rel_pos.z * rel_pos.z)

    if v_sq < 1e-9:
        # Stationary relative motion
        return 0.0, current_dist

    # Dot product of position and velocity vectors
    dot_rv = rel_pos.x * rel_vel.x + rel_pos.z * rel_vel.z

    # t_cpa is the unconstrained minimizer of ||r + v*t||^2: t = -(r . v) / ||v||^2
    unconstrained_t_cpa = -dot_rv / v_sq
    t_cpa = max(0.0, unconstrained_t_cpa)

    # Position at t_cpa
    cpa_x = rel_pos.x + rel_vel.x * t_cpa
    cpa_z = rel_pos.z + rel_vel.z * t_cpa
    d_cpa = math.sqrt(cpa_x * cpa_x + cpa_z * cpa_z)

    return t_cpa, d_cpa


def compute_constant_velocity_ttc(
    rel_pos: Vector2,
    rel_vel: Vector2,
    collision_radius: float,
) -> float | None:
    """Compute exact analytical Time-to-Collision (TTC) for two circular agents.

    Solves the quadratic equation:
        ||rel_pos + rel_vel * t||^2 = collision_radius^2
        A * t^2 + B * t + C = 0

    Args:
        rel_pos: Relative position vector (target - ego).
        rel_vel: Relative velocity vector (target - ego).
        collision_radius: Combined physical radii of ego and target (r1 + r2).

    Returns:
        Smallest non-negative future collision time t in seconds, or None if no collision.
    """
    current_dist_sq = rel_pos.x * rel_pos.x + rel_pos.z * rel_pos.z
    r_col_sq = collision_radius * collision_radius

    if current_dist_sq <= r_col_sq:
        # Already in physical contact / collision
        return 0.0

    v_sq = rel_vel.x * rel_vel.x + rel_vel.z * rel_vel.z
    if v_sq < 1e-9:
        # Relative speed zero, will never collide
        return None

    dot_rv = rel_pos.x * rel_vel.x + rel_pos.z * rel_vel.z

    # If relative velocity is diverging (dot_rv >= 0), objects are moving apart
    if dot_rv >= 0.0:
        return None

    # Quadratic coefficients
    a = v_sq
    b = 2.0 * dot_rv
    c = current_dist_sq - r_col_sq

    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        # Trajectories do not intersect within collision radius
        return None

    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    if t1 >= 0.0:
        return t1
    elif t2 >= 0.0:
        return 0.0

    return None


def evaluate_ground_truth_for_agent(
    current_time: float,
    ego_pos: Vector2,
    ego_vel: Vector2,
    ego_radius: float,
    target_spec: AgentSpecification,
) -> tuple[GroundTruthObjectState, GroundTruthCollisionState]:
    """Evaluate exact ground-truth state and collision metrics for an agent at timestamp t."""
    behavior = target_spec.behavior
    params = target_spec.behavior_params

    # 1. Kinematic propagation based on behavior
    if behavior == "direction_change":
        turn_time = float(params.get("turn_time", 2.5))
        turn_angle_rad = float(params.get("turn_angle_rad", math.pi / 2.0))
        init_vx = target_spec.initial_velocity.x
        init_vz = target_spec.initial_velocity.z

        cos_a = math.cos(turn_angle_rad)
        sin_a = math.sin(turn_angle_rad)
        post_vx = init_vx * cos_a - init_vz * sin_a
        post_vz = init_vx * sin_a + init_vz * cos_a

        if current_time < turn_time:
            # Segment 1: initial velocity
            pos_x = target_spec.initial_position.x + init_vx * current_time
            pos_z = target_spec.initial_position.z + init_vz * current_time
            vel_x = init_vx
            vel_z = init_vz
        else:
            # Segment 2: post-turn velocity
            p_turn_x = target_spec.initial_position.x + init_vx * turn_time
            p_turn_z = target_spec.initial_position.z + init_vz * turn_time
            dt_post = current_time - turn_time
            pos_x = p_turn_x + post_vx * dt_post
            pos_z = p_turn_z + post_vz * dt_post
            vel_x = post_vx
            vel_z = post_vz

        target_pos = Vector2(x=pos_x, z=pos_z)
        target_vel = Vector2(x=vel_x, z=vel_z)
        heading = math.atan2(vel_x, vel_z) if (vel_x != 0 or vel_z != 0) else 0.0

    else:
        # Constant velocity
        pos_x = target_spec.initial_position.x + target_spec.initial_velocity.x * current_time
        pos_z = target_spec.initial_position.z + target_spec.initial_velocity.z * current_time
        target_pos = Vector2(x=pos_x, z=pos_z)
        target_vel = target_spec.initial_velocity
        heading = math.atan2(target_vel.x, target_vel.z) if (target_vel.x != 0 or target_vel.z != 0) else 0.0

    target_state = GroundTruthObjectState(
        object_id=target_spec.agent_id,
        class_id=target_spec.class_id,
        x=round(target_pos.x, 6),
        z=round(target_pos.z, 6),
        vx=round(target_vel.x, 6),
        vz=round(target_vel.z, 6),
        ax=round(target_spec.initial_acceleration.x, 6),
        az=round(target_spec.initial_acceleration.z, 6),
        radius=target_spec.physical_radius,
        heading_rad=round(heading, 6),
    )

    # 2. Collision Reference Calculation
    rel_pos, rel_vel = compute_relative_kinematics(ego_pos, ego_vel, target_pos, target_vel)
    collision_radius = ego_radius + target_spec.physical_radius
    dist = math.sqrt(rel_pos.x * rel_pos.x + rel_pos.z * rel_pos.z)
    has_collided = (dist <= collision_radius)

    if behavior == "direction_change":
        # Piecewise analytical evaluation
        turn_time = float(params.get("turn_time", 2.5))
        if current_time < turn_time:
            # Check segment 1 first
            t_rem_seg1 = turn_time - current_time
            t_cpa_1, d_cpa_1 = compute_constant_velocity_dcpa(rel_pos, rel_vel)
            ttc_1 = compute_constant_velocity_ttc(rel_pos, rel_vel, collision_radius)

            if ttc_1 is not None and ttc_1 <= t_rem_seg1:
                ttc = ttc_1
                t_cpa = t_cpa_1
                d_cpa = d_cpa_1
            else:
                # Evaluate segment 2 from turn point
                turn_ego_pos = Vector2(x=ego_pos.x + ego_vel.x * t_rem_seg1, z=ego_pos.z + ego_vel.z * t_rem_seg1)
                turn_target_pos = Vector2(x=target_pos.x + target_vel.x * t_rem_seg1, z=target_pos.z + target_vel.z * t_rem_seg1)
                
                init_vx = target_spec.initial_velocity.x
                init_vz = target_spec.initial_velocity.z
                turn_angle_rad = float(params.get("turn_angle_rad", math.pi / 2.0))
                cos_a = math.cos(turn_angle_rad)
                sin_a = math.sin(turn_angle_rad)
                post_vel = Vector2(x=init_vx * cos_a - init_vz * sin_a, z=init_vx * sin_a + init_vz * cos_a)

                rel_pos_2, rel_vel_2 = compute_relative_kinematics(turn_ego_pos, ego_vel, turn_target_pos, post_vel)
                t_cpa_2, d_cpa_2 = compute_constant_velocity_dcpa(rel_pos_2, rel_vel_2)
                ttc_2 = compute_constant_velocity_ttc(rel_pos_2, rel_vel_2, collision_radius)

                t_cpa = t_rem_seg1 + t_cpa_2
                d_cpa = min(d_cpa_1, d_cpa_2)
                ttc = (t_rem_seg1 + ttc_2) if ttc_2 is not None else None
        else:
            # Already in segment 2
            t_cpa, d_cpa = compute_constant_velocity_dcpa(rel_pos, rel_vel)
            ttc = compute_constant_velocity_ttc(rel_pos, rel_vel, collision_radius)
    else:
        # Direct constant-velocity closed-form evaluation
        t_cpa, d_cpa = compute_constant_velocity_dcpa(rel_pos, rel_vel)
        ttc = compute_constant_velocity_ttc(rel_pos, rel_vel, collision_radius)

    collision_state = GroundTruthCollisionState(
        has_collided=has_collided,
        time_to_collision=round(ttc, 4) if ttc is not None else None,
        closest_point_of_approach=round(d_cpa, 4),
        t_cpa=round(t_cpa, 4),
        first_collision_timestamp=current_time if has_collided else None,
        colliding_agent_id=target_spec.agent_id if has_collided else None,
    )

    return target_state, collision_state

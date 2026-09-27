"""Deterministic kinematic motion models for synthetic entities.

Motion models compute exact future ground-truth positions given current state and kinematics.
Noise injection is explicitly isolated outside the motion model to preserve ground-truth purity.
"""

from __future__ import annotations

import math

from app.core.types import Vector2
from simulation.entities import EgoEntity, SimulatedEntity


class ConstantVelocityModel:
    """Deterministic linear constant-velocity motion model.

    Equation:
        p(t + dt) = p(t) + v * dt
    """

    @staticmethod
    def advance(position: Vector2, velocity: Vector2, dt: float) -> Vector2:
        """Propagate a ground-plane position forward in time at constant velocity.

        Args:
            position: Initial position on the ground plane (meters).
            velocity: Constant velocity vector (m/s).
            dt: Time delta to advance (seconds, non-negative).

        Returns:
            New ground-plane position Vector2 after dt seconds.

        Raises:
            ValueError: If dt is negative or not finite.
            TypeError: If position or velocity are not Vector2 instances.
        """
        if not isinstance(position, Vector2):
            raise TypeError(f"position must be a Vector2, got {type(position)}")
        if not isinstance(velocity, Vector2):
            raise TypeError(f"velocity must be a Vector2, got {type(velocity)}")
        if not isinstance(dt, (int, float)) or not math.isfinite(dt):
            raise ValueError(f"dt must be a finite float, got {dt!r}")
        if dt < 0.0:
            raise ValueError(f"dt cannot be negative, got {dt}")

        return Vector2(
            x=position.x + velocity.x * dt,
            z=position.z + velocity.z * dt,
        )

    @classmethod
    def advance_entity(cls, entity: SimulatedEntity, dt: float) -> SimulatedEntity:
        """Propagate a simulated entity forward by dt seconds.

        Args:
            entity: Current simulated entity state.
            dt: Time delta in seconds.

        Returns:
            New SimulatedEntity at timestamp = entity.timestamp + dt.
        """
        new_position = cls.advance(entity.position, entity.velocity, dt)
        return SimulatedEntity(
            object_id=entity.object_id,
            class_id=entity.class_id,
            position=new_position,
            velocity=entity.velocity,
            timestamp=entity.timestamp + dt,
        )

    @classmethod
    def advance_ego(cls, ego: EgoEntity, dt: float) -> EgoEntity:
        """Propagate the ego agent forward by dt seconds.

        Args:
            ego: Current ego entity state.
            dt: Time delta in seconds.

        Returns:
            New EgoEntity at timestamp = ego.timestamp + dt.
        """
        new_position = cls.advance(ego.position, ego.velocity, dt)
        return EgoEntity(
            position=new_position,
            velocity=ego.velocity,
            timestamp=ego.timestamp + dt,
        )

"""Constant Velocity trajectory predictor with analytical uncertainty covariance propagation.

Implements the PRISM Predictor protocol:
ObjectState -> Trajectory [TrajectoryPoint(timestamp, position, velocity, covariance)]
Operates purely in physical ground-plane metric coordinates (meters and m/s).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from app.core.interfaces import Predictor
from app.core.types import (
    Covariance4x4,
    ObjectState,
    Trajectory,
    TrajectoryPoint,
    Vector2,
)

# Lightweight matrix operations for 4x4 covariance propagation
Mat4x4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]


def _mat4_mul(A: Mat4x4, B: Mat4x4) -> Mat4x4:
    """Multiply two 4x4 matrices: C = A * B."""
    return (
        (
            A[0][0]*B[0][0] + A[0][1]*B[1][0] + A[0][2]*B[2][0] + A[0][3]*B[3][0],
            A[0][0]*B[0][1] + A[0][1]*B[1][1] + A[0][2]*B[2][1] + A[0][3]*B[3][1],
            A[0][0]*B[0][2] + A[0][1]*B[1][2] + A[0][2]*B[2][2] + A[0][3]*B[3][2],
            A[0][0]*B[0][3] + A[0][1]*B[1][3] + A[0][2]*B[2][3] + A[0][3]*B[3][3],
        ),
        (
            A[1][0]*B[0][0] + A[1][1]*B[1][0] + A[1][2]*B[2][0] + A[1][3]*B[3][0],
            A[1][0]*B[0][1] + A[1][1]*B[1][1] + A[1][2]*B[2][1] + A[1][3]*B[3][1],
            A[1][0]*B[0][2] + A[1][1]*B[1][2] + A[1][2]*B[2][2] + A[1][3]*B[3][2],
            A[1][0]*B[0][3] + A[1][1]*B[1][3] + A[1][2]*B[2][3] + A[1][3]*B[3][3],
        ),
        (
            A[2][0]*B[0][0] + A[2][1]*B[1][0] + A[2][2]*B[2][0] + A[2][3]*B[3][0],
            A[2][0]*B[0][1] + A[2][1]*B[1][1] + A[2][2]*B[2][1] + A[2][3]*B[3][1],
            A[2][0]*B[0][2] + A[2][1]*B[1][2] + A[2][2]*B[2][2] + A[2][3]*B[3][2],
            A[2][0]*B[0][3] + A[2][1]*B[1][3] + A[2][2]*B[2][3] + A[2][3]*B[3][3],
        ),
        (
            A[3][0]*B[0][0] + A[3][1]*B[1][0] + A[3][2]*B[2][0] + A[3][3]*B[3][0],
            A[3][0]*B[0][1] + A[3][1]*B[1][1] + A[3][2]*B[2][1] + A[3][3]*B[3][1],
            A[3][0]*B[0][2] + A[3][1]*B[1][2] + A[3][2]*B[2][2] + A[3][3]*B[3][2],
            A[3][0]*B[0][3] + A[3][1]*B[1][3] + A[3][2]*B[2][3] + A[3][3]*B[3][3],
        ),
    )


def _mat4_transpose(A: Mat4x4) -> Mat4x4:
    """Transpose a 4x4 matrix: A^T."""
    return (
        (A[0][0], A[1][0], A[2][0], A[3][0]),
        (A[0][1], A[1][1], A[2][1], A[3][1]),
        (A[0][2], A[1][2], A[2][2], A[3][2]),
        (A[0][3], A[1][3], A[2][3], A[3][3]),
    )


def _mat4_add(A: Mat4x4, B: Mat4x4) -> Mat4x4:
    """Add two 4x4 matrices: C = A + B."""
    return (
        (A[0][0]+B[0][0], A[0][1]+B[0][1], A[0][2]+B[0][2], A[0][3]+B[0][3]),
        (A[1][0]+B[1][0], A[1][1]+B[1][1], A[1][2]+B[1][2], A[1][3]+B[1][3]),
        (A[2][0]+B[2][0], A[2][1]+B[2][1], A[2][2]+B[2][2], A[2][3]+B[2][3]),
        (A[3][0]+B[3][0], A[3][1]+B[3][1], A[3][2]+B[3][2], A[3][3]+B[3][3]),
    )


def _symmetrize_mat4(A: Mat4x4) -> Mat4x4:
    """Symmetrize a 4x4 matrix: A_sym = 0.5 * (A + A^T)."""
    return (
        (A[0][0], 0.5*(A[0][1]+A[1][0]), 0.5*(A[0][2]+A[2][0]), 0.5*(A[0][3]+A[3][0])),
        (0.5*(A[1][0]+A[0][1]), A[1][1], 0.5*(A[1][2]+A[2][1]), 0.5*(A[1][3]+A[3][1])),
        (0.5*(A[2][0]+A[0][2]), 0.5*(A[2][1]+A[1][2]), A[2][2], 0.5*(A[2][3]+A[3][2])),
        (0.5*(A[3][0]+A[0][3]), 0.5*(A[3][1]+A[1][3]), 0.5*(A[3][2]+A[2][3]), A[3][3]),
    )


@dataclass(frozen=True)
class PredictionConfig:
    """Configuration for kinematic trajectory prediction.

    Attributes:
        timestep: Temporal resolution (seconds) between consecutive predicted points.
        process_noise_ax: Continuous lateral acceleration disturbance std dev (m/s^2).
        process_noise_az: Continuous forward acceleration disturbance std dev (m/s^2).
        max_horizon: Maximum allowable prediction horizon (seconds).
    """

    timestep: float = 0.1
    process_noise_ax: float = 0.0
    process_noise_az: float = 0.0
    max_horizon: float = 10.0

    def __post_init__(self) -> None:
        if not isinstance(self.timestep, (int, float)) or not math.isfinite(self.timestep) or self.timestep <= 0.0:
            raise ValueError(f"timestep must be a positive finite float, got {self.timestep!r}")
        for name, val in [("process_noise_ax", self.process_noise_ax), ("process_noise_az", self.process_noise_az)]:
            if not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0.0:
                raise ValueError(f"{name} must be a non-negative finite float, got {val!r}")
        if not isinstance(self.max_horizon, (int, float)) or not math.isfinite(self.max_horizon) or self.max_horizon <= 0.0:
            raise ValueError(f"max_horizon must be a positive finite float, got {self.max_horizon!r}")
        object.__setattr__(self, "timestep", float(self.timestep))
        object.__setattr__(self, "process_noise_ax", float(self.process_noise_ax))
        object.__setattr__(self, "process_noise_az", float(self.process_noise_az))
        object.__setattr__(self, "max_horizon", float(self.max_horizon))


class ConstantVelocityPredictor:
    """Deterministic linear constant-velocity trajectory predictor.

    Implements the Predictor protocol from app.core.interfaces.
    """

    def __init__(self, config: PredictionConfig | None = None) -> None:
        self.config: PredictionConfig = config if config is not None else PredictionConfig()

    def _build_transition_matrix(self, dt: float) -> Mat4x4:
        """Construct 4x4 state transition matrix F(dt) for [X, Z, Vx, Vz]."""
        return (
            (1.0, 0.0, dt,  0.0),
            (0.0, 1.0, 0.0, dt),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )

    def _build_process_noise(self, dt: float) -> Mat4x4:
        """Construct continuous white noise acceleration covariance Q(dt) for [X, Z, Vx, Vz]."""
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt

        q_ax = self.config.process_noise_ax * self.config.process_noise_ax
        q_az = self.config.process_noise_az * self.config.process_noise_az

        return (
            (q_ax * dt4 / 4.0, 0.0,              q_ax * dt3 / 2.0, 0.0),
            (0.0,              q_az * dt4 / 4.0, 0.0,              q_az * dt3 / 2.0),
            (q_ax * dt3 / 2.0, 0.0,              q_ax * dt2,       0.0),
            (0.0,              q_az * dt3 / 2.0, 0.0,              q_az * dt2),
        )

    def predict(self, state: ObjectState, horizon: float) -> Trajectory:
        """Extrapolate an ObjectState into a future Trajectory across the given horizon.

        Args:
            state: Estimated physical state of the object.
            horizon: Future prediction duration in seconds (0.0 <= horizon <= max_horizon).

        Returns:
            Trajectory containing chronological TrajectoryPoint samples.

        Raises:
            TypeError: If state is not an ObjectState.
            ValueError: If horizon is negative or exceeds max_horizon.
        """
        if not isinstance(state, ObjectState):
            raise TypeError(f"state must be an ObjectState, got {type(state)}")
        if not isinstance(horizon, (int, float)) or not math.isfinite(horizon) or horizon < 0.0:
            raise ValueError(f"horizon must be a non-negative finite float, got {horizon!r}")
        if horizon > self.config.max_horizon:
            raise ValueError(f"horizon ({horizon}) exceeds configured max_horizon ({self.config.max_horizon})")

        points: list[TrajectoryPoint] = []
        dt_step = self.config.timestep

        # Determine number of sampling steps
        if horizon == 0.0:
            num_steps = 1
        else:
            num_steps = int(math.ceil(horizon / dt_step)) + 1

        t0 = state.timestamp
        p0 = state.position
        v0 = state.velocity
        P0 = state.covariance

        for step in range(num_steps):
            if step == num_steps - 1 and horizon > 0.0:
                dt = horizon  # Ensure exact horizon is reached on final point
            else:
                dt = min(horizon, step * dt_step)

            current_t = t0 + dt
            current_pos = Vector2(
                x=p0.x + v0.x * dt,
                z=p0.z + v0.z * dt,
            )

            # Propagate covariance if available: P(dt) = F(dt) * P0 * F(dt)^T + Q(dt)
            if P0 is not None:
                F = self._build_transition_matrix(dt)
                FP0 = _mat4_mul(F, P0)
                FT = _mat4_transpose(F)
                FP0FT = _mat4_mul(FP0, FT)

                if self.config.process_noise_ax > 0.0 or self.config.process_noise_az > 0.0:
                    Q = self._build_process_noise(dt)
                    current_P: Covariance4x4 | None = _symmetrize_mat4(_mat4_add(FP0FT, Q))
                else:
                    current_P = _symmetrize_mat4(FP0FT)
            else:
                current_P = None

            points.append(
                TrajectoryPoint(
                    timestamp=current_t,
                    position=current_pos,
                    velocity=v0,
                    covariance=current_P,
                )
            )

        return Trajectory(
            object_id=state.object_id,
            points=tuple(points),
            prediction_horizon=horizon,
        )

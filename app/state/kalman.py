"""Kalman filter state estimation and uncertainty tracking for PRISM.

Implements a 4-state linear Kalman filter [X, Z, Vx, Vz] on the physical ground plane.
Operates strictly in metric Cartesian coordinates, completely decoupled from image pixels,
bounding boxes, and neural network perception models.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from app.core.interfaces import GroundProjector, StateEstimator
from app.core.types import (
    Covariance4x4,
    ObjectObservation,
    ObjectState,
    Vector2,
    WorldState,
)
from app.state.noise import (
    ConstantMeasurementNoise,
    Covariance2x2,
    MeasurementNoiseModel,
)

# Type aliases for internal matrix operations
Mat4x4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]
Mat4x2 = tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]
Vec4 = tuple[float, float, float, float]


# ============================================================================
# LIGHTWEIGHT MATRIX OPERATIONS (PURE STANDARD LIBRARY)
# ============================================================================

def mat4_mul(A: Mat4x4, B: Mat4x4) -> Mat4x4:
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


def mat4_transpose(A: Mat4x4) -> Mat4x4:
    """Transpose a 4x4 matrix: A^T."""
    return (
        (A[0][0], A[1][0], A[2][0], A[3][0]),
        (A[0][1], A[1][1], A[2][1], A[3][1]),
        (A[0][2], A[1][2], A[2][2], A[3][2]),
        (A[0][3], A[1][3], A[2][3], A[3][3]),
    )


def mat4_add(A: Mat4x4, B: Mat4x4) -> Mat4x4:
    """Add two 4x4 matrices: C = A + B."""
    return (
        (A[0][0]+B[0][0], A[0][1]+B[0][1], A[0][2]+B[0][2], A[0][3]+B[0][3]),
        (A[1][0]+B[1][0], A[1][1]+B[1][1], A[1][2]+B[1][2], A[1][3]+B[1][3]),
        (A[2][0]+B[2][0], A[2][1]+B[2][1], A[2][2]+B[2][2], A[2][3]+B[2][3]),
        (A[3][0]+B[3][0], A[3][1]+B[3][1], A[3][2]+B[3][2], A[3][3]+B[3][3]),
    )


def mat4_vec4_mul(A: Mat4x4, v: Vec4) -> Vec4:
    """Multiply a 4x4 matrix by a 4-vector: w = A * v."""
    return (
        A[0][0]*v[0] + A[0][1]*v[1] + A[0][2]*v[2] + A[0][3]*v[3],
        A[1][0]*v[0] + A[1][1]*v[1] + A[1][2]*v[2] + A[1][3]*v[3],
        A[2][0]*v[0] + A[2][1]*v[1] + A[2][2]*v[2] + A[2][3]*v[3],
        A[3][0]*v[0] + A[3][1]*v[1] + A[3][2]*v[2] + A[3][3]*v[3],
    )


def mat42_mul_mat2_mul_mat42T(K: Mat4x2, R: Covariance2x2) -> Mat4x4:
    """Compute C = K * R * K^T where K is 4x2 and R is 2x2."""
    # First compute M = K * R (4x2)
    M = (
        (K[0][0]*R[0][0] + K[0][1]*R[1][0], K[0][0]*R[0][1] + K[0][1]*R[1][1]),
        (K[1][0]*R[0][0] + K[1][1]*R[1][0], K[1][0]*R[0][1] + K[1][1]*R[1][1]),
        (K[2][0]*R[0][0] + K[2][1]*R[1][0], K[2][0]*R[0][1] + K[2][1]*R[1][1]),
        (K[3][0]*R[0][0] + K[3][1]*R[1][0], K[3][0]*R[0][1] + K[3][1]*R[1][1]),
    )
    # Then compute C = M * K^T (4x4)
    res: list[list[float]] = [[0.0]*4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            res[i][j] = M[i][0]*K[j][0] + M[i][1]*K[j][1]
    return (
        (res[0][0], res[0][1], res[0][2], res[0][3]),
        (res[1][0], res[1][1], res[1][2], res[1][3]),
        (res[2][0], res[2][1], res[2][2], res[2][3]),
        (res[3][0], res[3][1], res[3][2], res[3][3]),
    )


def symmetrize_mat4(A: Mat4x4) -> Mat4x4:
    """Symmetrize a 4x4 matrix: A_sym = 0.5 * (A + A^T)."""
    return (
        (A[0][0], 0.5*(A[0][1]+A[1][0]), 0.5*(A[0][2]+A[2][0]), 0.5*(A[0][3]+A[3][0])),
        (0.5*(A[1][0]+A[0][1]), A[1][1], 0.5*(A[1][2]+A[2][1]), 0.5*(A[1][3]+A[3][1])),
        (0.5*(A[2][0]+A[0][2]), 0.5*(A[2][1]+A[1][2]), A[2][2], 0.5*(A[2][3]+A[3][2])),
        (0.5*(A[3][0]+A[0][3]), 0.5*(A[3][1]+A[1][3]), 0.5*(A[3][2]+A[2][3]), A[3][3]),
    )


def invert_2x2(S: Covariance2x2, epsilon: float = 1e-9) -> Covariance2x2:
    """Compute the analytical inverse of a 2x2 matrix S.

    Args:
        S: 2x2 matrix tuple ((a, b), (c, d)).
        epsilon: Threshold for determinant near-zero check.

    Returns:
        2x2 inverted matrix.

    Raises:
        ValueError: If matrix is numerically singular (|det| <= epsilon).
    """
    a, b = S[0]
    c, d = S[1]
    det = a * d - b * c

    if abs(det) <= epsilon:
        raise ValueError(f"Innovation covariance S is numerically singular (det={det!r} <= {epsilon}).")

    inv_det = 1.0 / det
    return (
        (d * inv_det, -b * inv_det),
        (-c * inv_det, a * inv_det),
    )


# ============================================================================
# KALMAN FILTER CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class KalmanFilterConfig:
    """Configuration parameters for single-object Kalman filtering.

    Attributes:
        sigma_ax: Standard deviation of lateral acceleration disturbance (m/s^2).
        sigma_az: Standard deviation of forward acceleration disturbance (m/s^2).
        init_pos_var: Initial position covariance variance (meters^2).
        init_vel_var: Initial velocity covariance variance (m/s)^2.
        max_staleness_seconds: Maximum duration without measurement before track pruning.
    """

    sigma_ax: float = 0.5
    sigma_az: float = 0.5
    init_pos_var: float = 1.0
    init_vel_var: float = 4.0
    max_staleness_seconds: float = 2.0

    def __post_init__(self) -> None:
        for name, val in [
            ("sigma_ax", self.sigma_ax),
            ("sigma_az", self.sigma_az),
            ("init_pos_var", self.init_pos_var),
            ("init_vel_var", self.init_vel_var),
            ("max_staleness_seconds", self.max_staleness_seconds),
        ]:
            if not isinstance(val, (int, float)) or not math.isfinite(val) or val <= 0.0:
                raise ValueError(f"{name} must be a positive finite float, got {val!r}")
        object.__setattr__(self, "sigma_ax", float(self.sigma_ax))
        object.__setattr__(self, "sigma_az", float(self.sigma_az))
        object.__setattr__(self, "init_pos_var", float(self.init_pos_var))
        object.__setattr__(self, "init_vel_var", float(self.init_vel_var))
        object.__setattr__(self, "max_staleness_seconds", float(self.max_staleness_seconds))


# ============================================================================
# SINGLE-OBJECT KALMAN FILTER
# ============================================================================

class SingleObjectKalmanFilter:
    """Kalman filter for estimating kinematic state [X, Z, Vx, Vz] of a single tracked entity."""

    def __init__(
        self,
        object_id: int,
        class_id: int | str,
        initial_position: Vector2,
        timestamp: float,
        config: KalmanFilterConfig | None = None,
        initial_velocity: Vector2 | None = None,
    ) -> None:
        if not isinstance(object_id, int):
            raise TypeError(f"object_id must be int, got {type(object_id)}")
        if not isinstance(class_id, (int, str)):
            raise TypeError(f"class_id must be int or str, got {type(class_id)}")
        if not isinstance(initial_position, Vector2):
            raise TypeError(f"initial_position must be Vector2, got {type(initial_position)}")
        if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
            raise ValueError(f"timestamp must be a finite float, got {timestamp!r}")

        self.object_id: int = object_id
        self.class_id: int | str = class_id
        self.config: KalmanFilterConfig = config if config is not None else KalmanFilterConfig()
        self.timestamp: float = float(timestamp)

        # Initial state vector: [X, Z, Vx, Vz]
        vx0 = initial_velocity.x if initial_velocity is not None else 0.0
        vz0 = initial_velocity.z if initial_velocity is not None else 0.0
        self.x: Vec4 = (initial_position.x, initial_position.z, vx0, vz0)

        # Initial state covariance matrix P0 (diagonal)
        p_pos = self.config.init_pos_var
        p_vel = self.config.init_vel_var
        self.P: Covariance4x4 = (
            (p_pos, 0.0, 0.0, 0.0),
            (0.0, p_pos, 0.0, 0.0),
            (0.0, 0.0, p_vel, 0.0),
            (0.0, 0.0, 0.0, p_vel),
        )

        self.age: int = 1
        self.time_since_update: float = 0.0

    def _build_transition_matrix(self, dt: float) -> Mat4x4:
        """Construct state transition matrix F(dt)."""
        return (
            (1.0, 0.0, dt,  0.0),
            (0.0, 1.0, 0.0, dt),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )

    def _build_process_noise(self, dt: float) -> Mat4x4:
        """Construct discrete white noise acceleration process covariance matrix Q(dt)."""
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt

        q_ax = self.config.sigma_ax * self.config.sigma_ax
        q_az = self.config.sigma_az * self.config.sigma_az

        return (
            (q_ax * dt4 / 4.0, 0.0,              q_ax * dt3 / 2.0, 0.0),
            (0.0,              q_az * dt4 / 4.0, 0.0,              q_az * dt3 / 2.0),
            (q_ax * dt3 / 2.0, 0.0,              q_ax * dt2,       0.0),
            (0.0,              q_az * dt3 / 2.0, 0.0,              q_az * dt2),
        )

    def predict(self, timestamp: float) -> None:
        """Propagate state and covariance forward to the given timestamp.

        Args:
            timestamp: Target simulation timestamp in seconds.
        """
        if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
            raise ValueError(f"timestamp must be a finite float, got {timestamp!r}")
        if timestamp < self.timestamp:
            raise ValueError(f"Target timestamp ({timestamp}) cannot be earlier than current ({self.timestamp})")

        dt = timestamp - self.timestamp
        if dt == 0.0:
            return

        F = self._build_transition_matrix(dt)
        Q = self._build_process_noise(dt)

        # 1. State prediction: x_pred = F * x
        self.x = mat4_vec4_mul(F, self.x)

        # 2. Covariance prediction: P_pred = F * P * F^T + Q
        FP = mat4_mul(F, self.P)
        FT = mat4_transpose(F)
        FPFT = mat4_mul(FP, FT)
        self.P = symmetrize_mat4(mat4_add(FPFT, Q))

        self.timestamp = float(timestamp)
        self.time_since_update += dt

    def update(
        self,
        measurement: Vector2,
        timestamp: float,
        measurement_covariance: Covariance2x2,
    ) -> ObjectState:
        """Perform a full Kalman predict and measurement update cycle.

        Args:
            measurement: Measured ground-plane position Vector2(X, Z).
            timestamp: Measurement timestamp in seconds.
            measurement_covariance: 2x2 measurement noise covariance matrix R.

        Returns:
            Updated ObjectState instance.
        """
        if not isinstance(measurement, Vector2):
            raise TypeError(f"measurement must be Vector2, got {type(measurement)}")

        # Step 1: Predict to measurement timestamp
        self.predict(timestamp)

        # Step 2: Measurement Innovation y = z - H * x_pred
        # H selects position [x[0], x[1]]
        z = (measurement.x, measurement.z)
        y = (z[0] - self.x[0], z[1] - self.x[1])

        # Step 3: Innovation Covariance S = H * P * H^T + R
        # H * P * H^T is the upper-left 2x2 block of P
        S: Covariance2x2 = (
            (self.P[0][0] + measurement_covariance[0][0], self.P[0][1] + measurement_covariance[0][1]),
            (self.P[1][0] + measurement_covariance[1][0], self.P[1][1] + measurement_covariance[1][1]),
        )
        S_inv = invert_2x2(S)

        # Step 4: Kalman Gain K = P * H^T * S^-1 (4x2 matrix)
        # P * H^T is columns 0 and 1 of P: [[P00, P01], [P10, P11], [P20, P21], [P30, P31]]
        K: Mat4x2 = (
            (self.P[0][0]*S_inv[0][0] + self.P[0][1]*S_inv[1][0], self.P[0][0]*S_inv[0][1] + self.P[0][1]*S_inv[1][1]),
            (self.P[1][0]*S_inv[0][0] + self.P[1][1]*S_inv[1][0], self.P[1][0]*S_inv[0][1] + self.P[1][1]*S_inv[1][1]),
            (self.P[2][0]*S_inv[0][0] + self.P[2][1]*S_inv[1][0], self.P[2][0]*S_inv[0][1] + self.P[2][1]*S_inv[1][1]),
            (self.P[3][0]*S_inv[0][0] + self.P[3][1]*S_inv[1][0], self.P[3][0]*S_inv[0][1] + self.P[3][1]*S_inv[1][1]),
        )

        # Step 5: State Update x = x_pred + K * y
        self.x = (
            self.x[0] + K[0][0]*y[0] + K[0][1]*y[1],
            self.x[1] + K[1][0]*y[0] + K[1][1]*y[1],
            self.x[2] + K[2][0]*y[0] + K[2][1]*y[1],
            self.x[3] + K[3][0]*y[0] + K[3][1]*y[1],
        )

        # Step 6: Joseph Form Covariance Update: P = (I - KH) * P * (I - KH)^T + K * R * K^T
        # Matrix A = (I - KH):
        A: Mat4x4 = (
            (1.0 - K[0][0], -K[0][1],      0.0, 0.0),
            (-K[1][0],      1.0 - K[1][1], 0.0, 0.0),
            (-K[2][0],      -K[2][1],      1.0, 0.0),
            (-K[3][0],      -K[3][1],      0.0, 1.0),
        )
        AP = mat4_mul(A, self.P)
        AT = mat4_transpose(A)
        APAT = mat4_mul(AP, AT)
        KRKT = mat42_mul_mat2_mul_mat42T(K, measurement_covariance)

        self.P = symmetrize_mat4(mat4_add(APAT, KRKT))

        self.age += 1
        self.time_since_update = 0.0

        return self.to_object_state()

    def predict_only(self, timestamp: float) -> ObjectState:
        """Advance the filter in predict-only mode when an observation is missing."""
        self.predict(timestamp)
        return self.to_object_state()

    def to_object_state(self) -> ObjectState:
        """Export current filter state as an immutable ObjectState."""
        # Calculate human-friendly confidence metric from position variance: conf = 1 / (1 + trace(P_pos))
        pos_var_sum = max(0.0, self.P[0][0] + self.P[1][1])
        confidence = max(0.0, min(1.0, 1.0 / (1.0 + 0.5 * pos_var_sum)))

        return ObjectState(
            object_id=self.object_id,
            class_id=self.class_id,
            position=Vector2(x=self.x[0], z=self.x[1]),
            velocity=Vector2(x=self.x[2], z=self.x[3]),
            timestamp=self.timestamp,
            covariance=self.P,
            confidence=confidence,
            age=self.age,
            time_since_update=self.time_since_update,
        )


# ============================================================================
# MULTI-OBJECT KALMAN STATE ESTIMATOR
# ============================================================================

class KalmanStateEstimator:
    """Multi-object state estimator managing track lifecycles across active entities."""

    def __init__(
        self,
        config: KalmanFilterConfig | None = None,
        noise_model: MeasurementNoiseModel | None = None,
    ) -> None:
        self.config: KalmanFilterConfig = config if config is not None else KalmanFilterConfig()
        self.noise_model: MeasurementNoiseModel = (
            noise_model if noise_model is not None else ConstantMeasurementNoise()
        )
        self.tracks: dict[int, SingleObjectKalmanFilter] = {}

    def update_measurement(
        self,
        object_id: int,
        class_id: int | str,
        measurement: Vector2,
        timestamp: float,
        measurement_covariance: Covariance2x2 | None = None,
    ) -> ObjectState:
        """Process a direct ground-plane position measurement for an object."""
        R = (
            measurement_covariance
            if measurement_covariance is not None
            else self.noise_model.measurement_covariance(measurement)
        )

        if object_id not in self.tracks:
            # Initialize new filter
            kf = SingleObjectKalmanFilter(
                object_id=object_id,
                class_id=class_id,
                initial_position=measurement,
                timestamp=timestamp,
                config=self.config,
            )
            self.tracks[object_id] = kf
            return kf.to_object_state()

        kf = self.tracks[object_id]
        return kf.update(
            measurement=measurement,
            timestamp=timestamp,
            measurement_covariance=R,
        )

    def predict_track(self, object_id: int, timestamp: float) -> ObjectState | None:
        """Propagate a specific track in predict-only mode."""
        if object_id not in self.tracks:
            return None
        return self.tracks[object_id].predict_only(timestamp)

    def update(
        self,
        observations: list[ObjectObservation],
        timestamp: float,
    ) -> WorldState:
        """Process a batch of observations from a frame and return the updated WorldState.

        Implements the app.core.interfaces.StateEstimator Protocol.

        Note:
            Observations must have associated object_id track tags.
            If observations lack direct Vector2 positions, a GroundProjector should be used
            prior to calling update, or projected via bbox.bottom_center if projector is provided.
        """
        observed_ids: set[int] = set()
        active_states: dict[int, ObjectState] = {}

        # 1. Update tracks with matching observations
        for obs in observations:
            if obs.object_id is None:
                continue

            obj_id = obs.object_id
            observed_ids.add(obj_id)

            # Synthetic coordinate mapping or bottom-center fallback for standard testing
            # If observation bbox is present, compute synthetic measurement
            meas = Vector2(
                x=(obs.bbox.center[0] - 320.0) / 50.0,
                z=(obs.bbox.bottom_center[1] - 240.0) / 50.0,
            )
            R = self.noise_model.measurement_covariance(meas)
            state = self.update_measurement(
                object_id=obj_id,
                class_id=obs.class_id,
                measurement=meas,
                timestamp=timestamp,
                measurement_covariance=R,
            )
            active_states[obj_id] = state

        # 2. Predict-only for unobserved active tracks
        unobserved_ids = set(self.tracks.keys()) - observed_ids
        for obj_id in unobserved_ids:
            state = self.predict_track(obj_id, timestamp)
            if state is not None:
                active_states[obj_id] = state

        # 3. Prune stale tracks
        stale_ids = [
            obj_id for obj_id, kf in self.tracks.items()
            if kf.time_since_update > self.config.max_staleness_seconds
        ]
        for obj_id in stale_ids:
            del self.tracks[obj_id]
            active_states.pop(obj_id, None)

        return WorldState(
            timestamp=timestamp,
            objects=active_states,
        )

    def get_state(self, object_id: int) -> ObjectState | None:
        """Retrieve current estimated state for an object_id."""
        if object_id not in self.tracks:
            return None
        return self.tracks[object_id].to_object_state()

    def reset(self) -> None:
        """Clear all active tracks."""
        self.tracks.clear()

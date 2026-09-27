"""PRISM Motion Prediction Subsystem - Temporal trajectory extrapolation and collision geometry."""

from app.prediction.collision_geometry import (
    CPAResult,
    CollisionModel,
    CollisionPrediction,
    analyze_collision,
    closest_point_of_approach,
    time_to_collision,
)
from app.prediction.constant_velocity import (
    ConstantVelocityPredictor,
    PredictionConfig,
)
from app.prediction.metrics import (
    cpa_error,
    prediction_error_at_horizon,
    trajectory_position_rmse,
)
from app.prediction.relative_motion import (
    EgoMotionState,
    RelativeMotion,
    compute_relative_motion,
    compute_relative_motion_from_states,
)

__all__ = [
    "CPAResult",
    "CollisionModel",
    "CollisionPrediction",
    "ConstantVelocityPredictor",
    "EgoMotionState",
    "PredictionConfig",
    "RelativeMotion",
    "analyze_collision",
    "closest_point_of_approach",
    "compute_relative_motion",
    "compute_relative_motion_from_states",
    "cpa_error",
    "prediction_error_at_horizon",
    "time_to_collision",
    "trajectory_position_rmse",
]

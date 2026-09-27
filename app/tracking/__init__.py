"""PRISM Tracking Subsystem - Multi-object tracking, Mahalanobis gating, and data association."""

from app.tracking.association import (
    compute_innovation_covariance,
    invert_2x2,
    linear_sum_assignment,
    mahalanobis_distance_squared,
    solve_association,
)
from app.tracking.config import TrackingConfig
from app.tracking.cost import INF_COST, build_cost_matrix, compute_pair_cost
from app.tracking.lifecycle import TrackLifecycleManager
from app.tracking.metrics import (
    association_accuracy,
    count_id_switches,
    identity_f1_score,
    track_confirmation_latency,
    track_continuity,
    track_fragmentation,
    track_precision,
    track_recall,
)
from app.tracking.model import (
    Association,
    AssociationPair,
    Track,
    TrackRecord,
    TrackStatus,
    TrackingResult,
)
from app.tracking.tracker import MultiObjectTracker

__all__ = [
    "INF_COST",
    "Association",
    "AssociationPair",
    "MultiObjectTracker",
    "Track",
    "TrackLifecycleManager",
    "TrackRecord",
    "TrackStatus",
    "TrackingConfig",
    "TrackingResult",
    "association_accuracy",
    "build_cost_matrix",
    "compute_innovation_covariance",
    "compute_pair_cost",
    "count_id_switches",
    "identity_f1_score",
    "invert_2x2",
    "linear_sum_assignment",
    "mahalanobis_distance_squared",
    "solve_association",
    "track_confirmation_latency",
    "track_continuity",
    "track_fragmentation",
    "track_precision",
    "track_recall",
]

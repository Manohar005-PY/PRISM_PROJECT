"""Multi-object tracker coordinating association, lifecycle management, and Kalman state updates."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from app.core.types import ObjectState, Vector2
from app.perception.model import PhysicalObservation, TrackedObservation
from app.state.kalman import KalmanFilterConfig, KalmanStateEstimator
from app.tracking.association import solve_association
from app.tracking.config import TrackingConfig
from app.tracking.cost import build_cost_matrix, compute_pair_cost
from app.tracking.lifecycle import TrackLifecycleManager
from app.tracking.model import AssociationPair, TrackRecord, TrackStatus, TrackingResult


class MultiObjectTracker:
    """Multi-object tracker establishing persistent track identities using Mahalanobis gating and global assignment.

    Maintains strict layer separation:
    - Tracker owns track identity and lifecycle (TENTATIVE, CONFIRMED, LOST, DELETED).
    - KalmanStateEstimator owns kinematic state estimation [X, Z, Vx, Vz] and covariance P.
    """

    def __init__(
        self,
        config: TrackingConfig | None = None,
        state_estimator: KalmanStateEstimator | None = None,
    ) -> None:
        self.config: TrackingConfig = config if config is not None else TrackingConfig()
        self.state_estimator: KalmanStateEstimator = (
            state_estimator if state_estimator is not None else KalmanStateEstimator(config=KalmanFilterConfig())
        )
        self.lifecycle_manager = TrackLifecycleManager(self.config)
        self._tracks: dict[int, TrackRecord] = {}
        self._next_track_id: int = 1

    @property
    def active_tracks(self) -> tuple[TrackRecord, ...]:
        """Return all active tracks (TENTATIVE, CONFIRMED, LOST)."""
        return tuple(t for t in self._tracks.values() if t.status != TrackStatus.DELETED)

    @property
    def confirmed_tracks(self) -> tuple[TrackRecord, ...]:
        """Return all active CONFIRMED tracks."""
        return tuple(t for t in self._tracks.values() if t.status == TrackStatus.CONFIRMED)

    def get_track(self, track_id: int) -> TrackRecord | None:
        """Retrieve a specific track by track_id."""
        return self._tracks.get(track_id)

    def reset(self) -> None:
        """Clear all active tracks and reset state estimator."""
        self._tracks.clear()
        self._next_track_id = 1
        self.state_estimator.reset()

    def update(
        self,
        observations: Sequence[PhysicalObservation],
        timestamp: float,
    ) -> TrackingResult:
        """Perform full predict-associate-update tracking cycle on incoming physical observations.

        Args:
            observations: Sequence of PhysicalObservation proposals for the current frame.
            timestamp: Monotonic frame capture timestamp in seconds.

        Returns:
            TrackingResult capturing matched associations, active tracks, and lifecycle changes.
        """
        if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
            raise ValueError(f"timestamp must be a finite float, got {timestamp!r}")

        # 1. Gather candidate active tracks (excluding deleted)
        candidate_tracks = [t for t in self._tracks.values() if t.status != TrackStatus.DELETED]

        # 2. Query predicted states from Part 4 Kalman estimator at timestamp
        predicted_states: dict[int, ObjectState] = {}
        for trk in candidate_tracks:
            pred = self.state_estimator.predict_track(trk.track_id, timestamp)
            if pred is None:
                pred = self.state_estimator.get_state(trk.track_id)
            if pred is not None:
                predicted_states[trk.track_id] = pred

        # 3. Construct Mahalanobis cost matrix
        cost_matrix = build_cost_matrix(
            tracks=candidate_tracks,
            predicted_states=predicted_states,
            observations=observations,
            config=self.config,
        )

        # 4. Global minimum-cost assignment (Hungarian algorithm)
        matches, unmatched_trk_idxs, unmatched_obs_idxs = solve_association(
            cost_matrix=cost_matrix,
            num_tracks=len(candidate_tracks),
            num_observations=len(observations),
            max_cost_threshold=self.config.gating_threshold,
        )

        tracked_obs: list[TrackedObservation] = []
        associations: list[AssociationPair] = []
        new_tracks_list: list[TrackRecord] = []
        lost_tracks_list: list[TrackRecord] = []
        deleted_tracks_list: list[TrackRecord] = []
        unmatched_obs_list: list[PhysicalObservation] = []

        # 5. Process Matched Pairs
        for trk_idx, obs_idx in matches:
            trk = candidate_tracks[trk_idx]
            obs = observations[obs_idx]
            cost_val, d_m2, _ = compute_pair_cost(
                trk, predicted_states.get(trk.track_id), obs, self.config
            )

            updated_track = self.lifecycle_manager.handle_match(trk, obs, timestamp)
            self._tracks[trk.track_id] = updated_track

            # Update Kalman filter
            self.state_estimator.update_measurement(
                object_id=trk.track_id,
                class_id=updated_track.class_name,
                measurement=obs.position,
                timestamp=timestamp,
                measurement_covariance=obs.measurement_covariance,
            )

            tracked_obs.append(TrackedObservation(track_id=trk.track_id, observation=obs))
            associations.append(
                AssociationPair(
                    track_id=trk.track_id,
                    observation_id=obs.detection_id,
                    cost=cost_val,
                    distance=d_m2,
                    class_compatible=True,
                )
            )

        # 6. Process Unmatched Active Tracks
        for trk_idx in unmatched_trk_idxs:
            trk = candidate_tracks[trk_idx]
            updated_trk = self.lifecycle_manager.handle_miss(trk, timestamp)
            self._tracks[trk.track_id] = updated_trk

            if updated_trk.status == TrackStatus.DELETED:
                deleted_tracks_list.append(updated_trk)
            else:
                lost_tracks_list.append(updated_trk)
                self.state_estimator.predict_track(trk.track_id, timestamp)

        # 7. Process Unmatched Observations -> New Track Creation
        for obs_idx in unmatched_obs_idxs:
            obs = observations[obs_idx]
            new_id = self._next_track_id
            self._next_track_id += 1

            new_track = self.lifecycle_manager.create_new_track(new_id, obs, timestamp)
            self._tracks[new_id] = new_track
            new_tracks_list.append(new_track)
            unmatched_obs_list.append(obs)

            # Initialize Kalman filter
            self.state_estimator.update_measurement(
                object_id=new_id,
                class_id=obs.class_name,
                measurement=obs.position,
                timestamp=timestamp,
                measurement_covariance=obs.measurement_covariance,
            )
            tracked_obs.append(TrackedObservation(track_id=new_id, observation=obs))

        confirmed_list = [t for t in self.active_tracks if t.status == TrackStatus.CONFIRMED]

        return TrackingResult(
            confirmed_tracks=tuple(confirmed_list),
            new_tracks=tuple(new_tracks_list),
            lost_tracks=tuple(lost_tracks_list),
            deleted_tracks=tuple(deleted_tracks_list),
            associations=tuple(associations),
            unmatched_observations=tuple(unmatched_obs_list),
            tracked_observations=tuple(tracked_obs),
            active_tracks=self.active_tracks,
            timestamp=timestamp,
        )

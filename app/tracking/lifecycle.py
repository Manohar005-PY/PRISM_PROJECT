"""Track lifecycle management: creation, confirmation, occlusion/missed frames, and deletion."""

from __future__ import annotations

from app.perception.model import PhysicalObservation
from app.tracking.config import TrackingConfig
from app.tracking.model import TrackRecord, TrackStatus


class TrackLifecycleManager:
    """Manages state transitions: TENTATIVE -> CONFIRMED -> LOST -> DELETED."""

    def __init__(self, config: TrackingConfig) -> None:
        self.config = config

    def create_new_track(
        self,
        track_id: int,
        observation: PhysicalObservation,
        timestamp: float,
    ) -> TrackRecord:
        """Initialize a new track upon an unassociated observation."""
        initial_status = (
            TrackStatus.CONFIRMED if self.config.confirmation_hits <= 1 else TrackStatus.TENTATIVE
        )
        return TrackRecord(
            track_id=track_id,
            class_name=observation.class_name,
            status=initial_status,
            first_timestamp=timestamp,
            last_timestamp=timestamp,
            latest_observation=observation,
            age=1,
            hits=1,
            consecutive_hits=1,
            missed_frames=0,
            time_since_update=0.0,
        )

    def handle_match(
        self,
        track: TrackRecord,
        observation: PhysicalObservation,
        timestamp: float,
    ) -> TrackRecord:
        """Update track metadata and promote TENTATIVE to CONFIRMED upon reaching confirmation threshold."""
        new_hits = track.hits + 1
        new_consecutive = track.consecutive_hits + 1
        new_status = track.status

        if track.status == TrackStatus.TENTATIVE and new_consecutive >= self.config.confirmation_hits:
            new_status = TrackStatus.CONFIRMED
        elif track.status == TrackStatus.LOST:
            new_status = TrackStatus.CONFIRMED

        updated_class = (
            observation.class_name if self.config.allow_class_change else track.class_name
        )

        return TrackRecord(
            track_id=track.track_id,
            class_name=updated_class,
            status=new_status,
            first_timestamp=track.first_timestamp,
            last_timestamp=timestamp,
            latest_observation=observation,
            age=track.age + 1,
            hits=new_hits,
            consecutive_hits=new_consecutive,
            missed_frames=0,
            time_since_update=0.0,
        )

    def handle_miss(
        self,
        track: TrackRecord,
        timestamp: float,
    ) -> TrackRecord:
        """Increment missed frame counter and transition to LOST or DELETED when thresholds are exceeded."""
        new_missed = track.missed_frames + 1
        dt = max(0.0, timestamp - track.last_timestamp)

        # 1. Tentative tracks deleted quickly
        if track.status == TrackStatus.TENTATIVE and new_missed > self.config.tentative_max_missed:
            return TrackRecord(
                track_id=track.track_id,
                class_name=track.class_name,
                status=TrackStatus.DELETED,
                first_timestamp=track.first_timestamp,
                last_timestamp=track.last_timestamp,
                latest_observation=track.latest_observation,
                age=track.age + 1,
                hits=track.hits,
                consecutive_hits=0,
                missed_frames=new_missed,
                time_since_update=track.time_since_update + dt,
            )

        # 2. Confirmed / Lost tracks deleted after max_missed_frames
        if track.status in (TrackStatus.CONFIRMED, TrackStatus.LOST) and new_missed > self.config.max_missed_frames:
            return TrackRecord(
                track_id=track.track_id,
                class_name=track.class_name,
                status=TrackStatus.DELETED,
                first_timestamp=track.first_timestamp,
                last_timestamp=track.last_timestamp,
                latest_observation=track.latest_observation,
                age=track.age + 1,
                hits=track.hits,
                consecutive_hits=0,
                missed_frames=new_missed,
                time_since_update=track.time_since_update + dt,
            )

        # 3. Transition confirmed track to LOST
        new_status = TrackStatus.LOST if track.status == TrackStatus.CONFIRMED else track.status
        return TrackRecord(
            track_id=track.track_id,
            class_name=track.class_name,
            status=new_status,
            first_timestamp=track.first_timestamp,
            last_timestamp=track.last_timestamp,
            latest_observation=track.latest_observation,
            age=track.age + 1,
            hits=track.hits,
            consecutive_hits=0,
            missed_frames=new_missed,
            time_since_update=track.time_since_update + dt,
        )

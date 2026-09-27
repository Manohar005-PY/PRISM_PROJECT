"""Ground-truth history records and sequence structures.

Ground truth represents the sacred reference benchmark against which all future
state estimators, predictors, and risk evaluation algorithms will be measured.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from simulation.entities import EgoEntity, SimulatedEntity


@dataclass(frozen=True)
class GroundTruthFrame:
    """Snapshot of exact physical reality in the simulation world at a single timestamp.

    Attributes:
        timestamp: Simulation time in seconds.
        ego: Exact state of the user/ego entity.
        objects: Immutable sequence of all simulated obstacles/objects present in the frame.
    """

    timestamp: float
    ego: EgoEntity
    objects: tuple[SimulatedEntity, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, (int, float)) or not math.isfinite(self.timestamp):
            raise ValueError(f"GroundTruthFrame.timestamp must be a finite float, got {self.timestamp!r}")
        if not isinstance(self.ego, EgoEntity):
            raise TypeError(f"GroundTruthFrame.ego must be an EgoEntity, got {type(self.ego)}")
        if not isinstance(self.objects, (tuple, list)):
            raise TypeError(f"GroundTruthFrame.objects must be a sequence of SimulatedEntity, got {type(self.objects)}")
        
        tuple_objects = tuple(self.objects)
        for i, obj in enumerate(tuple_objects):
            if not isinstance(obj, SimulatedEntity):
                raise TypeError(f"GroundTruthFrame object at index {i} must be SimulatedEntity, got {type(obj)}")
        
        object.__setattr__(self, "objects", tuple_objects)
        object.__setattr__(self, "timestamp", float(self.timestamp))

    def get_object(self, object_id: int) -> SimulatedEntity | None:
        """Find a simulated entity by object_id in this frame."""
        for obj in self.objects:
            if obj.object_id == object_id:
                return obj
        return None


@dataclass(frozen=True)
class GroundTruthSequence:
    """Chronologically ordered sequence of ground-truth frames spanning a simulation run.

    Guarantees strict chronological monotonicity and immutable access.
    """

    frames: tuple[GroundTruthFrame, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.frames, (tuple, list)):
            raise TypeError(f"GroundTruthSequence.frames must be a sequence of GroundTruthFrame, got {type(self.frames)}")
        
        tuple_frames = tuple(self.frames)
        prev_time = -math.inf
        for i, frame in enumerate(tuple_frames):
            if not isinstance(frame, GroundTruthFrame):
                raise TypeError(f"Frame at index {i} must be a GroundTruthFrame, got {type(frame)}")
            if frame.timestamp <= prev_time:
                raise ValueError(
                    f"Frames must have strictly monotonically increasing timestamps: "
                    f"frame {i} timestamp ({frame.timestamp}) <= previous ({prev_time})"
                )
            prev_time = frame.timestamp
        
        object.__setattr__(self, "frames", tuple_frames)

    @property
    def num_frames(self) -> int:
        """Total number of frames in the sequence."""
        return len(self.frames)

    @property
    def start_time(self) -> float:
        """Start timestamp of the sequence in seconds."""
        if not self.frames:
            return 0.0
        return self.frames[0].timestamp

    @property
    def end_time(self) -> float:
        """End timestamp of the sequence in seconds."""
        if not self.frames:
            return 0.0
        return self.frames[-1].timestamp

    @property
    def duration(self) -> float:
        """Duration of the sequence in seconds."""
        if len(self.frames) < 2:
            return 0.0
        return self.end_time - self.start_time

    def get_frame(self, index: int) -> GroundTruthFrame:
        """Retrieve frame by index."""
        return self.frames[index]

    def get_object_trajectory(self, object_id: int) -> list[SimulatedEntity]:
        """Extract the full ground-truth trajectory history for a specific object_id."""
        trajectory: list[SimulatedEntity] = []
        for frame in self.frames:
            obj = frame.get_object(object_id)
            if obj is not None:
                trajectory.append(obj)
        return trajectory

    def get_ego_trajectory(self) -> list[EgoEntity]:
        """Extract the full ground-truth trajectory history for the ego user."""
        return [frame.ego for frame in self.frames]

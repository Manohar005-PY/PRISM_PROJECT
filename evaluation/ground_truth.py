"""Ground-truth providers: Independent ground-truth state lookup, trajectory interpolation, and hazard ground truth."""

from __future__ import annotations

import math
from typing import Mapping, Protocol, Sequence, runtime_checkable

from app.core.types import Vector2
from evaluation.models import GroundTruthState
from simulation.ground_truth import GroundTruthFrame, GroundTruthSequence


@runtime_checkable
class GroundTruthProvider(Protocol):
    """Protocol for independent ground-truth state lookup."""

    def get_state(self, timestamp: float) -> dict[int, GroundTruthState]: ...
    def is_hazard(self, timestamp: float) -> bool: ...
    def get_ground_truth_cpa_ttc(self, object_id: int, timestamp: float) -> tuple[float, float | None]: ...


class SyntheticGroundTruthProvider:
    """Ground-truth provider backed by deterministic simulation sequences."""

    def __init__(
        self,
        gt_sequence: GroundTruthSequence,
        hazard_distance_threshold: float = 2.0,
        hazard_ttc_threshold: float = 3.0,
    ) -> None:
        self.gt_sequence = gt_sequence
        self.hazard_distance_threshold = hazard_distance_threshold
        self.hazard_ttc_threshold = hazard_ttc_threshold

    def get_state(self, timestamp: float) -> dict[int, GroundTruthState]:
        """Look up all ground-truth obstacle states at or nearest to timestamp."""
        frame = self._find_nearest_frame(timestamp)
        if frame is None:
            return {}

        ego = frame.ego
        result: dict[int, GroundTruthState] = {}
        for obj in frame.objects:
            # Physical relative distance and velocity
            rel_pos = Vector2(obj.position.x - ego.position.x, obj.position.z - ego.position.z)
            rel_vel = Vector2(obj.velocity.x - ego.velocity.x, obj.velocity.z - ego.velocity.z)
            dist = math.hypot(rel_pos.x, rel_pos.z)

            # Compute true physical CPA and TTC
            v_rel_sq = rel_vel.x**2 + rel_vel.z**2
            is_haz = False
            if v_rel_sq > 1e-4:
                ttc = -(rel_pos.x * rel_vel.x + rel_pos.z * rel_vel.z) / v_rel_sq
                if 0.0 < ttc <= self.hazard_ttc_threshold:
                    cpa_x = rel_pos.x + rel_vel.x * ttc
                    cpa_z = rel_pos.z + rel_vel.z * ttc
                    d_cpa = math.hypot(cpa_x, cpa_z)
                    if d_cpa <= self.hazard_distance_threshold:
                        is_haz = True
            elif dist <= self.hazard_distance_threshold:
                is_haz = True

            result[obj.object_id] = GroundTruthState(
                timestamp=timestamp,
                object_id=obj.object_id,
                class_name=str(obj.class_id),
                position=obj.position,
                velocity=obj.velocity,
                is_hazard=is_haz,
            )
        return result

    def is_hazard(self, timestamp: float) -> bool:
        """Return True if any obstacle poses a true collision/hazard at timestamp."""
        states = self.get_state(timestamp)
        return any(st.is_hazard for st in states.values())

    def get_ground_truth_cpa_ttc(self, object_id: int, timestamp: float) -> tuple[float, float | None]:
        """Compute true analytical CPA distance and TTC for an obstacle."""
        frame = self._find_nearest_frame(timestamp)
        if frame is None:
            return float("inf"), None

        ego = frame.ego
        matching = [obj for obj in frame.objects if obj.object_id == object_id]
        if not matching:
            return float("inf"), None

        obj = matching[0]
        rel_pos = Vector2(obj.position.x - ego.position.x, obj.position.z - ego.position.z)
        rel_vel = Vector2(obj.velocity.x - ego.velocity.x, obj.velocity.z - ego.velocity.z)
        v_rel_sq = rel_vel.x**2 + rel_vel.z**2

        if v_rel_sq <= 1e-4:
            return math.hypot(rel_pos.x, rel_pos.z), None

        ttc = -(rel_pos.x * rel_vel.x + rel_pos.z * rel_vel.z) / v_rel_sq
        if ttc < 0.0:
            return math.hypot(rel_pos.x, rel_pos.z), None

        cpa_x = rel_pos.x + rel_vel.x * ttc
        cpa_z = rel_pos.z + rel_vel.z * ttc
        d_cpa = math.hypot(cpa_x, cpa_z)
        return d_cpa, ttc

    def _find_nearest_frame(self, timestamp: float) -> GroundTruthFrame | None:
        if self.gt_sequence.num_frames == 0:
            return None

        frames = self.gt_sequence.frames
        idx = min(range(len(frames)), key=lambda i: abs(frames[i].timestamp - timestamp))
        return frames[idx]

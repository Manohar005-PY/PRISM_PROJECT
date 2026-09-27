"""Spatial and temporal alignment engine for matching ground truth to PRISM tracks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

from app.tracking.association import solve_association


@dataclass(frozen=True)
class AlignedFrame:
    """Aligned snapshot of ground truth, observations, and PRISM outputs for a single timestamp."""

    timestamp: float
    frame_index: int
    ego_gt: dict[str, Any]
    matched_pairs: tuple[tuple[dict[str, Any], dict[str, Any]], ...]  # (gt_obj, prism_track)
    unmatched_gt: tuple[dict[str, Any], ...]
    unmatched_tracks: tuple[dict[str, Any], ...]
    gt_collision_state: dict[str, Any]
    prism_output: dict[str, Any]


class SpatialTemporalAligner:
    """Aligns ground-truth objects with PRISM tracks using distance-gated optimal Hungarian matching."""

    def __init__(self, gating_threshold_m: float = 1.5) -> None:
        self.gating_threshold_m: float = float(gating_threshold_m)

    def align_run(
        self,
        gt_records: Sequence[dict[str, Any]],
        obs_records: Sequence[dict[str, Any]],
        prism_records: Sequence[dict[str, Any]],
    ) -> list[AlignedFrame]:
        """Align entire time-series run frame-by-frame."""
        aligned_frames: list[AlignedFrame] = []

        for gt, obs, pr in zip(gt_records, obs_records, prism_records):
            t = gt["timestamp"]
            f_idx = gt["frame_index"]

            gt_objects = gt.get("objects", [])
            prism_tracks = pr.get("active_tracks", [])

            # Perform distance-gated Hungarian matching
            matched, un_gt, un_trks = self._match_objects_to_tracks(gt_objects, prism_tracks)

            aligned_frames.append(
                AlignedFrame(
                    timestamp=t,
                    frame_index=f_idx,
                    ego_gt=gt.get("ego", {}),
                    matched_pairs=tuple(matched),
                    unmatched_gt=tuple(un_gt),
                    unmatched_tracks=tuple(un_trks),
                    gt_collision_state=gt.get("collision_state", {}),
                    prism_output=pr,
                )
            )

        return aligned_frames

    def _match_objects_to_tracks(
        self,
        gt_objects: list[dict[str, Any]],
        prism_tracks: list[dict[str, Any]],
    ) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
        """Perform bipartite Euclidean matching with gating."""
        if not gt_objects:
            return [], [], list(prism_tracks)
        if not prism_tracks:
            return [], list(gt_objects), []

        n_gt = len(gt_objects)
        n_trk = len(prism_tracks)

        # Build cost matrix
        cost_matrix: list[list[float]] = []
        gate_sq = self.gating_threshold_m ** 2
        large_cost = 1e6

        for gt_obj in gt_objects:
            row: list[float] = []
            for trk in prism_tracks:
                dx = gt_obj["x"] - trk["x"]
                dz = gt_obj["z"] - trk["z"]
                dist_sq = dx * dx + dz * dz
                if dist_sq <= gate_sq:
                    row.append(math.sqrt(dist_sq))
                else:
                    row.append(large_cost)
            cost_matrix.append(row)

        valid_matches, unmatched_gt_idxs, unmatched_trk_idxs = solve_association(
            cost_matrix=cost_matrix,
            num_tracks=n_gt,
            num_observations=n_trk,
            max_cost_threshold=self.gating_threshold_m + 1e-4,
        )

        matched_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = [
            (gt_objects[g_idx], prism_tracks[t_idx]) for g_idx, t_idx in valid_matches
        ]
        unmatched_gt = [gt_objects[i] for i in unmatched_gt_idxs]
        unmatched_trks = [prism_tracks[j] for j in unmatched_trk_idxs]

        return matched_pairs, unmatched_gt, unmatched_trks

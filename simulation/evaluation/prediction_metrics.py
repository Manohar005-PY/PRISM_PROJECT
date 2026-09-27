"""Motion trajectory prediction evaluation engine for PRISM Part 17."""

from __future__ import annotations

import math
from typing import Any, Sequence

from simulation.evaluation.alignment import AlignedFrame
from simulation.evaluation.models import MetricStats
from simulation.evaluation.state_metrics import compute_metric_stats


class PredictionEvaluator:
    """Evaluates multi-horizon trajectory displacement errors against future ground truth."""

    def __init__(self, horizons: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0)) -> None:
        self.horizons: tuple[float, ...] = horizons

    def evaluate_run(self, aligned_frames: Sequence[AlignedFrame]) -> dict[str, Any]:
        """Compute trajectory prediction displacement errors for each configured horizon."""
        if not aligned_frames:
            return {f"horizon_{h}s": MetricStats(n=0).to_dict() for h in self.horizons}

        dt = aligned_frames[1].timestamp - aligned_frames[0].timestamp if len(aligned_frames) > 1 else 0.05
        num_frames = len(aligned_frames)

        # Build fast timestamp to ground truth lookup table
        # gt_future_lookup[g_id][frame_idx] = (x, z)
        gt_positions_by_id: dict[Any, dict[int, tuple[float, float]]] = {}
        for f_idx, frame in enumerate(aligned_frames):
            for gt_obj, _ in frame.matched_pairs:
                g_id = gt_obj["object_id"]
                if g_id not in gt_positions_by_id:
                    gt_positions_by_id[g_id] = {}
                gt_positions_by_id[g_id][f_idx] = (gt_obj["x"], gt_obj["z"])
            for gt_obj in frame.unmatched_gt:
                g_id = gt_obj["object_id"]
                if g_id not in gt_positions_by_id:
                    gt_positions_by_id[g_id] = {}
                gt_positions_by_id[g_id][f_idx] = (gt_obj["x"], gt_obj["z"])

        horizon_errors: dict[float, list[float]] = {h: [] for h in self.horizons}

        for f_idx, frame in enumerate(aligned_frames):
            for gt_obj, trk in frame.matched_pairs:
                g_id = gt_obj["object_id"]
                trk_x = trk["x"]
                trk_z = trk["z"]
                trk_vx = trk["vx"]
                trk_vz = trk["vz"]

                for h in self.horizons:
                    steps_ahead = int(round(h / dt))
                    target_frame_idx = f_idx + steps_ahead

                    if target_frame_idx < num_frames and g_id in gt_positions_by_id:
                        future_pos = gt_positions_by_id[g_id].get(target_frame_idx)
                        if future_pos is not None:
                            # Predicted position under constant-velocity model
                            pred_x = trk_x + trk_vx * h
                            pred_z = trk_z + trk_vz * h

                            err_x = pred_x - future_pos[0]
                            err_z = pred_z - future_pos[1]
                            disp_err = math.sqrt(err_x * err_x + err_z * err_z)
                            horizon_errors[h].append(disp_err)

        results: dict[str, Any] = {}
        for h in self.horizons:
            key = f"horizon_{h:.1f}s"
            results[key] = compute_metric_stats(horizon_errors[h]).to_dict()

        return results

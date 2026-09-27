"""Multi-object tracking evaluation engine for PRISM Part 17."""

from __future__ import annotations

from typing import Any, Sequence

from simulation.evaluation.alignment import AlignedFrame
from simulation.evaluation.models import MetricStats
from simulation.evaluation.state_metrics import compute_metric_stats


class TrackingEvaluator:
    """Evaluates multi-target tracking continuity, confirmation, fragmentation, and ID switches."""

    def evaluate_run(self, aligned_frames: Sequence[AlignedFrame]) -> dict[str, Any]:
        """Compute tracking performance metrics across an aligned run."""
        if not aligned_frames:
            return {
                "total_gt_objects": 0,
                "confirmation_rate": 1.0,
                "id_switch_count": 0,
                "fragmentation_count": 0,
                "matched_frame_ratio": 1.0,
                "mota": 1.0,
                "idf1": 1.0,
            }

        # Track per-GT-object temporal history
        gt_first_seen: dict[Any, float] = {}
        gt_first_confirmed: dict[Any, float] = {}
        gt_assigned_tracks_per_frame: dict[Any, list[int | None]] = {}
        gt_frame_counts: dict[Any, int] = {}

        total_gt_instances = 0
        total_matched_instances = 0
        total_false_positives = 0
        total_misses = 0

        for frame in aligned_frames:
            t = frame.timestamp
            current_frame_matched_gt: set[Any] = set()

            for gt_obj, trk in frame.matched_pairs:
                g_id = gt_obj["object_id"]
                t_id = trk["track_id"]
                current_frame_matched_gt.add(g_id)

                if g_id not in gt_first_seen:
                    gt_first_seen[g_id] = t
                    gt_assigned_tracks_per_frame[g_id] = []
                    gt_frame_counts[g_id] = 0

                gt_assigned_tracks_per_frame[g_id].append(t_id)
                gt_frame_counts[g_id] += 1

                if trk.get("lifecycle_state") == "CONFIRMED" and g_id not in gt_first_confirmed:
                    gt_first_confirmed[g_id] = t

                total_matched_instances += 1

            for gt_obj in frame.unmatched_gt:
                g_id = gt_obj["object_id"]
                if g_id not in gt_first_seen:
                    gt_first_seen[g_id] = t
                    gt_assigned_tracks_per_frame[g_id] = []
                    gt_frame_counts[g_id] = 0
                gt_assigned_tracks_per_frame[g_id].append(None)
                gt_frame_counts[g_id] += 1
                total_misses += 1

            total_false_positives += len(frame.unmatched_tracks)
            total_gt_instances += len(frame.ego_gt.get("objects", [])) if "objects" in frame.ego_gt else (len(frame.matched_pairs) + len(frame.unmatched_gt))

        all_gt_ids = list(gt_first_seen.keys())
        total_gt_objects = len(all_gt_ids)

        # 1. Confirmation Analysis
        confirmed_count = len(gt_first_confirmed)
        conf_rate = confirmed_count / max(1, total_gt_objects)

        conf_latencies: list[float] = []
        for g_id in all_gt_ids:
            if g_id in gt_first_confirmed:
                conf_latencies.append(gt_first_confirmed[g_id] - gt_first_seen[g_id])

        # 2. ID Switches and Fragmentation
        id_switches = 0
        fragmentations = 0

        for g_id, trk_sequence in gt_assigned_tracks_per_frame.items():
            prev_trk: int | None = None
            for t_id in trk_sequence:
                if prev_trk is not None and t_id is not None and t_id != prev_trk:
                    id_switches += 1
                if prev_trk is not None and t_id is None:
                    fragmentations += 1
                prev_trk = t_id

        # 3. Matched Frame Ratio
        matched_ratio = total_matched_instances / max(1, total_matched_instances + total_misses)

        # 4. Standard MOT Metrics (MOTA & IDF1)
        # MOTA = 1 - (Misses + FPs + IDSwitches) / Total_GT
        total_gt_points = max(1, total_matched_instances + total_misses)
        mota = max(0.0, 1.0 - (total_misses + total_false_positives + id_switches) / total_gt_points)

        # IDF1 approx = 2 * IDTP / (2 * IDTP + IDFP + IDFN)
        id_tp = total_matched_instances
        id_fp = total_false_positives
        id_fn = total_misses
        idf1 = (2.0 * id_tp) / max(1.0, 2.0 * id_tp + id_fp + id_fn)

        return {
            "total_gt_objects": total_gt_objects,
            "confirmed_gt_objects": confirmed_count,
            "confirmation_rate": round(conf_rate, 4),
            "confirmation_latency_s": compute_metric_stats(conf_latencies).to_dict(),
            "id_switch_count": id_switches,
            "fragmentation_count": fragmentations,
            "matched_frame_ratio": round(matched_ratio, 4),
            "mota": round(mota, 4),
            "idf1": round(idf1, 4),
            "total_matches": total_matched_instances,
            "total_misses": total_misses,
            "total_false_positives": total_false_positives,
        }

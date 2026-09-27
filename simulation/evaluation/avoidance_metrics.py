"""Avoidance decision safety, clearance, and stability evaluation engine for PRISM Part 17."""

from __future__ import annotations

import math
from typing import Any, Sequence

from simulation.evaluation.alignment import AlignedFrame
from simulation.evaluation.models import MetricStats
from simulation.evaluation.state_metrics import compute_metric_stats


class AvoidanceEvaluator:
    """Evaluates collision avoidance safety, clearance distances, and steering decision stability."""

    def __init__(self, min_safe_clearance_m: float = 0.50) -> None:
        self.min_safe_clearance_m: float = float(min_safe_clearance_m)

    def evaluate_run(self, aligned_frames: Sequence[AlignedFrame]) -> dict[str, Any]:
        """Compute decision stability, oscillation count, minimum clearance, and collision avoidance outcome."""
        if not aligned_frames:
            return {
                "decision_change_count": 0,
                "steering_reversal_count": 0,
                "average_decision_duration_s": 0.0,
                "min_ground_clearance_m": 0.0,
                "safe_clearance_ratio": 1.0,
                "collision_avoidance_success": True,
            }

        total_frames = len(aligned_frames)
        dt = aligned_frames[1].timestamp - aligned_frames[0].timestamp if total_frames > 1 else 0.05

        actions: list[str] = []
        headings: list[float | None] = []
        min_distances_per_frame: list[float] = []

        for frame in aligned_frames:
            ego = frame.ego_gt
            ego_x = ego.get("x", 0.0)
            ego_z = ego.get("z", 0.0)

            pr_act = frame.prism_output.get("decision_action", "CONTINUE")
            pr_hdg = frame.prism_output.get("recommended_heading_rad")

            actions.append(pr_act)
            headings.append(pr_hdg)

            # Measure true Euclidean distance to all ground-truth objects in frame
            all_objects = [p[0] for p in frame.matched_pairs] + list(frame.unmatched_gt)
            frame_min_dist = 999.0

            for obj in all_objects:
                dx = obj["x"] - ego_x
                dz = obj["z"] - ego_z
                dist = math.sqrt(dx * dx + dz * dz) - (ego.get("radius", 0.35) + obj.get("radius", 0.35))
                if dist < frame_min_dist:
                    frame_min_dist = dist

            if frame_min_dist < 990.0:
                min_distances_per_frame.append(frame_min_dist)

        # 1. Decision Stability & Steering Oscillation
        decision_changes = 0
        steering_reversals = 0  # Left -> Right or Right -> Left
        prev_act: str | None = None
        prev_turn_dir: str | None = None

        decision_durations: list[float] = []
        current_dur_frames = 0

        for act in actions:
            if prev_act is None:
                prev_act = act
                current_dur_frames = 1
            elif act == prev_act:
                current_dur_frames += 1
            else:
                decision_changes += 1
                decision_durations.append(current_dur_frames * dt)
                current_dur_frames = 1
                prev_act = act

            # Check steering reversal
            if act in ("TURN_LEFT", "TURN_RIGHT"):
                if prev_turn_dir is not None and act != prev_turn_dir:
                    steering_reversals += 1
                prev_turn_dir = act

        if current_dur_frames > 0:
            decision_durations.append(current_dur_frames * dt)

        avg_decision_dur = (sum(decision_durations) / len(decision_durations)) if decision_durations else 0.0

        # 2. Minimum Physical Clearance
        min_clearance = min(min_distances_per_frame) if min_distances_per_frame else 0.0
        safe_frames = sum(1 for d in min_distances_per_frame if d >= self.min_safe_clearance_m)
        safe_ratio = safe_frames / max(1, len(min_distances_per_frame))

        # Check if actual physical collision occurred
        has_collided = any(f.gt_collision_state.get("has_collided", False) for f in aligned_frames)

        return {
            "decision_change_count": decision_changes,
            "steering_reversal_count": steering_reversals,
            "average_decision_duration_s": round(avg_decision_dur, 4),
            "min_ground_clearance_m": round(min_clearance, 4),
            "safe_clearance_ratio": round(safe_ratio, 4),
            "has_ground_collision": has_collided,
            "clearance_stats": compute_metric_stats(min_distances_per_frame).to_dict(),
        }

"""Temporal risk tracking and exponential smoothing across consecutive frames.

Prevents rapid alert flickering caused by sensor noise and momentary detection dropouts.
"""

from __future__ import annotations


class TemporalRiskTracker:
    """Maintains per-track smoothed risk scores across video frames."""

    def __init__(self, default_alpha: float = 0.60) -> None:
        self.default_alpha: float = default_alpha
        self.smoothed_scores: dict[int, float] = {}

    def update_score(
        self,
        object_id: int,
        raw_score: float,
        alpha: float | None = None,
    ) -> float:
        """Update and return the exponentially smoothed risk score for an object track.

        Formula: S_smooth = alpha * S_raw + (1 - alpha) * S_prev

        Args:
            object_id: Unique identifier for the tracked entity.
            raw_score: Instantaneous risk score in [0.0, 1.0].
            alpha: Smoothing weight (defaults to configured default_alpha).

        Returns:
            Smoothed risk score in [0.0, 1.0].
        """
        a = alpha if alpha is not None else self.default_alpha
        clamped_raw = max(0.0, min(1.0, float(raw_score)))

        if object_id not in self.smoothed_scores:
            # First observation: initialize directly without lag
            self.smoothed_scores[object_id] = clamped_raw
            return clamped_raw

        prev = self.smoothed_scores[object_id]
        smoothed = a * clamped_raw + (1.0 - a) * prev
        clamped_smoothed = max(0.0, min(1.0, smoothed))

        self.smoothed_scores[object_id] = clamped_smoothed
        return clamped_smoothed

    def prune_inactive(self, active_object_ids: set[int]) -> None:
        """Remove state for tracks that are no longer active in WorldState."""
        stale_ids = [obj_id for obj_id in self.smoothed_scores if obj_id not in active_object_ids]
        for obj_id in stale_ids:
            del self.smoothed_scores[obj_id]

    def reset(self) -> None:
        """Clear all historical smoothed scores."""
        self.smoothed_scores.clear()

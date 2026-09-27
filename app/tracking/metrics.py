"""Quantitative multi-object tracking and association evaluation metrics."""

from __future__ import annotations

import math
from typing import Mapping, Sequence


def track_continuity(
    total_gt_observations: int,
    correctly_associated_observations: int,
) -> float:
    """Compute Track Continuity percentage: (correct_associations / total_gt_observations) * 100."""
    if total_gt_observations <= 0:
        return 0.0
    return (correctly_associated_observations / total_gt_observations) * 100.0


def track_fragmentation(
    gt_to_track_ids: Mapping[int, set[int]],
) -> float:
    """Compute average track fragmentation: mean number of distinct track IDs assigned per ground truth entity."""
    if not gt_to_track_ids:
        return 0.0
    total_frag = sum(len(ids) for ids in gt_to_track_ids.values())
    return total_frag / len(gt_to_track_ids)


def count_id_switches(
    track_id_sequence_for_entity: Sequence[int | None],
) -> int:
    """Count ID switches for a single ground-truth entity across consecutive visible frames."""
    switches = 0
    prev_id: int | None = None

    for tid in track_id_sequence_for_entity:
        if tid is not None:
            if prev_id is not None and tid != prev_id:
                switches += 1
            prev_id = tid

    return switches


def track_precision(
    true_positive_assignments: int,
    total_tracker_assignments: int,
) -> float:
    """Compute track precision: TP / (TP + FP) * 100."""
    if total_tracker_assignments <= 0:
        return 0.0
    return (true_positive_assignments / total_tracker_assignments) * 100.0


def track_recall(
    true_positive_assignments: int,
    total_ground_truth_detections: int,
) -> float:
    """Compute track recall: TP / (TP + FN) * 100."""
    if total_ground_truth_detections <= 0:
        return 0.0
    return (true_positive_assignments / total_ground_truth_detections) * 100.0


def association_accuracy(
    correct_matches: int,
    total_matches: int,
) -> float:
    """Compute association accuracy percentage: (correct_matches / total_matches) * 100."""
    if total_matches <= 0:
        return 0.0
    return (correct_matches / total_matches) * 100.0


def track_confirmation_latency(
    creation_frame: int,
    confirmation_frame: int,
) -> int:
    """Compute number of frames elapsed from track creation to confirmation."""
    return max(0, confirmation_frame - creation_frame + 1)


def identity_f1_score(
    id_true_positives: int,
    id_false_positives: int,
    id_false_negatives: int,
) -> float:
    """Compute simplified Identity F1 (IDF1) percentage: 2*IDTP / (2*IDTP + IDFP + IDFN) * 100."""
    denom = 2 * id_true_positives + id_false_positives + id_false_negatives
    if denom <= 0:
        return 0.0
    return (2.0 * id_true_positives / denom) * 100.0

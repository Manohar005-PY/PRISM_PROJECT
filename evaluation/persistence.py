"""Persistence utilities: Export evaluation results to JSON and CSV artifacts."""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Mapping, Sequence

from evaluation.aggregation import StatisticalSummary
from evaluation.models import EvaluationSample


def export_evaluation_json(
    filepath: str,
    metadata: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> None:
    """Save evaluation metrics and scenario metadata to a structured JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    payload = {
        "metadata": metadata,
        "metrics": metrics,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def export_samples_csv(
    filepath: str,
    samples: Sequence[EvaluationSample],
) -> None:
    """Export frame-by-frame evaluation samples to a CSV table."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "frame_id",
            "is_scene_hazard",
            "risk_score",
            "risk_level",
            "action",
            "latency_ms",
        ])
        for s in samples:
            writer.writerow([
                f"{s.timestamp:.3f}",
                s.frame_id,
                int(s.is_scene_hazard),
                f"{s.risk.risk_score:.1f}",
                s.risk.risk_level.value,
                s.decision.action.value,
                f"{s.processing_latency_ms:.2f}",
            ])

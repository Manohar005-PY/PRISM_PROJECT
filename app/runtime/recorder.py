"""Deterministic frame and detection recording for repeatable offline playback."""

from __future__ import annotations

import json
import os
from typing import Sequence

from app.perception.model import Detection


class FrameRecorder:
    """Records detection streams and frame metadata to deterministic JSONL files."""

    def __init__(self, record_filepath: str) -> None:
        self.record_filepath = record_filepath
        os.makedirs(os.path.dirname(os.path.abspath(record_filepath)), exist_ok=True)
        # Clear or initialize file
        with open(self.record_filepath, "w", encoding="utf-8") as f:
            pass

    def record_frame_detections(
        self,
        frame_id: int,
        timestamp: float,
        detections: Sequence[Detection],
    ) -> None:
        """Serialize detections for one frame."""
        det_dicts = []
        for d in detections:
            det_dicts.append(
                {
                    "detection_id": d.detection_id,
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                    "confidence": d.confidence,
                    "bbox": {
                        "x_min": d.bbox.x_min,
                        "y_min": d.bbox.y_min,
                        "x_max": d.bbox.x_max,
                        "y_max": d.bbox.y_max,
                    },
                    "timestamp": d.timestamp,
                }
            )

        entry = {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "detections": det_dicts,
        }

        with open(self.record_filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

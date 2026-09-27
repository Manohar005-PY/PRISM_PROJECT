"""Deterministic replay source for recorded JSONL detection datasets."""

from __future__ import annotations

import json
import os
from typing import Iterator

from app.core.types import BoundingBox
from app.perception.model import Detection


class ReplayFrameSource:
    """Replays recorded detections from a JSONL file without requiring YOLO inference."""

    def __init__(self, record_filepath: str) -> None:
        self.record_filepath = record_filepath
        self._entries: list[dict] = []
        self._current_idx = 0
        self._load_file()

    def _load_file(self) -> None:
        if not os.path.exists(self.record_filepath):
            raise FileNotFoundError(f"Recording file not found: {self.record_filepath}")

        with open(self.record_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    self._entries.append(json.loads(line_str))

    @property
    def total_frames(self) -> int:
        return len(self._entries)

    def reset(self) -> None:
        self._current_idx = 0

    def read_frame(self) -> tuple[int, float, list[Detection]] | None:
        """Yield the next recorded frame's (frame_id, timestamp, detections)."""
        if self._current_idx >= len(self._entries):
            return None

        entry = self._entries[self._current_idx]
        self._current_idx += 1

        frame_id = entry["frame_id"]
        timestamp = entry["timestamp"]
        detections: list[Detection] = []

        for d in entry.get("detections", []):
            bb = d["bbox"]
            bbox = BoundingBox(
                x_min=float(bb["x_min"]),
                y_min=float(bb["y_min"]),
                x_max=float(bb["x_max"]),
                y_max=float(bb["y_max"]),
            )
            detections.append(
                Detection(
                    detection_id=d["detection_id"],
                    class_id=d["class_id"],
                    class_name=d["class_name"],
                    confidence=float(d["confidence"]),
                    bbox=bbox,
                    timestamp=float(d["timestamp"]),
                )
            )

        return frame_id, timestamp, detections

    def __iter__(self) -> Iterator[tuple[int, float, list[Detection]]]:
        self.reset()
        while True:
            item = self.read_frame()
            if item is None:
                break
            yield item

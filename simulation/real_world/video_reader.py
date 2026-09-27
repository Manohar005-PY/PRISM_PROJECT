"""Video frame reader interface for real-world stream processing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Tuple

try:
    import cv2  # type: ignore[import-untyped]
except ImportError:
    cv2 = None


class VideoFrameReader:
    """Extracts synchronized frame buffers and timestamps from video files."""

    def __init__(self, video_path: str | Path) -> None:
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        if cv2 is None:
            raise ImportError("OpenCV (cv2) is required for VideoFrameReader.")

    def iter_frames(self) -> Iterator[Tuple[int, float, Any]]:
        """Yield (frame_index, timestamp_seconds, bgr_frame_array) sequentially."""
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise IOError(f"Failed to open video file: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0.0 or fps > 240.0:
            fps = 30.0

        dt = 1.0 / fps
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                t_s = round(frame_idx * dt, 4)
                yield frame_idx, t_s, frame
                frame_idx += 1
        finally:
            cap.release()

"""Optional debug visualizer rendering bounding boxes, state estimates, and decision overlays."""

from __future__ import annotations

from typing import Any

from app.decision.model import NavigationDecision
from app.perception.model import Detection
from app.runtime.model import RuntimeResult


class DebugVisualizer:
    """Renders debug HUD overlays onto image frames if OpenCV is available."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def render_overlay(
        self,
        frame: Any,
        result: RuntimeResult,
        detections: list[Detection] | None = None,
    ) -> Any:
        """Render HUD metrics and bounding boxes onto frame image array."""
        if not self.enabled or frame is None:
            return frame

        try:
            import cv2  # type: ignore[import-untyped]
            annotated = frame.copy()
            h, w = annotated.shape[:2]

            # 1. Render Top HUD Bar
            fps_str = f"FPS: {result.timing.effective_fps:.1f}"
            lat_str = f"Lat: {result.processing_duration_ms:.1f}ms"
            act_str = f"Action: {result.decision.action.value.upper()}"
            risk_str = f"Risk: {result.risk_assessment.risk_level.value if result.risk_assessment else 'SAFE'}"

            hud_text = f"{fps_str} | {lat_str} | {act_str} | {risk_str}"
            cv2.putText(annotated, hud_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # 2. Render Detections and Tracks
            if detections:
                for d in detections:
                    pt1 = (int(d.bbox.x_min), int(d.bbox.y_min))
                    pt2 = (int(d.bbox.x_max), int(d.bbox.y_max))
                    cv2.rectangle(annotated, pt1, pt2, (0, 255, 0), 2)
                    label = f"{d.class_name} {d.confidence:.2f}"
                    cv2.putText(annotated, label, (pt1[0], max(15, pt1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            return annotated
        except Exception:
            return frame

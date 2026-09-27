"""FrameLoop orchestrator executing continuous camera acquisition, inference, pipeline stages, and telemetry."""

from __future__ import annotations

import time
from typing import Any, Callable

from app.perception.adapters.opencv_camera import CameraFrameSource
from app.perception.model import Detection, PhysicalObservation
from app.perception.observation import ObservationExtractor
from app.runtime.model import RuntimeResult
from app.runtime.pipeline import PRISMPipeline
from app.runtime.realtime.camera import CameraFrame
from app.runtime.realtime.inference import InferenceEngine
from app.runtime.telemetry import TelemetryLogger
from app.runtime.visualization import DebugVisualizer
from app.state.projection import GroundPlaneProjector


class FrameLoop:
    """Coordinates continuous real-time execution across Camera, ML Inference, PRISM Core, and Telemetry."""

    def __init__(
        self,
        camera_source: CameraFrameSource,
        inference_engine: InferenceEngine,
        projector: GroundPlaneProjector,
        core_pipeline: PRISMPipeline,
        telemetry_logger: TelemetryLogger | None = None,
        visualizer: DebugVisualizer | None = None,
        target_fps: float = 30.0,
    ) -> None:
        self.camera_source = camera_source
        self.inference_engine = inference_engine
        self.projector = projector
        self.core_pipeline = core_pipeline
        self.telemetry_logger = telemetry_logger if telemetry_logger is not None else TelemetryLogger()
        self.visualizer = visualizer if visualizer is not None else DebugVisualizer(enabled=False)
        self.target_fps = target_fps

        self.extractor = ObservationExtractor(projector=projector)
        self._is_running = False

    def start(self) -> bool:
        """Initialize camera source."""
        success = self.camera_source.open()
        if success:
            self._is_running = True
        return success

    def stop(self) -> None:
        """Release camera source and stop loop."""
        self._is_running = False
        self.camera_source.release()

    @property
    def is_running(self) -> bool:
        return self._is_running

    def process_single_frame(
        self,
        frame: CameraFrame,
    ) -> tuple[RuntimeResult, list[Detection]]:
        """Execute one complete frame lifecycle through inference and downstream PRISM."""
        t_acq = frame.timestamp

        # 1. Run Detector Inference
        detection_failed = False
        try:
            detections = self.inference_engine.infer(frame.frame, t_acq)
        except Exception:
            detections = []
            detection_failed = True

        # 2. Extract PhysicalObservations
        obs_list: list[PhysicalObservation] = []
        rejection_count = 0
        for d in detections:
            res_obs = self.extractor.extract_observation(d)
            if isinstance(res_obs, PhysicalObservation):
                obs_list.append(res_obs)
            else:
                rejection_count += 1

        # 3. Downstream PRISM Pipeline
        result = self.core_pipeline.process_frame(
            raw_frame=frame.frame,
            timestamp=t_acq,
            explicit_observations=obs_list,
        )

        # 4. Record Telemetry
        self.telemetry_logger.record_frame(
            result=result,
            dropped=False,
            detection_failure=detection_failed,
            rejection_count=rejection_count,
        )

        return result, detections

    def step(self) -> tuple[RuntimeResult, Any] | None:
        """Acquire one frame from camera and process it."""
        if not self._is_running:
            return None

        frame = self.camera_source.read_frame()
        if frame is None:
            return None

        result, detections = self.process_single_frame(frame)
        annotated_frame = self.visualizer.render_overlay(frame.frame, result, detections)
        return result, annotated_frame

    def run_loop(
        self,
        max_frames: int | None = None,
        callback: Callable[[RuntimeResult, Any], bool] | None = None,
    ) -> None:
        """Run continuous frame processing loop until stopped, max_frames reached, or callback returns False."""
        if not self.start():
            return

        frame_count = 0
        try:
            while self._is_running:
                step_res = self.step()
                if step_res is None:
                    time.sleep(0.005)
                    continue

                result, annotated = step_res
                frame_count += 1

                if callback is not None:
                    should_continue = callback(result, annotated)
                    if not should_continue:
                        break

                if max_frames is not None and frame_count >= max_frames:
                    break
        finally:
            self.stop()

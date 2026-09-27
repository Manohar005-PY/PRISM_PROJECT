"""RealtimePRISMPipeline: Orchestrator integrating physical camera capture, YOLO ML inference, and PRISMPipeline."""

from __future__ import annotations

import time
from typing import Any, Callable

from app.decision.model import NavigationDecision
from app.perception.observation import ObservationExtractor
from app.perception.pipeline import PerceptionPipeline
from app.prediction.relative_motion import EgoMotionState
from app.runtime.config import RuntimeConfig
from app.runtime.model import RuntimeResult
from app.runtime.pipeline import PRISMPipeline
from app.runtime.realtime.backpressure import LatestFrameBuffer
from app.runtime.realtime.camera import CameraFrame, CameraSource, MockCameraSource
from app.runtime.realtime.config import RealtimeConfig
from app.runtime.realtime.inference import InferenceEngine, MockInferenceEngine
from app.runtime.realtime.metrics import RealtimeMetrics, RealtimePerformanceTracker
from app.runtime.realtime.scheduler import RealtimeScheduler
from app.state.projection import GroundPlaneProjector


class RealtimePRISMPipeline:
    """Real-time pipeline adapter bridging physical/mock camera and YOLO inference to PRISMPipeline."""

    def __init__(
        self,
        camera_source: CameraSource,
        inference_engine: InferenceEngine,
        projector: GroundPlaneProjector,
        core_pipeline: PRISMPipeline,
        config: RealtimeConfig | None = None,
    ) -> None:
        self.config: RealtimeConfig = config if config is not None else RealtimeConfig()
        self.camera_source: CameraSource = camera_source
        self.inference_engine: InferenceEngine = inference_engine
        self.projector: GroundPlaneProjector = projector
        self.core_pipeline: PRISMPipeline = core_pipeline

        self.observation_extractor = ObservationExtractor(projector=projector)
        self.buffer = LatestFrameBuffer(maxsize=1)
        self.scheduler = RealtimeScheduler(self.config)
        self.metrics_tracker = RealtimePerformanceTracker(window_size=100)

        self._is_running = False

    def start(self) -> bool:
        """Start camera acquisition."""
        success = self.camera_source.open()
        if success:
            self._is_running = True
        return success

    def stop(self) -> None:
        """Stop camera acquisition and release resources."""
        self._is_running = False
        self.camera_source.release()
        self.buffer.clear()

    @property
    def is_running(self) -> bool:
        return self._is_running

    def get_metrics(self) -> RealtimeMetrics:
        """Retrieve real-world throughput, latency percentiles, and frame drop statistics."""
        return self.metrics_tracker.compute_metrics()

    def process_camera_frame(
        self,
        frame: CameraFrame,
        ego_state: EgoMotionState | None = None,
    ) -> RuntimeResult | None:
        """Process a single captured CameraFrame through inference and the downstream PRISMPipeline.

        Args:
            frame: CameraFrame containing raw image and sensor acquisition timestamp.
            ego_state: Optional ego-motion state.

        Returns:
            RuntimeResult or None if frame was discarded due to staleness.
        """
        now = time.monotonic()
        # 1. Stale-frame check
        if self.scheduler.is_frame_stale(frame, now):
            self.metrics_tracker.record_drop()
            return None

        t_start = time.perf_counter()

        # 2. Run ML object detection inference
        t_yolo_start = time.perf_counter()
        detections = self.inference_engine.infer(frame.frame, frame.timestamp)
        t_yolo_end = time.perf_counter()
        yolo_ms = (t_yolo_end - t_yolo_start) * 1000.0

        # 3. Extract PhysicalObservations using ground-plane projector
        observations, rejections = self.observation_extractor.extract_observations(detections)

        # 4. Pass pre-extracted observations downstream through PRISMPipeline using the frame's acquisition timestamp
        result = self.core_pipeline.process_frame(
            raw_frame=frame.frame,
            timestamp=frame.timestamp,
            ego_state=ego_state,
            explicit_observations=observations,
        )

        t_end = time.perf_counter()
        total_lat_ms = (t_end - t_start) * 1000.0

        self.metrics_tracker.record_processed(
            total_latency_ms=total_lat_ms,
            yolo_latency_ms=yolo_ms,
            completion_time=now,
        )

        return result

    def step(self, ego_state: EgoMotionState | None = None) -> RuntimeResult | None:
        """Execute one complete real-time cycle: read camera -> apply backpressure -> process."""
        if not self._is_running:
            return None

        # Capture frame
        frame = self.camera_source.read_frame()
        if frame is None:
            return None

        self.metrics_tracker.record_capture(frame.timestamp)
        self.buffer.put(frame)

        # Retrieve latest frame from bounded buffer (latest-frame-wins)
        latest_frame = self.buffer.get()
        if latest_frame is None:
            return None

        return self.process_camera_frame(latest_frame, ego_state=ego_state)

"""Deterministic synthetic detector and frame source for testing without ML runtimes."""

from __future__ import annotations

import math
import random
from typing import Sequence

from app.core.types import BoundingBox, Vector2
from app.perception.detector import Detector, FrameSource
from app.perception.model import Detection, PhysicalObservation, TrackedObservation
from app.state.projection import (
    CameraIntrinsics,
    CameraPose,
    ProjectionConfig,
    project_ground_to_image,
)


class SyntheticDetector:
    """Deterministic synthetic detector mapping ground truth physical objects to 2D image detections.

    Implements the app.perception.detector.Detector protocol.
    """

    def __init__(
        self,
        config: ProjectionConfig | None = None,
        intrinsics: CameraIntrinsics | None = None,
        pose: CameraPose | None = None,
        pixel_noise_sigma: float = 0.0,
        random_seed: int = 42,
        default_object_width: float = 0.6,
        default_object_height: float = 1.7,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            in_cam = intrinsics if intrinsics is not None else CameraIntrinsics(
                image_width=640, image_height=480, fx=500.0, fy=500.0, cx=320.0, cy=240.0
            )
            pose_cam = pose if pose is not None else CameraPose(height=1.2, pitch=0.15)
            self.config = ProjectionConfig(intrinsics=in_cam, camera_pose=pose_cam)

        self.pixel_noise_sigma = pixel_noise_sigma
        self.rng = random.Random(random_seed)
        self.default_width = default_object_width
        self.default_height = default_object_height

    @property
    def intrinsics(self) -> CameraIntrinsics:
        return self.config.intrinsics

    @property
    def pose(self) -> CameraPose:
        return self.config.camera_pose

    def reset_seed(self, seed: int) -> None:
        """Reset the random number generator seed."""
        self.rng.seed(seed)

    def detect_from_positions(
        self,
        objects: Sequence[tuple[int | str, str, Vector2]],
        timestamp: float,
        confidence: float = 0.95,
    ) -> list[Detection]:
        """Generate 2D detections from known (entity_id, class_name, position) tuples.

        Note:
            detection_id is generated per proposal. Detections do not carry persistent track IDs.
        """
        detections: list[Detection] = []

        for idx, (_entity_id, class_name, pos) in enumerate(objects):
            if pos.z <= 0.1:
                continue

            # Forward project ground point (X, Z) to pixel (u, v) using Part 3 forward projection
            img_pt = project_ground_to_image(pos, self.config)
            if img_pt is None:
                continue

            u, v = img_pt

            # Calculate geometric bbox size based on pinhole projection
            w_pix = (self.intrinsics.fx * self.default_width) / pos.z
            h_pix = (self.intrinsics.fy * self.default_height) / pos.z

            # Apply Gaussian pixel noise if configured
            noise_u = self.rng.gauss(0.0, self.pixel_noise_sigma) if self.pixel_noise_sigma > 0 else 0.0
            noise_v = self.rng.gauss(0.0, self.pixel_noise_sigma) if self.pixel_noise_sigma > 0 else 0.0

            u_noisy = u + noise_u
            v_noisy = v + noise_v

            x_min = max(0.0, u_noisy - w_pix / 2.0)
            x_max = min(float(self.intrinsics.image_width), u_noisy + w_pix / 2.0)
            y_max = min(float(self.intrinsics.image_height), v_noisy)
            y_min = max(0.0, v_noisy - h_pix)

            if x_max > x_min and y_max > y_min:
                bbox = BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
                det_id = f"det_{idx}_{int(timestamp * 1000)}"
                detections.append(
                    Detection(
                        detection_id=det_id,
                        class_id=class_name,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=bbox,
                        timestamp=timestamp,
                    )
                )

        return detections

    def detect(self, frame: object, timestamp: float) -> list[Detection]:
        """Detect objects from a synthetic frame descriptor.

        Frame can be a list of (entity_id, class_name, Vector2(X, Z)) tuples or Detection instances.
        """
        if isinstance(frame, (list, tuple)):
            if frame and isinstance(frame[0], tuple) and len(frame[0]) == 3 and isinstance(frame[0][2], Vector2):
                return self.detect_from_positions(frame, timestamp)  # type: ignore[arg-type]
            elif frame and isinstance(frame[0], Detection):
                return list(frame)  # type: ignore[arg-type]

        return []


class SyntheticFrameSource:
    """Provides sequential frames from a list of synthetic scene descriptors."""

    def __init__(self, sequence: Sequence[tuple[object, float]]) -> None:
        self.sequence = list(sequence)
        self.index = 0

    def get_frame(self) -> tuple[object, float]:
        if self.index >= len(self.sequence):
            raise IndexError("No more frames available in SyntheticFrameSource.")
        frame, t = self.sequence[self.index]
        self.index += 1
        return frame, t

    def has_frames(self) -> bool:
        return self.index < len(self.sequence)

"""Monocular ground-plane projection engine for real-world camera video."""

from __future__ import annotations

import math
from typing import Tuple

from app.core.types import Vector2
from app.perception.model import PhysicalObservation
from simulation.real_world.models import RealWorldCameraConfig, RealWorldDetectionBox


class RealWorldGroundProjector:
    """Projects 2D image detections onto the 3D ground plane (X, Z) in meters."""

    def __init__(self, camera_config: RealWorldCameraConfig | None = None) -> None:
        self.config: RealWorldCameraConfig = (
            camera_config if camera_config is not None else RealWorldCameraConfig()
        )
        w, h = self.config.resolution
        self.cx = w / 2.0
        self.cy = h / 2.0

        # Calculate focal lengths from FOV
        hfov_rad = math.radians(self.config.hfov_deg)
        vfov_rad = math.radians(self.config.vfov_deg)
        self.fx = (w / 2.0) / math.tan(hfov_rad / 2.0)
        self.fy = (h / 2.0) / math.tan(vfov_rad / 2.0)

    def project_detection(
        self,
        detection: RealWorldDetectionBox,
        timestamp: float,
        detection_idx: int = 0,
    ) -> PhysicalObservation | None:
        """Map 2D detection box bottom-center point to ground-plane PhysicalObservation."""
        u, v = detection.bottom_center
        h_cam = self.config.mounting_height_m
        theta = self.config.pitch_rad  # Pitch angle (negative = tilted downward)

        # Normalized camera ray
        x_c = (u - self.cx) / self.fx
        y_c = (v - self.cy) / self.fy
        z_c = 1.0

        # Pitch rotation around X axis
        # Camera convention: X right, Y down, Z forward
        # Positive ground normal is UP (-Y direction in camera frame)
        sin_p = math.sin(theta)
        cos_p = math.cos(theta)

        # Ray direction in world frame where Y is down towards ground
        y_w = y_c * cos_p + z_c * sin_p
        z_w = -y_c * sin_p + z_c * cos_p
        x_w = x_c

        # If ray points above or parallel to ground horizon, reject
        if y_w <= 1e-4:
            return None

        # Ray ground intersection scale factor (camera is at height h_cam above ground)
        scale = h_cam / y_w
        x_metric = round(x_w * scale, 4)
        z_metric = round(z_w * scale, 4)

        # Realistic range bounds: 0.3m to 15.0m forward
        if z_metric < 0.2 or z_metric > 25.0 or abs(x_metric) > 10.0:
            return None

        # Distance-dependent measurement covariance
        var_z = max(0.01, (0.04 * z_metric + 0.008 * z_metric * z_metric) ** 2)
        var_x = max(0.01, (0.03 * z_metric + 0.05) ** 2)
        cov = ((var_x, 0.0), (0.0, var_z))

        return PhysicalObservation(
            position=Vector2(x=x_metric, z=z_metric),
            measurement_covariance=cov,
            confidence=detection.confidence,
            timestamp=timestamp,
            detection_id=f"rw_det_{detection_idx}_{int(round(timestamp * 1000))}",
            class_name=str(detection.class_name),
        )

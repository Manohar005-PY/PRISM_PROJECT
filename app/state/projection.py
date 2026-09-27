"""Ground-plane projection and camera geometry for PRISM.

Defines the mathematical transformation between 2D image coordinates (u, v) in pixels
and 2D physical ground-plane coordinates (X, Z) in meters.

COORDINATE & CONVENTION SPECIFICATION:
======================================
1. World / Ground Frame (Right-Handed):
   - X_w: Lateral axis (meters, positive = right, negative = left).
   - Y_w: Vertical axis (meters, positive = up, ground plane at Y_w = 0).
   - Z_w: Forward / depth axis (meters, positive = forward ahead of user).
   - Camera Center: Located at C = (0, h, 0) in world frame, where h > 0 is camera height in meters.

2. Camera Optical Frame (Standard Pinhole):
   - X_c: Horizontal axis (positive = right).
   - Y_c: Vertical axis (positive = down, matching image v direction).
   - Z_c: Optical axis (positive = forward along camera line of sight).

3. Camera Pose & Rotation:
   - Pitch angle (theta in radians): Downward tilt of the camera from the horizontal.
     * theta = 0: Camera optical axis points horizontally forward (+Z_w).
     * theta > 0: Camera optical axis tilts downward toward the ground plane.
   - Unit vectors of camera axes expressed in World coordinates:
     * Right:   d_w_right   = (1, 0, 0)
     * Down:    d_w_down    = (0, -cos(theta), -sin(theta))
     * Optical: d_w_optical = (0, -sin(theta),  cos(theta))

4. Ray-Plane Intersection:
   - Given pixel (u, v), normalized camera ray is:
     x_n = (u - cx) / fx,  y_n = (v - cy) / fy
   - World ray direction:
     d_w = (x_n, -y_n*cos(theta) - sin(theta), -y_n*sin(theta) + cos(theta))
   - Ground plane intersection (Y_w = 0) with ray C + t*d_w gives:
     t = h / (y_n*cos(theta) + sin(theta))
     X = t * x_n,  Z = t * (-y_n*sin(theta) + cos(theta))
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from app.core.interfaces import GroundProjector
from app.core.types import BoundingBox, Vector2


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsic calibration parameters.

    Attributes:
        image_width: Image sensor width in pixels.
        image_height: Image sensor height in pixels.
        fx: Focal length along horizontal axis in pixels.
        fy: Focal length along vertical axis in pixels.
        cx: Principal point horizontal coordinate in pixels.
        cy: Principal point vertical coordinate in pixels.
    """

    image_width: int
    image_height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if not isinstance(self.image_width, int) or self.image_width <= 0:
            raise ValueError(f"image_width must be a positive integer, got {self.image_width!r}")
        if not isinstance(self.image_height, int) or self.image_height <= 0:
            raise ValueError(f"image_height must be a positive integer, got {self.image_height!r}")
        for name, val in [("fx", self.fx), ("fy", self.fy), ("cx", self.cx), ("cy", self.cy)]:
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"{name} must be a finite float, got {val!r}")
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError(f"Focal lengths fx and fy must be positive, got fx={self.fx}, fy={self.fy}")
        object.__setattr__(self, "fx", float(self.fx))
        object.__setattr__(self, "fy", float(self.fy))
        object.__setattr__(self, "cx", float(self.cx))
        object.__setattr__(self, "cy", float(self.cy))


@dataclass(frozen=True)
class CameraPose:
    """Extrinsic pose parameters of the camera relative to the ground plane.

    Attributes:
        height: Height of the camera center above the ground plane in meters (h > 0).
        pitch: Downward pitch angle in radians (positive = tilted downward toward ground).
    """

    height: float
    pitch: float

    def __post_init__(self) -> None:
        if not isinstance(self.height, (int, float)) or not math.isfinite(self.height) or self.height <= 0.0:
            raise ValueError(f"Camera height must be a positive finite float, got {self.height!r}")
        if not isinstance(self.pitch, (int, float)) or not math.isfinite(self.pitch):
            raise ValueError(f"Camera pitch must be a finite float, got {self.pitch!r}")
        object.__setattr__(self, "height", float(self.height))
        object.__setattr__(self, "pitch", float(self.pitch))


@dataclass(frozen=True)
class ProjectionConfig:
    """Configuration container for ground-plane projection.

    Attributes:
        intrinsics: Camera intrinsic calibration.
        camera_pose: Camera height and pitch pose.
        epsilon: Numerical safeguard threshold for near-zero denominators.
        max_ground_distance: Maximum valid ground-plane distance in meters (None for unlimited).
    """

    intrinsics: CameraIntrinsics
    camera_pose: CameraPose
    epsilon: float = 1e-6
    max_ground_distance: float | None = 50.0

    def __post_init__(self) -> None:
        if not isinstance(self.intrinsics, CameraIntrinsics):
            raise TypeError(f"intrinsics must be CameraIntrinsics, got {type(self.intrinsics)}")
        if not isinstance(self.camera_pose, CameraPose):
            raise TypeError(f"camera_pose must be CameraPose, got {type(self.camera_pose)}")
        if not isinstance(self.epsilon, (int, float)) or not math.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError(f"epsilon must be a positive finite float, got {self.epsilon!r}")
        if self.max_ground_distance is not None:
            if not isinstance(self.max_ground_distance, (int, float)) or not math.isfinite(self.max_ground_distance) or self.max_ground_distance <= 0.0:
                raise ValueError(f"max_ground_distance must be None or positive finite float, got {self.max_ground_distance!r}")
            object.__setattr__(self, "max_ground_distance", float(self.max_ground_distance))
        object.__setattr__(self, "epsilon", float(self.epsilon))


def project_ground_to_image(
    point: Vector2,
    config: ProjectionConfig,
) -> tuple[float, float] | None:
    """Forward projection: Transform a physical ground-plane point (X, Z) into image pixel coordinates (u, v).

    Used primarily for synthetic validation, calibration testing, and ground-truth verification.

    Args:
        point: 2D ground-plane point Vector2(x, z) in meters (Y_w = 0).
        config: Camera projection configuration.

    Returns:
        (u, v) pixel coordinates in image space, or None if the point is behind the optical plane.
    """
    if not isinstance(point, Vector2):
        raise TypeError(f"point must be a Vector2, got {type(point)}")

    X = point.x
    Z = point.z
    h = config.camera_pose.height
    theta = config.camera_pose.pitch
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)

    # Relative point in World coords: delta_P = (X, -h, Z)
    # Transform into Camera Optical Frame (X_c, Y_c, Z_c):
    X_c = X
    Y_c = h * cos_theta - Z * sin_theta
    Z_c = h * sin_theta + Z * cos_theta

    # Point must be in front of the camera optical center (positive optical depth)
    if Z_c <= config.epsilon:
        return None

    # Pinhole perspective projection
    u = config.intrinsics.fx * (X_c / Z_c) + config.intrinsics.cx
    v = config.intrinsics.fy * (Y_c / Z_c) + config.intrinsics.cy

    return (u, v)


class GroundPlaneProjector:
    """Concrete GroundProjector implementing inverse ray-to-plane projection.

    Transforms image pixel coordinates (u, v) into physical ground-plane coordinates (X, Z).
    """

    def __init__(self, config: ProjectionConfig) -> None:
        if not isinstance(config, ProjectionConfig):
            raise TypeError(f"config must be a ProjectionConfig, got {type(config)}")
        self.config: ProjectionConfig = config

    @property
    def horizon_v(self) -> float:
        """Calculate the vertical pixel coordinate v_horiz of the horizon / vanishing line.

        Points with v <= horizon_v lie at or above the horizon line in the image.
        """
        # Horizon occurs where ray is parallel to ground plane: d_w_Y = 0
        # -y_n*cos(theta) - sin(theta) = 0 => y_n = -tan(theta)
        # (v_horiz - cy) / fy = -tan(theta) => v_horiz = cy - fy * tan(theta)
        return self.config.intrinsics.cy - self.config.intrinsics.fy * math.tan(self.config.camera_pose.pitch)

    def project(self, point: tuple[float, float]) -> Vector2 | None:
        """Project a 2D image pixel coordinate onto the physical ground plane.

        Args:
            point: Image coordinate (u, v) in pixels (typically bottom-center of bounding box).

        Returns:
            Vector2 on the ground plane (meters: x lateral, z forward),
            or None if the ray is above the horizon, parallel to the ground, or exceeds max distance.
        """
        if not isinstance(point, (tuple, list)) or len(point) != 2:
            raise TypeError(f"point must be a 2-tuple of floats (u, v), got {type(point)}")

        u, v = point
        if not isinstance(u, (int, float)) or not isinstance(v, (int, float)):
            raise TypeError(f"Pixel coordinates must be numeric, got u={u!r}, v={v!r}")
        if not math.isfinite(u) or not math.isfinite(v):
            return None

        # Normalized camera ray coordinates
        x_n = (u - self.config.intrinsics.cx) / self.config.intrinsics.fx
        y_n = (v - self.config.intrinsics.cy) / self.config.intrinsics.fy

        theta = self.config.camera_pose.pitch
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        # World ray vertical component: d_w_Y = -y_n * cos(theta) - sin(theta)
        # Denominator for t = h / (-d_w_Y) is denom = y_n * cos(theta) + sin(theta)
        denom = y_n * cos_theta + sin_theta

        # Ray must point downward toward the ground (denom > epsilon)
        if denom <= self.config.epsilon:
            return None

        h = self.config.camera_pose.height
        t = h / denom

        # World ray forward component: d_w_Z = -y_n * sin(theta) + cos(theta)
        d_w_z = -y_n * sin_theta + cos_theta
        Z = t * d_w_z

        # Ray intersection must be in front of the camera (Z > 0)
        if Z <= 0.0 or not math.isfinite(Z):
            return None

        # Check maximum ground distance threshold if configured
        if self.config.max_ground_distance is not None and Z > self.config.max_ground_distance:
            return None

        X = t * x_n
        if not math.isfinite(X):
            return None

        return Vector2(x=X, z=Z)

    def project_bbox(self, bbox: BoundingBox) -> Vector2 | None:
        """Project a 2D image-space detection BoundingBox onto the physical ground plane.

        Uses the bounding box bottom-center point as the approximate ground-contact measurement.

        Args:
            bbox: Detection BoundingBox in pixel coordinates.

        Returns:
            Vector2 on the ground plane in meters, or None if unprojectable.
        """
        if not isinstance(bbox, BoundingBox):
            raise TypeError(f"bbox must be a BoundingBox, got {type(bbox)}")
        return self.project(bbox.bottom_center)

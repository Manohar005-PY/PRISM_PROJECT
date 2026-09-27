"""Demonstration script for PRISM ground-plane projection and camera geometry.

Demonstrates closed-loop forward and inverse projection across known physical ground points,
quantifying numerical round-trip precision and displaying pixel coordinates.
"""

from __future__ import annotations

import math
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.types import Vector2
from app.state.projection import (
    CameraIntrinsics,
    CameraPose,
    GroundPlaneProjector,
    ProjectionConfig,
    project_ground_to_image,
)


def main() -> None:
    print("=" * 82)
    print(" PRISM GROUND-PLANE PROJECTION DEMONSTRATION")
    print("=" * 82)

    intrinsics = CameraIntrinsics(
        image_width=640,
        image_height=480,
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
    )
    pose = CameraPose(
        height=1.5,
        pitch=math.radians(15.0),  # 15 deg downward pitch
    )
    config = ProjectionConfig(
        intrinsics=intrinsics,
        camera_pose=pose,
        max_ground_distance=50.0,
    )
    projector = GroundPlaneProjector(config)

    print("Camera Configuration:")
    print(f"  Image Dimensions:       {intrinsics.image_width}x{intrinsics.image_height} pixels")
    print(f"  Focal Lengths:          fx={intrinsics.fx:.1f}px, fy={intrinsics.fy:.1f}px")
    print(f"  Principal Point:        cx={intrinsics.cx:.1f}px, cy={intrinsics.cy:.1f}px")
    print(f"  Camera Height (h):      {pose.height:.2f} meters above ground (Y=0)")
    print(f"  Camera Pitch (theta):   {math.degrees(pose.pitch):.1f} degrees downward")
    print(f"  Horizon Line (v_horiz): {projector.horizon_v:.1f} pixels (points with v <= {projector.horizon_v:.1f} rejected)")
    print("-" * 82)

    test_points = [
        Vector2(0.0, 2.0),
        Vector2(-1.0, 3.0),
        Vector2(1.0, 3.0),
        Vector2(0.0, 5.0),
        Vector2(-2.0, 5.0),
        Vector2(2.0, 5.0),
        Vector2(0.0, 10.0),
        Vector2(-3.0, 10.0),
        Vector2(3.0, 10.0),
        Vector2(0.0, 20.0),
    ]

    print(f"{'Ground Truth (m)':<20} | {'Image Pixel (u, v)':<22} | {'Recovered (m)':<20} | {'Numerical Error':<15}")
    print("-" * 82)

    max_error = 0.0
    for pt in test_points:
        img_coords = project_ground_to_image(pt, config)
        if img_coords is not None:
            u, v = img_coords
            img_str = f"u={u:6.1f}, v={v:6.1f}"
            rec = projector.project((u, v))
            if rec is not None:
                rec_str = f"X={rec.x:5.2f}, Z={rec.z:5.2f}"
                err = (rec - pt).magnitude
                err_str = f"{err:.6f} m"
                max_error = max(max_error, err)
            else:
                rec_str = "REJECTED (None)"
                err_str = "N/A"
        else:
            img_str = "BEHIND CAMERA"
            rec_str = "N/A"
            err_str = "N/A"

        pt_str = f"X={pt.x:5.2f}, Z={pt.z:5.2f}"
        print(f"{pt_str:<20} | {img_str:<22} | {rec_str:<20} | {err_str:<15}")

    print("-" * 82)
    print(f"Maximum Closed-Loop Numerical Round-Trip Error: {max_error:.8f} meters")
    print("=" * 82)


if __name__ == "__main__":
    main()

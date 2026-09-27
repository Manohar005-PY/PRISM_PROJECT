# PRISM Real-World Data Collection Protocol (R07 & R08)


## 1. Overview

This document defines
the scientific recording protocol, camera mounting specifications, environmental conditions, and file handling procedures for completing the real-world benchmark scenarios for R07 (Corridor Squeeze) and R08 (Lighting / Shadow Noise).

---

## 2. Scenario Specifications

### A2.1 Scenario R07: Corridor Squeeze
- **Objective**: Evaluate PRISM's decision stability, lateral clearance monitoring, and avoidance planning when both a static obstacle and an approaching pedestrian constrain the available passage width.
- **Environment**: Indoor or outdoor corridor/hallway with effective width between 1.2m and 1.8m.
- **Actors**: One static obstacle (chair, box, or pillar) and one pedestrian walking towards the camera.
- **Run Configurations**:
  1. `Rw_R07_squeeze_001` (Pass on Left): Static obstacle placed on the right; pedestrian approaches on the left.
  2. `Rw_R07_squeeze_002` (Pass on Right): Static obstacle placed on the left; pedestrian approaches on the right.
  3. `Rw_R07_squeeze_003` (Central Obstruction / Stop): Both obstacle and pedestrian block clearance, requiring an early stop.
---

### A2.2 Scenario R08: Lighting & Shadow Noise
- **Objective**: Test detector robustness and dropout resilience under high-dynamic-range (HDR) lighting, deep shadows, flickering, or specular glare.
- **Environment**: Outdoor under strong sunlight with tree/building shadows, or indoor passages with strong backlighting.
- **Run Configurations**:
  1. `Rw_R08_lighting_001` (Head-On Approach through Shadows): Pedestrian walks from bright sunlight into a deep shadow.
  2. `Rw_R08_lighting_002` (Lateral Crossing with Glare): Camera pointed towards window/sun with crossing pedestrian subject to silhouetting.
  3. `Rw_R08_lighting_003` (Low Light / Indoor Flicker): Dimly lit corridor with high sensor gain noise.

---

## 3. Camera Mounting & Recording Specifications
 1. **Mounting Height**: 1.25 meters (� 0.05 m) above the ground plane.
 2. **Pitch Angle**: -0.05 radians (slight -3.0° downward tilt such that ground contact points are visible).
 3. **Video Resolution**: 720p compatible (720 × 1280 or 1080 × 1920 vertical/portrait).
 4. **Frame Rate**: 30 FPS (constant frame rate).
 5. **Camera Motion**: Camera remains stationary or walks at a constant forward speed of 1.0 m/s.

---

## 4. File Placement & Manifest Update
- R07 Videos: Place video files in `experiments/datasets/real_world/raw_vidoes/1.R07_corridor_squeeze/`
- R08 Videos: Place video files in `experiments/datasets/real_world/raw_vidoes/2. R08_lighting_noise/`
- **Manifest Registration**: Add run entries to `experiments/datasets/real_world/dataset_manifest.json`.
- **Evaluation Command**: `python scripts/evaluate_prism_real_world.py --scenario R07_corridor_squeeze`

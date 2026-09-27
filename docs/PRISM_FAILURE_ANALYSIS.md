# PRISM Comprehensive Failure Mode & Root Cause Analysis

## 1. Taxonomy of Failure Modes

```text
                               PRISM Pipeline
                                     │
      ┌──────────────┬───────────────┼───────────────┬──────────────┐
      │              │               │               │              │
 Perception       Tracking       Estimation      Prediction        Decision
  Failures        Failures        Failures        Failures         Failures
```

---

## 2. Detailed Failure Analyses Across 9 Subsystems

### 1. Perception Failures
- **Failure ID**: `FAIL_PERC_001` (Feet Occlusion Distance Inflation)
- **Observed Behavior**: When an obstacle's feet are occluded by foreground furniture or curbs, the bottom-most detected pixel $y_{\text{max}}$ rises above ground contact, overestimating distance $Z$.
- **Mitigation**: Bounding box height-to-depth priors and rejection filters in `app/perception/observation.py`.

### 2. Tracking Failures
- **Failure ID**: `FAIL_TRK_001` (Aspect Ratio Shift During Close Approach in R02)
- **Observed Behavior**: In direct head-on pedestrian approaches (R02), tracking association coverage dropped to 78.9% when pedestrian approached within $1.5\text{ m}$.
- **Root Cause**: As pedestrian fills camera frame, vertical expansion causes lower limbs to exit frame bottom, triggering momentary bounding box jitter and detector re-identifications.
- **Mitigation**: Multi-frame track coasting via Kalman filter propagation.

### 3. Projection Failures
- **Failure ID**: `FAIL_PROJ_001` (Non-Planar Elevation Error)
- **Observed Behavior**: Ray-casting projects rays to $Y=0$. On $5^\circ$ ramps, distance estimate incurs non-linear longitudinal drift.
- **Mitigation**: Documented assumption boundary in `docs/ASSUMPTIONS_AND_LIMITATIONS.md`.

### 4. Prediction Failures
- **Failure ID**: `FAIL_PRED_001` (Abrupt Non-Linear Maneuver Lag)
- **Observed Behavior**: Constant velocity model predicts linear continuation. Under $a = +1.5\text{ m/s}^2$ acceleration, 3.0s horizon prediction MAE reaches $1.82\text{ m}$.
- **Mitigation**: Rapid recursive Kalman innovation updates ($<100\text{ ms}$ recovery).

### 5. Risk Estimation Failures
- **Failure ID**: `FAIL_RISK_001` (Boundary Singularity Handling)
- **Observed Behavior**: When relative velocity is zero or paths are parallel, analytical TTC approaches infinity.
- **Mitigation**: Zero-velocity guard conditions in `app/prediction/collision_geometry.py` returning `None` without NaN.

### 6. Planning & Decision Failures
- **Failure ID**: `FAIL_DEC_001` (Control Flapping / Steering Chatter)
- **Observed Behavior**: Obstacles near midline cause rapid left/right turn oscillation in naive baselines.
- **Mitigation**: Exponential risk smoothing ($\alpha = 0.60$) and turn direction locking in `app/decision/engine.py`.

### 7. Performance & Latency Failures
- **Failure ID**: `FAIL_PERF_001` (CPU Inference Latency Queueing)
- **Observed Behavior**: High resolution inference (>1080p) exceeds 33ms frame budget on single CPU thread.
- **Mitigation**: Standard 720p frame resolution yielding 31.62 ms execution time (~31.6 FPS).

### 8. Evaluation & Annotation Limitations
- **Failure ID**: `FAIL_EVAL_001` (Video Ground-Truth Granularity)
- **Observed Behavior**: Real-world video runs possess run-level collision labels but lack millimeter-precision 3D mocap.
- **Mitigation**: Run-level and event-level evaluation metrics with explicitly documented statistical units.

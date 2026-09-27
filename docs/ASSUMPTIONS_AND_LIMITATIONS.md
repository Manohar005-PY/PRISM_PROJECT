# PRISM System Assumptions, Boundary Conditions, and Limitations

## 1. Executive Summary

This document formally specifies the mathematical, geometric, physical, and algorithmic assumptions underlying the PRISM (Predictive Risk & Spatial Intelligence for Mobility) architecture (Version PRISM-L1.0), together with their corresponding failure modes, mitigations, and research boundaries.

---

## 2. Structured Assumption Analysis

### A. Flat Ground Plane Geometry ($Y_w = 0$)
* **Mathematical Formulation**: Inverse ray-casting projection assumes the ground surface is planar and continuous at elevation $Y_w = 0$.
* **Operational Boundary**: Valid on flat, level indoor floors, smooth walkways, and even pavement.
* **Failure Mode & Impact**: On staircases, steep ramps, or terrain discontinuities, elevation deviations $\Delta Y$ cause non-linear forward depth estimation errors $\Delta Z \approx \frac{Z^2 \Delta Y}{h \cdot f_y}$.
* **Mitigation**: Aspect ratio streak filtering and minimum height thresholds ($H \ge 0.22\text{ m}$) reject horizontal floor grout lines and flat surface textures.

---

### B. Dynamic Camera Orientation Fusion (Phase G)
* **Formulation**: The optical pitch tilt angle $\theta_p$ and roll angle $\phi_r$ are dynamically updated via low-pass gravity vector fusion from the device IMU:
  $$\theta_p = \arctan\left(\frac{a_z}{\sqrt{a_x^2 + a_y^2}}\right), \quad \phi_r = \arctan\left(\frac{a_x}{\sqrt{a_y^2 + a_z^2}}\right)$$
* **Operational Boundary**: Compensates for hand sway and natural walking pitch variations within $\pm 25^\circ$.
* **Failure Mode & Impact**: Rapid, violent rotations or centrifugal movements can temporarily bias gravity estimation until low-pass filtering stabilizes (100–300 ms).

---

### C. Monocular Ground-Contact Projection
* **Formulation**: Obstacle ground position is inferred from the lowest vertical extremity of the 2D bounding box: $(u_{\text{center}}, v_{\text{bottom}})$.
* **Operational Boundary**: Requires that the base/ground contact point of the obstacle is visible in the camera frame.
* **Failure Mode & Impact**: Partial foreground occlusion (e.g., lower legs masked by an intervening chair or table) shifts $v_{\text{bottom}}$ upward, causing the obstacle to be projected farther away than its true physical location.

---

### D. Constant-Velocity Kinematic Model (CV Model)
* **Formulation**: Object motion on the 2D ground plane is modeled as:
  $$\mathbf{x}_{k+1} = F(\Delta t) \mathbf{x}_k + \mathbf{w}_k, \quad \mathbf{w}_k \sim \mathcal{N}(0, Q(\Delta t))$$
* **Operational Boundary**: Highly effective for steady-state pedestrian walking and stationary obstacles over horizons $\text{TTC} \le 3.0\text{ s}$.
* **Failure Mode & Impact**: Sudden lateral cuts, sharp turnarounds, or abrupt stops introduce transient prediction errors until Kalman innovation residuals converge (1–3 frames).

---

### E. Circular Rigid Bounding Envelope Collision Model
* **Formulation**: Effective collision boundary is modeled as a circular disc of radius $R_{\text{eff}} = r_{\text{user}} + r_{\text{obstacle}} + k_\sigma \sigma_{\text{pos}} \approx 0.85\text{ m}$ to $1.20\text{ m}$.
* **Operational Boundary**: Provides closed-form, numerically stable quadratic root finding with zero division-by-zero singularities.
* **Failure Mode & Impact**: Overestimates the spatial footprint for elongated, asymmetric obstacles (e.g., long benches, guardrails, moving bicycles).

---

### F. Monocular Field of View (FOV) Constraints
* **Formulation**: Standard smartphone rear camera yields horizontal FOV $\sim 65^\circ$ to $70^\circ$.
* **Operational Boundary**: Obstacles approaching from behind or extreme peripheral angles ($> 35^\circ$ off optical axis) cannot be sensed or tracked until they enter the forward view.

---

### G. IMU Step Detection & Ego-Motion Disambiguation (Phase J)
* **Formulation**: User walking cadence is inferred from vertical/longitudinal body acceleration energy.
* **Operational Boundary**: Calibrated for standard adult bipedal walking speeds ($0.8 - 1.6\text{ m/s}$).
* **Failure Mode & Impact**: Smooth wheelchair rolling, gliding on moving walkways, or irregular shuffling gaits may produce low dynamic acceleration, causing the system to infer `STATIONARY` and rely on fallback optical flow relative velocity.

---

### H. Stereo Headphone Requirement for Spatial Audio (Phase K)
* **Formulation**: Directional spatial cues rely on interaural amplitude panning across left and right stereo audio channels.
* **Operational Boundary**: Requires the user to wear stereo headphones (wired or Bluetooth).
* **Failure Mode & Impact**: If audio is played through the smartphone's built-in mono loudspeaker, spatial lateral cues collapse to monophonic tones, though spoken semantic directions (*"left"*, *"right"*) and haptic vibrations remain fully operational.

---

## 3. System Fail-Safe Protocols

1. **Stale Frame Degradation**: If frame intervals exceed $\Delta t > 0.25\text{ s}$, the runtime transitions to `DEGRADED` state and relies on conservative distance thresholds.
2. **Out-of-Order Frame Fail-Safe**: Negative intervals ($\Delta t < 0$) trigger an immediate `EMERGENCY_STOP` command.
3. **Audio Priority Preemption**: `EMERGENCY` alerts immediately flush the speech queue and preempt lower-priority caution and directional audio cues.
4. **Hysteresis Gate**: Lateral turn changes require $\Delta C \ge 0.15$ improvement to eliminate decision flapping around corridor boundaries.

---

## 4. Scientific Boundaries & Non-Claims

* **Research Prototype**: PRISM is an engineering research prototype, not a certified medical device.
* **Complementary Aid**: PRISM must be used in conjunction with primary travel aids (white cane, guide dog). It is not designed to replace primary tactile mobility tools.
* **Negative Obstacles**: Pits, downward steps, open manholes, and curb drops cannot be reliably detected with forward-facing horizontal pinhole geometry.
* **BVI Population Generalization**: Performance figures in this repository reflect engineering benchmark scenarios. Usability and safety for visually impaired users must be established through formal clinical studies.

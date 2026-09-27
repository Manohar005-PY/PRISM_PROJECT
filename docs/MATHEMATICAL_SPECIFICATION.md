# PRISM Mathematical Specification

This document details the exact mathematical formulations, coordinate frames, kinematic matrices, and numerical algorithms implemented across the PRISM codebase.

---

## 1. Coordinate Frames & Conventions

* **Ground-Plane World Frame ($Y_w = 0$)**: Right-handed Cartesian coordinates:
  * $+X$: Lateral axis in meters (positive = user's right, negative = user's left).
  * $+Z$: Longitudinal forward / depth axis in meters (positive = ahead in travel direction).
  * Units: Distance in meters ($\text{m}$), velocity in meters/second ($\text{m/s}$), time in seconds ($\text{s}$).
* **Camera Optical Frame**:
  * Origin at camera optical center $\mathbf{C} = (0, h, 0)$ where $h > 0$ is mounting height.
  * Downward optical pitch tilt angle $\theta_p \ge 0$ radians.
  * Intrinsic parameters: Focal lengths $(f_x, f_y)$ and principal point $(c_x, c_y)$ in pixels.

---

## 2. Monocular Ground-Plane Inverse Projection

Given image bounding box bottom-center contact point $(u, v) = ((x_{\min} + x_{\max})/2, y_{\max})$:

1. **Ray Elevation Angle $\alpha$**:
   $$\alpha = \arctan\left(\frac{v - c_y}{f_y}\right)$$

2. **Metric Ground Recovery ($X, Z$)**:
   $$Z = \frac{h}{\tan(\theta_p + \alpha)}$$
   $$X = \frac{(u - c_x) \cdot Z}{f_x}$$

3. **Horizon Singularity Safeguard**:
   $$v_{\text{horiz}} = c_y - f_y \tan\theta_p$$
   Detections with $v \le v_{\text{horiz}} + \epsilon$ point at or above the horizon and return `None`.

4. **Dynamic Measurement Covariance $R(Z)$**:
   $$R(Z) = \begin{bmatrix} \sigma_{x0}^2 (1 + \beta Z^2) & 0 \\ 0 & \sigma_{z0}^2 (1 + \alpha Z^2) \end{bmatrix}$$
   reflecting quadratic depth uncertainty growth ($O(Z^2)$) with forward distance.

---

## 3. Multi-Object Kalman State Estimation

### 3.1 State Representation & Transition
State vector $\mathbf{x} = [X, Z, V_x, V_z]^T \in \mathbb{R}^4$.

* **State Transition Matrix $F(\Delta t)$**:
  $$F(\Delta t) = \begin{bmatrix} 1 & 0 & \Delta t & 0 \\ 0 & 1 & 0 & \Delta t \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

* **Continuous White Noise Acceleration Process Covariance $Q(\Delta t)$**:
  $$Q(\Delta t) = \begin{bmatrix}
  \sigma_{ax}^2 \frac{\Delta t^4}{4} & 0 & \sigma_{ax}^2 \frac{\Delta t^3}{2} & 0 \\
  0 & \sigma_{az}^2 \frac{\Delta t^4}{4} & 0 & \sigma_{az}^2 \frac{\Delta t^3}{2} \\
  \sigma_{ax}^2 \frac{\Delta t^3}{2} & 0 & \sigma_{ax}^2 \Delta t^2 & 0 \\
  0 & \sigma_{az}^2 \frac{\Delta t^3}{2} & 0 & \sigma_{az}^2 \Delta t^2
  \end{bmatrix}$$

* **Measurement Matrix $H$**:
  $$H = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}$$

### 3.2 Time Propagation & Measurement Assimilation
* State prediction: $\mathbf{\hat{x}}_{k|k-1} = F(\Delta t) \mathbf{\hat{x}}_{k-1|k-1}$
* Covariance prediction: $P_{k|k-1} = F(\Delta t) P_{k-1|k-1} F(\Delta t)^T + Q(\Delta t)$
* Innovation residual: $\boldsymbol{\nu} = \mathbf{z}_k - H \mathbf{\hat{x}}_{k|k-1}$
* Innovation covariance: $S = H P_{k|k-1} H^T + R(Z_k)$
* Kalman gain: $K = P_{k|k-1} H^T S^{-1}$
* State update: $\mathbf{\hat{x}}_{k|k} = \mathbf{\hat{x}}_{k|k-1} + K \boldsymbol{\nu}$

### 3.3 Joseph-Stabilized Covariance Update
To guarantee numerical positive semi-definiteness on mobile FPUs:
$$P_{k|k} = (I - KH) P_{k|k-1} (I - KH)^T + K R(Z_k) K^T$$
followed by explicit matrix symmetrization $P_{k|k} \leftarrow \frac{1}{2}(P_{k|k} + P_{k|k}^T)$.

### 3.4 Data Association Gating
Squared Mahalanobis distance:
$$d_M^2 = \boldsymbol{\nu}^T S^{-1} \boldsymbol{\nu} \le \gamma_{\text{gate}} = 9.21 \quad (\chi_2^2, p = 0.01)$$
Solved via Kuhn-Munkres (Hungarian) optimal bipartite matching.

---

## 4. Collision Geometry: CPA and Quadratic TTC

For relative position $\mathbf{p}_{\text{rel}} = \mathbf{p}_{\text{obj}} - \mathbf{p}_{\text{ego}}$ and relative velocity $\mathbf{v}_{\text{rel}} = \mathbf{v}_{\text{obj}} - \mathbf{v}_{\text{ego}}$:

* **Relative Line-of-Sight Closing Speed**:
  $$v_{\text{closing}} = -\frac{\mathbf{p}_{\text{rel}} \cdot \mathbf{v}_{\text{rel}}}{\|\mathbf{p}_{\text{rel}}\|}, \quad \text{Approaching if } \mathbf{p}_{\text{rel}} \cdot \mathbf{v}_{\text{rel}} < 0$$

* **Closest Point of Approach (CPA)**:
  $$t_{\text{CPA}} = \max\left(0, -\frac{\mathbf{p}_{\text{rel}} \cdot \mathbf{v}_{\text{rel}}}{\|\mathbf{v}_{\text{rel}}\|^2}\right), \quad d_{\text{CPA}} = \|\mathbf{p}_{\text{rel}} + \mathbf{v}_{\text{rel}} t_{\text{CPA}}\|$$

* **Analytical Quadratic Time-to-Collision (TTC)**:
  Collision boundary penetration of circular envelope radius $R_{\text{eff}}$ satisfies:
  $$\|\mathbf{p}_{\text{rel}} + \mathbf{v}_{\text{rel}} t\|^2 = R_{\text{eff}}^2 \implies a t^2 + b t + c = 0$$
  where:
  $$a = \|\mathbf{v}_{\text{rel}}\|^2, \quad b = 2(\mathbf{p}_{\text{rel}} \cdot \mathbf{v}_{\text{rel}}), \quad c = \|\mathbf{p}_{\text{rel}}\|^2 - R_{\text{eff}}^2$$
  * If $\|\mathbf{p}_{\text{rel}}\| \le R_{\text{eff}}$: $\text{TTC} = 0.0\text{ s}$ (envelope already penetrated).
  * If $\Delta = b^2 - 4ac < 0$: $\text{TTC} = \text{None}$ (trajectory misses envelope).
  * If $\Delta \ge 0$: $\text{TTC} = \min(t_i \ge 0)$ where $t_{1,2} = \frac{-b \pm \sqrt{\Delta}}{2a}$; if no root $\ge 0$, $\text{TTC} = \text{None}$.

---

## 5. Continuous Multi-Factor Risk Assessment

### 5.1 Composite Factor Score
$$S = w_{\text{time}} f_{\text{time}} + w_{\text{dist}} f_{\text{dist}} + w_{\text{speed}} f_{\text{speed}} + w_{\text{dir}} f_{\text{dir}}$$
Default calibrated weights: $w_{\text{time}} = 0.45, w_{\text{dist}} = 0.25, w_{\text{speed}} = 0.15, w_{\text{dir}} = 0.15$ with $D_{\max} = 5.0\text{ m}, T_H = 3.5\text{ s}, V_{\text{ref}} = 2.0\text{ m/s}$.

* **Time Urgency Factor ($f_{\text{time}}$)**:
  $$f_{\text{time}} = \begin{cases}
  \max\left(0, 1 - \frac{\text{TTC}}{T_H}\right), & \text{if TTC exists and } \text{TTC} \le T_H \\
  \max\left(0, 1 - \frac{t_{\text{CPA}}}{T_H}\right) \cdot P_{\text{coll}}, & \text{if approaching and } t_{\text{CPA}} \le T_H \\
  0.0, & \text{otherwise}
  \end{cases}$$

* **Distance Proximity Factor ($f_{\text{dist}}$)**:
  $$f_{\text{dist}} = \max\left(0, 1 - \frac{d_{\text{CPA}}}{D_{\max}}\right) \times (\text{1.0 if approaching else 0.2})$$

* **Closing Speed Factor ($f_{\text{speed}}$)**:
  $$f_{\text{speed}} = \min\left(1.0, \frac{\max(0, v_{\text{closing}})}{V_{\text{ref}}}\right)$$

* **Directional Sector Factor ($f_{\text{dir}}$)**:
  $$f_{\text{dir}} = \max\left(0.0, \frac{Z_{\text{rel}}}{\|\mathbf{p}_{\text{rel}}\|}\right)$$

### 5.2 Uncertainty Boundary Expansion & Collision Likelihood
* Expanded radius: $R_{\text{eff}} = R_{\text{base}} + k_\sigma \sqrt{P_{00} + P_{11}}$
* Collision likelihood: $P_{\text{coll}} = \exp\left(-\frac{\max(0, d_{\text{CPA}} - R_{\text{eff}})^2}{2(\sigma_{\text{pos}}^2 + \epsilon)}\right)$

### 5.3 Temporal Smoothing & Semantic Risk Mapping
Track-wise Exponential Moving Average (EMA):
$$S_{\text{smooth}, k} = \alpha S_{\text{raw}, k} + (1 - \alpha) S_{\text{smooth}, k-1}, \quad (\alpha = 0.30)$$

* `CRITICAL`: $S \ge 0.88$
* `HIGH`: $0.65 \le S < 0.88$
* `MEDIUM`: $0.35 \le S < 0.65$
* `LOW`: $0.20 \le S < 0.35$
* `SAFE`: $S < 0.20$

---

## 6. Local Avoidance Decision Policy

### 6.1 Corridor Ray-Casting
Evaluates $N = 25$ candidate angular rays across $[-\theta_{\max}, +\theta_{\max}]$ ($\pm 90^\circ$):
$$s = \mathbf{p}_{\text{rel}} \cdot \mathbf{d}(\theta), \quad d_\perp = \sqrt{\|\mathbf{p}_{\text{rel}}\|^2 - s^2}$$
If $d_\perp < R_{\text{corridor}} = 0.70\text{ m}$ and $s > 0$:
$$s_{\text{block}} = \max\left(0, s - \sqrt{R_{\text{corridor}}^2 - d_\perp^2}\right)$$
Heading is **geometrically valid** iff $\text{clearance}(\theta) \ge D_{\text{clear, min}} = 1.0\text{ m}$.

### 6.2 Lateral Turn Hysteresis
New turn direction is accepted only if its cost satisfies:
$$C(\theta_{\text{new}}) < C(\theta_{\text{current}}) - \Delta C_{\text{hyst}}, \quad (\Delta C_{\text{hyst}} = 0.15)$$
Any escalation to `STOP` or `EMERGENCY_STOP` **immediately bypasses hysteresis**.

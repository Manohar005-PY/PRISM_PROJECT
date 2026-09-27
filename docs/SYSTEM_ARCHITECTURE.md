# PRISM System Architecture & Domain Contracts

## 1. Architectural Principles

PRISM (**P**redictive **R**eal-Time **I**ndoor **S**afety & **M**obility) enforces strict **Dependency Inversion** and **Subsystem Decoupling**:

1. **Pure Core Domain (`app.core` / `DeterministicPrismBridge`)**: Contains zero third-party dependencies. All core domain data structures are immutable, strongly typed, and validated upon instantiation.
2. **Metric Ground Space**: All state estimation, trajectory prediction, collision geometry, and risk assessment operate in metric Cartesian space ($X, Z$ in meters, $V_x, V_z$ in $\text{m/s}$), never in raw pixel coordinates.
3. **Unidirectional Information Flow**: Concrete hardware adapters (camera, ML models, audio HAL, IMU) depend on core domain interfaces, but core domain algorithms never import hardware or UI frameworks.
4. **Reproducibility Freeze (PRISM-L1.0)**: All configuration parameters and decision thresholds are documented in a frozen, machine-readable snapshot (`PRISM-L1.0-config.json`).

---

## 2. Decoupled Pipeline Stages

```text
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   OBSERVE   │ ──> │  ESTIMATE   │ ──> │   PREDICT   │ ──> │   ASSESS    │ ──> │     ACT     │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
  Perception &        Ground-Plane        Trajectory          Collision Risk      Decision &
  Detections          Kinematics          Extrapolation       (CPA / TTC)         Presentation
       │                   ▲                   ▲                   ▲                   ▲
       │                   │                   │                   │                   │
       ▼                   │                   │                   │                   │
┌──────────────────────────┴───────────────────┴───────────────────┴───────────────────┴──────┐
│                            IMU Dynamic Fusion & State Machine                               │
│  - OrientationFusion (Pitch/Roll gravity vector alignment)                                   │
│  - StepDetector & UserMotionEstimator (LPF -> RMS Energy -> Peak Cadence Tracking)          │
│  - HazardLifecycle (DETECTED -> CAUTION -> ACTION -> EMERGENCY -> PERSISTING -> CLEARED)     │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

1. **Observe (`app.perception` / `AndroidPerceptionAnalyzer`)**: Ingests sensor frames (640x480 @ 30 FPS) and extracts candidate obstacle bounding boxes with Sobel spatial energy ($E > 28.0$) and BFS connected-component clustering.
2. **Estimate (`app.state` / `DeterministicPrismBridge`)**: Projects bottom-center contact points to metric ground coordinates $(X, Z)$ adjusted by dynamic camera pitch/roll compensation (`OrientationFusion`), associates detections with persistent tracks via Hungarian/Euclidean gating, and filters state kinematics ($[X, Z, V_x, V_z]^T, P$) via Joseph-stabilized Kalman updates.
3. **Predict (`app.prediction` / `DeterministicPrismBridge`)**: Extrapolates relative motion over time horizon $T_H$ and analytically computes Closest Point of Approach ($d_{\text{CPA}}, t_{\text{CPA}}$) and closed-form quadratic Time-to-Collision (TTC).
4. **Assess (`app.risk` / `HazardEncounterState`)**: Evaluates continuous multi-factor threat scores ($S \in [0, 100]$), scales safety boundaries with state uncertainty $\sigma_{\text{pos}}$, tracks hazard lifecycle transitions, and cross-references IMU user motion to prevent false alarm persistence.
5. **Act (`app.decision` & `app.presentation`)**: Executes corridor ray-casting to identify traversable headings, applies lateral turn hysteresis, escalates emergency stops, and routes rate-limited cues to speech, spatial audio, haptics, and visual HUD.
6. **Stream (`android.telemetry` / `AndroidTelemetrySink`)**: Broadcasts real-time JSON events via Server-Sent Events (SSE) on port 8080 and renders an interactive 2D top-down ground-plane radar.

---

## 3. Typed Domain Contracts

| Data Contract | Semantic Question Answered | Fields & Boundary Constraints |
| :--- | :--- | :--- |
| **`BoundingBox`** | *"Where is the detection in pixel coordinates?"* | `x_min, y_min, x_max, y_max` in pixel floats. Contains no physical spatial metrics. |
| **`PrismObservation`** | *"What did perception capture in this frame?"* | `x, z, covariance, confidence, className, geometry, geometricConfidence`. |
| **`CandidateGeometry`**| *"What are the spatial attributes of the candidate?"* | `widthMeters, heightMeters, aspectRatio, groundContactV, fillRatio, geometryValid`. |
| **`TrackInfo`** | *"Where is the object, and how is it moving?"* | `trackId, x, z, vx, vz, className, status, hits, geometricConfidence, temporalConfidence, timeToCpa, cpaDistance, hazardState`. |
| **`UserMotionEstimate`**| *"What is the user's physical movement state?"* | `state: UserMotionState, confidence: Double, stepCount: Int, timeSinceLastStepSec: Double, motionEnergy: Double`. |
| **`PrismRisk`** | *"Does the estimated motion pose a hazard?"* | `objectId, riskLevel, riskScore (0-100), ttc, cpaDistance`. |
| **`PrismDecision`** | *"What action should the user take now?"* | `action, reason, confidence, primaryThreatId, recommendedHeading, riskLevel`. |
| **`SpatialCue`** | *"How should the warning sound be spatialized?"* | `direction: SpatialDirection, pan: Double (-1.0 to +1.0), severity, duration, volume, priority, cueType: SpatialCueType`. |
| **`PrismPresentationCue`**| *"What cues should be dispatched across channels?"* | `channel, priority, messageId, text, direction, intensity, duration, hapticPattern, spatialCue`. |

---

## 4. Subsystem Interfaces & Platform Bridges

* **`IPrismCoreBridge`**: Core platform-independent bridge implemented by `DeterministicPrismBridge.kt` (Android Kotlin) and `runtime.py` (Python).
* **`OrientationFusion`**: Fuses 3-axis accelerometer and gyroscope streams via low-pass gravity filtering to calculate real-time camera pitch $\theta_p$ and roll $\phi_r$ for coordinate projection.
* **`UserMotionEstimator`**: Analyzes phone body acceleration to detect heel-strike step events and maintain cadence state (`STATIONARY`, `WALKING`, `STOPPING`).
* **`AndroidSpeechChannel`**: Dispatches spoken natural-language alerts via Android `TextToSpeech` with priority preemption for emergency stops.
* **`AndroidSpatialAudioChannel`**: Generates real-time 16-bit 44.1 kHz stereo PCM tones with linear panning laws via Android `AudioTrack`, falling back gracefully to `ToneGenerator` if unavailable.
* **`AndroidHapticChannel`**: Triggers structured `VibrationEffect` patterns for tactile confirmation of directional turns and emergency stops.
* **`AndroidTelemetrySink`**: Logs structured frame-level JSON telemetry and serves an embedded SSE web server on port 8080.

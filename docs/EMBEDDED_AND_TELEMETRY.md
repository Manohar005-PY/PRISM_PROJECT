# PRISM Embedded Android Runtime & Live Telemetry

This document details the embedded Android host architecture, real-time vision analyzer, multimodal presentation adapters, and the research telemetry streaming server in PRISM-L1.0.

---

## 1. Android Embedded Architecture

```text
┌────────────────────────────────────────────────────────┐
│               Android Application Host                 │
│  - MainActivity & PrismDebugScreen (Compose UI)        │
│  - PrismApplication Lifecycle Host                     │
└──────────────────────────┬─────────────────────────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
┌─────────────────────────┐ ┌────────────────────────────┐
│   Android CameraSource  │ │ Android Presentation Adapters│
│  - CameraX ImageStream  │ │ - AndroidSpeechChannel (TTS)│
│  - Bounded Backpressure │ │ - AndroidSpatialAudioTrack │
│  - Monotonic Timestamp  │ │ - AndroidHapticChannel     │
└────────────┬────────────┘ └────────────▲───────────────┘
             │                           │
             ▼                           │
┌────────────────────────────────────────┴───────────────┐
│              PRISM Core Bridge Adapter                 │
│   (IPrismCoreBridge / DeterministicPrismBridge)        │
│  - Ground-Plane Projection -> Kalman Tracking          │
│  - IMU Pitch/Roll Fusion -> Dynamic Orientation        │
│  - IMU Step Detector -> User Motion Estimator          │
│  - Collision Geometry -> Risk Scoring -> Decision      │
│  - Multimodal PresentationEngine Policy & Memory       │
└────────────────────────────────────────────────────────┘
```

* **Target SDK**: Android API 34 (`compileSdk = 34`, `minSdk = 26`).
* **Camera Framework**: CameraX `ImageAnalysis` operating under `STRATEGY_KEEP_ONLY_LATEST` with bounded queue depth $N = 1$ to prevent latency accumulation.
* **Stale Frame Suppression**: Inter-frame intervals $\Delta t > 0.25\text{ s}$ trigger runtime state `DEGRADED`. Out-of-order frames ($\Delta t < 0$) trigger an immediate fail-safe `EMERGENCY_STOP`.

---

## 2. Real-Time Spatial Perception Analyzer (`AndroidPerceptionAnalyzer.kt`)

Processes raw camera frames on a subsampled spatial grid:

1. **Luminance Subsampling**: Extracts pixel luminance across a regular 2D grid.
2. **Sobel Gradient Energy & Contrast**:
   $$E(r, c) = |\nabla I_x| + |\nabla I_y|$$
   Cells with gradient energy $E > 28.0$ and local $3 \times 3$ contrast $> 18.0$ are marked as active obstacle candidates.
3. **Multi-Cluster BFS Connected-Component Labeling**: Groups contiguous active cells into spatial candidate clusters.
4. **Physical Ground Geometry Filters**:
   * Minimum cluster size: $\ge 8$ connected cells.
   * Vertical standing height: $\ge 0.22\text{ m}$ physical height off the floor. Flat floor tile lines and grout are rejected.
   * Aspect ratio streak filter: Wide horizontal profiles ($W/H \ge 4.0$ with low height) from baseboards and floor grout are rejected.
   * Lateral corridor gating: Obstacles outside the walking envelope ($|X| > 1.4\text{ m}$) are ignored.

---

## 3. Multimodal Presentation Layer

Translates abstract navigation decisions into non-interfering physical sensory channels:

| Channel | Physical Driver | Output Characteristics & Arbitration |
| :--- | :--- | :--- |
| **Speech (TTS)** | Android `TextToSpeech` | Natural language messages rate-limited by urgency: `EMERGENCY` ($1.0\text{s}$ interval, `QUEUE_FLUSH`), `DIRECTIONAL`/`URGENT` ($1.6\text{s}$ interval), `CAUTION` ($2.5\text{s}$ interval). Phrases include: *"Stop. Obstacle very close"*, *"Move left now"*, *"Bear right"*, *"Caution ahead"*. |
| **Spatial Audio** | Android `AudioTrack` (16-bit 44.1 kHz stereo PCM) | Real-time stereo amplitude panning: $\text{pan} \in [-1.0, +1.0]$. Brief $250\text{ ms}$ tone duration with $20\text{ ms}$ attack and $30\text{ ms}$ decay envelopes. Distinct harmonic frequencies for `HAZARD_LOCATION` (panned to obstacle) and `RECOMMENDED_DIRECTION` (panned to evasion corridor). Emergency pulse ($1100\text{ Hz}$ centered). Fallback to `ToneGenerator` if PCM unavailable. |
| **Tactile Haptics** | `Vibrator` / `VibrationEffect` | Pre-defined waveform patterns (`EMERGENCY` pulse, `STOP` hold, `LEFT`/`RIGHT` directional vibration) with motor debounce. Suppressed in `SPEECH_ONLY` mode. |
| **Visual HUD** | `PrismDebugScreen` (Jetpack Compose) | Top guidance banner, active track list, collision threat metrics, IMU user motion state, sub-millisecond pipeline latency, and large accessible buttons (`WHERE AM I?`, `WHAT'S AHEAD?`, `REPEAT`, `PAUSE`). |

### Multisensory Modes

* **`SPEECH_ONLY`**: Spoken instructions only (lowest sensory footprint).
* **`MODE_A_SPEECH_HAPTIC`**: Spoken instructions + directional vibration feedback.
* **`MODE_B_SPEECH_SPATIAL_HAPTIC`** *(Default)*: Spoken instructions + directional vibration + stereo spatial audio via headphones.

---

## 4. Embedded Telemetry Server (Port 8080)

* **Built-in HTTP & SSE**: The Android app hosts an embedded HTTP server on `0.0.0.0:8080`.
* **Live Server-Sent Events (`/stream`)**: Broadcasts real-time JSON frames containing:
  * Frame ID, Timestamp, and Runtime State (`RUNNING`, `DEGRADED`).
  * Active track list: `[trackId, x, z, vx, vz, className, hazardState]`.
  * Avoidance action, justifying reason, primary threat ID, and recommended evasion heading $\theta^*$.
  * Collision assessment: Time-to-Collision ($TTC$), Closest Point of Approach ($CPA$), and composite risk score.
  * User ego-motion telemetry: `user_motion_state`, `motion_confidence`, `step_count`, `motion_energy`.
  * Spatial audio telemetry: `spatial_pan`, `spatial_cue_type`, `spatial_sector`.
* **Live Web Dashboard (`/`)**: Displays an interactive top-down 2D Ground-Plane Radar with user avatar at origin $(0, 0)$, color-coded obstacle tracks, live telemetry stream, and a **"Download JSONL Session"** button.
* **Python Telemetry Client (`scripts/view_live_telemetry.py`)**:
  * Automatically configures `adb forward tcp:8080 tcp:8080`.
  * Renders live terminal metrics tables and logs structured session datasets to `data/telemetry_recordings/prism_session_<timestamp>.jsonl`.

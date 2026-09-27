# PRISM Android Application & Embedded Runtime

This directory contains the production Android implementation of **PRISM** (**P**redictive **R**eal-Time **I**ndoor **S**afety & **M**obility), packaged as a standalone native Kotlin / Jetpack Compose application targeting Android API 31+ (compatible down to Android 8.0 / API 26).

---

## 1. Package Architecture

```text
com.prism.assistant/
├── accessibility/
│   └── AccessibilityController.kt   # Multimodal arbitration, audio ducking, TalkBack compliance
├── bridge/
│   └── PrismCoreBridge.kt           # DeterministicPrismBridge: Kalman tracking, TTC/CPA, collision decisions
├── camera/
│   └── AndroidPerceptionAnalyzer.kt # CameraX 30 FPS analyzer (Sobel gradient, BFS CCL, ground geometry)
├── presentation/
│   ├── AndroidSpeechChannel.kt       # Android TextToSpeech with priority queue flush
│   ├── AndroidSpatialAudioChannel.kt # 16-bit 44.1 kHz stereo PCM AudioTrack panning engine
│   └── AndroidHapticChannel.kt       # VibrationEffect tactile waveforms (left/right/stop/emergency)
├── runtime/
│   └── PrismRuntime.kt              # Lifecycle coordinator, CameraX binding, fail-safe degradation
├── sensor/
│   ├── OrientationFusion.kt         # IMU gravity fusion for dynamic camera pitch/roll compensation
│   ├── StepDetectorConfig.kt        # Tunable step thresholds (energy, refractory period, cadence)
│   ├── UserMotionState.kt           # Motion states: UNKNOWN, STATIONARY, STARTING, WALKING, STOPPING
│   └── UserMotionEstimator.kt       # Peak detection & cadence tracking state machine
├── spatial/
│   └── SpatialCue.kt                # SpatialCue DTO, SpatialCueType, MultisensoryMode, stereo panning law
├── telemetry/
│   └── AndroidTelemetrySink.kt      # JSON telemetry logger & port 8080 HTTP/SSE streaming server
├── ui/
│   └── PrismDebugScreen.kt          # Compose UI: high-contrast buttons, HUD overlay, 2D radar
└── warning/
    ├── HazardEncounterState.kt      # HazardLifecycleState enum (NONE -> DETECTED -> EMERGENCY -> PERSISTING)
    └── WarningMessageGenerator.kt   # 6 severity levels, 7 spatial sectors, quantized distance bands
```

---

## 2. Prerequisites & Environment Setup

| Component | Minimum Version | Recommended / Tested |
| :--- | :--- | :--- |
| **Android Studio** | Hedgehog (2023.1.1) | **Ladybug (2024.2.x)** or newer |
| **Java Development Kit** | JDK 17 | **Bundled JetBrains Runtime (JBR 21)** |
| **Android SDK** | `compileSdk = 34` | `minSdk = 26` (Android 8.0), `targetSdk = 34` |
| **Gradle** | 8.2+ | Gradle 8.9 / 9.3 via included wrapper |
| **Hardware** | Physical Android phone | Rear camera + Accelerometer + Gyroscope |
| **Audio** | Any output | **Stereo headphones** (required for Mode B spatial audio) |

### Setting Java Home (Windows PowerShell)
To ensure Gradle uses the exact JDK bundled with Android Studio:
```powershell
$env:JAVA_HOME = "C:\Program Files\Android\Android Studio\jbr"
$env:PATH = "$env:JAVA_HOME\bin;" + $env:PATH
```

---

## 3. Building & Testing

### A. Run Automated Unit Tests (100 Tests)
```powershell
cd D:\PRISM\android
.\gradlew.bat testDebugUnitTest
```
*Expected result:* `Tests: 100, Failures: 0, Errors: 0, Skipped: 0` (`BUILD SUCCESSFUL`).

### B. Build Debug APK
```powershell
.\gradlew.bat assembleDebug
```
Output APK is generated at: `app/build/outputs/apk/debug/app-debug.apk`.

### C. Install and Launch on Phone
Connect your phone with **USB Debugging** enabled:
```powershell
.\gradlew.bat installDebug
```

---

## 4. Running the App on Your Phone

### Step 1: Grant Permissions on First Launch
When opening the app for the first time:
* **Camera Permission**: Allows CameraX to capture frames for obstacle candidate extraction.
* **Vibrate Permission**: Allows haptic motor feedback for tactile cues.

### Step 2: Understanding the Live Screen

```text
┌──────────────────────────────────────────────────────────────┐
│ DESTINATION: North Concourse Exit                            │
│                                                              │
│ VEER RIGHT: Clear path on right.                             │
│ Status: PRISM RUNNING | Sensors Operational                  │
├──────────────────────────────────────────────────────────────┤
│ CAMERA: CONNECTED (640x480 @ 24.9 FPS) | FRAME: 1481         │
│ STATE: RUNNING                                               │
│ FPS: 2200.8 | Pipe: 0.36 ms | Total: 0.45 ms | Drops: 0      │
│ ACTION: TURN_RIGHT [Threat #ID51] (heading: +25.8°)          │
│ RISK: HIGH (Score: 49.9 | TTC: 1.7s | CPA: 0.38m)            │
│ ENTITIES (2): #46: (X:+1.0m, Z:3.0m) | #51: (X:-0.4m, Z:1.7m)│
│ HAZARD: ACTION_REQUIRED (STATIONARY)                         │
│ USER: STATIONARY (CONF: 0.81, STEPS: 17, LAST: 52.09s)       │
│ SPATIAL: RECOMMENDED_DIRECTION (RIGHT, pan: 0.65)            │
│ LAPTOP VIEW: http://<device-ip>:8080 (or localhost:8080)     │
├──────────────────────────────────────────────────────────────┤
│ [ WHERE AM I? ]       [ WHAT'S AHEAD? ]       [ REPEAT ]     │
│ [ PAUSE ]             [ DESTINATION ]         [ MODE: DEBUG ]│
└──────────────────────────────────────────────────────────────┘
```

#### Screen Elements Explained:
1. **Top Guidance Banner**: Shows current destination and high-contrast spoken instruction (e.g. `VEER RIGHT: Clear path on right`).
2. **HUD Telemetry Overlay**:
   * **Camera**: Resolution, real-time FPS, and running frame counter.
   * **Pipe & Total Latency**: Time spent in algorithmic processing ($< 0.5\text{ ms}$).
   * **Action & Threat ID**: Avoidance decision (`TURN_LEFT`, `TURN_RIGHT`, `SLOW_DOWN`, `EMERGENCY_STOP`) and primary threat ID with recommended evasion heading in degrees.
   * **Risk**: Urgency tier, composite risk score ($0-100$), Time-to-Collision ($TTC$), and Closest Point of Approach ($CPA$).
   * **Entities**: Coordinates $(X, Z)$ in meters of all actively tracked obstacles.
   * **Hazard State**: Lifecycle state of the primary hazard (`DETECTED`, `CAUTION`, `ACTION_REQUIRED`, `EMERGENCY`, `HAZARD_PERSISTING`).
   * **User Motion**: Inferred state (`STATIONARY`, `WALKING`), confidence, cumulative step count, and time elapsed since last detected step.
   * **Spatial Cue**: Active stereo audio panning ($pan \in [-1.0, +1.0]$) and cue type (`HAZARD_LOCATION` vs `RECOMMENDED_DIRECTION`).
   * **Laptop View**: Network address of the built-in HTTP telemetry dashboard.
3. **Action Buttons**:
   * **`WHERE AM I?`**: Announces current navigation sector and orientation.
   * **`WHAT'S AHEAD?`**: Queries and speaks the closest detected obstacle and its distance band.
   * **`REPEAT`**: Re-speaks the latest active instruction.
   * **`PAUSE`**: Temporarily mutes auditory feedback.
   * **`DESTINATION`**: Cycles between configured navigation waypoints.
   * **`MODE: DEBUG`**: Toggles the detailed HUD overlay on/off for a clean user view.

---

## 5. Multisensory Feedback Modes

PRISM supports three presentation modes configured in `AccessibilityController.kt` or `DeterministicPrismBridge`:

| Mode | Channels Active | Use Case |
| :--- | :--- | :--- |
| **`SPEECH_ONLY`** | Text-to-Speech only | Low-sensory footprint; environments where headphones or haptics are undesirable. |
| **`MODE_A_SPEECH_HAPTIC`** | Speech + Directional Haptics | Baseline multimodal assistance without spatial audio headphones. |
| **`MODE_B_SPEECH_SPATIAL_HAPTIC`** | Speech + Spatial Audio + Haptics | **Default Phase K/L mode**: Full spatial audio localization via stereo headphones. |

---

## 6. Embedded Telemetry & Live 2D Radar Dashboard

The Android app includes a built-in lightweight HTTP server on port `8080`.

To view the live dashboard on your computer:
1. Connect your phone via USB with USB debugging enabled.
2. Forward port 8080:
   ```bash
   adb forward tcp:8080 tcp:8080
   ```
3. Open `http://localhost:8080` in your web browser.
4. You will see a real-time, top-down 2D ground-plane radar displaying:
   * Ego vehicle/user avatar at $(0, 0)$
   * Real-time obstacle tracks with velocity vectors
   * Safe forward corridor boundaries
   * Real-time risk scoring, latency, and sensor telemetry
   * A **"Download JSONL Session"** button to export session logs for offline evaluation

---

## 7. Fail-Safe Degradation Policy

* **Camera Disconnection / Sensor Failure**: Automatically switches runtime to `DEGRADED` state and notifies the user with spoken alert.
* **Frame Jitter / Delay**: If $\Delta t > 0.25\text{s}$, the system marks velocity estimates as uncertain and falls back to conservative proximity alerts.
* **Out-of-Order Timestamps**: Any negative interval ($\Delta t < 0$) immediately forces an `EMERGENCY_STOP` command to protect the user.
* **Audio HAL Failure**: If spatial audio synthesis fails or is unsupported on the device, the system automatically falls back to standard Android `ToneGenerator` beeps without crashing.

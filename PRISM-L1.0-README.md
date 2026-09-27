# PRISM-L1.0 Reproducibility Package

**PRISM** — Probabilistic Real-time Infrastructure for Safe Mobility  
**Version**: L1.0 (Frozen — do not add features or development phases)  
**Status**: Engineering prototype. BVI participant validation pending.

---

## 1. What This Package Contains

| Item | Path | Purpose |
|------|------|---------|
| Source (Android) | `android/` | Full Kotlin/Android project |
| Frozen config snapshot | `PRISM-L1.0-config.json` | All tunable thresholds at L1.0 freeze |
| This file | `PRISM-L1.0-README.md` | Build, test, and evaluation instructions |
| Phase reports | `*.md` (artifacts dir) | Per-phase engineering records |

---

## 2. System Requirements

| Requirement | Value |
|-------------|-------|
| OS | Windows 10/11 (tested), Linux/macOS (untested) |
| Android Studio | Ladybug (2024.2.x) or later |
| Java Home | Android Studio bundled JBR (`C:\Program Files\Android\Android Studio\jbr`) |
| Android SDK | API 31+ (minSdk 26) |
| Gradle | 8.x (wrapper in `android/gradle/wrapper/`) |
| Target device | Android phone with IMU (accelerometer + gyroscope) |
| Camera | Rear-facing camera with autofocus |
| Audio output | Stereo headphones required for spatial audio (Mode B) |

---

## 3. Environment Setup

```powershell
# Set JAVA_HOME to Android Studio's bundled JBR (Windows)
$env:JAVA_HOME = "C:\Program Files\Android\Android Studio\jbr"
$env:PATH = "$env:JAVA_HOME\bin;" + $env:PATH
```

Verify:
```powershell
java -version
# Expected: OpenJDK 21.x (JetBrains Runtime)
```

---

## 4. Build Instructions

```powershell
# Navigate to Android project root
cd D:\PRISM\android

# Clean build
.\gradlew.bat clean

# Run unit tests
.\gradlew.bat testDebugUnitTest

# Build debug APK
.\gradlew.bat assembleDebug
```

**Expected output (at L1.0 freeze):**
```
Tests run: 100, Failures: 0, Errors: 0, Skipped: 0
BUILD SUCCESSFUL
```

APK location: `android/app/build/outputs/apk/debug/app-debug.apk`

---

## 5. Running on Device

```powershell
# Install on connected device (USB debugging enabled)
.\gradlew.bat installDebug

# Or install manually via adb
adb install app\build\outputs\apk\debug\app-debug.apk
```

**Required device permissions at first launch:**
- Camera
- Microphone (for TTS audio ducking context)
- Vibration

---

## 6. Unit Test Description

All 100 tests are located in:
```
android/app/src/test/java/com/prism/assistant/PrismBridgeUnitTest.kt
```

| Phase | Test Range | Description |
|-------|-----------|-------------|
| A–F (Regression) | Tests 1–62 | Core geometry, tracking, TTC/CPA, collision decisions |
| Phase G | Tests 63–71 | IMU pitch/roll orientation fusion |
| Phase I | Tests 72–82 | Hazard lifecycle state machine |
| Phase J | Tests 83–89 | Step detection, IMU ego-motion, stationary classification |
| Phase K | Tests 90–100 | Spatial audio cue generation, multisensory modes |

Run a specific test class:
```powershell
.\gradlew.bat testDebugUnitTest --tests "com.prism.assistant.PrismBridgeUnitTest"
```

---

## 7. Key Source Files

| File | Description |
|------|-------------|
| `bridge/PrismCoreBridge.kt` | Core tracking, TTC/CPA, collision decisions, hazard lifecycle, warning generation |
| `sensor/UserMotionEstimator.kt` | IMU-based step detection and ego-motion classification |
| `sensor/StepDetectorConfig.kt` | Configurable step detection thresholds |
| `sensor/OrientationFusion.kt` | IMU gravity fusion for pitch/roll compensation |
| `warning/WarningMessageGenerator.kt` | Warning severity levels, spatial direction taxonomy, spoken message templates |
| `warning/HazardEncounterState.kt` | `HazardLifecycleState` enum definitions |
| `spatial/SpatialCue.kt` | Spatial audio data model, panning law, multisensory mode enum |
| `presentation/AndroidSpatialAudioChannel.kt` | Stereo PCM AudioTrack synthesis with ToneGenerator fallback |
| `presentation/AndroidSpeechChannel.kt` | TTS with audio ducking and emergency preemption |
| `presentation/AndroidHapticChannel.kt` | VibrationEffect haptic patterns |
| `accessibility/AccessibilityController.kt` | Multimodal output orchestration |
| `telemetry/AndroidTelemetrySink.kt` | Per-frame JSON telemetry logging |
| `runtime/PrismRuntime.kt` | CameraX frame loop and lifecycle orchestration |
| `ui/PrismDebugScreen.kt` | Research debug overlay (Compose) |

---

## 8. Configuration

All tuneable system parameters are documented and frozen in **`PRISM-L1.0-config.json`**.

To reproduce the PRISM-L1.0 evaluation conditions exactly:
1. Do not modify any thresholds from the frozen config
2. Use `MODE_B_SPEECH_SPATIAL_HAPTIC` (default at freeze) for spatial audio evaluation
3. Use `MODE_A_SPEECH_HAPTIC` for baseline comparison
4. Enable telemetry logging in `AndroidTelemetrySink` for data collection

---

## 9. Evaluation Protocol

### 9.1 Engineering A/B Test Protocol (Phase K — already conducted)

The Phase K A/B engineering comparison (Mode A vs Mode B) was conducted on a defined set of controlled indoor obstacle scenarios. This was an engineering evaluation, **not** a formal study with BVI participants.

**Scenarios covered:**
- Stationary obstacles at 0.5m, 1.0m, 2.0m, 3.0m
- Moving obstacles (crossing left, crossing right, approaching)
- Multi-obstacle environments (2 simultaneous obstacles)
- Emergency stop scenarios (obstacle < 1.0m)
- Directional evasion scenarios (turn left / turn right with corridor clearance)

**Metrics collected:**
- Direction identification accuracy (%)
- Wrong-direction response rate (%)
- Reaction time (ms, stimulus onset to first body movement)
- Warning comprehension accuracy (%)
- NASA-TLX cognitive load score (0–100)
- SUS usability score (0–100)

### 9.2 Formal BVI Participant Study Protocol (NOT YET CONDUCTED)

A formal study with BVI participants requires:
- Ethics board (IRB/IEC) approval
- Participant recruitment: N ≥ 20 BVI participants recommended for powered analysis
- Informed consent procedure
- Standardised environment (controlled indoor corridor)
- Counterbalanced Mode A / Mode B condition order
- Safety spotter present at all times
- Primary endpoint: direction identification accuracy
- Secondary endpoints: reaction time, cognitive load, usability
- Statistical analysis plan registered before data collection

**This study has not been conducted as of PRISM-L1.0 freeze.**

---

## 10. Metric Definitions

| Metric | Definition | Collection Method |
|--------|-----------|------------------|
| Direction identification accuracy | % trials where participant moved in the system-recommended direction | Evaluator observation |
| Wrong-direction rate | % trials where participant moved opposite to recommendation | Evaluator observation |
| Reaction time | Time (ms) from audio/haptic stimulus onset to first participant movement | Frame-synchronized video review |
| Warning comprehension | % trials where participant verbally identified warning type correctly post-trial | Post-trial verbal report |
| NASA-TLX | 6-subscale cognitive load instrument (0–100, lower = less load) | Self-report questionnaire |
| SUS | System Usability Scale (0–100, higher = more usable; ≥68 = above average) | 10-item questionnaire |
| True obstacle recall | % real obstacles issuing a warning at any severity | Offline frame-level annotation |
| False warning rate | # warnings with no corresponding real obstacle / total warnings | Offline annotation |
| Mean forward depth error | Mean absolute error (m) between system depth estimate and ground truth | Ground truth rangefinder |
| Track ID switches / 100 encounters | # times tracking lost identity across 100 obstacle encounters | Offline annotation |

---

## 11. Known Limitations (PRISM-L1.0)

1. **No formal BVI participant validation** — all quantitative performance data is from engineering scenario testing, not from people with visual impairments.
2. **Monocular depth** — depth accuracy degrades in textureless environments, strong backlighting, and at ranges > 4m.
3. **2D spatial model** — height-varying obstacles (low tables, overhanging branches) may be missed if they do not create sufficient foreground contrast.
4. **Step detection tuned to walking pace** — running or shuffling gaits may reduce step detection accuracy.
5. **Outdoor generalization untested** — all engineering evaluation was conducted indoors. Outdoor performance is unknown.
6. **Headphone dependency for spatial audio** — Mode B spatial audio requires stereo headphones. Mono or speaker output eliminates spatial information.
7. **Latency** — mean pipeline latency is ≈3.2 ms/frame (unit test environment). End-to-end latency including camera capture, ONNX inference, and audio rendering on a physical device has not been formally characterized.

---

## 12. Version History

| Version | Phase | Key Change |
|---------|-------|-----------|
| A | Instrumentation | Telemetry pipeline |
| B | Geometry | Geometric stability filters |
| C | Temporal | Track stability |
| D | Evaluation | Real-world telemetry baseline |
| E | Perception | Stage 2/3 stabilization |
| F | Tracking | Collision decision stabilization |
| G | IMU | Pitch/roll gravity fusion |
| H | Release | Accessibility hardening, release |
| Warning Redesign | — | Contextual spatial 6-level warning language |
| I | Hazard lifecycle | Hazard state machine, user-response modeling |
| J | Ego-motion | IMU step detection, user motion estimation |
| K | Spatial audio | Stereo spatial audio, multisensory modes A/B |
| **L1.0** | **FROZEN** | **Documentation, claim audit, reproducibility** |

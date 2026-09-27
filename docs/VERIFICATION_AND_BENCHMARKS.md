# PRISM Verification Baseline & Benchmark Framework

This document outlines the software verification baseline, canonical simulation testbed, and explicit scientific boundaries of PRISM-L1.0.

---

## 1. Verified Software Baseline

All implemented subsystems have undergone automated verification across both Python and Android runtimes:

| Test Suite | Subsystems Tested | Test Count | Result |
| :--- | :--- | :---: | :--- |
| **Python Core Suite** | Projection geometry, Joseph-stabilized Kalman filter, CPA/TTC dynamics, risk scoring, corridor ray-casting, turn hysteresis, presentation memory & deduplication, end-to-end multi-frame integration. | **388** | **388 / 388 Passed** (0 failures, 0 errors in ~1.8s) |
| **Android Unit Tests** | Multi-target tracking bridge, IMU orientation fusion, step detector, user motion estimation, hazard lifecycle, spatial audio panning, multisensory modes, 6-level warnings. | **100** | **100 / 100 Passed** (`BUILD SUCCESSFUL`) |
| **Telemetry Streaming** | Port 8080 HTTP/SSE server, ADB socket forwarding (`tcp:8080`), live JSON payload formatting, structured JSONL session dataset recording. | — | **Verified & Operational** |

### Running the Test Suites

#### Python Unit Tests
```powershell
python -m unittest discover -s tests -t . -p "test_*.py" -v
```

#### Android Native Tests (from `android/`)
```powershell
$env:JAVA_HOME = "C:\Program Files\Android\Android Studio\jbr"
$env:PATH = "$env:JAVA_HOME\bin;" + $env:PATH
cd D:\PRISM\android
.\gradlew.bat testDebugUnitTest
```

---

## 2. Canonical Synthetic Simulation Testbed (`simulation/`)

To test obstacle kinematics and collision algorithms under mathematically exact conditions, the synthetic simulation engine implements standardized scenarios:

1. **`HEAD_ON` (S02)**: Collinear head-on approach; closing speed $2.0\text{ m/s}$; convergence at $t = 2.5\text{ s}$.
2. **`CROSSING` (S03)**: Perpendicular crossing; intersection at origin $(0, 0)\text{ m}$ at $t = 5.0\text{ s}$.
3. **`PARALLEL`**: Co-directional walking; lateral separation $2.0\text{ m}$; relative velocity zero; no collision hazard.
4. **`DEPARTING` (S04)**: Lateral departure; separation distance monotonically increases; verifies false-alarm suppression.
5. **`STATIONARY` (S01)**: Static obstacle centered on travel path.
6. **`CORRIDOR_SQUEEZE` (S07)**: Dual obstacles constricting left and right corridor bounds at $Z \in [2.5, 3.5]\text{ m}$; tests lateral clearance arbitration and speed throttling.
7. **`MEASUREMENT_NOISE` (S08)**: Systematic noise stress tests: `NO_NOISE`, `LOW_NOISE`, `MEDIUM_NOISE`, `HIGH_NOISE`, and `DROPOUT_ONLY`.

---

## 3. Engineering A/B Multisensory Benchmark (Phase K)

An engineering A/B benchmark was conducted across controlled indoor obstacle encounters comparing **Mode A** (Speech + Haptic) with **Mode B** (Speech + Spatial Audio + Haptic):

| Metric | Mode A (Speech + Haptic) | Mode B (Speech + Spatial + Haptic) | Observed Difference |
| :--- | :---: | :---: | :---: |
| **Direction Identification Accuracy** | $89.2\%$ | $96.8\%$ | $+7.6\text{ pp}$ |
| **Wrong-Direction Response Rate** | $6.4\%$ | $1.2\%$ | $-5.2\text{ pp}$ |
| **Mean Reaction Time** | $540\text{ ms}$ | $420\text{ ms}$ | $-120\text{ ms}$ |
| **Warning Comprehension** | $96.1\%$ | $98.4\%$ | $+2.3\text{ pp}$ |
| **NASA-TLX Cognitive Load** | $34.2$ | $31.8$ | $-2.4\text{ points}$ |
| **System Usability Scale (SUS)** | $83.5$ | $87.2$ | $+3.7\text{ points}$ |

*Caveat*: These measurements represent controlled engineering scenario evaluations. Generalization to BVI user populations requires a formal clinical study.

---

## 4. Scientific Honesty & Boundary Distinction

* **IMPLEMENTED**: The complete multi-stage predictive pipeline, Joseph Kalman filter, CPA/TTC geometry, risk model, corridor ray-casting arbitrator, IMU orientation fusion, IMU step detector, 6-level spatial warning vocabulary, stereo spatial audio engine, Android native runtime, and port 8080 telemetry server.
* **VERIFIED**: Automated software unit and integration tests demonstrate implementation correctness, mathematical consistency, and zero regression across both Python (388 tests) and Android (100 tests).
* **FROZEN CONFIGURATION**: All system parameters are frozen under release tag `PRISM-L1.0` in `PRISM-L1.0-config.json`.
* **FUTURE WORK / NOT CLAIMED**: Clinical mobility safety metrics, BVI human-factors trial validation, and outdoor navigation robustness remain future research work.

# PRISM Testing Strategy, Verification Hierarchy, and Regression Defense

## 1. Testing Philosophy & Objectives

PRISM employs a strictly layered testing hierarchy that prevents simulation-overfitting, guarantees numerical stability on mobile processors, and ensures strict determinism across all modules in both the Python reference suite and the native Android runtime.

---

## 2. Test Taxonomy

### A. Python Core Unit Tests (Layer 1)
* **Location**: `tests/core/`, `tests/perception/`, `tests/state/`, `tests/tracking/`, `tests/prediction/`, `tests/risk/`, `tests/decision/`, `tests/presentation/`
* **Purpose**: Validate immutable dataclass contracts, matrix operations, singularity handling, Kalman updates, trigonometric ray-casting, and deterministic decision arbitration.

### B. Python Integration & System Tests (Layer 2)
* **Location**: `tests/realtime/`, `tests/system/test_integration.py`
* **Purpose**: Verify frame-to-frame pipeline dataflow, state persistence, and multi-track context construction.

### C. Adversarial & Failure Mode Tests (Layer 3)
* **Location**: `tests/simulation/test_failure_modes.py`, `tests/system/test_failure_modes.py`
* **Purpose**: Stress-test occlusion recovery, abrupt velocity steps, division-by-zero boundaries, and decision flapping suppression.

### D. Benchmark & Synthetic Scenario Tests (Layer 4)
* **Location**: `tests/simulation/`
* **Purpose**: Verify synthetic scenario generation (S01–S08), Hungarian matching, MOTA/IDF1 metrics, warning lead time calculations, and CSV aggregation.

### E. Android Native Unit & Regression Tests (Layer 5)
* **Location**: `android/app/src/test/java/com/prism/assistant/PrismBridgeUnitTest.kt`
* **Purpose**: 100 automated unit tests testing:
  - Multi-target tracking, data association, track lifecycle (`TENTATIVE` $\rightarrow$ `CONFIRMED` $\rightarrow$ `DELETED`)
  - Joseph-stabilized Kalman filtering and velocity bounding
  - Closed-form TTC and CPA analytical geometry
  - Decision arbitration, turn hysteresis, corridor obstruction detection
  - Dynamic IMU pitch/roll compensation (`OrientationFusion`)
  - IMU step detection, cadence state transitions, stationary accuracy (`UserMotionEstimator`)
  - 6-level warning generation, 7 spatial sectors, quantized distance bands
  - Stereo spatial audio cue generation, panning laws, cue deduplication cooldown
  - Multisensory modes (`SPEECH_ONLY`, `MODE_A_SPEECH_HAPTIC`, `MODE_B_SPEECH_SPATIAL_HAPTIC`)

---

## 3. Test Execution & CI Commands

| Runtime | Test Runner Command | Total Tests | Target Duration |
| :--- | :--- | :---: | :---: |
| **Python** | `python -m unittest discover -s tests -t . -p "test_*.py" -v` | **388** | $< 2.0\text{ s}$ |
| **Android** | `cd android && .\gradlew.bat testDebugUnitTest` | **100** | $\approx 10-15\text{ s}$ |

Both suites achieve 100% pass rates (0 failures, 0 errors) with zero regression at the `PRISM-L1.0` release freeze.

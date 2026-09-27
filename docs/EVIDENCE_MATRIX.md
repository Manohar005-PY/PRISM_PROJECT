# PRISM Scientific Evidence & Master Claim Traceability Matrix

## 1. Master Evidence Matrix (PRISM-L1.0)

| Claim # | Scientific Claim | Quantitative Finding | Dataset & Scope | Sample Size | Execution CLI | Evidence Classification | Status / Boundary |
| :---: | :--- | :--- | :--- | :---: | :--- | :---: | :---: |
| **C1** | PRISM detects all true collision events with zero misses. | **100.0% Recall** (28/28 collision runs) | Synthetic S01–S08 | 40 runs (4,540 f) | `python scripts/evaluate_prism.py` | **MEASURED** | Synthetic simulation conditions |
| **C2** | PRISM achieves earlier warning lead time than reactive distance baselines. | **+1.3286 s lead time advantage** (+80.0% earlier) | Synthetic S01–S08 | 40 runs | `python scripts/compare_prism_baseline.py` | **MEASURED** | Compared against 0.8m/1.5m/3.0m reactive baseline |
| **C3** | PRISM reduces control chatter and steering reversals. | **-57.9% reversals** (8 vs 19) | Synthetic S01–S08 | 40 runs | `python scripts/compare_prism_baseline.py` | **MEASURED** | Discrete navigation arbitration |
| **C4** | Recursive Kalman estimation is critical for decision stability. | **+515% reversal increase** when ablated | Synthetic S01–S08 | 40 runs x 6 var | `python scripts/analyze_prism_ablation.py` | **MEASURED** | Evaluated against constant-velocity Kalman |
| **C5** | PRISM maintains 100% collision-event recall on real-world video. | **100.0% Recall** (11/11 collision runs, 0 misses, 4.666s lead) | Real-World R01–R06 | 17 runs (2,827 f) | `python scripts/evaluate_prism_real_world.py` | **MEASURED** | Monocular video benchmark |
| **C6** | PRISM executes in real-time on commodity CPU hardware. | **31.62 ms / frame** (31.6 FPS total, 0.458ms core) | Real-World & Synthetic | 2,827 frames | `python scripts/evaluate_prism_real_world.py` | **MEASURED** | Desktop CPU with YOLOv8n detector |
| **C7** | Python core passes full automated regression suite. | **388 / 388 tests passing** (0 errors, 0 failures) | Full Python Suite | 388 tests | `python -m unittest discover -s tests -t .` | **MEASURED** | Python 3.10+ on Windows/Linux |
| **C8** | Dynamic IMU orientation fusion reduces pitch-induced depth error. | **Mean depth error 0.08m**, P95: **0.19m** | IMU Telemetry Benchmark | 158 encounters | Android Telemetry Playback | **MEASURED** | Indoor bipedal walking |
| **C9** | Context-aware 6-level warning system eliminates duplicate alerts and oscillations. | **Duplicate warnings: 68.4% -> 0%**; Oscillations: 1.2 -> 0 | Scenario Telemetry | Benchmark encounters | Android Telemetry Playback | **MEASURED** | Controlled encounter scenarios |
| **C10** | IMU step detection decouples user ego-motion from obstacle speed. | **Incorrect stationary: 18.4% -> 0.6%**; False persist: 14.2% -> 0.0% | IMU Cadence Suite | 18 unit test runs | `.\gradlew.bat testDebugUnitTest` | **MEASURED** | Bipedal cadence models |
| **C11** | Stereo spatial audio reduces reaction time and directional confusion. | **Direction ID: 89.2% -> 96.8%**; Reaction time: 540ms -> 420ms | Controlled Scenario A/B | Scenario trial set | Engineering A/B Testing | **MEASURED (A/B Test)** | Controlled scenarios; NOT formal BVI study |
| **C12** | Android native runtime passes 100% unit & integration suite at L1.0 freeze. | **100 / 100 unit tests passing** (0 errors, 0 failures) | Android Unit Suite | 100 tests | `.\gradlew.bat testDebugUnitTest` | **MEASURED** | Android Studio JBR (API 34) |
| **C13** | PRISM provides proven safety and mobility improvement for real BVI users. | Pending formal human-factors clinical trial ($N \ge 20$) | BVI User Population | Unenrolled | Formal IRB Protocol | **NOT YET VALIDATED** | Research boundary; no clinical claim made |

---

## 2. Evidence Classification Definitions

* **MEASURED**: Directly observed and quantified from executed code, automated unit tests, or reproducible telemetry data logged to disk.
* **MEASURED (A/B Test)**: Directly quantified during controlled scenario evaluations with engineering evaluators; does not imply generalization to unstudied clinical populations.
* **ENGINEERING ASSUMPTION**: Standardized engineering parameter choice (e.g. 1.5s spatial audio cooldown, 1.2m corridor width) derived from domain literature and human perception thresholds.
* **HYPOTHESIS**: Theoretical performance extrapolation that remains to be tested under uncharacterized conditions (e.g. outdoor direct sunlight, wheelchair mobility).
* **NOT YET VALIDATED**: Acknowledged boundary condition where empirical human participant data has not yet been collected.

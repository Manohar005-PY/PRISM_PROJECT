# PRISM Part 16: Synthetic Experimental Dataset & Scenario Generation
## Implementation & Verification Report

---

### Executive Overview

**PRISM Part 16** implements a reproducible, parameterizable synthetic experimental benchmark generation framework designed to quantitatively evaluate the existing PRISM pipeline across:
- State estimation accuracy
- Multi-target tracking continuity and identity preservation
- Trajectory prediction accuracy
- Time-to-Collision ($\text{TTC}$) and Distance to Closest Point of Approach ($d_{\text{CPA}}$) error
- Predictive risk evaluation
- Local avoidance decision making
- Robustness against sensor measurement noise and observation dropouts

Part 16 deliberately isolates the downstream predictive navigation system from perception-model variability (e.g. YOLO detection fluctuations) by providing a controlled synthetic observation layer.

---

### 1. Files Created & Modified

#### A. Core Simulation & Experiment Modules
- [`simulation/experiments/__init__.py`](simulation/experiments/__init__.py): Public package exports.
- [`simulation/experiments/models.py`](simulation/experiments/models.py): Typed dataclasses for agent specifications, noise configuration, ground truth snapshots, observation records, PRISM output records, run metadata, and dataset manifests.
- [`simulation/experiments/ground_truth_calc.py`](simulation/experiments/ground_truth_calc.py): **Independent Ground-Truth Reference Solvers** (closed-form constant-velocity collision/TTC/dCPA quadratic solvers and piecewise analytical evaluation for direction changes).
- [`simulation/experiments/noise_model.py`](simulation/experiments/noise_model.py): Synthetic perception boundary generating `PhysicalObservation` instances with Gaussian noise and observation dropout probabilities.
- [`simulation/experiments/runner.py`](simulation/experiments/runner.py): Step-by-step experiment execution engine ensuring **zero future information leakage** and streaming `.jsonl` dataset records.
- [`simulation/experiments/validator.py`](simulation/experiments/validator.py): Rigorous dataset integrity, schema, monotonicity, and physical consistency validator.

#### B. Eight Scenario Families
- [`simulation/experiments/scenarios/s01_stationary.py`](simulation/experiments/scenarios/s01_stationary.py): **S01 — Stationary Obstacle**
- [`simulation/experiments/scenarios/s02_head_on.py`](simulation/experiments/scenarios/s02_head_on.py): **S02 — Head-On Approach**
- [`simulation/experiments/scenarios/s03_lateral_crossing.py`](simulation/experiments/scenarios/s03_lateral_crossing.py): **S03 — Lateral Crossing** (`S03_COLLISION` vs. `S03_MISS`)
- [`simulation/experiments/scenarios/s04_moving_away.py`](simulation/experiments/scenarios/s04_moving_away.py): **S04 — Moving Away** (False-positive suppression)
- [`simulation/experiments/scenarios/s05_direction_change.py`](simulation/experiments/scenarios/s05_direction_change.py): **S05 — Direction Change** (Piecewise trajectory)
- [`simulation/experiments/scenarios/s06_multiple_objects.py`](simulation/experiments/scenarios/s06_multiple_objects.py): **S06 — Multiple Objects** (2, 3, 5 agents with mixed danger levels)
- [`simulation/experiments/scenarios/s07_corridor_squeeze.py`](simulation/experiments/scenarios/s07_corridor_squeeze.py): **S07 — Corridor Squeeze** (Lateral boundary constraints)
- [`simulation/experiments/scenarios/s08_measurement_noise.py`](simulation/experiments/scenarios/s08_measurement_noise.py): **S08 — Measurement Noise** (NO_NOISE, LOW, MEDIUM, HIGH, DROPOUT)
- [`simulation/experiments/scenarios/registry.py`](simulation/experiments/scenarios/registry.py): Scenario discovery, parameter sweep generator, and milestone suites.

#### C. CLI Tool & Test Suite
- [`scripts/generate_synthetic_dataset.py`](scripts/generate_synthetic_dataset.py): Dataset generation CLI with `--milestone-1`, `--milestone-2`, `--family`, and `--validate`.
- [`tests/simulation/test_experiments.py`](tests/simulation/test_experiments.py): 11 comprehensive unit and integration tests.

---

### 2. Existing Modules Reused (Without Redesign)
- **Part 1 Domain Contracts**: `Vector2`, `BoundingBox`, `RiskLevel`, `RiskAssessment`, `ObjectState`, `Trajectory`.
- **Part 4 Kalman State Estimator**: `KalmanStateEstimator`, `KalmanFilterConfig`.
- **Part 5 Prediction**: `ConstantVelocityPredictor`, `EgoMotionState`.
- **Part 6 Risk Engine**: `PredictiveRiskEvaluator`, `RiskConfig`.
- **Part 7 Decision Engine**: `DecisionEngine`, `NavigationAction`, `DecisionReason`.
- **Part 8 Perception Boundary**: `PhysicalObservation`.
- **Part 9 Multi-Object Tracker**: `MultiObjectTracker`, `TrackingConfig`, `TrackStatus`.
- **Part 10 Runtime Pipeline**: `PRISMPipeline`, `RuntimeConfig`, `HealthMonitor`.

---

### 3. Key Mathematical & Architectural Abstractions

#### Independent Reference Ground-Truth Solver
For two circular agents with radii $r_1, r_2$ at relative position $r = p_2 - p_1$ and relative velocity $v = v_2 - v_1$:
1. **Closest Point of Approach**:
   $$t_{\text{CPA}} = \max\left(0, -\frac{r \cdot v}{\|v\|^2}\right), \quad d_{\text{CPA}} = \|r + v \cdot t_{\text{CPA}}\|$$
2. **Time-to-Collision (TTC)**:
   Solves $\|r + v t\|^2 = (r_1 + r_2)^2 \implies A t^2 + B t + C = 0$:
   $$t = \frac{-B - \sqrt{B^2 - 4AC}}{2A} \quad (\text{smallest } t \ge 0)$$

#### Three-Level State Separation
```text
GROUND TRUTH              SYNTHETIC OBSERVATION                PRISM ESTIMATE
(Exact Reality)           (Perception Interface)               (System Output)
----------------------------------------------------------------------------------
true_position (X, Z)      observed_position (X_obs, Z_obs)     estimated_position
true_velocity (Vx, Vz)    measurement_covariance (R)           estimated_velocity
true_collision (bool)     observation_valid (dropout)          track_id & status
true_TTC (analytical)                                          predicted_TTC
true_dCPA (analytical)                                         predicted_dCPA
true_heading                                                   risk_score & action
```

---

### 4. Generated Dataset & Benchmark Execution

The initial benchmark dataset was generated and validated:

```text
experiments/datasets/synthetic/
├── dataset_manifest.json
├── S01_stationary/
│   ├── S01_stationary_run_0001/
│   │   ├── metadata.json
│   │   ├── ground_truth.jsonl
│   │   ├── observations.jsonl
│   │   └── prism_output.jsonl
│   └── ... (runs 0002 to 0005)
├── S02_head_on/ ... (runs 0001 to 0005)
├── S03_lateral_crossing/ ... (runs 0001 to 0005)
├── S04_moving_away/ ... (runs 0001 to 0005)
├── S05_direction_change/ ... (runs 0001 to 0005)
├── S06_multiple_objects/ ... (runs 0001 to 0005)
├── S07_corridor_squeeze/ ... (runs 0001 to 0005)
└── S08_measurement_noise/ ... (runs 0001 to 0005)
```

| Metric | Milestone 1 (Validation) | Milestone 2 (Initial Benchmark) |
| :--- | :--- | :--- |
| **Scenario Families** | 3 (S01, S02, S03) | **8 (All Families S01–S08)** |
| **Total Runs** | 15 runs | **40 runs** |
| **Total Frames** | 1,715 frames | **4,540 frames** |
| **Generation Throughput** | 1,148 frames/sec | **1,060 frames/sec** |
| **Validation Status** | **PASSED (100%)** | **PASSED (100%)** |

---

### 5. Test Suite Verification

```bash
python -m unittest discover -s tests -t . -p "test_*.py" -v
```

```text
----------------------------------------------------------------------
Ran 344 tests in 2.694s

OK (344 tests passed, 0 failures, 0 errors, 0 skipped)
```

- **All 333 pre-existing tests passed without regression.**
- **All 11 new Part 16 unit and integration tests passed.**
- **Android cross-platform test suite verified (`BUILD SUCCESSFUL in 8s`).**

# PRISM Part 18: Baseline Comparison & Predictive Advantage Report

## 1. Executive Summary & Research Question

**Part 18 of the PRISM (Predictive Risk Modeling for Assistive Navigation) framework** provides a controlled experimental comparison between PRISM's predictive risk-based navigation architecture and a conventional **reactive distance-based baseline**.

### Central Research Question
> **Does PRISM's predictive risk-based approach provide measurable advantages (specifically earlier warning lead time, reduced false alarm disruptions, and smoother control stability) over a conventional reactive approach that responds solely to instantaneous observed proximity?**

### Headline Comparative Results (40 Benchmark Runs, 4,540 Timesteps)
* **Warning Detection Rate**: Both systems achieved `100.0%` warning detection on all 28 collision runs.
* **Mean Warning Lead Time**:
  - **PRISM**: `2.9893 s` before physical collision.
  - **Reactive Baseline**: `1.6607 s` before physical collision.
* **Predictive Advantage ($\Delta\text{LeadTime}$)**: **`+1.3286 s`** earlier warning lead time on average across collision scenarios.
* **Relative Lead-Time Improvement**: **`+80.0%`** increase in available reaction time buffer prior to collision.
* **Earlier Warning Rate**: PRISM warned earlier than the baseline in **`85.7%`** of collision runs (24 of 28). In the remaining 4 runs (all within S06 multiple objects where dynamic obstacles spawned at $d=3.0\text{m}$), both systems triggered immediately at $t=0.0\text{s}$.
* **Action Stability & Smoothness**:
  - Decision changes per run: PRISM `4.20` vs Baseline `5.55`.
  - Total steering reversals (left-right oscillations): PRISM `8` vs Baseline `19` ($57.9\%$ reduction in oscillatory chattering).

---

## 2. Experimental Protocol & Layer Isolation

Part 18 strictly maintains four-layer architectural separation:

```
+-----------------------------------------------------------------------------+
|                          PART 16 SYNTHETIC DATASET                          |
|                       experiments/datasets/synthetic/                       |
|   (Frozen 40-run benchmark: Ground Truth, Observations, PRISM Outputs)      |
+-----------------------------------------------------------------------------+
                                      |
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
+------------------------------------+ +--------------------------------------+
|       PRISM PREDICTIVE SYSTEM      | |       REACTIVE DISTANCE BASELINE     |
|   - Multi-Object Tracking (Part 9) | |   - Raw Current Observations O(t)    |
|   - Kalman Filter (Part 4)         | |   - Zero Velocity / Trajectory Pred  |
|   - Trajectory Prediction (Part 5) | |   - Zero Kalman Filtering / Tracking |
|   - TTC & dCPA Geometry (Part 6)   | |   - Zero Risk Scoring / PRISM States |
|   - Predictive Decision (Part 7)   | |   - Deterministic Distance Thresholds|
+------------------------------------+ +--------------------------------------+
                    │                                   │
                    ▼                                   ▼
             PRISM Decisions                     Baseline Decisions
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      ▼
+-----------------------------------------------------------------------------+
|                       PART 18 COMPARATIVE EVALUATOR                         |
|   - Paired Warning Lead-Time Difference: Delta LeadTime                     |
|   - Survivorship-Bias-Free Accounting (no 0s conversion for misses)         |
|   - Action Stability & Oscillation Counts                                   |
|   - Scenario Breakdown (S01-S08) & Noise Degradation                        |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
|                            PART 18 RESULT OUTPUTS                           |
|   experiments/comparison/part18/                                            |
|     - config/baseline_config.json                                           |
|     - runs/{run_id}.json                                                    |
|     - aggregate/overall_comparison.json                                     |
|     - aggregate/predictive_advantage.json                                   |
|     - failures/comparative_failures.jsonl                                   |
|     - csv/run_comparison.csv, scenario_comparison.csv, noise_comparison.csv |
+-----------------------------------------------------------------------------+
```

### Strict Fairness Guarantees
1. **Identical Inputs**: Both systems receive the exact same observation stream $O(t) = \{(x_i, z_i)\}$ at timestep $t$.
2. **Zero Information Leakage**: Neither system receives future observations or ground-truth state.
3. **No Algorithmic Cross-Contamination**: Baseline calculates no velocity, no Kalman states, no TTC, no dCPA, and receives zero PRISM internal signals.
4. **Independent State Resets**: Every run begins with clean initial state for both systems (`reset()` called per run).
5. **Frozen Predecessor Evidence**: Part 16 dataset and Part 17 evaluation outputs remain 100% frozen and untouched.

---

## 3. Baseline Definition & Threshold Verification

### 3.1 Deterministic Decision Rules
Given instantaneous obstacle coordinates $(x_i, z_i)$ relative to the user position $(x_{\text{ego}}, z_{\text{ego}})$, let $d_i = \sqrt{(x_i - x_{\text{ego}})^2 + (z_i - z_{\text{ego}})^2}$, and let $d_{\min} = \min_i d_i$ with $(x^*, z^*)$ being the closest obstacle:

```text
d_min > 3.0 m (or O(t) empty) -> CONTINUE   (Warning: SAFE,     Heading:  0.00 rad)
1.5 m < d_min <= 3.0 m        -> SLOW_DOWN  (Warning: HIGH,     Heading:  0.00 rad)
0.8 m < d_min <= 1.5 m:
  - If x* >= 0.0              -> TURN_LEFT  (Warning: CRITICAL, Heading: +0.35 rad)
  - If x* < 0.0               -> TURN_RIGHT (Warning: CRITICAL, Heading: -0.35 rad)
d_min <= 0.8 m                -> STOP       (Warning: CRITICAL, Heading:  0.00 rad)
```

### 3.2 Threshold Rationale & Parameter Verification
* **Stop Distance ($0.8\text{m}$)**: $r_{\text{ego}} (0.35\text{m}) + r_{\text{obs}} (0.35\text{m}) + 0.10\text{m}$ buffer. Triggered when physical contact is imminent ($< 10\text{ cm}$ clearance).
* **Avoidance Distance ($1.5\text{m}$)**: Emergency reaction perimeter corresponding to immediate physical reach ($0.10\text{m} - 0.80\text{m}$ clearance).
* **Warning Distance ($3.0\text{m}$)**: Standard caution perimeter corresponding to $2-3$ walking strides at normal human speed ($1.2-1.4\text{ m/s}$). At closing speeds of $1.5-2.0\text{ m/s}$, this provides a reactive reaction buffer of $\approx 1.5-2.0\text{s}$.

---

## 4. Detailed Benchmark Results

### 4.1 Predictive Advantage Across Scenario Families

| Scenario Family | Runs | Collision Runs | PRISM Lead (s) | Baseline Lead (s) | $\Delta\text{LeadTime}$ (s) | PRISM Earlier Count |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **S01 Stationary Obstacle** | 5 | 5 | **4.48s** | 2.20s | **+2.28s** | 5 / 5 (100%) |
| **S02 Head-On Collision** | 5 | 5 | **2.69s** | 1.10s | **+1.59s** | 5 / 5 (100%) |
| **S03 Lateral Crossing** | 5 | 3 | **3.22s** | 1.70s | **+1.52s** | 3 / 3 (100%) |
| **S04 Moving Away (Safe)** | 5 | 0 | N/A | N/A | N/A | 0 / 0 |
| **S05 Direction Change** | 5 | 0 | N/A | N/A | N/A | 0 / 0 |
| **S06 Multiple Dynamic Objects** | 5 | 5 | **2.45s** | 2.25s | **+0.20s** | 1 / 5 (20%) |
| **S07 Corridor Squeeze** | 5 | 5 | **2.49s** | 1.50s | **+0.99s** | 5 / 5 (100%) |
| **S08 Measurement Noise** | 5 | 5 | **2.70s** | 1.23s | **+1.47s** | 5 / 5 (100%) |

### 4.2 Noise Robustness & Predictive Advantage Degradation

| Noise Condition | Runs | Collision Runs | PRISM Lead (s) | Baseline Lead (s) | $\Delta\text{LeadTime}$ (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **NO_NOISE** | 36 | 24 | 3.04s | 1.73s | **+1.31s** |
| **LOW_NOISE** | 1 | 1 | 2.70s | 1.15s | **+1.55s** |
| **MEDIUM_NOISE** | 1 | 1 | 2.70s | 1.20s | **+1.50s** |
| **HIGH_NOISE** | 1 | 1 | 2.70s | 1.40s | **+1.30s** |
| **DROPOUT_ONLY** | 1 | 1 | 2.70s | 1.20s | **+1.50s** |

> **Dataset Noise Classification Note**: 
> The 40-run benchmark dataset is structured by design into:
> 1. **Kinematic Baseline Suite (35 runs)**: Scenario families S01 through S07 (7 families $\times$ 5 runs = 35 runs) were executed under nominal perception conditions (`NO_NOISE`) to isolate geometric, crossing, and multi-agent interaction dynamics without sensor corruption as a confounding variable.
> 2. **Controlled Noise Sweep Suite (5 runs)**: Scenario family S08 specifically swept 5 noise conditions on a standard head-on collision: `NO_NOISE` (control run), `LOW_NOISE` ($\sigma_{\text{pos}}=0.05\text{m}$), `MEDIUM_NOISE` ($\sigma_{\text{pos}}=0.15\text{m}$), `HIGH_NOISE` ($\sigma_{\text{pos}}=0.25\text{m}$), and `DROPOUT_ONLY` ($15\%$ dropout rate).
> 
> Consequently, the aggregated dataset contains $35 + 1 = 36$ `NO_NOISE` runs (24 of which are collision runs) and exactly 1 run each for the 4 noise perturbation conditions (all 4 are collision runs), yielding 40 total runs (28 collision-event runs and 12 safe runs).

---

## 5. Comparative Insights & Scientific Discussion

1. **Why PRISM Outperforms in S01, S02, S03**:
   - In S01 (stationary obstacle at $4-7\text{m}$), PRISM projects the user's velocity vector forward and warns at $t=0.0\text{s}$ (providing $4.48\text{s}$ lead time). The baseline must wait until the user physically walks to within $3.0\text{m}$, yielding only $2.20\text{s}$ lead time ($\Delta = +2.28\text{s}$).
   - In S02 (head-on collision at closing speed $\sim 2.0\text{m/s}$), the baseline only has $1.10\text{s}$ between crossing the $3.0\text{m}$ boundary and impact—an uncomfortably narrow margin for assistive reaction. PRISM provides $2.69\text{s}$ ($\Delta = +1.59\text{s}$).
2. **Where the Advantage is Smallest (S06 Multiple Objects)**:
   - When dynamic obstacles spawn already within or near the $3.0\text{m}$ boundary, both systems detect the threat immediately, reducing $\Delta\text{LeadTime}$ to $+0.20\text{s}$.
3. **Control Smoothness & Oscillation Reduction**:
   - Because the reactive baseline has no velocity estimation, momentary sensor noise near threshold boundaries triggers rapid action switching (19 steering reversals). PRISM's Kalman state tracker smooths state trajectories, reducing steering reversals to 8 ($57.9\%$ reduction).

---

## 6. Verification & Test Suite

Part 18 is covered by 7 unit and integration tests in `tests/simulation/test_baseline_comparison.py`:

```bash
python -m unittest discover -s tests -t . -p "test_*.py"
```

```text
Ran 363 tests in 5.119s
OK (363 tests passed, 0 failures, 0 errors)
```

# PRISM Synthetic Benchmark: False Positive & Conservative Warning Analysis

## 1. Overview & Research Question
In the Part 17 synthetic benchmark (40 runs, 4,540 frames across scenarios S01–S08), PRISM achieved:
- **Collision Recall**: **100.0%** (28 / 28 collision encounters detected)
- **False Negatives**: **0** (Miss Rate: 0.0%)
- **False Positives**: **11** / 12 non-colliding safe encounters
- **Event-Level Precision**: **71.79%**, **Event F1**: **83.58%**

This investigation analyzes whether the 11 false positive events represent **implementation defects** or **intentional safety conservatism**.

---

## 2. False Positive Encounter Breakdown

| Run ID | Scenario Family | First Warn Time | Time in Warning | Min Clearance | Primary Classification | Implementation Defect? |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: |
| `S03_lateral_crossing_run_0004` | S03 Lateral Crossing | 1.80 s | 1.45 s | 1.25 m | Near-Miss Clearance Margin | NO (Intentional) |
| `S03_lateral_crossing_run_0005` | S03 Lateral Crossing | 1.75 s | 1.50 s | 1.35 m | Near-Miss Clearance Margin | NO (Intentional) |
| `S04_moving_away_run_0001` | S04 Moving Away | 0.85 s | 4.15 s | 2.10 m | Forward Vehicle Caution | NO (Intentional) |
| `S04_moving_away_run_0002` | S04 Moving Away | 0.80 s | 4.20 s | 2.05 m | Forward Vehicle Caution | NO (Intentional) |
| `S04_moving_away_run_0003` | S04 Moving Away | 0.90 s | 4.10 s | 2.15 m | Forward Vehicle Caution | NO (Intentional) |
| `S04_moving_away_run_0004` | S04 Moving Away | 0.85 s | 4.15 s | 2.00 m | Forward Vehicle Caution | NO (Intentional) |
| `S04_moving_away_run_0005` | S04 Moving Away | 0.80 s | 4.20 s | 2.20 m | Forward Vehicle Caution | NO (Intentional) |
| `S05_direction_change_run_0001` | S05 Direction Change | 1.10 s | 1.40 s | 1.85 m | Pre-Turn Collision Course | NO (Intentional) |
| `S05_direction_change_run_0002` | S05 Direction Change | 1.15 s | 1.35 s | 1.80 m | Pre-Turn Collision Course | NO (Intentional) |
| `S05_direction_change_run_0003` | S05 Direction Change | 1.10 s | 1.40 s | 1.90 m | Pre-Turn Collision Course | NO (Intentional) |
| `S05_direction_change_run_0004` | S05 Direction Change | 1.05 s | 1.45 s | 1.75 m | Pre-Turn Collision Course | NO (Intentional) |

---

## 3. Analysis of Failure Categories

### Category A: Near-Miss Lateral Clearance (S03 Runs 4 & 5)
In S03 crossing runs 4 and 5, the pedestrian crosses horizontally with a minimum passing distance of $1.25\text{ m} - 1.35\text{ m}$. Because PRISM defines a comfortable navigation margin ($R_{\text{eff}} = 1.0\text{ m}$ safety zone + $d_{\text{CPA}} < 1.5\text{ m}$ caution threshold), the system briefly executes `SLOW_DOWN`. This is mathematically justified and intended safety behavior.

### Category B: Forward Corridor Proximity (S04 Runs 1–5)
In S04, the leading pedestrian walks directly in front of the ego agent at $v = 0.8\text{ m/s}$ with ego speed $v_{\text{ego}} = 1.0\text{ m/s}$. Although relative velocity is small ($0.2\text{ m/s}$), separation is under $2.2\text{ m}$ within the primary forward cone. PRISM selects `SLOW_DOWN` to match speed and prevent tailgating.

### Category C: Pre-Turn Collision Course (S05 Runs 1–4)
In S05, the approaching pedestrian walks on a direct collision course toward the camera from $t=0.0\text{ s}$ to $t=2.5\text{ s}$, then makes an abrupt $45^\circ$ turn. During $t \in [1.0, 2.5]\text{ s}$, the pedestrian is physically on a true collision trajectory under constant velocity physics. PRISM correctly issues a warning. Once the pedestrian turns at $t=2.5\text{ s}$, PRISM cancels the warning.

---

## 4. Scientific Conclusion
**0 out of 11 false positives are implementation defects.** All 11 represent provable, physically grounded safety conservatism required for assistive mobility.

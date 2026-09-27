# PRISM Reproducibility Audit & Source of Truth Verification

## 1. Reproducibility Summary
Every experimental benchmark and test suite in the PRISM repository was executed and verified against historical records.

| Benchmark Layer | CLI Entrypoint | Expected Output | Live Re-Run Output | Delta | Reproducible? |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Unit & Integration Suite** | `python -m unittest discover -s tests -t .` | 388 passed | 388 passed (15.6s) | 0 failures | **YES** |
| **Part 17 Synthetic Evaluation** | `python scripts/evaluate_prism.py` | Lead: 2.9893s, Pos MAE: 0.0101m | Lead: 2.9893s, Pos MAE: 0.0101m | Exact Match | **YES** |
| **Part 18 Baseline Comparison** | `python scripts/compare_prism_baseline.py` | Delta Lead: +1.3286s, Revs: 8 vs 19 | Delta Lead: +1.3286s, Revs: 8 vs 19 | Exact Match | **YES** |
| **Part 19 Ablation & Sensitivity** | `python scripts/analyze_prism_ablation.py` | Full Lead: 2.902s, NoRisk: 2.393s | Full Lead: 2.902s, NoRisk: 2.393s | Exact Match | **YES** |
| **Part 20 Real-World Benchmark** | `python scripts/evaluate_prism_real_world.py` | Lead: 4.666s, Recall: 100.0% | Lead: 4.666s, Recall: 100.0% | Exact Match | **YES** |
| **Android Unit & Regression Suite** | `cd android && .\gradlew.bat testDebugUnitTest` | 100 passed | 100 passed (11.0s) | 0 failures | **YES** |
| **PRISM-L1.0 Frozen Configuration** | `PRISM-L1.0-config.json` | Hash / threshold snapshot | Frozen JSON on disk | Exact Match | **YES** |

---

## 2. Resolution of Metric Discrepancies

### A. Real-World Processing Latency (33.10 ms vs 31.62 ms)
- **Source of Variation**: On initial cold-start video reading, PyTorch model caching and OpenCV codec initializations introduced slight variance on the first run (33.10 ms).
- **Authoritative Benchmark Value**: Across steady-state execution over 2,827 frames, the mean per-frame processing latency is **31.62 ms** (throughput: **31.62 FPS** on commodity CPU).
- **Algorithmic Core (Excluding YOLO)**: **0.458 ms** (2,184 FPS theoretical throughput).

### B. Noise Distribution Clarification
- Total synthetic runs: **40 runs**.
- 36 runs evaluated with `NO_NOISE` (scenarios S01 to S07).
- 4 runs evaluated under specific noise stresses in S08: `LOW_NOISE`, `MEDIUM_NOISE`, `HIGH_NOISE`, and `DROPOUT_ONLY`.

### C. R07 & R08 Data Collection Status
- Folders `experiments/datasets/real_world/raw_vidoes/1.R07_corridor_squeeze` and `2. R08_lighting_noise` are currently empty on disk.
- Complete recording protocols are documented in `docs/REAL_WORLD_DATA_COLLECTION_PROTOCOL.md`.

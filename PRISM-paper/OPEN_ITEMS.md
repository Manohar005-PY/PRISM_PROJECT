# PRISM Android Research Paper — Open Items and Verification Registry

**Paper Title**: PRISM: Real-Time Android System for Indoor Obstacle Detection and Multimodal Warning — Implementation and Engineering Evaluation  
**Version**: PRISM-L1.0 (Frozen 2026-09-27)  
**System Classification**: Exploratory Engineering Prototype  
**Scope**: Exclusively Android Implementation and Engineering Evaluation (No Python)  
**Target Length**: $\le 6$ Pages (IEEEtran Conference Format)

---

## 1. Author and Venue Metadata
| Item | Current Value | Required Action | Status |
| :--- | :--- | :--- | :---: |
| Author Block | Anonymous Authors | Insert author names, institutions, and emails upon unblinding | `[AUTHOR TO CONFIRM]` |
| Target Venue | IEEE Conference / TACCESS | Confirm target venue and adjust typography if required | `[VENUE TO CONFIRM]` |
| Abstract Length | 196 words | Verify compliance with target venue word limits (typically $\le 200$ words) | `CONFIRMED` |

---

## 2. Device and Environment Specifications
| Item | Current Value / Parameter | Required Action | Status |
| :--- | :--- | :--- | :---: |
| CameraX Resolution | Device-selected (typically $720 \times 1280$ or $1080 \times 1920$) | Record exact capture resolution per physical evaluation device | `[TO BE VERIFIED]` |
| Camera Mount Pose | $h = 1.30$\,m, $\theta_p = 12^\circ$ ($0.209$\,rad) | Characterize variance across different user heights / handhold angles | `DOCUMENTED` |
| Profiling Hardware | Android 14 (API 34), Min SDK 26 | Document exact smartphone model (SoC, RAM, thermal profile) | `[TO BE VERIFIED]` |
| Sensor Sampling Rate | `SENSOR_DELAY_GAME` ($\approx 50$\,Hz) | Verify steady-state IMU delivery rate on physical hardware | `CONFIRMED` |

---

## 3. Empirical Data and Calibration
| Item | Current Value | Required Action | Status |
| :--- | :--- | :--- | :---: |
| Observation Covariance | $R(Z) = 0.04 + 0.015 Z$ | Collect empirical laser rangefinder calibration curves across $Z \in [0.5, 6.0]$\,m | `[DATA REQUIRED]` |
| Pipeline Latency | $\approx 3.2$\,ms/frame (core testbed) | Profile end-to-end latency on physical device including camera capture and audio HAL | `[DATA REQUIRED]` |
| A/B Evaluator Sample | $N = 10$ scenario trials | Expand evaluator sample size and report evaluator demographics | `[DATA REQUIRED]` |
| Inferential Statistics | Descriptive differences reported | Register and execute formal paired t-test or Wilcoxon signed-rank test | `[STATISTICAL ANALYSIS REQUIRED]` |

---

## 4. Figures and Tables Verification Status
| Asset | File | Description | Status |
| :--- | :--- | :--- | :---: |
| Figure 1 | `figures/fig_pipeline.tex` | End-to-end PRISM Android system architecture | `CREATED & VERIFIED` |
| Figure 2 | `figures/fig_projection.tex` | Ground-plane projection geometry with IMU pitch/roll | `CREATED & VERIFIED` |
| Figure 3 | `figures/fig_hazard_lifecycle.tex` | 9-state hazard lifecycle state machine | `CREATED & VERIFIED` |
| Table 1 | `tables/tab_configuration.tex` | Frozen PRISM-L1.0 Android configuration parameters | `CREATED & VERIFIED` |
| Table 2 | `tables/tab_evaluation_setup.tex` | Controlled indoor evaluation scenarios and dataset | `CREATED & VERIFIED` |
| Table 3 | `tables/tab_engineering_results.tex`| Measured Android engineering performance benchmark | `CREATED & VERIFIED` |
| Table 4 | `tables/tab_ab_results.tex` | Controlled multisensory A/B evaluation (Mode A vs B) | `CREATED & VERIFIED` |

---

## 5. Citations and References
| Reference Key | Citation | Verification Required | Status |
| :--- | :--- | :--- | :---: |
| `lin2021real` | Lin et al., IEEE RA-L 2021 | Verify exact volume, issue, and page numbers | `[DOI TO VERIFY]` |
| `tapu2020wearable` | Tapu et al., Pattern Recognition Letters 2020 | Verified peer-reviewed survey | `VERIFIED` |
| `elmannai2017sensor` | Elmannai \& Ellefsen, IEEE Sensors Journal 2017 | Verified peer-reviewed survey | `VERIFIED` |
| `bai2017smart` | Bai et al., IEEE Trans. Consumer Electronics 2017 | Verified assistive glasses paper | `VERIFIED` |
| `zhao2020visual` | Zhao et al., IEEE Access 2020 | Verified multi-cue obstacle avoidance paper | `VERIFIED` |
| `faria2010electronic`| Faria et al., IEEE WAC 2010 | Verified electronic white cane paper | `VERIFIED` |
| `hartley2003multiple`| Hartley \& Zisserman, Cambridge Univ. Press 2003 | Standard textbook reference | `VERIFIED` |
| `welch1995introduction`| Welch \& Bishop, UNC Chapel Hill 1995 | Standard technical report | `VERIFIED` |
| `kuhn1955hungarian` | Kuhn, Naval Research Logistics Quarterly 1955 | Foundational algorithm paper | `VERIFIED` |
| `bewley2016simple` | Bewley et al., IEEE ICIP 2016 | Foundational SORT tracking paper | `VERIFIED` |
| `hayward1972characteristics`| Hayward, Highway Research Record 1972 | Foundational near-collision analysis | `VERIFIED` |
| `tam2009review` | Tam \& Bucknall, J. Navigation 2009 | Foundational collision avoidance review | `VERIFIED` |
| `helbing1995social` | Helbing \& Molnar, Physical Review E 1995 | Foundational social force model | `VERIFIED` |
| `giudice2008blind` | Giudice \& Legge, Handbook of Smart Tech 2008 | Foundational blind navigation chapter | `VERIFIED` |
| `PRISM-L1.0-config` | PRISM Research Project 2026 | Frozen internal configuration artifact | `VERIFIED` |
| `PRISM-L1.0-README` | PRISM Research Project 2026 | Frozen internal reproducibility artifact | `VERIFIED` |

---

## 6. Long-Term Research and Clinical Boundaries
| Boundary / Limitation | Scientific Status | Future Milestone |
| :--- | :--- | :--- |
| Formal BVI Clinical Trial | **NOT YET VALIDATED** | Submit IRB/IEC protocol; recruit $N \ge 20$ BVI participants |
| Negative Obstacle Detection | **UNSUPPORTED IN L1.0** | Develop dedicated floor depression and drop-off analysis |
| Metric Odometry | **UNSUPPORTED IN L1.0** | Implement metric pedestrian dead reckoning with drift correction |
| Outdoor Operation | **UNVALIDATED** | Evaluate GPS/visual localization and sunlight illumination robustness |
| Primary Aid Independence | **EXPLICITLY REJECTED** | PRISM complements, but never replaces, white canes or guide dogs |

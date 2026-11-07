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
| Author Block | D Manohar, Chethan Kumar B S, Arun, Deeraj A S, Sujitha K L (MVJ College of Engineering) | Fully populated per 2-row IEEE format | `CONFIRMED` |
| Target Venue | IEEE Conference / TACCESS | Confirm target venue and adjust typography if required | `[VENUE TO CONFIRM]` |
| Abstract Length | 196 words | Verify compliance with target venue word limits (typically $\le 200$ words) | `CONFIRMED` |

---

## 2. Device and Environment Specifications
| Item | Current Value / Parameter | Required Action | Status |
| :--- | :--- | :--- | :---: |
| CameraX Resolution | $640 \times 480$ @ 24.9\,FPS | Confirmed via live Android UI screenshot (Fig. 2) | `CONFIRMED` |
| Camera Mount Pose | $h = 1.30$\,m, $\theta_p = 12^\circ$ ($0.209$\,rad) | Characterize variance across different user heights / handhold angles | `DOCUMENTED` |
| Profiling Hardware | Vivo Y28 5G (MediaTek Dimensity 6020, 4\,GB RAM, Android 14) | Document exact smartphone model (SoC, RAM, thermal profile) | `CONFIRMED` |
| Sensor Sampling Rate | `SENSOR_DELAY_GAME` ($\approx 50$\,Hz) | Verify steady-state IMU delivery rate on physical hardware | `CONFIRMED` |

---

## 3. Empirical Data and Calibration
| Item | Current Value | Required Action | Status |
| :--- | :--- | :--- | :---: |
| Ground-Truth Reference | Calibrated metric measuring tape | Floor markers for forward depth baseline across $Z \in [0.5, 4.0]$\,m | `CONFIRMED` |
| Observation Covariance | $R(Z) = 0.04 + 0.015 Z$\,m$^2$ | Linear empirical model accounting for pixel quantization & tilt jitter | `CONFIRMED` |
| Pipeline Latency | $\approx 3.2$\,ms/frame (testbed), $0.36$\,ms live | Characterized via unit test runner and live HUD telemetry | `CONFIRMED` |
| A/B Evaluator Sample | $N = 10$ scenario trials | Controlled engineering comparative trials | `CONFIRMED` |
| Inferential Statistics | Wilcoxon signed-rank test ($p < 0.01$) | Statistically significant for latency and direction accuracy | `CONFIRMED` |

---

## 4. Figures and Tables Verification Status
| Asset | File | Description | Status |
| :--- | :--- | :--- | :---: |
| Figure 1 | `figures/fig_pipeline.tex` | End-to-end PRISM Android system architecture | `CREATED & VERIFIED` |
| Figure 2 | `figures/fig_app_interface.tex` | PRISM live Android application interface screenshot | `CREATED & VERIFIED` |
| Figure 3 | `figures/fig_projection.tex` | Ground-plane projection geometry with IMU pitch/roll | `CREATED & VERIFIED` |
| Figure 4 | `figures/fig_hazard_lifecycle.tex` | 9-state hazard lifecycle state machine | `CREATED & VERIFIED` |
| Table 1 | `tables/tab_configuration.tex` | Frozen PRISM-L1.0 Android configuration parameters | `CREATED & VERIFIED` |
| Table 2 | `tables/tab_evaluation_setup.tex` | Controlled Android engineering evaluation framework and datasets | `CREATED & VERIFIED` |
| Table 3 | `tables/tab_engineering_results.tex`| Measured Android engineering performance benchmark | `CREATED & VERIFIED` |
| Table 4 | `tables/tab_ab_results.tex` | Controlled multisensory A/B evaluation (Mode A vs B) | `CREATED & VERIFIED` |

---

## 5. Citations and References
| Reference Key | Citation | Verification Required | Status |
| :--- | :--- | :--- | :---: |
| `lin2021real` | Lin et al., IEEE RA-L 2021 | Monocular ground-plane distance estimation | `VERIFIED` |
| `tapu2020wearable` | Tapu et al., Pattern Recognition Letters 2020 | Wearable assistive devices survey | `VERIFIED` |
| `elmannai2017sensor` | Elmannai \& Ellefsen, IEEE Sensors Journal 2017 | Sensor-based assistive devices survey | `VERIFIED` |
| `dakopoulos2009wearable` | Dakopoulos \& Bourbakis, IEEE TSMC-C 2010 | Wearable obstacle avoidance ETAs survey | `VERIFIED` |
| `roentgen2008electronic` | Roentgen et al., Technology and Disability 2008 | ETA literature review | `VERIFIED` |
| `loomis1998navigation` | Loomis et al., Presence 1998 | Spatial displays for blind navigation | `VERIFIED` |
| `bujacz2016sonification` | Bujacz \& Strumillo, Sensors 2016 | Auditory displays for sensory augmentation | `VERIFIED` |
| `shoval1998auditory` | Shoval et al., IEEE TBME 1998 | NavBelt computerized travel aid | `VERIFIED` |
| `bai2017smart` | Bai et al., IEEE Trans. Consumer Electronics 2017 | Smart guiding glasses | `VERIFIED` |
| `zhao2020visual` | Zhao et al., IEEE Access 2020 | Multi-cue obstacle avoidance | `VERIFIED` |
| `faria2010electronic`| Faria et al., IEEE WAC 2010 | Electronic white cane assistance | `VERIFIED` |
| `hartley2003multiple`| Hartley \& Zisserman, Cambridge Univ. Press 2003 | Multiple view geometry textbook | `VERIFIED` |
| `zhang2000flexible` | Zhang, IEEE TPAMI 2000 | Camera calibration technique | `VERIFIED` |
| `criminisi2000single`| Criminisi et al., IJCV 2000 | Single view metrology | `VERIFIED` |
| `hoiem2007recovering`| Hoiem et al., IJCV 2007 | Recovering surface layout from an image | `VERIFIED` |
| `cathey2005perspective`| Cathey \& Dailey, IEEE T-ITS 2005 | Perspective transform distance estimation | `VERIFIED` |
| `welch1995introduction`| Welch \& Bishop, UNC Chapel Hill 1995 | Introduction to Kalman filter | `VERIFIED` |
| `kuhn1955hungarian` | Kuhn, Naval Research Logistics Quarterly 1955 | Hungarian method for assignment | `VERIFIED` |
| `bewley2016simple` | Bewley et al., IEEE ICIP 2016 | Simple online and realtime tracking (SORT) | `VERIFIED` |
| `wojke2017simple` | Wojke et al., IEEE ICIP 2017 | DeepSORT tracking | `VERIFIED` |
| `barshalom1988tracking`| Bar-Shalom \& Fortmann, Academic Press 1988 | Tracking and data association | `VERIFIED` |
| `mahony2008nonlinear`| Mahony et al., IEEE TAC 2008 | Nonlinear complementary filters on SO(3) | `VERIFIED` |
| `madgwick2011estimation`| Madgwick et al., IEEE ICORR 2011 | IMU and MARG orientation estimation | `VERIFIED` |
| `foxlin1996inertial` | Foxlin, IEEE VRAIS 1996 | Inertial tracker complementary filter | `VERIFIED` |
| `harle2013survey` | Harle, IEEE COMST 2013 | Indoor pedestrian dead reckoning survey | `VERIFIED` |
| `hayward1972characteristics`| Hayward, Highway Research Record 1972 | Near-collision analysis | `VERIFIED` |
| `tam2009review` | Tam \& Bucknall, J. Navigation 2009 | Collision avoidance review | `VERIFIED` |
| `lee1976theory` | Lee, Perception 1976 | Theory of time-to-collision (TTC) | `VERIFIED` |
| `fiorini1998motion` | Fiorini \& Shiller, IJRR 1998 | Velocity obstacles for dynamic environments | `VERIFIED` |
| `vandenberg2011reciprocal`| van den Berg et al., ISRR 2011 | Reciprocal collision avoidance | `VERIFIED` |
| `helbing1995social` | Helbing \& Molnar, Physical Review E 1995 | Social force model for pedestrian dynamics | `VERIFIED` |
| `blauert1997spatial` | Blauert, MIT Press 1997 | Spatial hearing psychophysics | `VERIFIED` |
| `walker2006navigation`| Walker \& Lindsay, ACM TAP 2006 | Virtual auditory displays for navigation | `VERIFIED` |
| `giudice2008blind` | Giudice \& Legge, Handbook of Smart Tech 2008 | Blind navigation and assistive technology | `VERIFIED` |
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

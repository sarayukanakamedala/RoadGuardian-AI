# Dataset Selection Decision Document

**Project:** RoadGuardian AI: Video-Based Road Accident Detection and Intelligent Emergency Incident Management  
**Decision Status:** APPROVED & FINALIZED  
**Selected Benchmark:** Car Crash Dataset (CCD)  
**Official Repository:** [Cogito2012/CarCrashDataset (GitHub)](https://github.com/Cogito2012/CarCrashDataset)  
**Academic Publication:** ACM Multimedia Conference 2020 (ACM MM '20)  

---

## 1. Executive Summary & Selection Decision

Following an exhaustive architectural evaluation of candidate video datasets against the 14 dimensions defined in [`docs/dataset_selection_criteria.md`](dataset_selection_criteria.md), **Car Crash Dataset (CCD)** has been officially approved and selected as the foundational benchmark for RoadGuardian AI.

CCD provides 4,500 standardized video clips (1,500 accident sequences and 3,000 normal driving sequences), formatted consistently as 50-frame, 10-FPS `.mp4` video files spanning approximately 5 seconds each. This temporal structure is mathematically aligned with RoadGuardian AI's **ResNet-18 + GRU** spatio-temporal architecture.

---

## 2. Why the Car Crash Dataset (CCD) Was Selected

### 2.1 Architectural Alignment with ResNet-18 + GRU
- **Uniform Temporal Geometry:** Each clip contains exactly 50 frames at 10 frames per second ($5.0\text{ seconds}$). This uniform duration eliminates erratic sequence padding during temporal batch generation.
- **Continuous Motion Sequences:** Unlike static image collections, CCD captures the full kinetic progression of a collision: normal trajectory $\rightarrow$ critical pre-impact deceleration/swerve $\rightarrow$ impact shock $\rightarrow$ post-collision halt.
- **Sliding Window Compatibility:** A sliding temporal window of $T=16$ frames with stride $s=2$ cleanly decomposes each 50-frame clip into overlapping sequential inputs for GRU recurrent modeling.

### 2.2 Rich Environmental Metadata
CCD includes fine-grained environmental annotations across every clip:
- **Lighting Conditions:** Day / Night.
- **Weather Variations:** Normal / Snowy / Rainy.
- **Vehicle Involvement:** Ego-vehicle involved vs. Non-ego vehicle (bystander) collisions.
These metadata attributes permit rigorous subgroup validation to ensure RoadGuardian AI does not exhibit biased performance in low-light or adverse weather scenarios.

### 2.3 Temporal Ground Truth Annotations
CCD provides frame-level accident occurrence annotations identifying the precise moment of impact within the 50-frame sequence. This provides objective ground truth for evaluating our automated evidence snapshot capture and temporal warning latency.

---

## 3. Comparison with Rejected Alternatives

| Dataset | Status | Architectural Evaluation & Rejection Rationale |
| :--- | :--- | :--- |
| **Static CCTV Frame Datasets** | **REJECTED** | Image-only datasets (e.g., static bounding box crash snapshots) fail to capture velocity drops, motion flow, and temporal dynamics. Using static images would defeat the purpose of recurrent temporal modeling with GRUs. |
| **UCF-Crime (Accident Subset)** | **REJECTED** | Composed of long, untrimmed surveillance recordings (1 to 5 minutes each). The collision represents an isolated 5-second event within hundreds of seconds of unrelated footage, requiring weakly supervised temporal localization before classification can even begin. |
| **DoTA (Detection of Traffic Anomaly)** | **REJECTED** | Massive storage footprint (>60 GB) spanning 18 multi-class anomaly categories. Downloading multi-part Baidu archives presents significant accessibility friction, and the taxonomy introduces unnecessary multi-class complexity for binary collision detection. |
| **CADP (Car Accident Detection & Prediction)** | **REJECTED** | Relies on a YouTube web-scraping script where 30–40% of originating videos have been deleted, suspended, or geo-restricted over time, severely compromising experimental reproducibility. |

---

## 4. Dataset Strengths & Operational Limitations

### 4.1 Strengths
1. **Curated & Verified:** Peer-reviewed benchmark published at ACM Multimedia 2020.
2. **Standardized Encoding:** Consistent H.264 / MP4 encoding with verified decodability via OpenCV (`cv2.VideoCapture`).
3. **Manageable Footprint:** Clean 4,500-clip archive totaling approximately 3 to 5 GB, making it practical for local student workstations and Google Colab GPUs.
4. **Clean Binary Class Distinction:** Clear ground truth labels without ambiguous near-miss edge cases.

### 4.2 Limitations & Mitigation Strategy
- **Limitation 1: Camera Perspective (Dashcam Bias):** Most CCD clips are captured from dashcam or front-facing viewpoints rather than high-angle municipal pole-mounted CCTV cameras.
  - *Mitigation:* ResNet-18 spatial feature representations generalize across perspective viewpoints because vehicle deformation and sudden kinetic deceleration remain visually invariant. In Phase 8, external CCTV test clips will be evaluated for out-of-domain robustness.
- **Limitation 2: Fixed 10 FPS Sampling Rate:** Professional surveillance cameras typically operate at 25 or 30 FPS.
  - *Mitigation:* 10 FPS is computationally advantageous for deep learning sequence modeling, as it captures 5 seconds of motion dynamics in just 50 frames, reducing recurrent unrolling length and GPU memory pressure.

---

## 5. Class Imbalance Analysis & Mitigation Strategy

### 5.1 Observed Imbalance Ratio
- **Accident Clips:** 1,500 ($33.33\%$)
- **Normal Driving Clips:** 3,000 ($66.67\%$)
- **Natural Class Ratio:** $1 : 2$ (Accident : Normal)

While a $1:2$ ratio is common in real-world traffic where accidents are rare events, naive cross-entropy loss could cause the neural network to bias toward predicting the majority (normal) class.

### 5.2 Mitigation Strategy for RoadGuardian AI
1. **Stratified Video-Level Splitting:** Partitioning (`scripts/create_video_splits.py`) will preserve the exact $1:2$ ratio across `TRAIN`, `VALIDATION`, and `TEST` splits, ensuring no class shift across partitions.
2. **Class-Weighted Loss Function:** The binary cross-entropy loss will apply an inverse class weight to penalize false negatives (missed accidents) more heavily:
   $$w_{\text{accident}} = \frac{N_{\text{total}}}{2 \times N_{\text{accident}}} = \frac{4500}{2 \times 1500} = 1.5$$
   $$w_{\text{normal}} = \frac{4500}{2 \times 3000} = 0.75$$
3. **Threshold Calibration:** The default classification decision threshold will be empirically tuned on the validation set using Precision-Recall Curves rather than assuming an uncalibrated $\tau = 0.50$.

---

## 6. Academic Citation & Provenance

When citing the Car Crash Dataset in academic reports, research papers, or project viva presentations, the following reference must be used:

```bibtex
@inproceedings{bao2020uncertainty,
  title={Uncertainty-based Traffic Accident Anticipation with Spatio-Temporal Relational Learning},
  author={Bao, Wentao and Yu, Qi and Kong, Yu},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia (ACM MM '20)},
  pages={4282--4290},
  year={2020},
  doi={10.1145/3394171.3413941}
}
```

---

## 7. Ethical, Privacy & Licensing Guidelines

1. **Non-Commercial Academic Use:** The Car Crash Dataset is provided strictly for non-commercial educational and research purposes.
2. **Privacy Protection:** The dataset comprises public roadway footage. Individual license plates and pedestrian identities are not high-resolution or indexed, minimizing privacy risks.
3. **Safe Simulation Boundary:** In adherence to [`docs/project_scope.md`](project_scope.md), all emergency service dispatches triggered upon detecting accidents in CCD videos remain strictly simulated.

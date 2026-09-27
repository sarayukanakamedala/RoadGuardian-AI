# Road Accident Dataset Selection Criteria & Evaluation Rubric

**Document Version:** 1.0.0  
**Project:** RoadGuardian AI: Video-Based Road Accident Detection and Intelligent Emergency Incident Management  
**Status:** Evaluation Framework Active (Dataset Selection Pending Manual Review)  

---

## 1. Purpose & Scope

This document defines the formal criteria and scoring rubric used to evaluate candidate video datasets for training and benchmarking **RoadGuardian AI**. 

Because video-based deep learning models (ResNet-18 + GRU) require coherent spatio-temporal sequences rather than isolated static frames, dataset selection directly governs model generalization, training feasibility, and academic credibility during B.Tech project defense.

> [!IMPORTANT]
> In adherence to academic research guidelines, **no dataset is selected automatically**. Candidate datasets must be systematically scored against the criteria below before acquisition.

---

## 2. The 14 Evaluation Dimensions

Every candidate dataset is evaluated across fourteen explicit dimensions:

### 2.1 Legal, Access & Licensing Dimensions
1. **Free Availability:** The dataset must be publicly accessible without requiring paid access or proprietary enterprise agreements.
2. **Academic & Research Usage Rights:** Licensing terms (e.g., CC BY-NC 4.0, MIT, or verified university research release) must legally permit academic usage, model training, and portfolio publication.
3. **Download Accessibility & Reliability:** The dataset must have stable, verifiable hosting (e.g., GitHub, Zenodo, Hugging Face, Kaggle, university servers) with direct download options rather than deprecated links or unmaintained mirrors.

### 2.2 Volume, Balance & Label Quality Dimensions
4. **Number of Accident Samples:** Sufficient volume of distinct collision events (recommended minimum: 200–500 distinct accident videos) to prevent severe overfitting on spatial representations.
5. **Number of Normal Samples:** Comparable or balanced volume of normal traffic sequences (recommended 1:1 to 1:3 ratio) spanning congested, free-flow, and rural traffic.
6. **Video Availability vs. Static Frames:** The dataset **must provide raw or reconstructed video clips** (`.mp4`, `.avi`, `.mkv`), not merely isolated bounding box cropped images, as GRU sequence modeling relies on continuous frame sequences.
7. **Label Quality & Annotation Granularity:** Labels must be unambiguous (clean separation between normal maneuvers and genuine collisions).
8. **Accident Timestamps / Temporal Boundaries:** Preference is given to datasets providing temporal bounds (start frame, impact frame, post-crash recovery frame) rather than unsegmented whole-video labels.

### 2.3 Technical, Diversity & Hardware Feasibility Dimensions
9. **Dataset Size (Storage Footprint):** The uncompressed video archive should be manageable on a student workstation / development machine (recommended: 5 GB to 30 GB total).
10. **Video Format & Codec Compatibility:** Video files must decode reliably via standard `cv2.VideoCapture` (H.264, MPEG-4) without requiring proprietary decoders.
11. **Environmental & Scene Diversity:** Videos should encompass diverse lighting (day, night, dusk), weather conditions (clear, rain, glare), and camera viewpoints (fixed surveillance CCTV, dashcam, overhead).
12. **Relevance to ResNet-18 + GRU:** Video durations must suit sliding temporal sequence windows ($T=16$ or $30$ frames), exhibiting discernible motion transitions leading up to and following the collision.
13. **Suitability for Final-Year Academic Capstone:** The dataset must be recognized or cited in peer-reviewed literature, establishing a credible benchmark for project viva defense.
14. **Hardware Feasibility:** Spatial resolutions and total frame counts must allow feature extraction on accessible consumer GPUs (e.g., NVIDIA RTX 3050/3060/4060 or Google Colab T4) within reasonable training times.

---

## 3. Candidate Dataset Profiles (For Evaluation)

The following public datasets are prominent candidates in traffic anomaly and road accident literature:

| Candidate Dataset | Origin / Paper | Primary Camera Type | Key Characteristics |
| :--- | :--- | :--- | :--- |
| **CADP (Car Accident Detection and Prediction)** | Shah et al. (CVPR Workshops) | Traffic CCTV / YouTube | 1,416 video clips with temporal labels for accident occurrence. |
| **UCF-Crime (Road Accident Subset)** | Sultani et al. (CVPR) | Fixed Surveillance CCTV | Real-world surveillance footage; subset contains genuine road collisions. |
| **DoTA (Detection of Traffic Anomaly)** | Yao et al. (TPAMI) | Dashcam / Vehicle Ego | Extensive accident taxonomy (head-on, rear-end, collision with pedestrian), temporal and spatial bounding boxes. |
| **Kaggle / Dashcam Road Accident Datasets** | Public Community Curations | Dashcam Videos | Readily accessible dashcam accident and normal driving clips; variable metadata. |
| **SOHAS (Smart Online Highway Accident Surveillance)** | Academic Benchmark | Highway CCTV | High-definition overhead highway CCTV accident sequences. |

---

## 4. Evaluation Scoring Rubric (Template)

Each criterion is scored on a scale of **1 (Poor)** to **5 (Exceptional)**.

### Priority Weighting:
- **Category A: Accessibility & Integrity (Weight: 30%)** - Free, licensed, decodable video format, stable link.
- **Category B: Spatio-Temporal Suitability (Weight: 40%)** - True video sequences, temporal timestamps, balance, sample volume, relevance to ResNet+GRU.
- **Category C: Engineering & Defense Practicality (Weight: 30%)** - Hardware feasibility, manageable storage footprint, academic citation value.

### Candidate Comparison Table (To be completed during manual selection):

```
+------------------------------------+--------+-----------+--------+---------------+--------+
| Evaluation Criterion (1 to 5)      | Weight | Candidate | Candidate | Candidate  | Candidate |
|                                    |        | 1: CADP   | 2: UCF | 3: DoTA       | 4: Dashcam|
+------------------------------------+--------+-----------+--------+---------------+--------+
| 1. Free Public Availability        |   6%   |    [ ]    |  [ ]   |      [ ]      |    [ ]    |
| 2. Academic / Research License     |   8%   |    [ ]    |  [ ]   |      [ ]      |    [ ]    |
| 3. Download Accessibility & Host   |   6%   |    [ ]    |  [ ]   |      [ ]      |    [ ]    |
| 4. Video Availability (Continuous) |  10%   |    [ ]    |  [ ]   |      [ ]      |    [ ]    |
| 5. Volume of Accident Samples      |   8%   |    [ ]    |  [ ]   |      [ ]      |    [ ]    |
| 6. Volume of Normal Samples        |   6%   |    [ ]    |  [ ]   |      [ ]      |    [ ]    |
| 7. Label Quality & Accuracy        |   6%   |    [ ]    |  [ ]   |      [ ]      |    [ ]    |
| 8. Accident Timestamp Granularity  |   6%   |    [ ]    |  [ ]   |      [ ]      |    [ ]    |
| 9. Storage Footprint Feasibility   |   6%   |    [ ]    |  [ ]   |      [ ]      |    [ ]    |
| 10. Codec / OpenCV Decodability    |  10%   |    [ ]    |  [ ]   |      [ ]      |    [ ]    |
| 11. Environmental Diversity        |   6%   |    [ ]    |  [ ]   |      [ ]      |    [ ]    |
| 12. Relevance to ResNet18 + GRU    |   8%   |    [ ]    |  [ ]   |      [ ]      |    [ ]    |
| 13. Academic Citation & Viva Value |   6%   |    [ ]    |  [ ]   |      [ ]      |    [ ]    |
| 14. Local Hardware Feasibility     |   6%   |    [ ]    |  [ ]   |      [ ]      |    [ ]    |
+------------------------------------+--------+-----------+--------+---------------+--------+
| Weighted Total Score (0 - 100%)    |  100%  |    TBD    |  TBD   |      TBD      |    TBD    |
+------------------------------------+--------+-----------+--------+---------------+--------+
```

---

## 5. Next Steps for Dataset Finalization

1. **Candidate Review:** The project team reviews the candidate datasets against available network bandwidth and local disk storage.
2. **Download & Sample Audit:** A small representative sample (~10 clips) is downloaded to verify `cv2.VideoCapture` decodability using `scripts/verify_dataset.py`.
3. **Formal Selection Sign-Off:** The chosen dataset is documented with its citation in `dataset/README.md`.
4. **Manifest Generation:** Videos are indexed into `dataset/metadata/dataset_manifest.csv`.
5. **Video-Level Partitioning:** `scripts/create_video_splits.py` is executed to finalize `TRAIN`, `VALIDATION`, and `TEST` splits.

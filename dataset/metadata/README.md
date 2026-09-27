# Dataset Metadata, Partitioning & Provenance Protocol

This document governs the metadata architecture, split methodology, and provenance tracking for **RoadGuardian AI**.

---

## 1. Role of the Dataset Manifest

The dataset manifest (`dataset_manifest.csv`) serves as the authoritative single source of truth for the entire machine learning lifecycle. It catalogs all physical video clips alongside their ground truth labels, container properties, split assignments, and provenance attributes.

Downstream pipelines (preprocessing, feature caching, PyTorch `Dataset` loaders, and model evaluation routines) must **never** infer labels or splits by scanning raw folder trees directly. Instead, all loaders read from the validated manifest, guaranteeing determinism, reproducibility, and verifiable data pipelines.

---

## 2. The Video-Level Splitting Imperative

> [!CAUTION]
> **CRITICAL ARCHITECTURAL RULE: Splitting MUST Occur Strictly at the VIDEO Level, Never at the Frame Level.**

In video-based deep learning, one of the most pervasive methodological flaws is **temporal data leakage** (frame leakage).

### 2.1 The Frame Leakage Pitfall
Consider a 10-second traffic video recorded at 30 FPS, containing 300 frames. Consecutive frames ($f_t$ and $f_{t+1}$) recorded $\frac{1}{30}\text{th}$ of a second apart share virtually identical visual information:
- Identical background scenery (buildings, road geometry, trees, lane markers).
- Identical weather and lighting conditions (shadow angles, overcast cloud cover).
- Identical vehicles in almost identical pixel positions.

If a dataset is shuffled and split at the **frame level**:
- Frames from the *same accident sequence* will populate both the `TRAIN` set and the `TEST` set.
- The neural network (ResNet-18) will easily memorize static scene artifacts (e.g., a specific billboard or road texture) rather than generalizing to the spatio-temporal dynamics of a vehicular collision.
- The resulting test evaluation will report spuriously inflated metrics (e.g., 99% accuracy) that catastrophically degrade when evaluated on novel, unseen video footage.

### 2.2 Strict Video-Level Isolation
To ensure genuine out-of-sample generalization:
1. Every unique video clip is assigned **in its entirety** to exactly one partition: `TRAIN`, `VALIDATION`, or `TEST`.
2. All temporal windows ($T$-frame sequences) generated from `video_id = VID_001` belong exclusively to the split containing `VID_001`.
3. No scene background or video identifier may ever span multiple splits.

```
┌────────────────────────────────────────────────────────┐
│               Raw Candidate Video Pool                 │
│         [VID_001] [VID_002] [VID_003] ... [VID_N]       │
└──────────────────────────┬─────────────────────────────┘
                           │
             Video-Level Stratified Partition
            (Random Seed = 42, No Cross-Split Leakage)
                           │
       ┌───────────────────┼───────────────────┐
       ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  TRAIN (70%) │    │   VAL (15%)  │    │  TEST (15%)  │
│  VID_001     │    │  VID_002     │    │  VID_003     │
│  All frames  │    │  All frames  │    │  All frames  │
│  stay here   │    │  stay here   │    │  stay here   │
└──────────────┘    └──────────────┘    └──────────────┘
```

---

## 3. Dataset Provenance & Academic Integrity

In an academic B.Tech capstone and portfolio project, provenance—the verifiable history of data origin, licensing, and transformation—is vital for academic defense.

### Provenance Requirements:
1. **Source Citation:** Every entry in `dataset_manifest.csv` must record `source_dataset` and `source_url`.
2. **Licensing Compliance:** Models must only be trained on data authorized for educational and academic research (e.g., Creative Commons, MIT, academic non-commercial agreements).
3. **Imbalance Documentation:** True class proportions (accident vs. normal) must be empirically recorded. Imbalances must be handled explicitly through stratified splitting and weighted loss functions rather than hidden data filtering.
4. **Integrity Validation:** Every video must be verified with `cv2.VideoCapture` to ensure the file stream contains valid, uncorrupted video packets before entering the manifest (`is_decodable = True`).

---

## 4. Reproducibility Requirements

To guarantee that any project examiner, peer reviewer, or placement interviewer can replicate the experimental results:
1. **Deterministic Random Seeds:** Partitioning scripts (`scripts/create_video_splits.py`) must enforce a fixed random seed (default: `42`).
2. **Immutable Manifest Artifact:** Once audited and verified, `dataset_manifest.csv` is committed to version control as a lightweight text record.
3. **Zero Untracked Local Dependencies:** No local hardcoded absolute paths (`C:\...`) are permitted in metadata or code. All references are relative to the project root.

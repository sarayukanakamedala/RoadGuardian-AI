# RoadGuardian AI - Dataset Directory & Specifications

This directory houses the raw video clips, processed spatial-temporal frame representations, and metadata manifests for the **RoadGuardian AI** accident detection pipeline.

> [!IMPORTANT]
> **No dataset has been selected or pre-packaged.**  
> In accordance with academic research integrity, dataset acquisition requires formal evaluation against our [Dataset Selection Criteria](../docs/dataset_selection_criteria.md). This document specifies the target schema and directory contract required by our data ingestion scripts.

---

## 1. Directory Structure

```
dataset/
├── raw/                          # Original video files (strictly gitignored)
│   ├── accident/                 # Raw video clips containing traffic collisions
│   │   └── .gitkeep
│   └── normal/                   # Raw video clips of normal driving/traffic flow
│       └── .gitkeep
│
├── processed/                    # Preprocessed assets (strictly gitignored)
│   ├── frames/                   # Extracted and resized video frames (224x224)
│   ├── sequences/                # Uniform sliding temporal sequences (T=16 frames)
│   └── features/                 # Cached ResNet-18 spatial embedding tensors (512-d)
│
└── metadata/                     # Structured manifests and audit trails (tracked)
    ├── README.md                 # Manifest schema and split rules
    ├── dataset_manifest_template.csv # Empty schema template for recording video metadata
    └── dataset_manifest.csv      # (Populated after dataset selection & audit)
```

---

## 2. Target Dataset Specifications & Classes

To support binary spatio-temporal classification with ResNet-18 + GRU, the candidate dataset must provide samples for two discrete classes:

| Class Identifier | Class Name | Description | Example Visual Context |
| :--- | :--- | :--- | :--- |
| `1` | `accident` | Video footage depicting a collision, rollover, sudden impact, or motorcycle fall. | Vehicle-to-vehicle collision, vehicle-to-barrier impact, pedestrian incident, intersection crash. |
| `0` | `normal` | Video footage depicting continuous, non-collision traffic flow or normal driving maneuvers. | Highway cruising, intersection queueing, stop-and-go congestion, lane changes, adverse weather driving. |

---

## 3. Metadata Field Definitions

When cataloging ingested videos into `dataset/metadata/dataset_manifest.csv`, every video must be documented against the following standardized fields:

1. **`video_id`** *(string)*: Unique alphanumeric identifier assigned to each video clip (e.g., `VID_ACC_001`, `VID_NORM_042`).
2. **`filename`** *(string)*: Exact local filename within `dataset/raw/` (e.g., `accident_clip_001.mp4`).
3. **`source_dataset`** *(string)*: Originating academic benchmark or public repository name (e.g., `CADP`, `UCF-Crime-Accident`, `DoTA`).
4. **`source_url`** *(string)*: Canonical URL or DOI link to the source repository or publisher publication.
5. **`license`** *(string)*: Terms of distribution (e.g., `CC BY-NC 4.0`, `MIT`, `Academic Use Only`). Must permit educational/research usage.
6. **`class_label`** *(string)*: Ground truth categorical designation: `accident` or `normal`.
7. **`duration_seconds`** *(float)*: Duration of the video in seconds (recommended 3.0s to 15.0s for concise temporal modeling).
8. **`fps`** *(float)*: Native frames per second of the video (e.g., 25.0, 30.0).
9. **`width`** *(integer)*: Native horizontal resolution in pixels (e.g., 1280, 1920).
10. **`height`** *(integer)*: Native vertical resolution in pixels (e.g., 720, 1080).
11. **`file_size_mb`** *(float)*: File size in megabytes on disk.
12. **`split`** *(string)*: Data partition assigned at the **video level**: `TRAIN`, `VALIDATION`, or `TEST`.
13. **`is_decodable`** *(boolean)*: Integrity verification flag (`True` / `False`) indicating whether `cv2.VideoCapture` successfully reads all frames without stream corruption.
14. **`notes`** *(string)*: Descriptive observational tags (e.g., `night_time`, `rain`, `intersection`, `rear_end`, `occluded`).

---

## 4. Ingestion Workflow

Once a dataset is selected following [`docs/dataset_selection_criteria.md`](../docs/dataset_selection_criteria.md):

1. Download the raw archive into a temporary extraction directory.
2. Place video clips into `dataset/raw/accident/` and `dataset/raw/normal/`.
3. Run `python scripts/verify_dataset.py --dataset_dir dataset/raw` to inspect decodability and extract dimensions into `dataset/metadata/dataset_audit.csv`.
4. Populate `dataset/metadata/dataset_manifest.csv` based on the audit findings.
5. Execute `python scripts/create_video_splits.py --manifest dataset/metadata/dataset_manifest.csv` to partition at the video level.
6. Conduct exploratory data analysis using `notebooks/dataset_analysis/01_dataset_eda.ipynb`.

---

## 5. PyTorch Video Preprocessing & Dataset Loader

The preprocessing pipeline (`ml/preprocessing/`) provides direct, zero-overhead ingestion into PyTorch without generating hundreds of thousands of loose frame images:

1. **`VideoDataset` (`ml/preprocessing/video_dataset.py`)**:
   - Reads directly from authoritative `dataset/metadata/dataset_manifest.csv`.
   - Strictly enforces video-level partition (`split='TRAIN'`, `'VALIDATION'`, or `'TEST'`).
   - Resolves all video paths relative to `PROJECT_ROOT`.
   - Performs on-the-fly OpenCV video decoding with explicit BGR $\rightarrow$ RGB conversion.
   - Deterministically samples $T=16$ frames using uniform temporal spacing across the 50 frames:
     `np.linspace(0, total_frames - 1, num=16, dtype=int)` (indices: 0, 3, 6, 9, 13, 16, 19, 22, 26, 29, 32, 35, 39, 42, 45, 49).
   - Returns `(frames: FloatTensor [16, 3, 224, 224], label: LongTensor, video_id: str)`.

2. **`VideoTransform` (`ml/preprocessing/transforms.py`)**:
   - Resizes frames to $224 \times 224$.
   - Scales pixel intensities to $[0.0, 1.0]$ and applies standard ImageNet normalization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).

3. **Smoke Verification (`scripts/test_preprocessing.py`)**:
   ```bash
   .venv\Scripts\python.exe scripts\test_preprocessing.py
   ```
   Validates tensor dimensions, label consistency, finite values, and zero cross-partition leakage across all splits.


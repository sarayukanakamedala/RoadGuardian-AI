# Car Crash Dataset (CCD) - Acquisition & Ingestion Protocol

**Document:** Dataset Acquisition & Verification Plan  
**Target Benchmark:** Car Crash Dataset (CCD - ACM MM '20)  
**Storage Target:** `dataset/raw/` (Accident: `dataset/raw/accident/`, Normal: `dataset/raw/normal/`)  
**Status:** ACQUISITION IN PROGRESS  

---

## 1. End-to-End Ingestion Workflow

To ensure data integrity, reproducibility, and prevent model training failure, the ingestion of CCD follows a strict, phased pipeline:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. CCD Download (Official Cogito2012 Repository / Mirrors)  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Small Sample Verification (~10 Clips via OpenCV)         │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Full Dataset Acquisition (1,500 Crash + 3,000 Normal)    │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Dataset Audit (scripts/verify_dataset.py)                │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Manifest Creation (dataset_manifest.csv)                 │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. Video-Level Stratified Partition (create_video_splits.py)│
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. Exploratory Data Analysis (01_dataset_eda.ipynb)         │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 8. Gateway to Baseline Model Development (Phase 3)          │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Step-by-Step Implementation Guide

### Step 1: CCD Download
- **Official Source:** [Cogito2012/CarCrashDataset (GitHub)](https://github.com/Cogito2012/CarCrashDataset)
- **Publication:** Bao et al., *"Uncertainty-based Traffic Accident Anticipation with Spatio-Temporal Relational Learning"*, ACM MM 2020.
- **Manual Acquisition:** The developer downloads the video archives from the official Google Drive / Baidu mirrors provided in the repository.
- *Rule:* Automated scraping or direct unauthorized downloading is prohibited; use official release mirrors.

### Step 2: Small Sample Verification (Sanity Check)
Before staging the full 4,500 video dataset, acquire and stage a minimal sample of **5 accident clips** and **5 normal clips**:
```
dataset/raw/
├── accident/
│   ├── sample_crash_01.mp4
│   └── ...
└── normal/
    ├── sample_normal_01.mp4
    └── ...
```
Run a dry run of the verification script:
```bash
python scripts/verify_dataset.py --dataset_dir dataset/raw --output_csv dataset/metadata/sample_audit.csv
```
**Acceptance Criteria:**
- OpenCV successfully initializes `cv2.VideoCapture`.
- Confirmed stream properties: 50 frames, 10 FPS, ~5.0s duration, zero container corruption.

### Step 3: Full Dataset Acquisition
Once the sanity check passes:
1. Extract all 1,500 crash videos into `dataset/raw/accident/`.
2. Extract all 3,000 normal driving videos into `dataset/raw/normal/`.
3. Confirm local storage footprint (~3 GB to 5 GB).
4. Verify `.gitignore` prevents `dataset/raw/*.mp4` from being tracked by Git.

### Step 4: Full Dataset Audit
Execute the automated audit utility across the entire raw repository:
```bash
python scripts/verify_dataset.py --dataset_dir dataset/raw --output_csv dataset/metadata/dataset_audit.csv
```
This utility examines every video file, checks for corrupted frames, logs FPS and resolution consistency, and outputs `dataset/metadata/dataset_audit.csv`.

### Step 5: Manifest Creation
Populate `dataset/metadata/dataset_manifest.csv` based on the template schema:
- `video_id`: Unique identifier (e.g., `CCD_ACC_0001` to `CCD_ACC_1500`, `CCD_NORM_0001` to `CCD_NORM_3000`).
- `filename`: Local relative filename.
- `source_dataset`: `CarCrashDataset (CCD)`.
- `source_url`: `https://github.com/Cogito2012/CarCrashDataset`.
- `license`: `Academic / Research Non-Commercial`.
- `class_label`: `accident` or `normal`.
- Physical metrics from `dataset_audit.csv` (`duration_seconds`, `fps`, `width`, `height`, `file_size_mb`).
- Environmental tags in `notes` (e.g., `day`, `night`, `rain`, `snow`, `ego_involved`).

### Step 6: Video-Level Stratified Partition
Execute the leakage-free partition script:
```bash
python scripts/create_video_splits.py \
    --manifest dataset/metadata/dataset_manifest.csv \
    --train_ratio 0.70 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --seed 42
```
**Validation Assertions:**
- Partitioning occurs **strictly at the video level**.
- Zero cross-partition video ID overlap ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$).
- Preserves natural 1:2 class stratification:
  - **Train (70%):** ~1,050 Accident, ~2,100 Normal.
  - **Val (15%):** ~225 Accident, ~450 Normal.
  - **Test (15%):** ~225 Accident, ~450 Normal.

### Step 7: Exploratory Data Analysis (EDA)
Launch Jupyter and execute `notebooks/dataset_analysis/01_dataset_eda.ipynb`:
1. Verify class distributions and proportions.
2. Confirm duration histogram centers sharply at 5.0 seconds.
3. Confirm 10 FPS native frame rate.
4. Verify environmental subset distributions (Day vs. Night, Weather).
5. Document any outliers or findings in Markdown.

### Step 8: Gateway to Baseline Model Development (Phase 3)
Phase 2 concludes, and Phase 3 commences only when:
- [ ] 100% of candidate clips in `dataset_manifest.csv` are marked `is_decodable = True`.
- [ ] Train/Validation/Test split columns are populated with zero leakage.
- [ ] EDA notebook has been executed and validated.

---

## 3. Mandatory Quality & Integrity Constraints

> [!CAUTION]
> 1. **No Synthetic Video Generation:** Do not generate synthetic or dummy video files to simulate dataset presence.
> 2. **No Fake Metadata Population:** Manifest entries must only reflect real video files audited on disk.
> 3. **Preserve Architecture:** Preprocessing parameters (`DEFAULT_FRAME_SIZE = (224, 224)`, `DEFAULT_SEQUENCE_LENGTH = 16`) remain configured for ResNet-18 + GRU.

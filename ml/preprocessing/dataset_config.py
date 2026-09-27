"""
RoadGuardian AI - Dataset Preprocessing Configuration
=====================================================
Centralized configuration placeholders and default hyperparameters for video ingestion,
frame extraction, spatial transformations, and temporal sequence generation.

NOTE:
Values defined below serve as configurable baseline defaults. They can be tuned
or overridden based on the physical specifications of the selected dataset.
"""

from pathlib import Path

# ==============================================================================
# 1. Project Filesystem Path Placeholders (Relative to Project Root)
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Directory containing raw, uncompressed video archives (gitignored)
RAW_DATASET_DIR = PROJECT_ROOT / "dataset" / "raw"

# Directory storing processed artifacts (frames, temporal sequences, cached features)
PROCESSED_DATASET_DIR = PROJECT_ROOT / "dataset" / "processed"
FEATURES_DIR = PROCESSED_DATASET_DIR / "features"
FEATURE_MANIFEST_PATH = FEATURES_DIR / "feature_manifest.csv"

# Directory storing manifest CSVs, split indices, and audit records
METADATA_DIR = PROJECT_ROOT / "dataset" / "metadata"

# Authoritative dataset manifest file path
DATASET_MANIFEST_PATH = METADATA_DIR / "dataset_manifest.csv"
DATASET_AUDIT_PATH = METADATA_DIR / "dataset_audit.csv"

# ==============================================================================
# 2. Ground Truth Target Labels & Encodings
# ==============================================================================
# Binary classification class labels
ACCIDENT_LABEL = "accident"
NORMAL_LABEL = "normal"

# Numeric mapping for model target tensors
LABEL_TO_ID = {
    NORMAL_LABEL: 0,
    ACCIDENT_LABEL: 1,
}

ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

# ==============================================================================
# 3. Spatial & Video Preprocessing Configurable Defaults
# ==============================================================================
# Spatial input resolution expected by standard ResNet backbones (height, width)
DEFAULT_FRAME_SIZE = (224, 224)

# Number of consecutive frames per temporal window input to GRU sequence head
DEFAULT_SEQUENCE_LENGTH = 16

# Sampling stride across input video (1 = every consecutive frame, 2 = every 2nd frame)
DEFAULT_FRAME_SAMPLING_RATE = 1

# Supported video container formats
SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov")

# Default dataset split ratios (must sum to 1.0)
DEFAULT_SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

# Fixed random seed for deterministic reproducibility across data splits
RANDOM_SEED = 42

"""
RoadGuardian AI - Cached Feature Dataset
========================================
PyTorch Dataset for loading pre-extracted, cached ResNet-18 spatial feature
sequences [16, 512] directly from disk without decoding raw video streams.

Key Guarantees:
- Strictly partitioned by dataset split (TRAIN, VALIDATION, TEST) via feature_manifest.csv.
- Zero disk scanning outside the authoritative feature manifest.
- Resolves all feature_path entries relative to PROJECT_ROOT.
- Enforces strict tensor shape validation: [16, 512].
- Returns (features: FloatTensor [16, 512], label: LongTensor, video_id: str).
"""

import csv
from pathlib import Path
from typing import Optional, Union

import torch
from torch.utils.data import Dataset

from ml.preprocessing.dataset_config import (
    FEATURE_MANIFEST_PATH as DEFAULT_FEATURE_MANIFEST_PATH,
    FEATURES_DIR as DEFAULT_FEATURES_DIR,
    PROJECT_ROOT,
)


class CachedFeatureDataset(Dataset):
    """
    PyTorch Dataset for precomputed ResNet-18 spatial feature sequences.

    Loads cached .pt feature files listed in feature_manifest.csv for a specific
    data split (TRAIN, VALIDATION, or TEST).
    """

    def __init__(
        self,
        split: str,
        manifest_path: Union[str, Path] = DEFAULT_FEATURE_MANIFEST_PATH,
        project_root: Union[str, Path] = PROJECT_ROOT,
        expected_seq_len: int = 16,
        expected_feature_dim: int = 512,
    ):
        """
        Args:
            split: Target data split: 'TRAIN', 'VALIDATION', or 'TEST'.
            manifest_path: Path to feature_manifest.csv.
            project_root: Root directory to resolve relative feature_path from.
            expected_seq_len: Expected temporal sequence length (default: 16).
            expected_feature_dim: Expected spatial feature dimension (default: 512).
        """
        self.split = split.upper().strip()
        if self.split not in ("TRAIN", "VALIDATION", "TEST"):
            raise ValueError(
                f"[CachedFeatureDataset] Invalid split '{split}'. Expected 'TRAIN', 'VALIDATION', or 'TEST'."
            )

        self.manifest_path = Path(manifest_path).resolve()
        self.project_root = Path(project_root).resolve()
        self.expected_seq_len = expected_seq_len
        self.expected_feature_dim = expected_feature_dim

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"[CachedFeatureDataset] Feature manifest file not found: {self.manifest_path}"
            )

        # Load records filtered by the designated split
        self.records = []
        with open(self.manifest_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("split", "").strip().upper() == self.split:
                    self.records.append(row)

        if not self.records:
            raise RuntimeError(
                f"[CachedFeatureDataset] Zero feature records found for split '{self.split}' "
                f"in manifest: {self.manifest_path}"
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        """
        Loads a cached feature sequence.

        Returns:
            features: FloatTensor of shape [16, 512]
            label: LongTensor scalar (0 for normal, 1 for accident)
            video_id: Unique string identifier for the video clip
        """
        record = self.records[idx]
        video_id = record["video_id"]
        rel_feature_path = record["feature_path"]
        feature_file = self.project_root / rel_feature_path

        if not feature_file.exists():
            raise FileNotFoundError(
                f"[CachedFeatureDataset] Feature cache file missing for video '{video_id}': {feature_file}"
            )

        # Load saved payload dictionary
        payload = torch.load(feature_file, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError(
                f"[CachedFeatureDataset] Corrupted cache payload in {feature_file}: expected dict, got {type(payload)}"
            )

        features = payload.get("features")
        if features is None or not isinstance(features, torch.Tensor):
            raise ValueError(
                f"[CachedFeatureDataset] Missing or invalid 'features' tensor in {feature_file}"
            )

        # Ensure float32 dtype
        if features.dtype != torch.float32:
            features = features.to(dtype=torch.float32)

        # Verify shape strictly matches [16, 512]
        expected_shape = (self.expected_seq_len, self.expected_feature_dim)
        if tuple(features.shape) != expected_shape:
            raise ValueError(
                f"[CachedFeatureDataset] Unexpected feature shape {tuple(features.shape)} "
                f"for video '{video_id}' in {feature_file}. Expected {expected_shape}."
            )

        # Verify finite values
        if not torch.isfinite(features).all():
            raise ValueError(
                f"[CachedFeatureDataset] Non-finite values detected in feature tensor for video '{video_id}'."
            )

        # Determine label
        label_val = payload.get("label")
        if label_val is None:
            label_val = int(record["label_id"])
        label_tensor = torch.tensor(int(label_val), dtype=torch.long)

        return features, label_tensor, video_id

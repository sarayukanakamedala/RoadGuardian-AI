"""
RoadGuardian AI - PyTorch Video Dataset
=======================================
PyTorch Dataset for loading video sequences from the authoritative
RoadGuardian AI dataset manifest (dataset/metadata/dataset_manifest.csv).

Key Architecture Guarantees:
- Enforces strict video-level partition (TRAIN, VALIDATION, TEST) with zero leakage.
- Reads only from the authoritative split column in dataset_manifest.csv.
- Resolves all video file paths relative to PROJECT_ROOT.
- Performs deterministic temporal frame sampling (default: uniform temporal sampling).
- Decodes video streams on the fly via OpenCV without loose frame disk overhead.
- Explicitly converts BGR to RGB and applies ImageNet normalization for ResNet-18.
- Returns (frames: FloatTensor [16, 3, 224, 224], label: LongTensor, video_id: str).
"""

import csv
from pathlib import Path
from typing import Callable, Optional, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ml.preprocessing.dataset_config import (
    DATASET_MANIFEST_PATH,
    DEFAULT_FRAME_SIZE,
    DEFAULT_SEQUENCE_LENGTH,
    LABEL_TO_ID,
    PROJECT_ROOT,
)
from ml.preprocessing.transforms import VideoTransform, get_default_video_transform


def compute_frame_indices(
    total_frames: int,
    sequence_length: int = 16,
    sampling_strategy: str = "uniform",
) -> list[int]:
    """
    Computes deterministic frame indices to sample from a video.

    Strategies:
    - 'uniform' (default): Evenly samples sequence_length frames across the entire clip
      using np.linspace(0, total_frames - 1, num=sequence_length, dtype=int).
      Covers full temporal progression (pre-crash, collision event, post-collision halt).
    - 'center': Samples sequence_length consecutive frames centered at clip midpoint.
    - 'start': Samples the initial sequence_length consecutive frames.
    """
    if total_frames < sequence_length:
        raise ValueError(
            f"Video frame count ({total_frames}) is less than requested sequence length ({sequence_length})."
        )

    if sampling_strategy == "uniform":
        return np.linspace(0, total_frames - 1, num=sequence_length, dtype=int).tolist()
    elif sampling_strategy == "center":
        start_idx = (total_frames - sequence_length) // 2
        return list(range(start_idx, start_idx + sequence_length))
    elif sampling_strategy == "start":
        return list(range(sequence_length))
    else:
        raise ValueError(
            f"Unknown sampling_strategy: '{sampling_strategy}'. Choose 'uniform', 'center', or 'start'."
        )


def decode_video_frames(
    video_path: Path,
    frame_indices: list[int],
    video_id: str,
) -> list[np.ndarray]:
    """
    Decodes specific frame indices from a video using OpenCV.
    Explicitly converts BGR to RGB.
    Raises RuntimeError if video cannot be opened or frames cannot be read.
    """
    if not video_path.exists():
        raise FileNotFoundError(
            f"[VideoDataset] Video file not found for video_id '{video_id}': {video_path}"
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(
            f"[VideoDataset] Failed to open video container for video_id '{video_id}': {video_path}"
        )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_set = set(frame_indices)
    max_target = max(frame_indices)

    # Read frames sequentially up to max_target for efficient stream access
    current_idx = 0
    read_frames = {}

    while current_idx <= max_target:
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            raise RuntimeError(
                f"[VideoDataset] Failed reading frame {current_idx} (container total: {total_frames}) "
                f"for video_id '{video_id}': {video_path}"
            )

        if current_idx in target_set:
            # Explicitly convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            read_frames[current_idx] = frame_rgb

        current_idx += 1

    cap.release()

    # Reconstruct frames in the requested order of frame_indices
    frames = []
    for idx in frame_indices:
        if idx not in read_frames:
            raise RuntimeError(
                f"[VideoDataset] Missing frame index {idx} after decoding video_id '{video_id}': {video_path}"
            )
        frames.append(read_frames[idx])

    return frames


class VideoDataset(Dataset):
    """
    PyTorch Dataset for RoadGuardian AI video sequences.

    Loads videos strictly from the authoritative split column in dataset_manifest.csv.
    Decodes on the fly via OpenCV and applies ImageNet normalization for ResNet-18.
    """

    def __init__(
        self,
        split: str,
        manifest_path: Union[str, Path] = DATASET_MANIFEST_PATH,
        project_root: Union[str, Path] = PROJECT_ROOT,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        frame_size: tuple[int, int] = DEFAULT_FRAME_SIZE,
        sampling_strategy: str = "uniform",
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            split: Data split to load: 'TRAIN', 'VALIDATION', or 'TEST'.
            manifest_path: Path to dataset_manifest.csv.
            project_root: Root directory to resolve relative_path from.
            sequence_length: Number of frames per sequence (default: 16).
            frame_size: Spatial target resolution (height, width) (default: 224, 224).
            sampling_strategy: 'uniform' (default), 'center', or 'start'.
            transform: Optional callable transform. If None, uses default VideoTransform.
        """
        self.split = split.upper().strip()
        if self.split not in ("TRAIN", "VALIDATION", "TEST"):
            raise ValueError(
                f"[VideoDataset] Invalid split '{split}'. Expected 'TRAIN', 'VALIDATION', or 'TEST'."
            )

        self.manifest_path = Path(manifest_path).resolve()
        self.project_root = Path(project_root).resolve()
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        self.sampling_strategy = sampling_strategy
        self.transform = transform or get_default_video_transform(target_size=frame_size)
        self.label_to_id = LABEL_TO_ID

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"[VideoDataset] Manifest file not found: {self.manifest_path}"
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
                f"[VideoDataset] Zero video records found for split '{self.split}' "
                f"in manifest: {self.manifest_path}"
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        """
        Retrieves and transforms a video sample.

        Returns:
            frames: FloatTensor of shape [16, 3, 224, 224] (T, C, H, W)
            label: LongTensor scalar (0 for normal, 1 for accident)
            video_id: Unique string identifier for the video clip
        """
        record = self.records[idx]
        video_id = record["video_id"]
        class_label = record["class_label"].strip().lower()
        relative_path = record["relative_path"]

        # Crucial architectural requirement: resolve relative_path from PROJECT_ROOT
        video_full_path = self.project_root / relative_path

        label_id = self.label_to_id.get(class_label)
        if label_id is None:
            raise ValueError(
                f"[VideoDataset] Unknown class_label '{class_label}' for video_id '{video_id}'. "
                f"Supported classes: {list(self.label_to_id.keys())}"
            )

        total_frames = int(record.get("frame_count", 50))
        frame_indices = compute_frame_indices(
            total_frames=total_frames,
            sequence_length=self.sequence_length,
            sampling_strategy=self.sampling_strategy,
        )

        frames_rgb = decode_video_frames(
            video_path=video_full_path,
            frame_indices=frame_indices,
            video_id=video_id,
        )

        frames_tensor = self.transform(frames_rgb)  # FloatTensor [16, 3, 224, 224]
        label_tensor = torch.tensor(label_id, dtype=torch.long)

        return frames_tensor, label_tensor, video_id

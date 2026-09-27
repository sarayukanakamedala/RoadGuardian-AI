#!/usr/bin/env python3
"""
RoadGuardian AI - Preprocessing Smoke Test Utility
==================================================
Tests and verifies VideoDataset and VideoTransform on a minimal sample
without processing the entire 4,500-video dataset.

Verification Steps:
1. Instantiates TRAIN, VALIDATION, and TEST datasets from dataset_manifest.csv.
2. Asserts and logs expected dataset lengths (3150 / 675 / 675).
3. Loads exactly 2 samples from each partition.
4. Asserts tensor shape is exactly [16, 3, 224, 224].
5. Asserts labels are valid torch.long scalar integers (0 or 1).
6. Asserts tensor values are finite (no NaN, no Inf).
7. Inspects and prints statistical range (min, max, mean) for normalized sample.
8. Exits with code 0 on complete success, or code 1 on any assertion failure.
"""

import sys
from pathlib import Path

# Add project root to sys.path so ml package can be imported cleanly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from ml.preprocessing.dataset_config import DATASET_MANIFEST_PATH, LABEL_TO_ID
from ml.preprocessing.video_dataset import VideoDataset


def run_smoke_test():
    print("=" * 70)
    print(" RoadGuardian AI - Video Preprocessing Smoke Test")
    print("=" * 70)
    print(f"Project Root  : {PROJECT_ROOT}")
    print(f"Manifest Path : {DATASET_MANIFEST_PATH}")
    print("-" * 70)

    if not DATASET_MANIFEST_PATH.exists():
        print(f"[FATAL] Authoritative manifest does not exist: {DATASET_MANIFEST_PATH}")
        sys.exit(1)

    expected_lengths = {
        "TRAIN": 3150,
        "VALIDATION": 675,
        "TEST": 675,
    }

    datasets = {}

    # 1. Instantiate datasets and verify lengths
    for split_name, expected_len in expected_lengths.items():
        print(f"\n[1] Initializing {split_name} VideoDataset ...", end=" ", flush=True)
        try:
            ds = VideoDataset(split=split_name, manifest_path=DATASET_MANIFEST_PATH)
            datasets[split_name] = ds
            actual_len = len(ds)
            print(f"OK (Length: {actual_len})")
            assert actual_len == expected_len, (
                f"Split {split_name} length mismatch: expected {expected_len}, got {actual_len}"
            )
        except Exception as e:
            print(f"FAILED!\n[ERROR] {e}")
            sys.exit(1)

    print("\n" + "-" * 70)
    print("[2] Loading 2 sample sequences per partition ...")
    print("-" * 70)

    first_sample_stats = None

    for split_name, ds in datasets.items():
        print(f"\n--- Split: {split_name} ---")
        # Sample first and last element of the split to test diversity
        sample_indices = [0, len(ds) - 1]

        for idx in sample_indices:
            try:
                frames, label, video_id = ds[idx]
            except Exception as e:
                print(f"[FATAL] Exception loading sample {idx} from {split_name}: {e}")
                sys.exit(1)

            # Print basic info
            label_val = label.item()
            label_name = "accident" if label_val == 1 else "normal"
            print(
                f"  Sample index {idx:>4} | video_id: {video_id:<18} | "
                f"label: {label_val} ({label_name}) | shape: {list(frames.shape)} | dtype: {frames.dtype}"
            )

            # Assertions
            assert isinstance(frames, torch.Tensor), "Frames must be a torch.Tensor"
            assert frames.dtype == torch.float32, f"Expected float32, got {frames.dtype}"
            assert list(frames.shape) == [16, 3, 224, 224], (
                f"Expected tensor shape [16, 3, 224, 224], got {list(frames.shape)}"
            )

            assert isinstance(label, torch.Tensor), "Label must be a torch.Tensor"
            assert label.dtype == torch.long, f"Expected long label dtype, got {label.dtype}"
            assert label_val in (0, 1), f"Label value must be 0 or 1, got {label_val}"

            assert isinstance(video_id, str) and len(video_id) > 0, "video_id must be non-empty string"

            # Check finite
            assert torch.isfinite(frames).all(), f"Frames tensor contains NaN or Inf for video {video_id}!"

            if first_sample_stats is None:
                first_sample_stats = {
                    "video_id": video_id,
                    "min": round(float(frames.min()), 4),
                    "max": round(float(frames.max()), 4),
                    "mean": round(float(frames.mean()), 4),
                    "std": round(float(frames.std()), 4),
                }

    # 3. Report sample tensor statistics
    print("\n" + "=" * 70)
    print(f" Sample Tensor Statistics (video_id: {first_sample_stats['video_id']})")
    print("=" * 70)
    print(f"  Min  : {first_sample_stats['min']}")
    print(f"  Max  : {first_sample_stats['max']}")
    print(f"  Mean : {first_sample_stats['mean']}")
    print(f"  Std  : {first_sample_stats['std']}")
    print("  Note : Statistics reflect ImageNet normalization [(x - mean) / std].")
    print("=" * 70)

    print("\n[SUCCESS] Preprocessing smoke test completed with 0 errors!")
    sys.exit(0)


if __name__ == "__main__":
    run_smoke_test()

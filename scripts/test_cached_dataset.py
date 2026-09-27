"""
RoadGuardian AI - Cached Feature Dataset Validation Test
========================================================
Validates the integrity, partition isolation, and tensor specifications of
pre-extracted ResNet-18 feature sequences loaded via CachedFeatureDataset.

Validations:
- Dataset instantiation for TRAIN, VALIDATION, and TEST.
- Length validation (smoke test count or full dataset: 3150 / 675 / 675).
- Tensor shape validation: strictly [16, 512].
- Tensor dtype validation: float32.
- Finite numerical checks: zero NaN, zero Inf.
- Partition isolation: zero video ID leakage across TRAIN / VALIDATION / TEST.
- Compatibility test with GRUClassifier forward pass.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from ml.preprocessing.cached_dataset import CachedFeatureDataset, DEFAULT_FEATURE_MANIFEST_PATH
from ml.resnet_gru.model import GRUClassifier


def test_cached_dataset(manifest_path: Path = DEFAULT_FEATURE_MANIFEST_PATH):
    print("=" * 70)
    print("ROADGUARDIAN AI - CACHED FEATURE DATASET VALIDATION")
    print("=" * 70)
    print(f"Manifest Path: {manifest_path}\n")

    if not manifest_path.exists():
        print(f"[ERROR] Feature manifest not found at: {manifest_path}")
        print("Please run scripts/extract_features.py first.")
        sys.exit(1)

    splits = ["TRAIN", "VALIDATION", "TEST"]
    expected_full_counts = {"TRAIN": 3150, "VALIDATION": 675, "TEST": 675}
    datasets = {}
    split_video_ids = {}

    for split in splits:
        print(f">>> Instantiating CachedFeatureDataset for split: {split} ...")
        try:
            ds = CachedFeatureDataset(split=split, manifest_path=manifest_path)
            datasets[split] = ds
            video_ids = [r["video_id"] for r in ds.records]
            split_video_ids[split] = set(video_ids)

            actual_len = len(ds)
            exp_full = expected_full_counts[split]
            if actual_len == exp_full:
                print(f"  [OK] Full dataset split size confirmed: {actual_len} samples.")
            else:
                print(f"  [INFO] Smoke-test split size: {actual_len} samples (Full expectation: {exp_full}).")

            assert actual_len > 0, f"Zero records loaded for split {split}!"

        except Exception as e:
            print(f"  [FAIL] Failed instantiating dataset for split {split}: {e}")
            raise e

    # 1. Verify Split Isolation (Zero Leakage)
    print("\n--- Verifying Split Isolation ---")
    train_ids = split_video_ids["TRAIN"]
    val_ids = split_video_ids["VALIDATION"]
    test_ids = split_video_ids["TEST"]

    train_val_overlap = train_ids.intersection(val_ids)
    train_test_overlap = train_ids.intersection(test_ids)
    val_test_overlap = val_ids.intersection(test_ids)

    assert len(train_val_overlap) == 0, f"Leakage detected between TRAIN and VAL: {train_val_overlap}"
    assert len(train_test_overlap) == 0, f"Leakage detected between TRAIN and TEST: {train_test_overlap}"
    assert len(val_test_overlap) == 0, f"Leakage detected between VAL and TEST: {val_test_overlap}"
    print("  [OK] Zero video ID overlap between TRAIN, VALIDATION, and TEST.")

    # 2. Verify Sample Inspection
    print("\n--- Verifying Feature Tensors & Data Types ---")
    for split, ds in datasets.items():
        num_to_test = min(len(ds), 3)
        print(f"Testing {num_to_test} sample(s) from {split} split:")
        for idx in range(num_to_test):
            features, label, video_id = ds[idx]

            # Check shapes
            assert features.ndim == 2, f"Expected 2D tensor [16, 512], got {features.ndim}D"
            assert tuple(features.shape) == (16, 512), f"Expected (16, 512), got {tuple(features.shape)}"
            assert features.dtype == torch.float32, f"Expected float32, got {features.dtype}"

            # Check finite
            assert torch.isfinite(features).all(), f"Non-finite values detected in {video_id}"

            # Check label
            assert isinstance(label, torch.Tensor), f"Label is not a Tensor: {type(label)}"
            assert label.item() in (0, 1), f"Unexpected label value: {label.item()}"

            # Check video_id
            assert isinstance(video_id, str) and len(video_id) > 0, f"Invalid video_id: {video_id}"

            print(
                f"  Sample {idx}: video_id='{video_id}', "
                f"shape={list(features.shape)}, dtype={features.dtype}, "
                f"label={label.item()} ({'accident' if label.item() == 1 else 'normal'}), "
                f"finite=True"
            )

    # 3. DataLoader & Temporal Head Compatibility Test
    print("\n--- Verifying DataLoader & Temporal GRU Compatibility ---")
    train_loader = DataLoader(datasets["TRAIN"], batch_size=2, shuffle=False)
    batch_features, batch_labels, batch_ids = next(iter(train_loader))

    B, T, D = batch_features.shape
    print(f"DataLoader batch shape: [B={B}, T={T}, D={D}]")
    assert B == min(2, len(datasets["TRAIN"])), f"Unexpected batch size {B}"
    assert (T, D) == (16, 512), f"Unexpected sequence dimensions ({T}, {D})"

    # Pass through GRUClassifier to verify end-to-end compatibility
    gru_head = GRUClassifier(input_size=512, hidden_size=256, num_layers=1)
    gru_head.eval()
    with torch.no_grad():
        logits = gru_head(batch_features)

    assert tuple(logits.shape) == (B, 1), f"Expected logits shape ({B}, 1), got {tuple(logits.shape)}"
    assert torch.isfinite(logits).all(), "Non-finite logits produced by GRUClassifier"
    print(f"  [OK] GRUClassifier forward pass succeeded on cached features! Logits shape: {list(logits.shape)}")

    print("\n" + "=" * 70)
    print("[SUCCESS] All CachedFeatureDataset checks passed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    test_cached_dataset()

"""
RoadGuardian AI - Feature Cache Comprehensive Audit Tool
========================================================
Performs rigorous audit of cached ResNet-18 spatial features against
the authoritative dataset_manifest.csv.
"""

import csv
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from ml.preprocessing.dataset_config import DATASET_MANIFEST_PATH, FEATURE_MANIFEST_PATH


def audit_cache(
    dataset_manifest_path: Path = DATASET_MANIFEST_PATH,
    feature_manifest_path: Path = FEATURE_MANIFEST_PATH,
    expected_full: bool = False,
) -> dict:
    print("=" * 70)
    print("ROADGUARDIAN AI - FEATURE CACHE AUDIT")
    print("=" * 70)
    print(f"Dataset Manifest: {dataset_manifest_path}")
    print(f"Feature Manifest: {feature_manifest_path}")
    print(f"Mode:             {'FULL AUDIT (4500)' if expected_full else 'INCREMENTAL / SMOKE AUDIT'}\n")

    # 1. Load authoritative dataset manifest
    if not dataset_manifest_path.exists():
        raise FileNotFoundError(f"Missing dataset manifest: {dataset_manifest_path}")

    with open(dataset_manifest_path, mode="r", encoding="utf-8") as f:
        auth_records = list(csv.DictReader(f))

    auth_by_id = {r["video_id"]: r for r in auth_records}
    print(f"[Authoritative] Total videos: {len(auth_records)}")
    auth_split_counts = {}
    for r in auth_records:
        s = r["split"]
        auth_split_counts[s] = auth_split_counts.get(s, 0) + 1
    print(f"[Authoritative] Split distribution: {auth_split_counts}")

    assert len(auth_records) == 4500, f"Expected 4500 videos, got {len(auth_records)}"
    assert auth_split_counts.get("TRAIN") == 3150, f"Expected 3150 TRAIN, got {auth_split_counts.get('TRAIN')}"
    assert auth_split_counts.get("VALIDATION") == 675, f"Expected 675 VALIDATION, got {auth_split_counts.get('VALIDATION')}"
    assert auth_split_counts.get("TEST") == 675, f"Expected 675 TEST, got {auth_split_counts.get('TEST')}"

    # 2. Load feature manifest
    if not feature_manifest_path.exists():
        print(f"[WARNING] Feature manifest does not exist yet at {feature_manifest_path}")
        return {"total_cached": 0}

    with open(feature_manifest_path, mode="r", encoding="utf-8") as f:
        feat_records = list(csv.DictReader(f))

    print(f"[Feature Cache] Total records in manifest: {len(feat_records)}")

    feat_by_id = {}
    duplicates = []
    for r in feat_records:
        vid = r["video_id"]
        if vid in feat_by_id:
            duplicates.append(vid)
        feat_by_id[vid] = r

    assert len(duplicates) == 0, f"Duplicate video IDs found in feature manifest: {duplicates}"

    feat_split_counts = {}
    for r in feat_records:
        s = r["split"]
        feat_split_counts[s] = feat_split_counts.get(s, 0) + 1
    print(f"[Feature Cache] Split distribution: {feat_split_counts}")

    if expected_full:
        assert len(feat_records) == 4500, f"Expected 4500 cached records, found {len(feat_records)}"
        assert feat_split_counts.get("TRAIN") == 3150, f"Expected 3150 TRAIN, got {feat_split_counts.get('TRAIN')}"
        assert feat_split_counts.get("VALIDATION") == 675, f"Expected 675 VALIDATION, got {feat_split_counts.get('VALIDATION')}"
        assert feat_split_counts.get("TEST") == 675, f"Expected 675 TEST, got {feat_split_counts.get('TEST')}"

    # 3. Check Partition Isolation & Metadata Consistency
    train_ids = {r["video_id"] for r in feat_records if r["split"] == "TRAIN"}
    val_ids = {r["video_id"] for r in feat_records if r["split"] == "VALIDATION"}
    test_ids = {r["video_id"] for r in feat_records if r["split"] == "TEST"}

    assert len(train_ids.intersection(val_ids)) == 0, "Leakage between TRAIN and VAL"
    assert len(train_ids.intersection(test_ids)) == 0, "Leakage between TRAIN and TEST"
    assert len(val_ids.intersection(test_ids)) == 0, "Leakage between VAL and TEST"
    print("  [OK] Zero partition leakage across TRAIN, VALIDATION, and TEST.")

    # 4. Audit Physical Files and Tensors
    total_size_bytes = 0
    checked_count = 0

    for r in feat_records:
        vid = r["video_id"]
        rel_path = r["feature_path"]
        target_file = PROJECT_ROOT / rel_path

        assert target_file.exists(), f"Feature file missing for video '{vid}': {target_file}"
        file_size = target_file.stat().st_size
        total_size_bytes += file_size

        # Check against authoritative dataset_manifest.csv
        auth_row = auth_by_id.get(vid)
        assert auth_row is not None, f"Video '{vid}' in feature manifest not found in authoritative dataset_manifest.csv"
        assert r["split"] == auth_row["split"], f"Split mismatch for '{vid}': {r['split']} vs {auth_row['split']}"
        assert r["class_label"] == auth_row["class_label"], f"Class mismatch for '{vid}'"
        expected_label = 1 if auth_row["class_label"] == "accident" else 0
        assert int(r["label_id"]) == expected_label, f"Label ID mismatch for '{vid}'"

        # Check metadata fields
        assert r["sequence_length"] == "16", f"Metadata sequence_length mismatch: {r['sequence_length']}"
        assert r["feature_dim"] == "512", f"Metadata feature_dim mismatch: {r['feature_dim']}"
        assert r["frame_size"] == "224x224", f"Metadata frame_size mismatch: {r['frame_size']}"
        assert r["sampling_strategy"] == "uniform", f"Metadata sampling_strategy mismatch: {r['sampling_strategy']}"
        assert r["backbone"] == "resnet18", f"Metadata backbone mismatch: {r['backbone']}"
        assert "ResNet18_Weights" in r["weights"], f"Metadata weights mismatch: {r['weights']}"

        # Load payload to verify tensors
        payload = torch.load(target_file, map_location="cpu", weights_only=True)
        assert payload["video_id"] == vid, f"Payload video_id mismatch in {target_file}"
        assert payload["split"] == r["split"], f"Payload split mismatch in {target_file}"
        assert payload["label"] == expected_label, f"Payload label mismatch in {target_file}"
        assert tuple(payload["feature_shape"]) == (16, 512), f"Payload shape mismatch in {target_file}"

        features = payload["features"]
        assert isinstance(features, torch.Tensor), f"Features is not a Tensor in {target_file}"
        assert tuple(features.shape) == (16, 512), f"Feature tensor shape is not (16, 512) in {target_file}"
        assert features.dtype == torch.float32, f"Feature tensor dtype is not float32 in {target_file}"
        assert torch.isfinite(features).all(), f"Non-finite values found in {target_file}"

        checked_count += 1
        if checked_count % 500 == 0:
            print(f"  Verified {checked_count}/{len(feat_records)} cached files...", flush=True)

    total_size_mb = total_size_bytes / (1024 * 1024)
    print(f"\n[Audit Summary]")
    print(f"  Verified Files:         {checked_count}/{len(feat_records)}")
    print(f"  Total Cache Size:       {total_size_mb:.2f} MB ({total_size_bytes} bytes)")
    print(f"  Average Size / File:    {total_size_bytes / max(checked_count, 1) / 1024:.2f} KB")
    print(f"  Zero Leakage:           PASSED")
    print(f"  Tensor Specifications:  PASSED ([16, 512], float32, finite)")
    print(f"  Manifest Concordance:   PASSED (100% match with dataset_manifest.csv)")
    print("=" * 70)

    return {
        "total_cached": checked_count,
        "split_counts": feat_split_counts,
        "total_size_mb": total_size_mb,
    }


if __name__ == "__main__":
    is_full = "--full" in sys.argv
    audit_cache(expected_full=is_full)

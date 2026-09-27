"""
RoadGuardian AI - ResNet18 Feature Extraction & Caching Pipeline
================================================================
Extracts 512-dimensional spatial feature vectors from video frames using an
ImageNet-pretrained ResNet-18 backbone and caches the sequence representations
to disk in a structured, partition-isolated hierarchy.

Workflow:
---------
Raw Video (.mp4)
  -> 16 Preprocessed Frames [16, 3, 224, 224] via VideoDataset
  -> Pretrained ResNet-18 Spatial Extractor (eval, no_grad)
  -> Spatial Feature Embeddings [16, 512]
  -> Cached to disk: dataset/processed/features/<split>/<video_id>.pt
  -> Documented in: dataset/processed/features/feature_manifest.csv
"""

import csv
import time
from pathlib import Path
from typing import Optional, Union

import torch
from torch.utils.data import DataLoader

from ml.preprocessing.dataset_config import (
    DATASET_MANIFEST_PATH,
    DEFAULT_FRAME_SIZE,
    DEFAULT_SEQUENCE_LENGTH,
    FEATURES_DIR as DEFAULT_FEATURES_DIR,
    PROJECT_ROOT,
)
from ml.preprocessing.video_dataset import VideoDataset
from ml.resnet_gru.model import ResNetSpatialExtractor


def extract_split_features(
    split: str,
    output_dir: Union[str, Path] = DEFAULT_FEATURES_DIR,
    manifest_path: Union[str, Path] = DATASET_MANIFEST_PATH,
    batch_size: int = 8,
    device: str = "cpu",
    limit: Optional[int] = None,
    overwrite: bool = False,
) -> list[dict]:
    """
    Extracts ResNet-18 spatial features for a specific partition split.

    Args:
        split: 'TRAIN', 'VALIDATION', or 'TEST'.
        output_dir: Base directory for cached features (default: dataset/processed/features).
        manifest_path: Path to dataset_manifest.csv.
        batch_size: Number of video sequences processed simultaneously.
        device: 'cpu' or 'cuda'.
        limit: Optional cap on number of videos to extract (useful for smoke tests).
        overwrite: If True, overwrites existing .pt files; if False, validates and skips.

    Returns:
        List of metadata dicts corresponding to the extracted features.
    """
    split_name = split.upper().strip()
    output_dir = Path(output_dir).resolve()
    split_dir = output_dir / split_name.lower()
    split_dir.mkdir(parents=True, exist_ok=True)

    # 1. Instantiate authoritative VideoDataset for the designated split
    dataset = VideoDataset(
        split=split_name,
        manifest_path=manifest_path,
        project_root=PROJECT_ROOT,
        sequence_length=DEFAULT_SEQUENCE_LENGTH,
        frame_size=DEFAULT_FRAME_SIZE,
        sampling_strategy="uniform",
    )

    if limit is not None and limit > 0:
        dataset.records = dataset.records[:limit]

    total_samples = len(dataset)
    t0 = time.time()

    # Partition dataset into already-cached (valid) vs un-cached
    cached_records = []
    uncached_dataset_records = []

    for rec in dataset.records:
        vid = rec["video_id"]
        target_file = split_dir / f"{vid}.pt"
        if target_file.exists() and not overwrite:
            # Validate existing cache file integrity
            try:
                existing_data = torch.load(target_file, map_location="cpu", weights_only=True)
                assert isinstance(existing_data, dict), f"Expected dict payload, got {type(existing_data)}"
                assert existing_data.get("video_id") == vid, f"video_id mismatch: {existing_data.get('video_id')} vs {vid}"
                assert existing_data.get("split") == split_name, f"split mismatch: {existing_data.get('split')} vs {split_name}"
                assert tuple(existing_data.get("feature_shape", [])) == (16, 512), f"shape mismatch"
                features = existing_data["features"]
                assert tuple(features.shape) == (16, 512) and features.dtype == torch.float32, "features mismatch"
                assert torch.isfinite(features).all(), "non-finite values in cache"

                file_size_kb = round(target_file.stat().st_size / 1024, 2)
                rel_feature_path = target_file.relative_to(PROJECT_ROOT).as_posix()
                label_val = existing_data["label"]
                class_label = existing_data["class_label"]

                cached_records.append({
                    "video_id": vid,
                    "class_label": class_label,
                    "label_id": label_val,
                    "split": split_name,
                    "feature_path": rel_feature_path,
                    "num_frames": 16,
                    "feature_dim": 512,
                    "dtype": "float32",
                    "file_size_kb": file_size_kb,
                    "sampling_strategy": "uniform",
                    "frame_size": "224x224",
                    "sequence_length": 16,
                    "backbone": "resnet18",
                    "weights": "ResNet18_Weights.DEFAULT",
                })
            except Exception as e:
                print(f"[{split_name}] Existing cache file {target_file} invalid ({e}). Queued for re-extraction.")
                uncached_dataset_records.append(rec)
        else:
            uncached_dataset_records.append(rec)

    extracted_records = list(cached_records)
    skipped_count = len(cached_records)
    newly_extracted_count = 0
    failed_count = 0

    if skipped_count > 0:
        print(f"[{split_name}] Validated and retained {skipped_count} pre-existing cache file(s).")

    to_extract_count = len(uncached_dataset_records)
    print(f"[{split_name}] Total: {total_samples} | Retained: {skipped_count} | To Extract: {to_extract_count}")

    if to_extract_count > 0:
        dataset.records = uncached_dataset_records
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # Enforce strictly deterministic order
            num_workers=0,  # Stable sequential worker execution
        )

        # 2. Instantiate pretrained ResNet-18 feature extractor in eval mode
        extractor = ResNetSpatialExtractor(
            pretrained=True,
            freeze_backbone=True,
            feature_dim=512,
        ).to(device)
        extractor.eval()

        with torch.no_grad():
            for batch_idx, (frames_batch, labels_batch, video_ids_batch) in enumerate(loader):
                B, T, C, H, W = frames_batch.shape
                frames_batch = frames_batch.to(device)

                # Extract features via modular spatial backbone: [B, 16, 512]
                features_batch = extractor(frames_batch).cpu()

                for i in range(B):
                    video_id = video_ids_batch[i]
                    label_val = int(labels_batch[i].item())
                    class_label = "accident" if label_val == 1 else "normal"
                    features = features_batch[i]  # [16, 512] FloatTensor

                    # Cache safety: verify metadata and tensor shape
                    assert list(features.shape) == [16, 512], (
                        f"Unexpected feature shape {list(features.shape)} for video_id {video_id}"
                    )
                    assert torch.isfinite(features).all(), (
                        f"Non-finite values detected in extracted features for video_id {video_id}"
                    )

                    file_name = f"{video_id}.pt"
                    target_file = split_dir / file_name

                    payload = {
                        "video_id": video_id,
                        "class_label": class_label,
                        "label": label_val,
                        "split": split_name,
                        "feature_shape": [16, 512],
                        "features": features.to(dtype=torch.float32),
                    }
                    torch.save(payload, target_file)

                    file_size_kb = round(target_file.stat().st_size / 1024, 2)
                    rel_feature_path = target_file.relative_to(PROJECT_ROOT).as_posix()

                    record = {
                        "video_id": video_id,
                        "class_label": class_label,
                        "label_id": label_val,
                        "split": split_name,
                        "feature_path": rel_feature_path,
                        "num_frames": 16,
                        "feature_dim": 512,
                        "dtype": "float32",
                        "file_size_kb": file_size_kb,
                        "sampling_strategy": "uniform",
                        "frame_size": "224x224",
                        "sequence_length": 16,
                        "backbone": "resnet18",
                        "weights": "ResNet18_Weights.DEFAULT",
                    }
                    extracted_records.append(record)
                    newly_extracted_count += 1

                progress_total = skipped_count + newly_extracted_count
                if newly_extracted_count % 100 == 0 or progress_total == total_samples:
                    elapsed = time.time() - t0
                    sec_per_vid = elapsed / max(progress_total, 1)
                    pct = (progress_total / total_samples) * 100
                    print(
                        f"  [{split_name}] Progress: {progress_total}/{total_samples} ({pct:.1f}%) "
                        f"| Newly extracted: {newly_extracted_count} | Pace: {sec_per_vid:.2f}s/vid",
                        flush=True,
                    )

    elapsed_total = time.time() - t0
    stats = {
        "split": split_name,
        "total": total_samples,
        "skipped": skipped_count,
        "newly_extracted": newly_extracted_count,
        "failed": failed_count,
        "elapsed_sec": round(elapsed_total, 2),
    }

    print(
        f"[{split_name}] Finished split in {elapsed_total:.2f}s "
        f"(Retained: {skipped_count}, Newly Extracted: {newly_extracted_count}, Failed: {failed_count})."
    )
    return extracted_records, stats


def update_feature_manifest(
    records: list[dict],
    output_manifest_path: Union[str, Path] = DEFAULT_FEATURES_DIR / "feature_manifest.csv",
):
    """
    Writes or merges extracted feature records into the authoritative feature_manifest.csv.
    """
    output_manifest_path = Path(output_manifest_path).resolve()
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "video_id",
        "class_label",
        "label_id",
        "split",
        "feature_path",
        "num_frames",
        "feature_dim",
        "dtype",
        "file_size_kb",
        "sampling_strategy",
        "frame_size",
        "sequence_length",
        "backbone",
        "weights",
    ]

    # Merge with existing records if present
    existing_records = {}
    if output_manifest_path.exists():
        with open(output_manifest_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing_records[r["video_id"]] = r

    # Update with newly extracted records
    for r in records:
        existing_records[r["video_id"]] = r

    sorted_records = sorted(existing_records.values(), key=lambda r: (r["split"], r["video_id"]))

    with open(output_manifest_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted_records)

    print(f"Feature manifest updated with {len(sorted_records)} entries at: {output_manifest_path}")
    return sorted_records


def verify_feature_manifest(
    manifest_path: Union[str, Path] = DEFAULT_FEATURES_DIR / "feature_manifest.csv",
    project_root: Union[str, Path] = PROJECT_ROOT,
) -> dict:
    """
    Audits the generated feature_manifest.csv and cached feature .pt files.

    Performs comprehensive safety checks:
    - Verifies no duplicate video_ids.
    - Verifies zero overlap between train, validation, and test splits.
    - Verifies all feature files exist on disk.
    - Verifies feature tensor shapes are [16, 512], dtype is float32, and all values are finite.
    - Verifies stored payload metadata matches manifest entries.
    """
    manifest_path = Path(manifest_path).resolve()
    project_root = Path(project_root).resolve()

    if not manifest_path.exists():
        raise FileNotFoundError(f"Feature manifest does not exist: {manifest_path}")

    records = []
    with open(manifest_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(reader)

    if not records:
        raise ValueError(f"Feature manifest is empty: {manifest_path}")

    video_ids = [r["video_id"] for r in records]
    if len(video_ids) != len(set(video_ids)):
        duplicates = [vid for vid in set(video_ids) if video_ids.count(vid) > 1]
        raise ValueError(f"Duplicate video_ids detected in feature manifest: {duplicates}")

    splits = {}
    for r in records:
        splits.setdefault(r["split"].upper(), []).append(r["video_id"])

    # Verify partition isolation across splits
    all_splits = list(splits.keys())
    for i in range(len(all_splits)):
        for j in range(i + 1, len(all_splits)):
            s1, s2 = all_splits[i], all_splits[j]
            overlap = set(splits[s1]).intersection(set(splits[s2]))
            if overlap:
                raise ValueError(f"Partition leakage detected between {s1} and {s2}: {overlap}")

    # Verify each cache file
    for r in records:
        vid = r["video_id"]
        rel_path = r["feature_path"]
        feature_file = project_root / rel_path
        if not feature_file.exists():
            raise FileNotFoundError(f"Feature file missing on disk for video '{vid}': {feature_file}")

        payload = torch.load(feature_file, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid payload type for '{vid}': expected dict, got {type(payload)}")

        if payload.get("video_id") != vid:
            raise ValueError(f"Payload video_id mismatch: manifest '{vid}' vs payload '{payload.get('video_id')}'")

        if payload.get("split") != r["split"]:
            raise ValueError(f"Payload split mismatch for '{vid}': manifest '{r['split']}' vs payload '{payload.get('split')}'")

        if payload.get("label") != int(r["label_id"]):
            raise ValueError(f"Payload label mismatch for '{vid}': manifest '{r['label_id']}' vs payload '{payload.get('label')}'")

        features = payload.get("features")
        if not isinstance(features, torch.Tensor):
            raise ValueError(f"Payload 'features' is not a Tensor for '{vid}'")

        if tuple(features.shape) != (16, 512):
            raise ValueError(f"Payload shape mismatch for '{vid}': expected (16, 512), got {tuple(features.shape)}")

        if features.dtype != torch.float32:
            raise ValueError(f"Payload dtype mismatch for '{vid}': expected float32, got {features.dtype}")

        if not torch.isfinite(features).all():
            raise ValueError(f"Non-finite values detected in features for '{vid}'")

    summary = {
        "total_records": len(records),
        "split_counts": {s: len(vids) for s, vids in splits.items()},
        "zero_leakage": True,
        "all_files_exist": True,
        "all_shapes_valid": True,
    }
    print(f"[Cache Audit OK] Verified {len(records)} cached records across splits: {summary['split_counts']}")
    return summary

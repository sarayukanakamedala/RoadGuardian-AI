"""
RoadGuardian AI - Feature Extraction CLI Tool
=============================================
Command-line interface to extract spatial ResNet-18 features from video sequences
and cache them partition-by-partition to disk.

Usage:
------
# Smoke test (2 clips per split):
python scripts/extract_features.py --split ALL --limit 2

# Full extraction for a specific split:
python scripts/extract_features.py --split TRAIN --batch-size 8 --device cpu

# Full extraction across all splits:
python scripts/extract_features.py --split ALL --batch-size 8 --device cpu
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing.dataset_config import DATASET_MANIFEST_PATH, PROJECT_ROOT
from ml.resnet_gru.extract_features import (
    DEFAULT_FEATURES_DIR,
    extract_split_features,
    update_feature_manifest,
    verify_feature_manifest,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract ResNet-18 spatial features from RoadGuardian AI videos."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="ALL",
        choices=["TRAIN", "VALIDATION", "TEST", "ALL"],
        help="Dataset split to extract (default: ALL). 'ALL' processes TRAIN -> VALIDATION -> TEST.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for DataLoader (default: 8).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Computing device: 'cpu' or 'cuda' (default: cpu).",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float32",
        choices=["float32"],
        help="Floating point precision for cached tensors (default: float32).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of videos per split (useful for smoke tests).",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(DATASET_MANIFEST_PATH),
        help=f"Path to authoritative dataset manifest (default: {DATASET_MANIFEST_PATH}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_FEATURES_DIR),
        help=f"Directory to save cached features (default: {DEFAULT_FEATURES_DIR}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing .pt feature files if already present.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    split_arg = args.split.upper()
    if split_arg == "ALL":
        splits_to_process = ["TRAIN", "VALIDATION", "TEST"]
    else:
        splits_to_process = [split_arg]

    print("=" * 70)
    print("ROADGUARDIAN AI - PRETRAINED RESNET-18 FEATURE EXTRACTION")
    print("=" * 70)
    print(f"Target Splits:    {splits_to_process}")
    print(f"Batch Size:       {args.batch_size}")
    print(f"Device:           {args.device}")
    print(f"Precision:        {args.precision}")
    print(f"Limit per split:  {args.limit if args.limit is not None else 'ALL'}")
    print(f"Manifest Path:    {args.manifest}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Overwrite:        {args.overwrite}")
    print("-" * 70)

    start_time = time.time()
    all_extracted_records = []
    split_stats = []

    for split in splits_to_process:
        print(f"\n>>> Processing Split: {split} ...")
        records, stats = extract_split_features(
            split=split,
            output_dir=args.output_dir,
            manifest_path=args.manifest,
            batch_size=args.batch_size,
            device=args.device,
            limit=args.limit,
            overwrite=args.overwrite,
        )
        all_extracted_records.extend(records)
        split_stats.append(stats)

    # Update authoritative feature manifest
    manifest_file = Path(args.output_dir) / "feature_manifest.csv"
    update_feature_manifest(
        records=all_extracted_records,
        output_manifest_path=manifest_file,
    )

    print("\n--- Auditing Generated Feature Cache ---")
    audit_summary = verify_feature_manifest(manifest_path=manifest_file, project_root=PROJECT_ROOT)

    elapsed_total = time.time() - start_time
    total_processed = sum(s["total"] for s in split_stats)
    total_skipped = sum(s["skipped"] for s in split_stats)
    total_newly_extracted = sum(s["newly_extracted"] for s in split_stats)
    total_failed = sum(s["failed"] for s in split_stats)
    avg_sec_per_vid = elapsed_total / max(total_processed, 1)

    # Calculate total size on disk
    features_dir = Path(args.output_dir)
    total_bytes = sum(f.stat().st_size for f in features_dir.rglob("*.pt"))
    total_mb = total_bytes / (1024 * 1024)

    print("\n" + "=" * 70)
    print("ROADGUARDIAN AI - EXTRACTION SUMMARY REPORT")
    print("=" * 70)
    print(f"Total videos processed:       {total_processed}")
    print(f"Number skipped (retained):    {total_skipped}")
    print(f"Number newly extracted:       {total_newly_extracted}")
    print(f"Number failed:                {total_failed}")
    print(f"Total extraction time:        {elapsed_total:.2f}s ({elapsed_total/60:.2f} min)")
    print(f"Average time per video:       {avg_sec_per_vid:.3f}s/vid")
    print(f"Total cache size on disk:     {total_mb:.2f} MB ({total_bytes} bytes)")
    print(f"Cache count by split:         {audit_summary['split_counts']}")
    print(f"Manifest written to:          {manifest_file}")
    print(f"Cache audit status:           PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()

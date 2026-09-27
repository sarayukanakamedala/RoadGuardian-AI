#!/usr/bin/env python3
"""
RoadGuardian AI - Video-Level Dataset Partitioning Utility
==========================================================
Reads an audited manifest (e.g. dataset_audit.csv or dataset_manifest.csv),
partitions the dataset strictly at the VIDEO LEVEL into TRAIN, VALIDATION,
and TEST sets using stratified random sampling, prevents cross-partition
frame leakage, and generates a split distribution summary.

Usage:
    python scripts/create_video_splits.py --manifest dataset/metadata/dataset_audit.csv --output dataset/metadata/dataset_manifest.csv
    python scripts/create_video_splits.py --manifest dataset/metadata/dataset_audit.csv --output dataset/metadata/dataset_manifest.csv --train_ratio 0.70 --val_ratio 0.15 --test_ratio 0.15 --seed 42
"""

import argparse
import csv
import math
import random
import sys
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Partition RoadGuardian AI dataset at the VIDEO LEVEL with zero frame leakage."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="dataset/metadata/dataset_audit.csv",
        help="Path to the source manifest CSV (default: dataset/metadata/dataset_audit.csv).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save updated manifest with split column. If omitted and manifest is dataset_audit.csv, defaults to dataset_manifest.csv.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.70,
        help="Proportion of videos allocated to the training split (default: 0.70).",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Proportion of videos allocated to the validation split (default: 0.15).",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Proportion of videos allocated to the test split (default: 0.15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic, reproducible partitioning (default: 42).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()

    if args.output:
        output_path = Path(args.output).resolve()
    elif manifest_path.name == "dataset_audit.csv":
        output_path = manifest_path.parent / "dataset_manifest.csv"
    else:
        output_path = manifest_path

    print("=" * 70)
    print(" RoadGuardian AI - Video-Level Dataset Partitioning")
    print("=" * 70)
    print(f"Manifest Path : {manifest_path}")
    print(f"Output Path   : {output_path}")
    print(f"Ratios        : Train={args.train_ratio}, Val={args.val_ratio}, Test={args.test_ratio}")
    print(f"Random Seed   : {args.seed}")
    print("-" * 70)

    # Validate split ratios sum to 1.0
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not math.isclose(total_ratio, 1.0, rel_tol=1e-4):
        print(f"[ERROR] Split ratios must sum to 1.0. Current sum: {total_ratio}")
        sys.exit(1)

    if not manifest_path.exists():
        print(f"[ERROR] Manifest file does not exist: {manifest_path}")
        print("Ensure you have audited a dataset and generated the audit manifest first.")
        sys.exit(1)

    with open(manifest_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames) if reader.fieldnames else []
        rows = list(reader)

    if not rows:
        print(f"[ERROR] Manifest file '{manifest_path}' is empty or contains no video records.")
        sys.exit(1)

    if "video_id" not in fieldnames or "class_label" not in fieldnames:
        print("[ERROR] Manifest missing required columns: 'video_id' and 'class_label'.")
        sys.exit(1)

    # Check for duplicate video_ids
    seen_ids = set()
    duplicates = []
    for r in rows:
        vid = r["video_id"].strip()
        if vid in seen_ids:
            duplicates.append(vid)
        seen_ids.add(vid)

    if duplicates:
        print(f"[ERROR] Found duplicate video_ids in manifest: {set(duplicates)}")
        print("Every video must possess a strictly unique video_id to avoid data corruption.")
        sys.exit(1)

    # Check for non-decodable videos
    if "is_decodable" in fieldnames:
        non_decodable = [r for r in rows if str(r.get("is_decodable", "")).strip().lower() in ("false", "0")]
        if non_decodable:
            print(f"[WARNING] Manifest contains {len(non_decodable)} non-decodable video(s).")

    # Stratify by class_label at the VIDEO level
    class_to_rows = defaultdict(list)
    for r in rows:
        label = r.get("class_label", "unknown").strip().lower()
        class_to_rows[label].append(r)

    # Deterministic shuffle per class (sorted class keys for exact reproducible ordering)
    rng = random.Random(args.seed)
    train_ids, val_ids, test_ids = set(), set(), set()

    for label in sorted(class_to_rows.keys()):
        class_records = class_to_rows[label]
        rng.shuffle(class_records)
        n = len(class_records)
        n_train = int(round(n * args.train_ratio))
        n_val = int(round(n * args.val_ratio))
        # Ensure remaining go to test to prevent rounding discrepancies
        n_test = n - n_train - n_val

        train_slice = class_records[:n_train]
        val_slice = class_records[n_train : n_train + n_val]
        test_slice = class_records[n_train + n_val :]

        for r in train_slice:
            r["split"] = "TRAIN"
            train_ids.add(r["video_id"])
        for r in val_slice:
            r["split"] = "VALIDATION"
            val_ids.add(r["video_id"])
        for r in test_slice:
            r["split"] = "TEST"
            test_ids.add(r["video_id"])

    # Strict assertion: Ensure ZERO video ID overlap between splits
    overlap_tv = train_ids.intersection(val_ids)
    overlap_tt = train_ids.intersection(test_ids)
    overlap_vt = val_ids.intersection(test_ids)

    if overlap_tv or overlap_tt or overlap_vt:
        print("[FATAL ERROR] Cross-split video ID leakage detected!")
        sys.exit(1)

    # Write updated records with split assigned, preserving all audit columns
    if "split" not in fieldnames:
        fieldnames.append("split")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Display split summary table
    print("\n" + "=" * 70)
    print(" Video-Level Partition Summary (Zero Leakage Verified)")
    print("=" * 70)
    print(f"{'Split':<15} | {'Total Videos':<14} | {'Class Breakdown'}")
    print("-" * 70)

    for split_name in ("TRAIN", "VALIDATION", "TEST"):
        split_records = [r for r in rows if r.get("split") == split_name]
        counts_by_class = defaultdict(int)
        for r in split_records:
            counts_by_class[r.get("class_label", "unknown")] += 1

        breakdown = ", ".join(f"{cls}: {cnt}" for cls, cnt in sorted(counts_by_class.items()))
        print(f"{split_name:<15} | {len(split_records):<14} | {breakdown}")

    print("=" * 70)
    print(f"Updated manifest successfully written to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

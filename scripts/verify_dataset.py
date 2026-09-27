#!/usr/bin/env python3
"""
RoadGuardian AI - Dataset Verification & Audit Utility
======================================================
Scans a target dataset directory, validates video decodability via OpenCV,
extracts structural stream metadata (resolution, frame count, FPS, duration),
infers class labels and stable video IDs, and exports an audit manifest
without modifying or deleting any source files.

Usage:
    python scripts/verify_dataset.py --dataset_dir dataset/raw
    python scripts/verify_dataset.py --dataset_dir dataset/raw --output_csv dataset/metadata/dataset_audit.csv
"""

import argparse
import concurrent.futures
import csv
import os
import sys
from pathlib import Path

# Supported video extensions
DEFAULT_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit and verify video decodability and physical properties for RoadGuardian AI."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset/raw",
        help="Path to the directory containing raw video files to audit (default: dataset/raw).",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="dataset/metadata/dataset_audit.csv",
        help="Path to the output audit CSV file (default: dataset/metadata/dataset_audit.csv).",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=DEFAULT_EXTENSIONS,
        help="List of video file extensions to scan for.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent worker threads for video decodability auditing (default: 8).",
    )
    return parser.parse_args()


def infer_class_label(video_path: Path, dataset_dir: Path) -> str:
    """
    Infers class label ('accident' or 'normal') based on video location within dataset/raw/<class>/.
    """
    try:
        rel_parts = [p.lower() for p in video_path.relative_to(dataset_dir).parts]
    except ValueError:
        rel_parts = [p.lower() for p in video_path.parts]

    for part in rel_parts:
        if part in ("accident", "accidents", "crash"):
            return "accident"
        if part in ("normal", "non_accident", "non-accident"):
            return "normal"

    # Fallback to immediate parent directory name
    parent_name = video_path.parent.name.lower()
    if parent_name in ("accident", "normal"):
        return parent_name

    return "unknown"


def compute_relative_path(video_path: Path, dataset_dir: Path) -> str:
    """
    Computes a clean, POSIX-style relative path to the video file from the project root or CWD.
    """
    try:
        return video_path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        pass

    try:
        project_root = Path(__file__).resolve().parents[1]
        return video_path.relative_to(project_root).as_posix()
    except ValueError:
        pass

    try:
        return video_path.relative_to(dataset_dir).as_posix()
    except ValueError:
        return video_path.name


def generate_video_id(class_label: str, video_path: Path) -> str:
    """
    Generates a deterministic, stable, collision-free video ID.
    Format: <class_label>_<stem> (e.g. accident_000001, normal_000001).
    """
    stem = video_path.stem
    if stem.lower().startswith(f"{class_label.lower()}_"):
        return stem
    return f"{class_label.lower()}_{stem}"


def inspect_video(video_path: Path, dataset_dir: Path):
    """
    Attempts to decode a video file using OpenCV and extracts structural stream metadata.
    Does NOT modify or delete the file in any way.
    """
    file_size_bytes = video_path.stat().st_size
    file_size_mb = round(file_size_bytes / (1024 * 1024), 3)

    class_label = infer_class_label(video_path, dataset_dir)
    video_id = generate_video_id(class_label, video_path)
    rel_path_str = compute_relative_path(video_path, dataset_dir)

    record = {
        "video_id": video_id,
        "filename": video_path.name,
        "relative_path": rel_path_str,
        "class_label": class_label,
        "file_size_mb": file_size_mb,
        "is_decodable": False,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "frame_count": 0,
        "duration_seconds": 0.0,
        "notes": "",
    }

    try:
        import cv2
    except ImportError:
        record["notes"] = "OpenCV (cv2) is not installed in the current environment."
        return record

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            record["notes"] = "Failed to open video container (corrupt or unsupported codec)."
            return record

        # Read container properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Test reading the first frame to confirm decodability
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None or width <= 0 or height <= 0:
            record["notes"] = "Opened container but failed to read initial frame."
            return record

        fps_val = round(float(fps), 2) if fps and fps > 0 else 0.0
        duration_val = round(frame_count / fps_val, 2) if fps_val > 0 else 0.0

        record["is_decodable"] = True
        record["width"] = width
        record["height"] = height
        record["fps"] = fps_val
        record["frame_count"] = frame_count
        record["duration_seconds"] = duration_val
        record["notes"] = "Valid video stream."

    except Exception as e:
        record["notes"] = f"Decoding exception: {str(e)}"

    return record


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    output_csv = Path(args.output_csv).resolve()

    print("=" * 70)
    print(" RoadGuardian AI - Dataset Verification & Audit")
    print("=" * 70)
    print(f"Target Directory : {dataset_dir}")
    print(f"Output Audit CSV : {output_csv}")
    print(f"Extensions       : {', '.join(args.extensions)}")
    print(f"Worker Threads   : {args.workers}")
    print("-" * 70)

    if not dataset_dir.exists():
        print(f"[ERROR] Specified dataset directory does not exist: {dataset_dir}")
        print("Please check the path or acquire a candidate dataset first.")
        sys.exit(1)

    # Collect candidate video files recursively and sort deterministically
    normalized_exts = tuple(ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in args.extensions)
    video_files = sorted([p for p in dataset_dir.rglob("*") if p.is_file() and p.suffix.lower() in normalized_exts])

    if not video_files:
        print(f"[INFO] No video files found in '{dataset_dir}' matching extensions: {args.extensions}")
        print("Audit terminated. (No files modified)")
        return

    print(f"[INFO] Found {len(video_files)} candidate video files. Commencing decodability audit...\n")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    records = []
    decodable_count = 0
    corrupt_count = 0
    total = len(video_files)

    fieldnames = [
        "video_id",
        "filename",
        "relative_path",
        "class_label",
        "file_size_mb",
        "is_decodable",
        "width",
        "height",
        "fps",
        "frame_count",
        "duration_seconds",
        "notes",
    ]

    if args.workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_video = {executor.submit(inspect_video, vp, dataset_dir): vp for vp in video_files}
            completed = 0
            for future in concurrent.futures.as_completed(future_to_video):
                record = future.result()
                records.append(record)
                completed += 1
                if record["is_decodable"]:
                    decodable_count += 1
                else:
                    corrupt_count += 1
                    print(f"\n[CORRUPT] {record['relative_path']}: {record['notes']}")

                if completed % 250 == 0 or completed == total:
                    print(f"[{completed}/{total}] Audited ({decodable_count} valid, {corrupt_count} corrupt)...", flush=True)

        # Sort records deterministically by video_id
        records.sort(key=lambda r: r["video_id"])
    else:
        for idx, video_path in enumerate(video_files, start=1):
            record = inspect_video(video_path, dataset_dir)
            records.append(record)
            if record["is_decodable"]:
                decodable_count += 1
            else:
                corrupt_count += 1
                print(f"\n[CORRUPT] {record['relative_path']}: {record['notes']}")

            if idx % 250 == 0 or idx == total:
                print(f"[{idx}/{total}] Audited ({decodable_count} valid, {corrupt_count} corrupt)...", flush=True)

    # Write audit records to CSV
    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print("\n" + "=" * 70)
    print(" Audit Summary")
    print("=" * 70)
    print(f"Total Videos Scanned : {len(records)}")
    print(f"Decodable Videos     : {decodable_count}")
    print(f"Corrupted / Failed   : {corrupt_count}")
    print(f"Audit Manifest Saved : {output_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()

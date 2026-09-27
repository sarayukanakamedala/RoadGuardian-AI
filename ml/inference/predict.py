"""
RoadGuardian AI - Single Video Inference
========================================
Runs accident/normal prediction on an arbitrary .mp4 video.

Pipeline:
Raw Video
    -> 16 uniformly sampled frames
    -> VideoTransform [16, 3, 224, 224]
    -> pretrained ResNet-18
    -> [16, 512] spatial features
    -> trained GRU
    -> accident probability
    -> prediction
"""

import argparse
from pathlib import Path

import cv2
import torch

from ml.preprocessing.video_dataset import compute_frame_indices, decode_video_frames
from ml.preprocessing.transforms import get_default_video_transform
from ml.resnet_gru.model import ResNetSpatialExtractor, GRUClassifier


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT = (
    PROJECT_ROOT
    / "ml"
    / "artifacts"
    / "checkpoints"
    / "best_gru.pt"
)

SEQUENCE_LENGTH = 16
FRAME_SIZE = (224, 224)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():

    print("Loading ResNet-18 + GRU models...")

    # Same ResNet configuration used during feature extraction
    extractor = ResNetSpatialExtractor(
        pretrained=True,
        freeze_backbone=True,
        feature_dim=512,
    ).to(DEVICE)

    extractor.eval()

    # Same GRU configuration used during training
    classifier = GRUClassifier(
        input_size=512,
        hidden_size=256,
        num_layers=1,
        dropout=0.0,
        bidirectional=False,
    ).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True,
    )

    classifier.load_state_dict(
        checkpoint["model_state_dict"]
    )

    classifier.eval()

    return extractor, classifier


# ============================================================
# VIDEO INFERENCE
# ============================================================

def predict_video(video_path: Path):

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video file not found: {video_path}"
        )

    # --------------------------------------------------------
    # Open video and determine frame count
    # --------------------------------------------------------

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()

    if total_frames < SEQUENCE_LENGTH:
        raise ValueError(
            f"Video contains only {total_frames} frames. "
            f"At least {SEQUENCE_LENGTH} frames are required."
        )

    print(f"Video          : {video_path}")
    print(f"Total frames   : {total_frames}")
    print(f"FPS            : {fps:.2f}")

    # --------------------------------------------------------
    # Uniform temporal sampling
    # --------------------------------------------------------

    frame_indices = compute_frame_indices(
        total_frames=total_frames,
        sequence_length=SEQUENCE_LENGTH,
        sampling_strategy="uniform",
    )

    print(f"Sampled frames : {frame_indices}")

    # --------------------------------------------------------
    # Decode RGB frames
    # --------------------------------------------------------

    frames_rgb = decode_video_frames(
        video_path=video_path,
        frame_indices=frame_indices,
        video_id=video_path.stem,
    )

    # --------------------------------------------------------
    # Same transformation used during feature extraction
    # --------------------------------------------------------

    transform = get_default_video_transform(
        target_size=FRAME_SIZE
    )

    frames = transform(frames_rgb)

    # [16, 3, 224, 224]
    frames = frames.unsqueeze(0).to(DEVICE)

    print(f"Frame tensor   : {tuple(frames.shape)}")

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    extractor, classifier = load_models()

    # --------------------------------------------------------
    # ResNet -> GRU
    # --------------------------------------------------------

    with torch.no_grad():

        # [1, 16, 512]
        features = extractor(frames)

        # [1, 1]
        logits = classifier(features)

        # Convert logit to probability
        probability = torch.sigmoid(
            logits
        ).item()

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    accident_probability = probability
    normal_probability = 1.0 - probability

    if accident_probability >= 0.5:
        prediction = "ACCIDENT"
    else:
        prediction = "NORMAL"

    return {
        "video": str(video_path),
        "prediction": prediction,
        "accident_probability": accident_probability,
        "normal_probability": normal_probability,
        "threshold": 0.5,
        "device": str(DEVICE),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="RoadGuardian AI - Accident Detection"
    )

    parser.add_argument(
        "video",
        type=str,
        help="Path to input .mp4 video",
    )

    args = parser.parse_args()

    video_path = Path(args.video).expanduser().resolve()

    print("=" * 70)
    print("ROADGUARDIAN AI - VIDEO INFERENCE")
    print("=" * 70)

    print(f"Project Root : {PROJECT_ROOT}")
    print(f"Checkpoint   : {CHECKPOINT}")
    print(f"Device       : {DEVICE}")
    print()

    result = predict_video(video_path)

    print()
    print("-" * 70)
    print("PREDICTION")
    print("-" * 70)

    print(f"Result              : {result['prediction']}")
    print(
        f"Accident probability: "
        f"{result['accident_probability']:.4f} "
        f"({result['accident_probability'] * 100:.2f}%)"
    )
    print(
        f"Normal probability  : "
        f"{result['normal_probability']:.4f} "
        f"({result['normal_probability'] * 100:.2f}%)"
    )
    print(f"Decision threshold  : {result['threshold']:.2f}")

    print()
    print("=" * 70)
    print("INFERENCE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()


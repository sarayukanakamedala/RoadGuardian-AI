import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ml.resnet_gru.model import GRUClassifier
from ml.preprocessing.cached_dataset import CachedFeatureDataset


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_MANIFEST = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "features"
    / "feature_manifest.csv"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "ml"
    / "artifacts"
    / "checkpoints"
    / "best_gru.pt"
)

OUTPUT_DIR = PROJECT_ROOT / "ml" / "artifacts" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(labels, predictions):

    labels = torch.tensor(labels)
    predictions = torch.tensor(predictions)

    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    f1 = (
        2 * precision * recall / max(precision + recall, 1e-12)
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ROADGUARDIAN AI - TEST SET EVALUATION")
    print("=" * 70)

    print(f"Project Root : {PROJECT_ROOT}")
    print(f"Feature Manifest : {FEATURE_MANIFEST}")
    print(f"Checkpoint : {CHECKPOINT}")
    print(f"Device : {DEVICE}")

    # --------------------------------------------------------
    # Load TEST dataset
    # --------------------------------------------------------

    test_dataset = CachedFeatureDataset(
        manifest_path=FEATURE_MANIFEST,
        split="TEST",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(f"Test samples : {len(test_dataset)}")

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = GRUClassifier(
        input_size=512,
        hidden_size=256,
        num_layers=1,
        dropout=0.0,
        bidirectional=False,
    ).to(DEVICE)

    # --------------------------------------------------------
    # Load best checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Checkpoint loaded successfully.")

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    criterion = torch.nn.BCEWithLogitsLoss()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for features, labels, video_ids in test_loader:

            features = features.to(DEVICE)
            labels = labels.float().to(DEVICE)

            logits = model(features).squeeze(1)

            loss = criterion(logits, labels)

            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).long()

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_labels.extend(labels.cpu().long().tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_probabilities.extend(probabilities.cpu().tolist())

    test_loss = total_loss / total_samples

    metrics = calculate_metrics(
        all_labels,
        all_predictions,
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("TEST RESULTS")
    print("-" * 70)

    print(f"Test Loss  : {test_loss:.4f}")
    print(f"Accuracy   : {metrics['accuracy']:.4f}")
    print(f"Precision  : {metrics['precision']:.4f}")
    print(f"Recall     : {metrics['recall']:.4f}")
    print(f"F1 Score   : {metrics['f1']:.4f}")

    print()
    print("CONFUSION MATRIX")
    print("-" * 70)

    print(f"True Negatives  : {metrics['true_negative']}")
    print(f"False Positives  : {metrics['false_positive']}")
    print(f"False Negatives  : {metrics['false_negative']}")
    print(f"True Positives   : {metrics['true_positive']}")

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results = {
        "split": "TEST",
        "num_samples": len(test_dataset),
        "test_loss": test_loss,
        **metrics,
        "checkpoint": str(CHECKPOINT),
        "device": str(DEVICE),
    }

    output_file = OUTPUT_DIR / "test_results.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print()
    print(f"Results saved to:")
    print(output_file)

    print()
    print("=" * 70)
    print("TEST EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()


"""
RoadGuardian AI - GRU Training Pipeline
=======================================
Trains the temporal GRU classifier using pre-extracted ResNet-18
feature sequences from CachedFeatureDataset.

Input:
    [B, 16, 512]

Output:
    Binary accident/normal classification.
"""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.preprocessing.cached_dataset import CachedFeatureDataset
from ml.resnet_gru.model import GRUClassifier


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_MANIFEST = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "features"
    / "feature_manifest.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "ml" / "artifacts"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
HISTORY_DIR = OUTPUT_DIR / "history"

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_gru.pt"
HISTORY_PATH = HISTORY_DIR / "training_history.json"

BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(labels, predictions):
    """Calculate binary classification metrics."""

    labels = torch.tensor(labels, dtype=torch.long)
    predictions = torch.tensor(predictions, dtype=torch.long)

    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())

    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# ============================================================
# One epoch
# ============================================================

def run_epoch(model, loader, criterion, optimizer=None):
    """Run one training or validation epoch."""

    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_labels = []
    all_predictions = []

    for features, labels, _video_ids in loader:

        features = features.to(DEVICE)
        labels = labels.to(DEVICE)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(features).squeeze(1)

            loss = criterion(
                logits,
                labels.float()
            )

            if training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                optimizer.step()

        total_loss += loss.item() * features.size(0)

        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).long()

        all_labels.extend(labels.detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())

    average_loss = total_loss / len(loader.dataset)

    metrics = calculate_metrics(
        all_labels,
        all_predictions
    )

    metrics["loss"] = average_loss

    return metrics


# ============================================================
# Main training
# ============================================================

def main():

    print("=" * 70)
    print("ROADGUARDIAN AI - GRU TRAINING")
    print("=" * 70)

    print(f"Project Root : {PROJECT_ROOT}")
    print(f"Feature Manifest : {FEATURE_MANIFEST}")
    print(f"Device : {DEVICE}")
    print(f"Batch Size : {BATCH_SIZE}")
    print(f"Epochs : {EPOCHS}")
    print(f"Learning Rate : {LEARNING_RATE}")
    print("-" * 70)

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    print("\nLoading datasets...")

    train_dataset = CachedFeatureDataset(
        split="TRAIN",
        manifest_path=FEATURE_MANIFEST,
        project_root=PROJECT_ROOT,
    )

    val_dataset = CachedFeatureDataset(
        split="VALIDATION",
        manifest_path=FEATURE_MANIFEST,
        project_root=PROJECT_ROOT,
    )

    print(f"TRAIN samples      : {len(train_dataset)}")
    print(f"VALIDATION samples : {len(val_dataset)}")

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # --------------------------------------------------------
    # Class imbalance handling
    # --------------------------------------------------------

    train_labels = [
        int(record["label_id"])
        for record in train_dataset.records
    ]

    positive_count = sum(train_labels)
    negative_count = len(train_labels) - positive_count

    pos_weight_value = negative_count / max(positive_count, 1)

    print("\nClass distribution:")
    print(f"Normal   : {negative_count}")
    print(f"Accident : {positive_count}")
    print(f"Positive weight : {pos_weight_value:.4f}")

    pos_weight = torch.tensor(
        [pos_weight_value],
        dtype=torch.float32,
        device=DEVICE,
    )

    # --------------------------------------------------------
    # GRU model
    # --------------------------------------------------------

    model = GRUClassifier(
        input_size=512,
        hidden_size=256,
        num_layers=1,
        dropout=0.0,
        bidirectional=False,
    ).to(DEVICE)

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"\nTrainable parameters : {trainable_parameters:,}")

    # --------------------------------------------------------
    # Loss + optimizer
    # --------------------------------------------------------

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    history = []

    best_val_f1 = -1.0
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    training_start = time.time()

    for epoch in range(1, EPOCHS + 1):

        epoch_start = time.time()

        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer=None,
        )

        scheduler.step(val_metrics["loss"])

        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
            "learning_rate": current_lr,
            "time_seconds": epoch_time,
        }

        history.append(epoch_record)

        print(
            f"\nEpoch {epoch:02d}/{EPOCHS}"
        )

        print(
            f"  Train | "
            f"Loss: {train_metrics['loss']:.4f} | "
            f"Acc: {train_metrics['accuracy']:.4f} | "
            f"F1: {train_metrics['f1']:.4f}"
        )

        print(
            f"  Val   | "
            f"Loss: {val_metrics['loss']:.4f} | "
            f"Acc: {val_metrics['accuracy']:.4f} | "
            f"Precision: {val_metrics['precision']:.4f} | "
            f"Recall: {val_metrics['recall']:.4f} | "
            f"F1: {val_metrics['f1']:.4f}"
        )

        print(
            f"  LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # ----------------------------------------------------
        # Save best model based on validation F1
        # ----------------------------------------------------

        improved = (
            val_metrics["f1"] > best_val_f1
            or (
                val_metrics["f1"] == best_val_f1
                and val_metrics["loss"] < best_val_loss
            )
        )

        if improved:

            best_val_f1 = val_metrics["f1"]
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "input_size": 512,
                    "hidden_size": 256,
                    "num_layers": 1,
                    "dropout": 0.0,
                    "bidirectional": False,
                },
                "epoch": epoch,
                "validation_metrics": val_metrics,
                "train_metrics": train_metrics,
            }

            torch.save(
                checkpoint,
                BEST_MODEL_PATH,
            )

            print(
                f"  [CHECKPOINT] Best model saved → {BEST_MODEL_PATH}"
            )

        else:
            epochs_without_improvement += 1

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if epochs_without_improvement >= PATIENCE:

            print(
                f"\nEarly stopping triggered after "
                f"{epoch} epochs."
            )

            break

    total_training_time = time.time() - training_start

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_payload = {
        "device": str(DEVICE),
        "epochs_requested": EPOCHS,
        "epochs_completed": len(history),
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "best_validation_f1": best_val_f1,
        "best_validation_loss": best_val_loss,
        "total_training_seconds": total_training_time,
        "history": history,
    }

    with open(
        HISTORY_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            history_payload,
            f,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(f"Best Validation F1 : {best_val_f1:.4f}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Training Time       : {total_training_time:.2f}s")
    print(f"Best Model          : {BEST_MODEL_PATH}")
    print(f"Training History    : {HISTORY_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()


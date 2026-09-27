#!/usr/bin/env python3
"""
RoadGuardian AI - ResNet18 + GRU Architecture Smoke Test
=========================================================
Performs an offline architectural validation of the ResNetGRU model using
purely synthetic tensors.

Verification Criteria:
- Model instantiation without downloading weights (pretrained=False).
- ResNet-18 spatial feature dimension assertion (512-dim).
- GRU configuration verification (input_size=512, hidden_size=256, num_layers=1).
- Output shape validation [B, 1] on synthetic input [2, 16, 3, 224, 224].
- Finite floating-point output verification (no NaN, no Inf).
- End-to-end backpropagation gradient generation test.
- Zero reliance on dataset files, zero real video loading, zero checkpoint saving.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from ml.resnet_gru.model import ResNetGRU


def run_architecture_test():
    print("=" * 70)
    print(" RoadGuardian AI - ResNet18 + GRU Architecture Verification Test")
    print("=" * 70)

    # A. Instantiate model with pretrained=False (offline, no downloads)
    print("\n[A] Instantiating ResNetGRU(pretrained=False) ...", end=" ", flush=True)
    model = ResNetGRU(
        pretrained=False,
        freeze_backbone=False,
        hidden_size=256,
        num_layers=1,
        dropout=0.0,
    )
    print("OK")

    # E. Verify backbone feature dimension is 512
    print("[E] Verifying backbone feature dimension ...", end=" ", flush=True)
    assert model.feature_dim == 512, f"Expected feature_dim=512, got {model.feature_dim}"
    print(f"OK (feature_dim: {model.feature_dim})")

    # F. Verify GRU configuration
    print("[F] Verifying GRU configuration ...", end=" ", flush=True)
    assert model.gru.input_size == 512, f"Expected GRU input_size=512, got {model.gru.input_size}"
    assert model.gru.hidden_size == 256, f"Expected GRU hidden_size=256, got {model.gru.hidden_size}"
    assert model.gru.num_layers == 1, f"Expected GRU num_layers=1, got {model.gru.num_layers}"
    assert model.gru.batch_first is True, "Expected GRU batch_first=True"
    print(
        f"OK (input_size={model.gru.input_size}, hidden_size={model.gru.hidden_size}, num_layers={model.gru.num_layers})"
    )

    # B. Create synthetic input tensor
    print("\n[B] Generating synthetic video tensor [B=2, T=16, C=3, H=224, W=224] ...", end=" ", flush=True)
    x = torch.randn(2, 16, 3, 224, 224, dtype=torch.float32)
    print(f"OK (shape: {list(x.shape)})")

    # Intermediate check: verify spatial feature extraction produces [2, 16, 512]
    print("[*] Testing spatial feature extraction ...", end=" ", flush=True)
    with torch.no_grad():
        features = model.extract_spatial_features(x)
    assert list(features.shape) == [2, 16, 512], f"Expected spatial shape [2, 16, 512], got {list(features.shape)}"
    print(f"OK (features shape: {list(features.shape)})")

    # C. Run end-to-end forward pass
    print("\n[C] Executing end-to-end forward pass ...", end=" ", flush=True)
    logits = model(x)
    print("OK")

    # D. Verify logits properties
    print("[D] Verifying output logits properties ...", end=" ", flush=True)
    assert logits.shape == (2, 1), f"Expected logits shape (2, 1), got {logits.shape}"
    assert logits.is_floating_point(), f"Expected floating point logits, got {logits.dtype}"
    assert torch.isfinite(logits).all(), "Logits contain non-finite values (NaN or Inf)!"
    print(f"OK (shape: {list(logits.shape)}, dtype: {logits.dtype}, finite: True)")

    # G. Backward smoke test using synthetic data
    print("\n[G] Running backward pass smoke test on synthetic loss ...", end=" ", flush=True)
    model.zero_grad()
    loss = logits.mean()
    loss.backward()

    # Verify gradients are produced for both classifier and backbone
    assert model.classifier.weight.grad is not None, "Classifier weight gradient is None!"
    assert torch.isfinite(model.classifier.weight.grad).all(), "Classifier weight gradient is non-finite!"

    assert model.gru.weight_ih_l0.grad is not None, "GRU weight_ih_l0 gradient is None!"
    assert torch.isfinite(model.gru.weight_ih_l0.grad).all(), "GRU weight_ih_l0 gradient is non-finite!"

    backbone_has_grad = False
    for p in model.backbone.parameters():
        if p.grad is not None and torch.isfinite(p.grad).all():
            backbone_has_grad = True
            break
    assert backbone_has_grad, "No valid gradients found in ResNet-18 backbone parameters!"
    print("OK (Gradients verified on backbone, GRU, and classifier)")

    # H. Summary report
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 70)
    print(" Model Architecture Summary")
    print("=" * 70)
    print(f"Model Class           : {type(model).__name__}")
    print(f"Total Parameters      : {total_params:,}")
    print(f"Trainable Parameters  : {trainable_params:,}")
    print(f"Input Shape           : {list(x.shape)} [B, T, C, H, W]")
    print(f"Backbone Feature Shape: {list(features.shape)} [B, T, 512]")
    print(
        f"GRU Configuration     : input_size={model.gru.input_size}, hidden_size={model.gru.hidden_size}, "
        f"num_layers={model.gru.num_layers}, batch_first={model.gru.batch_first}"
    )
    print(f"Output Shape          : {list(logits.shape)} [B, 1]")
    print(f"Finite Output Status  : PASSED (No NaN / Inf)")
    print(f"Backward Pass Status  : PASSED (Gradients successfully generated)")
    print("=" * 70)

    print("\n[SUCCESS] ResNet18 + GRU architecture test passed with 0 errors!")
    sys.exit(0)


if __name__ == "__main__":
    run_architecture_test()

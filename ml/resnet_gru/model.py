"""
RoadGuardian AI - ResNet18 + GRU Spatio-Temporal Model
======================================================
Modular implementation combining a 2D convolutional spatial backbone (ResNet-18)
with a Recurrent Neural Network (GRU) for temporal video classification.

Architecture Components:
------------------------
1. ResNetSpatialExtractor:
   - Extracts 512-dimensional spatial representations from per-frame video tensors.
   - Input: [B, T, 3, 224, 224] -> Output: [B, T, 512].
   - Reusable independently for offline feature extraction and caching.

2. GRUClassifier:
   - Models temporal kinetic progression across sequential frame embeddings.
   - Input: [B, T, 512] -> Output: [B, 1] raw logits.
   - Reusable independently for training/inference on pre-cached feature tensors.

3. ResNetGRU:
   - Composite end-to-end model coupling ResNetSpatialExtractor + GRUClassifier.
   - Supports forward(raw_frames) -> logits for end-to-end inference.
   - Supports forward_features(cached_features) -> logits for cached evaluation.
"""

from typing import Optional
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNetSpatialExtractor(nn.Module):
    """
    Spatial feature extraction backbone based on ResNet-18.

    Replaces the final ImageNet classification layer (fc) with nn.Identity(),
    producing a 512-dimensional feature embedding for each input frame.
    """

    def __init__(
        self,
        pretrained: bool = False,
        freeze_backbone: bool = False,
        feature_dim: int = 512,
    ):
        super().__init__()
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.feature_dim = feature_dim

        # Load ResNet-18 with official torchvision weights or random initialization
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)

        # Replace final classification head with Identity
        if self.backbone.fc.in_features != feature_dim:
            raise ValueError(
                f"Expected backbone.fc.in_features to be {feature_dim}, got {self.backbone.fc.in_features}"
            )
        self.backbone.fc = nn.Identity()

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Video tensor of shape [B, T, C, H, W]
        Returns:
            Spatial embeddings tensor of shape [B, T, 512]
        """
        if x.ndim != 5:
            raise ValueError(
                f"Expected 5D input tensor of shape [B, T, C, H, W], but got {x.ndim} dimensions."
            )

        B, T, C, H, W = x.shape
        if C != 3:
            raise ValueError(
                f"Expected 3 color channels (RGB) at dimension 2, but got {C} channels."
            )

        # Flatten batch and time dimensions for parallel 2D convolution passes
        x_flat = x.view(B * T, C, H, W)
        features_flat = self.backbone(x_flat)  # [B * T, 512]
        features = features_flat.view(B, T, self.feature_dim)  # [B, T, 512]
        return features


class GRUClassifier(nn.Module):
    """
    Temporal sequence classifier based on Gated Recurrent Units (GRU).

    Ingests sequential spatial feature representations [B, T, 512] and produces
    unnormalized binary collision logits [B, 1] from the final temporal hidden state.
    """

    def __init__(
        self,
        input_size: int = 512,
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.bidirectional = bidirectional

        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=bidirectional,
        )

        num_directions = 2 if bidirectional else 1
        classifier_in = hidden_size * num_directions

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.classifier = nn.Linear(classifier_in, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Spatial feature sequence of shape [B, T, 512]
        Returns:
            logits: Output classification logits of shape [B, 1]
        """
        if features.ndim != 3:
            raise ValueError(
                f"Expected 3D feature tensor [B, T, {self.input_size}], got {features.ndim} dimensions."
            )

        B, T, F = features.shape
        if F != self.input_size:
            raise ValueError(
                f"Expected feature dimension {self.input_size}, but got {F}."
            )

        # gru_out shape: [B, T, hidden_size * num_directions]
        gru_out, _ = self.gru(features)

        # Slice final temporal representation
        if self.bidirectional:
            forward_final = gru_out[:, -1, : self.hidden_size]
            backward_final = gru_out[:, 0, self.hidden_size :]
            temporal_repr = torch.cat([forward_final, backward_final], dim=1)
        else:
            temporal_repr = gru_out[:, -1, :]  # [B, hidden_size]

        temporal_repr = self.dropout(temporal_repr)
        logits = self.classifier(temporal_repr)  # [B, 1]
        return logits


class ResNetGRU(nn.Module):
    """
    Composite Spatio-Temporal Model coupling ResNetSpatialExtractor + GRUClassifier.

    Supports both:
    1. forward(raw_frames): [B, T, 3, 224, 224] -> [B, 1] (end-to-end video inference)
    2. forward_features(features): [B, T, 512] -> [B, 1] (fast inference on cached features)
    """

    def __init__(
        self,
        pretrained: bool = False,
        freeze_backbone: bool = False,
        feature_dim: int = 512,
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()

        # Modular sub-components
        self.spatial_extractor = ResNetSpatialExtractor(
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            feature_dim=feature_dim,
        )

        self.temporal_classifier = GRUClassifier(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        # Backward-compatible attribute references
        self.backbone = self.spatial_extractor.backbone
        self.gru = self.temporal_classifier.gru
        self.classifier = self.temporal_classifier.classifier
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.bidirectional = bidirectional

    def extract_spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        """Exposes spatial extraction directly on raw video tensors."""
        return self.spatial_extractor(x)

    def forward_features(self, features: torch.Tensor) -> torch.Tensor:
        """Runs only the temporal GRU + classifier on pre-cached spatial feature tensors."""
        return self.temporal_classifier(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """End-to-end forward pass: [B, T, 3, 224, 224] -> [B, 1]."""
        features = self.spatial_extractor(x)
        logits = self.temporal_classifier(features)
        return logits

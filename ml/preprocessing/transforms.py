"""
RoadGuardian AI - Video Frame Transformations
=============================================
Transformation utilities for video frames preparing tensor inputs for
torchvision ResNet-18 feature extraction.

Standard ImageNet normalization parameters:
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
"""

from typing import Sequence, Union
import cv2
import numpy as np
import torch

# Standard ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class VideoTransform:
    """
    Transforms a sequence of raw video frames (in RGB format) into a normalized
    PyTorch FloatTensor of shape [T, C, H, W] suitable for ResNet-18 backbones.
    """

    def __init__(
        self,
        target_size: tuple[int, int] = (224, 224),
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
    ):
        self.target_size = target_size  # (height, width)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def transform_frame(self, frame_rgb: np.ndarray) -> torch.Tensor:
        """
        Transforms a single RGB frame (H, W, 3) to a normalized tensor (3, H_target, W_target).
        """
        h_target, w_target = self.target_size
        if frame_rgb.shape[:2] != (h_target, w_target):
            frame_resized = cv2.resize(
                frame_rgb, (w_target, h_target), interpolation=cv2.INTER_LINEAR
            )
        else:
            frame_resized = frame_rgb

        # Convert HWC uint8 [0, 255] to CHW float32 [0.0, 1.0]
        tensor = (
            torch.from_numpy(frame_resized)
            .permute(2, 0, 1)
            .to(dtype=torch.float32)
            / 255.0
        )

        # Apply ImageNet normalization: (x - mean) / std
        tensor = (tensor - self.mean) / self.std
        return tensor

    def __call__(
        self, frames: Union[Sequence[np.ndarray], np.ndarray]
    ) -> torch.Tensor:
        """
        Args:
            frames: Sequence of T RGB frames (numpy arrays) of shape (H, W, 3)
        Returns:
            torch.FloatTensor of shape [T, 3, target_height, target_width]
        """
        transformed_frames = [self.transform_frame(f) for f in frames]
        return torch.stack(transformed_frames, dim=0)


def get_default_video_transform(
    target_size: tuple[int, int] = (224, 224),
) -> VideoTransform:
    """Factory helper to obtain a standard VideoTransform instance."""
    return VideoTransform(target_size=target_size)

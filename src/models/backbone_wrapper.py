"""
ECG Backbone Wrappers (Phase 3 Foundation Model Interface)
==========================================================
Provides standard interfaces for ECG encoder backbones:
    - BaseECGBackbone: Abstract base class enforcing strict gradient freezing and feature dimension contracts.
    - ResNet1DBackbone: Standard 1D convolutional baseline encoder for 12-lead waveforms.
    - PretrainedBackboneWrapper: Generic wrapper for external foundation models (ECG-FM, K-MERL, GEM).
"""

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


class BaseECGBackbone(nn.Module, ABC):
    """
    Abstract Base Class for ECG encoder backbones.
    Enforces freeze mechanisms and feature dimension contracts.
    """

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Dimensionality of the pooled feature vector h_raw output by the backbone."""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor, lead_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the backbone.

        Args:
            x: Input tensor of shape (B, 12, T).
            lead_mask: Optional binary mask of shape (B, 12) or (12,).

        Returns:
            h_raw: Extracted feature representation of shape (B, feature_dim).
        """
        pass

    def freeze(self) -> None:
        """Freezes all backbone parameters to prevent gradient updates."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def unfreeze(self) -> None:
        """Unfreezes backbone parameters for end-to-end fine-tuning if needed."""
        for param in self.parameters():
            param.requires_grad = True
        self.train()

    def is_frozen(self) -> bool:
        """Checks if all parameters are frozen."""
        return all(not p.requires_grad for p in self.parameters())


class ConvBlock1D(nn.Module):
    """Basic 1D Residual Convolutional Block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += res
        return self.relu(out)


class ResNet1DBackbone(BaseECGBackbone):
    """
    Standard 1D-ResNet baseline encoder for 12-lead ECG waveforms.
    Maps (B, 12, T) -> (B, feature_dim).
    """

    def __init__(
        self,
        in_channels: int = 12,
        base_filters: int = 64,
        feature_dim: int = 512,
    ) -> None:
        super().__init__()
        self._feature_dim = feature_dim

        # Initial stem convolution
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # Residual stages
        self.stage1 = ConvBlock1D(base_filters, base_filters * 2, stride=2)       # 64 -> 128
        self.stage2 = ConvBlock1D(base_filters * 2, base_filters * 4, stride=2)   # 128 -> 256
        self.stage3 = ConvBlock1D(base_filters * 4, feature_dim, stride=2)        # 256 -> feature_dim

        # Adaptive global average pooling over temporal dimension
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, x: torch.Tensor, lead_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (B, 12, T).
            lead_mask: Optional binary mask (B, 12) or (12,). If supplied, enforces masking.
        """
        if x.ndim != 3 or x.shape[1] != 12:
            raise ValueError(f"Input must be of shape (B, 12, T), got {x.shape}")

        if lead_mask is not None:
            if lead_mask.ndim == 1:
                mask = lead_mask.view(1, 12, 1).to(x.device, dtype=x.dtype)
            elif lead_mask.ndim == 2:
                mask = lead_mask.unsqueeze(-1).to(x.device, dtype=x.dtype)
            else:
                raise ValueError(f"lead_mask must be 1D or 2D, got shape {lead_mask.shape}")
            x = x * mask

        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.global_pool(out).squeeze(-1)  # (B, feature_dim)
        return out


class PretrainedBackboneWrapper(BaseECGBackbone):
    """
    Adapter wrapper around arbitrary external ECG foundation models
    (e.g., ECG-FM, K-MERL, GEM) or custom embedding extractors.
    """

    def __init__(
        self,
        model: nn.Module,
        feature_dim: int,
        extractor_fn: Optional[Callable[[nn.Module, torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self._feature_dim = feature_dim
        self.extractor_fn = extractor_fn

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, x: torch.Tensor, lead_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if lead_mask is not None:
            if lead_mask.ndim == 1:
                mask = lead_mask.view(1, 12, 1).to(x.device, dtype=x.dtype)
            elif lead_mask.ndim == 2:
                mask = lead_mask.unsqueeze(-1).to(x.device, dtype=x.dtype)
            else:
                raise ValueError(f"lead_mask must be 1D or 2D, got shape {lead_mask.shape}")
            x = x * mask

        if self.extractor_fn is not None:
            return self.extractor_fn(self.model, x)
        return self.model(x)

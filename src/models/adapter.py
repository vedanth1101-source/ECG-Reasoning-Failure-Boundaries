"""
Bottleneck MLP Adapter Module (Phase 3 Master Implementation)
=============================================================
Implements the 2-layer residual bottleneck MLP adapter mapping continuous
backbone representations h_raw in R^{d_in} to the standardized latent vector:
    Z_ECG in R^768

Mathematical Formulation:
    h_norm = LayerNorm(h_raw)
    h_down = GELU(W_down @ h_norm + b_down),      W_down in R^{512 x d_in}
    h_mid  = Dropout(h_down, p=dropout)
    h_up   = W_up @ h_mid + b_up,                W_up in R^{768 x 512}
    Z_ECG  = LayerNorm(h_up + W_skip @ h_norm)

Invariants:
    - Guarantees exact output dimensionality (B, 768).
    - Preserves deterministic latent extraction across Control and Experimental paths (TC-02 Parity).
    - Lightweight footprint (<1M parameters).
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


class BottleneckMLPAdapter(nn.Module):
    """
    2-Layer Residual Bottleneck MLP Adapter mapping backbone features -> R^768.

    Args:
        d_in: Feature dimension produced by the upstream foundation backbone.
        d_mid: Bottleneck hidden layer dimension (default: 512).
        d_out: Target standardized latent vector dimension (default: 768).
        dropout: Dropout probability applied at the bottleneck layer (default: 0.1).
    """

    def __init__(
        self,
        d_in: int,
        d_mid: int = 512,
        d_out: int = 768,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_in <= 0 or d_mid <= 0 or d_out <= 0:
            raise ValueError(f"All dimensions must be positive integers. Got d_in={d_in}, d_mid={d_mid}, d_out={d_out}")

        self.d_in = d_in
        self.d_mid = d_mid
        self.d_out = d_out
        self.dropout_rate = dropout

        # 1. Input normalization to stabilize backbone feature scale
        self.input_norm = nn.LayerNorm(d_in)

        # 2. Down-projection to bottleneck dimension
        self.down_proj = nn.Linear(d_in, d_mid)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # 3. Up-projection to standardized latent dimension
        self.up_proj = nn.Linear(d_mid, d_out)

        # 4. Residual skip connection
        if d_in != d_out:
            self.skip_proj = nn.Linear(d_in, d_out, bias=False)
        else:
            self.skip_proj = nn.Identity()

        # 5. Output normalization to ensure Z_ECG is well-scaled for Phase 4 Reasoning Transformer
        self.output_norm = nn.LayerNorm(d_out)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize weights with Xavier/Glorot normal initialization."""
        nn.init.xavier_normal_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.xavier_normal_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

        if isinstance(self.skip_proj, nn.Linear):
            nn.init.xavier_normal_(self.skip_proj.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass projecting backbone features into standardized latent space Z_ECG.

        Args:
            h: Backbone feature tensor of shape (B, d_in).

        Returns:
            Z_ECG: Standardized latent vector of shape (B, 768).
        """
        if h.ndim != 2:
            raise ValueError(f"Expected 2D tensor of shape (batch_size, d_in), got shape {h.shape}")
        if h.shape[1] != self.d_in:
            raise ValueError(f"Expected feature dimension {self.d_in}, got {h.shape[1]}")

        # LayerNorm on raw backbone features
        h_norm = self.input_norm(h)

        # Bottleneck non-linear transformation
        h_down = self.act(self.down_proj(h_norm))
        h_mid = self.dropout(h_down)
        h_up = self.up_proj(h_mid)

        # Residual skip connection + Output LayerNorm
        h_res = self.skip_proj(h_norm)
        z_ecg = self.output_norm(h_up + h_res)

        return z_ecg

    def get_parameter_summary(self) -> Dict[str, int]:
        """Returns parameter counts of the adapter."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": total_params - trainable_params,
            "d_in": self.d_in,
            "d_mid": self.d_mid,
            "d_out": self.d_out,
        }


class TemporaryClassificationHead(nn.Module):
    """
    Temporary supervised classification head attached to Z_ECG (in R^768)
    for pre-alignment / diagnostic loss training on PTB-XL classes.
    """

    def __init__(self, d_in: int = 768, num_classes: int = 5, dropout: float = 0.2) -> None:
        super().__init__()
        self.d_in = d_in
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_in, num_classes),
        )

    def forward(self, z_ecg: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing diagnostic logits.

        Args:
            z_ecg: Latent vector of shape (B, 768).

        Returns:
            logits: Diagnostic logits of shape (B, num_classes).
        """
        return self.classifier(z_ecg)


class AdapterAlignmentModel(nn.Module):
    """
    Combined training container housing the BottleneckMLPAdapter and
    TemporaryClassificationHead for supervised alignment.
    """

    def __init__(
        self,
        d_in: int,
        d_mid: int = 512,
        d_out: int = 768,
        num_classes: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.adapter = BottleneckMLPAdapter(d_in=d_in, d_mid=d_mid, d_out=d_out, dropout=dropout)
        self.classifier = TemporaryClassificationHead(d_in=d_out, num_classes=num_classes, dropout=dropout)

    def extract_latent(self, h: torch.Tensor) -> torch.Tensor:
        """Extracts standardized Z_ECG in R^768."""
        return self.adapter(h)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both latent representation Z_ECG and classification logits.

        Returns:
            (z_ecg, logits): Tuple of tensors of shapes (B, 768) and (B, num_classes).
        """
        z_ecg = self.adapter(h)
        logits = self.classifier(z_ecg)
        return z_ecg, logits

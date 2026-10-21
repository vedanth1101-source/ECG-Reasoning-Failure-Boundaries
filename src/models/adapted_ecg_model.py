"""
Adapted ECG Model Module (Phase 3 Master Model Container)
=========================================================
Encapsulates an ECG foundation/baseline backbone together with the
2-layer residual BottleneckMLPAdapter into an integrated, modular unit.

Guarantees:
    - Enforces freeze state on backbone (TC-02 Parity).
    - Exposes extract_latent() -> Z_ECG in R^768.
    - Transparently routes masked ECG input (B, 12, T).
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import torch
import torch.nn as nn

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.models.adapter import BottleneckMLPAdapter
from src.models.backbone_wrapper import BaseECGBackbone


class AdaptedECGModel(nn.Module):
    """
    Unified end-to-end container combining an ECG backbone with the
    BottleneckMLPAdapter to produce standardized Z_ECG in R^768.

    Args:
        backbone: An instance of BaseECGBackbone.
        adapter: Optional custom BottleneckMLPAdapter. If None, initialized
                 automatically using backbone.feature_dim and d_out=768.
        freeze_backbone: If True, freezes all parameters of the backbone.
    """

    def __init__(
        self,
        backbone: BaseECGBackbone,
        adapter: Optional[BottleneckMLPAdapter] = None,
        freeze_backbone: bool = True,
        d_mid: int = 512,
        d_out: int = 768,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = backbone

        if adapter is not None:
            if adapter.d_in != backbone.feature_dim:
                raise ValueError(
                    f"Adapter d_in ({adapter.d_in}) must match backbone feature_dim ({backbone.feature_dim})"
                )
            self.adapter = adapter
        else:
            self.adapter = BottleneckMLPAdapter(
                d_in=backbone.feature_dim,
                d_mid=d_mid,
                d_out=d_out,
                dropout=dropout,
            )

        if freeze_backbone:
            self.backbone.freeze()

    def forward(
        self,
        x: torch.Tensor,
        lead_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Processes masked ECG input and outputs standardized latent representation Z_ECG.

        Args:
            x: Masked ECG tensor of shape (B, 12, T).
            lead_mask: Optional binary availability mask of shape (B, 12) or (12,).

        Returns:
            z_ecg: Latent vector of shape (B, 768).
        """
        # 1. Extract continuous backbone features (B, feature_dim)
        h_raw = self.backbone(x, lead_mask=lead_mask)

        # 2. Project into standardized latent space Z_ECG in R^768
        z_ecg = self.adapter(h_raw)

        return z_ecg

    def extract_raw_features(
        self,
        x: torch.Tensor,
        lead_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extracts unadapted backbone features h_raw in R^{d_in}."""
        return self.backbone(x, lead_mask=lead_mask)

    def verify_backbone_frozen(self) -> bool:
        """Verifies that all parameters in the backbone have requires_grad=False."""
        return self.backbone.is_frozen()

    def get_parameter_summary(self) -> Dict[str, Any]:
        """Returns comprehensive parameter breakdown across backbone and adapter."""
        backbone_total = sum(p.numel() for p in self.backbone.parameters())
        backbone_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)

        adapter_total = sum(p.numel() for p in self.adapter.parameters())
        adapter_trainable = sum(p.numel() for p in self.adapter.parameters() if p.requires_grad)

        return {
            "backbone": {
                "total": backbone_total,
                "trainable": backbone_trainable,
                "frozen": backbone_total - backbone_trainable,
                "is_frozen": self.backbone.is_frozen(),
                "feature_dim": self.backbone.feature_dim,
            },
            "adapter": {
                "total": adapter_total,
                "trainable": adapter_trainable,
                "frozen": adapter_total - adapter_trainable,
                "d_in": self.adapter.d_in,
                "d_mid": self.adapter.d_mid,
                "d_out": self.adapter.d_out,
            },
            "total_model_parameters": backbone_total + adapter_total,
            "total_trainable_parameters": backbone_trainable + adapter_trainable,
        }

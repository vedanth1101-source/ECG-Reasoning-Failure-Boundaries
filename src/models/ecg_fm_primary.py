"""
Frozen ECG-FM Primary Foundation Model Interface (Phase 3 Master Implementation)
================================================================================
Provides the direct, frozen foundation model representation pipeline:

                 MASKED ECG (B, 12, 1000)
                            |
                            v
                 ECG-FM Input Formatter
               (shape & channel contract)
                            |
                            v
                      FROZEN ECG-FM
                 (wav2vec2_cmsc Backbone)
                            |
                            v
                     Z_ECG in R^768
                            |
                            v
                     MODULE 4 LATER

CRITICAL RESEARCH CONSTRAINTS (Phase 3 Final Design):
1. Zero trainable representation adapters.
2. Zero projection heads or bottleneck networks.
3. ECG-FM weights are strictly frozen (requires_grad = False).
4. Direct pooling of transformer encoder features yields Z_ECG in R^768.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, Any
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Add local fairseq_signals path
fairseq_signals_path = Path(__file__).resolve().parent / "fairseq_signals"
if str(fairseq_signals_path.parent) not in sys.path:
    sys.path.insert(0, str(fairseq_signals_path.parent))

from fairseq_signals.models import build_model


class FrozenECGFM(nn.Module):
    """
    Primary Foundation Model Container: Frozen ECG-FM (Wav2Vec2CMSCModel).
    
    Loads the official pretrained checkpoint `mimic_iv_ecg_physionet_pretrained.pt`
    and produces standardized latent representations Z_ECG in R^768 without
    any downstream trainable neural adapters.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Union[str, torch.device] = "cpu"
    ) -> None:
        super().__init__()
        self.device = torch.device(device)

        if checkpoint_path is None:
            checkpoint_path = repo_root / "checkpoints" / "ecg_fm" / "mimic_iv_ecg_physionet_pretrained.pt"
        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"ECG-FM checkpoint not found at: {self.checkpoint_path}")

        # 1. Load checkpoint
        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        model_cfg = OmegaConf.create(ckpt["cfg"]["model"])

        # 2. Instantiate official Wav2Vec2CMSCModel architecture
        self.model = build_model(model_cfg, task=None)
        self.model.load_state_dict(ckpt["model"], strict=True)

        # 3. Strictly freeze all parameters (Zero trainable parameters)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        self.model.to(self.device)

        self._feature_dim = 768

    @property
    def feature_dim(self) -> int:
        """Native representation dimension of ECG-FM."""
        return self._feature_dim

    def format_input(
        self,
        x: Union[np.ndarray, torch.Tensor],
        lead_mask: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Deterministic ECG-FM Input Formatting (Input Adapter).
        Verifies and enforces tensor shape (B, 12, 1000) and applies mask if provided.
        NOTE: This is NOT a trainable neural network.
        """
        if isinstance(x, np.ndarray):
            x_t = torch.from_numpy(x).float()
        else:
            x_t = x.float()

        if x_t.ndim == 2:
            x_t = x_t.unsqueeze(0)  # (1, 12, T)

        if x_t.shape[1] != 12:
            raise ValueError(f"ECG-FM requires 12 input leads along dimension 1, got shape {x_t.shape}")

        if lead_mask is not None:
            if isinstance(lead_mask, np.ndarray):
                m_t = torch.from_numpy(lead_mask).float()
            else:
                m_t = lead_mask.float()
            if m_t.ndim == 1:
                m_t = m_t.view(1, 12, 1)
            elif m_t.ndim == 2:
                m_t = m_t.unsqueeze(-1)
            x_t = x_t * m_t.to(x_t.device)

        return x_t.to(self.device)

    def forward(
        self,
        x: Union[np.ndarray, torch.Tensor],
        lead_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
        pool_method: str = "mean"
    ) -> torch.Tensor:
        """
        Extracts standardized latent vector Z_ECG in R^768 from the frozen ECG-FM.

        Args:
            x: Input tensor of shape (B, 12, 1000) or (12, 1000).
            lead_mask: Optional binary availability mask vector.
            pool_method: Temporal pooling ('mean').

        Returns:
            z_ecg: Latent representation of shape (B, 768).
        """
        x_formatted = self.format_input(x, lead_mask=lead_mask)

        with torch.no_grad():
            # features_only=True extracts output of the 12-layer Transformer encoder
            res = self.model(source=x_formatted, mask=False, features_only=True)
            seq_reps = res["x"]  # Shape: (B, T_frames, 768)

            if pool_method == "mean":
                z_ecg = seq_reps.mean(dim=1)  # (B, 768)
            else:
                raise ValueError(f"Unsupported pooling method: {pool_method}")

        return z_ecg

    def extract_latent(
        self,
        x: Union[np.ndarray, torch.Tensor],
        lead_mask: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Alias for forward pass to extract Z_ECG in R^768."""
        return self.forward(x, lead_mask=lead_mask)

    def is_completely_frozen(self) -> bool:
        """Confirms that all parameters have requires_grad == False."""
        return all(not p.requires_grad for p in self.parameters())

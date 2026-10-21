"""
Unit Tests for Phase 3 Bottleneck MLP Adapter and Model Containers
==================================================================
Tests:
    - Shape invariants: (B, d_in) -> (B, 768).
    - TC-02 Parity: Identical masked inputs yield identical Z_ECG.
    - Gradient Isolation: Backbone is completely frozen; only adapter trains.
    - Modular Backbone Integration: ResNet1D baseline + Adapter.
"""

import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.models.adapter import (
    BottleneckMLPAdapter,
    TemporaryClassificationHead,
    AdapterAlignmentModel,
)
from src.models.backbone_wrapper import ResNet1DBackbone, PretrainedBackboneWrapper
from src.models.adapted_ecg_model import AdaptedECGModel


class TestBottleneckMLPAdapter:
    """Tests for BottleneckMLPAdapter module."""

    @pytest.mark.parametrize("d_in", [512, 768, 1024])
    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_output_shape(self, d_in: int, batch_size: int):
        """Verifies output shape is strictly (batch_size, 768)."""
        adapter = BottleneckMLPAdapter(d_in=d_in, d_mid=512, d_out=768)
        x = torch.randn(batch_size, d_in)
        z = adapter(x)

        assert z.shape == (batch_size, 768)
        assert not torch.isnan(z).any()
        assert not torch.isinf(z).any()

    def test_invalid_input_dimensions(self):
        """Verifies error handling on mismatched or malformed inputs."""
        adapter = BottleneckMLPAdapter(d_in=512, d_out=768)

        # Wrong feature dimension
        with pytest.raises(ValueError, match="Expected feature dimension 512"):
            adapter(torch.randn(4, 256))

        # 3D input instead of 2D
        with pytest.raises(ValueError, match="Expected 2D tensor"):
            adapter(torch.randn(4, 12, 512))

    def test_residual_projection_types(self):
        """Verifies identity skip when dimensions match, linear when mismatched."""
        adapter_same = BottleneckMLPAdapter(d_in=768, d_out=768)
        assert isinstance(adapter_same.skip_proj, nn.Identity)

        adapter_diff = BottleneckMLPAdapter(d_in=512, d_out=768)
        assert isinstance(adapter_diff.skip_proj, nn.Linear)

    def test_parameter_summary(self):
        """Verifies parameter summary telemetry."""
        adapter = BottleneckMLPAdapter(d_in=512, d_mid=256, d_out=768)
        summary = adapter.get_parameter_summary()

        assert summary["d_in"] == 512
        assert summary["d_mid"] == 256
        assert summary["d_out"] == 768
        assert summary["total_parameters"] > 0
        assert summary["trainable_parameters"] == summary["total_parameters"]


class TestAlignmentAndClassificationHead:
    """Tests for classification head and alignment model."""

    def test_temporary_classification_head(self):
        head = TemporaryClassificationHead(d_in=768, num_classes=5)
        z = torch.randn(8, 768)
        logits = head(z)
        assert logits.shape == (8, 5)

    def test_adapter_alignment_model(self):
        alignment_model = AdapterAlignmentModel(d_in=512, d_out=768, num_classes=71)
        alignment_model.eval()
        h = torch.randn(4, 512)

        z_ecg, logits = alignment_model(h)
        assert z_ecg.shape == (4, 768)
        assert logits.shape == (4, 71)

        z_direct = alignment_model.extract_latent(h)
        assert torch.allclose(z_ecg, z_direct)


class TestAdaptedECGModel:
    """Tests for AdaptedECGModel and Backbone integration."""

    def test_resnet1d_backbone_forward(self):
        """Verifies standard 1D CNN backbone forward pass."""
        backbone = ResNet1DBackbone(in_channels=12, feature_dim=512)
        x = torch.randn(2, 12, 1000)
        h = backbone(x)
        assert h.shape == (2, 512)

    def test_adapted_model_end_to_end(self):
        """Verifies complete end-to-end forward mapping (B, 12, T) -> (B, 768)."""
        backbone = ResNet1DBackbone(in_channels=12, feature_dim=512)
        model = AdaptedECGModel(backbone=backbone, d_out=768, freeze_backbone=True)

        x = torch.randn(4, 12, 1000)
        z = model(x)
        assert z.shape == (4, 768)

    def test_gradient_isolation_freeze(self):
        """
        CRITICAL TEST: Ensures backbone parameters have requires_grad=False
        and receive NO gradient updates during backward pass.
        """
        backbone = ResNet1DBackbone(in_channels=12, feature_dim=512)
        model = AdaptedECGModel(backbone=backbone, freeze_backbone=True)

        assert model.verify_backbone_frozen() is True

        # Forward pass
        x = torch.randn(2, 12, 500)
        z = model(x)
        loss = z.sum()
        loss.backward()

        # Check that backbone parameters have NO gradients
        for name, param in model.backbone.named_parameters():
            assert param.grad is None, f"Backbone parameter {name} received unexpected gradient!"

        # Check that adapter parameters DO receive gradients
        for name, param in model.adapter.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Adapter parameter {name} should have received gradient!"

    def test_tc02_parity(self):
        """
        TEST TC-02: Control vs Experimental Parity.
        Given identical masked input X_masked, output Z_ECG must be identical.
        """
        backbone = ResNet1DBackbone(in_channels=12, feature_dim=512)
        model = AdaptedECGModel(backbone=backbone, freeze_backbone=True)
        model.eval()

        x = torch.randn(3, 12, 1000)
        lead_mask = torch.tensor([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0], dtype=torch.float32)

        with torch.no_grad():
            z1 = model(x, lead_mask=lead_mask)
            z2 = model(x, lead_mask=lead_mask)

        # Bitwise parity assertion
        assert torch.allclose(z1, z2, atol=1e-7), "TC-02 Failure: Non-deterministic latent vector extraction!"

"""
Phase 3: Model Adaptation and Latent Representation Subsystem
=============================================================
Provides:
    - BottleneckMLPAdapter: 2-layer residual bottleneck projection to Z_ECG in R^768.
    - BaseECGBackbone: Abstract backbone interface for ECG encoders.
    - ResNet1DBackbone: Standard 1D-CNN baseline encoder for 12-lead waveforms.
    - AdaptedECGModel: Unified backbone + adapter module with freeze enforcement.
    - TemporaryClassificationHead: Detachable classification head for supervised alignment.
"""

from src.models.adapter import (
    BottleneckMLPAdapter,
    TemporaryClassificationHead,
    AdapterAlignmentModel,
)
from src.models.backbone_wrapper import (
    BaseECGBackbone,
    ResNet1DBackbone,
    PretrainedBackboneWrapper,
)
from src.models.adapted_ecg_model import AdaptedECGModel
from src.models.ecg_fm_primary import FrozenECGFM

__all__ = [
    "FrozenECGFM",
    "BottleneckMLPAdapter",
    "TemporaryClassificationHead",
    "AdapterAlignmentModel",
    "BaseECGBackbone",
    "ResNet1DBackbone",
    "PretrainedBackboneWrapper",
    "AdaptedECGModel",
]

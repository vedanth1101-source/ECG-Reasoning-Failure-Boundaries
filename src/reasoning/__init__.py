"""
Clinical Reasoning Package
==========================
Contains modules for multimodal context construction, reasoning transformers,
trace generation, and evaluation under controlled lead degradation.
"""

from src.reasoning.context_builder import (
    ControlContextBuilder,
    LEAD_NAMES,
    CANDIDATE_ACRONYMS,
    CANDIDATE_NAMES,
)
from src.reasoning.reasoning_transformer import ControlReasoningTransformer
from src.reasoning.trace_generator import StructuredTraceGenerator
from src.reasoning.dataset import (
    ControlReasoningDataset,
    collate_control_reasoning_batch,
)

__all__ = [
    "ControlContextBuilder",
    "ControlReasoningTransformer",
    "StructuredTraceGenerator",
    "ControlReasoningDataset",
    "collate_control_reasoning_batch",
    "LEAD_NAMES",
    "CANDIDATE_ACRONYMS",
    "CANDIDATE_NAMES",
]

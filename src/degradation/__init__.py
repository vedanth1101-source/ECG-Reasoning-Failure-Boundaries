"""
Phase 2: Controlled Degradation & Lead Availability Masking Subsystem
=====================================================================
Implements deterministic, un-reconstructed lead masking across anatomical
and progressive degradation schedules.
"""

from .lead_masking import (
    CANONICAL_LEADS,
    LEAD_TO_INDEX,
    AnatomicalRegion,
    LeadMask,
    MaskedECG,
    LeadMaskingOperator,
    get_anatomical_schedule,
    get_stepwise_schedule,
    create_custom_mask
)

__all__ = [
    "CANONICAL_LEADS",
    "LEAD_TO_INDEX",
    "AnatomicalRegion",
    "LeadMask",
    "MaskedECG",
    "LeadMaskingOperator",
    "get_anatomical_schedule",
    "get_stepwise_schedule",
    "create_custom_mask"
]

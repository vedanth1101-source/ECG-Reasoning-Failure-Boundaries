"""
Lead Ordering Module (P_1)
--------------------------
Reorganizes heterogeneous raw input channels into the strict, canonical
clinical 12-lead sequence:
['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'].
"""

from typing import List, Tuple, Dict, Optional
import numpy as np


CANONICAL_LEADS: List[str] = [
    "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"
]

# Standardize common dataset naming variations
LEAD_SYNONYMS: Dict[str, str] = {
    # Limb leads
    "i": "I", "lead1": "I", "lead_1": "I", "leadi": "I",
    "ii": "II", "lead2": "II", "lead_2": "II", "leadii": "II",
    "iii": "III", "lead3": "III", "lead_3": "III", "leadiii": "III",
    "avr": "aVR", "leadavr": "aVR", "lead_avr": "aVR",
    "avl": "aVL", "leadavl": "aVL", "lead_avl": "aVL",
    "avf": "aVF", "leadavf": "aVF", "lead_avf": "aVF",
    # Precordial chest leads
    "v1": "V1", "leadv1": "V1", "lead_v1": "V1",
    "v2": "V2", "leadv2": "V2", "lead_v2": "V2",
    "v3": "V3", "leadv3": "V3", "lead_v3": "V3",
    "v4": "V4", "leadv4": "V4", "lead_v4": "V4",
    "v5": "V5", "leadv5": "V5", "lead_v5": "V5",
    "v6": "V6", "leadv6": "V6", "lead_v6": "V6",
}


class LeadOrderingError(ValueError):
    """Raised when lead ordering cannot be resolved."""
    pass


def normalize_lead_name(name: str) -> str:
    """Normalizes a raw lead name to canonical notation."""
    cleaned = name.strip()
    # Check exact match first
    if cleaned in CANONICAL_LEADS:
        return cleaned
    
    key = cleaned.lower().replace("-", "_").replace(" ", "")
    if key in LEAD_SYNONYMS:
        return LEAD_SYNONYMS[key]
    
    # Try uppercase for roman numerals and V-leads
    upper = cleaned.upper()
    if upper in ["I", "II", "III", "V1", "V2", "V3", "V4", "V5", "V6"]:
        return upper
    if upper == "AVR":
        return "aVR"
    if upper == "AVL":
        return "aVL"
    if upper == "AVF":
        return "aVF"
        
    return cleaned


class LeadOrdering:
    """
    P_1: Canonical Lead Ordering Component.
    Maps raw multi-channel ECG matrices into the canonical 12-lead sequence:
    [I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6].
    """

    def __init__(self, canonical_leads: Optional[List[str]] = None):
        self.canonical_leads = canonical_leads or list(CANONICAL_LEADS)

    def reorder(
        self, signal: np.ndarray, lead_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Reorder input channels to match canonical lead order.

        Args:
            signal: numpy array of shape (C, T) or (T, C).
            lead_names: list of channel names corresponding to channels in signal.

        Returns:
            Tuple of (reordered_signal of shape (12, T), canonical_lead_names).
        """
        if signal.ndim != 2:
            raise LeadOrderingError(
                f"Expected 2D signal matrix, got ndim={signal.ndim} with shape {signal.shape}"
            )

        # Determine channel axis: standard is (C, T)
        # If signal is (T, C) where C == len(lead_names), transpose it
        if signal.shape[0] != len(lead_names) and signal.shape[1] == len(lead_names):
            signal = signal.T

        if signal.shape[0] != len(lead_names):
            raise LeadOrderingError(
                f"Channel dimension mismatch: signal has {signal.shape[0]} channels, "
                f"but {len(lead_names)} lead names were provided."
            )

        # Normalize lead names
        normalized_names = [normalize_lead_name(name) for name in lead_names]
        
        # Build index mapping
        name_to_idx: Dict[str, int] = {}
        for idx, name in enumerate(normalized_names):
            if name in name_to_idx:
                raise LeadOrderingError(f"Duplicate lead channel detected: '{name}'")
            name_to_idx[name] = idx

        # Verify all canonical leads are available
        missing_leads = [lead for lead in self.canonical_leads if lead not in name_to_idx]
        if missing_leads:
            raise LeadOrderingError(
                f"Missing mandatory canonical leads: {missing_leads}. "
                f"Provided channels: {normalized_names}"
            )

        # Assemble reordered permutation matrix
        indices = [name_to_idx[lead] for lead in self.canonical_leads]
        reordered_signal = np.copy(signal[indices, :])

        return reordered_signal, list(self.canonical_leads)

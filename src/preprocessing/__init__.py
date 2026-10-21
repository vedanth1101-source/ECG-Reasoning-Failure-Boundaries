"""
ECG Preprocessing Package (Phase 1)
-----------------------------------
Canonical pipeline implementing the 5 sequential theoretical operations:
- P_1: Lead Ordering (Canonical 12-lead sequence)
- P_2: Sampling and Duration Standardization (Resampling & Windowing)
- P_3: Unit Calibration and Normalization (ADC-to-mV conversion & Scaling)
- P_4: Mathematical Filtering (Zero-phase Butterworth highpass, lowpass, notch)
- P_5: Quality Screening and Integrity Audit (Flatline, NaN/Inf, Dimension & Saturation check)
"""

from .lead_ordering import LeadOrdering, CANONICAL_LEADS, LeadOrderingError, normalize_lead_name
from .sampling_standardization import SamplingStandardizer
from .unit_calibration import UnitCalibrator
from .filtering import SignalFilter
from .quality_screening import QualityScreening, QualityAuditResult
from .pipeline import ECGPreprocessingPipeline, PreprocessedECG
from .dataset_loader import PTBXLDatasetLoader

__all__ = [
    "LeadOrdering",
    "CANONICAL_LEADS",
    "LeadOrderingError",
    "normalize_lead_name",
    "SamplingStandardizer",
    "UnitCalibrator",
    "SignalFilter",
    "QualityScreening",
    "QualityAuditResult",
    "ECGPreprocessingPipeline",
    "PreprocessedECG",
    "PTBXLDatasetLoader",
]

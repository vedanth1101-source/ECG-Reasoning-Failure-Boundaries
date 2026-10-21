"""
Phase 3: Structured Clinical Observation Extraction Subsystem
=============================================================
Provides:
    - ClinicalObservationExtractor: Deterministic physiological biomarker extractor.
    - ClinicalObservation: Typed dataclass representation of discrete observations O_l.
"""

from src.observations.clinical_extractor import (
    ClinicalObservationExtractor,
    ClinicalObservation,
)

__all__ = [
    "ClinicalObservationExtractor",
    "ClinicalObservation",
]

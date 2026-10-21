"""
Unit Tests for Phase 3 Subsystem B: Clinical Observation Extraction
===================================================================
Tests:
    - Schema conformity matching Section 3.2 JSON specification.
    - Observability invariant: masked leads are NEVER reported in observations.
    - Determinism: identical inputs produce identical observations.
    - Edge cases: flatline, single lead, zero leads.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.observations.clinical_extractor import (
    ClinicalObservationExtractor,
    ClinicalObservation,
)
from src.degradation.lead_masking import LeadMaskingOperator, CANONICAL_LEADS


class TestClinicalObservationExtractor:

    def test_schema_keys_and_types(self):
        """Verifies observation dictionary matches Section 3.2 specification."""
        extractor = ClinicalObservationExtractor(fs=100.0)

        # Create synthetic 12-lead signal (10 seconds, 72 bpm sinus rhythm)
        t = np.linspace(0, 10, 1000)
        signal = np.zeros((12, 1000), dtype=np.float32)
        # Add synthetic periodic R-peaks at ~1.2 Hz (72 bpm)
        for i in range(12):
            signal[i] = 0.5 * np.sin(2 * np.pi * 1.2 * t)

        obs = extractor.extract(signal)
        d = obs.to_dict()

        assert "rhythm" in d
        assert "heart_rate" in d
        assert "pr_interval_ms" in d
        assert "qrs_duration_ms" in d
        assert "st_elevation" in d
        assert "st_depression" in d
        assert "t_wave_inversion" in d

        assert isinstance(d["st_elevation"]["present"], bool)
        assert isinstance(d["st_elevation"]["leads"], list)
        assert isinstance(d["st_elevation"]["amplitude_mm"], dict)

        assert isinstance(d["st_depression"]["present"], bool)
        assert isinstance(d["st_depression"]["leads"], list)
        assert isinstance(d["st_depression"]["amplitude_mm"], dict)

        assert isinstance(d["t_wave_inversion"]["present"], bool)
        assert isinstance(d["t_wave_inversion"]["leads"], list)

    def test_mask_observability_invariant(self):
        """
        CRITICAL TEST: Ensures masked leads are NEVER reported in st_elevation,
        st_depression, or t_wave_inversion.
        """
        extractor = ClinicalObservationExtractor(fs=100.0)
        operator = LeadMaskingOperator()

        # Create signal with high voltage on all leads
        signal = np.ones((12, 1000), dtype=np.float32) * 0.3  # ST elevation on all channels

        # Apply 4-lead mask: ["I", "II", "III", "V1"]
        masked_ecg = operator.apply(signal, "4_lead")
        active_leads = set(masked_ecg.mask.active_leads)

        obs = extractor.extract(masked_ecg.signal, lead_mask=masked_ecg.mask.mask_vector)
        d = obs.to_dict()

        # All reported leads MUST be a subset of active_leads
        for lead in d["st_elevation"]["leads"]:
            assert lead in active_leads, f"Masked lead {lead} illegally reported in ST elevation!"

        for lead in d["st_depression"]["leads"]:
            assert lead in active_leads, f"Masked lead {lead} illegally reported in ST depression!"

        for lead in d["t_wave_inversion"]["leads"]:
            assert lead in active_leads, f"Masked lead {lead} illegally reported in T-wave inversion!"

    def test_zero_leads_edge_case(self):
        """Tests handling when all leads are masked."""
        extractor = ClinicalObservationExtractor(fs=100.0)
        signal = np.zeros((12, 1000), dtype=np.float32)
        zero_mask = np.zeros(12, dtype=np.uint8)

        obs = extractor.extract(signal, lead_mask=zero_mask)
        d = obs.to_dict()

        assert d["heart_rate"] is None
        assert d["st_elevation"]["present"] is False
        assert len(d["st_elevation"]["leads"]) == 0

    def test_determinism(self):
        """Tests that identical signals yield bitwise identical observations."""
        extractor = ClinicalObservationExtractor(fs=100.0)
        rng = np.random.default_rng(42)
        signal = rng.standard_normal((12, 1000)).astype(np.float32)

        obs1 = extractor.extract(signal).to_dict()
        obs2 = extractor.extract(signal).to_dict()

        assert obs1 == obs2

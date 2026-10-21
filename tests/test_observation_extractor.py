"""
Unit Tests for Module 4 (Part 1): Linear Observation Extractor & Schema Validation
===================================================================================
Validates:
    - Projection head output shapes (Equations 4.6 & 4.7)
    - Rigid clinical taxonomy adherence
    - Observability invariant: masked leads never leak into positive findings
    - Electrophysiological sanity checks (AFib P-wave absence)
    - Zero patient leakage across multi-cohort splits
"""

import sys
from pathlib import Path
import pytest
import torch
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.reasoning.clinical_taxonomy import (
    LEAD_NAMES,
    RHYTHM_CLASSES,
    RATE_CLASSES,
    PR_CLASSES,
    QRS_CLASSES,
    ST_STATES,
)
from src.reasoning.observation_extractor import LinearObservationExtractor
from src.reasoning.schema_validator import SchemaValidationLayer
from src.reasoning.multi_cohort_loader import MultiCohortPartitionManager


class TestLinearObservationExtractor:
    def test_linear_head_output_shapes(self):
        B = 4
        z = torch.randn(B, 768)
        extractor = LinearObservationExtractor(d_in=768)
        logits = extractor(z)

        assert "rhythm" in logits
        assert "rate" in logits
        assert "PR" in logits
        assert "QRS" in logits
        assert "ST" in logits

        assert logits["rhythm"].shape == (B, len(RHYTHM_CLASSES))
        assert logits["rate"].shape == (B, len(RATE_CLASSES))
        assert logits["PR"].shape == (B, len(PR_CLASSES))
        assert logits["QRS"].shape == (B, len(QRS_CLASSES))
        assert logits["ST"].shape == (B, len(LEAD_NAMES), len(ST_STATES))

    def test_discretization_equations_4_6_and_4_7(self):
        B = 2
        z = torch.randn(B, 768)
        mask = torch.ones(B, 12, dtype=torch.float32)

        extractor = LinearObservationExtractor(d_in=768)
        findings = extractor.predict_discretized(z, mask)

        assert len(findings) == B
        for f in findings:
            assert f["rhythm"]["value"] in RHYTHM_CLASSES
            assert f["rate"]["category"] in RATE_CLASSES
            assert f["PR"]["category"] in PR_CLASSES
            assert f["QRS"]["category"] in QRS_CLASSES
            assert "lead_states" in f["ST"]
            assert len(f["ST"]["lead_states"]) == 12


class TestSchemaValidationLayer:
    def test_unadmitted_key_rejection(self):
        validator = SchemaValidationLayer()
        corrupted_finding = {
            "rhythm": {"value": "Sinus rhythm", "derived_leads": ["II"]},
            "rate": {"category": "Normal Rate (60-100 bpm)"},
            "PR": {"category": "Normal (120-200 ms)"},
            "QRS": {"category": "Normal (<120 ms)"},
            "ST": {"elevation_leads": [], "depression_leads": []},
            "illegal_hallucinated_key": "some_value",
        }
        mask = [1] * 12
        sanitized, is_valid, log = validator.validate_single_observation(corrupted_finding, mask)

        assert not is_valid
        assert any("illegal_hallucinated_key" in note for note in log)
        assert "illegal_hallucinated_key" not in sanitized

    def test_masked_lead_observability_invariant(self):
        validator = SchemaValidationLayer()
        # Lead V6 (index 11) is masked (mask[11] == 0)
        mask = [1] * 11 + [0]
        finding_with_leak = {
            "rhythm": {"value": "Sinus rhythm", "derived_leads": ["II"]},
            "rate": {"category": "Normal Rate (60-100 bpm)"},
            "PR": {"category": "Normal (120-200 ms)"},
            "QRS": {"category": "Normal (<120 ms)"},
            "ST": {
                "elevation_leads": ["V6"],  # V6 is masked!
                "depression_leads": [],
            },
        }

        sanitized, is_valid, log = validator.validate_single_observation(finding_with_leak, mask)

        assert not is_valid
        assert "V6" not in sanitized["ST"]["elevation_leads"]
        assert sanitized["ST"]["elevation_present"] is False
        assert any("V6" in note for note in log)

    def test_electrophysiological_sanity_afib_pr(self):
        validator = SchemaValidationLayer()
        finding_afib = {
            "rhythm": {"value": "Atrial fibrillation", "derived_leads": ["II"]},
            "rate": {"category": "Normal Rate (60-100 bpm)"},
            "PR": {"category": "Normal (120-200 ms)"},  # Conflict: AFib has no discrete PR
            "QRS": {"category": "Normal (<120 ms)"},
            "ST": {"elevation_leads": [], "depression_leads": []},
        }
        mask = [1] * 12
        sanitized, is_valid, log = validator.validate_single_observation(finding_afib, mask)

        assert sanitized["PR"]["category"] == "Absent / Indeterminate"
        assert any("AFib has absent P waves" in note for note in log)


class TestMultiCohortZeroLeakage:
    def test_multi_cohort_zero_patient_leakage(self):
        mgr = MultiCohortPartitionManager()
        audit = mgr.verify_zero_patient_leakage()

        for cohort_name, res in audit.items():
            assert res["zero_leakage_passed"] is True, f"Patient leakage detected in {cohort_name}!"
            assert res["train_val_leakage"] == 0
            assert res["train_test_leakage"] == 0
            assert res["val_test_leakage"] == 0

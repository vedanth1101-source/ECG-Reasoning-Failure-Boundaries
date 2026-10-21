"""
Unit Tests for Phase 2: Lead Availability Masking Subsystem
============================================================
Tests:
  1. Canonical Cabrera lead ordering and index mapping
  2. Anatomical reduction schedule definitions and lead counts
  3. Progressive stepwise elimination schedule definitions
  4. Single recording masking: (12, 1000)
  5. Batched recording masking: (32, 12, 1000)
  6. Inactive channel suppression: strictly 0.0
  7. Active channel fidelity: exactly unchanged
  8. Research constraint verification: is_reconstructed is False
  9. Error handling on anomalous channel dimensions
"""

import pytest
import numpy as np

from src.degradation.lead_masking import (
    CANONICAL_LEADS,
    LEAD_TO_INDEX,
    LeadMask,
    LeadMaskingOperator,
    get_anatomical_schedule,
    get_stepwise_schedule,
    create_custom_mask
)


@pytest.fixture
def operator():
    return LeadMaskingOperator()


@pytest.fixture
def sample_single_ecg():
    # Deterministic dummy signal (12, 1000)
    np.random.seed(123)
    return np.random.randn(12, 1000).astype(np.float32)


@pytest.fixture
def sample_batch_ecg():
    # Batch dummy signal (16, 12, 1000)
    np.random.seed(456)
    return np.random.randn(16, 12, 1000).astype(np.float32)


def test_canonical_leads_order():
    expected = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
    assert CANONICAL_LEADS == expected
    assert len(CANONICAL_LEADS) == 12
    for idx, lead in enumerate(expected):
        assert LEAD_TO_INDEX[lead] == idx


def test_anatomical_schedules_coverage(operator):
    sched = operator.anatomical_schedules
    assert "12_lead" in sched
    assert "8_lead_independent" in sched
    assert "6_lead_limb" in sched
    assert "6_lead_precordial" in sched
    assert "4_lead" in sched
    assert "2_lead_orthogonal" in sched
    assert "2_lead_rhythm" in sched
    assert "1_lead" in sched

    # Check lead counts
    assert sched["12_lead"].num_leads == 12
    assert sched["8_lead_independent"].num_leads == 8
    assert sched["6_lead_limb"].num_leads == 6
    assert sched["6_lead_precordial"].num_leads == 6
    assert sched["4_lead"].num_leads == 4
    assert sched["2_lead_orthogonal"].num_leads == 2
    assert sched["2_lead_rhythm"].num_leads == 2
    assert sched["1_lead"].num_leads == 1


def test_stepwise_schedules_coverage(operator):
    sched = operator.stepwise_schedules
    expected_counts = {
        "12_lead": 12,
        "10_lead": 10,
        "8_lead": 8,
        "6_lead": 6,
        "4_lead": 4,
        "2_lead": 2,
        "1_lead": 1
    }
    for name, cnt in expected_counts.items():
        assert name in sched
        assert sched[name].num_leads == cnt


def test_single_signal_masking(operator, sample_single_ecg):
    # Mask to 6 limb leads (V1-V6 should be zeroed)
    res = operator.apply(sample_single_ecg, "6_lead_limb")

    assert res.signal.shape == sample_single_ecg.shape
    assert res.signal.dtype == np.float32
    assert res.is_reconstructed is False

    # Limb leads (indices 0 to 5) must be preserved identically
    np.testing.assert_allclose(res.signal[0:6, :], sample_single_ecg[0:6, :])

    # Precordial leads (indices 6 to 11) must be strictly 0.0
    np.testing.assert_allclose(res.signal[6:12, :], 0.0)


def test_batch_signal_masking(operator, sample_batch_ecg):
    # Mask to single lead (Lead II at index 1)
    res = operator.apply(sample_batch_ecg, "1_lead")

    assert res.signal.shape == sample_batch_ecg.shape
    assert res.signal.dtype == np.float32

    # Lead II (index 1) preserved across all batch items
    np.testing.assert_allclose(res.signal[:, 1, :], sample_batch_ecg[:, 1, :])

    # All other leads must be strictly 0.0
    other_indices = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    np.testing.assert_allclose(res.signal[:, other_indices, :], 0.0)


def test_custom_mask(operator, sample_single_ecg):
    # Custom mask: Lateral leads only (I, aVL, V5, V6)
    custom = create_custom_mask(["I", "aVL", "V5", "V6"], name="lateral_only")
    res = operator.apply(sample_single_ecg, custom)

    assert res.num_active_leads == 4
    assert set(res.active_leads) == {"I", "aVL", "V5", "V6"}

    # Check active leads
    for lead in ["I", "aVL", "V5", "V6"]:
        idx = LEAD_TO_INDEX[lead]
        np.testing.assert_allclose(res.signal[idx, :], sample_single_ecg[idx, :])

    # Check non-lateral leads are zeroed
    for lead in ["II", "III", "aVR", "aVF", "V1", "V2", "V3", "V4"]:
        idx = LEAD_TO_INDEX[lead]
        np.testing.assert_allclose(res.signal[idx, :], 0.0)


def test_invalid_channel_dimension(operator):
    # Signal with 10 channels instead of 12
    bad_signal = np.random.randn(10, 1000).astype(np.float32)
    with pytest.raises(ValueError, match="Expected 12 channels"):
        operator.apply(bad_signal, "12_lead")


def test_invalid_lead_name():
    with pytest.raises(ValueError, match="Unknown lead 'V7'"):
        create_custom_mask(["I", "II", "V7"])


def test_zero_reconstruction_invariant(operator, sample_single_ecg):
    # Enforces research constraint: is_reconstructed must never be True
    res = operator.apply(sample_single_ecg, "2_lead_rhythm")
    assert res.is_reconstructed is False


def test_lead_provenance_and_observability():
    from src.degradation.lead_masking import (
        LEAD_PROVENANCE,
        get_lead_observability_status,
    )
    # 1. Measured leads
    assert LEAD_PROVENANCE["I"].source_type == "measured"
    assert LEAD_PROVENANCE["I"].independent_observation is True
    assert LEAD_PROVENANCE["II"].source_type == "measured"
    assert LEAD_PROVENANCE["V1"].source_type == "measured"

    # 2. Derived leads
    assert LEAD_PROVENANCE["III"].source_type == "derived"
    assert LEAD_PROVENANCE["III"].parent_leads == ("I", "II")
    assert LEAD_PROVENANCE["III"].independent_observation is False
    assert LEAD_PROVENANCE["aVR"].source_type == "derived"

    # 3. Observability when both parents present
    status_12 = get_lead_observability_status("III", CANONICAL_LEADS)
    assert status_12 == "observed_derived"
    status_I = get_lead_observability_status("I", CANONICAL_LEADS)
    assert status_I == "observed_measured"

    # 4. Observability when lead is masked
    status_masked = get_lead_observability_status("V5", ["I", "II", "V1"])
    assert status_masked == "masked_lead"

    # 5. Observability when derived lead is in active list BUT parent lead I is missing
    # e.g., active = ["II", "III", "V1"] -> I is missing!
    status_broken_parent = get_lead_observability_status("III", ["II", "III", "V1"])
    assert status_broken_parent == "unavailable_derived"

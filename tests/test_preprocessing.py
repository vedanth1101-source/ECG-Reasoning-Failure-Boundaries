"""
Unit & Integration Tests for Phase 1 Data Preprocessing Pipeline
=================================================================
Covers P_1 through P_5 and end-to-end PTB-XL dataset processing.
"""

import os
from pathlib import Path
import pytest
import numpy as np

from src.preprocessing.lead_ordering import (
    LeadOrdering, CANONICAL_LEADS, LeadOrderingError, normalize_lead_name
)
from src.preprocessing.sampling_standardization import SamplingStandardizer
from src.preprocessing.unit_calibration import UnitCalibrator
from src.preprocessing.filtering import SignalFilter
from src.preprocessing.quality_screening import QualityScreening, QualityAuditResult
from src.preprocessing.pipeline import ECGPreprocessingPipeline, PreprocessedECG
from src.preprocessing.dataset_loader import PTBXLDatasetLoader


# =====================================================================
# P_1: Lead Ordering Tests
# =====================================================================
def test_lead_ordering_canonical():
    """Verify standard canonical order is preserved."""
    orderer = LeadOrdering()
    # Provide scrambled channels
    scrambled_names = ["V6", "V5", "V4", "V3", "V2", "V1", "aVF", "aVL", "aVR", "III", "II", "I"]
    signal = np.arange(12)[:, None] * np.ones((1, 100))  # shape (12, 100)
    
    reordered, leads = orderer.reorder(signal, scrambled_names)
    assert leads == CANONICAL_LEADS
    assert reordered.shape == (12, 100)
    # Channel 'I' was at index 11 in scrambled_names, so reordered[0] should match signal[11]
    np.testing.assert_array_equal(reordered[0], signal[11])


def test_lead_ordering_synonyms():
    """Verify case-insensitivity and synonym handling (e.g. AVR -> aVR)."""
    orderer = LeadOrdering()
    ptbxl_names = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    signal = np.ones((12, 500))
    reordered, leads = orderer.reorder(signal, ptbxl_names)
    assert leads == CANONICAL_LEADS
    assert leads[3] == "aVR"
    assert leads[4] == "aVL"
    assert leads[5] == "aVF"


def test_lead_ordering_missing_lead():
    """Verify missing mandatory lead raises LeadOrderingError."""
    orderer = LeadOrdering()
    incomplete_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5"]  # missing V6
    signal = np.ones((11, 100))
    with pytest.raises(LeadOrderingError):
        orderer.reorder(signal, incomplete_names)


def test_lead_ordering_transposition():
    """Verify (T, C) array is correctly transposed to (C, T)."""
    orderer = LeadOrdering()
    signal_tc = np.ones((1000, 12))
    reordered, leads = orderer.reorder(signal_tc, CANONICAL_LEADS)
    assert reordered.shape == (12, 1000)


# =====================================================================
# P_2: Sampling and Duration Standardization Tests
# =====================================================================
def test_sampling_standardization_upsample_and_pad():
    """Verify resampling from 100 Hz to 500 Hz and padding 8s to 10s."""
    standardizer = SamplingStandardizer(target_fs=500.0, target_duration_sec=10.0)
    # 8 seconds at 100 Hz = 800 samples
    signal = np.ones((12, 800))
    std_sig, meta = standardizer.standardize(signal, fs_in=100.0)

    # 10s at 500 Hz = 5000 samples
    assert std_sig.shape == (12, 5000)
    assert meta["target_samples"] == 5000
    assert meta["pad_left"] > 0 or meta["pad_right"] > 0


def test_sampling_standardization_trim():
    """Verify centered trimming of excess duration (12s -> 10s)."""
    standardizer = SamplingStandardizer(target_fs=100.0, target_duration_sec=10.0, windowing_mode="center")
    # 12 seconds at 100 Hz = 1200 samples
    signal = np.ones((12, 1200))
    std_sig, meta = standardizer.standardize(signal, fs_in=100.0)

    assert std_sig.shape == (12, 1000)
    assert meta["crop_start"] == 100
    assert meta["crop_end"] == 1100


# =====================================================================
# P_3: Unit Calibration Tests
# =====================================================================
def test_unit_calibration_adc_conversion():
    """Verify ADC raw counts converted to mV using gain and baseline."""
    calibrator = UnitCalibrator(scaling_method="none")
    # Raw counts: baseline=2048, gain=1000 counts/mV
    # Sample value 3048 should equal (3048 - 2048) / 1000 = 1.0 mV
    raw_signal = np.full((12, 500), 3048.0)
    calibrated, meta = calibrator.calibrate(raw_signal, adc_gain=1000.0, baseline=2048.0)

    np.testing.assert_allclose(calibrated, 1.0, atol=1e-5)
    assert meta["calibrated_min_mv"] == pytest.approx(1.0)


def test_unit_calibration_zscore():
    """Verify lead-wise z-score normalization."""
    calibrator = UnitCalibrator(scaling_method="zscore")
    rng = np.random.default_rng(42)
    signal = rng.normal(loc=5.0, scale=2.0, size=(12, 1000))
    calibrated, meta = calibrator.calibrate(signal)

    means = np.mean(calibrated, axis=1)
    stds = np.std(calibrated, axis=1)
    np.testing.assert_allclose(means, 0.0, atol=1e-2)
    np.testing.assert_allclose(stds, 1.0, atol=1e-2)


# =====================================================================
# P_4: Filtering Tests
# =====================================================================
def test_filtering_baseline_wander_and_powerline():
    """Verify baseline drift and 50 Hz powerline interference attenuation."""
    filt = SignalFilter(highpass_cutoff=0.5, notch_freqs=[50.0], notch_q=30.0)
    fs = 500.0
    t = np.linspace(0, 10.0, 5000, endpoint=False)
    
    # Construct synthetic signal:
    # 0.1 Hz baseline drift + 1.0 Hz ECG proxy + 50 Hz powerline hum
    baseline_drift = 2.0 * np.sin(2 * np.pi * 0.1 * t)
    cardiac_proxy = 1.0 * np.sin(2 * np.pi * 1.5 * t)
    powerline_noise = 0.5 * np.sin(2 * np.pi * 50.0 * t)
    
    noisy_signal = np.tile(baseline_drift + cardiac_proxy + powerline_noise, (12, 1))
    filtered, meta = filt.filter_signal(noisy_signal, fs=fs)
    
    assert filtered.shape == (12, 5000)
    # The 0.1 Hz drift and 50 Hz hum should be significantly reduced
    # Peak-to-peak amplitude should be lower than raw noisy signal
    assert np.std(filtered[0, :]) < np.std(noisy_signal[0, :])


# =====================================================================
# P_5: Quality Screening Tests
# =====================================================================
def test_quality_screening_clean_pass():
    """Verify clean physiological signal passes quality audit."""
    screener = QualityScreening(min_channel_std_mv=1e-4)
    rng = np.random.default_rng(123)
    clean_sig = rng.normal(0, 0.5, size=(12, 1000))
    audit = screener.audit(clean_sig, CANONICAL_LEADS, fs=100.0)
    assert audit.passed is True
    assert len(audit.rejection_reasons) == 0


def test_quality_screening_flatline_detection():
    """Verify flatlined lead is flagged and causes rejection."""
    screener = QualityScreening(min_channel_std_mv=1e-4)
    rng = np.random.default_rng(123)
    signal = rng.normal(0, 0.5, size=(12, 1000))
    # Flatline lead 'V3' (index 8)
    signal[8, :] = 0.0
    audit = screener.audit(signal, CANONICAL_LEADS, fs=100.0)
    assert audit.passed is False
    assert any("V3" in r and "Flatline" in r for r in audit.rejection_reasons)


def test_quality_screening_nan_inf_detection():
    """Verify non-finite values (NaN / Inf) trigger rejection."""
    screener = QualityScreening()
    signal = np.ones((12, 1000))
    signal[0, 50] = np.nan
    signal[2, 100] = np.inf
    audit = screener.audit(signal, CANONICAL_LEADS, fs=100.0)
    assert audit.passed is False
    assert any("non-finite" in r for r in audit.rejection_reasons)


def test_quality_screening_saturation():
    """Verify amplifier saturation / voltage clipping triggers rejection."""
    screener = QualityScreening(max_valid_amplitude_mv=15.0)
    signal = np.ones((12, 1000))
    signal[1, 200] = 25.0  # Excessive voltage spike
    audit = screener.audit(signal, CANONICAL_LEADS, fs=100.0)
    assert audit.passed is False
    assert any("Saturation" in r for r in audit.rejection_reasons)


# =====================================================================
# End-to-End Pipeline & PTB-XL Dataset Integration Tests
# =====================================================================
def test_full_pipeline_synthetic():
    """Verify end-to-end pipeline execution P_1 through P_5 on synthetic input."""
    pipeline = ECGPreprocessingPipeline(target_fs=100.0, target_duration_sec=10.0)
    rng = np.random.default_rng(99)
    raw_signal = rng.normal(0.0, 0.5, size=(12, 1200))  # 12s at 100 Hz
    lead_names = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    
    result = pipeline.process(raw_signal, lead_names, fs_in=100.0)
    assert isinstance(result, PreprocessedECG)
    assert result.signal.shape == (12, 1000)
    assert result.leads == CANONICAL_LEADS
    assert result.is_valid is True
    assert "p1_lead_ordering" in result.pipeline_metadata
    assert "p5_quality_screening" in result.pipeline_metadata


def test_real_ptbxl_record_processing():
    """Verify Phase 1 preprocessing on actual PTB-XL record from local repository."""
    ptbxl_path = Path("ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
    if not ptbxl_path.exists():
        pytest.skip("Local PTB-XL dataset directory not found.")
        
    loader = PTBXLDatasetLoader(dataset_root=str(ptbxl_path), sampling_rate=100)
    # Get first ecg_id in database
    first_id = int(loader.df.index[0])
    record = loader.load_record(first_id)

    assert isinstance(record, PreprocessedECG)
    assert record.signal.shape == (12, 1000)
    assert record.leads == CANONICAL_LEADS
    assert record.fs == 100.0
    assert record.duration_sec == 10.0
    assert record.is_valid is True
    assert len(record.audit.rejection_reasons) == 0

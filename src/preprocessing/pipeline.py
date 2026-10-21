"""
ECG Preprocessing Pipeline (Phase 1 Master Orchestrator)
--------------------------------------------------------
Executes the sequential, theoretical operations to standardize heterogeneous
raw physiological recordings into uniform (12, 1000) tensors at 100 Hz.

CORRECTED PREPROCESSING EXECUTION SEQUENCE (Critical Fix 1):
1. Unit Calibration (P_3): ADC machine counts -> physical millivolts (mV).
2. Canonical Lead Ordering (P_1): Spatial alignment to Cabrera sequence.
3. Pre-filtering Quality Screening: Fast check for valid dimensions and finite values.
4. Mathematical Filtering (P_4) AT NATIVE RATE (e.g. 500 Hz):
   - 0.5 Hz highpass (baseline wander)
   - Dataset-aware 50 Hz / 60 Hz IIR notch filter (mains powerline interference)
   - 45.0 Hz lowpass anti-aliasing filter
   Notice: Operates at native rate so 50/60 Hz resides far below Nyquist (250 Hz).
5. Sampling & Duration Standardization (P_2):
   - Polyphase downsampling from 500 Hz -> 100 Hz via `scipy.signal.resample_poly`.
   - Fixed temporal windowing to 10.0 seconds (exactly 1,000 samples).
6. Quality Screening and Integrity Audit (P_5):
   - Final audit on standardized 100 Hz signal (flatline, rails, non-finite values).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
import numpy as np

from .lead_ordering import LeadOrdering, CANONICAL_LEADS
from .sampling_standardization import SamplingStandardizer
from .unit_calibration import UnitCalibrator
from .filtering import SignalFilter
from .quality_screening import QualityScreening, QualityAuditResult


@dataclass
class PreprocessedECG:
    """Standardized physiological ECG recording ready for downstream representation and reasoning."""
    signal: np.ndarray
    leads: List[str]
    fs: float
    duration_sec: float
    audit: QualityAuditResult
    pipeline_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Returns True if the recording passed Phase 1 quality screening."""
        return self.audit.passed


class ECGPreprocessingPipeline:
    """
    Master Phase 1 Preprocessing Pipeline orchestrating P_1 through P_5 in corrected sequence.
    """

    def __init__(
        self,
        target_fs: float = 100.0,
        target_duration_sec: float = 10.0,
        scaling_method: str = "none",
        highpass_cutoff: float = 0.5,
        lowpass_cutoff: Optional[float] = 45.0,
        notch_freqs: Optional[List[float]] = None,
        dataset_name: Optional[str] = None,
        min_channel_std_mv: float = 1e-4,
        max_valid_amplitude_mv: float = 15.0,
        resample_method: str = "polyphase"
    ):
        """
        Initializes pipeline components with standardized parameters.
        """
        self.target_fs = target_fs
        self.target_duration_sec = target_duration_sec
        self.dataset_name = dataset_name

        # P_1: Lead Ordering
        self.p1_lead_ordering = LeadOrdering(canonical_leads=list(CANONICAL_LEADS))

        # P_2: Sampling and Duration Standardization (applied AFTER 500 Hz filtering)
        self.p2_sampling = SamplingStandardizer(
            target_fs=target_fs,
            target_duration_sec=target_duration_sec,
            windowing_mode="center",
            resample_method=resample_method
        )

        # P_3: Unit Calibration and Normalization
        self.p3_calibration = UnitCalibrator(
            scaling_method=scaling_method,
            clip_range_mv=(-max_valid_amplitude_mv, max_valid_amplitude_mv)
        )

        # P_4: Filtering (executed at native raw sampling rate, e.g. 500 Hz)
        self.p4_filtering = SignalFilter(
            highpass_cutoff=highpass_cutoff,
            lowpass_cutoff=lowpass_cutoff,
            notch_freqs=notch_freqs,
            dataset_name=dataset_name
        )

        # P_5: Final Quality Screening on standardized (12, 1000) array
        self.p5_screening = QualityScreening(
            min_channel_std_mv=min_channel_std_mv,
            max_valid_amplitude_mv=max_valid_amplitude_mv,
            expected_channels=12,
            expected_samples=int(round(target_fs * target_duration_sec))
        )

    def process(
        self,
        raw_signal: np.ndarray,
        lead_names: List[str],
        fs_in: float,
        adc_gain: Optional[Union[float, List[float], np.ndarray]] = None,
        baseline: Optional[Union[float, List[float], np.ndarray]] = None,
        units: Optional[Union[str, List[str]]] = "mV",
        is_digital_counts: Optional[bool] = None,
        dataset_name: Optional[str] = None
    ) -> PreprocessedECG:
        """
        Executes corrected preprocessing sequence:
        Raw -> P_3 (Calibration) -> P_1 (Ordering) -> P_4 (Filtering at fs_in) -> P_2 (Polyphase Downsampling) -> P_5 (Validation).
        """
        pipeline_log: Dict[str, Any] = {}
        effective_dataset = dataset_name or self.dataset_name

        # ------------------------------------------------------------------
        # Step 1: Unit Calibration & Voltage Conversion (P_3) at native fs
        # ------------------------------------------------------------------
        calibrated_signal, p3_meta = self.p3_calibration.calibrate(
            signal=raw_signal,
            adc_gain=adc_gain,
            baseline=baseline,
            units=units,
            is_digital_counts=is_digital_counts,
            dataset_name=effective_dataset
        )
        pipeline_log["step1_unit_calibration"] = p3_meta
        pipeline_log["p3_unit_calibration"] = p3_meta

        # ------------------------------------------------------------------
        # Step 2: Canonical Spatial Lead Ordering (P_1)
        # ------------------------------------------------------------------
        ordered_signal, canonical_leads = self.p1_lead_ordering.reorder(
            signal=calibrated_signal, lead_names=lead_names
        )
        p1_info = {
            "input_channels": lead_names,
            "output_canonical_leads": canonical_leads,
            "shape_after_p1": list(ordered_signal.shape),
        }
        pipeline_log["step2_lead_ordering"] = p1_info
        pipeline_log["p1_lead_ordering"] = p1_info

        # ------------------------------------------------------------------
        # Step 3: Mathematical Filtering at Native fs (P_4)
        # Filters at native 500 Hz (0.5 Hz highpass, 50/60 Hz notch, 45 Hz lowpass)
        # ------------------------------------------------------------------
        filtered_signal, p4_meta = self.p4_filtering.filter_signal(
            signal=ordered_signal, fs=fs_in
        )
        pipeline_log["step3_filtering_at_native_fs"] = p4_meta
        pipeline_log["p4_filtering"] = p4_meta

        # ------------------------------------------------------------------
        # Step 4: Polyphase Downsampling & Temporal Windowing (P_2)
        # Decimates cleanly from fs_in (e.g. 500 Hz) to target_fs (100 Hz)
        # ------------------------------------------------------------------
        resampled_signal, p2_meta = self.p2_sampling.standardize(
            signal=filtered_signal, fs_in=fs_in
        )
        pipeline_log["step4_sampling_and_duration"] = p2_meta
        pipeline_log["p2_sampling_and_duration"] = p2_meta

        # ------------------------------------------------------------------
        # Step 5: Final Quality Screening & Integrity Audit (P_5)
        # ------------------------------------------------------------------
        audit_result = self.p5_screening.audit(
            signal=resampled_signal,
            lead_names=canonical_leads,
            fs=self.target_fs
        )
        pipeline_log["step5_quality_screening"] = audit_result.to_dict()
        pipeline_log["p5_quality_screening"] = audit_result.to_dict()

        return PreprocessedECG(
            signal=resampled_signal,
            leads=canonical_leads,
            fs=self.target_fs,
            duration_sec=self.target_duration_sec,
            audit=audit_result,
            pipeline_metadata=pipeline_log
        )

"""
Unit Calibration and Normalization Module (P_3)
-----------------------------------------------
Converts raw device outputs (ADC machine counts) into true physical voltage units
(millivolts, mV) using hardware-specific gains and baseline offsets, followed
by an optional configurable amplitude scaling process.

ADC TO mV CONVERSION SPECIFICATION (Critical Fix 8):
- If raw signal is in digital machine counts:
    V_mV = (raw_counts - baseline) / adc_gain
- If raw signal is already loaded as physical voltage by WFDB reader (e.g. units='mV'):
    V_mV = raw_signal (calibrated pass-through)

AMPLITUDE CLAMP AUDIT SPECIFICATION (Critical Fix 9):
The [-15.0, +15.0] mV range acts as an amplifier rail guard and saturation detection mechanism.
Normal physiological surface ECG potentials rarely exceed 3-5 mV (even in severe left ventricular
hypertrophy). Values exceeding 15 mV represent hardware rail clipping, electrode motion artifact,
or defibrillator discharge.
"""

from typing import Tuple, Dict, Any, Optional, Union, List
import numpy as np


class UnitCalibrator:
    """
    P_3: Unit Calibration and Amplitude Normalization Component.
    Transforms raw ADC counts into calibrated millivolts (mV), ensuring
    physical consistency across diverse hardware acquisition apparatus.
    """

    def __init__(
        self,
        scaling_method: str = "none",
        clip_range_mv: Optional[Tuple[float, float]] = (-15.0, 15.0),
        eps: float = 1e-8
    ):
        """
        Args:
            scaling_method: 'none' (retains physical mV), 'zscore' (standardizes each lead),
                            or 'robust_zscore' (median and IQR).
            clip_range_mv: Optional physical clamp tuple (min_mv, max_mv) to eliminate
                           unrealistic amplifier rail spikes. Default: (-15.0, 15.0) mV.
            eps: Epsilon for numerical division stability.
        """
        if scaling_method not in ("none", "zscore", "robust_zscore"):
            raise ValueError(
                f"scaling_method must be 'none', 'zscore', or 'robust_zscore', got '{scaling_method}'"
            )
        self.scaling_method = scaling_method
        self.clip_range_mv = clip_range_mv
        self.eps = eps

    def calibrate(
        self,
        signal: np.ndarray,
        adc_gain: Optional[Union[float, List[float], np.ndarray]] = None,
        baseline: Optional[Union[float, List[float], np.ndarray]] = None,
        units: Optional[Union[str, List[str]]] = "mV",
        is_digital_counts: Optional[bool] = None,
        dataset_name: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Calibrates raw recording into physical mV and applies normalization.

        Args:
            signal: numpy array of shape (12, T).
            adc_gain: ADC counts per physical unit (e.g. 1000 counts/mV). If None or 0,
                      assumed already in physical units.
            baseline: ADC baseline offset.
            units: Unit string or list of unit strings per channel.
            is_digital_counts: If True, explicitly applies (signal - baseline) / adc_gain.
                               If False, assumes signal is already in physical mV.
                               If None, auto-detects based on amplitude range and units.
            dataset_name: Optional dataset identifier.

        Returns:
            Tuple of (calibrated_signal of shape (12, T), calibration_metadata).
        """
        if signal.ndim != 2:
            raise ValueError(f"Expected 2D array of shape (12, T), got {signal.shape}")

        channels, num_samples = signal.shape
        signal_mv = signal.astype(np.float64, copy=True)

        # 1. ADC Count to Physical Millivolts Conversion
        applied_gain: np.ndarray = np.ones((channels, 1), dtype=np.float64)
        applied_baseline: np.ndarray = np.zeros((channels, 1), dtype=np.float64)

        if adc_gain is not None:
            gains = np.asarray(adc_gain, dtype=np.float64)
            if gains.ndim == 0:
                gains = np.full((channels, 1), gains)
            elif gains.ndim == 1:
                gains = gains.reshape((channels, 1))
            gains[gains == 0] = 1.0
            applied_gain = gains

        if baseline is not None:
            baselines = np.asarray(baseline, dtype=np.float64)
            if baselines.ndim == 0:
                baselines = np.full((channels, 1), baselines)
            elif baselines.ndim == 1:
                baselines = baselines.reshape((channels, 1))
            applied_baseline = baselines

        # Determine whether input is digital counts or already physical mV
        max_abs_val = float(np.nanmax(np.abs(signal))) if np.any(np.isfinite(signal)) else 0.0

        # Check unit strings
        unit_str = ""
        if isinstance(units, list) and len(units) > 0:
            unit_str = str(units[0]).lower()
        elif isinstance(units, str):
            unit_str = units.lower()

        if is_digital_counts is None:
            if unit_str in ("mv", "millivolt", "millivolts") and max_abs_val < 50.0:
                # WFDB rdsamp already converted digital counts to physical mV
                is_digital_counts = False
            elif max_abs_val > 50.0:
                # Raw integer counts from ADC hardware (e.g. 1000 - 4096)
                is_digital_counts = True
            else:
                is_digital_counts = False

        conversion_applied = False
        formula_used = "pass_through_already_physical_mV"
        if is_digital_counts and adc_gain is not None:
            signal_mv = (signal_mv - applied_baseline) / applied_gain
            conversion_applied = True
            formula_used = "mV = (raw_counts - baseline) / adc_gain"

        # 2. Physical Clipping Audit & Safety Rail Clamping (Critical Fix 9)
        pre_clamp_min = float(np.nanmin(signal_mv)) if np.any(np.isfinite(signal_mv)) else 0.0
        pre_clamp_max = float(np.nanmax(signal_mv)) if np.any(np.isfinite(signal_mv)) else 0.0
        clamped_samples_count = 0
        clamped_leads_list: List[int] = []

        if self.clip_range_mv is not None:
            min_val, max_val = self.clip_range_mv
            below_mask = signal_mv < min_val
            above_mask = signal_mv > max_val
            clamped_mask = below_mask | above_mask
            clamped_samples_count = int(np.sum(clamped_mask))
            if clamped_samples_count > 0:
                clamped_leads_list = [ch for ch in range(channels) if np.any(clamped_mask[ch])]
            signal_mv = np.clip(signal_mv, min_val, max_val)

        post_clamp_min = float(np.nanmin(signal_mv)) if np.any(np.isfinite(signal_mv)) else 0.0
        post_clamp_max = float(np.nanmax(signal_mv)) if np.any(np.isfinite(signal_mv)) else 0.0

        # 3. Amplitude Scaling
        scaling_params: Dict[str, Any] = {}
        if self.scaling_method == "none":
            scaled_signal = signal_mv
        elif self.scaling_method == "zscore":
            means = np.mean(signal_mv, axis=1, keepdims=True)
            stds = np.std(signal_mv, axis=1, keepdims=True)
            stds[stds < self.eps] = 1.0
            scaled_signal = (signal_mv - means) / stds
            scaling_params = {"means": means.flatten().tolist(), "stds": stds.flatten().tolist()}
        elif self.scaling_method == "robust_zscore":
            medians = np.median(signal_mv, axis=1, keepdims=True)
            q75 = np.percentile(signal_mv, 75, axis=1, keepdims=True)
            q25 = np.percentile(signal_mv, 25, axis=1, keepdims=True)
            iqr = q75 - q25
            iqr[iqr < self.eps] = 1.0
            scaled_signal = (signal_mv - medians) / iqr
            scaling_params = {"medians": medians.flatten().tolist(), "iqr": iqr.flatten().tolist()}

        metadata = {
            "dataset_name": dataset_name,
            "units_before": units,
            "units_after": "mV",
            "conversion_formula": formula_used,
            "raw_min": float(np.nanmin(signal)) if np.any(np.isfinite(signal)) else 0.0,
            "raw_max": float(np.nanmax(signal)) if np.any(np.isfinite(signal)) else 0.0,
            "is_digital_counts": is_digital_counts,
            "adc_conversion_applied": conversion_applied,
            "calibrated_min_mv": post_clamp_min,
            "calibrated_max_mv": post_clamp_max,
            "pre_clamp_min_mv": pre_clamp_min,
            "pre_clamp_max_mv": pre_clamp_max,
            "post_clamp_min_mv": post_clamp_min,
            "post_clamp_max_mv": post_clamp_max,
            "clip_range_mv": self.clip_range_mv,
            "clamped_samples_count": clamped_samples_count,
            "clamped_leads": clamped_leads_list,
            "scaling_method": self.scaling_method,
            "scaling_params": scaling_params,
        }

        return scaled_signal.astype(np.float32), metadata

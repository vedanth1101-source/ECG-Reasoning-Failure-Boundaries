"""
Deterministic Clinical Observation Extractor (Phase 3 Subsystem B)
===================================================================
Extracts discrete structured physiological observations O_l strictly from
the observable leads at degradation level l.

Schema Compliance (Section 3.2):
{
  "rhythm": "Sinus rhythm",
  "heart_rate": 72,
  "pr_interval_ms": 210,
  "qrs_duration_ms": 142,
  "st_elevation": {
    "present": true,
    "leads": ["II", "III", "aVF"],
    "amplitude_mm": { "II": 2.0, "III": 1.8, "aVF": 2.1 }
  },
  "st_depression": {
    "present": false,
    "leads": [],
    "amplitude_mm": {}
  },
  "t_wave_inversion": {
    "present": false,
    "leads": []
  }
}

CRITICAL RESEARCH CONSTRAINT:
Under NO circumstances are unobserved leads imputed or evaluated.
If a lead is masked (mask[lead] == 0), it is strictly excluded from observation.
"""

import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import scipy.signal as signal

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.degradation.lead_masking import CANONICAL_LEADS, LEAD_TO_INDEX


@dataclass
class STObservation:
    present: bool
    leads: List[str]
    amplitude_mm: Dict[str, float]


@dataclass
class TWaveObservation:
    present: bool
    leads: List[str]


@dataclass
class ClinicalObservation:
    rhythm: str
    heart_rate: Optional[int]
    pr_interval_ms: Optional[int]
    qrs_duration_ms: Optional[int]
    st_elevation: STObservation
    st_depression: STObservation
    t_wave_inversion: TWaveObservation

    def to_dict(self) -> Dict[str, Any]:
        """Returns JSON-serializable dictionary matching Section 3.2."""
        return {
            "rhythm": self.rhythm,
            "heart_rate": self.heart_rate,
            "pr_interval_ms": self.pr_interval_ms,
            "qrs_duration_ms": self.qrs_duration_ms,
            "st_elevation": {
                "present": self.st_elevation.present,
                "leads": self.st_elevation.leads,
                "amplitude_mm": self.st_elevation.amplitude_mm,
            },
            "st_depression": {
                "present": self.st_depression.present,
                "leads": self.st_depression.leads,
                "amplitude_mm": self.st_depression.amplitude_mm,
            },
            "t_wave_inversion": {
                "present": self.t_wave_inversion.present,
                "leads": self.t_wave_inversion.leads,
            },
        }


class ClinicalObservationExtractor:
    """
    Deterministic rule-based clinical observation extractor.

    Args:
        fs: Sampling frequency in Hz (default: 100.0 Hz).
        st_elevation_threshold_mv: Voltage threshold in mV for ST elevation (default: 0.10 mV = 1.0 mm).
        st_depression_threshold_mv: Voltage threshold in mV for ST depression (default: -0.05 mV = -0.5 mm).
    """

    def __init__(
        self,
        fs: float = 100.0,
        st_elevation_threshold_mv: float = 0.10,
        st_depression_threshold_mv: float = -0.05,
    ) -> None:
        self.fs = float(fs)
        self.st_elev_thresh = float(st_elevation_threshold_mv)
        self.st_dep_thresh = float(st_depression_threshold_mv)

    def extract(
        self,
        ecg_signal: np.ndarray,
        lead_mask: Optional[Union[np.ndarray, List[str], Tuple[str, ...]]] = None,
    ) -> ClinicalObservation:
        """
        Extracts structured observations from visible channels.

        Args:
            ecg_signal: NumPy array of shape (12, T).
            lead_mask: Binary mask vector (12,) or list/tuple of active lead names.
                       If None, all 12 leads are considered active.

        Returns:
            ClinicalObservation dataclass instance.
        """
        if ecg_signal.ndim != 2 or ecg_signal.shape[0] != 12:
            raise ValueError(f"ecg_signal must have shape (12, T), got {ecg_signal.shape}")

        # Resolve active leads
        active_leads = self._resolve_active_leads(lead_mask)
        if not active_leads:
            # Degraded to zero leads: emit unavailable
            return ClinicalObservation(
                rhythm="Unavailable (No active leads)",
                heart_rate=None,
                pr_interval_ms=None,
                qrs_duration_ms=None,
                st_elevation=STObservation(present=False, leads=[], amplitude_mm={}),
                st_depression=STObservation(present=False, leads=[], amplitude_mm={}),
                t_wave_inversion=TWaveObservation(present=False, leads=[]),
            )

        # 1. Select best lead for global rhythm and interval metrics (Lead II preferred, then V1, then first available)
        rhythm_lead_name = self._select_rhythm_lead(active_leads)
        rhythm_lead_idx = LEAD_TO_INDEX[rhythm_lead_name]
        rhythm_sig = ecg_signal[rhythm_lead_idx]

        # 2. Extract Heart Rate and Rhythm
        heart_rate, r_peaks, rr_intervals = self._compute_heart_rate(rhythm_sig)
        rhythm = self._classify_rhythm(heart_rate, rr_intervals)

        # 3. Extract PR Interval and QRS Duration
        pr_interval_ms, qrs_duration_ms = self._compute_intervals(rhythm_sig, r_peaks)

        # 4. Extract ST-Segment Deviations across ALL ACTIVE LEADS
        st_elev_leads, st_elev_amps, st_dep_leads, st_dep_amps = self._compute_st_deviations(
            ecg_signal, active_leads, r_peaks
        )

        # 5. Extract T-Wave Inversions across ACTIVE LEADS
        t_inv_leads = self._compute_t_wave_inversions(ecg_signal, active_leads, r_peaks)

        return ClinicalObservation(
            rhythm=rhythm,
            heart_rate=heart_rate,
            pr_interval_ms=pr_interval_ms,
            qrs_duration_ms=qrs_duration_ms,
            st_elevation=STObservation(
                present=len(st_elev_leads) > 0,
                leads=st_elev_leads,
                amplitude_mm=st_elev_amps,
            ),
            st_depression=STObservation(
                present=len(st_dep_leads) > 0,
                leads=st_dep_leads,
                amplitude_mm=st_dep_amps,
            ),
            t_wave_inversion=TWaveObservation(
                present=len(t_inv_leads) > 0,
                leads=t_inv_leads,
            ),
        )

    def _resolve_active_leads(
        self,
        lead_mask: Optional[Union[np.ndarray, List[str], Tuple[str, ...]]],
    ) -> List[str]:
        if lead_mask is None:
            return list(CANONICAL_LEADS)
        if isinstance(lead_mask, (list, tuple)):
            return [lead for lead in lead_mask if lead in LEAD_TO_INDEX]
        if isinstance(lead_mask, np.ndarray):
            if lead_mask.ndim == 1 and len(lead_mask) == 12:
                return [CANONICAL_LEADS[i] for i in range(12) if lead_mask[i] > 0]
        return list(CANONICAL_LEADS)

    def _select_rhythm_lead(self, active_leads: List[str]) -> str:
        for preferred in ("II", "V1", "I", "V5"):
            if preferred in active_leads:
                return preferred
        return active_leads[0]

    def _compute_heart_rate(self, sig: np.ndarray) -> Tuple[Optional[int], np.ndarray, np.ndarray]:
        """Detects R-peaks and computes resting heart rate in bpm."""
        if np.all(sig == 0) or np.std(sig) < 1e-4:
            return None, np.array([]), np.array([])

        # Bandpass filter for QRS energy enhancement (5 to 15 Hz)
        sos = signal.butter(2, [5.0, 15.0], btype="bandpass", fs=self.fs, output="sos")
        filtered = signal.sosfiltfilt(sos, sig)
        diff = np.diff(filtered)
        squared = diff ** 2

        # Moving integration window (100 ms)
        win_size = max(int(0.1 * self.fs), 3)
        integrated = np.convolve(squared, np.ones(win_size) / win_size, mode="same")

        # Threshold detection
        thresh = np.mean(integrated) + 1.5 * np.std(integrated)
        min_distance = int(0.25 * self.fs)  # Max 240 bpm

        peaks, _ = signal.find_peaks(integrated, height=thresh, distance=min_distance)

        # Refine peaks to local maxima in original signal
        refined_peaks = []
        search_radius = int(0.05 * self.fs)  # 50 ms window
        for p in peaks:
            start = max(0, p - search_radius)
            end = min(len(sig), p + search_radius + 1)
            refined_peaks.append(start + np.argmax(sig[start:end]))

        refined_peaks = np.unique(refined_peaks)
        if len(refined_peaks) < 2:
            return None, refined_peaks, np.array([])

        rr_samples = np.diff(refined_peaks)
        rr_sec = rr_samples / self.fs
        mean_rr = float(np.median(rr_sec))

        if mean_rr <= 0:
            return None, refined_peaks, rr_sec

        hr = int(round(60.0 / mean_rr))
        # Physiological bounding
        hr = max(30, min(hr, 250))
        return hr, refined_peaks, rr_sec

    def _classify_rhythm(self, hr: Optional[int], rr_sec: np.ndarray) -> str:
        if hr is None or len(rr_sec) < 2:
            return "Indeterminate rhythm"

        rr_cv = float(np.std(rr_sec) / np.mean(rr_sec)) if np.mean(rr_sec) > 0 else 0.0

        if rr_cv > 0.20:
            return "Atrial Fibrillation / Irregular rhythm"
        if hr < 60:
            return "Sinus Bradycardia"
        if hr > 100:
            return "Sinus Tachycardia"
        return "Sinus rhythm"

    def _compute_intervals(
        self, sig: np.ndarray, r_peaks: np.ndarray
    ) -> Tuple[Optional[int], Optional[int]]:
        """Computes PR interval and QRS duration in milliseconds."""
        if len(r_peaks) < 2:
            return None, None

        # Sample around median valid peak
        qrs_durations = []
        pr_intervals = []

        for r in r_peaks[1:-1]:
            # QRS window: [-60ms, +60ms]
            left = max(0, r - int(0.06 * self.fs))
            right = min(len(sig), r + int(0.06 * self.fs))

            q_candidate = left + np.argmin(sig[left:r]) if r > left else left
            s_candidate = r + np.argmin(sig[r:right]) if right > r else r

            qrs_samples = max(s_candidate - q_candidate, int(0.04 * self.fs))
            qrs_durations.append(qrs_samples * (1000.0 / self.fs))

            # PR window: [r - 250ms, r - 60ms]
            p_start = max(0, r - int(0.25 * self.fs))
            p_end = max(0, r - int(0.06 * self.fs))
            if p_end > p_start:
                p_peak = p_start + np.argmax(sig[p_start:p_end])
                pr_samples = r - p_peak
                pr_intervals.append(pr_samples * (1000.0 / self.fs))

        qrs_ms = int(round(np.median(qrs_durations))) if qrs_durations else 90
        qrs_ms = max(50, min(qrs_ms, 220))

        pr_ms = int(round(np.median(pr_intervals))) if pr_intervals else 160
        pr_ms = max(100, min(pr_ms, 350))

        return pr_ms, qrs_ms

    def _compute_st_deviations(
        self,
        ecg_signal: np.ndarray,
        active_leads: List[str],
        r_peaks: np.ndarray,
    ) -> Tuple[List[str], Dict[str, float], List[str], Dict[str, float]]:
        """Computes ST elevation and depression strictly for active leads."""
        st_elev_leads = []
        st_elev_amps = {}
        st_dep_leads = []
        st_dep_amps = {}

        if len(r_peaks) < 2:
            return st_elev_leads, st_elev_amps, st_dep_leads, st_dep_amps

        j_offset = int(0.06 * self.fs)   # 60 ms after R peak (J-point)
        base_offset = int(0.12 * self.fs) # 120 ms before R peak (PR isoelectric segment)

        for lead_name in active_leads:
            idx = LEAD_TO_INDEX[lead_name]
            sig = ecg_signal[idx]

            st_diffs = []
            for r in r_peaks[1:-1]:
                j_idx = min(len(sig) - 1, r + j_offset)
                base_idx = max(0, r - base_offset)
                st_diffs.append(sig[j_idx] - sig[base_idx])

            if not st_diffs:
                continue

            median_st_mv = float(np.median(st_diffs))
            # Convert mV to mm (Standard 10 mm / mV calibration)
            st_mm = round(median_st_mv * 10.0, 2)

            if median_st_mv >= self.st_elev_thresh:
                st_elev_leads.append(lead_name)
                st_elev_amps[lead_name] = max(st_mm, 1.0)
            elif median_st_mv <= self.st_dep_thresh:
                st_dep_leads.append(lead_name)
                st_dep_amps[lead_name] = min(st_mm, -0.5)

        return st_elev_leads, st_elev_amps, st_dep_leads, st_dep_amps

    def _compute_t_wave_inversions(
        self,
        ecg_signal: np.ndarray,
        active_leads: List[str],
        r_peaks: np.ndarray,
    ) -> List[str]:
        """Detects pathological T-wave inversions in active leads."""
        t_inv_leads = []
        if len(r_peaks) < 2:
            return t_inv_leads

        t_window = int(0.20 * self.fs)  # 200 ms after R peak

        for lead_name in active_leads:
            idx = LEAD_TO_INDEX[lead_name]
            sig = ecg_signal[idx]

            t_amplitudes = []
            for r in r_peaks[1:-1]:
                t_idx = min(len(sig) - 1, r + t_window)
                base_idx = max(0, r - int(0.12 * self.fs))
                t_amplitudes.append(sig[t_idx] - sig[base_idx])

            if t_amplitudes and np.median(t_amplitudes) < -0.10:
                t_inv_leads.append(lead_name)

        return t_inv_leads

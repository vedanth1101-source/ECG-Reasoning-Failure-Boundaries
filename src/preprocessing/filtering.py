"""
Filtering Module (P_4)
----------------------
Applies mathematical digital signal processing to clean the ECG waveform at native sampling rate (e.g., 500 Hz):
1. Baseline wander removal via high-pass zero-phase Butterworth filtering (0.5 Hz).
2. Powerline electrical interference elimination via dataset-aware IIR notch filtering (50 Hz for PTB-XL/Chapman, 60 Hz for MIMIC).
3. High-frequency artifact and EMG noise suppression via low-pass zero-phase Butterworth anti-aliasing filtering (45 Hz).

CRITICAL MORPHOLOGICAL NOTICE (Critical Fix 10):
filtfilt applies forward-backward filtering and avoids phase shift / temporal displacement
associated with causal one-direction filtering, but it does not guarantee exact preservation
of ECG amplitude or morphology. Filter order, cutoff frequencies, and boundary transient conditions
can subtly alter sharp fiducial peaks (e.g., R-peak amplitudes) and low-frequency ST-segment baselines.
"""

from typing import Tuple, Dict, Any, List, Optional, Union
import numpy as np
from scipy import signal as sp_signal


class SignalFilter:
    """
    P_4: Mathematical ECG Signal Filtering Component.
    Applies zero-phase digital filtering at native sampling frequency (prior to downsampling)
    to suppress baseline wander, powerline hum, and high-frequency noise.
    """

    def __init__(
        self,
        highpass_cutoff: float = 0.5,
        lowpass_cutoff: Optional[float] = 45.0,
        notch_freqs: Optional[List[float]] = None,
        dataset_name: Optional[str] = None,
        notch_q: float = 30.0,
        highpass_order: int = 3,
        lowpass_order: int = 4
    ):
        """
        Args:
            highpass_cutoff: Low cutoff frequency in Hz for baseline wander (default: 0.5 Hz).
            lowpass_cutoff: High cutoff frequency in Hz for high-frequency noise and anti-aliasing (default: 45.0 Hz).
            notch_freqs: Explicit list of powerline notch frequencies in Hz.
                         If None and dataset_name provided:
                           - 'ptbxl' or 'chapman' -> [50.0]
                           - 'mimic' -> [60.0]
                         If both None, defaults to [50.0, 60.0].
            dataset_name: Optional dataset identifier ('ptbxl', 'chapman', 'mimic') for metadata-driven notch selection.
            notch_q: Quality factor Q for the IIR notch filter (default: 30.0).
            highpass_order: Butterworth filter order for highpass stage (default: 3).
            lowpass_order: Butterworth filter order for lowpass stage (default: 4).
        """
        self.highpass_cutoff = highpass_cutoff
        self.lowpass_cutoff = lowpass_cutoff
        self.notch_q = notch_q
        self.highpass_order = highpass_order
        self.lowpass_order = lowpass_order

        # Dataset-aware notch configuration (Critical Fix 1)
        if notch_freqs is not None:
            self.notch_freqs = [float(f) for f in notch_freqs]
        elif dataset_name is not None:
            dname = dataset_name.lower()
            if "mimic" in dname:
                self.notch_freqs = [60.0]  # US mains frequency (Beth Israel Deaconess Medical Center, Boston)
            elif "ptb" in dname:
                self.notch_freqs = [50.0]  # European mains frequency (Germany)
            elif "chapman" in dname:
                self.notch_freqs = [50.0]  # Chinese mains frequency (Shaoxing Hospital, China)
            else:
                self.notch_freqs = [50.0, 60.0]
        else:
            self.notch_freqs = [50.0, 60.0]

    def filter_signal(
        self, signal: np.ndarray, fs: float
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Applies sequential zero-phase filtering across all leads.
        Must be executed at the raw sampling rate (e.g. fs=500 Hz) before downsampling to 100 Hz.

        Args:
            signal: numpy array of shape (12, T).
            fs: Current sampling frequency in Hz (must be > 2 * max(notch_freqs, lowpass_cutoff)).

        Returns:
            Tuple of (filtered_signal of shape (12, T), filtering_metadata).
        """
        if signal.ndim != 2:
            raise ValueError(f"Expected 2D array of shape (12, T), got {signal.shape}")
        if fs <= 0:
            raise ValueError(f"Sampling frequency must be positive, got {fs}")

        nyquist = 0.5 * fs
        filtered = signal.astype(np.float64, copy=True)
        applied_stages: List[str] = []

        # 1. High-Pass Filter: Baseline Wander Removal (0.5 Hz)
        if self.highpass_cutoff > 0 and self.highpass_cutoff < nyquist:
            b_hp, a_hp = sp_signal.butter(
                self.highpass_order,
                self.highpass_cutoff / nyquist,
                btype="highpass"
            )
            filtered = sp_signal.filtfilt(b_hp, a_hp, filtered, axis=1)
            applied_stages.append(f"highpass_{self.highpass_cutoff}Hz_order{self.highpass_order}")

        # 2. Notch Filters: Powerline Interference (50 Hz / 60 Hz)
        # Operates safely at fs >= 250 Hz (e.g. 500 Hz where nyquist=250 Hz)
        for f_notch in self.notch_freqs:
            if 0 < f_notch < (0.95 * nyquist):
                b_notch, a_notch = sp_signal.iirnotch(
                    w0=f_notch,
                    Q=self.notch_q,
                    fs=fs
                )
                filtered = sp_signal.filtfilt(b_notch, a_notch, filtered, axis=1)
                applied_stages.append(f"notch_{f_notch}Hz_Q{self.notch_q}")
            else:
                applied_stages.append(f"notch_{f_notch}Hz_SKIPPED_above_nyquist_limit_{0.95*nyquist:.1f}Hz")

        # 3. Low-Pass Filter: High-Frequency Artifact Suppression & Anti-Aliasing
        effective_lp = self.lowpass_cutoff
        if effective_lp is not None and 0 < effective_lp < nyquist:
            b_lp, a_lp = sp_signal.butter(
                self.lowpass_order,
                effective_lp / nyquist,
                btype="lowpass"
            )
            filtered = sp_signal.filtfilt(b_lp, a_lp, filtered, axis=1)
            applied_stages.append(f"lowpass_{effective_lp}Hz_order{self.lowpass_order}")

        metadata = {
            "fs_at_filtering": fs,
            "nyquist": nyquist,
            "applied_stages": applied_stages,
            "effective_lowpass_hz": effective_lp,
            "notch_freqs": self.notch_freqs,
            "zero_phase": True,
            "morphology_disclaimer": (
                "filtfilt applies forward-backward filtering and avoids phase shift / temporal displacement "
                "associated with causal one-direction filtering, but it does not guarantee exact preservation "
                "of ECG amplitude or morphology."
            )
        }

        return filtered, metadata

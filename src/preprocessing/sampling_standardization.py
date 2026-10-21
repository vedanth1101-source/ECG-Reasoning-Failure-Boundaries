"""
Sampling and Duration Standardization Module (P_2)
--------------------------------------------------
Standardizes varying ECG sampling frequencies to a uniform target rate (e.g. 100 Hz)
and duration (e.g. 10.0 seconds, 1000 samples).

RESAMPLING METHODOLOGY (Critical Fix 1):
- Primary Method (Default): Polyphase filtering and decimation via `scipy.signal.resample_poly`.
  This uses an anti-aliasing FIR filter and efficient polyphase decomposition, avoiding the
  boundary ringing and cyclic wraparound artifacts inherent to Fourier-domain methods.
- Secondary Method: Fourier/FFT-based interpolation via `scipy.signal.resample`, retained only
  for non-rational or arbitrary resample configurations and documented strictly as FFT-based.
"""

from typing import Tuple, Dict, Any, Optional
from math import gcd
import numpy as np
from scipy import signal as sp_signal


class SamplingStandardizer:
    """
    P_2: Sampling Rate and Duration Standardization Component.
    Converts 500 Hz physiological signals to 100 Hz standardized tensors (1,000 samples)
    using polyphase anti-aliasing resampling.
    """

    def __init__(
        self,
        target_fs: float = 100.0,
        target_duration_sec: float = 10.0,
        windowing_mode: str = "center",
        resample_method: str = "polyphase"
    ):
        """
        Args:
            target_fs: Target sampling frequency in Hz (default: 100.0).
            target_duration_sec: Target signal duration in seconds (default: 10.0).
            windowing_mode: 'center' (crop/pad symmetrically) or 'start' (crop/pad at end).
            resample_method: 'polyphase' (default, uses resample_poly) or 'fourier' (uses resample).
        """
        if target_fs <= 0:
            raise ValueError(f"target_fs must be > 0, got {target_fs}")
        if target_duration_sec <= 0:
            raise ValueError(f"target_duration_sec must be > 0, got {target_duration_sec}")
        if windowing_mode not in ("center", "start"):
            raise ValueError(f"windowing_mode must be 'center' or 'start', got {windowing_mode}")
        if resample_method not in ("polyphase", "fourier"):
            raise ValueError(f"resample_method must be 'polyphase' or 'fourier', got {resample_method}")

        self.target_fs = float(target_fs)
        self.target_duration_sec = float(target_duration_sec)
        self.target_samples = int(round(self.target_fs * self.target_duration_sec))
        self.windowing_mode = windowing_mode
        self.resample_method = resample_method

    def standardize(
        self, signal: np.ndarray, fs_in: float
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resamples and duration-standardizes the ECG signal.

        Args:
            signal: numpy array of shape (12, T).
            fs_in: Source sampling frequency in Hz (e.g. 500 Hz).

        Returns:
            Tuple of (standardized_signal of shape (12, target_samples), metadata_dict).
        """
        if signal.ndim != 2:
            raise ValueError(f"Expected 2D array of shape (12, T), got {signal.shape}")
        if fs_in <= 0:
            raise ValueError(f"fs_in must be > 0, got {fs_in}")

        channels, orig_samples = signal.shape
        orig_duration = orig_samples / fs_in

        # 1. Resampling
        if abs(fs_in - self.target_fs) > 1e-3:
            if self.resample_method == "polyphase":
                # Compute integer rational factors
                up = int(round(self.target_fs))
                down = int(round(fs_in))
                common = gcd(up, down)
                up //= common
                down //= common
                # Apply polyphase filtering and decimation along time axis (axis=1)
                resampled_signal = sp_signal.resample_poly(signal, up=up, down=down, axis=1)
            else:
                # Fourier-based FFT resample (documented accurately as FFT-based)
                resampled_len = int(round(orig_samples * (self.target_fs / fs_in)))
                resampled_signal = sp_signal.resample(signal, resampled_len, axis=1)
        else:
            resampled_signal = np.copy(signal)

        # 2. Duration Standardization (Centering, Padding, or Trimming)
        current_len = resampled_signal.shape[1]
        pad_left = 0
        pad_right = 0
        crop_start = 0
        crop_end = current_len

        if current_len == self.target_samples:
            standardized_signal = resampled_signal
        elif current_len > self.target_samples:
            # Excess samples: trim
            excess = current_len - self.target_samples
            if self.windowing_mode == "center":
                crop_start = excess // 2
                crop_end = crop_start + self.target_samples
            else:
                crop_start = 0
                crop_end = self.target_samples
            standardized_signal = resampled_signal[:, crop_start:crop_end]
        else:
            # Insufficient samples: pad symmetrically with zeros
            deficit = self.target_samples - current_len
            if self.windowing_mode == "center":
                pad_left = deficit // 2
                pad_right = deficit - pad_left
            else:
                pad_left = 0
                pad_right = deficit
            standardized_signal = np.pad(
                resampled_signal,
                ((0, 0), (pad_left, pad_right)),
                mode="constant",
                constant_values=0.0
            )

        metadata = {
            "original_fs": fs_in,
            "target_fs": self.target_fs,
            "resample_method": self.resample_method,
            "resample_ratio": f"{self.target_fs}/{fs_in}",
            "original_samples": orig_samples,
            "original_duration_sec": round(orig_duration, 3),
            "target_samples": self.target_samples,
            "target_duration_sec": self.target_duration_sec,
            "pad_left": pad_left,
            "pad_right": pad_right,
            "crop_start": crop_start,
            "crop_end": crop_end,
            "windowing_mode": self.windowing_mode,
        }

        return standardized_signal.astype(np.float32), metadata

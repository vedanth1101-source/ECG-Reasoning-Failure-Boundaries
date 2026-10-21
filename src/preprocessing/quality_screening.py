"""
Quality Screening Module (P_5)
------------------------------
Executes a structural and numerical integrity audit that screens and rejects
any ECG recording containing:
1. Flatlined or disconnected channels
2. Non-finite values (NaN, +Inf, -Inf)
3. Dimensional anomalies (non-12 lead, improper sample length)
4. Extreme amplifier saturation / clipping

Provides structured logging and audit reports detailing exact causes of rejection.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class QualityAuditResult:
    """Detailed structural and numerical integrity audit report."""
    passed: bool
    rejection_reasons: List[str] = field(default_factory=list)
    channel_reports: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "rejection_reasons": self.rejection_reasons,
            "channel_reports": self.channel_reports,
            "metadata": self.metadata,
        }


class QualityScreening:
    """
    P_5: Quality Screening and Integrity Audit Component.
    Enforces quality gates before signals enter downstream model reasoning.
    """

    def __init__(
        self,
        min_channel_std_mv: float = 1e-4,
        max_valid_amplitude_mv: float = 15.0,
        max_consecutive_flatline_sec: float = 1.5,
        expected_channels: int = 12,
        expected_samples: Optional[int] = None,
    ):
        """
        Args:
            min_channel_std_mv: Minimum required standard deviation (mV) to not be considered flatlined.
            max_valid_amplitude_mv: Maximum allowable absolute voltage (mV) before flagging saturation.
            max_consecutive_flatline_sec: Maximum allowable consecutive constant samples in seconds.
            expected_channels: Expected channel dimension (default 12).
            expected_samples: Optional expected temporal sample count.
        """
        self.min_channel_std_mv = min_channel_std_mv
        self.max_valid_amplitude_mv = max_valid_amplitude_mv
        self.max_consecutive_flatline_sec = max_consecutive_flatline_sec
        self.expected_channels = expected_channels
        self.expected_samples = expected_samples

    def audit(
        self,
        signal: np.ndarray,
        lead_names: List[str],
        fs: float
    ) -> QualityAuditResult:
        """
        Executes structural and numerical integrity check on the preprocessed ECG signal.

        Args:
            signal: numpy array of shape (C, T).
            lead_names: list of channel names.
            fs: sampling rate in Hz.

        Returns:
            QualityAuditResult with passed flag and comprehensive rejection logs.
        """
        rejection_reasons: List[str] = []
        channel_reports: Dict[str, Dict[str, Any]] = {}

        # 1. Dimensional Integrity Check
        if signal.ndim != 2:
            rejection_reasons.append(
                f"Dimensional anomaly: Expected 2D signal array, got ndim={signal.ndim} (shape={signal.shape})"
            )
            return QualityAuditResult(
                passed=False,
                rejection_reasons=rejection_reasons,
                metadata={"shape": list(signal.shape)}
            )

        channels, num_samples = signal.shape
        if channels != self.expected_channels:
            rejection_reasons.append(
                f"Dimensional anomaly: Expected {self.expected_channels} channels, found {channels} channels."
            )

        if self.expected_samples is not None and num_samples != self.expected_samples:
            rejection_reasons.append(
                f"Dimensional anomaly: Expected {self.expected_samples} samples, found {num_samples} samples."
            )

        # 2. Global Non-Finite Check (NaN / Inf)
        non_finite_mask = ~np.isfinite(signal)
        total_non_finite = int(np.sum(non_finite_mask))
        if total_non_finite > 0:
            rejection_reasons.append(
                f"Numerical anomaly: Found {total_non_finite} non-finite (NaN / Inf) values in signal."
            )

        # 3. Channel-Specific Auditing
        max_flatline_samples = int(round(self.max_consecutive_flatline_sec * fs))

        for c_idx in range(channels):
            lead_name = lead_names[c_idx] if c_idx < len(lead_names) else f"ch_{c_idx}"
            channel_data = signal[c_idx, :]

            finite_channel_data = channel_data[np.isfinite(channel_data)]
            if len(finite_channel_data) == 0:
                channel_reports[lead_name] = {
                    "lead": lead_name,
                    "all_non_finite": True,
                }
                rejection_reasons.append(f"Channel '{lead_name}' contains only non-finite values.")
                continue

            c_min = float(np.min(finite_channel_data))
            c_max = float(np.max(finite_channel_data))
            c_std = float(np.std(finite_channel_data))
            c_mean = float(np.mean(finite_channel_data))

            # Flatline check via std
            flatline_detected = False
            if c_std < self.min_channel_std_mv:
                flatline_detected = True
                rejection_reasons.append(
                    f"Flatline anomaly on lead '{lead_name}': standard deviation {c_std:.2e} mV < threshold {self.min_channel_std_mv:.2e} mV."
                )

            # Flatline check via consecutive constant values
            diff = np.diff(finite_channel_data)
            zero_diff = (np.abs(diff) < 1e-7).astype(np.int32)
            # Fast check of consecutive flatline length
            if np.any(zero_diff):
                # Run-length check
                padded = np.concatenate(([0], zero_diff, [0]))
                changes = np.diff(padded)
                starts = np.where(changes == 1)[0]
                ends = np.where(changes == -1)[0]
                run_lengths = ends - starts
                max_run = int(np.max(run_lengths)) if len(run_lengths) > 0 else 0
                if max_run >= max_flatline_samples:
                    flatline_detected = True
                    rejection_reasons.append(
                        f"Flatline run anomaly on lead '{lead_name}': {max_run} consecutive static samples ({max_run / fs:.2f}s) >= threshold ({self.max_consecutive_flatline_sec:.2f}s)."
                    )
            else:
                max_run = 0

            # Amplifier Saturation / Clipping Check
            saturation_detected = False
            if abs(c_min) > self.max_valid_amplitude_mv or abs(c_max) > self.max_valid_amplitude_mv:
                saturation_detected = True
                rejection_reasons.append(
                    f"Saturation/clipping anomaly on lead '{lead_name}': voltage bounds [{c_min:.2f}, {c_max:.2f}] mV exceed valid limit ±{self.max_valid_amplitude_mv:.2f} mV."
                )

            channel_reports[lead_name] = {
                "lead": lead_name,
                "std_mv": c_std,
                "mean_mv": c_mean,
                "min_mv": c_min,
                "max_mv": c_max,
                "flatline": flatline_detected,
                "saturation": saturation_detected,
                "max_static_run_samples": max_run,
            }

        passed = len(rejection_reasons) == 0

        return QualityAuditResult(
            passed=passed,
            rejection_reasons=rejection_reasons,
            channel_reports=channel_reports,
            metadata={
                "channels": channels,
                "samples": num_samples,
                "fs": fs,
                "duration_sec": num_samples / fs if fs > 0 else 0,
            }
        )

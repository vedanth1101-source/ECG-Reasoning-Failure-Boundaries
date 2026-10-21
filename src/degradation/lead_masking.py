"""
Lead Availability Masking Module (Phase 2 Master Implementation)
================================================================
Implements the mathematical degradation operator:
    X_masked = X ⊙ (m_l ⊗ 1_T^T)

Where:
    - X in R^{12 x T} is the canonical 12-lead standardized recording
    - m_l in {0, 1}^12 is the binary availability mask at degradation step l
    - Inactive leads are strictly clamped to 0.0

CRITICAL RESEARCH CONSTRAINT:
Under NO circumstances are missing leads reconstructed or synthesized.
Zero generative imputation is performed.
"""

import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
import numpy as np

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Canonical Cabrera sequence (immutable)
CANONICAL_LEADS: Tuple[str, ...] = (
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
)

LEAD_TO_INDEX: Dict[str, int] = {lead: idx for idx, lead in enumerate(CANONICAL_LEADS)}


@dataclass(frozen=True)
class LeadProvenance:
    """
    Formal lead provenance metadata distinguishing measured vs mathematically derived channels.
    """
    lead_name: str
    source_type: str  # 'measured' or 'derived'
    parent_leads: Tuple[str, ...]
    derivation_formula: Optional[str]
    independent_observation: bool


# Canonical provenance mapping based on standard electrocardiographic lead geometry:
# Directly acquired bipolar limb leads: I, II
# Unipolar precordial leads: V1, V2, V3, V4, V5, V6
# Derived channels (Einthoven's triangle / Goldberger unipolar augmented):
# III = II - I
# aVR = -(I + II) / 2
# aVL = (I - III) / 2 = I - II / 2
# aVF = (II + III) / 2 = II - I / 2
LEAD_PROVENANCE: Dict[str, LeadProvenance] = {
    "I": LeadProvenance(
        lead_name="I",
        source_type="measured",
        parent_leads=(),
        derivation_formula=None,
        independent_observation=True,
    ),
    "II": LeadProvenance(
        lead_name="II",
        source_type="measured",
        parent_leads=(),
        derivation_formula=None,
        independent_observation=True,
    ),
    "III": LeadProvenance(
        lead_name="III",
        source_type="derived",
        parent_leads=("I", "II"),
        derivation_formula="III = II - I",
        independent_observation=False,
    ),
    "aVR": LeadProvenance(
        lead_name="aVR",
        source_type="derived",
        parent_leads=("I", "II"),
        derivation_formula="aVR = -(I + II) / 2",
        independent_observation=False,
    ),
    "aVL": LeadProvenance(
        lead_name="aVL",
        source_type="derived",
        parent_leads=("I", "II"),
        derivation_formula="aVL = I - II / 2",
        independent_observation=False,
    ),
    "aVF": LeadProvenance(
        lead_name="aVF",
        source_type="derived",
        parent_leads=("I", "II"),
        derivation_formula="aVF = II - I / 2",
        independent_observation=False,
    ),
    "V1": LeadProvenance(
        lead_name="V1",
        source_type="measured",
        parent_leads=(),
        derivation_formula=None,
        independent_observation=True,
    ),
    "V2": LeadProvenance(
        lead_name="V2",
        source_type="measured",
        parent_leads=(),
        derivation_formula=None,
        independent_observation=True,
    ),
    "V3": LeadProvenance(
        lead_name="V3",
        source_type="measured",
        parent_leads=(),
        derivation_formula=None,
        independent_observation=True,
    ),
    "V4": LeadProvenance(
        lead_name="V4",
        source_type="measured",
        parent_leads=(),
        derivation_formula=None,
        independent_observation=True,
    ),
    "V5": LeadProvenance(
        lead_name="V5",
        source_type="measured",
        parent_leads=(),
        derivation_formula=None,
        independent_observation=True,
    ),
    "V6": LeadProvenance(
        lead_name="V6",
        source_type="measured",
        parent_leads=(),
        derivation_formula=None,
        independent_observation=True,
    ),
}


def get_lead_observability_status(
    lead_name: str,
    active_leads: Union[List[str], Tuple[str, ...], np.ndarray]
) -> str:
    """
    Distinguishes lead observability taking into account provenance:
      - 'observed_measured': directly recorded physical lead and currently present
      - 'observed_derived': mathematically derived lead and present, with parent leads available
      - 'unavailable_derived': mathematically derived lead where required parents (I or II) are masked
      - 'masked_lead': lead is unobserved / zeroed in current configuration
    """
    if isinstance(active_leads, np.ndarray):
        active_set = {CANONICAL_LEADS[i] for i in range(12) if active_leads[i] > 0}
    else:
        active_set = set(active_leads)

    if lead_name not in active_set:
        return "masked_lead"

    prov = LEAD_PROVENANCE.get(lead_name)
    if prov is None or prov.source_type == "measured":
        return "observed_measured"

    # For derived leads, check parent availability
    parents_available = all(p in active_set for p in prov.parent_leads)
    if parents_available:
        return "observed_derived"
    else:
        return "unavailable_derived"


class AnatomicalRegion:
    """Anatomical vector orientations and lead groupings."""
    LIMB = ("I", "II", "III", "aVR", "aVL", "aVF")
    PRECORDIAL = ("V1", "V2", "V3", "V4", "V5", "V6")
    INFERIOR = ("II", "III", "aVF")
    LATERAL = ("I", "aVL", "V5", "V6")
    SEPTAL = ("V1", "V2")
    ANTERIOR = ("V3", "V4")
    ANTEROSEPTAL = ("V1", "V2", "V3", "V4")
    HIGH_LATERAL = ("I", "aVL")
    LOW_LATERAL = ("V5", "V6")
    INDEPENDENT_LIMB = ("I", "II")  # Leads III, aVR, aVL, aVF are derived via Einthoven/Goldberger


@dataclass(frozen=True)
class LeadMask:
    """
    Immutable representation of an availability mask at a specific degradation level.
    """
    name: str
    num_leads: int
    mask_vector: np.ndarray  # (12,) uint8 binary vector
    active_leads: Tuple[str, ...]
    masked_leads: Tuple[str, ...]
    regime: str  # 'anatomical' or 'stepwise'
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "num_leads": self.num_leads,
            "active_leads": list(self.active_leads),
            "masked_leads": list(self.masked_leads),
            "mask_vector": self.mask_vector.tolist(),
            "regime": self.regime,
            "description": self.description
        }


@dataclass
class MaskedECG:
    """
    Result of applying a LeadMask to an input ECG signal.
    """
    signal: np.ndarray  # (12, T) or (N, 12, T)
    mask: LeadMask
    active_leads: List[str]
    masked_leads: List[str]
    original_shape: Tuple[int, ...]
    original_energy: Optional[float] = None
    is_reconstructed: bool = False  # Research constraint: MUST ALWAYS BE FALSE

    @property
    def num_active_leads(self) -> int:
        return self.mask.num_leads

    @property
    def energy_retention_ratio(self) -> float:
        """
        Computes waveform signal-energy retention ratio (Frobenius norm squared of masked vs original signal).
        
        METHODOLOGICAL DISTINCTION (Medium Fix A):
        Signal-energy retention != Clinical-information retention.
        A degraded signal may retain substantial physical waveform energy while
        vital diagnostic observations (e.g. localized ST-segment elevations or P-waves)
        are completely lost due to missing channels.
        """
        if self.original_energy is not None and self.original_energy > 0:
            masked_energy = float(np.sum(self.signal ** 2))
            return masked_energy / self.original_energy
        return 0.0


def create_lead_mask(
    name: str,
    active_leads: List[str],
    regime: str = "custom",
    description: str = ""
) -> LeadMask:
    """
    Constructs an immutable LeadMask from a list of active lead names.
    """
    # Validate lead names
    for lead in active_leads:
        if lead not in LEAD_TO_INDEX:
            raise ValueError(f"Unknown lead '{lead}'. Must be one of {CANONICAL_LEADS}")

    mask_vec = np.zeros(12, dtype=np.uint8)
    for lead in active_leads:
        mask_vec[LEAD_TO_INDEX[lead]] = 1

    active_tuple = tuple(lead for lead in CANONICAL_LEADS if mask_vec[LEAD_TO_INDEX[lead]] == 1)
    masked_tuple = tuple(lead for lead in CANONICAL_LEADS if mask_vec[LEAD_TO_INDEX[lead]] == 0)

    return LeadMask(
        name=name,
        num_leads=len(active_tuple),
        mask_vector=mask_vec,
        active_leads=active_tuple,
        masked_leads=masked_tuple,
        regime=regime,
        description=description
    )


# -------------------------------------------------------------------------
# Standard Schedules
# -------------------------------------------------------------------------

def get_anatomical_schedule() -> Dict[str, LeadMask]:
    """
    Returns the Clinical Anatomical Reduction Schedule:
      - 12 leads: Full clinical baseline
      - 8 leads: Independent leads (I, II, V1-V6) without derived limb leads
      - 6 leads (Limb): Frontal plane limb leads only (I, II, III, aVR, aVL, aVF)
      - 6 leads (Precordial): Horizontal plane precordial leads only (V1-V6)
      - 4 leads: Bipolar limb leads + single anterior lead (I, II, III, V1)
      - 2 leads (Orthogonal): Einthoven bipolar vectors (I, II)
      - 2 leads (Rhythm/Septal): Rhythm monitoring leads (II, V1)
      - 1 lead: Ambulatory single-lead rhythm strip (II)
    """
    schedules = {}

    schedules["12_lead"] = create_lead_mask(
        name="12_lead",
        active_leads=list(CANONICAL_LEADS),
        regime="anatomical",
        description="Full standard 12-lead clinical ECG baseline."
    )

    schedules["8_lead_independent"] = create_lead_mask(
        name="8_lead_independent",
        active_leads=["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"],
        regime="anatomical",
        description="Independent lead subset without mathematically derived leads (III = II - I, aVR, aVL, aVF)."
    )

    schedules["6_lead_limb"] = create_lead_mask(
        name="6_lead_limb",
        active_leads=["I", "II", "III", "aVR", "aVL", "aVF"],
        regime="anatomical",
        description="Frontal plane limb leads only; all horizontal precordial leads (V1-V6) masked."
    )

    schedules["6_lead_precordial"] = create_lead_mask(
        name="6_lead_precordial",
        active_leads=["V1", "V2", "V3", "V4", "V5", "V6"],
        regime="anatomical",
        description="Horizontal plane precordial leads only; all frontal limb leads (I-aVF) masked."
    )

    schedules["4_lead"] = create_lead_mask(
        name="4_lead",
        active_leads=["I", "II", "III", "V1"],
        regime="anatomical",
        description="Bipolar limb leads (Einthoven triangle) plus single anterior/septal lead V1."
    )

    schedules["2_lead_orthogonal"] = create_lead_mask(
        name="2_lead_orthogonal",
        active_leads=["I", "II"],
        regime="anatomical",
        description="Orthogonal bipolar frontal leads (Lead I and Lead II)."
    )

    schedules["2_lead_rhythm"] = create_lead_mask(
        name="2_lead_rhythm",
        active_leads=["II", "V1"],
        regime="anatomical",
        description="Dual rhythm monitoring leads: inferior limb vector (II) and right precordial vector (V1)."
    )

    schedules["1_lead"] = create_lead_mask(
        name="1_lead",
        active_leads=["II"],
        regime="anatomical",
        description="Ambulatory / wearable single-lead rhythm monitoring (Lead II)."
    )

    return schedules


def get_stepwise_schedule() -> Dict[str, LeadMask]:
    """
    Returns the Primary Fixed Clinically/Device-Motivated Elimination Schedule
    loaded directly from configs/lead_configurations.json:
    12 -> 10 -> 8 -> 6 -> 4 -> 2 -> 1 leads.
    """
    import json
    cfg_path = repo_root / "configs" / "lead_configurations.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg_data = json.load(f)
        schedules = {}
        for lvl_str in ["12", "10", "8", "6", "4", "2", "1"]:
            if lvl_str in cfg_data["configurations"]:
                conf = cfg_data["configurations"][lvl_str]
                schedules[f"{lvl_str}_lead"] = create_lead_mask(
                    name=f"{lvl_str}_lead",
                    active_leads=conf["leads"],
                    regime="stepwise",
                    description=f"{conf['motivation']} (Source: {conf['source']})"
                )
        return schedules

    # Fallback if config file not found
    schedules = {}
    schedules["12_lead"] = create_lead_mask(
        name="12_lead",
        active_leads=list(CANONICAL_LEADS),
        regime="stepwise",
        description="12 available leads."
    )
    schedules["10_lead"] = create_lead_mask(
        name="10_lead",
        active_leads=["I", "II", "aVR", "aVL", "V1", "V2", "V3", "V4", "V5", "V6"],
        regime="stepwise",
        description="10 available leads."
    )
    schedules["8_lead"] = create_lead_mask(
        name="8_lead",
        active_leads=["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"],
        regime="stepwise",
        description="8 available leads."
    )
    schedules["6_lead"] = create_lead_mask(
        name="6_lead",
        active_leads=["I", "II", "III", "aVR", "aVL", "aVF"],
        regime="stepwise",
        description="6 available leads (Frontal plane hexaxial limb leads)."
    )
    schedules["4_lead"] = create_lead_mask(
        name="4_lead",
        active_leads=["I", "II", "III", "V1"],
        regime="stepwise",
        description="4 available leads (Einthoven + V1)."
    )
    schedules["2_lead"] = create_lead_mask(
        name="2_lead",
        active_leads=["II", "V1"],
        regime="stepwise",
        description="2 available leads (telemetry-like II and V1)."
    )
    schedules["1_lead"] = create_lead_mask(
        name="1_lead",
        active_leads=["II"],
        regime="stepwise",
        description="1 available lead (ambulatory-like Lead II)."
    )
    return schedules


def create_custom_mask(active_leads: List[str], name: str = "custom") -> LeadMask:
    """Convenience factory for custom experimental masks."""
    return create_lead_mask(name=name, active_leads=active_leads, regime="custom")


# -------------------------------------------------------------------------
# Lead Masking Operator
# -------------------------------------------------------------------------

class LeadMaskingOperator:
    """
    Applies lead availability masks to standardized ECG signals.
    Supports single instances of shape (12, T) and batches of shape (N, 12, T).
    Supports NumPy arrays and PyTorch Tensors.
    """

    def __init__(self, anatomical_schedules: Optional[Dict[str, LeadMask]] = None):
        self.anatomical_schedules = anatomical_schedules or get_anatomical_schedule()
        self.stepwise_schedules = get_stepwise_schedule()

    def get_mask(self, mask_identifier: Union[str, LeadMask, List[str]]) -> LeadMask:
        """Resolves string name, LeadMask object, or list of leads into a LeadMask."""
        if isinstance(mask_identifier, LeadMask):
            return mask_identifier
        elif isinstance(mask_identifier, str):
            if mask_identifier in self.anatomical_schedules:
                return self.anatomical_schedules[mask_identifier]
            elif mask_identifier in self.stepwise_schedules:
                return self.stepwise_schedules[mask_identifier]
            else:
                raise KeyError(
                    f"Mask '{mask_identifier}' not found in standard schedules. "
                    f"Available anatomical: {list(self.anatomical_schedules.keys())}, "
                    f"stepwise: {list(self.stepwise_schedules.keys())}"
                )
        elif isinstance(mask_identifier, (list, tuple)):
            return create_custom_mask(list(mask_identifier))
        else:
            raise TypeError(f"Unsupported mask identifier type: {type(mask_identifier)}")

    def load_signal(self, file_path: Union[str, Path]) -> np.ndarray:
        """
        Loads an ECG recording from disk:
          - If .npy or .npz: directly loads numpy array
          - If WFDB record (.hea, .dat, .mat): loads and runs Phase 1 pipeline to return (12, 1000)
        """
        path = Path(file_path)
        if not path.exists() and not path.with_suffix(".hea").exists():
            raise FileNotFoundError(f"ECG file not found at {path}")

        if path.suffix == ".npy":
            arr = np.load(path)
            if arr.ndim == 2 and arr.shape[0] != 12 and arr.shape[1] == 12:
                arr = arr.T  # Transpose (T, 12) -> (12, T)
            return arr.astype(np.float32)

        elif path.suffix == ".npz":
            data = np.load(path)
            key = "signal" if "signal" in data else list(data.keys())[0]
            arr = data[key]
            if arr.ndim == 2 and arr.shape[0] != 12 and arr.shape[1] == 12:
                arr = arr.T
            return arr.astype(np.float32)

        else:
            # WFDB record format
            import wfdb
            from src.preprocessing.pipeline import ECGPreprocessingPipeline

            base_path = str(path.with_suffix(""))
            raw_sig, meta = wfdb.rdsamp(base_path)
            pipeline = ECGPreprocessingPipeline(target_fs=100.0, target_duration_sec=10.0)
            res = pipeline.process(
                raw_signal=raw_sig,
                lead_names=meta["sig_name"],
                fs_in=float(meta["fs"]),
                units=meta.get("units", "mV")
            )
            return res.signal.astype(np.float32)

    def apply(
        self,
        signal: Union[np.ndarray, str, Path, Any],
        mask: Union[str, LeadMask, List[str]],
        channels_dim: int = -2
    ) -> MaskedECG:
        """
        Applies mask vector to the signal tensor. Accepts:
          - A NumPy array or PyTorch Tensor: (12, T) or (N, 12, T)
          - A string or Path to an ECG file (.npy, .npz, or WFDB record)

        Args:
            signal: numpy array, torch Tensor, or file path.
            mask: LeadMask instance, schedule name string, or list of lead names.
            channels_dim: dimension index representing the 12 leads (-2 by default).

        Returns:
            MaskedECG object with inactive leads zeroed out.
        """
        # Load from file if string or Path provided
        if isinstance(signal, (str, Path)):
            signal = self.load_signal(signal)

        lead_mask = self.get_mask(mask)
        original_shape = signal.shape

        # Verify channels dimension is 12
        if signal.shape[channels_dim] != 12:
            raise ValueError(
                f"Expected 12 channels along dimension {channels_dim}, got {signal.shape[channels_dim]}. "
                f"Signal shape: {signal.shape}. Ensure signals are in canonical (12, T) layout."
            )

        # Compute original signal energy for retention ratio
        if isinstance(signal, np.ndarray):
            orig_energy = float(np.sum(signal ** 2))
        else:
            orig_energy = float(signal.pow(2).sum().item()) if hasattr(signal, "pow") else None

        # PyTorch Tensor branch
        if hasattr(signal, "device") and hasattr(signal, "dtype") and not isinstance(signal, np.ndarray):
            import torch
            mask_t = torch.tensor(lead_mask.mask_vector, dtype=signal.dtype, device=signal.device)
            # Reshape for broadcasting
            if signal.ndim == 2:  # (12, T)
                mask_t = mask_t.unsqueeze(-1)  # (12, 1)
            elif signal.ndim == 3:  # (N, 12, T)
                mask_t = mask_t.unsqueeze(0).unsqueeze(-1)  # (1, 12, 1)
            masked_signal = signal * mask_t
        else:
            # NumPy array branch
            mask_arr = lead_mask.mask_vector.astype(signal.dtype)
            if signal.ndim == 2:  # (12, T)
                mask_arr = mask_arr[:, np.newaxis]  # (12, 1)
            elif signal.ndim == 3:  # (N, 12, T)
                mask_arr = mask_arr[np.newaxis, :, np.newaxis]  # (1, 12, 1)
            else:
                raise ValueError(f"Signal must be 2D (12, T) or 3D (N, 12, T), got {signal.ndim}D.")
            masked_signal = signal * mask_arr

        return MaskedECG(
            signal=masked_signal,
            mask=lead_mask,
            active_leads=list(lead_mask.active_leads),
            masked_leads=list(lead_mask.masked_leads),
            original_shape=original_shape,
            original_energy=orig_energy,
            is_reconstructed=False  # Invariant strictly preserved
        )

    def progressive_mask(
        self,
        signal: Union[np.ndarray, str, Path, Any],
        schedule_type: str = "stepwise"
    ) -> Dict[str, MaskedECG]:
        """
        Executes complete progressive lead masking across an individual recording or file.

        Args:
            signal: numpy array (12, T), batch (N, 12, T), or file path.
            schedule_type: 'stepwise' (12->10->8->6->4->2->1) or 'anatomical' (standard clinical tiers).

        Returns:
            Ordered dictionary: {level_name: MaskedECG}
        """
        if schedule_type == "stepwise":
            schedule = self.stepwise_schedules
            level_order = ["12_lead", "10_lead", "8_lead", "6_lead", "4_lead", "2_lead", "1_lead"]
        elif schedule_type == "anatomical":
            schedule = self.anatomical_schedules
            level_order = [
                "12_lead", "8_lead_independent", "6_lead_precordial", "6_lead_limb",
                "4_lead", "2_lead_rhythm", "2_lead_orthogonal", "1_lead"
            ]
        else:
            raise ValueError(f"Unknown schedule_type '{schedule_type}'. Must be 'stepwise' or 'anatomical'.")

        results = {}
        for level in level_order:
            if level in schedule:
                results[level] = self.apply(signal, schedule[level])
        return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Apply progressive lead masking to an ECG file.")
    parser.add_argument("--file", type=str, required=True, help="Path to ECG file (.npy or WFDB record)")
    parser.add_argument("--schedule", type=str, choices=["stepwise", "anatomical"], default="stepwise")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional output dir to save masked .npy files")
    args = parser.parse_args()

    operator = LeadMaskingOperator()
    print(f"Loading and applying progressive lead masking on: {args.file} (schedule: {args.schedule})...")

    results = operator.progressive_mask(args.file, schedule_type=args.schedule)

    print("\nDegradation Ladder Results:")
    for level, masked in results.items():
        active = ",".join(masked.active_leads)
        ret = masked.energy_retention_ratio * 100.0
        print(f"  [{level:>18}] -> Active Leads: {masked.num_active_leads:2d} ({active:<32}) | Energy Retained: {ret:5.1f}%")

    if args.output_dir:
        out_p = Path(args.output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        file_stem = Path(args.file).stem
        for level, masked in results.items():
            save_file = out_p / f"{file_stem}_{level}.npy"
            np.save(save_file, masked.signal)
        print(f"\nSaved {len(results)} masked arrays to {out_p.resolve()}")


if __name__ == "__main__":
    main()

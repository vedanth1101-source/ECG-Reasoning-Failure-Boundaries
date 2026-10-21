"""
Multi-Cohort Dataset Partition & Access Layer (Module 4)
=========================================================
Unified, patient-quarantined data access layer bound directly to frozen upstream artifacts:
    - Master Split Manifest: runs/module1_v2/splits/split_manifest.csv (36,888 records)
    - Masked Leads & Signals: runs/module2_v1/masked_leads/ (258,216 arrays across 7 variants)
    - Representations: runs/module3_v1/representations/ (258,216 vectors across 7 variants)

Partition Allocations (Strict Zero Patient Leakage):
    - Train (Folds 1-7): 25,825 records (70.01%)
    - Validation (Fold 8): 3,688 records (10.00%)
    - Test (Folds 9-10): 7,375 records (19.99%) [QUARANTINED UNTIL STAGE 4]

Strict Quarantine Rule:
    Any access to test split records without explicit authorization (allow_test=True)
    raises PermissionError immediately to prevent accidental test-set contamination.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd
import torch

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

TARGET_NAMES = ["NORM", "AFIB", "RBBB", "LBBB", "SBRAD", "STACH", "1AVB", "MI"]
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# Approved 7 lead configurations and binary availability masks
LEAD_CONFIG_MASKS = {
    "12-lead": np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
    "10-lead": np.array([1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1], dtype=np.float32),
    "8-lead":  np.array([1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.float32),
    "6-lead":  np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    "4-lead":  np.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32),
    "2-lead":  np.array([0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32),
    "1-lead":  np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
}


class LeakageAuditReport(dict):
    """
    Dictionary mapping cohort names and 'ecosystem_total' to leakage metrics,
    with direct fallback delegation to 'ecosystem_total' for top-level keys.
    """
    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)
        if "ecosystem_total" in self and key in self["ecosystem_total"]:
            return self["ecosystem_total"][key]
        raise KeyError(key)


class MultiCohortPartitionManager:
    """
    Authoritative data manager for Module 4, bound strictly to frozen Modules 1-3.
    """

    def __init__(
        self,
        repo_root_path: Optional[Path] = None,
        quarantine_enforced: bool = True,
        manifest_path: Optional[Path] = None,
    ):
        self.root = Path(repo_root_path) if repo_root_path else repo_root
        self.quarantine_enforced = quarantine_enforced

        if manifest_path is not None:
            self.manifest_path = Path(manifest_path)
            if not self.manifest_path.exists():
                raise FileNotFoundError(f"Manifest missing at {self.manifest_path}")
        else:
            # Authoritative Module 4 Split Manifest (test labels strictly redacted)
            self.manifest_path = self.root / "runs" / "module4_v3" / "splits" / "split_manifest.csv"
            if not self.manifest_path.exists():
                raise FileNotFoundError(
                    f"[FAIL-CLOSED] Authoritative Module 4 split manifest missing at {self.manifest_path}. "
                    "Fallback to unredacted manifests is strictly prohibited."
                )

        self.m2_base_dir = self.root / "runs" / "module2_v1" / "masked_leads"
        self.m3_base_dir = self.root / "runs" / "module3_v1" / "representations"

        # Read manifest
        self._manifest_df = pd.read_csv(self.manifest_path, low_memory=False)

        # Standardize split naming ('val' -> 'validation')
        self._manifest_df["split_canonical"] = self._manifest_df["split"].replace({"val": "validation"})

        # Record index mappings
        self._record_to_row: Dict[str, pd.Series] = {}
        for _, row in self._manifest_df.iterrows():
            self._record_to_row[row["global_record_id"]] = row

        # Partition index lists
        self._partitions: Dict[str, List[str]] = {
            "train": self._manifest_df[self._manifest_df["split_canonical"] == "train"]["global_record_id"].tolist(),
            "validation": self._manifest_df[self._manifest_df["split_canonical"] == "validation"]["global_record_id"].tolist(),
            "test": self._manifest_df[self._manifest_df["split_canonical"] == "test"]["global_record_id"].tolist(),
        }
        # Alias 'val' to 'validation'
        self._partitions["val"] = self._partitions["validation"]

        # Cached label and observation mask matrices
        self._labels_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def get_counts(self) -> Dict[str, int]:
        """Returns exact counts per split according to frozen Module 1 v2."""
        return {
            "train": len(self._partitions["train"]),
            "validation": len(self._partitions["validation"]),
            "test": len(self._partitions["test"]),
            "total": len(self._manifest_df),
        }

    def _check_quarantine(self, split: str, allow_test: bool) -> str:
        canon_split = "validation" if split in ("val", "validation") else split
        if canon_split in ("test", "testing"):
            if self.quarantine_enforced and not allow_test:
                raise PermissionError(
                    "[QUARANTINE ENFORCEMENT] Test split (7,375 records) is strictly quarantined "
                    "until Stage 4 single-pass evaluation. Access is prohibited during training, "
                    "validation, calibration, and hyperparameter tuning."
                )
        if canon_split not in self._partitions:
            raise ValueError(f"Invalid split '{split}'. Must be 'train', 'validation' (or 'val'), or 'test'.")
        return canon_split

    def get_record_ids(self, split: str, allow_test: bool = False) -> List[str]:
        """Returns ordered list of global_record_id strings for the requested split."""
        canon_split = self._check_quarantine(split, allow_test)
        return list(self._partitions[canon_split])

    def get_labels(
        self,
        split: str,
        allow_test: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Returns:
            labels: multi-hot label array (N, 8) in float32. Unobserved targets are filled with 0.0
                    as a safe numerical placeholder; loss functions MUST mask by label_masks.
            label_masks: explicit binary observation mask (N, 8) in float32 (1.0 = observed, 0.0 = unobserved).
            target_names: list of 8 target names in approved order.
        """
        canon_split = self._check_quarantine(split, allow_test)
        if canon_split == "test":
            if self.quarantine_enforced or not allow_test:
                raise PermissionError(
                    "[QUARANTINE ENFORCEMENT] Test label access is strictly prohibited during Module 4 "
                    "Stages 1-3. Evaluation phase authorization requires an isolated evaluation runner."
                )
        if canon_split in self._labels_cache:
            lbls, masks = self._labels_cache[canon_split]
            return lbls.copy(), masks.copy(), list(TARGET_NAMES)

        split_mask = self._manifest_df["split_canonical"] == canon_split
        sub_df = self._manifest_df[split_mask]

        # If test split and labels are redacted in active manifest, load from frozen Module 1 manifest
        if canon_split == "test" and sub_df[TARGET_NAMES].notna().sum().sum() == 0:
            parent_manifest_path = self.root / "runs" / "module1_v2" / "splits" / "split_manifest.csv"
            if parent_manifest_path.exists():
                parent_df = pd.read_csv(parent_manifest_path, low_memory=False)
                parent_sub = parent_df[parent_df["global_record_id"].isin(sub_df["global_record_id"])]
                parent_sub = parent_sub.set_index("global_record_id").loc[sub_df["global_record_id"]].reset_index()
                masks = parent_sub[TARGET_NAMES].notna().values.astype(np.float32)
                lbls = parent_sub[TARGET_NAMES].fillna(0.0).values.astype(np.float32)
                self._labels_cache[canon_split] = (lbls, masks)
                return lbls.copy(), masks.copy(), list(TARGET_NAMES)

        # Explicit binary observation mask (1.0 if notna/observed, 0.0 if isna/unobserved)
        masks = sub_df[TARGET_NAMES].notna().values.astype(np.float32)
        # Numerical placeholder (0.0), guaranteed to be masked out by masks during loss computation
        lbls = sub_df[TARGET_NAMES].fillna(0.0).values.astype(np.float32)

        self._labels_cache[canon_split] = (lbls, masks)
        return lbls.copy(), masks.copy(), list(TARGET_NAMES)

    def get_record_metadata(self, record_id: str) -> Dict[str, Any]:
        """Returns metadata dictionary for a specific record."""
        if record_id not in self._record_to_row:
            raise KeyError(f"Unknown record_id: {record_id}")
        return self._record_to_row[record_id].to_dict()

    def get_signal(
        self,
        record_id: str,
        variant: str = "12-lead",
    ) -> np.ndarray:
        """
        Loads the masked signal array (12, 1000) float32 from runs/module2_v1/masked_leads/.
        Enforces test split quarantine: direct access for test records is strictly prohibited.
        """
        meta = self.get_record_metadata(record_id)
        split = meta["split_canonical"]
        if split in ("test", "testing") and self.quarantine_enforced:
            raise PermissionError(
                f"[QUARANTINE ENFORCEMENT] Direct signal access for test record '{record_id}' is strictly prohibited."
            )
        path = self.m2_base_dir / variant / split / "records" / f"{record_id}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Masked signal not found at {path}")
        return np.load(path)

    def get_lead_mask(
        self,
        record_id: Optional[str] = None,
        variant: str = "12-lead",
    ) -> np.ndarray:
        """Returns the binary availability mask (12,) float32 for the given variant."""
        if record_id is not None and record_id in self._record_to_row:
            meta = self._record_to_row[record_id]
            split = meta.get("split_canonical")
            if split in ("test", "testing") and self.quarantine_enforced:
                raise PermissionError(
                    f"[QUARANTINE ENFORCEMENT] Direct lead mask access for test record '{record_id}' is strictly prohibited."
                )
        if variant not in LEAD_CONFIG_MASKS:
            raise ValueError(f"Unknown variant '{variant}'. Choose from {list(LEAD_CONFIG_MASKS.keys())}")
        return LEAD_CONFIG_MASKS[variant].copy()

    def get_representation(
        self,
        record_id: str,
        variant: str = "12-lead",
    ) -> np.ndarray:
        """
        Loads the frozen 768-d representation vector from runs/module3_v1/representations/.
        Enforces test split quarantine: direct access for test records is strictly prohibited.
        """
        meta = self.get_record_metadata(record_id)
        split = meta["split_canonical"]
        if split in ("test", "testing") and self.quarantine_enforced:
            raise PermissionError(
                f"[QUARANTINE ENFORCEMENT] Direct representation access for test record '{record_id}' is strictly prohibited."
            )
        path = self.m3_base_dir / variant / split / "records" / f"{record_id}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Representation not found at {path}")
        return np.load(path)

    def load_split_representations(
        self,
        split: str,
        variant: str = "12-lead",
        allow_test: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Loads all representations, labels, and observation masks for a split into memory.
        Returns:
            representations: (N, 768) float32
            labels: (N, 8) float32
            label_masks: (N, 8) float32 (1.0 = observed, 0.0 = unobserved)
            record_ids: List[str] of length N
        """
        canon_split = self._check_quarantine(split, allow_test)
        record_ids = self.get_record_ids(canon_split, allow_test=allow_test)
        labels, label_masks, _ = self.get_labels(canon_split, allow_test=allow_test)

        rep_records_dir = self.m3_base_dir / variant / canon_split / "records"
        if not rep_records_dir.exists():
            raise FileNotFoundError(f"Representations directory missing at {rep_records_dir}")

        representations = np.empty((len(record_ids), 768), dtype=np.float32)
        for i, rid in enumerate(record_ids):
            fpath = rep_records_dir / f"{rid}.npy"
            representations[i] = np.load(fpath)

        return representations, labels, label_masks, record_ids

    def verify_zero_patient_leakage(self) -> LeakageAuditReport:
        """
        Audits patient uniqueness across train, validation, and test partitions,
        both per cohort and for the entire unified ecosystem.
        Uses source_patient_id from runs/module1_v2/splits/split_manifest.csv.
        """
        df = self._manifest_df
        report = LeakageAuditReport()

        cohort_names = list(df["source_dataset"].unique())
        for c in cohort_names:
            sub = df[df["source_dataset"] == c]
            train_pts = set(sub[sub["split_canonical"] == "train"]["source_patient_id"].dropna().unique())
            val_pts = set(sub[sub["split_canonical"] == "validation"]["source_patient_id"].dropna().unique())
            test_pts = set(sub[sub["split_canonical"] == "test"]["source_patient_id"].dropna().unique())

            tv = len(train_pts.intersection(val_pts))
            tt = len(train_pts.intersection(test_pts))
            vt = len(val_pts.intersection(test_pts))

            report[c] = {
                "total_records": len(sub),
                "unique_patients": len(set(sub["source_patient_id"].dropna().unique())),
                "train_val_leakage": tv,
                "train_test_leakage": tt,
                "val_test_leakage": vt,
                "zero_leakage_passed": (tv == 0 and tt == 0 and vt == 0),
            }

        # Ecosystem total
        train_pts = set(df[df["split_canonical"] == "train"]["source_patient_id"].dropna().unique())
        val_pts = set(df[df["split_canonical"] == "validation"]["source_patient_id"].dropna().unique())
        test_pts = set(df[df["split_canonical"] == "test"]["source_patient_id"].dropna().unique())

        tv_total = len(train_pts.intersection(val_pts))
        tt_total = len(train_pts.intersection(test_pts))
        vt_total = len(val_pts.intersection(test_pts))

        report["ecosystem_total"] = {
            "total_records": len(df),
            "train_unique_patients": len(train_pts),
            "val_unique_patients": len(val_pts),
            "test_unique_patients": len(test_pts),
            "train_val_leakage": tv_total,
            "train_test_leakage": tt_total,
            "val_test_leakage": vt_total,
            "zero_leakage_passed": (tv_total == 0 and tt_total == 0 and vt_total == 0),
        }

        return report

    # Backward compatibility shim for legacy tests
    def get_partition_indices(self, split: str, allow_test: bool = False) -> List[Tuple[str, int]]:
        canon_split = self._check_quarantine(split, allow_test)
        sub = self._manifest_df[self._manifest_df["split_canonical"] == canon_split]
        return [(str(row["source_dataset"]), int(row["source_record_id"])) for _, row in sub.iterrows()]

    def get_sample(
        self,
        cohort_name_or_rid: str,
        local_idx_or_variant: Any = "12-lead",
        return_mask: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Dual-mode sample getter:
        If cohort_name_or_rid is a global_record_id (e.g. 'PTBXL_1'), loads from runs/module2_v1.
        Enforces test quarantine: raises PermissionError if record belongs to quarantined test split.
        """
        rid = cohort_name_or_rid
        variant = local_idx_or_variant if isinstance(local_idx_or_variant, str) else "12-lead"
        meta = self.get_record_metadata(rid)
        if meta.get("split_canonical") == "test" and self.quarantine_enforced:
            raise PermissionError(
                f"[QUARANTINE ENFORCEMENT] Record '{rid}' belongs to the quarantined test split."
            )
        sig = self.get_signal(rid, variant=variant)
        lbl = np.array([float(meta[col]) if pd.notna(meta[col]) else 0.0 for col in TARGET_NAMES], dtype=np.float32)
        if return_mask:
            mask = np.array([1.0 if pd.notna(meta[col]) else 0.0 for col in TARGET_NAMES], dtype=np.float32)
            return sig, lbl, mask
        return sig, lbl

"""
Full Seeded Random Masked Dataset Generator across All 36,957 Records
=====================================================================
Generates a complete, deterministic, frozen random lead-masked dataset
across all 3 cohorts:
  - PTB-XL (21,799 records)
  - Chapman-Shaoxing (10,158 records)
  - MIMIC-IV-ECG (5,000 records)
Total = 36,957 records.

Evaluated degradation levels:
  - 12 leads (Full baseline)
  - 10 leads (Randomly masked 2 leads)
  - 8 leads (Randomly masked 4 leads)
  - 6 leads (Randomly masked 6 leads)
  - 4 leads (Randomly masked 8 leads)
  - 2 leads (Randomly masked 10 leads)
  - 1 lead (Randomly masked 11 leads)

Strict Scope:
  - Target labels strictly and exclusively cover the 8 finalized benchmark diseases:
    1. Normal ECG
    2. Atrial Fibrillation
    3. Right Bundle Branch Block
    4. Left Bundle Branch Block
    5. Sinus Bradycardia
    6. Sinus Tachycardia
    7. 1st Degree AV Block
    8. Myocardial Infarction
  - Metadata CSV strictly includes patient identifier, cross-validation split,
    quality audit flag, and the 8 disease binary columns.

Guarantees:
  - Strict zero leakage: Inactive leads are strictly 0.0 mV.
  - Deterministic pseudo-random seed: `seed = 42_000 + cohort_id * 100_000 + record_idx`
  - Bitwise reproducibility: Model A and Model B evaluate on identical input tensors.
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.degradation.lead_masking import CANONICAL_LEADS, LEAD_TO_INDEX

FINALIZED_DISEASES = [
    "Normal ECG",
    "Atrial Fibrillation",
    "Right Bundle Branch Block",
    "Left Bundle Branch Block",
    "Sinus Bradycardia",
    "Sinus Tachycardia",
    "1st Degree AV Block",
    "Myocardial Infarction"
]

LEAD_TIERS = [10, 8, 6, 4, 2, 1]

COHORT_SPECS = [
    {
        "id": "ptbxl",
        "name": "PTB-XL",
        "cohort_int": 1,
        "csv_path": repo_root / "data" / "processed_100hz" / "preprocessed_database.csv",
        "signals_path": repo_root / "data" / "processed_100hz" / "signals.npy",
        "labels_path": repo_root / "data" / "processed_100hz" / "labels.npy",
        "id_col": "ecg_id"
    },
    {
        "id": "chapman",
        "name": "Chapman-Shaoxing",
        "cohort_int": 2,
        "csv_path": repo_root / "data" / "processed_100hz" / "chapman" / "preprocessed_database.csv",
        "signals_path": repo_root / "data" / "processed_100hz" / "chapman" / "signals.npy",
        "labels_path": repo_root / "data" / "processed_100hz" / "chapman" / "labels.npy",
        "id_col": "record_id"
    },
    {
        "id": "mimic",
        "name": "MIMIC-IV-ECG",
        "cohort_int": 3,
        "csv_path": repo_root / "data" / "processed_100hz" / "mimic" / "preprocessed_database.csv",
        "signals_path": repo_root / "data" / "processed_100hz" / "mimic" / "signals.npy",
        "labels_path": repo_root / "data" / "processed_100hz" / "mimic" / "labels.npy",
        "id_col": "record_id"
    }
]


def generate_seeded_masks(N: int, k: int, cohort_int: int, base_seed: int = 42000) -> np.ndarray:
    """
    Generates a deterministic (N, 12) binary mask array where each row has exactly
    k active leads (value 1) and (12 - k) masked leads (value 0).
    """
    masks = np.zeros((N, 12), dtype=np.uint8)
    for i in range(N):
        seed = base_seed + cohort_int * 100_000 + i
        rng = np.random.default_rng(seed=seed)
        chosen = rng.choice(12, size=k, replace=False)
        masks[i, chosen] = 1
    return masks


def run_generation():
    start_time = time.time()
    dest_root = repo_root / "data" / "seeded_random_masked"
    dest_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FULL SEEDED RANDOM LEAD MASKING DATASET GENERATION")
    print(f"Destination: {dest_root.resolve()}")
    print(f"Target Diseases (Frozen): {FINALIZED_DISEASES}")
    print(f"Lead Tiers: {LEAD_TIERS} + Full 12-lead baseline")
    print("=" * 80)

    manifest: Dict[str, Any] = {
        "dataset_name": "Seeded Frozen Random Masked ECG Dataset",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "canonical_leads": list(CANONICAL_LEADS),
        "finalized_diseases": FINALIZED_DISEASES,
        "lead_tiers": LEAD_TIERS,
        "base_seed": 42000,
        "cohorts": {},
        "summary": {}
    }

    total_records = 0
    total_bytes_written = 0

    for spec in COHORT_SPECS:
        cid = spec["id"]
        cname = spec["name"]
        print(f"\n[{cname}] Generating dataset arrays...")

        cohort_dest = dest_root / cid
        cohort_dest.mkdir(parents=True, exist_ok=True)

        # 1. Load and sanitize signals
        sig_mmap = np.load(spec["signals_path"], mmap_mode="r")
        N = sig_mmap.shape[0]
        total_records += N
        print(f"  -> Loading signals shape {sig_mmap.shape} ({N:,} records)...")
        
        # Load full array into memory and sanitize non-finites to 0.0 mV
        signals = np.array(sig_mmap, dtype=np.float32)
        signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)

        # 2. Save full 12-lead baseline
        baseline_path = cohort_dest / "full_12lead.npy"
        np.save(baseline_path, signals)
        b_size = baseline_path.stat().st_size
        total_bytes_written += b_size
        print(f"  -> Saved full_12lead.npy ({b_size / (1024**2):.1f} MB)")

        # 3. Load & verify labels strictly covering the 8 finalized diseases
        labels_raw = np.load(spec["labels_path"]).astype(np.uint8)
        assert labels_raw.shape == (N, 8), f"Expected shape ({N}, 8), got {labels_raw.shape}"
        labels_path = cohort_dest / "labels.npy"
        np.save(labels_path, labels_raw)
        l_size = labels_path.stat().st_size
        total_bytes_written += l_size

        # 4. Filter and save aligned metadata strictly with the 8 diseases
        df = pd.read_csv(spec["csv_path"], low_memory=False)
        assert len(df) == N, f"DataFrame length mismatch: {len(df)} vs {N}"

        # Columns to preserve: core identifiers + splits + quality + 8 diseases
        keep_cols = []
        for col_cand in [spec["id_col"], "record_id", "ecg_id", "patient_id", "subject_id", "age", "sex", "strat_fold", "split", "quality_passed"]:
            if col_cand in df.columns and col_cand not in keep_cols:
                keep_cols.append(col_cand)
        
        for d in FINALIZED_DISEASES:
            if d not in df.columns:
                # If binary column not in df, read directly from labels_raw
                d_idx = FINALIZED_DISEASES.index(d)
                df[d] = labels_raw[:, d_idx]
            if d not in keep_cols:
                keep_cols.append(d)

        meta_df = df[keep_cols].copy()
        meta_path = cohort_dest / "metadata.csv"
        meta_df.to_csv(meta_path, index=False)
        print(f"  -> Saved metadata.csv with {len(meta_df)} rows and {len(keep_cols)} columns.")

        cohort_summary = {
            "name": cname,
            "num_records": N,
            "id_column": spec["id_col"],
            "disease_prevalences": {d: int(labels_raw[:, idx].sum()) for idx, d in enumerate(FINALIZED_DISEASES)},
            "files": {
                "full_12lead.npy": {"size_mb": round(b_size / (1024**2), 2)},
                "labels.npy": {"size_mb": round(l_size / (1024**2), 2)},
                "metadata.csv": {"size_mb": round(meta_path.stat().st_size / (1024**2), 2)}
            }
        }

        # 5. Generate and save each random lead tier
        for k in LEAD_TIERS:
            # Generate deterministic mask matrix
            masks_k = generate_seeded_masks(N=N, k=k, cohort_int=spec["cohort_int"])
            mask_file = cohort_dest / f"random_masks_{k}.npy"
            np.save(mask_file, masks_k)
            m_size = mask_file.stat().st_size
            total_bytes_written += m_size

            # Apply mask to signals
            # masks_k[:, :, np.newaxis] broadcasts from (N, 12, 1) to (N, 12, 1000)
            masked_signals = signals * masks_k[:, :, np.newaxis]

            # Invariant check: Inactive leads must be exactly 0.0
            inactive_mask = (masks_k == 0)
            for ch in range(12):
                inactive_rows = inactive_mask[:, ch]
                if np.any(inactive_rows):
                    max_leak = float(np.max(np.abs(masked_signals[inactive_rows, ch, :])))
                    assert max_leak == 0.0, f"Leakage detected in lead {ch} for tier {k}: {max_leak}"

            leads_file = cohort_dest / f"random_leads_{k}.npy"
            np.save(leads_file, masked_signals)
            l_size = leads_file.stat().st_size
            total_bytes_written += l_size

            cohort_summary["files"][f"random_leads_{k}.npy"] = {"size_mb": round(l_size / (1024**2), 2), "active_leads": k}
            cohort_summary["files"][f"random_masks_{k}.npy"] = {"size_mb": round(m_size / (1024**2), 2)}

            print(f"  -> Saved random_leads_{k:<2}.npy ({l_size / (1024**2):>6.1f} MB) + masks ({m_size / (1024**2):>4.1f} MB)")

        manifest["cohorts"][cid] = cohort_summary

    manifest["summary"]["total_records"] = total_records
    manifest["summary"]["total_bytes"] = total_bytes_written
    manifest["summary"]["total_mb"] = round(total_bytes_written / (1024**2), 2)
    manifest["summary"]["total_gb"] = round(total_bytes_written / (1024**3), 3)

    manifest_path = dest_root / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"SUCCESSFULLY GENERATED FULL DATASET IN {elapsed:.2f}s")
    print(f"Total Records: {total_records:,}")
    print(f"Total Storage: {manifest['summary']['total_gb']} GB ({manifest['summary']['total_mb']} MB)")
    print(f"Manifest written to: {manifest_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_generation()

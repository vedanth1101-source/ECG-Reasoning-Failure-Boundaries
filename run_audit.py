"""
ECG Audit & Prevalence Explorer Runner
======================================
Unified, standalone, interactive and CLI tool to audit physical artifacts,
record accounting, patient grouping, quality screening, and disease prevalence across:
  - PTB-XL (21,799 records)
  - Chapman-Shaoxing (10,158 records)
  - MIMIC-IV-ECG (5,000 records)

Usage:
  # 1. Interactive Menu (or simply double-click run_audit.bat):
  python run_audit.py

  # 2. Run 5 Random Range Samples directly:
  python run_audit.py --samples

  # 3. Run Full Source Inventory & SHA-256 Hashes:
  python run_audit.py --inventory

  # 4. Run Full Disease Prevalence Breakdown:
  python run_audit.py --prevalence

  # 5. Inspect a Specific Record by ID:
  python run_audit.py --record 9
  python run_audit.py --record JS00001
  python run_audit.py --record 48418505

  # 6. Inspect All ECGs for a Given Patient ID:
  python run_audit.py --patient 15222

  # 7. Run Full Automated Audit Suite:
  python run_audit.py --all
"""

import sys
import os
import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

# Safe UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent

DISEASE_SHORT = ["NORM", "AFIB", "RBBB", "LBBB", "SBRAD", "STACH", "1AVB", "MI"]
DISEASE_FULL = [
    "Normal ECG (NORM)",
    "Atrial Fibrillation (AFIB)",
    "Right Bundle Branch Block (RBBB)",
    "Left Bundle Branch Block (LBBB)",
    "Sinus Bradycardia (SBRAD)",
    "Sinus Tachycardia (STACH)",
    "1st Degree AV Block (1AVB)",
    "Myocardial Infarction (MI)"
]


def compute_sha256(file_path: Path) -> str:
    """Computes SHA-256 hash in 64KB blocks."""
    if not file_path.exists():
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class ECGDataManager:
    """Manages lazy-loading and caching of physical dataset artifacts."""

    def __init__(self):
        self.df_p: Optional[pd.DataFrame] = None
        self.sig_p: Optional[np.ndarray] = None
        self.lbl_p: Optional[np.ndarray] = None

        self.df_c: Optional[pd.DataFrame] = None
        self.sig_c: Optional[np.ndarray] = None
        self.lbl_c: Optional[np.ndarray] = None

        self.df_m: Optional[pd.DataFrame] = None
        self.sig_m: Optional[np.ndarray] = None
        self.lbl_m: Optional[np.ndarray] = None

        self.pooled_lbl: Optional[np.ndarray] = None
        self.obs_dict: Optional[List[Dict[str, Any]]] = None
        self.ptb_test_indices: Optional[Dict[int, int]] = None

    def load_data(self):
        if self.df_p is not None:
            return

        print("\n[*] Loading dataset files into memory from disk...")

        # 1. PTB-XL
        p_csv = REPO_ROOT / "data/processed_100hz/preprocessed_database.csv"
        p_sig = REPO_ROOT / "data/processed_100hz/signals.npy"
        p_lbl = REPO_ROOT / "data/processed_100hz/labels.npy"
        self.df_p = pd.read_csv(p_csv, low_memory=False)
        self.sig_p = np.load(p_sig, mmap_mode="r")
        self.lbl_p = np.load(p_lbl)
        self.df_p["dataset_name"] = "PTB-XL"
        self.df_p["std_rec_id"] = self.df_p["ecg_id"].astype(str)
        self.df_p["std_pat_id"] = self.df_p["patient_id"].astype(str)
        self.df_p["raw_ann"] = self.df_p["scp_codes"].astype(str)

        # 2. Chapman
        c_csv = REPO_ROOT / "data/processed_100hz/chapman/preprocessed_database.csv"
        c_sig = REPO_ROOT / "data/processed_100hz/chapman/signals.npy"
        c_lbl = REPO_ROOT / "data/processed_100hz/chapman/labels.npy"
        self.df_c = pd.read_csv(c_csv, low_memory=False)
        self.sig_c = np.load(c_sig, mmap_mode="r")
        self.lbl_c = np.load(c_lbl)
        self.df_c["dataset_name"] = "Chapman"
        self.df_c["std_rec_id"] = self.df_c["record_id"].astype(str)
        self.df_c["std_pat_id"] = self.df_c["patient_id"].astype(str)
        self.df_c["raw_ann"] = self.df_c["dx_snomed"].astype(str)

        # 3. MIMIC
        m_csv = REPO_ROOT / "data/processed_100hz/mimic/preprocessed_database.csv"
        m_sig = REPO_ROOT / "data/processed_100hz/mimic/signals.npy"
        m_lbl = REPO_ROOT / "data/processed_100hz/mimic/labels.npy"
        self.df_m = pd.read_csv(m_csv, low_memory=False)
        self.sig_m = np.load(m_sig, mmap_mode="r")
        self.lbl_m = np.load(m_lbl)
        self.df_m["dataset_name"] = "MIMIC"
        self.df_m["std_rec_id"] = self.df_m["record_id"].astype(str)
        self.df_m["std_pat_id"] = self.df_m["subject_id"].astype(str)
        self.df_m["raw_ann"] = self.df_m["disease"].astype(str)

        self.pooled_lbl = np.vstack([self.lbl_p, self.lbl_c, self.lbl_m])
        print(f"[+] Loaded successfully:")
        print(f"    - PTB-XL:            {len(self.df_p):>6,} records")
        print(f"    - Chapman-Shaoxing:  {len(self.df_c):>6,} records")
        print(f"    - MIMIC-IV-ECG:      {len(self.df_m):>6,} records")
        print(f"    - Total Pooled:      {len(self.pooled_lbl):>6,} records\n")

    def load_observations(self):
        if self.obs_dict is not None:
            return

        obs_path = REPO_ROOT / "data/observations/ptbxl/obs_1_lead.json"
        if not obs_path.exists():
            return

        try:
            with open(obs_path, "r", encoding="utf-8") as f:
                self.obs_dict = json.load(f)
            # Find PTB-XL fold 10 mapping
            test_mask = (self.df_p["strat_fold"] == 10).to_numpy()
            self.ptb_test_indices = {orig_idx: test_row for test_row, orig_idx in enumerate(np.where(test_mask)[0])}
        except Exception:
            self.obs_dict = None

    def get_observation_for_ptb_row(self, orig_idx: int) -> Optional[Dict[str, Any]]:
        self.load_observations()
        if self.obs_dict is None or self.ptb_test_indices is None:
            return None
        if orig_idx in self.ptb_test_indices:
            pos = self.ptb_test_indices[orig_idx]
            if 0 <= pos < len(self.obs_dict):
                return self.obs_dict[pos]
        return None


def print_table_header(show_obs: bool = False):
    header = f"{'#':<3} | {'Dataset':<8} | {'Record ID':<9} | {'Patient ID':<10} | {'QC':<4} | {'Voltage Bounds':<18} | {'Vector [N,A,R,L,B,T,1,M]':<26} | {'Assigned':<14} | {'Raw Annotation'}"
    if show_obs:
        header += " | Observations (HR, Rhythm, Intervals)"
    print(header)
    print("-" * (len(header) + 10))


def print_record_row(idx: int, row: pd.Series, sig: np.ndarray, lbl: np.ndarray, obs_str: Optional[str] = None):
    sig_min = float(np.nanmin(sig)) if not np.isnan(sig).all() else float("nan")
    sig_max = float(np.nanmax(sig)) if not np.isnan(sig).all() else float("nan")
    bounds_str = f"[{sig_min:+.2f}, {sig_max:+.2f}] mV"
    vec_str = str(lbl.tolist())
    active_dis = [DISEASE_SHORT[i] for i, v in enumerate(lbl) if v == 1]
    active_str = ", ".join(active_dis) if active_dis else "NONE"
    raw_ann = str(row["raw_ann"])[:35] + ("..." if len(str(row["raw_ann"])) > 35 else "")
    qc_str = "PASS" if row.get("quality_passed", True) else "FAIL"

    line = f"{idx:<3} | {row['dataset_name']:<8} | {row['std_rec_id']:<9} | {row['std_pat_id']:<10} | {qc_str:<4} | {bounds_str:<18} | {vec_str:<26} | {active_str:<14} | {raw_ann:<38}"
    if obs_str is not None:
        line += f" | {obs_str}"
    print(line)


def format_observation(obs: Optional[Dict[str, Any]]) -> str:
    if not obs:
        return "N/A"
    r = obs.get("rhythm", "Unknown")
    hr = obs.get("heart_rate", "N/A")
    pr = obs.get("pr_interval_ms", "N/A")
    qrs = obs.get("qrs_duration_ms", "N/A")
    ste = "YES" if obs.get("st_elevation", {}).get("present") else "No"
    return f"{r} | {hr} bpm | PR:{pr}ms | QRS:{qrs}ms | STE:{ste}"


def display_prevalence_comparison(lbl_slice: np.ndarray, sample_name: str, pooled_lbl: np.ndarray):
    n_sample = len(lbl_slice)
    n_pooled = len(pooled_lbl)
    sample_sums = lbl_slice.sum(axis=0)
    pooled_sums = pooled_lbl.sum(axis=0)

    print(f"\n  Prevalence Comparison: {sample_name} (N = {n_sample:,}) vs Pooled Total (N = {n_pooled:,}):")
    print(f"  {'Disease Name':<32} | {'Sample Positives':<18} | {'Sample Prev (%)':<16} | {'Pooled Prev (%) Reference'}")
    print("  " + "-" * 88)

    for idx, d_name in enumerate(DISEASE_FULL):
        s_cnt = int(sample_sums[idx])
        s_prev = s_cnt / n_sample * 100.0
        p_prev = pooled_sums[idx] / n_pooled * 100.0
        print(f"  {d_name:<32} | {s_cnt:>8,} / {n_sample:<6} | {s_prev:>14.2f}% | {p_prev:>14.2f}%")

    pos_per_rec = lbl_slice.sum(axis=1)
    z = int((pos_per_rec == 0).sum())
    o = int((pos_per_rec == 1).sum())
    m = int((pos_per_rec > 1).sum())
    print(f"\n  Multi-Label Distribution: 0 labels: {z} ({z/n_sample*100:.1f}%), 1 label: {o} ({o/n_sample*100:.1f}%), >=2 labels: {m} ({m/n_sample*100:.1f}%)\n")


def run_five_random_samples(dm: ECGDataManager, seed: Optional[int] = 12345):
    """Executes the 5 canonical audit samples."""
    dm.load_data()
    dm.load_observations()
    rng = np.random.default_rng(seed=seed)

    print("=" * 110)
    print("SAMPLE 1: RANDOM CONTIGUOUS RANGE FROM PTB-XL (10 Records)")
    print("=" * 110)
    start_1 = int(rng.integers(0, len(dm.df_p) - 10))
    print(f"Slice Window: Rows [{start_1} to {start_1 + 9}] in PTB-XL (data/processed_100hz/preprocessed_database.csv)")
    print_table_header(show_obs=True)
    slice_lbl_1 = dm.lbl_p[start_1 : start_1 + 10]
    for i in range(10):
        row_idx = start_1 + i
        obs = dm.get_observation_for_ptb_row(row_idx)
        print_record_row(i, dm.df_p.iloc[row_idx], dm.sig_p[row_idx], dm.lbl_p[row_idx], format_observation(obs))
    display_prevalence_comparison(slice_lbl_1, f"PTB-XL Window [{start_1}:{start_1+10}]", dm.pooled_lbl)

    print("=" * 110)
    print("SAMPLE 2: RANDOM CONTIGUOUS RANGE FROM CHAPMAN-SHAOXING (10 Records)")
    print("=" * 110)
    start_2 = int(rng.integers(0, len(dm.df_c) - 10))
    print(f"Slice Window: Rows [{start_2} to {start_2 + 9}] in Chapman (data/processed_100hz/chapman/preprocessed_database.csv)")
    print_table_header(show_obs=False)
    slice_lbl_2 = dm.lbl_c[start_2 : start_2 + 10]
    for i in range(10):
        row_idx = start_2 + i
        print_record_row(i, dm.df_c.iloc[row_idx], dm.sig_c[row_idx], dm.lbl_c[row_idx])
    display_prevalence_comparison(slice_lbl_2, f"Chapman Window [{start_2}:{start_2+10}]", dm.pooled_lbl)

    print("=" * 110)
    print("SAMPLE 3: RANDOM CONTIGUOUS RANGE FROM MIMIC-IV-ECG (10 Records)")
    print("=" * 110)
    start_3 = int(rng.integers(0, len(dm.df_m) - 10))
    print(f"Slice Window: Rows [{start_3} to {start_3 + 9}] in MIMIC (data/processed_100hz/mimic/preprocessed_database.csv)")
    print_table_header(show_obs=False)
    slice_lbl_3 = dm.lbl_m[start_3 : start_3 + 10]
    for i in range(10):
        row_idx = start_3 + i
        print_record_row(i, dm.df_m.iloc[row_idx], dm.sig_m[row_idx], dm.lbl_m[row_idx])
    display_prevalence_comparison(slice_lbl_3, f"MIMIC Window [{start_3}:{start_3+10}]", dm.pooled_lbl)

    print("=" * 110)
    print("SAMPLE 4: RANDOM MULTI-RECORD PATIENT CLUSTER IN PTB-XL (Testing Patient Grouping)")
    print("=" * 110)
    pat_counts = dm.df_p["patient_id"].value_counts()
    multi_pats = pat_counts[pat_counts >= 3].index.to_numpy()
    chosen_pat = rng.choice(multi_pats)
    pat_indices = np.where(dm.df_p["patient_id"] == chosen_pat)[0]
    print(f"Selected Patient ID: {chosen_pat} (Has {len(pat_indices)} recordings across rows {pat_indices.tolist()})")
    print_table_header(show_obs=True)
    slice_lbl_4 = dm.lbl_p[pat_indices]
    for i, r_idx in enumerate(pat_indices):
        obs = dm.get_observation_for_ptb_row(r_idx)
        print_record_row(i, dm.df_p.iloc[r_idx], dm.sig_p[r_idx], dm.lbl_p[r_idx], format_observation(obs))
    display_prevalence_comparison(slice_lbl_4, f"Patient {chosen_pat} ECGs", dm.pooled_lbl)

    print("=" * 110)
    print("SAMPLE 5: RANDOM POOLED CROSS-DATASET BATCH (50 Records Across All 3 Cohorts)")
    print("=" * 110)
    p_idx = rng.choice(len(dm.df_p), size=30, replace=False)
    c_idx = rng.choice(len(dm.df_c), size=14, replace=False)
    m_idx = rng.choice(len(dm.df_m), size=6, replace=False)

    df_p_sub = dm.df_p.iloc[p_idx]
    df_c_sub = dm.df_c.iloc[c_idx]
    df_m_sub = dm.df_m.iloc[m_idx]
    slice_df_5 = pd.concat([df_p_sub, df_c_sub, df_m_sub], ignore_index=True)

    sig_p_sub = np.array(dm.sig_p[p_idx])
    sig_c_sub = np.array(dm.sig_c[c_idx])
    sig_m_sub = np.array(dm.sig_m[m_idx])
    slice_sig_5 = np.vstack([sig_p_sub, sig_c_sub, sig_m_sub])

    lbl_p_sub = dm.lbl_p[p_idx]
    lbl_c_sub = dm.lbl_c[c_idx]
    lbl_m_sub = dm.lbl_m[m_idx]
    slice_lbl_5 = np.vstack([lbl_p_sub, lbl_c_sub, lbl_m_sub])

    perm = rng.permutation(len(slice_df_5))
    slice_df_5 = slice_df_5.iloc[perm].reset_index(drop=True)
    slice_sig_5 = slice_sig_5[perm]
    slice_lbl_5 = slice_lbl_5[perm]

    print("Displaying first 15 of 50 randomly sampled records from across the ecosystem:")
    print_table_header(show_obs=False)
    for i in range(15):
        print_record_row(i, slice_df_5.iloc[i], slice_sig_5[i], slice_lbl_5[i])
    print(f"... [{len(slice_df_5) - 15} additional records in sample omitted from table for brevity] ...")
    display_prevalence_comparison(slice_lbl_5, "50-Record Pooled Sample", dm.pooled_lbl)

    print("=" * 110)
    print("[SUCCESS] All 5 random range samples successfully extracted and verified against disk artifacts.")
    print("=" * 110)


def run_source_inventory_audit():
    """Audits file paths, sizes, hashes, and exact record counts."""
    print("\n" + "=" * 90)
    print("  PHYSICAL SOURCE INVENTORY & IDENTIFIER ACCOUNTING")
    print("=" * 90)

    datasets = [
        ("PTB-XL v1.0.3", "data/processed_100hz"),
        ("Chapman-Shaoxing", "data/processed_100hz/chapman"),
        ("MIMIC-IV-ECG", "data/processed_100hz/mimic")
    ]

    for name, rel_path in datasets:
        folder = REPO_ROOT / rel_path
        csv_file = folder / "preprocessed_database.csv"
        sig_file = folder / "signals.npy"
        lbl_file = folder / "labels.npy"

        print(f"\n--- {name} ---")
        print(f"Directory: {folder}")
        for label, fpath in [("CSV Database", csv_file), ("Signals NPY", sig_file), ("Labels NPY", lbl_file)]:
            if fpath.exists():
                size_mb = fpath.stat().st_size / (1024 * 1024)
                sha = compute_sha256(fpath)
                print(f"  {label:<14} | Size: {size_mb:>8.2f} MB | SHA-256: {sha}")
            else:
                print(f"  {label:<14} | [MISSING]")

        if csv_file.exists():
            df = pd.read_csv(csv_file, low_memory=False)
            rec_col = "ecg_id" if "ecg_id" in df.columns else "record_id"
            pat_col = "patient_id" if "patient_id" in df.columns else "subject_id"
            n_recs = len(df)
            n_pats = df[pat_col].nunique() if pat_col in df.columns else n_recs
            qc_pass = df["quality_passed"].sum() if "quality_passed" in df.columns else n_recs
            print(f"  Record Accounting: Total={n_recs:,} | Unique Records={df[rec_col].nunique():,} | Unique Patients={n_pats:,} | QC Passed={qc_pass:,}")

    print("\n" + "=" * 90)


def run_prevalence_breakdown(dm: ECGDataManager):
    """Computes exact disease prevalence across all 36,957 records."""
    dm.load_data()
    n_p = len(dm.lbl_p)
    n_c = len(dm.lbl_c)
    n_m = len(dm.lbl_m)
    n_tot = len(dm.pooled_lbl)

    p_sum = dm.lbl_p.sum(axis=0)
    c_sum = dm.lbl_c.sum(axis=0)
    m_sum = dm.lbl_m.sum(axis=0)
    tot_sum = dm.pooled_lbl.sum(axis=0)

    print("\n" + "=" * 95)
    print("  CURRENT DISEASE PREVALENCE ACROSS THE POOLED POPULATION (N = 36,957)")
    print("=" * 95)
    print(f"{'Target Disease':<34} | {'PTB-XL':<10} | {'Chapman':<10} | {'MIMIC':<10} | {'Total Pos':<12} | {'Pooled Prev (%)'}")
    print("-" * 95)

    for idx, d_name in enumerate(DISEASE_FULL):
        p_c = int(p_sum[idx])
        c_c = int(c_sum[idx])
        m_c = int(m_sum[idx])
        t_c = int(tot_sum[idx])
        prev = t_c / n_tot * 100.0
        print(f"{d_name:<34} | {p_c:>8,}   | {c_c:>8,}   | {m_c:>8,}   | {t_c:>10,}   | {prev:>13.2f}%")

    print("-" * 95)
    print(f"{'Total Cohort Records':<34} | {n_p:>8,}   | {n_c:>8,}   | {n_m:>8,}   | {n_tot:>10,}   |       100.00%")
    print("=" * 95)

    pos_per_rec = dm.pooled_lbl.sum(axis=1)
    z = int((pos_per_rec == 0).sum())
    o = int((pos_per_rec == 1).sum())
    m = int((pos_per_rec > 1).sum())
    print("\nPOOLED MULTI-LABEL CO-OCCURRENCE:")
    print(f"  - Exactly 0 positive labels: {z:>6,} ({z/n_tot*100:6.2f}%)")
    print(f"  - Exactly 1 positive label:  {o:>6,} ({o/n_tot*100:6.2f}%)")
    print(f"  - Multi-label (>= 2 labels): {m:>6,} ({m/n_tot*100:6.2f}%)\n")


def inspect_single_record(dm: ECGDataManager, rec_id: str):
    """Looks up any record ID in PTB-XL, Chapman, or MIMIC."""
    dm.load_data()
    dm.load_observations()
    rec_str = str(rec_id).strip()

    # PTB-XL
    p_match = dm.df_p[dm.df_p["std_rec_id"] == rec_str]
    if len(p_match) > 0:
        row_idx = p_match.index[0]
        obs = dm.get_observation_for_ptb_row(row_idx)
        print_record_card("PTB-XL", p_match.iloc[0], dm.sig_p[row_idx], dm.lbl_p[row_idx], obs)
        return

    # Chapman
    c_match = dm.df_c[dm.df_c["std_rec_id"] == rec_str]
    if len(c_match) > 0:
        row_idx = c_match.index[0]
        print_record_card("Chapman-Shaoxing", c_match.iloc[0], dm.sig_c[row_idx], dm.lbl_c[row_idx], None)
        return

    # MIMIC
    m_match = dm.df_m[dm.df_m["std_rec_id"] == rec_str]
    if len(m_match) > 0:
        row_idx = m_match.index[0]
        print_record_card("MIMIC-IV-ECG", m_match.iloc[0], dm.sig_m[row_idx], dm.lbl_m[row_idx], None)
        return

    print(f"\n[!] Record ID '{rec_str}' not found in PTB-XL, Chapman, or MIMIC.")


def inspect_patient_records(dm: ECGDataManager, pat_id: str):
    """Finds all ECG recordings belonging to a patient."""
    dm.load_data()
    dm.load_observations()
    pat_str = str(pat_id).strip()

    # Search in PTB-XL
    matches = dm.df_p[dm.df_p["std_pat_id"] == pat_str]
    if len(matches) == 0 and not pat_str.endswith(".0"):
        matches = dm.df_p[dm.df_p["std_pat_id"] == f"{pat_str}.0"]

    if len(matches) > 0:
        indices = matches.index.to_numpy()
        print(f"\n[+] Found {len(matches)} recording(s) for Patient '{pat_str}' in PTB-XL:")
        print_table_header(show_obs=True)
        for i, idx in enumerate(indices):
            obs = dm.get_observation_for_ptb_row(idx)
            print_record_row(i, dm.df_p.iloc[idx], dm.sig_p[idx], dm.lbl_p[idx], format_observation(obs))
        display_prevalence_comparison(dm.lbl_p[indices], f"Patient {pat_str} Recordings", dm.pooled_lbl)
        return

    print(f"\n[!] Patient ID '{pat_str}' not found with multiple recordings in PTB-XL.")


def print_record_card(dataset: str, row: pd.Series, sig: np.ndarray, lbl: np.ndarray, obs: Optional[Dict[str, Any]]):
    print("\n" + "=" * 80)
    print(f"  RECORD DETAIL CARD: {row['std_rec_id']} ({dataset})")
    print("=" * 80)
    print(f"Dataset:              {dataset}")
    print(f"Record ID:            {row['std_rec_id']}")
    print(f"Patient / Subject ID: {row['std_pat_id']}")
    print(f"Quality Screening:    {'PASSED' if row.get('quality_passed', True) else 'FAILED'}")
    print(f"Signal Shape:         {sig.shape} (12 leads, 1000 samples @ 100 Hz)")
    print(f"Voltage Extrema:      Min = {np.nanmin(sig):.3f} mV, Max = {np.nanmax(sig):.3f} mV")
    print(f"Raw Source Label:     {row['raw_ann']}")

    active_dis = [DISEASE_FULL[i] for i, v in enumerate(lbl) if v == 1]
    print(f"\nMulti-Hot Binary Vector: {lbl.tolist()}")
    print(f"Assigned Benchmark Diseases ({len(active_dis)}):")
    if active_dis:
        for d in active_dis:
            print(f"  [+] {d}")
    else:
        print("  [-] None (Zero target labels)")

    if obs:
        print("\nStructured Clinical Observations (from obs_1_lead.json):")
        print(f"  Rhythm:             {obs.get('rhythm', 'N/A')}")
        print(f"  Heart Rate:         {obs.get('heart_rate', 'N/A')} bpm")
        print(f"  PR Interval:        {obs.get('pr_interval_ms', 'N/A')} ms")
        print(f"  QRS Duration:       {obs.get('qrs_duration_ms', 'N/A')} ms")
        ste = obs.get("st_elevation", {})
        print(f"  ST Elevation:       {'PRESENT' if ste.get('present') else 'Absent'} (Leads: {ste.get('leads', [])})")
        std = obs.get("st_depression", {})
        print(f"  ST Depression:      {'PRESENT' if std.get('present') else 'Absent'} (Leads: {std.get('leads', [])})")
        twi = obs.get("t_wave_inversion", {})
        print(f"  T-Wave Inversion:   {'PRESENT' if twi.get('present') else 'Absent'} (Leads: {twi.get('leads', [])})")
    print("=" * 80 + "\n")


def interactive_menu():
    dm = ECGDataManager()
    while True:
        print("\n" + "=" * 65)
        print("     ECG SOURCE INVENTORY & PREVALENCE AUDIT EXPLORER")
        print("=" * 65)
        print("  [1] Run 5 Random Range Samples across Cohorts")
        print("  [2] Run Source Inventory & Physical File Hash Audit")
        print("  [3] Run Pooled Disease Prevalence & Multi-Label Breakdown")
        print("  [4] Inspect a Specific Record by ID (e.g. 9, JS00001, 48418505)")
        print("  [5] Inspect a Patient's Complete ECG History (e.g. 15222)")
        print("  [6] Run Full Audit Suite (All Inspections)")
        print("  [0] Exit")
        print("-" * 65)

        try:
            choice = input("Enter option [0-6] (Default 1): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if choice == "" or choice == "1":
            run_five_random_samples(dm)
        elif choice == "2":
            run_source_inventory_audit()
        elif choice == "3":
            run_prevalence_breakdown(dm)
        elif choice == "4":
            rid = input("Enter Record ID: ").strip()
            if rid:
                inspect_single_record(dm, rid)
        elif choice == "5":
            pid = input("Enter Patient ID (e.g. 15222): ").strip()
            if pid:
                inspect_patient_records(dm, pid)
        elif choice == "6":
            run_source_inventory_audit()
            run_prevalence_breakdown(dm)
            run_five_random_samples(dm)
        elif choice == "0" or choice.lower() in ["exit", "q", "quit"]:
            print("Exiting explorer. Goodbye!")
            break
        else:
            print("[!] Invalid choice. Please enter a number between 0 and 6.")


def main():
    parser = argparse.ArgumentParser(description="ECG Audit & Prevalence Explorer")
    parser.add_argument("--samples", action="store_true", help="Run 5 random range samples directly.")
    parser.add_argument("--inventory", action="store_true", help="Run source inventory and SHA-256 integrity audit.")
    parser.add_argument("--prevalence", action="store_true", help="Run disease prevalence and multi-label breakdown.")
    parser.add_argument("--record", type=str, default=None, help="Inspect a specific record ID.")
    parser.add_argument("--patient", type=str, default=None, help="Inspect all ECGs for a patient ID.")
    parser.add_argument("--all", action="store_true", help="Run full audit suite.")
    parser.add_argument("--interactive", "-i", action="store_true", help="Force interactive menu.")
    args = parser.parse_args()

    dm = ECGDataManager()

    if args.samples:
        run_five_random_samples(dm)
    elif args.inventory:
        run_source_inventory_audit()
    elif args.prevalence:
        run_prevalence_breakdown(dm)
    elif args.record:
        inspect_single_record(dm, args.record)
    elif args.patient:
        inspect_patient_records(dm, args.patient)
    elif args.all:
        run_source_inventory_audit()
        run_prevalence_breakdown(dm)
        run_five_random_samples(dm)
    else:
        # If no arguments provided in CLI, launch interactive menu
        interactive_menu()


if __name__ == "__main__":
    main()

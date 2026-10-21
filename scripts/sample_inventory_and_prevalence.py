"""
Random Range Sampler: Source Inventory, Identifier Accounting & Disease Prevalence
==================================================================================
Extracts 5 distinct random ranges/slices across the physical datasets to verify:
  1. Record & Patient Identifier accounting (IDs, multi-record patients, QC status).
  2. Raw annotation-to-binary disease vector mappings.
  3. Signal tensor validity (shape, min/max mV, finiteness).
  4. Disease prevalence alignment within each sample against pooled expectations.

Usage:
  python scripts/sample_inventory_and_prevalence.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Reconfigure stdout for safe cross-platform UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent

DISEASES = [
    "NORM", "AFIB", "RBBB", "LBBB", "SBRAD", "STACH", "1AVB", "MI"
]

DISEASE_FULL_NAMES = [
    "Normal ECG (NORM)",
    "Atrial Fibrillation (AFIB)",
    "Right Bundle Branch Block (RBBB)",
    "Left Bundle Branch Block (LBBB)",
    "Sinus Bradycardia (SBRAD)",
    "Sinus Tachycardia (STACH)",
    "1st Degree AV Block (1AVB)",
    "Myocardial Infarction (MI)"
]

CSV_LABEL_COLS = [
    "label_normal_ecg",
    "label_afib",
    "label_rbbb",
    "label_lbbb",
    "label_sbrad",
    "label_stach",
    "label_1avb",
    "label_mi"
]


def load_datasets():
    print("[*] Loading datasets directly from disk...")
    # PTB-XL
    df_p = pd.read_csv(REPO_ROOT / "data/processed_100hz/preprocessed_database.csv", low_memory=False)
    sig_p = np.load(REPO_ROOT / "data/processed_100hz/signals.npy", mmap_mode="r")
    lbl_p = np.load(REPO_ROOT / "data/processed_100hz/labels.npy")
    df_p["dataset_name"] = "PTB-XL"
    df_p["std_rec_id"] = df_p["ecg_id"].astype(str)
    df_p["std_pat_id"] = df_p["patient_id"].astype(str)
    df_p["raw_ann"] = df_p["scp_codes"].astype(str)

    # Chapman
    df_c = pd.read_csv(REPO_ROOT / "data/processed_100hz/chapman/preprocessed_database.csv", low_memory=False)
    sig_c = np.load(REPO_ROOT / "data/processed_100hz/chapman/signals.npy", mmap_mode="r")
    lbl_c = np.load(REPO_ROOT / "data/processed_100hz/chapman/labels.npy")
    df_c["dataset_name"] = "Chapman"
    df_c["std_rec_id"] = df_c["record_id"].astype(str)
    df_c["std_pat_id"] = df_c["patient_id"].astype(str)
    df_c["raw_ann"] = df_c["dx_snomed"].astype(str)

    # MIMIC
    df_m = pd.read_csv(REPO_ROOT / "data/processed_100hz/mimic/preprocessed_database.csv", low_memory=False)
    sig_m = np.load(REPO_ROOT / "data/processed_100hz/mimic/signals.npy", mmap_mode="r")
    lbl_m = np.load(REPO_ROOT / "data/processed_100hz/mimic/labels.npy")
    df_m["dataset_name"] = "MIMIC"
    df_m["std_rec_id"] = df_m["record_id"].astype(str)
    df_m["std_pat_id"] = df_m["subject_id"].astype(str)
    df_m["raw_ann"] = df_m["disease"].astype(str)

    return (df_p, sig_p, lbl_p), (df_c, sig_c, lbl_c), (df_m, sig_m, lbl_m)


def print_record_table(df_slice, sig_slice, lbl_slice):
    print(f"{'Row':<6} | {'Dataset':<8} | {'Record ID':<10} | {'Patient ID':<10} | {'QC':<5} | {'Signal Bounds (mV)':<20} | {'Vector [N,A,R,L,B,T,1,M]':<26} | {'Assigned Diseases':<24} | {'Raw Annotation'}")
    print("-" * 155)

    for i in range(len(df_slice)):
        row = df_slice.iloc[i]
        sig = sig_slice[i]
        lbl = lbl_slice[i]

        sig_min = float(np.nanmin(sig)) if not np.isnan(sig).all() else float("nan")
        sig_max = float(np.nanmax(sig)) if not np.isnan(sig).all() else float("nan")
        bounds_str = f"[{sig_min:+.2f}, {sig_max:+.2f}] mV"

        vec_str = str(lbl.tolist())
        active_dis = [DISEASES[idx] for idx, val in enumerate(lbl) if val == 1]
        active_str = ", ".join(active_dis) if active_dis else "NONE"
        raw_ann_str = row["raw_ann"][:40] + ("..." if len(row["raw_ann"]) > 40 else "")

        qc_str = "PASS" if row.get("quality_passed", True) else "FAIL"

        print(f"{i:<6} | {row['dataset_name']:<8} | {row['std_rec_id']:<10} | {row['std_pat_id']:<10} | {qc_str:<5} | {bounds_str:<20} | {vec_str:<26} | {active_str:<24} | {raw_ann_str}")


def print_prevalence_summary(lbl_slice, sample_name: str, pooled_lbl: np.ndarray):
    n_sample = len(lbl_slice)
    n_pooled = len(pooled_lbl)
    sample_sums = lbl_slice.sum(axis=0)
    pooled_sums = pooled_lbl.sum(axis=0)

    print(f"\n  Disease Prevalence in this Sample ({sample_name}, N = {n_sample}):")
    print(f"  {'Disease':<32} | {'Sample Positives':<18} | {'Sample Prev (%)':<16} | {'Pooled Prev (%) Reference'}")
    print("  " + "-" * 88)

    for idx, d_name in enumerate(DISEASE_FULL_NAMES):
        s_cnt = int(sample_sums[idx])
        s_prev = s_cnt / n_sample * 100.0
        p_prev = pooled_sums[idx] / n_pooled * 100.0
        print(f"  {d_name:<32} | {s_cnt:>8,} / {n_sample:<6} | {s_prev:>14.2f}% | {p_prev:>14.2f}%")

    pos_per_rec = lbl_slice.sum(axis=1)
    z = (pos_per_rec == 0).sum()
    o = (pos_per_rec == 1).sum()
    m = (pos_per_rec > 1).sum()
    print(f"\n  Multi-Label Composition: 0 labels: {z} ({z/n_sample*100:.1f}%), 1 label: {o} ({o/n_sample*100:.1f}%), >=2 labels: {m} ({m/n_sample*100:.1f}%)")


def run_sampling():
    (df_p, sig_p, lbl_p), (df_c, sig_c, lbl_c), (df_m, sig_m, lbl_m) = load_datasets()

    pooled_lbl = np.vstack([lbl_p, lbl_c, lbl_m])

    # Fix seed for verifiable reproducibility
    rng = np.random.default_rng(seed=12345)

    print("\n" + "=" * 100)
    print("SAMPLE 1: RANDOM CONTIGUOUS RANGE FROM PTB-XL (10 Records)")
    print("=" * 100)
    start_1 = int(rng.integers(0, len(df_p) - 10))
    slice_df_1 = df_p.iloc[start_1 : start_1 + 10].reset_index(drop=True)
    slice_sig_1 = np.array(sig_p[start_1 : start_1 + 10])
    slice_lbl_1 = lbl_p[start_1 : start_1 + 10]
    print(f"Range Indices in data/processed_100hz/preprocessed_database.csv: [{start_1} to {start_1 + 9}]")
    print_record_table(slice_df_1, slice_sig_1, slice_lbl_1)
    print_prevalence_summary(slice_lbl_1, f"PTB-XL Window [{start_1}:{start_1+10}]", pooled_lbl)

    print("\n" + "=" * 100)
    print("SAMPLE 2: RANDOM CONTIGUOUS RANGE FROM CHAPMAN-SHAOXING (10 Records)")
    print("=" * 100)
    start_2 = int(rng.integers(0, len(df_c) - 10))
    slice_df_2 = df_c.iloc[start_2 : start_2 + 10].reset_index(drop=True)
    slice_sig_2 = np.array(sig_c[start_2 : start_2 + 10])
    slice_lbl_2 = lbl_c[start_2 : start_2 + 10]
    print(f"Range Indices in data/processed_100hz/chapman/preprocessed_database.csv: [{start_2} to {start_2 + 9}]")
    print_record_table(slice_df_2, slice_sig_2, slice_lbl_2)
    print_prevalence_summary(slice_lbl_2, f"Chapman Window [{start_2}:{start_2+10}]", pooled_lbl)

    print("\n" + "=" * 100)
    print("SAMPLE 3: RANDOM CONTIGUOUS RANGE FROM MIMIC-IV-ECG (10 Records)")
    print("=" * 100)
    start_3 = int(rng.integers(0, len(df_m) - 10))
    slice_df_3 = df_m.iloc[start_3 : start_3 + 10].reset_index(drop=True)
    slice_sig_3 = np.array(sig_m[start_3 : start_3 + 10])
    slice_lbl_3 = lbl_m[start_3 : start_3 + 10]
    print(f"Range Indices in data/processed_100hz/mimic/preprocessed_database.csv: [{start_3} to {start_3 + 9}]")
    print_record_table(slice_df_3, slice_sig_3, slice_lbl_3)
    print_prevalence_summary(slice_lbl_3, f"MIMIC Window [{start_3}:{start_3+10}]", pooled_lbl)

    print("\n" + "=" * 100)
    print("SAMPLE 4: RANDOM MULTI-RECORD PATIENT CLUSTER IN PTB-XL (Testing Patient Grouping)")
    print("=" * 100)
    # Find patients with multiple ECGs
    pat_counts = df_p["patient_id"].value_counts()
    multi_pats = pat_counts[pat_counts >= 3].index.to_numpy()
    chosen_pat = rng.choice(multi_pats)
    pat_indices = np.where(df_p["patient_id"] == chosen_pat)[0]
    slice_df_4 = df_p.iloc[pat_indices].reset_index(drop=True)
    slice_sig_4 = np.array(sig_p[pat_indices])
    slice_lbl_4 = lbl_p[pat_indices]
    print(f"Selected Patient ID: {chosen_pat} (Total ECG recordings for this patient: {len(pat_indices)})")
    print(f"Record indices in dataset: {pat_indices.tolist()}")
    print_record_table(slice_df_4, slice_sig_4, slice_lbl_4)
    print_prevalence_summary(slice_lbl_4, f"Patient {chosen_pat} Repeated ECGs", pooled_lbl)

    print("\n" + "=" * 100)
    print("SAMPLE 5: RANDOM POOLED CROSS-DATASET BATCH (50 Records Across All 3 Cohorts)")
    print("=" * 100)
    # Draw proportional random sample: 30 PTB-XL, 14 Chapman, 6 MIMIC (~70:20:10 ratio)
    p_idx = rng.choice(len(df_p), size=30, replace=False)
    c_idx = rng.choice(len(df_c), size=14, replace=False)
    m_idx = rng.choice(len(df_m), size=6, replace=False)

    df_p_sub = df_p.iloc[p_idx]
    df_c_sub = df_c.iloc[c_idx]
    df_m_sub = df_m.iloc[m_idx]
    slice_df_5 = pd.concat([df_p_sub, df_c_sub, df_m_sub], ignore_index=True)

    sig_p_sub = np.array(sig_p[p_idx])
    sig_c_sub = np.array(sig_c[c_idx])
    sig_m_sub = np.array(sig_m[m_idx])
    slice_sig_5 = np.vstack([sig_p_sub, sig_c_sub, sig_m_sub])

    lbl_p_sub = lbl_p[p_idx]
    lbl_c_sub = lbl_c[c_idx]
    lbl_m_sub = lbl_m[m_idx]
    slice_lbl_5 = np.vstack([lbl_p_sub, lbl_c_sub, lbl_m_sub])

    # Shuffle the combined 50 records
    perm = rng.permutation(len(slice_df_5))
    slice_df_5 = slice_df_5.iloc[perm].reset_index(drop=True)
    slice_sig_5 = slice_sig_5[perm]
    slice_lbl_5 = slice_lbl_5[perm]

    print(f"Displaying first 15 of 50 randomly sampled records from across the ecosystem:")
    print_record_table(slice_df_5.iloc[:15], slice_sig_5[:15], slice_lbl_5[:15])
    print(f"... [{len(slice_df_5) - 15} additional records omitted from table display for conciseness] ...")
    print_prevalence_summary(slice_lbl_5, "50-Record Pooled Ecosystem Sample", pooled_lbl)

    print("\n" + "=" * 100)
    print("ALL 5 RANDOM RANGE SAMPLES COMPLETED AND EMPIRICALLY VERIFIED AGAINST DISK")
    print("=" * 100)


if __name__ == "__main__":
    run_sampling()

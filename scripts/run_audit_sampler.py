"""
Interactive & Automated Audit Sampler
=====================================
Inspects Source Inventory, Record & Patient Identifiers, Disease Prevalence,
and Clinical Observations across PTB-XL, Chapman-Shaoxing, and MIMIC-IV-ECG.

Usage:
  # 1. Run 5 random range samples directly:
  python scripts/run_audit_sampler.py

  # 2. Run with clinical observation details (rhythm, HR, PR, QRS, ST-elevation):
  python scripts/run_audit_sampler.py --show-obs

  # 3. Inspect a specific record by ID:
  python scripts/run_audit_sampler.py --record-id 9
  python scripts/run_audit_sampler.py --record-id JS00001
  python scripts/run_audit_sampler.py --record-id 48418505

  # 4. Interactive menu mode:
  python scripts/run_audit_sampler.py -i
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

# Safe cross-platform UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent

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


class DatasetManager:
    """Loads and caches dataset arrays and metadata on demand."""

    def __init__(self):
        self.df_p = None
        self.sig_p = None
        self.lbl_p = None

        self.df_c = None
        self.sig_c = None
        self.lbl_c = None

        self.df_m = None
        self.sig_m = None
        self.lbl_m = None

        self.obs_train = None
        self.obs_val = None
        self.obs_test = None
        self.ptb_test_indices = None

    def load_all(self):
        if self.df_p is not None:
            return

        print("[*] Loading dataset manifests, signals, and multi-hot labels from disk...")
        # 1. PTB-XL
        p_csv = REPO_ROOT / "data/processed_100hz/preprocessed_database.csv"
        self.df_p = pd.read_csv(p_csv, low_memory=False)
        self.sig_p = np.load(REPO_ROOT / "data/processed_100hz/signals.npy", mmap_mode="r")
        self.lbl_p = np.load(REPO_ROOT / "data/processed_100hz/labels.npy")
        self.df_p["dataset_name"] = "PTB-XL"
        self.df_p["std_rec_id"] = self.df_p["ecg_id"].astype(str)
        self.df_p["std_pat_id"] = self.df_p["patient_id"].astype(str)
        self.df_p["raw_ann"] = self.df_p["scp_codes"].astype(str)

        # 2. Chapman
        c_csv = REPO_ROOT / "data/processed_100hz/chapman/preprocessed_database.csv"
        self.df_c = pd.read_csv(c_csv, low_memory=False)
        self.sig_c = np.load(REPO_ROOT / "data/processed_100hz/chapman/signals.npy", mmap_mode="r")
        self.lbl_c = np.load(REPO_ROOT / "data/processed_100hz/chapman/labels.npy")
        self.df_c["dataset_name"] = "Chapman"
        self.df_c["std_rec_id"] = self.df_c["record_id"].astype(str)
        self.df_c["std_pat_id"] = self.df_c["patient_id"].astype(str)
        self.df_c["raw_ann"] = self.df_c["dx_snomed"].astype(str)

        # 3. MIMIC
        m_csv = REPO_ROOT / "data/processed_100hz/mimic/preprocessed_database.csv"
        self.df_m = pd.read_csv(m_csv, low_memory=False)
        self.sig_m = np.load(REPO_ROOT / "data/processed_100hz/mimic/signals.npy", mmap_mode="r")
        self.lbl_m = np.load(REPO_ROOT / "data/processed_100hz/mimic/labels.npy")
        self.df_m["dataset_name"] = "MIMIC"
        self.df_m["std_rec_id"] = self.df_m["record_id"].astype(str)
        self.df_m["std_pat_id"] = self.df_m["subject_id"].astype(str)
        self.df_m["raw_ann"] = self.df_m["disease"].astype(str)

        self.pooled_lbl = np.vstack([self.lbl_p, self.lbl_c, self.lbl_m])
        print(f"[+] Loaded {len(self.df_p):,} PTB-XL, {len(self.df_c):,} Chapman, {len(self.df_m):,} MIMIC (Total: {len(self.pooled_lbl):,} records).")

    def load_observations(self):
        if self.obs_test is not None:
            return

        obs_dir = REPO_ROOT / "data/observations/ptbxl"
        if not obs_dir.exists():
            return

        print("[*] Loading structured clinical observation dictionaries...")
        try:
            with open(obs_dir / "obs_1_lead.json") as f:
                self.obs_test = json.load(f)
            with open(obs_dir / "obs_train.json") as f:
                self.obs_train = json.load(f)
            with open(obs_dir / "obs_val.json") as f:
                self.obs_val = json.load(f)

            # Map test row index in Fold 10
            test_mask = (self.df_p["strat_fold"] == 10).to_numpy()
            self.ptb_test_indices = {orig_idx: test_row for test_row, orig_idx in enumerate(np.where(test_mask)[0])}
            print(f"[+] Observations loaded: {len(self.obs_train):,} train, {len(self.obs_val):,} val, {len(self.obs_test):,} test.")
        except Exception as e:
            print(f"[!] Warning: Observation loading error: {e}")

    def get_observation_for_ptb_row(self, orig_idx: int) -> Optional[Dict[str, Any]]:
        self.load_observations()
        if self.ptb_test_indices is None:
            return None

        fold = self.df_p.iloc[orig_idx]["strat_fold"]
        if fold == 10 and orig_idx in self.ptb_test_indices:
            test_pos = self.ptb_test_indices[orig_idx]
            if test_pos < len(self.obs_test):
                return self.obs_test[test_pos]
        return None


def format_observation_string(obs: Dict[str, Any]) -> str:
    if not obs:
        return "N/A"
    rhythm = obs.get("rhythm", "Unknown")
    hr = obs.get("heart_rate", "N/A")
    pr = obs.get("pr_interval_ms", "N/A")
    qrs = obs.get("qrs_duration_ms", "N/A")
    ste = obs.get("st_elevation", {}).get("present", False)
    ste_str = "ST-Elev: YES" if ste else "ST-Elev: No"
    return f"{rhythm} | HR: {hr} bpm | PR: {pr}ms | QRS: {qrs}ms | {ste_str}"


def print_record_table(df_slice, sig_slice, lbl_slice, dm: Optional[DatasetManager] = None, show_obs: bool = False):
    header = f"{'Row':<4} | {'Dataset':<7} | {'Record ID':<9} | {'Patient ID':<10} | {'QC':<4} | {'Voltage Bounds':<18} | {'Vector [N,A,R,L,B,T,1,M]':<26} | {'Assigned':<16} | {'Raw Annotation'}"
    if show_obs:
        header += " | Structured Observations"
    print(header)
    print("-" * (len(header) + 15))

    for i in range(len(df_slice)):
        row = df_slice.iloc[i]
        sig = sig_slice[i]
        lbl = lbl_slice[i]

        sig_min = float(np.nanmin(sig)) if not np.isnan(sig).all() else float("nan")
        sig_max = float(np.nanmax(sig)) if not np.isnan(sig).all() else float("nan")
        bounds_str = f"[{sig_min:+.2f}, {sig_max:+.2f}] mV"

        vec_str = str(lbl.tolist())
        active_dis = [DISEASE_SHORT[idx] for idx, val in enumerate(lbl) if val == 1]
        active_str = ", ".join(active_dis) if active_dis else "NONE"
        raw_ann_str = row["raw_ann"][:35] + ("..." if len(row["raw_ann"]) > 35 else "")

        qc_str = "PASS" if row.get("quality_passed", True) else "FAIL"

        line = f"{i:<4} | {row['dataset_name']:<7} | {row['std_rec_id']:<9} | {row['std_pat_id']:<10} | {qc_str:<4} | {bounds_str:<18} | {vec_str:<26} | {active_str:<16} | {raw_ann_str:<38}"

        if show_obs:
            obs_str = "N/A"
            if dm and row["dataset_name"] == "PTB-XL":
                # find original index in df_p
                orig_idx = row.name if hasattr(row, "name") and isinstance(row.name, int) else None
                if orig_idx is not None:
                    obs_dict = dm.get_observation_for_ptb_row(orig_idx)
                    if obs_dict:
                        obs_str = format_observation_string(obs_dict)
            line += f" | {obs_str}"

        print(line)


def print_prevalence_comparison(lbl_slice: np.ndarray, sample_title: str, pooled_lbl: np.ndarray):
    n_sample = len(lbl_slice)
    n_pooled = len(pooled_lbl)
    sample_sums = lbl_slice.sum(axis=0)
    pooled_sums = pooled_lbl.sum(axis=0)

    print(f"\n  Prevalence Comparison: {sample_title} (N = {n_sample:,}) vs Pooled Reference (N = {n_pooled:,}):")
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
    print(f"\n  Multi-Label Distribution: 0 labels: {z} ({z/n_sample*100:.1f}%), 1 label: {o} ({o/n_sample*100:.1f}%), >=2 labels: {m} ({m/n_sample*100:.1f}%)")


def run_five_samples(dm: DatasetManager, rng: np.random.Generator, show_obs: bool = False, range_size: int = 10):
    dm.load_all()
    if show_obs:
        dm.load_observations()

    print("\n" + "=" * 110)
    print(f"SAMPLE 1: RANDOM CONTIGUOUS RANGE FROM PTB-XL ({range_size} Records)")
    print("=" * 110)
    start_1 = int(rng.integers(0, len(dm.df_p) - range_size))
    slice_df_1 = dm.df_p.iloc[start_1 : start_1 + range_size]
    slice_sig_1 = np.array(dm.sig_p[start_1 : start_1 + range_size])
    slice_lbl_1 = dm.lbl_p[start_1 : start_1 + range_size]
    print(f"Row Range in PTB-XL: [{start_1} to {start_1 + range_size - 1}]")
    print_record_table(slice_df_1, slice_sig_1, slice_lbl_1, dm=dm, show_obs=show_obs)
    print_prevalence_comparison(slice_lbl_1, f"PTB-XL Range [{start_1}:{start_1+range_size}]", dm.pooled_lbl)

    print("\n" + "=" * 110)
    print(f"SAMPLE 2: RANDOM CONTIGUOUS RANGE FROM CHAPMAN-SHAOXING ({range_size} Records)")
    print("=" * 110)
    start_2 = int(rng.integers(0, len(dm.df_c) - range_size))
    slice_df_2 = dm.df_c.iloc[start_2 : start_2 + range_size]
    slice_sig_2 = np.array(dm.sig_c[start_2 : start_2 + range_size])
    slice_lbl_2 = dm.lbl_c[start_2 : start_2 + range_size]
    print(f"Row Range in Chapman: [{start_2} to {start_2 + range_size - 1}]")
    print_record_table(slice_df_2, slice_sig_2, slice_lbl_2, dm=dm, show_obs=show_obs)
    print_prevalence_comparison(slice_lbl_2, f"Chapman Range [{start_2}:{start_2+range_size}]", dm.pooled_lbl)

    print("\n" + "=" * 110)
    print(f"SAMPLE 3: RANDOM CONTIGUOUS RANGE FROM MIMIC-IV-ECG ({range_size} Records)")
    print("=" * 110)
    start_3 = int(rng.integers(0, len(dm.df_m) - range_size))
    slice_df_3 = dm.df_m.iloc[start_3 : start_3 + range_size]
    slice_sig_3 = np.array(dm.sig_m[start_3 : start_3 + range_size])
    slice_lbl_3 = dm.lbl_m[start_3 : start_3 + range_size]
    print(f"Row Range in MIMIC: [{start_3} to {start_3 + range_size - 1}]")
    print_record_table(slice_df_3, slice_sig_3, slice_lbl_3, dm=dm, show_obs=show_obs)
    print_prevalence_comparison(slice_lbl_3, f"MIMIC Range [{start_3}:{start_3+range_size}]", dm.pooled_lbl)

    print("\n" + "=" * 110)
    print("SAMPLE 4: RANDOM MULTI-RECORD PATIENT CLUSTER IN PTB-XL (Testing Patient Grouping)")
    print("=" * 110)
    pat_counts = dm.df_p["patient_id"].value_counts()
    multi_pats = pat_counts[pat_counts >= 3].index.to_numpy()
    chosen_pat = rng.choice(multi_pats)
    pat_indices = np.where(dm.df_p["patient_id"] == chosen_pat)[0]
    slice_df_4 = dm.df_p.iloc[pat_indices]
    slice_sig_4 = np.array(dm.sig_p[pat_indices])
    slice_lbl_4 = dm.lbl_p[pat_indices]
    print(f"Patient ID: {chosen_pat} (Has {len(pat_indices)} recordings across rows {pat_indices.tolist()})")
    print_record_table(slice_df_4, slice_sig_4, slice_lbl_4, dm=dm, show_obs=show_obs)
    print_prevalence_comparison(slice_lbl_4, f"Patient {chosen_pat} Cluster", dm.pooled_lbl)

    print("\n" + "=" * 110)
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

    print("Displaying first 15 of 50 randomly drawn records from across all 3 datasets:")
    print_record_table(slice_df_5.iloc[:15], slice_sig_5[:15], slice_lbl_5[:15], dm=dm, show_obs=show_obs)
    print(f"... [{len(slice_df_5) - 15} additional records in sample omitted from table for readability] ...")
    print_prevalence_comparison(slice_lbl_5, "50-Record Pooled Sample", dm.pooled_lbl)


def inspect_single_record(dm: DatasetManager, rec_id: str):
    dm.load_all()
    dm.load_observations()

    # Search in PTB-XL
    match_p = dm.df_p[dm.df_p["std_rec_id"] == str(rec_id)]
    if len(match_p) > 0:
        row_idx = match_p.index[0]
        row = match_p.iloc[0]
        sig = np.array(dm.sig_p[row_idx])
        lbl = dm.lbl_p[row_idx]
        obs = dm.get_observation_for_ptb_row(row_idx)
        print_detailed_record_card("PTB-XL", row, sig, lbl, obs)
        return

    # Search in Chapman
    match_c = dm.df_c[dm.df_c["std_rec_id"] == str(rec_id)]
    if len(match_c) > 0:
        row_idx = match_c.index[0]
        row = match_c.iloc[0]
        sig = np.array(dm.sig_c[row_idx])
        lbl = dm.lbl_c[row_idx]
        print_detailed_record_card("Chapman", row, sig, lbl, None)
        return

    # Search in MIMIC
    match_m = dm.df_m[dm.df_m["std_rec_id"] == str(rec_id)]
    if len(match_m) > 0:
        row_idx = match_m.index[0]
        row = match_m.iloc[0]
        sig = np.array(dm.sig_m[row_idx])
        lbl = dm.lbl_m[row_idx]
        print_detailed_record_card("MIMIC", row, sig, lbl, None)
        return

    print(f"[!] Record ID '{rec_id}' not found in any dataset.")


def print_detailed_record_card(dataset: str, row: pd.Series, sig: np.ndarray, lbl: np.ndarray, obs: Optional[Dict[str, Any]]):
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


def interactive_menu(dm: DatasetManager):
    dm.load_all()
    dm.load_observations()

    while True:
        print("\n" + "=" * 60)
        print("  INTERACTIVE AUDIT MENU")
        print("=" * 60)
        print("1. Run 5 Fresh Random Range Samples")
        print("2. Inspect a Specific Record by ID (e.g. 9, JS00001, 48418505)")
        print("3. Inspect a Multi-Record Patient Cluster in PTB-XL")
        print("4. Exit")
        choice = input("\nEnter choice (1-4): ").strip()

        if choice == "1":
            rng = np.random.default_rng()
            run_five_samples(dm, rng, show_obs=True, range_size=10)
        elif choice == "2":
            rid = input("Enter Record ID: ").strip()
            if rid:
                inspect_single_record(dm, rid)
        elif choice == "3":
            rng = np.random.default_rng()
            pat_counts = dm.df_p["patient_id"].value_counts()
            multi_pats = pat_counts[pat_counts >= 3].index.to_numpy()
            chosen_pat = rng.choice(multi_pats)
            pat_indices = np.where(dm.df_p["patient_id"] == chosen_pat)[0]
            slice_df = dm.df_p.iloc[pat_indices]
            slice_sig = np.array(dm.sig_p[pat_indices])
            slice_lbl = dm.lbl_p[pat_indices]
            print(f"\nPatient ID: {chosen_pat} (Has {len(pat_indices)} recordings across rows {pat_indices.tolist()}):")
            print_record_table(slice_df, slice_sig, slice_lbl, dm=dm, show_obs=True)
            print_prevalence_comparison(slice_lbl, f"Patient {chosen_pat}", dm.pooled_lbl)
        elif choice == "4":
            print("Exiting.")
            break
        else:
            print("[!] Invalid choice.")


def main():
    parser = argparse.ArgumentParser(description="Audit and Random Range Sampler for ECG Datasets.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--range-size", type=int, default=10, help="Contiguous sample window size.")
    parser.add_argument("--show-obs", action="store_true", help="Display structured clinical observations.")
    parser.add_argument("--record-id", type=str, default=None, help="Inspect a specific record ID.")
    parser.add_argument("-i", "--interactive", action="store_true", help="Launch interactive menu.")
    args = parser.parse_args()

    dm = DatasetManager()

    if args.record_id:
        inspect_single_record(dm, args.record_id)
        return

    if args.interactive:
        interactive_menu(dm)
        return

    # Default execution: Run 5 random samples
    rng = np.random.default_rng(seed=args.seed)
    run_five_samples(dm, rng, show_obs=args.show_obs, range_size=args.range_size)


if __name__ == "__main__":
    main()

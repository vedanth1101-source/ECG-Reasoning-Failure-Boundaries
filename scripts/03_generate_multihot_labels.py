"""
Multi-Cohort Ground Truth Harmonization & Multi-Hot Matrix Generator (Step 2)
=============================================================================
Maps clinical annotations across all three preprocessed 100 Hz cohorts:
  1. PTB-XL (21,799 records)
  2. Chapman-Shaoxing (10,158 records)
  3. MIMIC-IV-ECG (5,000 records)

Produces multi-hot binary label matrices (N, 8) for the frozen 8-disease benchmark:
  0: Normal ECG
  1: Atrial Fibrillation
  2: Right Bundle Branch Block
  3: Left Bundle Branch Block
  4: Sinus Bradycardia
  5: Sinus Tachycardia
  6: 1st Degree AV Block
  7: Myocardial Infarction

Outputs:
  - data/processed_100hz/labels.npy (PTB-XL)
  - data/processed_100hz/chapman/labels.npy (Chapman)
  - data/processed_100hz/mimic/labels.npy (MIMIC)
  - Enriched preprocessed_database.csv in each directory with binary label columns
  - data/processed_100hz/labels_schema.json
  - reports/multi_cohort_label_harmonization.md
"""

import os
import sys
import json
import ast
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any
import numpy as np
import pandas as pd

# The Frozen 8-Disease Benchmark Schema
BENCHMARK_DISEASES = [
    "Normal ECG",
    "Atrial Fibrillation",
    "Right Bundle Branch Block",
    "Left Bundle Branch Block",
    "Sinus Bradycardia",
    "Sinus Tachycardia",
    "1st Degree AV Block",
    "Myocardial Infarction"
]

DISEASE_TO_IDX = {dis: idx for idx, dis in enumerate(BENCHMARK_DISEASES)}

# Column names for CSV metadata
LABEL_COLS = [
    "label_normal_ecg",
    "label_afib",
    "label_rbbb",
    "label_lbbb",
    "label_sbrad",
    "label_stach",
    "label_1avb",
    "label_mi"
]

# Clinical Metadata / Hallmarks for Knowledge Base
DISEASE_METADATA = {
    "Normal ECG": {
        "acronym": "NORM",
        "category": "Baseline Control",
        "lead_hallmarks": "Global across 12 leads",
        "diagnostic_criteria": "Normal sinus rhythm, normal P-QRS-T morphology and intervals"
    },
    "Atrial Fibrillation": {
        "acronym": "AFIB",
        "category": "Supraventricular Arrhythmia",
        "lead_hallmarks": "Rhythm leads (Lead II, V1)",
        "diagnostic_criteria": "Absent P waves, fibrillatory f-waves, irregularly irregular RR intervals"
    },
    "Right Bundle Branch Block": {
        "acronym": "RBBB",
        "category": "Conduction Delay (Right)",
        "lead_hallmarks": "V1, V2 (RSR') and I, aVL, V5, V6 (slurred S)",
        "diagnostic_criteria": "QRS >= 120 ms, secondary R wave in V1, wide slurred S wave in lateral leads"
    },
    "Left Bundle Branch Block": {
        "acronym": "LBBB",
        "category": "Conduction Delay (Left)",
        "lead_hallmarks": "I, aVL, V5, V6 (broad R) and V1, V2 (deep QS/rS)",
        "diagnostic_criteria": "QRS >= 120 ms, broad monophasic R in lateral leads, absent septal Q"
    },
    "Sinus Bradycardia": {
        "acronym": "SBRAD",
        "category": "Sinus Node Dysfunction (Slow)",
        "lead_hallmarks": "Global rate observable on any active lead",
        "diagnostic_criteria": "Sinus rhythm with resting ventricular rate < 60 bpm"
    },
    "Sinus Tachycardia": {
        "acronym": "STACH",
        "category": "Sinus Node Dysfunction (Fast)",
        "lead_hallmarks": "Global rate observable on any active lead",
        "diagnostic_criteria": "Sinus rhythm with resting ventricular rate > 100 bpm"
    },
    "1st Degree AV Block": {
        "acronym": "1AVB",
        "category": "Atrioventricular Conduction Delay",
        "lead_hallmarks": "Lead II, aVF, V1",
        "diagnostic_criteria": "Fixed PR interval prolongation > 200 ms with 1:1 AV conduction"
    },
    "Myocardial Infarction": {
        "acronym": "MI",
        "category": "Myocardial Ischemia / Infarction",
        "lead_hallmarks": "Regional: Inferior (II, III, aVF) or Anterior/Septal (V1-V4)",
        "diagnostic_criteria": "Pathological Q waves (> 40 ms) and/or ST elevation/depression"
    }
}


# -------------------------------------------------------------------------
# PTB-XL Multi-Hot Mapper
# -------------------------------------------------------------------------

def map_ptbxl_record(scp_str: str) -> np.ndarray:
    """Maps PTB-XL scp_codes dictionary to an 8-element binary multi-hot vector."""
    vec = np.zeros(8, dtype=np.uint8)
    try:
        d = ast.literal_eval(scp_str) if isinstance(scp_str, str) else {}
    except Exception:
        d = {}

    has_afib = "AFIB" in d
    has_rbbb = ("CRBBB" in d) or ("IRBBB" in d and d.get("IRBBB", 0) >= 50)
    has_lbbb = ("CLBBB" in d) or ("ILBBB" in d and d.get("ILBBB", 0) >= 50)
    has_sbrad = "SBRAD" in d
    has_stach = "STACH" in d
    has_1avb = "1AVB" in d

    # Any MI subclass
    mi_codes = ["IMI", "ASMI", "AMI", "ALMI", "ILMI", "LMI", "IPMI", "PMI"]
    has_mi = any(c in d for c in mi_codes)

    # Any pathological finding
    has_pathology = has_afib or has_rbbb or has_lbbb or has_1avb or has_mi

    # Normal ECG: explicit NORM statement or (SR without pathological disease)
    has_norm = ("NORM" in d and d.get("NORM", 0) >= 50) or ("SR" in d and not has_pathology)

    if has_norm and not has_pathology:
        vec[0] = 1
    if has_afib:
        vec[1] = 1
    if has_rbbb:
        vec[2] = 1
    if has_lbbb:
        vec[3] = 1
    if has_sbrad:
        vec[4] = 1
    if has_stach:
        vec[5] = 1
    if has_1avb:
        vec[6] = 1
    if has_mi:
        vec[7] = 1

    return vec


# -------------------------------------------------------------------------
# Chapman-Shaoxing Multi-Hot Mapper
# -------------------------------------------------------------------------

# SNOMED-CT code sets
SNOMED_NORM = {"426783006"}       # Sinus rhythm
SNOMED_AFIB = {"164889003"}       # Atrial fibrillation
SNOMED_RBBB = {"59118001"}        # Right bundle branch block
SNOMED_LBBB = {"164909002"}       # Left bundle branch block
SNOMED_SBRAD = {"426177001"}      # Sinus bradycardia
SNOMED_STACH = {"427084000"}      # Sinus tachycardia
SNOMED_1AVB = {"270492004", "164947007"}  # 1AVB + PR extension
SNOMED_MI = {"164865005"}         # Myocardial infarction

def map_chapman_record(dx_str: str) -> np.ndarray:
    """Maps Chapman-Shaoxing dx_snomed comma-separated string to an 8-element binary vector."""
    vec = np.zeros(8, dtype=np.uint8)
    codes = set(str(dx_str).replace(" ", "").split(",")) if pd.notna(dx_str) else set()

    has_afib = bool(codes.intersection(SNOMED_AFIB))
    has_rbbb = bool(codes.intersection(SNOMED_RBBB))
    has_lbbb = bool(codes.intersection(SNOMED_LBBB))
    has_sbrad = bool(codes.intersection(SNOMED_SBRAD))
    has_stach = bool(codes.intersection(SNOMED_STACH))
    has_1avb = bool(codes.intersection(SNOMED_1AVB))
    has_mi = bool(codes.intersection(SNOMED_MI))

    has_pathology = has_afib or has_rbbb or has_lbbb or has_1avb or has_mi
    has_norm = bool(codes.intersection(SNOMED_NORM)) and not has_pathology

    if has_norm:
        vec[0] = 1
    if has_afib:
        vec[1] = 1
    if has_rbbb:
        vec[2] = 1
    if has_lbbb:
        vec[3] = 1
    if has_sbrad:
        vec[4] = 1
    if has_stach:
        vec[5] = 1
    if has_1avb:
        vec[6] = 1
    if has_mi:
        vec[7] = 1

    return vec


# -------------------------------------------------------------------------
# MIMIC-IV-ECG Multi-Hot Mapper
# -------------------------------------------------------------------------

MIMIC_DISEASE_MAP = {
    "Normal ECG": 0,
    "Atrial Fibrillation": 1,
    "Right Bundle Branch Block": 2,
    "Left Bundle Branch Block": 3,
    "Sinus Bradycardia": 4,
    "Sinus Tachycardia": 5,
    "1st Degree AV Block": 6,
    "Myocardial Infarction": 7
}

def map_mimic_record(disease_str: str) -> np.ndarray:
    """Maps MIMIC-IV-ECG stratified disease string to an 8-element binary vector."""
    vec = np.zeros(8, dtype=np.uint8)
    if pd.notna(disease_str) and str(disease_str) in MIMIC_DISEASE_MAP:
        idx = MIMIC_DISEASE_MAP[str(disease_str)]
        vec[idx] = 1
    return vec


# -------------------------------------------------------------------------
# Main Execution Pipeline
# -------------------------------------------------------------------------

def process_all_cohorts(base_dir: Path, reports_dir: Path):
    print("="*75)
    print("STEP 2: MULTI-COHORT GROUND TRUTH HARMONIZATION (8-DISEASE SCHEMA)")
    print("="*75)

    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_stats: Dict[str, Any] = {}

    # ---------------------------------------------------------------------
    # 1. Process PTB-XL
    # ---------------------------------------------------------------------
    ptb_csv = base_dir / "preprocessed_database.csv"
    if not ptb_csv.exists():
        raise FileNotFoundError(f"PTB-XL database not found at {ptb_csv}")

    print(f"\n[1/3] Processing PTB-XL database from {ptb_csv}...")
    ptb_df = pd.read_csv(ptb_csv)
    N_ptb = len(ptb_df)
    ptb_labels = np.zeros((N_ptb, 8), dtype=np.uint8)

    for i, row in ptb_df.iterrows():
        ptb_labels[i] = map_ptbxl_record(row.get("scp_codes", ""))

    ptb_lbl_file = base_dir / "labels.npy"
    print(f"Saving PTB-XL labels to {ptb_lbl_file} (shape: {ptb_labels.shape})...")
    np.save(ptb_lbl_file, ptb_labels)

    # Add label columns to dataframe
    for col_idx, col_name in enumerate(LABEL_COLS):
        ptb_df[col_name] = ptb_labels[:, col_idx]
    ptb_df["num_positive_labels"] = ptb_labels.sum(axis=1)
    ptb_df.to_csv(ptb_csv, index=False)
    print(f"Updated {ptb_csv} with 8 binary label columns.")

    summary_stats["PTB-XL"] = {
        "total_records": N_ptb,
        "patients": int(ptb_df["patient_id"].nunique()),
        "labels_file": str(ptb_lbl_file),
        "disease_counts": {dis: int(ptb_labels[:, idx].sum()) for idx, dis in enumerate(BENCHMARK_DISEASES)},
        "multi_label_records": int((ptb_labels.sum(axis=1) > 1).sum()),
        "unassigned_records": int((ptb_labels.sum(axis=1) == 0).sum())
    }

    # ---------------------------------------------------------------------
    # 2. Process Chapman-Shaoxing
    # ---------------------------------------------------------------------
    chap_dir = base_dir / "chapman"
    chap_csv = chap_dir / "preprocessed_database.csv"
    if not chap_csv.exists():
        raise FileNotFoundError(f"Chapman database not found at {chap_csv}")

    print(f"\n[2/3] Processing Chapman-Shaoxing database from {chap_csv}...")
    chap_df = pd.read_csv(chap_csv)
    N_chap = len(chap_df)
    chap_labels = np.zeros((N_chap, 8), dtype=np.uint8)

    for i, row in chap_df.iterrows():
        chap_labels[i] = map_chapman_record(row.get("dx_snomed", ""))

    chap_lbl_file = chap_dir / "labels.npy"
    print(f"Saving Chapman labels to {chap_lbl_file} (shape: {chap_labels.shape})...")
    np.save(chap_lbl_file, chap_labels)

    for col_idx, col_name in enumerate(LABEL_COLS):
        chap_df[col_name] = chap_labels[:, col_idx]
    chap_df["num_positive_labels"] = chap_labels.sum(axis=1)
    chap_df.to_csv(chap_csv, index=False)
    print(f"Updated {chap_csv} with 8 binary label columns.")

    summary_stats["Chapman-Shaoxing"] = {
        "total_records": N_chap,
        "patients": int(chap_df["patient_id"].nunique()),
        "labels_file": str(chap_lbl_file),
        "disease_counts": {dis: int(chap_labels[:, idx].sum()) for idx, dis in enumerate(BENCHMARK_DISEASES)},
        "multi_label_records": int((chap_labels.sum(axis=1) > 1).sum()),
        "unassigned_records": int((chap_labels.sum(axis=1) == 0).sum())
    }

    # ---------------------------------------------------------------------
    # 3. Process MIMIC-IV-ECG
    # ---------------------------------------------------------------------
    mim_dir = base_dir / "mimic"
    mim_csv = mim_dir / "preprocessed_database.csv"
    if not mim_csv.exists():
        raise FileNotFoundError(f"MIMIC database not found at {mim_csv}")

    print(f"\n[3/3] Processing MIMIC-IV-ECG database from {mim_csv}...")
    mim_df = pd.read_csv(mim_csv)
    N_mim = len(mim_df)
    mim_labels = np.zeros((N_mim, 8), dtype=np.uint8)

    for i, row in mim_df.iterrows():
        mim_labels[i] = map_mimic_record(row.get("disease", ""))

    mim_lbl_file = mim_dir / "labels.npy"
    print(f"Saving MIMIC labels to {mim_lbl_file} (shape: {mim_labels.shape})...")
    np.save(mim_lbl_file, mim_labels)

    for col_idx, col_name in enumerate(LABEL_COLS):
        mim_df[col_name] = mim_labels[:, col_idx]
    mim_df["num_positive_labels"] = mim_labels.sum(axis=1)
    mim_df.to_csv(mim_csv, index=False)
    print(f"Updated {mim_csv} with 8 binary label columns.")

    summary_stats["MIMIC-IV-ECG"] = {
        "total_records": N_mim,
        "patients": int(mim_df["patient_id"].nunique()),
        "labels_file": str(mim_lbl_file),
        "disease_counts": {dis: int(mim_labels[:, idx].sum()) for idx, dis in enumerate(BENCHMARK_DISEASES)},
        "multi_label_records": int((mim_labels.sum(axis=1) > 1).sum()),
        "unassigned_records": int((mim_labels.sum(axis=1) == 0).sum())
    }

    # ---------------------------------------------------------------------
    # 4. Save Schema JSON
    # ---------------------------------------------------------------------
    schema_json_path = base_dir / "labels_schema.json"
    schema_dict = {
        "benchmark_diseases": BENCHMARK_DISEASES,
        "label_columns": LABEL_COLS,
        "disease_metadata": DISEASE_METADATA,
        "cohort_summaries": summary_stats
    }
    with open(schema_json_path, "w") as f:
        json.dump(schema_dict, f, indent=2)
    print(f"\nSaved schema metadata to {schema_json_path}.")

    # ---------------------------------------------------------------------
    # 5. Generate Markdown Report
    # ---------------------------------------------------------------------
    report_path = reports_dir / "multi_cohort_label_harmonization.md"
    generate_markdown_report(summary_stats, report_path)
    print(f"Generated comprehensive report at {report_path}.")

    print("\n" + "="*75)
    print("STEP 2 COMPLETE: ALL MULTI-HOT MATRICES SUCCESSFULLY GENERATED")
    print("="*75)


def generate_markdown_report(stats: Dict[str, Any], out_path: Path):
    """Generates an academic-grade markdown report of the harmonized 8-disease benchmark."""
    md = """# Multi-Cohort Ground Truth Label Harmonization Report (8-Disease Benchmark)

> **Step 2 Completed:** Aligned multi-hot ground truth label matrices $(N, 8)$ across the primary training cohort (**PTB-XL**) and external validation cohorts (**Chapman-Shaoxing** and **MIMIC-IV-ECG**).

---

## 1. Frozen 8-Disease Benchmark Specification

| Index | Disease Name | Category | Primary Lead Hallmarks | Clinical Diagnostic Hallmark |
| :---: | :--- | :--- | :--- | :--- |
"""
    for idx, dis in enumerate(BENCHMARK_DISEASES):
        m = DISEASE_METADATA[dis]
        md += f"| **{idx}** | **{dis}** | {m['category']} | `{m['lead_hallmarks']}` | {m['diagnostic_criteria']} |\n"

    md += """
---

## 2. Cross-Cohort Disease Prevalence Matrix

| Disease Class | PTB-XL (21,799) | Chapman (10,158) | MIMIC (5,000) | Total Positive Records | Cross-Cohort Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for dis in BENCHMARK_DISEASES:
        p_cnt = stats["PTB-XL"]["disease_counts"][dis]
        c_cnt = stats["Chapman-Shaoxing"]["disease_counts"][dis]
        m_cnt = stats["MIMIC-IV-ECG"]["disease_counts"][dis]
        total = p_cnt + c_cnt + m_cnt
        present_in = sum([p_cnt > 0, c_cnt > 0, m_cnt > 0])
        status = f"**{present_in}/3 Cohorts**"
        md += f"| **{dis}** | {p_cnt:,} | {c_cnt:,} | {m_cnt:,} | **{total:,}** | {status} |\n"

    md += f"""
---

## 3. Cohort Metadata & Multi-Label Breakdown

| Metric | PTB-XL | Chapman-Shaoxing | MIMIC-IV-ECG | Total Ecosystem |
| :--- | :---: | :---: | :---: | :---: |
| **Total Processed Records** | {stats['PTB-XL']['total_records']:,} | {stats['Chapman-Shaoxing']['total_records']:,} | {stats['MIMIC-IV-ECG']['total_records']:,} | **{stats['PTB-XL']['total_records'] + stats['Chapman-Shaoxing']['total_records'] + stats['MIMIC-IV-ECG']['total_records']:,}** |
| **Unique Patients** | {stats['PTB-XL']['patients']:,} | {stats['Chapman-Shaoxing']['patients']:,} | {stats['MIMIC-IV-ECG']['patients']:,} | **{stats['PTB-XL']['patients'] + stats['Chapman-Shaoxing']['patients'] + stats['MIMIC-IV-ECG']['patients']:,}** |
| **Multi-Label Records (>1 disease)** | {stats['PTB-XL']['multi_label_records']:,} | {stats['Chapman-Shaoxing']['multi_label_records']:,} | {stats['MIMIC-IV-ECG']['multi_label_records']:,} | **{stats['PTB-XL']['multi_label_records'] + stats['Chapman-Shaoxing']['multi_label_records'] + stats['MIMIC-IV-ECG']['multi_label_records']:,}** |
| **Unassigned / Other Records** | {stats['PTB-XL']['unassigned_records']:,} | {stats['Chapman-Shaoxing']['unassigned_records']:,} | {stats['MIMIC-IV-ECG']['unassigned_records']:,} | **{stats['PTB-XL']['unassigned_records'] + stats['Chapman-Shaoxing']['unassigned_records'] + stats['MIMIC-IV-ECG']['unassigned_records']:,}** |

---

## 4. Physical Storage Paths

Each cohort directory in [`data/processed_100hz/`](file:///e:/ECG/data/processed_100hz/) now contains synchronized pairs of signals and labels:
- **PTB-XL:** `data/processed_100hz/signals.npy` `(21799, 12, 1000)` + `labels.npy` `(21799, 8)`
- **Chapman:** `data/processed_100hz/chapman/signals.npy` `(10158, 12, 1000)` + `labels.npy` `(10158, 8)`
- **MIMIC:** `data/processed_100hz/mimic/signals.npy` `(5000, 12, 1000)` + `labels.npy` `(5000, 8)`
"""
    out_path.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    base_dir = Path("data/processed_100hz")
    reports_dir = Path("reports")
    process_all_cohorts(base_dir, reports_dir)

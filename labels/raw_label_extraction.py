"""
Raw Label Extraction Module
===========================
Step 2 of Dataset Preparation:
Extracts original diagnostic labels exactly as provided by each dataset.
Maintains full traceability:
- dataset
- record_id
- patient_id
- raw_label
- raw_label_code

Does NOT normalize or overwrite labels at this stage.
"""

import ast
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np


def extract_ptbxl_raw_labels(
    database_csv_path: Union[str, Path],
    scp_statements_csv_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Extracts raw SCP statements from PTB-XL ptbxl_database.csv.
    
    Each record contains a dictionary string in `scp_codes`, e.g.
    "{'NORM': 100.0, 'LVOLT': 0.0, 'SR': 0.0}".
    
    Returns DataFrame with columns:
    [dataset, record_id, patient_id, raw_label, raw_label_code, likelihood, original_report]
    """
    db_path = Path(database_csv_path)
    if not db_path.exists():
        raise FileNotFoundError(f"PTB-XL database file not found at: {db_path}")

    print(f"[PTB-XL] Reading {db_path}...")
    df = pd.read_csv(db_path)
    
    # Load scp descriptions if available
    scp_desc_map = {}
    if scp_statements_csv_path and Path(scp_statements_csv_path).exists():
        scp_df = pd.read_csv(scp_statements_csv_path, index_col=0)
        scp_desc_map = scp_df["description"].to_dict()

    rows = []
    for idx, row in df.iterrows():
        rec_id = str(row["ecg_id"])
        # Format patient_id cleanly without trailing floats
        pid_val = row["patient_id"]
        pat_id = str(int(pid_val)) if pd.notna(pid_val) else "UNKNOWN"
        report_text = str(row.get("report", "")) if pd.notna(row.get("report")) else ""
        
        raw_scp_str = str(row["scp_codes"])
        try:
            scp_dict = ast.literal_eval(raw_scp_str)
        except Exception:
            scp_dict = {}

        if not scp_dict:
            rows.append({
                "dataset": "PTB-XL",
                "record_id": rec_id,
                "patient_id": pat_id,
                "raw_label": "Unlabeled / Empty",
                "raw_label_code": "NONE",
                "likelihood": 0.0,
                "original_report": report_text
            })
        else:
            for code, likelihood in scp_dict.items():
                label_desc = scp_desc_map.get(code, code)
                rows.append({
                    "dataset": "PTB-XL",
                    "record_id": rec_id,
                    "patient_id": pat_id,
                    "raw_label": label_desc,
                    "raw_label_code": str(code),
                    "likelihood": float(likelihood),
                    "original_report": report_text
                })

    raw_labels_df = pd.DataFrame(rows)
    print(f"[PTB-XL] Extracted {len(raw_labels_df)} raw label instances across {df['ecg_id'].nunique()} records and {df['patient_id'].nunique()} patients.")
    return raw_labels_df


def parse_chapman_header_labels(header_text: str) -> Dict[str, Union[List[str], str]]:
    """
    Parses # Dx:, # Age:, # Sex: comments from a WFDB header string.
    """
    res = {"dx_codes": [], "age": "UNKNOWN", "sex": "UNKNOWN"}
    for line in header_text.splitlines():
        line = line.strip()
        if line.startswith("# Dx:"):
            dx_str = line.replace("# Dx:", "").strip()
            if dx_str:
                res["dx_codes"] = [c.strip() for c in dx_str.split(",") if c.strip()]
        elif line.startswith("# Age:"):
            res["age"] = line.replace("# Age:", "").strip()
        elif line.startswith("# Sex:"):
            res["sex"] = line.replace("# Sex:", "").strip()
    return res


def extract_chapman_raw_labels(
    chapman_dir: Union[str, Path],
    snomed_mapping_csv_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Extracts raw SNOMED-CT labels from Chapman-Shaoxing .hea headers.
    
    Returns DataFrame with columns:
    [dataset, record_id, patient_id, raw_label, raw_label_code, age, sex]
    """
    c_dir = Path(chapman_dir)
    if not c_dir.exists():
        raise FileNotFoundError(f"Chapman directory not found at: {c_dir}")

    hea_files = sorted(list(c_dir.glob("*.hea")))
    print(f"[Chapman-Shaoxing] Inspecting {len(hea_files)} .hea header files in {c_dir}...")

    # Load SNOMED code to name mapping if available
    snomed_name_map = {}
    if snomed_mapping_csv_path and Path(snomed_mapping_csv_path).exists():
        try:
            sn_df = pd.read_csv(snomed_mapping_csv_path, encoding="utf-8-sig")
            # Usually columns: Acronym Name, Full Name, Snomed_CT
            for _, r in sn_df.iterrows():
                code_str = str(r["Snomed_CT"]).strip()
                full_name = str(r["Full Name"]).strip()
                snomed_name_map[code_str] = full_name
        except Exception as e:
            print(f"Warning: Could not read SNOMED mapping file: {e}")

    rows = []
    for hf in hea_files:
        rec_id = hf.stem
        # In Chapman-Shaoxing, each recording corresponds to a distinct patient encounter
        # Record ID itself acts as the encounter/patient ID
        pat_id = rec_id
        
        try:
            content = hf.read_text(encoding="utf-8", errors="ignore")
            parsed = parse_chapman_header_labels(content)
            dx_codes = parsed["dx_codes"]
            age = parsed["age"]
            sex = parsed["sex"]
            
            if not dx_codes:
                rows.append({
                    "dataset": "Chapman-Shaoxing",
                    "record_id": rec_id,
                    "patient_id": pat_id,
                    "raw_label": "Unlabeled",
                    "raw_label_code": "NONE",
                    "age": age,
                    "sex": sex
                })
            else:
                for code in dx_codes:
                    raw_name = snomed_name_map.get(str(code), f"SNOMED_{code}")
                    rows.append({
                        "dataset": "Chapman-Shaoxing",
                        "record_id": rec_id,
                        "patient_id": pat_id,
                        "raw_label": raw_name,
                        "raw_label_code": str(code),
                        "age": age,
                        "sex": sex
                    })
        except Exception as e:
            print(f"Error parsing header {hf}: {e}")

    raw_labels_df = pd.DataFrame(rows)
    print(f"[Chapman-Shaoxing] Extracted {len(raw_labels_df)} raw label instances across {len(hea_files)} records.")
    return raw_labels_df


def extract_mimic_raw_labels(
    machine_measurements_csv_path: Union[str, Path]
) -> pd.DataFrame:
    """
    Extracts raw report phrases from MIMIC-IV-ECG machine_measurements.csv.
    
    Columns: subject_id, study_id, ecg_time, report_0 ... report_17.
    
    Returns DataFrame with columns:
    [dataset, record_id, patient_id, raw_label, raw_label_code, report_index, ecg_time]
    """
    mm_path = Path(machine_measurements_csv_path)
    if not mm_path.exists():
        raise FileNotFoundError(f"MIMIC machine measurements file not found at: {mm_path}")

    print(f"[MIMIC-IV-ECG] Loading report columns from {mm_path}...")
    report_cols = [f"report_{i}" for i in range(18)]
    use_cols = ["subject_id", "study_id", "ecg_time"] + report_cols
    
    # Read with chunking for memory efficiency
    chunks = []
    chunksize = 100000
    for chunk in pd.read_csv(mm_path, usecols=use_cols, chunksize=chunksize, low_memory=False):
        # Melt reports
        melted = chunk.melt(
            id_vars=["subject_id", "study_id", "ecg_time"],
            value_vars=report_cols,
            var_name="report_index",
            value_name="raw_label"
        )
        # Drop NaNs and empty whitespace strings
        melted = melted.dropna(subset=["raw_label"])
        melted["raw_label"] = melted["raw_label"].astype(str).str.strip()
        melted = melted[melted["raw_label"] != ""]
        melted = melted[melted["raw_label"].str.lower() != "nan"]
        
        melted["dataset"] = "MIMIC-IV-ECG"
        melted["record_id"] = melted["study_id"].astype(str)
        melted["patient_id"] = melted["subject_id"].astype(str)
        melted["raw_label_code"] = melted["report_index"]
        
        chunks.append(melted[["dataset", "record_id", "patient_id", "raw_label", "raw_label_code", "report_index", "ecg_time"]])

    raw_labels_df = pd.concat(chunks, ignore_index=True)
    print(f"[MIMIC-IV-ECG] Extracted {len(raw_labels_df)} raw label instances across {raw_labels_df['record_id'].nunique()} records and {raw_labels_df['patient_id'].nunique()} patients.")
    return raw_labels_df

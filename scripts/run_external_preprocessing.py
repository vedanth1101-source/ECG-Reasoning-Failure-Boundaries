"""
External Validation Cohort Batch Preprocessing Script (Phase 1)
===============================================================
Executes the standardized Phase 1 Preprocessing Pipeline (P_1 through P_5)
across external validation cohorts:
  1. Chapman-Shaoxing (10,158 paired records, MATLAB v4 format 16, 500 Hz)
  2. MIMIC-IV-ECG (5,000 sampled records, WFDB format 16, 500 Hz, 200 ADC/mV)

Standardizes all signals to:
  - Shape: (12, 1000) [100 Hz, 10.0 seconds]
  - Channel Sequence: Canonical Cabrera order [I, II, III, aVR, aVL, aVF, V1-V6]
  - Calibration: Physical millivolts (mV)
  - Filtering: Zero-phase bandpass (0.5 Hz highpass, 50/60 Hz notch, 45 Hz lowpass)
  - Quality Audit: P_5 screening for flatlines and rail clipping

Outputs per cohort in data/processed_100hz/<cohort>/:
  - signals.npy (float32 array of shape (N, 12, 1000))
  - preprocessed_database.csv (metadata + quality screening verdicts)
  - preprocessing_summary.json (execution statistics, throughput, rejection breakdown)
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import pandas as pd
import wfdb

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.preprocessing.pipeline import ECGPreprocessingPipeline


# -------------------------------------------------------------------------
# Header & Metadata Extraction Helpers
# -------------------------------------------------------------------------

def parse_chapman_header(hea_path: Path) -> Dict[str, str]:
    """Parses clinical comments from Chapman WFDB header."""
    info = {
        "age": "",
        "sex": "",
        "dx": "",
        "rx": "",
        "hx": "",
        "sx": ""
    }
    try:
        content = hea_path.read_text(encoding="utf-8", errors="ignore")
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("# Age:"):
                info["age"] = line.replace("# Age:", "").strip()
            elif line.startswith("# Sex:"):
                info["sex"] = line.replace("# Sex:", "").strip()
            elif line.startswith("# Dx:"):
                info["dx"] = line.replace("# Dx:", "").strip()
            elif line.startswith("# Rx:"):
                info["rx"] = line.replace("# Rx:", "").strip()
            elif line.startswith("# Hx:"):
                info["hx"] = line.replace("# Hx:", "").strip()
            elif line.startswith("# Sx:"):
                info["sx"] = line.replace("# Sx:", "").strip()
    except Exception:
        pass
    return info


# -------------------------------------------------------------------------
# Multiprocessing Worker
# -------------------------------------------------------------------------

def _process_single_record(task_args: Tuple[str, Dict[str, Any], float, float, str]) -> Dict[str, Any]:
    """
    Worker function executed in parallel CPU processes.
    task_args: (record_path_no_ext, meta_dict, target_fs, target_duration, scaling_method)
    """
    record_path_no_ext, meta_info, target_fs, target_duration, scaling_method = task_args
    rec_id = meta_info.get("record_id", Path(record_path_no_ext).name)
    try:
        raw_signal, wfdb_meta = wfdb.rdsamp(record_path_no_ext)
        pipeline = ECGPreprocessingPipeline(
            target_fs=target_fs,
            target_duration_sec=target_duration,
            scaling_method=scaling_method
        )
        res = pipeline.process(
            raw_signal=raw_signal,
            lead_names=wfdb_meta["sig_name"],
            fs_in=float(wfdb_meta["fs"]),
            units=wfdb_meta.get("units", "mV")
        )
        return {
            "record_id": rec_id,
            "success": True,
            "signal": res.signal.astype(np.float32),
            "passed": res.is_valid,
            "rejection_reasons": "; ".join(res.audit.rejection_reasons),
            "error": None,
            "meta": meta_info
        }
    except Exception as e:
        return {
            "record_id": rec_id,
            "success": False,
            "signal": None,
            "passed": False,
            "rejection_reasons": f"Processing crash: {str(e)}",
            "error": str(e),
            "meta": meta_info
        }


# -------------------------------------------------------------------------
# Batch Processors for Chapman and MIMIC
# -------------------------------------------------------------------------

def process_chapman_cohort(
    chapman_dir: Path,
    output_dir: Path,
    target_fs: float = 100.0,
    target_duration: float = 10.0,
    scaling_method: str = "none",
    workers: int = 8,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """Preprocesses Chapman-Shaoxing dataset into standardized 100 Hz tensors."""
    print("\n" + "="*70)
    print("STARTING CHAPMAN-SHAOXING BATCH PREPROCESSING")
    print("="*70)
    
    # 1. Discover matching .hea and .mat files
    print(f"Scanning directory: {chapman_dir}...")
    hea_files = {f.stem: f for f in chapman_dir.glob("*.hea")}
    mat_files = {f.stem: f for f in chapman_dir.glob("*.mat")}
    valid_stems = sorted(list(set(hea_files.keys()).intersection(mat_files.keys())))
    print(f"Discovered {len(valid_stems):,} complete paired records (.hea + .mat).")
    
    if limit is not None:
        valid_stems = valid_stems[:limit]
        print(f"Limited run to {len(valid_stems):,} records.")

    N = len(valid_stems)
    target_samples = int(round(target_fs * target_duration))

    # 2. Build task list with extracted header comments
    print("Parsing headers and building task catalog...")
    tasks = []
    for stem in valid_stems:
        hea_p = hea_files[stem]
        meta = parse_chapman_header(hea_p)
        meta["record_id"] = stem
        meta["patient_id"] = stem  # 1:1 patient-to-record in Chapman
        record_base = str(hea_p.with_suffix(""))
        tasks.append((record_base, meta, target_fs, target_duration, scaling_method))

    # 3. Parallel execution
    print(f"Launching parallel execution across {workers} CPU workers...")
    out_signals = np.zeros((N, 12, target_samples), dtype=np.float32)
    audit_rows = []
    passed_count = 0
    failed_count = 0
    start_time = time.time()
    chunk_size = 500

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for idx, res in enumerate(executor.map(_process_single_record, tasks, chunksize=50)):
            if res["success"] and res["signal"] is not None:
                out_signals[idx] = res["signal"]
            
            passed = res["passed"]
            if passed:
                passed_count += 1
            else:
                failed_count += 1

            meta = res["meta"]
            audit_rows.append({
                "record_id": meta["record_id"],
                "patient_id": meta["patient_id"],
                "age": meta.get("age", ""),
                "sex": meta.get("sex", ""),
                "dx_snomed": meta.get("dx", ""),
                "rx": meta.get("rx", ""),
                "hx": meta.get("hx", ""),
                "sx": meta.get("sx", ""),
                "preprocessing_success": res["success"],
                "quality_passed": passed,
                "rejection_reasons": res["rejection_reasons"],
            })

            completed = idx + 1
            if completed % chunk_size == 0 or completed == N:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                remaining = (N - completed) / rate if rate > 0 else 0
                print(f"  [Chapman] {completed}/{N} ({completed/N*100:.1f}%) | "
                      f"Passed: {passed_count} | Rejected: {failed_count} | "
                      f"Speed: {rate:.1f} rec/s | ETA: {remaining:.1f}s")

    elapsed_total = time.time() - start_time
    print(f"Chapman processing finished in {elapsed_total:.2f}s ({N/elapsed_total:.1f} rec/s).")

    # 4. Save artifacts
    cohort_out = output_dir / "chapman"
    cohort_out.mkdir(parents=True, exist_ok=True)

    sig_file = cohort_out / "signals.npy"
    print(f"Saving preprocessed signals to {sig_file} ({out_signals.nbytes / (1024**2):.1f} MB)...")
    np.save(sig_file, out_signals)

    meta_file = cohort_out / "preprocessed_database.csv"
    meta_df = pd.DataFrame(audit_rows)
    print(f"Saving enriched metadata to {meta_file}...")
    meta_df.to_csv(meta_file, index=False)

    summary = {
        "cohort": "Chapman-Shaoxing",
        "total_records": N,
        "sampling_rate_hz": target_fs,
        "duration_sec": target_duration,
        "target_samples": target_samples,
        "shape": [N, 12, target_samples],
        "passed_quality_screening": passed_count,
        "passed_percentage": round((passed_count / N) * 100, 2) if N > 0 else 0.0,
        "rejected_quality_screening": failed_count,
        "rejected_percentage": round((failed_count / N) * 100, 2) if N > 0 else 0.0,
        "execution_time_sec": round(elapsed_total, 2),
        "throughput_records_per_sec": round(N / elapsed_total, 2) if elapsed_total > 0 else 0.0,
        "workers_used": workers,
        "output_files": {
            "signals": str(sig_file),
            "metadata": str(meta_file)
        }
    }
    summary_file = cohort_out / "preprocessing_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary report written to {summary_file}.")
    return summary


def process_mimic_cohort(
    mimic_dir: Path,
    output_dir: Path,
    target_fs: float = 100.0,
    target_duration: float = 10.0,
    scaling_method: str = "none",
    workers: int = 8,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """Preprocesses sampled MIMIC-IV-ECG validation cohort into standardized 100 Hz tensors."""
    print("\n" + "="*70)
    print("STARTING MIMIC-IV-ECG BATCH PREPROCESSING")
    print("="*70)

    # 1. Load manifest
    manifest_path = mimic_dir / "records_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"MIMIC records manifest not found at {manifest_path}")

    print(f"Loading manifest from {manifest_path}...")
    manifest_df = pd.read_csv(manifest_path)
    print(f"Found {len(manifest_df):,} records across {manifest_df['subject_id'].nunique():,} unique patients.")

    if limit is not None:
        manifest_df = manifest_df.iloc[:limit]
        print(f"Limited run to {len(manifest_df):,} records.")

    N = len(manifest_df)
    target_samples = int(round(target_fs * target_duration))

    # 2. Build task catalog
    tasks = []
    for _, row in manifest_df.iterrows():
        study_id = str(row["study_id"])
        subject_id = str(row["subject_id"])
        disease = str(row["disease"])
        record_base = str(mimic_dir / study_id)
        meta = {
            "record_id": study_id,
            "study_id": study_id,
            "subject_id": subject_id,
            "patient_id": subject_id,
            "disease": disease,
            "relative_path": row.get("relative_path", "")
        }
        tasks.append((record_base, meta, target_fs, target_duration, scaling_method))

    # 3. Parallel execution
    print(f"Launching parallel execution across {workers} CPU workers...")
    out_signals = np.zeros((N, 12, target_samples), dtype=np.float32)
    audit_rows = []
    passed_count = 0
    failed_count = 0
    start_time = time.time()
    chunk_size = 500

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for idx, res in enumerate(executor.map(_process_single_record, tasks, chunksize=50)):
            if res["success"] and res["signal"] is not None:
                out_signals[idx] = res["signal"]

            passed = res["passed"]
            if passed:
                passed_count += 1
            else:
                failed_count += 1

            meta = res["meta"]
            audit_rows.append({
                "record_id": meta["record_id"],
                "study_id": meta["study_id"],
                "subject_id": meta["subject_id"],
                "patient_id": meta["patient_id"],
                "disease": meta["disease"],
                "relative_path": meta["relative_path"],
                "preprocessing_success": res["success"],
                "quality_passed": passed,
                "rejection_reasons": res["rejection_reasons"],
            })

            completed = idx + 1
            if completed % chunk_size == 0 or completed == N:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                remaining = (N - completed) / rate if rate > 0 else 0
                print(f"  [MIMIC] {completed}/{N} ({completed/N*100:.1f}%) | "
                      f"Passed: {passed_count} | Rejected: {failed_count} | "
                      f"Speed: {rate:.1f} rec/s | ETA: {remaining:.1f}s")

    elapsed_total = time.time() - start_time
    print(f"MIMIC processing finished in {elapsed_total:.2f}s ({N/elapsed_total:.1f} rec/s).")

    # 4. Save artifacts
    cohort_out = output_dir / "mimic"
    cohort_out.mkdir(parents=True, exist_ok=True)

    sig_file = cohort_out / "signals.npy"
    print(f"Saving preprocessed signals to {sig_file} ({out_signals.nbytes / (1024**2):.1f} MB)...")
    np.save(sig_file, out_signals)

    meta_file = cohort_out / "preprocessed_database.csv"
    meta_df = pd.DataFrame(audit_rows)
    print(f"Saving enriched metadata to {meta_file}...")
    meta_df.to_csv(meta_file, index=False)

    summary = {
        "cohort": "MIMIC-IV-ECG (Validation Sample)",
        "total_records": N,
        "sampling_rate_hz": target_fs,
        "duration_sec": target_duration,
        "target_samples": target_samples,
        "shape": [N, 12, target_samples],
        "passed_quality_screening": passed_count,
        "passed_percentage": round((passed_count / N) * 100, 2) if N > 0 else 0.0,
        "rejected_quality_screening": failed_count,
        "rejected_percentage": round((failed_count / N) * 100, 2) if N > 0 else 0.0,
        "execution_time_sec": round(elapsed_total, 2),
        "throughput_records_per_sec": round(N / elapsed_total, 2) if elapsed_total > 0 else 0.0,
        "workers_used": workers,
        "output_files": {
            "signals": str(sig_file),
            "metadata": str(meta_file)
        }
    }
    summary_file = cohort_out / "preprocessing_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary report written to {summary_file}.")
    return summary


# -------------------------------------------------------------------------
# CLI Main
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run Batch Preprocessing for External ECG Cohorts.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["chapman", "mimic", "all"],
        default="all",
        help="Which external validation dataset to preprocess (default: all)"
    )
    parser.add_argument(
        "--chapman_dir",
        type=str,
        default="data/chapman",
        help="Path to Chapman-Shaoxing records directory"
    )
    parser.add_argument(
        "--mimic_dir",
        type=str,
        default="data/external_validation/mimic",
        help="Path to MIMIC external validation directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed_100hz",
        help="Root output directory for preprocessed arrays"
    )
    parser.add_argument(
        "--sampling_rate",
        type=int,
        choices=[100, 500],
        default=100,
        help="Target sampling frequency in Hz (default: 100)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Target recording duration in seconds (default: 10.0)"
    )
    parser.add_argument(
        "--scaling_method",
        type=str,
        choices=["none", "zscore", "robust_zscore"],
        default="none",
        help="Voltage scaling method (default: none, preserves physical mV)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(12, os.cpu_count() or 4),
        help="Number of parallel CPU worker processes"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of records (for smoke tests)"
    )
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("\nPhase 1 Preprocessing Configuration:")
    print(f"  Target Sampling Rate: {args.sampling_rate} Hz")
    print(f"  Target Duration: {args.duration} s ({int(args.sampling_rate * args.duration)} samples)")
    print(f"  Voltage Scaling: {args.scaling_method}")
    print(f"  CPU Workers: {args.workers}")
    print(f"  Output Root: {out_path.resolve()}")
    if args.limit:
        print(f"  LIMIT: {args.limit} records per cohort")

    summaries = {}

    if args.dataset in ["chapman", "all"]:
        chap_dir = Path(args.chapman_dir)
        if not chap_dir.exists():
            print(f"Error: Chapman directory {chap_dir} does not exist.", file=sys.stderr)
        else:
            summaries["chapman"] = process_chapman_cohort(
                chapman_dir=chap_dir,
                output_dir=out_path,
                target_fs=float(args.sampling_rate),
                target_duration=float(args.duration),
                scaling_method=args.scaling_method,
                workers=args.workers,
                limit=args.limit
            )

    if args.dataset in ["mimic", "all"]:
        mim_dir = Path(args.mimic_dir)
        if not mim_dir.exists():
            print(f"Error: MIMIC directory {mim_dir} does not exist.", file=sys.stderr)
        else:
            summaries["mimic"] = process_mimic_cohort(
                mimic_dir=mim_dir,
                output_dir=out_path,
                target_fs=float(args.sampling_rate),
                target_duration=float(args.duration),
                scaling_method=args.scaling_method,
                workers=args.workers,
                limit=args.limit
            )

    print("\n" + "="*70)
    print("ALL REQUESTED PREPROCESSING TASKS COMPLETE!")
    print("="*70)
    for cohort, s in summaries.items():
        print(f"  {cohort.upper()}: {s['total_records']} records | Passed: {s['passed_quality_screening']} ({s['passed_percentage']}%) | Shape: {s['shape']}")


if __name__ == "__main__":
    main()

"""
Full Dataset Preprocessing Script for PTB-XL
=============================================
Executes the Phase 1 Preprocessing Pipeline (P_1 through P_5) across all
21,799 recordings using multi-core parallel processing.

Outputs:
  - signals.npy (float32 array of shape (N, 12, T))
  - preprocessed_metadata.csv (original metadata + quality audit verdicts)
  - preprocessed_summary.json (run statistics, timing, and rejection breakdown)
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import wfdb

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.preprocessing.pipeline import ECGPreprocessingPipeline


# Worker function executed in parallel processes
def _process_single_record(args: Tuple[str, int, float, float, str]) -> Dict[str, Any]:
    file_path, ecg_id, target_fs, target_duration, scaling_method = args
    try:
        raw_signal, meta = wfdb.rdsamp(file_path)
        pipeline = ECGPreprocessingPipeline(
            target_fs=target_fs,
            target_duration_sec=target_duration,
            scaling_method=scaling_method
        )
        res = pipeline.process(
            raw_signal=raw_signal,
            lead_names=meta["sig_name"],
            fs_in=float(meta["fs"]),
            units=meta.get("units", "mV")
        )
        return {
            "ecg_id": ecg_id,
            "success": True,
            "signal": res.signal.astype(np.float32),
            "passed": res.is_valid,
            "rejection_reasons": "; ".join(res.audit.rejection_reasons),
            "error": None
        }
    except Exception as e:
        return {
            "ecg_id": ecg_id,
            "success": False,
            "signal": None,
            "passed": False,
            "rejection_reasons": f"Processing crash: {str(e)}",
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="Run complete PTB-XL Phase 1 preprocessing.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
        help="Path to PTB-XL dataset directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed_100hz",
        help="Destination directory for processed arrays and metadata"
    )
    parser.add_argument(
        "--sampling_rate",
        type=int,
        choices=[100, 500],
        default=100,
        help="Target sampling frequency in Hz (100 or 500)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Target recording duration in seconds"
    )
    parser.add_argument(
        "--scaling_method",
        type=str,
        default="none",
        choices=["none", "zscore", "robust_zscore"],
        help="Voltage scaling method (default 'none' preserves physical mV)"
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
        help="Optional limit on number of records (for test runs)"
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    db_path = dataset_path / "ptbxl_database.csv"
    if not db_path.exists():
        print(f"Error: Database file not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading metadata from {db_path}...")
    df = pd.read_csv(db_path, index_col="ecg_id")
    total_records = len(df)
    if args.limit is not None:
        df = df.iloc[:args.limit]
        print(f"Limited execution to {len(df)} records.")
    else:
        print(f"Total records to process: {len(df)}")

    target_samples = int(round(args.sampling_rate * args.duration))
    print(f"Configuration:")
    print(f"  - Sampling Rate: {args.sampling_rate} Hz")
    print(f"  - Duration: {args.duration} s ({target_samples} samples)")
    print(f"  - Output Shape per signal: (12, {target_samples})")
    print(f"  - CPU Workers: {args.workers}")
    print(f"  - Output Directory: {output_path}")

    # Prepare worker arguments
    file_col = "filename_lr" if args.sampling_rate == 100 else "filename_hr"
    tasks = [
        (
            str(dataset_path / row[file_col]),
            int(ecg_id),
            float(args.sampling_rate),
            float(args.duration),
            args.scaling_method
        )
        for ecg_id, row in df.iterrows()
    ]

    print(f"\nLaunching parallel execution across {args.workers} workers...")
    start_time = time.time()

    # Pre-allocate master signal array (N, 12, target_samples)
    N = len(df)
    all_signals = np.zeros((N, 12, target_samples), dtype=np.float32)
    audit_results = []

    completed = 0
    passed_count = 0
    failed_count = 0
    chunk_size = 500

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for idx, result in enumerate(executor.map(_process_single_record, tasks, chunksize=50)):
            ecg_id = result["ecg_id"]
            if result["success"] and result["signal"] is not None:
                all_signals[idx] = result["signal"]
            
            passed = result["passed"]
            if passed:
                passed_count += 1
            else:
                failed_count += 1

            audit_results.append({
                "ecg_id": ecg_id,
                "preprocessing_success": result["success"],
                "quality_passed": passed,
                "rejection_reasons": result["rejection_reasons"],
            })

            completed += 1
            if completed % chunk_size == 0 or completed == N:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                remaining = (N - completed) / rate if rate > 0 else 0
                print(f"  Processed {completed}/{N} ({completed/N*100:.1f}%) | "
                      f"Passed: {passed_count} | Rejected: {failed_count} | "
                      f"Speed: {rate:.1f} rec/s | ETA: {remaining:.1f}s")

    elapsed_total = time.time() - start_time
    print(f"\nProcessing finished in {elapsed_total:.2f} seconds ({len(df)/elapsed_total:.1f} rec/s).")

    # 1. Save all signals to .npy
    signal_file = output_path / "signals.npy"
    print(f"Saving preprocessed signals to {signal_file} (approx {all_signals.nbytes / (1024**2):.1f} MB)...")
    np.save(signal_file, all_signals)

    # 2. Merge audit results with database metadata and save CSV
    audit_df = pd.DataFrame(audit_results).set_index("ecg_id")
    merged_df = df.join(audit_df)
    meta_file = output_path / "preprocessed_database.csv"
    print(f"Saving enriched metadata to {meta_file}...")
    merged_df.to_csv(meta_file)

    # 3. Save summary JSON
    summary = {
        "dataset": "PTB-XL v1.0.3",
        "total_records": N,
        "sampling_rate_hz": args.sampling_rate,
        "duration_sec": args.duration,
        "target_samples": target_samples,
        "shape": [N, 12, target_samples],
        "passed_quality_screening": passed_count,
        "passed_percentage": round((passed_count / N) * 100, 2),
        "rejected_quality_screening": failed_count,
        "rejected_percentage": round((failed_count / N) * 100, 2),
        "execution_time_sec": round(elapsed_total, 2),
        "throughput_records_per_sec": round(N / elapsed_total, 2),
        "workers_used": args.workers,
        "output_files": {
            "signals": str(signal_file),
            "metadata": str(meta_file)
        }
    }
    summary_file = output_path / "preprocessing_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary report written to {summary_file}.")
    print("\nPhase 1 Preprocessing Complete!")


if __name__ == "__main__":
    main()

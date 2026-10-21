"""
Frozen Benchmark Test Set Generator (Phase 2 Master Step - Option B)
=====================================================================
Extracts the held-out Fold 10 (`split == 'test'`) test records from:
  1. PTB-XL (2,198 test records)
  2. Chapman-Shaoxing (1,015 test records)
  3. MIMIC-IV-ECG (500 test records)
Total = 3,713 test records.

Applies the verified LeadMaskingOperator to pre-generate immutable,
frozen multi-level degraded evaluation arrays:
  - full_12lead.npy (12 leads - clinical baseline)
  - leads_10.npy (10 leads)
  - leads_8.npy (8 leads)
  - leads_6.npy (6 leads - lateral/septal)
  - leads_4.npy (4 leads)
  - leads_2.npy (2 leads - telemetry II, V1)
  - leads_1.npy (1 lead - rhythm strip II)
  - leads_6_limb.npy (frontal plane only)
  - leads_6_precordial.npy (horizontal plane only)

Also saves:
  - labels.npy: (N_test, 8) uint8 multi-hot ground truth
  - test_metadata.csv: index-aligned metadata
  - benchmark_manifest.json: specifications, active leads, and metrics

Research Constraint:
  Zero generative imputation. Masked channels are strictly clamped to 0.0 mV.
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.degradation.lead_masking import (
    CANONICAL_LEADS,
    LEAD_TO_INDEX,
    LeadMaskingOperator,
    get_stepwise_schedule,
    get_anatomical_schedule
)

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

COHORT_CONFIGS = [
    {
        "id": "ptbxl",
        "name": "PTB-XL",
        "dir": repo_root / "data" / "processed_100hz",
        "csv_path": repo_root / "data" / "processed_100hz" / "preprocessed_database.csv",
        "signals_path": repo_root / "data" / "processed_100hz" / "signals.npy",
        "labels_path": repo_root / "data" / "processed_100hz" / "labels.npy"
    },
    {
        "id": "chapman",
        "name": "Chapman-Shaoxing",
        "dir": repo_root / "data" / "processed_100hz" / "chapman",
        "csv_path": repo_root / "data" / "processed_100hz" / "chapman" / "preprocessed_database.csv",
        "signals_path": repo_root / "data" / "processed_100hz" / "chapman" / "signals.npy",
        "labels_path": repo_root / "data" / "processed_100hz" / "chapman" / "labels.npy"
    },
    {
        "id": "mimic",
        "name": "MIMIC-IV-ECG",
        "dir": repo_root / "data" / "processed_100hz" / "mimic",
        "csv_path": repo_root / "data" / "processed_100hz" / "mimic" / "preprocessed_database.csv",
        "signals_path": repo_root / "data" / "processed_100hz" / "mimic" / "signals.npy",
        "labels_path": repo_root / "data" / "processed_100hz" / "mimic" / "labels.npy"
    }
]


def generate_benchmark_test_sets():
    start_time = time.time()
    output_root = repo_root / "data" / "benchmark_test"
    output_root.mkdir(parents=True, exist_ok=True)

    operator = LeadMaskingOperator()
    stepwise_sched = get_stepwise_schedule()
    anatomical_sched = get_anatomical_schedule()

    # Define evaluation masks to persist
    eval_masks = {
        "full_12lead": stepwise_sched["12_lead"],
        "leads_10": stepwise_sched["10_lead"],
        "leads_8": stepwise_sched["8_lead"],
        "leads_6": stepwise_sched["6_lead"],
        "leads_4": stepwise_sched["4_lead"],
        "leads_2": stepwise_sched["2_lead"],
        "leads_1": stepwise_sched["1_lead"],
        "leads_6_limb": anatomical_sched["6_lead_limb"],
        "leads_6_precordial": anatomical_sched["6_lead_precordial"],
    }

    print("=" * 80)
    print("GENERATING FROZEN BENCHMARK TEST SETS (FOLD 10) - OPTION B")
    print(f"Output Root: {output_root}")
    print(f"Mask Variants: {list(eval_masks.keys())}")
    print("=" * 80)

    manifest: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "canonical_leads": list(CANONICAL_LEADS),
        "target_diseases": BENCHMARK_DISEASES,
        "splits_source": "Fold 10 (split == 'test')",
        "masks": {
            k: {
                "name": v.name,
                "num_leads": v.num_leads,
                "active_leads": list(v.active_leads),
                "masked_leads": list(v.masked_leads),
                "mask_vector": v.mask_vector.tolist(),
                "description": v.description
            }
            for k, v in eval_masks.items()
        },
        "cohorts": {},
        "storage_summary": {}
    }

    total_bytes_written = 0
    total_test_records = 0

    for cohort in COHORT_CONFIGS:
        cid = cohort["id"]
        cname = cohort["name"]
        print(f"\n[{cname}] Processing held-out test split...")

        cohort_out_dir = output_root / cid
        cohort_out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load metadata and locate test split
        df = pd.read_csv(cohort["csv_path"], low_memory=False)
        test_mask = (df["split"] == "test").to_numpy()
        test_indices = np.where(test_mask)[0]
        n_test = len(test_indices)
        total_test_records += n_test

        print(f"  -> Found {n_test:,} test records in Fold 10 (out of {len(df):,} total).")

        # Extract aligned metadata
        test_df = df.iloc[test_indices].copy().reset_index(drop=True)
        test_metadata_path = cohort_out_dir / "test_metadata.csv"
        test_df.to_csv(test_metadata_path, index=False)

        # 2. Extract labels
        labels_all = np.load(cohort["labels_path"])
        test_labels = labels_all[test_indices].astype(np.uint8)
        labels_out_path = cohort_out_dir / "labels.npy"
        np.save(labels_out_path, test_labels)

        # 3. Extract baseline signals
        sig_mmap = np.load(cohort["signals_path"], mmap_mode="r")
        test_signals = np.array(sig_mmap[test_indices], dtype=np.float32)

        # Sanitize any physical sensor disconnections (NaN/Inf) to 0.0 mV
        test_signals = np.nan_to_num(test_signals, nan=0.0, posinf=0.0, neginf=0.0)

        # Verify baseline signals
        assert test_signals.shape == (n_test, 12, 1000), f"Unexpected shape {test_signals.shape}"
        assert not np.isnan(test_signals).any(), "Found NaNs in baseline test signals!"
        assert not np.isinf(test_signals).any(), "Found Infs in baseline test signals!"

        # Baseline energy for retention metrics
        orig_energy = float(np.sum(test_signals ** 2))

        cohort_info = {
            "name": cname,
            "num_test_records": int(n_test),
            "signal_shape": list(test_signals.shape),
            "files": {},
            "energy_retention": {}
        }

        # 4. Generate and save each mask variant
        for mask_key, lead_mask in eval_masks.items():
            out_file = cohort_out_dir / f"{mask_key}.npy"

            # Apply mask to entire test batch
            masked_res = operator.apply(test_signals, lead_mask)
            masked_array = masked_res.signal.astype(np.float32)

            # Verification of masking fidelity
            # Inactive leads must be strictly 0.0
            for masked_lead in lead_mask.masked_leads:
                ch_idx = LEAD_TO_INDEX[masked_lead]
                ch_max = float(np.max(np.abs(masked_array[:, ch_idx, :])))
                assert ch_max == 0.0, f"Leaking inactive lead {masked_lead} in {mask_key}: max={ch_max}"

            # Active leads must match original exactly
            for active_lead in lead_mask.active_leads:
                ch_idx = LEAD_TO_INDEX[active_lead]
                diff = float(np.max(np.abs(masked_array[:, ch_idx, :] - test_signals[:, ch_idx, :])))
                assert diff == 0.0, f"Active lead {active_lead} altered in {mask_key}: max_diff={diff}"

            # Save to disk
            np.save(out_file, masked_array)
            file_size = out_file.stat().st_size
            total_bytes_written += file_size

            # Energy retention ratio
            masked_energy = float(np.sum(masked_array ** 2))
            retention_ratio = (masked_energy / orig_energy) if orig_energy > 0 else 0.0

            cohort_info["files"][f"{mask_key}.npy"] = {
                "size_bytes": file_size,
                "size_mb": round(file_size / (1024 * 1024), 2),
                "num_leads": lead_mask.num_leads,
                "active_leads": list(lead_mask.active_leads)
            }
            cohort_info["energy_retention"][mask_key] = round(retention_ratio, 4)

            print(f"  -> Saved {out_file.name:<25} ({file_size / (1024*1024):>6.1f} MB) | Leads: {lead_mask.num_leads:>2} | Energy Ret: {retention_ratio*100:>5.1f}%")

        manifest["cohorts"][cid] = cohort_info

    # Save manifest
    manifest["total_test_records"] = total_test_records
    manifest["storage_summary"]["total_bytes"] = total_bytes_written
    manifest["storage_summary"]["total_mb"] = round(total_bytes_written / (1024 * 1024), 2)
    manifest["storage_summary"]["total_gb"] = round(total_bytes_written / (1024 ** 3), 3)

    manifest_path = output_root / "benchmark_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"BENCHMARK GENERATION COMPLETED SUCCESSFULLY IN {elapsed:.2f}s")
    print(f"Total Test Records across all 3 cohorts: {total_test_records:,}")
    print(f"Total Frozen Benchmark Storage: {manifest['storage_summary']['total_gb']} GB ({manifest['storage_summary']['total_mb']} MB)")
    print(f"Manifest written to: {manifest_path}")
    print("=" * 80)

    # Generate markdown report
    generate_markdown_report(manifest)


def generate_markdown_report(manifest: Dict[str, Any]):
    report_path = repo_root / "reports" / "benchmark_test_set_summary.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Frozen Benchmark Test Set Summary (Option B)",
        "",
        f"**Generated:** {manifest['generated_at']}  ",
        f"**Source Folds:** Fold 10 (`split == 'test'`) from Patient-Quarantined Multi-Label Stratification  ",
        f"**Total Records:** {manifest['total_test_records']:,} test patients  ",
        f"**Total Disk Usage:** {manifest['storage_summary']['total_gb']} GB ({manifest['storage_summary']['total_mb']} MB)  ",
        "",
        "## 1. Cohort Breakdown",
        "",
        "| Cohort | Test Records | Baseline Array Shape | Labels Array Shape | Storage (MB) |",
        "| :--- | :---: | :---: | :---: | :---: |"
    ]

    for cid, info in manifest["cohorts"].items():
        cohort_mb = sum(f["size_mb"] for f in info["files"].values())
        lines.append(
            f"| **{info['name']}** | {info['num_test_records']:,} | `({info['num_test_records']}, 12, 1000)` | `({info['num_test_records']}, 8)` | {cohort_mb:.1f} MB |"
        )

    lines.extend([
        "",
        "## 2. Lead Degradation Levels & Energy Retention",
        "",
        "| Mask Key | Active Leads | Masked Leads | PTB-XL Retention | Chapman Retention | MIMIC Retention |",
        "| :--- | :--- | :--- | :---: | :---: | :---: |"
    ])

    for mask_key, mspec in manifest["masks"].items():
        act_str = ", ".join(mspec["active_leads"]) if len(mspec["active_leads"]) <= 6 else f"{len(mspec['active_leads'])} leads"
        mask_str = ", ".join(mspec["masked_leads"]) if mspec["masked_leads"] else "None"
        
        ptb_ret = manifest["cohorts"]["ptbxl"]["energy_retention"].get(mask_key, 0.0) * 100
        chap_ret = manifest["cohorts"]["chapman"]["energy_retention"].get(mask_key, 0.0) * 100
        mim_ret = manifest["cohorts"]["mimic"]["energy_retention"].get(mask_key, 0.0) * 100

        lines.append(
            f"| `{mask_key}` | {act_str} | {mask_str} | {ptb_ret:.1f}% | {chap_ret:.1f}% | {mim_ret:.1f}% |"
        )

    lines.extend([
        "",
        "## 3. Directory Layout",
        "",
        "```",
        "data/benchmark_test/",
        "├── benchmark_manifest.json",
        "├── ptbxl/",
        "│   ├── full_12lead.npy",
        "│   ├── labels.npy",
        "│   ├── test_metadata.csv",
        "│   ├── leads_10.npy",
        "│   ├── leads_8.npy",
        "│   ├── leads_6.npy",
        "│   ├── leads_4.npy",
        "│   ├── leads_2.npy",
        "│   ├── leads_1.npy",
        "│   ├── leads_6_limb.npy",
        "│   └── leads_6_precordial.npy",
        "├── chapman/",
        "│   └── ...",
        "└── mimic/",
        "    └── ...",
        "```",
        "",
        "## 4. Verification Guarantees",
        "",
        "- **Strict Zero Inactive Channel Leakage:** All masked channels strictly equal $0.0\\text{ mV}$.",
        "- **Bitwise Active Channel Fidelity:** All active channels preserve their raw standardized values with zero transformation.",
        "- **Zero Imputation:** No artificial synthesis or lead estimation was performed.",
        "- **Index Alignment:** Row $i$ in `test_metadata.csv`, `labels.npy`, and all `*.npy` signal variants represents the exact same patient."
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Summary report written to: {report_path}")


if __name__ == "__main__":
    generate_benchmark_test_sets()

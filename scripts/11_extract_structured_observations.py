"""
Script 11: Extract Structured Clinical Observations O_l across Test Benchmarks (Phase 3)
========================================================================================
Extracts discrete structured observations O_l (rhythm, heart rate, PR, QRS, ST, T)
for each record in the frozen test benchmarks across all lead levels (12, 10, 8, 6, 4, 2, 1).

Outputs:
    - data/observations/ptbxl/obs_*.json: Structured observations per test record
    - data/observations/ptbxl/observation_manifest.json: Telemetry summary
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.observations.clinical_extractor import ClinicalObservationExtractor


def extract_all_observations(args: argparse.Namespace) -> None:
    benchmark_dir = Path(args.benchmark_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = ClinicalObservationExtractor(fs=100.0)

    lead_files = {
        "12_lead": ("full_12lead.npy", "mask_full_12lead.npy"),
        "10_lead": ("leads_10.npy", "mask_leads_10.npy"),
        "8_lead": ("leads_8.npy", "mask_leads_8.npy"),
        "6_lead": ("leads_6.npy", "mask_leads_6.npy"),
        "4_lead": ("leads_4.npy", "mask_leads_4.npy"),
        "2_lead": ("leads_2.npy", "mask_leads_2.npy"),
        "1_lead": ("leads_1.npy", "mask_leads_1.npy"),
    }

    manifest: Dict[str, Any] = {
        "total_test_records": 2198,
        "sampling_rate_hz": 100.0,
        "benchmarks": {},
    }

    print("[*] Extracting structured clinical observations across lead levels...")

    for level_name, (sig_file, mask_file) in lead_files.items():
        sig_path = benchmark_dir / sig_file
        mask_path = benchmark_dir / mask_file

        if not sig_path.exists():
            print(f"[!] File {sig_path} not found. Skipping.")
            continue

        signals = np.load(sig_path, mmap_mode="r")
        masks = np.load(mask_path) if mask_path.exists() else None

        num_records = min(len(signals), args.limit) if args.limit else len(signals)
        print(f"  --> Extracting {level_name} ({num_records} records)...", end="", flush=True)

        observations_list: List[Dict[str, Any]] = []
        st_elevation_count = 0
        st_depression_count = 0

        for i in range(num_records):
            sig = np.array(signals[i], dtype=np.float32)
            mask = masks[i] if masks is not None else None

            obs = extractor.extract(sig, lead_mask=mask)
            d = obs.to_dict()
            observations_list.append(d)

            if d["st_elevation"]["present"]:
                st_elevation_count += 1
            if d["st_depression"]["present"]:
                st_depression_count += 1

        # Save observations JSON
        out_file = output_dir / f"obs_{level_name}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(observations_list, f, indent=2)

        manifest["benchmarks"][level_name] = {
            "records_extracted": num_records,
            "output_file": out_file.name,
            "st_elevation_prevalence": round(st_elevation_count / num_records, 4),
            "st_depression_prevalence": round(st_depression_count / num_records, 4),
        }

        print(f" Done. Saved to {out_file.name} (ST Elev: {st_elevation_count}, ST Dep: {st_depression_count})")

    # Save manifest
    manifest_path = output_dir / "observation_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[*] Complete! Observation extraction manifest written to {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract structured clinical observations across benchmarks.")
    parser.add_argument("--benchmark-dir", type=str, default="data/benchmark_test/ptbxl")
    parser.add_argument("--output-dir", type=str, default="data/observations/ptbxl")
    parser.add_argument("--limit", type=int, default=None, help="Optional record limit for quick testing.")

    args = parser.parse_args()
    extract_all_observations(args)

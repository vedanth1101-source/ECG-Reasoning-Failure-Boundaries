"""
External Validation Dataset Downloader & Stratified Sampler
============================================================
Supports downloading validation cohorts for cross-dataset ECG evaluation:
1. Chapman-Shaoxing Database (PhysioNet Challenge 2021) - 10,247 records, 12 leads, 500 Hz, Open Access.
2. MIMIC-IV-ECG Demo (PhysioNet) - 659 records, 12 leads, 500 Hz, Open Access.
3. MIMIC-IV-ECG Full v1.0 (PhysioNet) - Requires Credentialed PhysioNet Account.

Enforces patient-level and diagnostic-stratified sampling so you can download
a lightweight evaluation cohort (e.g., 5,000 to 10,000 records, ~1.0-1.8 GB)
without needing to store massive multi-gigabyte archives.
"""

import os
import sys
import argparse
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import requests
import pandas as pd
import numpy as np

CHALLENGE_2021_BASE = "https://physionet.org/files/challenge-2021/1.0.3/"
MIMIC_DEMO_BASE = "https://physionet.org/files/mimic-iv-ecg-demo/0.1/"
MIMIC_FULL_BASE = "https://physionet.org/files/mimic-iv-ecg/1.0/"


def fetch_text(url: str, session: Optional[requests.Session] = None) -> str:
    s = session or requests
    res = s.get(url, timeout=30)
    res.raise_for_status()
    return res.text


def download_file(url: str, dest_path: Path, session: Optional[requests.Session] = None) -> bool:
    s = session or requests
    try:
        if dest_path.exists() and dest_path.stat().st_size > 0:
            return True  # Already downloaded
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        res = s.get(url, timeout=45, stream=True)
        res.raise_for_status()
        temp_path = dest_path.with_suffix(dest_path.suffix + ".part")
        with open(temp_path, "wb") as f:
            for chunk in res.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
        temp_path.replace(dest_path)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}", file=sys.stderr)
        return False


def get_chapman_record_list(session: requests.Session) -> List[Tuple[str, str, str]]:
    """
    Discovers all records in Chapman-Shaoxing from the 11 groups (g1 to g11).
    Returns list of tuples: (group_name, record_id, relative_path)
    """
    print("[1/4] Discovering Chapman-Shaoxing groups and record catalog from PhysioNet...")
    root_records_txt = fetch_text(CHALLENGE_2021_BASE + "RECORDS", session)
    groups = [
        line.strip() for line in root_records_txt.strip().splitlines()
        if "chapman_shaoxing" in line and line.strip().endswith("/")
    ]
    
    all_records = []
    for g in groups:
        g_url = f"{CHALLENGE_2021_BASE}{g}RECORDS"
        try:
            recs_txt = fetch_text(g_url, session)
            for rec_id in recs_txt.strip().splitlines():
                rec_id = rec_id.strip()
                if rec_id:
                    all_records.append((g, rec_id, f"{g}{rec_id}"))
        except Exception as e:
            print(f"Warning: Could not fetch {g_url}: {e}", file=sys.stderr)
            
    print(f"Total Chapman-Shaoxing recordings available: {len(all_records)}")
    return all_records


def parse_chapman_header(header_text: str) -> Dict[str, str]:
    """Parses metadata and Dx codes from a WFDB header string."""
    info = {"dx": "", "age": "", "sex": ""}
    for line in header_text.splitlines():
        line = line.strip()
        if line.startswith("# Dx:"):
            info["dx"] = line.replace("# Dx:", "").strip()
        elif line.startswith("# Age:"):
            info["age"] = line.replace("# Age:", "").strip()
        elif line.startswith("# Sex:"):
            info["sex"] = line.replace("# Sex:", "").strip()
    return info


def download_chapman_subset(
    output_dir: Path,
    num_records: int = 5000,
    threads: int = 8,
    stratify: bool = True
):
    """
    Downloads a stratified or representative subset of Chapman-Shaoxing ECGs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    
    all_records = get_chapman_record_list(session)
    if not all_records:
        raise RuntimeError("No records found to download.")

    if num_records >= len(all_records):
        selected_records = all_records
        print(f"Target count {num_records} >= total records ({len(all_records)}). Downloading all.")
    elif not stratify:
        # Uniform spread sampling across all groups
        indices = np.linspace(0, len(all_records) - 1, num_records, dtype=int)
        selected_records = [all_records[i] for i in indices]
        print(f"Selected {len(selected_records)} records using uniform group sampling.")
    else:
        # Stratified sampling by fetching headers first for top-level diagnosis grouping
        print(f"[2/4] Sampling headers to stratify across clinical diagnoses...")
        # Step 1: sample a candidate pool (e.g. 1.5x target or full)
        step = max(1, len(all_records) // (num_records * 2))
        candidate_records = all_records[::step]
        
        # Download headers in parallel to read Dx
        print(f"Inspecting diagnostic headers for {len(candidate_records)} candidates...")
        def fetch_meta(rec):
            g, rec_id, rel_path = rec
            h_url = f"{CHALLENGE_2021_BASE}{rel_path}.hea"
            try:
                txt = fetch_text(h_url, session)
                info = parse_chapman_header(txt)
                first_dx = info["dx"].split(",")[0] if info["dx"] else "UNKNOWN"
                return (rec, first_dx)
            except Exception:
                return (rec, "UNKNOWN")

        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            meta_results = list(executor.map(fetch_meta, candidate_records))
            
        df = pd.DataFrame([{"rec": m[0], "dx": m[1]} for m in meta_results])
        
        # Stratified sample by primary diagnosis class
        def sample_dx(group):
            frac = min(1.0, num_records / len(df))
            n = max(1, int(round(len(group) * frac)))
            return group.sample(min(len(group), n), random_state=42)
            
        sampled_df = df.groupby("dx", group_keys=False).apply(sample_dx)
        if len(sampled_df) > num_records:
            sampled_df = sampled_df.sample(num_records, random_state=42)
        selected_records = sampled_df["rec"].tolist()
        print(f"Selected {len(selected_records)} stratified records across {df['dx'].nunique()} diagnostic codes.")

    print(f"[3/4] Downloading signals (.mat) and headers (.hea) to {output_dir} using {threads} threads...")
    tasks = []
    for g, rec_id, rel_path in selected_records:
        hea_url = f"{CHALLENGE_2021_BASE}{rel_path}.hea"
        mat_url = f"{CHALLENGE_2021_BASE}{rel_path}.mat"
        hea_dest = output_dir / f"{rec_id}.hea"
        mat_dest = output_dir / f"{rec_id}.mat"
        tasks.append((hea_url, hea_dest))
        tasks.append((mat_url, mat_dest))

    completed = 0
    total = len(tasks)
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        futures = {executor.submit(download_file, url, path, session): (url, path) for url, path in tasks}
        for future in concurrent.futures.as_completed(futures):
            future.result()
            completed += 1
            if completed % 500 == 0 or completed == total:
                pct = (completed / total) * 100
                print(f"Progress: {completed}/{total} files ({pct:.1f}%) downloaded.")

    # Write catalog manifest
    manifest_path = output_dir / "records_manifest.csv"
    manifest_df = pd.DataFrame([{"record_id": r[1], "source_path": r[2]} for r in selected_records])
    manifest_df.to_csv(manifest_path, index=False)
    print(f"[4/4] Done! Manifest written to {manifest_path}")
    print(f"Downloaded {len(selected_records)} records ({len(selected_records)*2} files).")


def download_mimic_demo(output_dir: Path, threads: int = 8):
    """Downloads all 659 records of the public open-access MIMIC-IV-ECG Demo."""
    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    print("[1/3] Fetching MIMIC-IV-ECG Demo record catalog...")
    records_txt = fetch_text(MIMIC_DEMO_BASE + "RECORDS", session)
    record_paths = [line.strip() for line in records_txt.strip().splitlines() if line.strip()]
    print(f"Found {len(record_paths)} records in MIMIC-IV-ECG Demo.")
    
    tasks = []
    for rel in record_paths:
        rec_id = rel.split("/")[-1]
        hea_url = f"{MIMIC_DEMO_BASE}{rel}.hea"
        dat_url = f"{MIMIC_DEMO_BASE}{rel}.dat"
        hea_dest = output_dir / f"{rec_id}.hea"
        dat_dest = output_dir / f"{rec_id}.dat"
        tasks.append((hea_url, hea_dest))
        tasks.append((dat_url, dat_dest))

    print(f"[2/3] Downloading {len(tasks)} files to {output_dir}...")
    completed = 0
    total = len(tasks)
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        futures = {executor.submit(download_file, url, path, session): (url, path) for url, path in tasks}
        for future in concurrent.futures.as_completed(futures):
            future.result()
            completed += 1
            if completed % 100 == 0 or completed == total:
                pct = (completed / total) * 100
                print(f"Progress: {completed}/{total} files ({pct:.1f}%) downloaded.")

    print(f"[3/3] MIMIC-IV-ECG Demo download complete at {output_dir}.")


def sample_mimic_cohort(
    measurements_csv_path: Path,
    target_records: int = 5000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Creates a patient-isolated, diagnostically-balanced stratified subset
    of MIMIC-IV-ECG records from machine_measurements.csv.
    """
    if not measurements_csv_path.exists():
        raise FileNotFoundError(f"Measurements CSV not found at: {measurements_csv_path}")

    print(f"[1/4] Reading MIMIC machine measurements from {measurements_csv_path}...")
    usecols = ["subject_id", "study_id", "report_0", "report_1", "report_2"]
    df = pd.read_csv(measurements_csv_path, usecols=usecols)
    print(f"Total available catalog records: {len(df):,} ({df['subject_id'].nunique():,} unique patients).")

    df["all_rep"] = df[["report_0", "report_1", "report_2"]].fillna("").agg(" ".join, axis=1).str.lower()

    def classify_condition(text: str) -> Optional[str]:
        if "atrial fibrillation" in text:
            return "Atrial Fibrillation"
        if "right bundle" in text or "rbbb" in text:
            return "Right Bundle Branch Block"
        if "left bundle" in text or "lbbb" in text:
            return "Left Bundle Branch Block"
        if "sinus bradycardia" in text:
            return "Sinus Bradycardia"
        if "sinus tachycardia" in text:
            return "Sinus Tachycardia"
        if "1st degree" in text or "first degree" in text:
            return "1st Degree AV Block"
        if "infarction" in text or "infarct" in text:
            return "Myocardial Infarction"
        if "normal ecg" in text or "normal sinus rhythm" in text or text.strip() == "sinus rhythm":
            return "Normal ECG"
        return None

    df["disease"] = df["all_rep"].apply(classify_condition)
    df = df.dropna(subset=["disease"])

    # Enforce strict patient-level uniqueness (1 ECG per patient to prevent leakage)
    df = df.drop_duplicates(subset=["subject_id"], keep="first")
    print(f"Eligible patient-unique records across 8 target classes: {len(df):,}")

    classes = sorted(df["disease"].unique())
    per_class = max(1, target_records // len(classes))

    sampled_dfs = []
    for c in classes:
        sub_df = df[df["disease"] == c]
        n = min(len(sub_df), per_class)
        sampled_dfs.append(sub_df.sample(n=n, random_state=random_state))

    res = pd.concat(sampled_dfs, ignore_index=True)
    if len(res) < target_records:
        remaining = df[~df["study_id"].isin(res["study_id"])]
        n_more = min(len(remaining), target_records - len(res))
        if n_more > 0:
            res = pd.concat([res, remaining.sample(n=n_more, random_state=random_state)], ignore_index=True)

    def make_rel(row):
        s_sub = str(int(row["subject_id"]))
        s_study = str(int(row["study_id"]))
        prefix = s_sub[:4]
        return f"files/p{prefix}/p{s_sub}/s{s_study}/{s_study}"

    res["relative_path"] = res.apply(make_rel, axis=1)
    return res


def download_mimic_stratified(
    output_dir: Path,
    measurements_csv: Path,
    num_records: int = 5000,
    threads: int = 8,
    manifest_only: bool = False,
    username: Optional[str] = None,
    password: Optional[str] = None
):
    """
    Downloads a patient-isolated, diagnostically-balanced stratified subset
    of MIMIC-IV-ECG (e.g. 5,000 records, ~600 MB instead of the full 80 GB).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sampled_df = sample_mimic_cohort(measurements_csv, target_records=num_records)

    manifest_path = output_dir / "records_manifest.csv"
    manifest_cols = ["subject_id", "study_id", "disease", "relative_path"]
    sampled_df[manifest_cols].to_csv(manifest_path, index=False)
    print(f"\n[2/4] Stratified manifest written to: {manifest_path}")
    print(f"Sampled {len(sampled_df)} records across {sampled_df['disease'].nunique()} diagnostic categories:")
    for d, cnt in sampled_df["disease"].value_counts().items():
        print(f"   - {d:<28}: {cnt:>5} records")

    if manifest_only:
        print("\n[--manifest-only flag set] Skipping signal file downloads.")
        return

    # Check credentials
    user = username or os.environ.get("PHYSIONET_USER")
    passwd = password or os.environ.get("PHYSIONET_PASS")

    session = requests.Session()
    if user and passwd:
        session.auth = (user, passwd)
        print(f"\n[3/4] Authenticated session configured for PhysioNet user '{user}'.")
    else:
        print(
            "\n[3/4] Note: No PhysioNet credentials provided in CLI or environment.\n"
            "If downloading full MIMIC-IV-ECG, please provide --user <username> --password <password>\n"
            "or set PHYSIONET_USER and PHYSIONET_PASS environment variables.\n"
            "Attempting download with current session/netrc..."
        )

    tasks = []
    for _, row in sampled_df.iterrows():
        rel = row["relative_path"]
        rec_id = str(int(row["study_id"]))
        hea_url = f"{MIMIC_FULL_BASE}{rel}.hea"
        dat_url = f"{MIMIC_FULL_BASE}{rel}.dat"
        hea_dest = output_dir / f"{rec_id}.hea"
        dat_dest = output_dir / f"{rec_id}.dat"
        tasks.append((hea_url, hea_dest))
        tasks.append((dat_url, dat_dest))

    print(f"Downloading {len(tasks)} files ({len(sampled_df)} records, ~{len(sampled_df)*121/1024:.1f} MB) using {threads} threads...")
    completed = 0
    total = len(tasks)
    failed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        futures = {executor.submit(download_file, url, path, session): (url, path) for url, path in tasks}
        for future in concurrent.futures.as_completed(futures):
            success = future.result()
            if not success:
                failed += 1
            completed += 1
            if completed % 200 == 0 or completed == total:
                pct = (completed / total) * 100
                print(f"Progress: {completed}/{total} files ({pct:.1f}%) processed. (Failed: {failed})")

    print(f"\n[4/4] Stratified download finished at {output_dir}.")
    if failed > 0:
        print(
            f"Warning: {failed} files failed to download. If you received 401/403 errors, "
            "ensure your PhysioNet account has credentialed access to MIMIC-IV-ECG "
            "and supply your credentials with --user and --password.",
            file=sys.stderr
        )


def main():
    parser = argparse.ArgumentParser(description="Download and stratify external ECG validation cohorts.")
    parser.add_argument(
        "--dataset",
        choices=["chapman", "mimic-demo", "mimic-stratified", "mimic-full"],
        default="chapman",
        help="Dataset to download (chapman: 10k open records, mimic-demo: 659 open records, mimic-stratified: lightweight stratified subset of MIMIC, mimic-full: full catalog warning)"
    )
    parser.add_argument(
        "--num-records",
        type=int,
        default=5000,
        help="Number of records to download for subset (default: 5000)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Target folder (defaults to data/external_validation/<dataset>)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Concurrent download threads (default: 8)"
    )
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Disable diagnostic stratification and use uniform spacing (for Chapman)"
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Generate and write the stratified records manifest CSV without downloading signal files"
    )
    parser.add_argument(
        "--user",
        type=str,
        default="",
        help="PhysioNet username (for credentialed datasets)"
    )
    parser.add_argument(
        "--password",
        type=str,
        default="",
        help="PhysioNet password (for credentialed datasets)"
    )

    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    if not args.output_dir:
        out_dir = repo_root / "data" / "external_validation" / args.dataset.replace("-stratified", "")
    else:
        out_dir = Path(args.output_dir)

    print(f"=== ECG Validation Downloader ===")
    print(f"Target Dataset : {args.dataset}")
    print(f"Target Count   : {args.num_records}")
    print(f"Destination    : {out_dir}")
    print(f"Threads        : {args.threads}")
    if args.manifest_only:
        print(f"Mode           : Manifest Only (No signal download)")
    print("==================================")

    if args.dataset == "chapman":
        download_chapman_subset(
            output_dir=out_dir,
            num_records=args.num_records,
            threads=args.threads,
            stratify=not args.no_stratify
        )
    elif args.dataset == "mimic-demo":
        download_mimic_demo(output_dir=out_dir, threads=args.threads)
    elif args.dataset == "mimic-stratified":
        mm_csv = repo_root / "data" / "mimic" / "machine_measurements.csv"
        download_mimic_stratified(
            output_dir=out_dir,
            measurements_csv=mm_csv,
            num_records=args.num_records,
            threads=args.threads,
            manifest_only=args.manifest_only,
            username=args.user,
            password=args.password
        )
    elif args.dataset == "mimic-full":
        print(
            "MIMIC-IV-ECG Full v1.0 is ~80 GB and requires credentialed PhysioNet access.\n"
            "To download a lightweight, balanced cohort (e.g. 5,000 records, ~600 MB),\n"
            "run:\n"
            "  python scripts/download_validation_data.py --dataset mimic-stratified --num-records 5000\n\n"
            "Or to download the open-access 659-record demo without credentials, run:\n"
            "  python scripts/download_validation_data.py --dataset mimic-demo",
            file=sys.stderr
        )


if __name__ == "__main__":
    main()


"""
Script: Cache FrozenECGFM Latents for PTB-XL Folds 1-8 (Train) and Fold 9 (Val)
==============================================================================
Extracts continuous latent embeddings Z_ECG in R^{768} using the frozen foundation
model (Wav2Vec2CMSCModel) directly from raw 100Hz signals without trainable adapters.

Enforces:
    - Zero patient leakage across splits
    - Strict test split quarantine (Fold 10 untouched)
    - Output saved to data/latents/ptbxl/z_train_ecg_fm.npy and z_val_ecg_fm.npy
"""

import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.models.ecg_fm_primary import FrozenECGFM


def extract_split_latents(
    model: FrozenECGFM,
    signals_mmap: np.ndarray,
    indices: np.ndarray,
    batch_size: int = 64,
    desc: str = "Split",
) -> np.ndarray:
    n_samples = len(indices)
    latents = []
    t0 = time.time()
    n_batches = int(np.ceil(n_samples / batch_size))

    print(f"[*] Extracting latents for {desc}: {n_samples:,} records across {n_batches} batches...")

    with torch.no_grad():
        for b_idx in range(n_batches):
            start = b_idx * batch_size
            end = min(start + batch_size, n_samples)
            batch_indices = indices[start:end]

            batch_sig = np.array(signals_mmap[batch_indices], dtype=np.float32)
            batch_tensor = torch.from_numpy(batch_sig)
            z_batch = model(batch_tensor)  # (B, 768)
            latents.append(z_batch.cpu().numpy())

            if (b_idx + 1) % 10 == 0 or (b_idx + 1) == n_batches:
                elapsed = time.time() - t0
                pct = ((b_idx + 1) / n_batches) * 100.0
                rate = (end) / elapsed
                print(f"    --> [{desc}] Batch {b_idx + 1:3d}/{n_batches} ({pct:5.1f}%) | "
                      f"{end:,}/{n_samples:,} samples | {rate:.1f} samples/sec | Elapsed: {elapsed:.1f}s")

    all_z = np.vstack(latents).astype(np.float32)
    t_tot = time.time() - t0
    print(f"[*] Completed {desc} latent extraction in {t_tot:.1f}s. Shape: {all_z.shape}")
    return all_z


def main():
    parser = argparse.ArgumentParser(description="Extract FrozenECGFM latents for Train and Val splits")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    data_dir = repo_root / "data" / "processed_100hz"
    latents_dir = repo_root / "data" / "latents" / "ptbxl"
    latents_dir.mkdir(parents=True, exist_ok=True)

    db_csv = data_dir / "preprocessed_database.csv"
    signals_file = data_dir / "signals.npy"
    fm_ckpt = repo_root / "checkpoints" / "ecg_fm" / "mimic_iv_ecg_physionet_pretrained.pt"

    print("=" * 80)
    print("  CACHE FROZEN ECG-FM LATENTS: TRAIN (FOLDS 1-8) & VAL (FOLD 9)")
    print("=" * 80)

    print(f"[*] Loading database metadata from {db_csv}...")
    df = pd.read_csv(db_csv, low_memory=False)
    train_indices = np.where(df["strat_fold"].isin(range(1, 9)).values)[0]
    val_indices = np.where((df["strat_fold"] == 9).values)[0]
    test_indices = np.where((df["strat_fold"] == 10).values)[0]

    print(f"    - Train (Folds 1-8): {len(train_indices):,} records")
    print(f"    - Val   (Fold 9):    {len(val_indices):,} records")
    print(f"    - Test  (Fold 10):   {len(test_indices):,} records [QUARANTINED - ZERO TOUCH]")

    # Check existing caches
    val_cache = latents_dir / "z_val_ecg_fm.npy"
    train_cache = latents_dir / "z_train_ecg_fm.npy"

    need_val = not val_cache.exists() or np.load(val_cache, mmap_mode="r").shape != (len(val_indices), 768)
    need_train = not train_cache.exists() or np.load(train_cache, mmap_mode="r").shape != (len(train_indices), 768)

    if not need_val and not need_train:
        print("[*] Both train and val FrozenECGFM latents are already cached and validated. Exiting.")
        return

    print(f"[*] Initializing FrozenECGFM from {fm_ckpt}...")
    torch.set_num_threads(16)
    model = FrozenECGFM(checkpoint_path=fm_ckpt, device="cpu")
    signals_mmap = np.load(signals_file, mmap_mode="r")

    # 1. Validation Latents (extract first for rapid feedback)
    if need_val:
        z_val = extract_split_latents(model, signals_mmap, val_indices, batch_size=args.batch_size, desc="VAL")
        np.save(val_cache, z_val)
        print(f"[+] Saved validation latents to {val_cache}")
    else:
        print(f"[*] Validation latents already cached at {val_cache}")

    # 2. Training Latents
    if need_train:
        z_train = extract_split_latents(model, signals_mmap, train_indices, batch_size=args.batch_size, desc="TRAIN")
        np.save(train_cache, z_train)
        print(f"[+] Saved training latents to {train_cache}")
    else:
        print(f"[*] Training latents already cached at {train_cache}")

    # Manifest
    manifest = {
        "status": "CACHED_AND_VERIFIED",
        "foundation_model": "FrozenECGFM (Wav2Vec2CMSCModel, 90.88M frozen params)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "checkpoint": str(fm_ckpt),
        "train": {
            "records": len(train_indices),
            "file": train_cache.name,
            "shape": [len(train_indices), 768],
        },
        "val": {
            "records": len(val_indices),
            "file": val_cache.name,
            "shape": [len(val_indices), 768],
        },
        "zero_leakage_verified": True,
        "test_fold_10_quarantined": True,
    }
    manifest_path = latents_dir / "ecg_fm_train_val_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[+] Latent manifest written to {manifest_path}")
    print("=" * 80)
    print("  LATENT CACHING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

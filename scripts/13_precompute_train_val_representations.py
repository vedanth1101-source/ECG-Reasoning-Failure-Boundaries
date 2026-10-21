"""
Script 13: Precompute Offline Representations (Z_ECG and O_l) for Entire Training & Validation Cohorts
======================================================================================================
Pre-extracts both:
    1. Continuous Latent Vectors Z_ECG (in R^768) via AdaptedECGModel
    2. Discrete Structured Observations O_l (rhythm, HR, PR, QRS, ST, T) via ClinicalObservationExtractor

Cohorts:
    - Train (Folds 1-8): 17,418 records
    - Val   (Fold 9):    2,183 records
    (Test   (Fold 10):   2,198 records already extracted in Scripts 10 & 11)

Outputs:
    - data/latents/ptbxl/z_train.npy (17418, 768)
    - data/latents/ptbxl/z_val.npy   (2183, 768)
    - data/latents/ptbxl/labels_train.npy (17418, 8)
    - data/latents/ptbxl/labels_val.npy   (2183, 8)
    - data/observations/ptbxl/obs_train.json (17418 records)
    - data/observations/ptbxl/obs_val.json   (2183 records)
    - data/latents/ptbxl/train_val_manifest.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import torch

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.models.adapter import BottleneckMLPAdapter
from src.models.backbone_wrapper import ResNet1DBackbone
from src.models.adapted_ecg_model import AdaptedECGModel
from src.observations.clinical_extractor import ClinicalObservationExtractor


def load_model(checkpoint_path: Path, device: torch.device) -> AdaptedECGModel:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    backbone = ResNet1DBackbone(in_channels=12, feature_dim=512)
    adapter = BottleneckMLPAdapter(d_in=512, d_mid=512, d_out=768, dropout=0.0)

    backbone.load_state_dict(checkpoint["backbone_state_dict"])
    adapter.load_state_dict(checkpoint["adapter_state_dict"])

    model = AdaptedECGModel(backbone=backbone, adapter=adapter, freeze_backbone=True).to(device)
    model.eval()
    return model


def precompute_cohort(
    split_name: str,
    indices: np.ndarray,
    signals_mmap: np.ndarray,
    labels_all: np.ndarray,
    model: AdaptedECGModel,
    extractor: ClinicalObservationExtractor,
    latents_dir: Path,
    obs_dir: Path,
    device: torch.device,
    batch_size: int = 128,
) -> Dict[str, Any]:
    num_records = len(indices)
    print(f"\n[*] Processing {split_name.upper()} cohort ({num_records:,} records)...")
    start_time = time.time()

    # 1. Latent Extraction (Z_ECG)
    print(f"  [1/2] Extracting continuous latents Z_ECG in R^768...", end="", flush=True)
    t0 = time.time()
    latents_list = []

    with torch.no_grad():
        for i in range(0, num_records, batch_size):
            batch_idx = indices[i : i + batch_size]
            batch_arr = np.array(signals_mmap[batch_idx], dtype=np.float32)
            batch_tensor = torch.from_numpy(batch_arr).to(device)

            z = model(batch_tensor)  # (B, 768)
            latents_list.append(z.cpu().numpy())

    z_matrix = np.vstack(latents_list).astype(np.float32)
    labels_subset = labels_all[indices]
    t_latent = time.time() - t0
    print(f" Done in {t_latent:.1f}s. Shape: {z_matrix.shape}")

    # Save latents & labels
    z_out_file = latents_dir / f"z_{split_name}.npy"
    labels_out_file = latents_dir / f"labels_{split_name}.npy"
    np.save(z_out_file, z_matrix)
    np.save(labels_out_file, labels_subset)

    # 2. Discrete Observation Extraction (O_l)
    print(f"  [2/2] Extracting discrete structured observations O_l...", end="", flush=True)
    t1 = time.time()
    observations_list: List[Dict[str, Any]] = []

    for idx in indices:
        sig = np.array(signals_mmap[idx], dtype=np.float32)
        obs = extractor.extract(sig)
        observations_list.append(obs.to_dict())

    t_obs = time.time() - t1
    print(f" Done in {t_obs:.1f}s.")

    # Save observations JSON
    obs_out_file = obs_dir / f"obs_{split_name}.json"
    with open(obs_out_file, "w", encoding="utf-8") as f:
        json.dump(observations_list, f, indent=2)

    total_time = time.time() - start_time
    print(f"  [+] {split_name.upper()} complete in {total_time:.1f}s. Saved to:")
    print(f"      - {z_out_file.name}")
    print(f"      - {labels_out_file.name}")
    print(f"      - {obs_out_file.name}")

    return {
        "records": num_records,
        "latent_shape": list(z_matrix.shape),
        "latent_file": z_out_file.name,
        "labels_file": labels_out_file.name,
        "obs_file": obs_out_file.name,
        "extraction_time_seconds": round(total_time, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Precompute train and val representations.")
    parser.add_argument("--data-dir", type=str, default="data/processed_100hz")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/baseline_resnet1d_adapter.pt")
    parser.add_argument("--latents-dir", type=str, default="data/latents/ptbxl")
    parser.add_argument("--obs-dir", type=str, default="data/observations/ptbxl")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cpu", action="store_true", help="Force CPU extraction.")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    latents_dir = Path(args.latents_dir)
    obs_dir = Path(args.obs_dir)
    latents_dir.mkdir(parents=True, exist_ok=True)
    obs_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[*] Precomputing representations using device: {device}")

    # Load database metadata
    db_csv = data_dir / "preprocessed_database.csv"
    if not db_csv.exists():
        raise FileNotFoundError(f"Database metadata not found at {db_csv}")

    df = pd.read_csv(db_csv, low_memory=False)
    train_indices = np.where(df["strat_fold"].isin(range(1, 9)).values)[0]
    val_indices = np.where((df["strat_fold"] == 9).values)[0]

    print(f"[*] Dataset split counts: Train={len(train_indices):,}, Val={len(val_indices):,}")

    # Memory-map signals & labels
    signals_mmap = np.load(data_dir / "signals.npy", mmap_mode="r")
    labels_all = np.load(data_dir / "labels.npy")

    # Load model and extractor
    model = load_model(Path(args.checkpoint), device)
    extractor = ClinicalObservationExtractor(fs=100.0)

    manifest: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": args.checkpoint,
        "device": str(device),
        "cohorts": {},
    }

    # 1. Validation Cohort (2,183 records)
    val_info = precompute_cohort(
        split_name="val",
        indices=val_indices,
        signals_mmap=signals_mmap,
        labels_all=labels_all,
        model=model,
        extractor=extractor,
        latents_dir=latents_dir,
        obs_dir=obs_dir,
        device=device,
        batch_size=args.batch_size,
    )
    manifest["cohorts"]["val"] = val_info

    # 2. Training Cohort (17,418 records)
    train_info = precompute_cohort(
        split_name="train",
        indices=train_indices,
        signals_mmap=signals_mmap,
        labels_all=labels_all,
        model=model,
        extractor=extractor,
        latents_dir=latents_dir,
        obs_dir=obs_dir,
        device=device,
        batch_size=args.batch_size,
    )
    manifest["cohorts"]["train"] = train_info

    # Save manifest
    manifest_file = latents_dir / "train_val_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[*] All training and validation representations successfully precomputed!")
    print(f"[*] Manifest written to {manifest_file}")


if __name__ == "__main__":
    main()

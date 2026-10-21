"""
Script 10: Extract Standardized Latent Representations Z_ECG across Degradation Benchmarks
========================================================================================
Extracts continuous latent embeddings Z_ECG in R^{2198 x 768} for each lead level
in the frozen test benchmark (12, 10, 8, 6, 4, 2, 1 leads) using the adapted model.

Outputs:
    - data/latents/ptbxl/z_*.npy: (2198, 768) float32 arrays
    - data/latents/ptbxl/latent_manifest.json: Verification telemetry and shape audits
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.models.adapter import BottleneckMLPAdapter
from src.models.backbone_wrapper import ResNet1DBackbone
from src.models.adapted_ecg_model import AdaptedECGModel


def extract_latents(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[*] Extracting latents using device: {device}")

    # Checkpoint resolution
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Run scripts/09_train_baseline_adapter.py first.")

    print(f"[*] Loading model weights from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Initialize model
    backbone = ResNet1DBackbone(in_channels=12, feature_dim=512)
    adapter = BottleneckMLPAdapter(d_in=512, d_mid=512, d_out=768, dropout=0.0)

    backbone.load_state_dict(checkpoint["backbone_state_dict"])
    adapter.load_state_dict(checkpoint["adapter_state_dict"])

    model = AdaptedECGModel(backbone=backbone, adapter=adapter, freeze_backbone=True).to(device)
    model.eval()

    # Benchmark inputs directory
    benchmark_dir = Path(args.benchmark_dir)
    if not benchmark_dir.exists():
        raise FileNotFoundError(f"Benchmark directory not found at {benchmark_dir}")

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lead levels in benchmark
    lead_files = {
        "12_lead": "full_12lead.npy",
        "10_lead": "leads_10.npy",
        "8_lead": "leads_8.npy",
        "6_lead": "leads_6.npy",
        "4_lead": "leads_4.npy",
        "2_lead": "leads_2.npy",
        "1_lead": "leads_1.npy",
    }

    manifest: Dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "latent_dimension": 768,
        "device": str(device),
        "benchmarks": {},
    }

    print("[*] Processing frozen test benchmark lead subsets...")

    for level_name, filename in lead_files.items():
        file_path = benchmark_dir / filename
        if not file_path.exists():
            print(f"[!] Warning: Benchmark file {file_path} not found. Skipping.")
            continue

        signals = np.load(file_path, mmap_mode="r")  # (N, 12, 1000)
        num_samples = len(signals)
        batch_size = args.batch_size
        latents_list = []

        print(f"  --> Extracting {level_name} ({num_samples} records)...", end="", flush=True)

        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                batch_arr = np.array(signals[i : i + batch_size], dtype=np.float32)
                batch_tensor = torch.from_numpy(batch_arr).to(device)

                z = model(batch_tensor)  # (B, 768)
                latents_list.append(z.cpu().numpy())

        all_z = np.vstack(latents_list).astype(np.float32)

        # Save latent array
        out_file = output_dir / f"z_{level_name}.npy"
        np.save(out_file, all_z)

        # Telemetry
        mean_val = float(np.mean(all_z))
        std_val = float(np.std(all_z))
        l2_norm = float(np.mean(np.linalg.norm(all_z, axis=1)))

        manifest["benchmarks"][level_name] = {
            "source_file": filename,
            "latent_file": out_file.name,
            "shape": list(all_z.shape),
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            "mean_l2_norm": round(l2_norm, 4),
            "has_nans": bool(np.isnan(all_z).any()),
            "has_infs": bool(np.isinf(all_z).any()),
        }

        print(f" Done. Saved to {out_file.name} (Shape: {all_z.shape}, L2 Norm: {l2_norm:.2f})")

    # Save manifest
    manifest_path = output_dir / "latent_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[*] Complete! Latent extraction manifest written to {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract standardized Z_ECG latents across benchmarks.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/baseline_resnet1d_adapter.pt")
    parser.add_argument("--benchmark-dir", type=str, default="data/benchmark_test/ptbxl")
    parser.add_argument("--output-dir", type=str, default="data/latents/ptbxl")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")

    args = parser.parse_args()
    extract_latents(args)

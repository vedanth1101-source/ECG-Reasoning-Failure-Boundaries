"""
Validation of Primary Frozen ECG-FM Masking Pipeline & Sensitivity (Phase 3 Final)
==================================================================================
Verifies:
  1. Primary fixed 7-level configurations loaded from configs/lead_configurations.json.
  2. For a test ECG, verifies exact tensors passed to FrozenECGFM:
     - Shape, active/masked leads, input checksums.
     - Confirms zero generative imputation / reconstruction.
  3. Evaluates representation sensitivity D_l = 1 - cosine_similarity(Z_12, Z_l)
     across all 7 primary levels directly on FrozenECGFM without any adapter.
"""

import sys
import json
import hashlib
import time
from pathlib import Path
import numpy as np
import torch

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.degradation.lead_masking import (
    LeadMaskingOperator,
    get_stepwise_schedule,
    CANONICAL_LEADS,
    LEAD_TO_INDEX
)
from src.models.ecg_fm_primary import FrozenECGFM


def run_ecg_fm_masking_audit():
    print("================================================================================")
    print("VERIFYING PRIMARY FROZEN ECG-FM MASKING INTEGRITY & INPUT PIPELINE")
    print("================================================================================")
    operator = LeadMaskingOperator()
    model = FrozenECGFM()

    # Load 1 real test ECG from PTB-XL
    ptb_file = repo_root / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3" / "records500" / "00000" / "00001_hr"
    from src.preprocessing.pipeline import ECGPreprocessingPipeline
    import wfdb
    pipeline = ECGPreprocessingPipeline(target_fs=100.0, target_duration_sec=10.0)

    raw, meta = wfdb.rdsamp(str(ptb_file))
    res = pipeline.process(
        raw_signal=raw,
        lead_names=meta["sig_name"],
        fs_in=float(meta["fs"]),
        units=meta.get("units", "mV")
    )
    ecg_12 = res.signal.astype(np.float32)  # (12, 1000)

    primary_levels = ["12_lead", "10_lead", "8_lead", "6_lead", "4_lead", "2_lead", "1_lead"]
    schedule = get_stepwise_schedule()

    single_ecg_log = {}
    representations = {}

    for lvl in primary_levels:
        mask_obj = schedule[lvl]
        masked = operator.apply(ecg_12, mask_obj)

        # Inactive channels check
        masked_idx = [LEAD_TO_INDEX[ld] for ld in masked.masked_leads]
        if masked_idx:
            leakage = float(np.max(np.abs(masked.signal[masked_idx, :])))
        else:
            leakage = 0.0

        # Input checksum
        tensor_bytes = masked.signal.tobytes()
        input_hash = hashlib.sha256(tensor_bytes).hexdigest()[:16]

        # Pass directly to frozen ECG-FM
        z_l = model(masked.signal).squeeze(0).cpu().numpy()  # (768,)
        rep_bytes = z_l.tobytes()
        rep_hash = hashlib.sha256(rep_bytes).hexdigest()[:16]

        representations[lvl] = z_l

        single_ecg_log[lvl] = {
            "num_active_leads": len(masked.active_leads),
            "active_leads": list(masked.active_leads),
            "masked_leads": list(masked.masked_leads),
            "max_inactive_leakage": leakage,
            "input_shape": list(masked.signal.shape),
            "input_sha256": input_hash,
            "representation_shape": list(z_l.shape),
            "representation_sha256": rep_hash,
            "is_reconstructed": masked.is_reconstructed
        }
        print(f"  Level {lvl:>10} ({len(masked.active_leads):2d} leads) -> Input Hash: {input_hash} | Rep Hash: {rep_hash}")

    # Compute cosine distances for this representative ECG
    z_12 = representations["12_lead"]
    norm_12 = np.linalg.norm(z_12)
    distances_sample = {}
    for lvl in primary_levels:
        z_curr = representations[lvl]
        sim = float(np.dot(z_12, z_curr) / (norm_12 * np.linalg.norm(z_curr) + 1e-8))
        distances_sample[lvl] = {
            "cosine_similarity": round(sim, 5),
            "cosine_distance_d_l": round(1.0 - sim, 5)
        }
        print(f"  Level {lvl:>10} -> Cosine Similarity: {sim:.4f} | D_l: {1.0 - sim:.4f}")

    print("\n================================================================================")
    print("RUNNING BATCH REPRESENTATION SENSITIVITY AUDIT ON 50 VALIDATION ECGS")
    print("================================================================================")
    ptb_dir = repo_root / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3" / "records500" / "00000"
    heas = list(ptb_dir.glob("*.hea"))[:50]
    batch_signals = []
    for h in heas:
        b = str(h.with_suffix(""))
        r, m = wfdb.rdsamp(b)
        p_res = pipeline.process(
            raw_signal=r,
            lead_names=m["sig_name"],
            fs_in=float(m["fs"]),
            units=m.get("units", "mV")
        )
        batch_signals.append(p_res.signal.astype(np.float32))
    batch_signals = np.stack(batch_signals)  # (N, 12, 1000)

    # Compute Z_12 for batch
    with torch.no_grad():
        z_12_batch = model(batch_signals).cpu()  # (N, 768)
        norm_12_batch = torch.norm(z_12_batch, p=2, dim=-1, keepdim=True)
        z_12_unit = z_12_batch / (norm_12_batch + 1e-8)

    batch_sensitivity = {}
    for lvl in primary_levels:
        masked_batch = operator.apply(batch_signals, schedule[lvl])
        with torch.no_grad():
            z_l_batch = model(masked_batch.signal).cpu()
            norm_l_batch = torch.norm(z_l_batch, p=2, dim=-1, keepdim=True)
            z_l_unit = z_l_batch / (norm_l_batch + 1e-8)
            sim_batch = torch.sum(z_12_unit * z_l_unit, dim=-1).numpy()
            d_l_batch = 1.0 - sim_batch

        batch_sensitivity[lvl] = {
            "num_active_leads": len(schedule[lvl].active_leads),
            "mean_cosine_similarity": round(float(np.mean(sim_batch)), 5),
            "mean_cosine_distance": round(float(np.mean(d_l_batch)), 5),
            "median_cosine_distance": round(float(np.median(d_l_batch)), 5),
            "std_cosine_distance": round(float(np.std(d_l_batch)), 5),
            "p25": round(float(np.percentile(d_l_batch, 25)), 5),
            "p75": round(float(np.percentile(d_l_batch, 75)), 5),
            "p95": round(float(np.percentile(d_l_batch, 95)), 5)
        }
        print(f"  Batch Level {lvl:>10} -> Mean Similarity: {np.mean(sim_batch):.4f} | Mean D_l: {np.mean(d_l_batch):.4f}")

    audit_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "foundation_model": "Frozen ECG-FM (wav2vec2_cmsc)",
        "checkpoint": "checkpoints/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt",
        "trainable_adapters": False,
        "single_ecg_verification": {
            "record_id": "00001_hr",
            "levels": single_ecg_log,
            "pairwise_distances": distances_sample
        },
        "batch_sensitivity_summary": batch_sensitivity
    }

    out_file = repo_root / "reports" / "ecg_fm_primary_audit.json"
    with open(out_file, "w") as f:
        json.dump(audit_payload, f, indent=2)
    print(f"\nSaved primary ECG-FM audit to {out_file}")


if __name__ == "__main__":
    run_ecg_fm_masking_audit()

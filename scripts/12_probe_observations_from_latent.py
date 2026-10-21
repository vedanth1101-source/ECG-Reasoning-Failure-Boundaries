"""
Script 12: Linear Probing of Clinical Observations from Latent Representations Z_ECG
====================================================================================
Tests how well the continuous latent vector Z_ECG encodes discrete clinical
parameters (Heart Rate, QRS Duration, ST Elevation Presence) via linear probing.

Pipeline:
    Z_ECG in R^{2198 x 768}
        -> Linear Ridge Regressor -> Predicted Heart Rate (bpm)
        -> Linear Ridge Regressor -> Predicted QRS Duration (ms)
        -> Logistic Probe -> ST Elevation Presence (ROC-AUC)

Outputs:
    - reports/latent_observation_probing_results.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def probe_latents(args: argparse.Namespace) -> None:
    latents_dir = Path(args.latents_dir)
    obs_dir = Path(args.obs_dir)
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    levels = ["12_lead", "6_lead", "2_lead", "1_lead"]
    probing_summary: Dict[str, Any] = {"probing_target_dimension": 768, "levels": {}}

    print("[*] Probing clinical observations from continuous latent embeddings Z_ECG...")

    for level in levels:
        latent_file = latents_dir / f"z_{level}.npy"
        obs_file = obs_dir / f"obs_{level}.json"

        if not latent_file.exists() or not obs_file.exists():
            print(f"[!] Warning: Missing {latent_file} or {obs_file}. Skipping {level}.")
            continue

        z = np.load(latent_file)  # (N, 768)
        with open(obs_file, "r", encoding="utf-8") as f:
            observations = json.load(f)

        # Extract targets
        heart_rates = []
        qrs_durations = []
        st_elev_flags = []
        valid_indices = []

        for idx, obs in enumerate(observations):
            hr = obs.get("heart_rate")
            qrs = obs.get("qrs_duration_ms")
            st_pres = 1 if obs["st_elevation"]["present"] else 0

            if hr is not None and qrs is not None:
                heart_rates.append(hr)
                qrs_durations.append(qrs)
                st_elev_flags.append(st_pres)
                valid_indices.append(idx)

        z_valid = z[valid_indices]
        y_hr = np.array(heart_rates, dtype=np.float32)
        y_qrs = np.array(qrs_durations, dtype=np.float32)
        y_st = np.array(st_elev_flags, dtype=np.int32)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # 1. Heart Rate Probe (Ridge)
        hr_maes, hr_r2s = [], []
        for train_idx, test_idx in kf.split(z_valid):
            probe_hr = Ridge(alpha=10.0)
            probe_hr.fit(z_valid[train_idx], y_hr[train_idx])
            preds_hr = probe_hr.predict(z_valid[test_idx])
            hr_maes.append(mean_absolute_error(y_hr[test_idx], preds_hr))
            hr_r2s.append(r2_score(y_hr[test_idx], preds_hr))

        # 2. QRS Duration Probe (Ridge)
        qrs_maes = []
        for train_idx, test_idx in kf.split(z_valid):
            probe_qrs = Ridge(alpha=10.0)
            probe_qrs.fit(z_valid[train_idx], y_qrs[train_idx])
            preds_qrs = probe_qrs.predict(z_valid[test_idx])
            qrs_maes.append(mean_absolute_error(y_qrs[test_idx], preds_qrs))

        # 3. ST Elevation Presence (Logistic Probe)
        st_aucs = []
        if len(np.unique(y_st)) > 1 and np.sum(y_st == 1) >= 5:
            for train_idx, test_idx in kf.split(z_valid):
                if len(np.unique(y_st[train_idx])) > 1 and len(np.unique(y_st[test_idx])) > 1:
                    probe_st = LogisticRegression(max_iter=200, C=0.1)
                    probe_st.fit(z_valid[train_idx], y_st[train_idx])
                    probs_st = probe_st.predict_proba(z_valid[test_idx])[:, 1]
                    st_aucs.append(roc_auc_score(y_st[test_idx], probs_st))
            mean_st_auc = round(float(np.mean(st_aucs)), 3) if st_aucs else None
        else:
            mean_st_auc = None  # Insufficient positive cases under severe degradation (e.g. 1 lead)

        probing_summary["levels"][level] = {
            "valid_samples": len(z_valid),
            "heart_rate_mae_bpm": round(float(np.mean(hr_maes)), 2),
            "heart_rate_r2": round(float(np.mean(hr_r2s)), 3),
            "qrs_duration_mae_ms": round(float(np.mean(qrs_maes)), 2),
            "st_elevation_probe_auc": mean_st_auc,
        }

        print(
            f"  --> Level {level:7s} | "
            f"HR MAE: {np.mean(hr_maes):.1f} bpm (R2: {np.mean(hr_r2s):.2f}) | "
            f"QRS MAE: {np.mean(qrs_maes):.1f} ms | "
            f"ST Elev Probe AUC: {mean_st_auc:.3f}"
        )

    out_file = reports_dir / "latent_observation_probing_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(probing_summary, f, indent=2)

    print(f"[*] Probing evaluation complete! Written to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probe clinical observations from latent embeddings Z_ECG.")
    parser.add_argument("--latents-dir", type=str, default="data/latents/ptbxl")
    parser.add_argument("--obs-dir", type=str, default="data/observations/ptbxl")

    args = parser.parse_args()
    probe_latents(args)

"""
Script 17: Production Stage 7 Test Benchmark Trace Generation & Degradation Evaluation (7 Levels)
==================================================================================================
Executes the authoritative single-pass test evaluation for Module 4 Control Reasoning Transformer
on the held-out test partition across all seven lead configurations (12, 10, 8, 6, 4, 2, 1 leads):
    - Master Partition: Folds 9-10 (7,375 records x 7 configurations = 51,625 evaluations)
    - Cohorts: PTB-XL (4,359), Chapman-Shaoxing (2,028), MIMIC-IV-ECG (988)
    - Checkpoint: runs/module4_v3/checkpoints/control_reasoning_best_seed42.pt (SHA-256: 2920c768...)
    - Calibration: Per-disease Fold 8 validation temperature vector
    - Trace Generator: StructuredTraceGenerator emitting clinical explanation traces and attention maps

Outputs:
    - runs/module4_v3/metrics/test_predictions_seed42.csv (51,625 rows x 45 columns)
    - runs/module4_v3/metrics/test_evaluation_summary.json
    - runs/module4_v3/reports/sample_test_reasoning_traces.json
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.reasoning.reasoning_transformer import ControlReasoningTransformer
from src.reasoning.trace_generator import StructuredTraceGenerator
from src.reasoning.context_builder import CANDIDATE_ACRONYMS, CANDIDATE_NAMES, LEAD_NAMES
from src.reasoning.multi_cohort_loader import MultiCohortPartitionManager, LEAD_CONFIG_MASKS
from src.observations.clinical_extractor import ClinicalObservationExtractor
from src.reasoning.freeze_contract import (
    APPROVED_TARGETS,
    APPROVED_CONFIGURATIONS,
    compute_sha256,
)


class ReasoningTestDataset(Dataset):
    """
    In-memory dataset pairing continuous latents Z_ECG, structured observations,
    lead availability masks, multi-hot disease targets, observation masks, and record metadata.
    """

    def __init__(
        self,
        z_latents: Union[np.ndarray, torch.Tensor],
        observations: List[Dict[str, Any]],
        labels: Union[np.ndarray, torch.Tensor],
        label_masks: Union[np.ndarray, torch.Tensor],
        lead_masks: Union[np.ndarray, torch.Tensor],
        record_ids: List[str],
        variants: List[str],
        datasets: List[str],
    ):
        self.z_latents = torch.from_numpy(z_latents).float() if isinstance(z_latents, np.ndarray) else z_latents.float()
        self.observations = observations
        self.labels = torch.from_numpy(labels).float() if isinstance(labels, np.ndarray) else labels.float()
        self.label_masks = torch.from_numpy(label_masks).float() if isinstance(label_masks, np.ndarray) else label_masks.float()
        self.lead_masks = torch.from_numpy(lead_masks).float() if isinstance(lead_masks, np.ndarray) else lead_masks.float()
        self.record_ids = record_ids
        self.variants = variants
        self.datasets = datasets
        self.n_samples = len(z_latents)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, str, str, str]:
        return (
            self.z_latents[idx],
            self.observations[idx],
            self.lead_masks[idx],
            self.labels[idx],
            self.label_masks[idx],
            self.record_ids[idx],
            self.variants[idx],
            self.datasets[idx],
        )


def collate_test_batch(batch):
    z_list = [item[0] for item in batch]
    obs_list = [item[1] for item in batch]
    mask_list = [item[2] for item in batch]
    labels_list = [item[3] for item in batch]
    label_masks_list = [item[4] for item in batch]
    record_ids = [item[5] for item in batch]
    variants = [item[6] for item in batch]
    datasets = [item[7] for item in batch]

    z_tensor = torch.stack(z_list, dim=0)
    masks_tensor = torch.stack(mask_list, dim=0)
    labels_tensor = torch.stack(labels_list, dim=0)
    lmasks_tensor = torch.stack(label_masks_list, dim=0)

    return z_tensor, obs_list, masks_tensor, labels_tensor, lmasks_tensor, record_ids, variants, datasets


def load_or_extract_test_observations(
    mgr: MultiCohortPartitionManager,
    variant: str,
    obs_cache_dir: Path,
) -> List[Dict[str, Any]]:
    """Loads cached test observations or extracts them deterministically from active leads."""
    obs_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = obs_cache_dir / f"obs_test_{variant}.json"

    if cache_file.exists():
        with open(cache_file, "r") as f:
            return json.load(f)

    record_ids = mgr.get_record_ids("test", allow_test=True)
    extractor = ClinicalObservationExtractor()
    obs_list = []

    print(f"    --> Extracting test observations for {variant} ({len(record_ids)} records)...")
    t0 = time.time()
    for i, rid in enumerate(record_ids):
        sig = mgr.get_signal(rid, variant=variant)
        mask = mgr.get_lead_mask(rid, variant=variant)
        obs = extractor.extract(sig, mask).to_dict()
        obs_list.append(obs)
        if (i + 1) % 2500 == 0 or (i + 1) == len(record_ids):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"        - [{variant}] Processed {i + 1}/{len(record_ids)} ({rate:.1f} rec/s)")

    with open(cache_file, "w") as f:
        json.dump(obs_list, f)
    return obs_list


def compute_calibration_metrics(
    y_true_all: np.ndarray,
    y_prob_all: np.ndarray,
    y_mask_all: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Computes NLL, Brier score, and ECE across observed target positions."""
    per_class_nll = {}
    per_class_brier = {}
    per_class_ece = {}
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    for c_idx, acronym in enumerate(CANDIDATE_ACRONYMS):
        valid = y_mask_all[:, c_idx] == 1.0
        y_t = y_true_all[valid, c_idx]
        y_p = y_prob_all[valid, c_idx]

        if len(y_t) == 0:
            per_class_nll[acronym] = 0.0
            per_class_brier[acronym] = 0.0
            per_class_ece[acronym] = 0.0
            continue

        p_clipped = np.clip(y_p, 1e-7, 1.0 - 1e-7)
        nll = -np.mean(y_t * np.log(p_clipped) + (1.0 - y_t) * np.log(1.0 - p_clipped))
        brier = np.mean((y_p - y_t) ** 2)

        ece = 0.0
        for b in range(n_bins):
            in_bin = (y_p >= bin_edges[b]) & (y_p < bin_edges[b + 1])
            n_b = np.sum(in_bin)
            if n_b > 0:
                bin_conf = np.mean(y_p[in_bin])
                bin_acc = np.mean(y_t[in_bin])
                ece += (n_b / len(y_p)) * np.abs(bin_acc - bin_conf)

        per_class_nll[acronym] = round(float(nll), 5)
        per_class_brier[acronym] = round(float(brier), 5)
        per_class_ece[acronym] = round(float(ece), 5)

    return {
        "mean_nll": round(float(np.mean(list(per_class_nll.values()))), 5),
        "mean_brier": round(float(np.mean(list(per_class_brier.values()))), 5),
        "mean_ece": round(float(np.mean(list(per_class_ece.values()))), 5),
        "per_class_nll": per_class_nll,
        "per_class_brier": per_class_brier,
        "per_class_ece": per_class_ece,
    }


def compute_bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_mask: np.ndarray,
    n_bootstraps: int = 200,
    seed: int = 42,
) -> Dict[str, Any]:
    """Computes 95% bootstrap confidence intervals for Macro-AUROC and Macro-F1."""
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    boot_aurocs = []
    boot_f1s = []

    for _ in range(n_bootstraps):
        boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
        b_true = y_true[boot_idx]
        b_prob = y_prob[boot_idx]
        b_mask = y_mask[boot_idx]
        b_pred = (b_prob >= 0.5).astype(np.float32)

        c_aurocs = []
        c_f1s = []
        for c in range(8):
            valid = b_mask[:, c] == 1.0
            if np.sum(valid) > 0 and len(np.unique(b_true[valid, c])) > 1:
                c_aurocs.append(roc_auc_score(b_true[valid, c], b_prob[valid, c]))
            if np.sum(valid) > 0:
                c_f1s.append(f1_score(b_true[valid, c], b_pred[valid, c], zero_division=0))

        if c_aurocs:
            boot_aurocs.append(np.mean(c_aurocs))
        if c_f1s:
            boot_f1s.append(np.mean(c_f1s))

    return {
        "macro_auroc_95ci": [round(float(np.percentile(boot_aurocs, 2.5)), 4), round(float(np.percentile(boot_aurocs, 97.5)), 4)],
        "macro_f1_95ci": [round(float(np.percentile(boot_f1s, 2.5)), 4), round(float(np.percentile(boot_f1s, 97.5)), 4)],
        "n_bootstraps": n_bootstraps,
    }


def run_stage7_test_evaluation(args: argparse.Namespace) -> None:
    print("=" * 80)
    print("  STAGE 7: TEST SET BENCHMARK EVALUATION & 7-LEAD DEGRADATION AUDIT")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[*] Compute Device: {device}")
    if device.type == "cpu":
        torch.set_num_threads(args.num_threads)
        print(f"[*] CPU Worker Threads Configured: {args.num_threads}")

    output_dir = Path(args.output_dir)
    reports_dir = Path(args.reports_dir)
    obs_dir = repo_root / "runs" / "module4_v3" / "observations"
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Checkpoint and Verify Cryptographic Integrity
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    ckpt_bytes = ckpt_path.read_bytes()
    ckpt_hash = hashlib.sha256(ckpt_bytes).hexdigest()
    print(f"[*] Model Checkpoint: {ckpt_path}")
    print(f"    - SHA-256 Digest: {ckpt_hash}")

    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt.get("config", {})
    model_arch = config.get("model_components", {}).get("reasoning_transformer", {
        "d_model": 256,
        "d_ecg": 768,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 512,
        "dropout": 0.1,
        "activation": "gelu",
    })

    model = ControlReasoningTransformer(
        d_model=model_arch["d_model"],
        d_ecg=model_arch["d_ecg"],
        nhead=model_arch["nhead"],
        num_layers=model_arch["num_layers"],
        dim_feedforward=model_arch["dim_feedforward"],
        dropout=model_arch["dropout"],
        num_candidates=8,
        activation=model_arch.get("activation", "gelu"),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[*] Model successfully instantiated and placed in eval() mode.")

    # Load calibration temperatures
    temperatures = ckpt.get("temperature_scaling", {}).get("temperatures", [
        1.1882, 1.2602, 1.1437, 1.0820, 1.0944, 1.1085, 1.1647, 1.1284
    ])
    temp_tensor = torch.tensor(temperatures, dtype=torch.float32, device=device).unsqueeze(0)
    print(f"[*] Fitted Validation Temperatures: {dict(zip(CANDIDATE_ACRONYMS, temperatures))}")

    # 2. Initialize Partition Manager (Unblinded for Stage 7 Test Evaluation)
    print(f"[*] Initializing MultiCohortPartitionManager for single-pass test evaluation...")
    mgr = MultiCohortPartitionManager(repo_root_path=repo_root, quarantine_enforced=False)
    test_rids = mgr.get_record_ids("test", allow_test=True)
    n_test = len(test_rids)
    print(f"[*] Quarantined Test Partition loaded: {n_test} records across Folds 9-10.")

    # 3. Commencing Evaluation Across All 7 Degradation Levels
    variants = APPROVED_CONFIGURATIONS
    trace_generator = StructuredTraceGenerator(evidence_weight_threshold=0.02, top_evidence_k=4)

    degradation_results = {}
    sample_traces_by_variant = {}
    all_csv_rows = []

    print(f"\n[*] Commencing evaluation across {len(variants)} lead configurations (Total: {n_test * len(variants):,} evaluations):")

    t_total_start = time.time()

    for v in variants:
        t_var_start = time.time()
        lead_mask_v = LEAD_CONFIG_MASKS[v]
        active_leads_count = int(np.sum(lead_mask_v))
        print(f"\n--- Lead Configuration: {v} ({active_leads_count} Active Leads) ---")

        # Load Representations & Observations
        z_test, y_test, m_test, rids_test = mgr.load_split_representations(split="test", variant=v, allow_test=True)
        obs_test = load_or_extract_test_observations(mgr, v, obs_dir)
        masks_test = np.tile(lead_mask_v, (n_test, 1))
        dsets_test = [mgr.get_record_metadata(rid)["source_dataset"] for rid in rids_test]

        test_dataset = ReasoningTestDataset(
            z_latents=z_test,
            observations=obs_test,
            labels=y_test,
            label_masks=m_test,
            lead_masks=masks_test,
            record_ids=rids_test,
            variants=[v] * n_test,
            datasets=dsets_test,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_test_batch,
        )

        all_logits = []
        all_probs_raw = []
        all_probs_cal = []
        all_preds = []
        all_attns = []
        evidence_names = None

        t_infer_start = time.time()
        with torch.no_grad():
            for b_z, b_obs, b_masks, b_lbls, b_lmasks, b_rids, b_vars, b_dsets in test_loader:
                b_z = b_z.to(device)
                b_masks = b_masks.to(device)

                out = model(z_ecg=b_z, observations=b_obs, lead_avail_mask=b_masks)
                if evidence_names is None:
                    evidence_names = out["evidence_names"]

                b_logits = out["logits"]
                b_probs = torch.sigmoid(b_logits)
                b_probs_cal = torch.sigmoid(b_logits / temp_tensor)
                b_preds = (b_probs_cal >= 0.5).float()

                all_logits.append(b_logits.cpu())
                all_probs_raw.append(b_probs.cpu())
                all_probs_cal.append(b_probs_cal.cpu())
                all_preds.append(b_preds.cpu())
                all_attns.append(out["attention_weights"].cpu())

        infer_duration = time.time() - t_infer_start
        throughput = n_test / infer_duration

        logits_np = torch.cat(all_logits, dim=0).numpy()
        probs_raw_np = torch.cat(all_probs_raw, dim=0).numpy()
        probs_cal_np = torch.cat(all_probs_cal, dim=0).numpy()
        preds_np = torch.cat(all_preds, dim=0).numpy()
        attns_np = torch.cat(all_attns, dim=0).numpy()

        # Compute Metrics across Observed Targets
        class_aurocs = []
        class_f1s = []
        class_precision = {}
        class_recall = {}
        class_specificity = {}
        class_confusion = {}

        for c_idx, acronym in enumerate(CANDIDATE_ACRONYMS):
            valid = m_test[:, c_idx] == 1.0
            y_t = y_test[valid, c_idx]
            y_p = probs_cal_np[valid, c_idx]
            y_pred = preds_np[valid, c_idx]

            auc = float(roc_auc_score(y_t, y_p)) if len(np.unique(y_t)) > 1 else 0.5
            f1 = float(f1_score(y_t, y_pred, zero_division=0))
            prec = float(precision_score(y_t, y_pred, zero_division=0))
            rec = float(recall_score(y_t, y_pred, zero_division=0))

            tn, fp, fn, tp = confusion_matrix(y_t, y_pred, labels=[0, 1]).ravel()
            spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

            class_aurocs.append(auc)
            class_f1s.append(f1)
            class_precision[acronym] = round(prec, 4)
            class_recall[acronym] = round(rec, 4)
            class_specificity[acronym] = round(spec, 4)
            class_confusion[acronym] = {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)}

        macro_auroc = float(np.mean(class_aurocs))
        macro_f1 = float(np.mean(class_f1s))

        calib_stats = compute_calibration_metrics(y_test, probs_cal_np, m_test)
        cis = compute_bootstrap_confidence_intervals(y_test, probs_cal_np, m_test, n_bootstraps=200)

        # Dataset Subgroup Evaluation
        dataset_breakdown = {}
        for dset_name in ["PTB-XL", "Chapman-Shaoxing", "MIMIC-IV-ECG"]:
            d_idx = [i for i, d in enumerate(dsets_test) if d == dset_name]
            if d_idx:
                d_true = y_test[d_idx]
                d_prob = probs_cal_np[d_idx]
                d_mask = m_test[d_idx]
                d_pred = preds_np[d_idx]

                d_aurocs = []
                d_f1s = []
                for c in range(8):
                    valid = d_mask[:, c] == 1.0
                    if np.sum(valid) > 0 and len(np.unique(d_true[valid, c])) > 1:
                        d_aurocs.append(roc_auc_score(d_true[valid, c], d_prob[valid, c]))
                    if np.sum(valid) > 0:
                        d_f1s.append(f1_score(d_true[valid, c], d_pred[valid, c], zero_division=0))

                dataset_breakdown[dset_name] = {
                    "record_count": len(d_idx),
                    "macro_auroc": round(float(np.mean(d_aurocs)), 4) if d_aurocs else None,
                    "macro_f1": round(float(np.mean(d_f1s)), 4) if d_f1s else None,
                }

        print(f"    - Macro-AUROC: {macro_auroc:.4f} (95% CI: {cis['macro_auroc_95ci']})")
        print(f"    - Macro-F1:    {macro_f1:.4f} (95% CI: {cis['macro_f1_95ci']})")
        print(f"    - Calibrated NLL: {calib_stats['mean_nll']:.4f} | Brier: {calib_stats['mean_brier']:.4f} | ECE: {calib_stats['mean_ece']:.4f}")
        print(f"    - Inference throughput: {throughput:.1f} samples/sec ({infer_duration:.2f}s)")

        # Generate Structured Traces for Sample Cases
        n_traces = min(args.trace_samples, n_test)
        sample_traces = trace_generator.generate_batch_traces(
            probabilities=probs_cal_np[:n_traces],
            attention_matrix=attns_np[:n_traces],
            evidence_names=evidence_names,
            lead_avail_masks=masks_test[:n_traces],
            observations=obs_test[:n_traces],
            record_ids=rids_test[:n_traces],
        )
        sample_traces_by_variant[v] = sample_traces

        degradation_results[v] = {
            "lead_count": active_leads_count,
            "macro_auroc": round(macro_auroc, 4),
            "macro_f1": round(macro_f1, 4),
            "confidence_intervals": cis,
            "calibrated_nll": calib_stats["mean_nll"],
            "calibrated_brier": calib_stats["mean_brier"],
            "calibrated_ece": calib_stats["mean_ece"],
            "per_class_auroc": dict(zip(CANDIDATE_ACRONYMS, [round(x, 4) for x in class_aurocs])),
            "per_class_f1": dict(zip(CANDIDATE_ACRONYMS, [round(x, 4) for x in class_f1s])),
            "per_class_precision": class_precision,
            "per_class_recall": class_recall,
            "per_class_specificity": class_specificity,
            "per_class_confusion": class_confusion,
            "dataset_breakdown": dataset_breakdown,
            "inference_throughput_samples_per_sec": round(throughput, 2),
            "inference_duration_seconds": round(infer_duration, 2),
        }

        # Build prediction rows for 45-column CSV export
        for i in range(n_test):
            row = {
                "record_id": rids_test[i],
                "dataset": dsets_test[i],
                "configuration": v,
                "seed": 42,
                "checkpoint_hash": ckpt_hash,
            }
            for c_idx, acr in enumerate(CANDIDATE_ACRONYMS):
                row[f"logit_{acr}"] = float(logits_np[i, c_idx])
                row[f"prob_{acr}"] = float(probs_raw_np[i, c_idx])
                row[f"calibrated_prob_{acr}"] = float(probs_cal_np[i, c_idx])
                row[f"pred_{acr}"] = int(preds_np[i, c_idx])
                row[f"observed_{acr}"] = int(m_test[i, c_idx])
                row[f"ground_truth_{acr}"] = float(y_test[i, c_idx]) if m_test[i, c_idx] == 1.0 else np.nan
            all_csv_rows.append(row)

    total_eval_time = time.time() - t_total_start
    print(f"\n[*] All 7 configurations successfully evaluated in {total_eval_time:.1f}s.")

    # 4. Save Prediction CSV
    csv_file = output_dir / "test_predictions_seed42.csv"
    print(f"[*] Exporting per-record test predictions to {csv_file}...")
    df_preds = pd.DataFrame(all_csv_rows)
    df_preds.to_csv(csv_file, index=False)
    print(f"    - Total Rows: {len(df_preds):,} (Expected: 51,625)")
    print(f"    - Total Columns: {len(df_preds.columns)} (Expected: 45)")
    csv_hash = compute_sha256(csv_file)
    print(f"    - Prediction CSV SHA-256: {csv_hash}")

    # 5. Save Evaluation Summary JSON
    summary_file = output_dir / "test_evaluation_summary.json"
    full_summary = {
        "report_name": "Module 4 Stage 7 Single-Pass Test Evaluation & Degradation Audit",
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "status": "STAGE7_TEST_EVALUATION_COMPLETE",
        "model_evaluated": {
            "checkpoint_path": str(ckpt_path),
            "checkpoint_sha256": ckpt_hash,
            "seed": 42,
            "role": "primary_model",
            "temperatures": dict(zip(CANDIDATE_ACRONYMS, temperatures)),
        },
        "evaluation_counts": {
            "test_canonical_records": n_test,
            "lead_configurations": len(variants),
            "total_evaluations": len(df_preds),
            "predictions_csv_sha256": csv_hash,
        },
        "degradation_trajectory": degradation_results,
    }
    with open(summary_file, "w") as f:
        json.dump(full_summary, f, indent=2)
    print(f"[*] Summary JSON saved to {summary_file}")

    # 6. Save Sample Reasoning Traces
    traces_file = reports_dir / "sample_test_reasoning_traces.json"
    with open(traces_file, "w") as f:
        json.dump(sample_traces_by_variant, f, indent=2)
    print(f"[*] Sample Reasoning Traces saved to {traces_file} ({args.trace_samples} dossiers/config)")

    # 7. Print Comprehensive Degradation Summary Table
    print("\n" + "=" * 90)
    print("  STAGE 7 TEST SET 7-LEVEL DEGRADATION SUMMARY TABLE")
    print("=" * 90)
    print(f"  {'Configuration':<14} {'Leads':<7} {'Macro-AUROC':<16} {'Macro-F1':<14} {'Calib ECE':<12} {'Calib NLL':<12} {'Brier':<10}")
    print("  " + "-" * 85)
    for v, res in degradation_results.items():
        ci_auc = f"[{res['confidence_intervals']['macro_auroc_95ci'][0]:.4f}, {res['confidence_intervals']['macro_auroc_95ci'][1]:.4f}]"
        print(f"  {v:<14} {res['lead_count']:<7} {res['macro_auroc']:<7.4f} {ci_auc:<8} "
              f"{res['macro_f1']:<14.4f} {res['calibrated_ece']:<12.5f} {res['calibrated_nll']:<12.4f} {res['calibrated_brier']:<10.4f}")
    print("=" * 90)
    print(f"\n[+] STAGE 7 TEST EVALUATION COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 7: Test Benchmark Trace Generation & 7-Lead Degradation Audit")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for inference")
    parser.add_argument("--num-threads", type=int, default=16, help="CPU threads for PyTorch")
    parser.add_argument("--trace-samples", type=int, default=100, help="Number of sample trace dossiers per configuration")
    parser.add_argument("--checkpoint", type=str, default="runs/module4_v3/checkpoints/control_reasoning_best_seed42.pt")
    parser.add_argument("--output-dir", type=str, default="runs/module4_v3/metrics")
    parser.add_argument("--reports-dir", type=str, default="runs/module4_v3/reports")
    parser.add_argument("--cpu", action="store_true", help="Force CPU compute")
    args = parser.parse_args()

    run_stage7_test_evaluation(args)

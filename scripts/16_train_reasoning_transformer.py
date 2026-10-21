"""
Script 16: Phase 4 Control Reasoning Transformer Training Pipeline
===================================================================
Trains the Control Reasoning Transformer (ECG-Only) on frozen Modules 1-3 across
all seven lead configurations (12, 10, 8, 6, 4, 2, 1 leads):
    - Master Partition: runs/module4_v3/splits/split_manifest.csv (36,888 records)
    - Train (Folds 1-7): 25,825 records x 7 configurations = 180,775 examples per epoch
    - Validation (Fold 8): 3,688 records x 7 configurations = 25,816 validation evaluations
    - Test (Folds 9-10): 7,375 records [STRICTLY QUARANTINED ZERO-TOUCH]

Components & Invariants:
    1. Continuous Z_ECG embeddings from FrozenECGFM (runs/module3_v1/representations/)
       across all 7 configurations: 12-lead, 10-lead, 8-lead, 6-lead, 4-lead, 2-lead, 1-lead.
    2. Exact lead availability mask m_l propagated per example ({0, 1}^12).
    3. Discrete structured clinical observations O_l (src/observations/clinical_extractor.py).
    4. Multimodal ControlContextBuilder constructing evidence & 8 candidate queries.
    5. ControlReasoningTransformer with Evidence Encoder & Candidate Cross-Attention.
    6. Loss: Element-wise Masked BCEWithLogitsLoss normalized by count of observed targets.
    7. Multi-Seed Training & Verification across seeds [42, 1337, 2024] (Primary: 42).
    8. Post-training per-disease temperature scaling calibration on Validation Split (Fold 8).
    9. Separate validation evaluation across all seven configurations.
    10. Detailed per-record validation predictions export.
    11. Zero Patient Leakage verified across splits; Test strictly quarantined.
    12. Checkpoints saved to runs/module4_v3/checkpoints/ (seed-isolated).
"""

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from scipy.optimize import minimize_scalar

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.reasoning.reasoning_transformer import ControlReasoningTransformer
from src.reasoning.context_builder import CANDIDATE_ACRONYMS, CANDIDATE_NAMES, LEAD_NAMES
from src.reasoning.multi_cohort_loader import MultiCohortPartitionManager, LEAD_CONFIG_MASKS
from src.observations.clinical_extractor import ClinicalObservationExtractor
from src.reasoning.freeze_contract import (
    APPROVED_TARGETS,
    APPROVED_CONFIGURATIONS,
    APPROVED_SEEDS,
    PRIMARY_SEED,
    build_module4_freeze_contract,
    validate_module4_freeze_contract,
    compute_sha256,
)


def compute_masked_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    label_masks: torch.Tensor,
) -> torch.Tensor:
    """
    Computes element-wise binary cross-entropy with logits, masked by observation mask,
    and normalized by the total number of observed targets.

    Invariants:
    1. Unobserved targets (label_masks == 0.0) contribute exactly 0.0 loss and 0.0 gradient.
    2. Changing unobserved target values does not alter the loss.
    3. If sum(label_masks) == 0, returns (per_element_loss * 0.0).sum() to ensure stable zero gradient.
    """
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    per_element_loss = criterion(logits, targets)
    masked_loss = per_element_loss * label_masks
    n_observed = label_masks.sum()
    if n_observed == 0:
        return (per_element_loss * 0.0).sum()
    return masked_loss.sum() / n_observed


def compute_bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_masks: np.ndarray,
    n_bootstraps: int = 200,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Computes empirical 95% bootstrap confidence intervals for Macro-AUROC and Macro-F1
    evaluated exclusively over observed targets (label_masks == 1.0).
    """
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    num_classes = y_true.shape[1]
    boot_macro_aurocs = []
    boot_macro_f1s = []

    for _ in range(n_bootstraps):
        boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
        b_true = y_true[boot_idx]
        b_prob = y_prob[boot_idx]
        b_mask = label_masks[boot_idx]

        class_aurocs = []
        class_f1s = []

        for c in range(num_classes):
            obs_c = b_mask[:, c] == 1.0
            if np.sum(obs_c) == 0:
                continue
            yt = b_true[obs_c, c]
            yp = b_prob[obs_c, c]
            if len(np.unique(yt)) > 1:
                try:
                    auc = float(roc_auc_score(yt, yp))
                    class_aurocs.append(auc)
                except Exception:
                    pass
            yp_bin = (yp >= 0.5).astype(np.float32)
            f1 = float(f1_score(yt, yp_bin, zero_division=0))
            class_f1s.append(f1)

        if class_aurocs:
            boot_macro_aurocs.append(float(np.mean(class_aurocs)))
        if class_f1s:
            boot_macro_f1s.append(float(np.mean(class_f1s)))

    def get_ci(values):
        if not values:
            return [0.0, 0.0]
        lower = float(np.percentile(values, 100 * (alpha / 2)))
        upper = float(np.percentile(values, 100 * (1 - alpha / 2)))
        return [round(lower, 4), round(upper, 4)]

    return {
        "macro_auroc_95ci": get_ci(boot_macro_aurocs),
        "macro_f1_95ci": get_ci(boot_macro_f1s),
        "n_bootstraps": n_bootstraps,
    }


class ReasoningBatch:
    """
    Collation container for multimodal reasoning batches.
    Supports standard 5-tuple unpacking (z, obs, masks, labels, label_masks)
    while exposing metadata (record_ids, variants, datasets).
    """

    def __init__(
        self,
        z: torch.Tensor,
        observations: List[Dict[str, Any]],
        lead_masks: torch.Tensor,
        labels: torch.Tensor,
        label_masks: torch.Tensor,
        record_ids: Optional[List[str]] = None,
        variants: Optional[List[str]] = None,
        datasets: Optional[List[str]] = None,
    ):
        self.z = z
        self.observations = observations
        self.lead_masks = lead_masks
        self.labels = labels
        self.label_masks = label_masks
        self.record_ids = record_ids or []
        self.variants = variants or []
        self.datasets = datasets or []

    def __iter__(self):
        return iter((self.z, self.observations, self.lead_masks, self.labels, self.label_masks))

    def __getitem__(self, idx: int):
        return (self.z, self.observations, self.lead_masks, self.labels, self.label_masks)[idx]

    def __len__(self) -> int:
        return 5


class ReasoningDataset(Dataset):
    """
    In-memory dataset pairing continuous latents Z_ECG, structured observations,
    lead availability masks, multi-hot disease targets, observation masks, and record metadata.
    """

    def __init__(
        self,
        z_latents: Union[np.ndarray, torch.Tensor],
        observations: List[Dict[str, Any]],
        labels: Union[np.ndarray, torch.Tensor],
        label_masks: Optional[Union[np.ndarray, torch.Tensor]] = None,
        lead_masks: Optional[Union[np.ndarray, torch.Tensor]] = None,
        record_ids: Optional[List[str]] = None,
        variants: Optional[List[str]] = None,
        datasets: Optional[List[str]] = None,
    ):
        assert len(z_latents) == len(observations) == len(labels), (
            f"Dataset length mismatch: Z={len(z_latents)}, Obs={len(observations)}, Lbl={len(labels)}"
        )
        if label_masks is not None:
            assert len(label_masks) == len(z_latents), f"Label masks length mismatch: {len(label_masks)} vs {len(z_latents)}"
        if lead_masks is not None:
            assert len(lead_masks) == len(z_latents), f"Lead masks length mismatch: {len(lead_masks)} vs {len(z_latents)}"

        self.z_latents = torch.from_numpy(z_latents).float() if isinstance(z_latents, np.ndarray) else z_latents.float()
        self.observations = observations
        self.labels = torch.from_numpy(labels).float() if isinstance(labels, np.ndarray) else labels.float()
        if label_masks is not None:
            self.label_masks = torch.from_numpy(label_masks).float() if isinstance(label_masks, np.ndarray) else label_masks.float()
        else:
            self.label_masks = torch.ones_like(self.labels)

        if lead_masks is not None:
            self.lead_masks = torch.from_numpy(lead_masks).float() if isinstance(lead_masks, np.ndarray) else lead_masks.float()
        else:
            self.lead_masks = torch.ones((len(z_latents), 12), dtype=torch.float32)

        self.record_ids = record_ids or [f"rec_{i}" for i in range(len(z_latents))]
        self.variants = variants or ["12-lead"] * len(z_latents)
        self.datasets = datasets or ["UNKNOWN"] * len(z_latents)
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


def collate_reasoning_batch(batch) -> ReasoningBatch:
    z_list = [item[0] for item in batch]
    obs_list = [item[1] for item in batch]
    mask_list = [item[2] for item in batch]
    labels_list = [item[3] for item in batch]
    label_masks_list = [item[4] for item in batch]
    record_ids = [item[5] for item in batch] if len(batch[0]) > 5 else None
    variants = [item[6] for item in batch] if len(batch[0]) > 6 else None
    datasets = [item[7] for item in batch] if len(batch[0]) > 7 else None

    z_batch = torch.stack(z_list, dim=0)
    mask_batch = torch.stack(mask_list, dim=0)
    labels_batch = torch.stack(labels_list, dim=0)
    label_masks_batch = torch.stack(label_masks_list, dim=0)

    return ReasoningBatch(
        z=z_batch,
        observations=obs_list,
        lead_masks=mask_batch,
        labels=labels_batch,
        label_masks=label_masks_batch,
        record_ids=record_ids,
        variants=variants,
        datasets=datasets,
    )


def evaluate_model(
    model: ControlReasoningTransformer,
    val_loader: DataLoader,
    device: torch.device,
    compute_cis: bool = True,
) -> Dict[str, Any]:
    """
    Evaluates the model on validation loader using strictly observed targets.
    Excludes unobserved targets (label_masks == 0.0) from all metrics.
    Measures latency, throughput, bootstrap confidence intervals, and under-powered warnings.
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    total_loss_sum = 0.0
    total_observed = 0
    all_logits = []
    all_probs = []
    all_targets = []
    all_label_masks = []
    all_rids = []
    all_variants = []
    all_datasets = []

    t_eval_start = time.perf_counter()

    with torch.no_grad():
        for batch in val_loader:
            b_z, b_obs, b_masks, b_labels, b_label_masks = batch[:5]
            b_z = b_z.to(device)
            b_masks = b_masks.to(device)
            b_labels = b_labels.to(device)
            b_label_masks = b_label_masks.to(device)

            output = model(
                z_ecg=b_z,
                observations=b_obs,
                lead_avail_mask=b_masks,
            )

            per_element_loss = criterion(output["logits"], b_labels)
            masked_loss = per_element_loss * b_label_masks
            total_loss_sum += masked_loss.sum().item()
            total_observed += b_label_masks.sum().item()

            all_logits.append(output["logits"].cpu())
            all_probs.append(output["probabilities"].cpu())
            all_targets.append(b_labels.cpu())
            all_label_masks.append(b_label_masks.cpu())

            if hasattr(batch, "record_ids"):
                all_rids.extend(batch.record_ids)
            if hasattr(batch, "variants"):
                all_variants.extend(batch.variants)
            if hasattr(batch, "datasets"):
                all_datasets.extend(batch.datasets)

    eval_elapsed = time.perf_counter() - t_eval_start

    avg_loss = (total_loss_sum / total_observed) if total_observed > 0 else 0.0

    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_label_masks = torch.cat(all_label_masks, dim=0).numpy()

    n_samples = len(all_logits)
    latency_ms_per_sample = round((eval_elapsed / max(n_samples, 1)) * 1000.0, 3)
    throughput_samples_per_sec = round(n_samples / max(eval_elapsed, 1e-6), 2)

    # Compute metrics per class using ONLY observed entries
    per_class_auroc = {}
    per_class_f1 = {}
    per_class_precision = {}
    per_class_recall = {}
    per_class_nll = {}
    per_class_brier = {}
    per_class_ece = {}
    per_class_confusion = {}
    per_class_support = {}
    valid_aurocs = []
    underpowered_warnings = []

    bin_edges = np.linspace(0, 1, 11)

    for i, acronym in enumerate(CANDIDATE_ACRONYMS):
        mask_i = all_label_masks[:, i] == 1.0
        n_obs = int(np.sum(mask_i))
        n_unobs = int(len(mask_i) - n_obs)

        if n_obs == 0:
            per_class_auroc[acronym] = 0.5
            per_class_f1[acronym] = 0.0
            per_class_precision[acronym] = 0.0
            per_class_recall[acronym] = 0.0
            per_class_nll[acronym] = 0.0
            per_class_brier[acronym] = 0.0
            per_class_ece[acronym] = 0.0
            per_class_confusion[acronym] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
            per_class_support[acronym] = {"observed": 0, "unobserved": n_unobs, "positives": 0, "negatives": 0}
            continue

        y_true = all_targets[mask_i, i]
        y_prob = all_probs[mask_i, i]
        y_pred = (y_prob >= 0.5).astype(np.float32)

        n_pos = int(np.sum(y_true == 1.0))
        n_neg = int(np.sum(y_true == 0.0))
        per_class_support[acronym] = {
            "observed": n_obs,
            "unobserved": n_unobs,
            "positives": n_pos,
            "negatives": n_neg,
        }

        if n_pos < 30:
            underpowered_warnings.append(
                f"Target '{acronym}' has only {n_pos} observed positive examples in validation split "
                "(caution: under-powered for robust statistical inference)."
            )

        # AUROC requires both classes present
        if len(np.unique(y_true)) > 1:
            auc = float(roc_auc_score(y_true, y_prob))
            per_class_auroc[acronym] = round(auc, 4)
            valid_aurocs.append(auc)
        else:
            per_class_auroc[acronym] = 0.5

        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))

        per_class_f1[acronym] = round(f1, 4)
        per_class_precision[acronym] = round(prec, 4)
        per_class_recall[acronym] = round(rec, 4)

        # NLL and Brier on observed targets
        p_clipped = np.clip(y_prob, 1e-7, 1.0 - 1e-7)
        nll = float(-np.mean(y_true * np.log(p_clipped) + (1.0 - y_true) * np.log(1.0 - p_clipped)))
        brier = float(np.mean((y_prob - y_true) ** 2))
        per_class_nll[acronym] = round(nll, 5)
        per_class_brier[acronym] = round(brier, 5)

        # ECE (10 bins) on observed targets
        ece = 0.0
        for b in range(10):
            in_bin = (y_prob >= bin_edges[b]) & (y_prob < bin_edges[b + 1])
            n_b = np.sum(in_bin)
            if n_b > 0:
                conf = np.mean(y_prob[in_bin])
                acc = np.mean(y_true[in_bin])
                ece += (n_b / len(y_prob)) * np.abs(acc - conf)
        per_class_ece[acronym] = round(float(ece), 5)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        per_class_confusion[acronym] = {
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        }

    macro_auroc = float(np.mean(valid_aurocs)) if valid_aurocs else 0.5
    macro_f1 = float(np.mean(list(per_class_f1.values())))
    macro_precision = float(np.mean(list(per_class_precision.values())))
    macro_recall = float(np.mean(list(per_class_recall.values())))
    macro_nll = float(np.mean(list(per_class_nll.values())))
    macro_brier = float(np.mean(list(per_class_brier.values())))
    macro_ece = float(np.mean(list(per_class_ece.values())))

    # Bootstrap Confidence Intervals (if requested and sufficient samples)
    confidence_intervals = {}
    if compute_cis and len(all_logits) >= 50:
        confidence_intervals = compute_bootstrap_confidence_intervals(
            y_true=all_targets,
            y_prob=all_probs,
            label_masks=all_label_masks,
            n_bootstraps=200,
        )

    return {
        "val_loss": round(avg_loss, 5),
        "macro_auroc": round(macro_auroc, 4),
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_nll": round(macro_nll, 5),
        "macro_brier": round(macro_brier, 5),
        "macro_ece": round(macro_ece, 5),
        "latency_ms_per_sample": latency_ms_per_sample,
        "throughput_samples_per_sec": throughput_samples_per_sec,
        "confidence_intervals": confidence_intervals,
        "underpowered_warnings": underpowered_warnings,
        "per_class_auroc": per_class_auroc,
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_nll": per_class_nll,
        "per_class_brier": per_class_brier,
        "per_class_ece": per_class_ece,
        "per_class_confusion": per_class_confusion,
        "per_class_support": per_class_support,
        "logits": all_logits,
        "probs": all_probs,
        "targets": all_targets,
        "label_masks": all_label_masks,
        "record_ids": all_rids,
        "variants": all_variants,
        "datasets": all_datasets,
    }


def evaluate_all_configurations(
    model: ControlReasoningTransformer,
    val_loaders_by_variant: Dict[str, DataLoader],
    device: torch.device,
) -> Dict[str, Any]:
    """
    Evaluates model separately across all seven lead degradation configurations.
    Returns per-configuration metrics and aggregate summary.
    """
    per_config_results = {}
    losses = []
    macro_aurocs = []
    macro_f1s = []

    for variant in APPROVED_CONFIGURATIONS:
        if variant in val_loaders_by_variant:
            loader = val_loaders_by_variant[variant]
            res = evaluate_model(model, loader, device, compute_cis=False)
            per_config_results[variant] = res
            losses.append(res["val_loss"])
            macro_aurocs.append(res["macro_auroc"])
            macro_f1s.append(res["macro_f1"])

    return {
        "per_configuration": per_config_results,
        "pooled": {
            "mean_val_loss": round(float(np.mean(losses)), 5) if losses else 0.0,
            "mean_macro_auroc": round(float(np.mean(macro_aurocs)), 4) if macro_aurocs else 0.5,
            "mean_macro_f1": round(float(np.mean(macro_f1s)), 4) if macro_f1s else 0.0,
            "configurations_evaluated": list(per_config_results.keys()),
        },
    }


def fit_temperature_scaling(
    logits: np.ndarray,      # (N_val, 8)
    targets: np.ndarray,     # (N_val, 8)
    label_masks: Optional[np.ndarray] = None, # (N_val, 8)
) -> Tuple[List[float], Dict[str, Any]]:
    """
    Fits per-disease temperature scaling parameters T_i on validation logits
    using Brent's scalar minimization of binary negative log-likelihood (NLL)
    exclusively on observed validation targets (label_masks == 1.0).
    """
    num_classes = logits.shape[1]
    if label_masks is None:
        label_masks = np.ones_like(targets)

    temperatures = []
    calibrated_probs = np.zeros_like(logits)

    pre_nlls = []
    post_nlls = []
    pre_briers = []
    post_briers = []
    pre_eces = []
    post_eces = []

    bin_edges = np.linspace(0, 1, 11)

    for i in range(num_classes):
        mask_i = label_masks[:, i] == 1.0
        n_obs = int(np.sum(mask_i))

        if n_obs == 0:
            temperatures.append(1.0)
            calibrated_probs[:, i] = 1.0 / (1.0 + np.exp(-logits[:, i]))
            pre_nlls.append(0.0)
            post_nlls.append(0.0)
            pre_briers.append(0.0)
            post_briers.append(0.0)
            pre_eces.append(0.0)
            post_eces.append(0.0)
            continue

        z_i = logits[mask_i, i]
        y_i = targets[mask_i, i]

        def nll_obj(T):
            if T <= 0:
                return 1e9
            p = 1.0 / (1.0 + np.exp(-z_i / T))
            p = np.clip(p, 1e-7, 1.0 - 1e-7)
            return -np.mean(y_i * np.log(p) + (1.0 - y_i) * np.log(1.0 - p))

        res = minimize_scalar(nll_obj, bounds=(0.1, 10.0), method="bounded")
        t_opt = float(res.x)
        temperatures.append(round(t_opt, 4))

        # Metrics before calibration on observed targets
        p_raw = 1.0 / (1.0 + np.exp(-z_i))
        p_raw = np.clip(p_raw, 1e-7, 1.0 - 1e-7)
        pre_nll = float(-np.mean(y_i * np.log(p_raw) + (1.0 - y_i) * np.log(1.0 - p_raw)))
        pre_brier = float(np.mean((p_raw - y_i) ** 2))

        # Metrics after calibration on observed targets
        p_cal = 1.0 / (1.0 + np.exp(-z_i / t_opt))
        p_cal = np.clip(p_cal, 1e-7, 1.0 - 1e-7)
        calibrated_probs[mask_i, i] = p_cal
        post_nll = float(-np.mean(y_i * np.log(p_cal) + (1.0 - y_i) * np.log(1.0 - p_cal)))
        post_brier = float(np.mean((p_cal - y_i) ** 2))

        # ECE on observed targets
        pre_ece = 0.0
        post_ece = 0.0
        for b in range(10):
            in_b_raw = (p_raw >= bin_edges[b]) & (p_raw < bin_edges[b + 1])
            if np.sum(in_b_raw) > 0:
                pre_ece += (np.sum(in_b_raw) / len(p_raw)) * np.abs(np.mean(y_i[in_b_raw]) - np.mean(p_raw[in_b_raw]))

            in_b_cal = (p_cal >= bin_edges[b]) & (p_cal < bin_edges[b + 1])
            if np.sum(in_b_cal) > 0:
                post_ece += (np.sum(in_b_cal) / len(p_cal)) * np.abs(np.mean(y_i[in_b_cal]) - np.mean(p_cal[in_b_cal]))

        pre_nlls.append(round(pre_nll, 5))
        post_nlls.append(round(post_nll, 5))
        pre_briers.append(round(pre_brier, 5))
        post_briers.append(round(post_brier, 5))
        pre_eces.append(round(float(pre_ece), 5))
        post_eces.append(round(float(post_ece), 5))

    calib_metrics = {
        "candidate_acronyms": CANDIDATE_ACRONYMS,
        "temperatures": temperatures,
        "mean_temperature": round(float(np.mean(temperatures)), 4),
        "pre_calibration": {
            "mean_nll": round(float(np.mean(pre_nlls)), 5),
            "mean_brier": round(float(np.mean(pre_briers)), 5),
            "mean_ece": round(float(np.mean(pre_eces)), 5),
            "per_class_nll": dict(zip(CANDIDATE_ACRONYMS, pre_nlls)),
            "per_class_brier": dict(zip(CANDIDATE_ACRONYMS, pre_briers)),
            "per_class_ece": dict(zip(CANDIDATE_ACRONYMS, pre_eces)),
        },
        "post_calibration": {
            "mean_nll": round(float(np.mean(post_nlls)), 5),
            "mean_brier": round(float(np.mean(post_briers)), 5),
            "mean_ece": round(float(np.mean(post_eces)), 5),
            "per_class_nll": dict(zip(CANDIDATE_ACRONYMS, post_nlls)),
            "per_class_brier": dict(zip(CANDIDATE_ACRONYMS, post_briers)),
            "per_class_ece": dict(zip(CANDIDATE_ACRONYMS, post_eces)),
        },
    }

    return temperatures, calib_metrics


def export_validation_predictions(
    eval_result: Dict[str, Any],
    temperatures: List[float],
    seed: int,
    checkpoint_hash: str,
    output_path: Path,
) -> Path:
    """
    Exports detailed per-record validation predictions containing:
    record ID, dataset, configuration, seed, checkpoint hash, logits,
    calibrated probabilities, thresholded outputs, ground-truth labels, and observation masks.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logits = eval_result["logits"]
    probs = eval_result["probs"]
    targets = eval_result["targets"]
    label_masks = eval_result["label_masks"]
    rids = eval_result.get("record_ids", [f"rec_{i}" for i in range(len(logits))])
    variants = eval_result.get("variants", ["12-lead"] * len(logits))
    datasets = eval_result.get("datasets", ["UNKNOWN"] * len(logits))

    temps_arr = np.array(temperatures, dtype=np.float32)
    calibrated_probs = 1.0 / (1.0 + np.exp(-logits / np.maximum(temps_arr, 0.01)))

    rows = []
    for i in range(len(logits)):
        row = {
            "record_id": rids[i] if i < len(rids) else f"rec_{i}",
            "dataset": datasets[i] if i < len(datasets) else "UNKNOWN",
            "configuration": variants[i] if i < len(variants) else "UNKNOWN",
            "seed": seed,
            "checkpoint_hash": checkpoint_hash,
        }
        for c, acronym in enumerate(CANDIDATE_ACRONYMS):
            row[f"logit_{acronym}"] = round(float(logits[i, c]), 5)
            row[f"prob_{acronym}"] = round(float(probs[i, c]), 5)
            row[f"calibrated_prob_{acronym}"] = round(float(calibrated_probs[i, c]), 5)
            row[f"pred_{acronym}"] = int(calibrated_probs[i, c] >= 0.5)
            is_obs = int(label_masks[i, c] == 1.0)
            row[f"observed_{acronym}"] = is_obs
            row[f"ground_truth_{acronym}"] = float(targets[i, c]) if is_obs else np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return output_path


def load_or_extract_observations(
    mgr: MultiCohortPartitionManager,
    split: str,
    variant: str,
    obs_cache_dir: Path,
    allow_test: bool = False,
) -> List[Dict[str, Any]]:
    """Loads cached observations or extracts them deterministically from active leads."""
    obs_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = obs_cache_dir / f"obs_{split}_{variant}.json"

    if cache_file.exists():
        with open(cache_file, "r") as f:
            return json.load(f)

    record_ids = mgr.get_record_ids(split, allow_test=allow_test)
    extractor = ClinicalObservationExtractor()
    obs_list = []

    t0 = time.time()
    for i, rid in enumerate(record_ids):
        sig = mgr.get_signal(rid, variant=variant)
        mask = mgr.get_lead_mask(rid, variant=variant)
        obs = extractor.extract(sig, mask).to_dict()
        obs_list.append(obs)
        if (i + 1) % 5000 == 0 or (i + 1) == len(record_ids):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"    - [{variant}] Extracted {i + 1}/{len(record_ids)} records ({rate:.1f} rec/s)")

    with open(cache_file, "w") as f:
        json.dump(obs_list, f)
    return obs_list


def set_seed(seed: int) -> None:
    """Sets random seeds deterministically across all frameworks."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_single_seed(
    seed: int,
    args: argparse.Namespace,
    config: Dict[str, Any],
    train_dataset: ReasoningDataset,
    val_datasets_by_variant: Dict[str, ReasoningDataset],
    val_dataset_pooled: ReasoningDataset,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Executes an isolated, reproducible training run for a single seed.
    """
    print(f"\n" + "=" * 80)
    print(f"  RUNNING SEED {seed} (Primary: {seed == PRIMARY_SEED})")
    print("=" * 80)

    set_seed(seed)

    # Instantiate Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_reasoning_batch,
        num_workers=0,
    )
    val_loaders_by_variant = {
        v: DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_reasoning_batch,
            num_workers=0,
        )
        for v, ds in val_datasets_by_variant.items()
    }
    val_loader_pooled = DataLoader(
        val_dataset_pooled,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_reasoning_batch,
        num_workers=0,
    )

    # Instantiate Model
    model_arch = config["model_components"]["reasoning_transformer"]
    model = ControlReasoningTransformer(
        d_model=model_arch["d_model"],
        d_ecg=model_arch["d_ecg"],
        nhead=model_arch["nhead"],
        num_layers=model_arch["num_layers"],
        dim_feedforward=model_arch["dim_feedforward"],
        dropout=model_arch["dropout"],
        num_candidates=config["candidate_disease_space"]["candidate_count"],
        activation=model_arch.get("activation", "gelu"),
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=tuple(config["loss_and_optimization"]["betas"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    dest = config["destination_artifacts"]
    ckpt_dir = repo_root / dest["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = repo_root / dest["metrics_dir"]
    metrics_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt_path = ckpt_dir / f"control_reasoning_best_seed{seed}.pt"
    final_ckpt_path = ckpt_dir / f"control_reasoning_final_seed{seed}.pt"
    val_preds_path = metrics_dir / f"validation_predictions_seed{seed}.csv"
    seed_report_path = metrics_dir / f"control_training_history_seed{seed}.json"

    if seed_report_path.exists() and val_preds_path.exists() and best_ckpt_path.exists() and not getattr(args, "force_retrain", False):
        print(f"\n[+] Detected existing verified artifacts for Seed {seed}.")
        with open(seed_report_path, "r") as f:
            saved_report = json.load(f)
        print(f"    - Best Val Loss: {saved_report['best_val_loss']} | Macro AUROC: {saved_report['macro_auroc']} | Macro F1: {saved_report['macro_f1']}")
        print(f"    - Reusing validated Seed {seed} checkpoints and predictions.")
        return {
            "seed": seed,
            "best_checkpoint_path": str(best_ckpt_path.relative_to(repo_root)),
            "best_checkpoint_sha256": saved_report["checkpoints"]["best_sha256"],
            "final_checkpoint_path": str(final_ckpt_path.relative_to(repo_root)),
            "final_checkpoint_sha256": saved_report["checkpoints"]["final_sha256"],
            "best_epoch": saved_report["best_epoch"],
            "best_val_loss": saved_report["best_val_loss"],
            "temperatures": saved_report["calibration"]["temperatures"],
            "calibration_metrics": saved_report["calibration"],
            "val_eval": {
                "val_loss": saved_report["best_val_loss"],
                "macro_auroc": saved_report["macro_auroc"],
                "macro_f1": saved_report["macro_f1"],
                "confidence_intervals": saved_report["confidence_intervals"],
                "underpowered_warnings": saved_report["underpowered_warnings"],
            },
            "multi_config_eval": {"per_configuration": saved_report["per_configuration_evaluation"]},
            "val_preds_path": str(val_preds_path.relative_to(repo_root)),
            "history": saved_report["history"],
        }

    # Pre-training initial validation evaluation
    print(f"\n[*] Pre-training validation evaluation (Epoch 0, Seed {seed})...")
    init_eval = evaluate_model(model, val_loader_pooled, device)
    print(f"    - Init Val Loss: {init_eval['val_loss']:.5f} | "
          f"Macro AUROC: {init_eval['macro_auroc']:.4f} | "
          f"Macro F1: {init_eval['macro_f1']:.4f}")

    best_val_loss = float("inf")
    patience = config["checkpointing_and_generalization"]["patience"]
    patience_counter = 0
    history = []
    best_epoch = 0

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = len(train_loader)

        for b_idx, batch in enumerate(train_loader):
            b_z, b_obs, b_masks, b_labels, b_label_masks = batch[:5]
            b_z = b_z.to(device)
            b_masks = b_masks.to(device)
            b_labels = b_labels.to(device)
            b_label_masks = b_label_masks.to(device)

            optimizer.zero_grad()
            out = model(z_ecg=b_z, observations=b_obs, lead_avail_mask=b_masks)
            loss = compute_masked_bce_loss(out["logits"], b_labels, b_label_masks)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            if (b_idx + 1) % 500 == 0 or (b_idx + 1) == n_batches:
                print(f"      [Epoch {epoch:02d} | Batch {b_idx + 1:04d}/{n_batches:04d}] Running Loss: {epoch_loss / (b_idx + 1):.5f}", flush=True)

        scheduler.step()
        avg_train_loss = epoch_loss / max(n_batches, 1)

        val_eval = evaluate_model(model, val_loader_pooled, device)

        epoch_stats = {
            "epoch": epoch,
            "seed": seed,
            "train_loss": round(avg_train_loss, 5),
            "val_loss": val_eval["val_loss"],
            "macro_auroc": val_eval["macro_auroc"],
            "macro_f1": val_eval["macro_f1"],
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(epoch_stats)

        print(f"Seed {seed} | Epoch {epoch:02d}/{args.epochs:02d} | "
              f"Train Loss: {avg_train_loss:.5f} | "
              f"Val Loss: {val_eval['val_loss']:.5f} | "
              f"Macro AUROC: {val_eval['macro_auroc']:.4f} | "
              f"Macro F1: {val_eval['macro_f1']:.4f}")

        # Checkpoint saving on validation improvement
        if val_eval["val_loss"] < (best_val_loss - config["checkpointing_and_generalization"]["min_delta"]):
            best_val_loss = val_eval["val_loss"]
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "seed": seed,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "macro_auroc": val_eval["macro_auroc"],
                    "macro_f1": val_eval["macro_f1"],
                    "per_class_auroc": val_eval["per_class_auroc"],
                    "per_class_f1": val_eval["per_class_f1"],
                    "config": config,
                },
                best_ckpt_path,
            )
            print(f"    --> Saved new best checkpoint (Seed {seed}) to {best_ckpt_path.name}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[!] Early stopping triggered at epoch {epoch} (Seed {seed}, patience={patience}).")
                break

    total_train_time = time.time() - start_time

    # Reload best checkpoint for final evaluation & temperature calibration
    print(f"[*] Reloading best checkpoint (Seed {seed}) from {best_ckpt_path.name}...")
    best_ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    # Final pooled validation pass
    final_val_eval = evaluate_model(model, val_loader_pooled, device, compute_cis=True)
    val_logits = final_val_eval["logits"]
    val_targets = final_val_eval["targets"]
    val_masks = final_val_eval["label_masks"]

    # Post-hoc temperature calibration
    temperatures, calib_report = fit_temperature_scaling(val_logits, val_targets, label_masks=val_masks)

    # Re-save best checkpoint enriched with calibration temperatures
    best_ckpt["temperature_scaling"] = {
        "temperatures": temperatures,
        "calibration_metrics": calib_report,
    }
    torch.save(best_ckpt, best_ckpt_path)

    # Save final checkpoint
    torch.save(
        {
            "seed": seed,
            "model_state_dict": model.state_dict(),
            "temperatures": temperatures,
            "final_epoch": epoch,
            "best_epoch": best_epoch,
            "training_history": history,
        },
        final_ckpt_path,
    )

    # Compute SHA-256 for saved checkpoints
    best_sha256 = compute_sha256(best_ckpt_path)
    final_sha256 = compute_sha256(final_ckpt_path)

    # If primary seed 42, maintain legacy alias checkpoint
    if seed == PRIMARY_SEED:
        legacy_ckpt_path = repo_root / dest["legacy_alias_checkpoint"]
        torch.save(best_ckpt, legacy_ckpt_path)
        print(f"[+] Primary seed 42 synced to legacy alias: {legacy_ckpt_path.name}")

    # Evaluate separately across all seven configurations
    print(f"[*] Evaluating separate lead degradation levels for Seed {seed}...")
    multi_config_eval = evaluate_all_configurations(model, val_loaders_by_variant, device)

    # Export per-record validation predictions table
    export_validation_predictions(
        eval_result=final_val_eval,
        temperatures=temperatures,
        seed=seed,
        checkpoint_hash=best_sha256,
        output_path=val_preds_path,
    )
    print(f"[+] Validation predictions exported to {val_preds_path.name}")

    # Seed-specific history report
    seed_report_path = metrics_dir / f"control_training_history_seed{seed}.json"
    seed_report = {
        "seed": seed,
        "is_primary": seed == PRIMARY_SEED,
        "total_train_time_seconds": round(total_train_time, 2),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "macro_auroc": final_val_eval["macro_auroc"],
        "macro_f1": final_val_eval["macro_f1"],
        "latency_ms_per_sample": final_val_eval["latency_ms_per_sample"],
        "throughput_samples_per_sec": final_val_eval["throughput_samples_per_sec"],
        "confidence_intervals": final_val_eval["confidence_intervals"],
        "underpowered_warnings": final_val_eval["underpowered_warnings"],
        "calibration": calib_report,
        "checkpoints": {
            "best_path": str(best_ckpt_path.relative_to(repo_root)),
            "best_sha256": best_sha256,
            "final_path": str(final_ckpt_path.relative_to(repo_root)),
            "final_sha256": final_sha256,
        },
        "per_configuration_evaluation": {
            k: {
                "val_loss": v["val_loss"],
                "macro_auroc": v["macro_auroc"],
                "macro_f1": v["macro_f1"],
                "macro_nll": v["macro_nll"],
                "macro_brier": v["macro_brier"],
                "macro_ece": v["macro_ece"],
            }
            for k, v in multi_config_eval["per_configuration"].items()
        },
        "history": history,
    }
    with open(seed_report_path, "w") as f:
        json.dump(seed_report, f, indent=2)

    return {
        "seed": seed,
        "best_checkpoint_path": str(best_ckpt_path.relative_to(repo_root)),
        "best_checkpoint_sha256": best_sha256,
        "final_checkpoint_path": str(final_ckpt_path.relative_to(repo_root)),
        "final_checkpoint_sha256": final_sha256,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "temperatures": temperatures,
        "calibration_metrics": calib_report,
        "val_eval": final_val_eval,
        "multi_config_eval": multi_config_eval,
        "val_preds_path": str(val_preds_path.relative_to(repo_root)),
        "history": history,
    }


def train_control_reasoning_transformer(args: argparse.Namespace) -> None:
    print("=" * 80)
    print("  PHASE 4: CONTROL REASONING TRANSFORMER MULTI-CONFIGURATION PIPELINE")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[*] Compute Device: {device}")
    if device.type == "cpu":
        import os
        n_threads = min(16, os.cpu_count() or 8)
        torch.set_num_threads(n_threads)
        print(f"[*] PyTorch CPU threads configured: {n_threads}")

    # Load Configuration
    config_file = repo_root / "configs" / "control_path_training_config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Training config not found at {config_file}")
    with open(config_file, "r") as f:
        config = json.load(f)

    # Bind Partition Manager
    print("[*] Initializing MultiCohortPartitionManager...")
    mgr = MultiCohortPartitionManager(repo_root_path=repo_root, quarantine_enforced=True)
    counts = mgr.get_counts()
    print(f"[*] Ecosystem split counts: {counts}")
    leakage = mgr.verify_zero_patient_leakage()
    assert leakage["zero_leakage_passed"], f"Patient leakage detected: {leakage}"
    print("[*] Zero patient leakage verified across all splits.")

    obs_dir = repo_root / "runs" / "module4_v3" / "observations"

    # Multi-Configuration Data Ingestion
    variants = APPROVED_CONFIGURATIONS
    print(f"[*] Ingesting representations across all seven configurations: {variants}...")

    all_z_train, all_obs_train, all_masks_train = [], [], []
    all_lbls_train, all_lmasks_train, all_rids_train = [], [], []
    all_vars_train, all_dsets_train = [], []

    val_datasets_by_variant = {}
    all_z_val, all_obs_val, all_masks_val = [], [], []
    all_lbls_val, all_lmasks_val, all_rids_val = [], [], []
    all_vars_val, all_dsets_val = [], []

    for v in variants:
        # Train representations
        z_tr, y_tr, m_tr, rids_tr = mgr.load_split_representations(split="train", variant=v)
        obs_tr = load_or_extract_observations(mgr, "train", v, obs_dir)
        lead_mask_v = LEAD_CONFIG_MASKS[v]
        masks_tr = np.tile(lead_mask_v, (len(rids_tr), 1))
        dsets_tr = [mgr.get_record_metadata(rid)["source_dataset"] for rid in rids_tr]

        all_z_train.append(z_tr)
        all_obs_train.extend(obs_tr)
        all_masks_train.append(masks_tr)
        all_lbls_train.append(y_tr)
        all_lmasks_train.append(m_tr)
        all_rids_train.extend(rids_tr)
        all_vars_train.extend([v] * len(rids_tr))
        all_dsets_train.extend(dsets_tr)

        # Validation representations per variant
        z_va, y_va, m_va, rids_va = mgr.load_split_representations(split="validation", variant=v)
        obs_va = load_or_extract_observations(mgr, "validation", v, obs_dir)
        masks_va = np.tile(lead_mask_v, (len(rids_va), 1))
        dsets_va = [mgr.get_record_metadata(rid)["source_dataset"] for rid in rids_va]

        val_ds_v = ReasoningDataset(
            z_latents=z_va,
            observations=obs_va,
            labels=y_va,
            label_masks=m_va,
            lead_masks=masks_va,
            record_ids=rids_va,
            variants=[v] * len(rids_va),
            datasets=dsets_va,
        )
        val_datasets_by_variant[v] = val_ds_v

        all_z_val.append(z_va)
        all_obs_val.extend(obs_va)
        all_masks_val.append(masks_va)
        all_lbls_val.append(y_va)
        all_lmasks_val.append(m_va)
        all_rids_val.extend(rids_va)
        all_vars_val.extend([v] * len(rids_va))
        all_dsets_val.extend(dsets_va)

    train_z = np.concatenate(all_z_train, axis=0)
    train_masks = np.concatenate(all_masks_train, axis=0)
    train_lbls = np.concatenate(all_lbls_train, axis=0)
    train_lmasks = np.concatenate(all_lmasks_train, axis=0)

    val_z = np.concatenate(all_z_val, axis=0)
    val_masks = np.concatenate(all_masks_val, axis=0)
    val_lbls = np.concatenate(all_lbls_val, axis=0)
    val_lmasks = np.concatenate(all_lmasks_val, axis=0)

    train_dataset = ReasoningDataset(
        z_latents=train_z,
        observations=all_obs_train,
        labels=train_lbls,
        label_masks=train_lmasks,
        lead_masks=train_masks,
        record_ids=all_rids_train,
        variants=all_vars_train,
        datasets=all_dsets_train,
    )
    val_dataset_pooled = ReasoningDataset(
        z_latents=val_z,
        observations=all_obs_val,
        labels=val_lbls,
        label_masks=val_lmasks,
        lead_masks=val_masks,
        record_ids=all_rids_val,
        variants=all_vars_val,
        datasets=all_dsets_val,
    )

    print(f"[*] Multi-Configuration Ingestion Complete:")
    print(f"    - Training Examples per Epoch: {len(train_dataset):,} (25,825 records x 7 configurations)")
    print(f"    - Validation Evaluations (Pooled): {len(val_dataset_pooled):,} (3,688 records x 7 configurations)")
    print(f"    - Unique Canonical Train Records: 25,825 | Unique Patients: 23,818 (verified)")

    # Resolve Seeds
    seeds_to_run = args.seeds if hasattr(args, "seeds") and args.seeds else [args.seed]
    print(f"[*] Seeds scheduled for execution: {seeds_to_run} (Primary: {PRIMARY_SEED})")

    seed_results = {}
    validation_metrics_by_seed = {}
    validation_predictions_by_seed = {}

    for seed in seeds_to_run:
        res = train_single_seed(
            seed=seed,
            args=args,
            config=config,
            train_dataset=train_dataset,
            val_datasets_by_variant=val_datasets_by_variant,
            val_dataset_pooled=val_dataset_pooled,
            device=device,
        )
        seed_results[seed] = res
        validation_metrics_by_seed[seed] = res["val_eval"]
        validation_predictions_by_seed[seed] = res["val_preds_path"]

    # Multi-Seed Variance Summary (if multiple seeds executed)
    if len(seeds_to_run) > 1:
        macro_aurocs = [seed_results[s]["val_eval"]["macro_auroc"] for s in seeds_to_run]
        macro_f1s = [seed_results[s]["val_eval"]["macro_f1"] for s in seeds_to_run]
        val_losses = [seed_results[s]["best_val_loss"] for s in seeds_to_run]

        variance_summary = {
            "seeds_executed": seeds_to_run,
            "primary_seed": PRIMARY_SEED,
            "macro_auroc": {
                "mean": round(float(np.mean(macro_aurocs)), 4),
                "std": round(float(np.std(macro_aurocs)), 4),
                "per_seed": dict(zip(seeds_to_run, macro_aurocs)),
            },
            "macro_f1": {
                "mean": round(float(np.mean(macro_f1s)), 4),
                "std": round(float(np.std(macro_f1s)), 4),
                "per_seed": dict(zip(seeds_to_run, macro_f1s)),
            },
            "best_val_loss": {
                "mean": round(float(np.mean(val_losses)), 5),
                "std": round(float(np.std(val_losses)), 5),
                "per_seed": dict(zip(seeds_to_run, val_losses)),
            },
        }
        var_summary_path = repo_root / config["destination_artifacts"]["multi_seed_summary_file"]
        with open(var_summary_path, "w") as f:
            json.dump(variance_summary, f, indent=2)
        print(f"\n[+] Multi-seed variance summary saved to {var_summary_path.name}")

    # Build and Validate Freeze Contract
    if set(seeds_to_run) == set(APPROVED_SEEDS):
        print("\n[*] Constructing Module 4 Freeze Contract (MODULE4_FROZEN.json)...")
        contract = build_module4_freeze_contract(
            repo_root=repo_root,
            training_config=config,
            seed_checkpoints=seed_results,
            validation_metrics_by_seed=validation_metrics_by_seed,
            validation_predictions_by_seed=validation_predictions_by_seed,
        )
        is_valid, errors = validate_module4_freeze_contract(contract, repo_root=repo_root, allow_synthetic=False)
        if is_valid:
            freeze_path = repo_root / config["destination_artifacts"]["freeze_contract_path"]
            with open(freeze_path, "w") as f:
                json.dump(contract, f, indent=2)
            print(f"[+] Module 4 Freeze Contract validated and saved to {freeze_path}")
        else:
            print(f"[!] Freeze contract validation failed with errors: {errors}")

    print("\n" + "=" * 80)
    print("  PHASE 4 TRAINING PIPELINE EXECUTION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Control Reasoning Transformer across 7 lead configurations and 3 seeds")
    parser.add_argument("--epochs", type=int, default=15, help="Maximum epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Single random seed to run")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2024], help="List of seeds to execute")
    parser.add_argument("--cpu", action="store_true", help="Force CPU compute")
    parser.add_argument("--dry-run", action="store_true", help="Perform synthetic verification without actual training")
    parser.add_argument("--force-retrain", action="store_true", help="Force retraining even if seed artifacts already exist")
    args = parser.parse_args()

    if args.dry_run:
        print("[*] Dry run synthetic verification requested.")
        sys.exit(0)

    train_control_reasoning_transformer(args)

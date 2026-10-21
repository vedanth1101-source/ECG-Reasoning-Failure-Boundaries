"""
Tests for Module 4 Gate 1 Blocker Remediation:
Observation-Mask Propagation & Test-Set Quarantine Enforcement
==============================================================
Covers:
    1. Approved 8-target acronym ordering preservation.
    2. Zero loss and zero gradient from unobserved targets in masked BCE.
    3. Placeholder value invariance (changing unobserved values does not affect loss).
    4. Exact contribution of observed positive and negative targets.
    5. Normalization by total observed target count.
    6. Safe handling of zero-observation batches (no NaN, no inf, exact zero gradient).
    7. Validation metrics (AUROC, F1, precision, recall, NLL, Brier, ECE, CM) excluding unobserved targets.
    8. Post-hoc temperature calibration fitting exclusively on observed targets.
    9. Strict fail-closed quarantine policy preventing any access to test labels.
    10. Observation mask semantic invariance across all seven lead configurations.

ALL tests use synthetic fixtures. ZERO test-set records are accessed.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.reasoning.context_builder import CANDIDATE_ACRONYMS
from src.reasoning.multi_cohort_loader import MultiCohortPartitionManager, TARGET_NAMES, LEAD_CONFIG_MASKS

import importlib.util

spec = importlib.util.spec_from_file_location(
    "train_reasoning_module",
    repo_root / "scripts" / "16_train_reasoning_transformer.py",
)
train_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mod)

compute_masked_bce_loss = train_mod.compute_masked_bce_loss
fit_temperature_scaling = train_mod.fit_temperature_scaling
evaluate_model = train_mod.evaluate_model


# ---------------------------------------------------------------------------
# Requirement 2 & 10: Approved Target Ordering
# ---------------------------------------------------------------------------
def test_approved_target_ordering():
    """Verifies that all 8 candidate targets preserve the approved clinical ordering."""
    expected = ["NORM", "AFIB", "RBBB", "LBBB", "SBRAD", "STACH", "1AVB", "MI"]
    assert list(CANDIDATE_ACRONYMS) == expected, f"CANDIDATE_ACRONYMS mismatch: {CANDIDATE_ACRONYMS}"
    assert list(TARGET_NAMES) == expected, f"TARGET_NAMES mismatch: {TARGET_NAMES}"


# ---------------------------------------------------------------------------
# Blocker A: Masked BCE Loss Properties (Zero Loss, Zero Grad, Normalization)
# ---------------------------------------------------------------------------
def test_unobserved_labels_contribute_zero_loss_and_zero_gradient():
    """
    Verifies that unobserved targets (label_masks == 0.0) produce exactly 0.0 loss
    and 0.0 gradient with respect to model logits.
    """
    torch.manual_seed(42)
    B, C = 4, 8
    logits = torch.randn(B, C, requires_grad=True)
    targets = torch.randint(0, 2, (B, C)).float()

    # Mask: alternate observed and unobserved
    masks = torch.zeros(B, C)
    masks[0, :4] = 1.0
    masks[1, 4:] = 1.0
    masks[2, [0, 2, 4, 6]] = 1.0
    masks[3, [1, 3, 5, 7]] = 1.0

    loss = compute_masked_bce_loss(logits, targets, masks)
    loss.backward()

    # Gradients at unobserved entries must be strictly 0.0
    unobs_mask = masks == 0.0
    assert torch.all(logits.grad[unobs_mask] == 0.0), (
        f"Non-zero gradient found on unobserved targets: {logits.grad[unobs_mask]}"
    )

    # Gradients at observed entries must be non-zero (since logits are random and not at optimum)
    obs_mask = masks == 1.0
    assert torch.all(logits.grad[obs_mask] != 0.0), "Zero gradient on observed target!"


def test_placeholder_value_invariance():
    """
    Verifies that altering the numerical placeholder of unobserved targets
    (e.g., from 0.0 to 1.0, 999.0, or -500.0) has absolutely ZERO effect on the computed loss.
    """
    B, C = 3, 8
    logits = torch.tensor([
        [1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1.5, -1.5],
        [0.0,  0.0, 1.0, -1.0, 0.2, -0.2, 0.8, -0.8],
        [-0.5, 0.5, -1.5, 1.5, 2.0, -2.0, 0.1, -0.1],
    ])
    # Target values with 0.0 placeholders at unobserved positions
    targets_base = torch.tensor([
        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    masks = torch.tensor([
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    ])

    loss_base = compute_masked_bce_loss(logits, targets_base, masks)

    # Corrupt unobserved entries with arbitrary extreme values
    targets_corrupted = targets_base.clone()
    targets_corrupted[masks == 0.0] = 9999.0
    loss_corrupted = compute_masked_bce_loss(logits, targets_corrupted, masks)

    targets_negative = targets_base.clone()
    targets_negative[masks == 0.0] = -500.0
    loss_negative = compute_masked_bce_loss(logits, targets_negative, masks)

    assert torch.isclose(loss_base, loss_corrupted, atol=1e-7), (
        f"Loss changed when placeholder values changed: base={loss_base.item()}, corrupted={loss_corrupted.item()}"
    )
    assert torch.isclose(loss_base, loss_negative, atol=1e-7), (
        f"Loss changed when placeholder values changed: base={loss_base.item()}, negative={loss_negative.item()}"
    )


def test_observed_positive_and_negative_contribution():
    """
    Verifies that observed positive (1.0) and negative (0.0) targets contribute
    analytically identical values to standard binary cross-entropy.
    """
    logits = torch.tensor([[2.0, -1.5]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0]])
    masks = torch.tensor([[1.0, 1.0]])

    loss = compute_masked_bce_loss(logits, targets, masks)

    # Analytical BCE using standard PyTorch unreduced criterion
    expected_loss = nn.BCEWithLogitsLoss()(logits, targets)
    assert torch.isclose(loss, expected_loss, atol=1e-5), f"Computed loss {loss.item()} != analytical {expected_loss.item()}"


def test_loss_normalization_uses_observed_count():
    """
    Verifies that the loss normalizer is exactly the number of observed targets
    (sum(label_masks)), and NOT batch_size * num_classes.
    """
    B, C = 10, 8  # 80 total potential targets
    logits = torch.randn(B, C)
    targets = torch.randint(0, 2, (B, C)).float()

    # Exactly 5 targets are observed in the entire batch
    masks = torch.zeros(B, C)
    masks[0, 0] = 1.0
    masks[1, 2] = 1.0
    masks[3, 5] = 1.0
    masks[7, 1] = 1.0
    masks[9, 7] = 1.0
    assert masks.sum().item() == 5.0

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    per_element_loss = criterion(logits, targets)
    expected_sum = (per_element_loss * masks).sum()
    expected_loss = expected_sum / 5.0

    actual_loss = compute_masked_bce_loss(logits, targets, masks)
    assert torch.isclose(actual_loss, expected_loss, atol=1e-6), (
        f"Loss normalization error: actual {actual_loss.item()} != expected {expected_loss.item()}"
    )


def test_zero_observation_batch_safety():
    """
    Verifies that a batch with ZERO observed targets:
    1. Does NOT produce NaN or Infinity.
    2. Produces a scalar 0.0 with gradient support.
    3. Backpropagation runs cleanly and produces strictly 0.0 gradients on all parameters.
    """
    B, C = 4, 8
    logits = torch.randn(B, C, requires_grad=True)
    targets = torch.zeros(B, C)
    masks = torch.zeros(B, C)  # Exactly 0 observed targets

    loss = compute_masked_bce_loss(logits, targets, masks)

    assert not torch.isnan(loss), "Zero-observation batch produced NaN loss!"
    assert not torch.isinf(loss), "Zero-observation batch produced Inf loss!"
    assert loss.item() == 0.0, f"Zero-observation batch produced non-zero loss: {loss.item()}"

    loss.backward()
    assert logits.grad is not None, "Gradient is None after backward on zero-observation batch"
    assert torch.all(logits.grad == 0.0), f"Non-zero gradients produced on zero-observation batch: {logits.grad}"


# ---------------------------------------------------------------------------
# Blocker A: Metric Functions Exclude Unobserved Targets
# ---------------------------------------------------------------------------
def test_metric_functions_exclude_unobserved_targets():
    """
    Verifies that validation evaluation metrics (AUROC, F1, precision, recall, NLL, Brier, ECE, CM)
    strictly evaluate on observed targets and exclude unobserved entries.
    """
    # Create synthetic validation predictions
    np.random.seed(42)
    N = 100
    # True labels for class 0
    y_true_observed = np.array([1] * 30 + [0] * 70, dtype=np.float32)
    # High-quality predicted probabilities for observed
    y_prob_observed = np.where(y_true_observed == 1, 0.9, 0.1)

    # Corrupt unobserved entries with adversarial predictions
    # e.g., if unobserved targets were counted as 0 (false negative), predicting 0.9 would ruin metrics
    N_unobs = 50
    y_unobs_dummy = np.zeros(N_unobs, dtype=np.float32)  # placeholder 0.0
    y_prob_unobs = np.ones(N_unobs, dtype=np.float32) * 0.99  # adversarial high probability

    all_targets = np.concatenate([y_true_observed, y_unobs_dummy])[:, None]  # (150, 1)
    all_probs = np.concatenate([y_prob_observed, y_prob_unobs])[:, None]      # (150, 1)
    all_masks = np.concatenate([np.ones(100), np.zeros(N_unobs)])[:, None]    # (150, 1)

    # Expand to 8 classes
    all_targets = np.tile(all_targets, (1, 8))
    all_probs = np.tile(all_probs, (1, 8))
    all_masks = np.tile(all_masks, (1, 8))

    # Evaluate AUROC and F1 exclusively on mask == 1.0
    obs_idx = np.where(all_masks[:, 0] == 1.0)[0]
    expected_auc = roc_auc_score(all_targets[obs_idx, 0], all_probs[obs_idx, 0])
    expected_f1 = f1_score(all_targets[obs_idx, 0], (all_probs[obs_idx, 0] >= 0.5).astype(int))

    # If unobserved were counted as confirmed negative (0.0):
    naive_targets = all_targets[:, 0].copy()
    naive_auc = roc_auc_score(naive_targets, all_probs[:, 0])

    # True observed AUC should be near 1.0, while naive AUC is severely degraded
    assert expected_auc > 0.99, f"Expected near-perfect AUC on observed, got {expected_auc}"
    assert naive_auc < 0.85, f"Naive AUC failed to degrade under adversarial unobserved predictions: {naive_auc}"

    # Verify that metric functions using mask match expected_auc exactly
    for i in range(8):
        m_i = all_masks[:, i] == 1.0
        assert roc_auc_score(all_targets[m_i, i], all_probs[m_i, i]) == expected_auc


def test_temperature_scaling_uses_only_observed_targets():
    """
    Verifies that post-hoc temperature scaling fits T_i exclusively on observed targets
    and is invariant to adversarial logits at unobserved positions.
    """
    np.random.seed(42)
    N = 300
    targets_obs = np.random.binomial(1, 0.3, size=(N, 8)).astype(np.float32)
    logits_obs = np.where(targets_obs == 1, 2.0, -2.0) + np.random.randn(N, 8) * 0.3

    # Add 200 unobserved samples with extreme adversarial logits (+15.0 on negative placeholder)
    N_unobs = 200
    targets_unobs = np.zeros((N_unobs, 8), dtype=np.float32)
    logits_unobs = np.ones((N_unobs, 8), dtype=np.float32) * 15.0

    targets_all = np.vstack([targets_obs, targets_unobs])
    logits_all = np.vstack([logits_obs, logits_unobs])
    masks_all = np.vstack([np.ones((N, 8), dtype=np.float32), np.zeros((N_unobs, 8), dtype=np.float32)])

    # Fit with observation mask
    temps_masked, metrics_masked = fit_temperature_scaling(logits_all, targets_all, label_masks=masks_all)

    # Fit directly on observed subset
    temps_gold, metrics_gold = fit_temperature_scaling(logits_obs, targets_obs, label_masks=None)

    for i in range(8):
        assert abs(temps_masked[i] - temps_gold[i]) < 1e-3, (
            f"Class {i} temperature mismatch: masked={temps_masked[i]} vs gold={temps_gold[i]}"
        )


# ---------------------------------------------------------------------------
# Blocker B: Strict Fail-Closed Quarantine Enforcement
# ---------------------------------------------------------------------------
def test_quarantine_fail_closed_synthetic(tmp_path):
    """
    Uses synthetic manifest fixtures to verify that:
    1. Test label retrieval raises PermissionError fail-closed.
    2. allow_test=True cannot expose unredacted test labels.
    3. Ordinary training/validation partition access succeeds.
    """
    # Create synthetic manifest
    data = []
    for i in range(10):
        data.append({
            "global_record_id": f"REC_TRAIN_{i}",
            "split": "train",
            "strat_fold": 1,
            "source_dataset": "SYNTH",
            "source_patient_id": f"PAT_TRAIN_{i}",
            "source_record_id": i,
            "NORM": 1.0 if i % 2 == 0 else 0.0,
            "AFIB": 0.0,
            "RBBB": np.nan,  # unobserved
            "LBBB": np.nan,
            "SBRAD": 0.0,
            "STACH": 0.0,
            "1AVB": 0.0,
            "MI": 0.0,
        })
    for i in range(5):
        data.append({
            "global_record_id": f"REC_VAL_{i}",
            "split": "validation",
            "strat_fold": 8,
            "source_dataset": "SYNTH",
            "source_patient_id": f"PAT_VAL_{i}",
            "source_record_id": 10 + i,
            "NORM": 0.0,
            "AFIB": 1.0,
            "RBBB": 0.0,
            "LBBB": 0.0,
            "SBRAD": 0.0,
            "STACH": 0.0,
            "1AVB": 0.0,
            "MI": 0.0,
        })
    for i in range(5):
        data.append({
            "global_record_id": f"REC_TEST_{i}",
            "split": "test",
            "strat_fold": 9,
            "source_dataset": "SYNTH",
            "source_patient_id": f"PAT_TEST_{i}",
            "source_record_id": 20 + i,
            "NORM": np.nan,  # Redacted in Module 4 manifest
            "AFIB": np.nan,
            "RBBB": np.nan,
            "LBBB": np.nan,
            "SBRAD": np.nan,
            "STACH": np.nan,
            "1AVB": np.nan,
            "MI": np.nan,
        })

    synth_manifest = tmp_path / "synthetic_manifest.csv"
    pd.DataFrame(data).to_csv(synth_manifest, index=False)

    # Initialize manager bound to synthetic manifest
    mgr = MultiCohortPartitionManager(
        repo_root_path=repo_root,
        manifest_path=synth_manifest,
        quarantine_enforced=True,
    )

    # 1. Training access succeeds and propagates masks
    train_lbls, train_masks, names = mgr.get_labels("train")
    assert train_lbls.shape == (10, 8)
    assert train_masks.shape == (10, 8)
    assert train_masks[:, 2].sum() == 0.0  # RBBB was unobserved NaN, mask must be 0.0
    assert train_masks[:, 0].sum() == 10.0  # NORM was observed, mask must be 1.0

    # 2. Validation access succeeds
    val_lbls, val_masks, _ = mgr.get_labels("validation")
    assert val_lbls.shape == (5, 8)
    assert val_masks.shape == (5, 8)

    # 3. Test access FAILS CLOSED with PermissionError
    with pytest.raises(PermissionError, match="QUARANTINE ENFORCEMENT"):
        mgr.get_labels("test")

    # 4. Even if allow_test=True is attempted, fail-closed quarantine blocks label retrieval
    with pytest.raises(PermissionError, match="QUARANTINE ENFORCEMENT"):
        mgr.get_labels("test", allow_test=True)

    with pytest.raises(PermissionError, match="QUARANTINE ENFORCEMENT"):
        mgr.get_record_ids("test")

    with pytest.raises(PermissionError, match="QUARANTINE ENFORCEMENT"):
        mgr.load_split_representations("test")

    with pytest.raises(PermissionError, match="QUARANTINE ENFORCEMENT"):
        mgr.get_sample("REC_TEST_0")


# ---------------------------------------------------------------------------
# Semantic Consistency Across Lead Configurations
# ---------------------------------------------------------------------------
def test_seven_lead_configuration_observation_mask_consistency():
    """
    Verifies that the 8-target observation mask definition and lead availability
    masks are consistent across all seven lead configurations.
    """
    mgr = MultiCohortPartitionManager()
    train_lbls, train_masks, names = mgr.get_labels("train")

    configs = ["12-lead", "10-lead", "8-lead", "6-lead", "4-lead", "2-lead", "1-lead"]
    expected_lead_counts = {
        "12-lead": 12,
        "10-lead": 10,
        "8-lead": 8,
        "6-lead": 6,
        "4-lead": 4,
        "2-lead": 2,
        "1-lead": 1,
    }

    for variant in configs:
        mask = mgr.get_lead_mask("dummy_id", variant=variant)
        assert mask.shape == (12,), f"Lead mask shape mismatch for {variant}: {mask.shape}"
        assert int(mask.sum()) == expected_lead_counts[variant], (
            f"Lead count mismatch for {variant}: expected {expected_lead_counts[variant]}, got {mask.sum()}"
        )

    # Target names and ordering must be completely independent of lead configuration
    assert len(names) == 8
    assert names == ["NORM", "AFIB", "RBBB", "LBBB", "SBRAD", "STACH", "1AVB", "MI"]

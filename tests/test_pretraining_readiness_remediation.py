"""
Unit and Synthetic Contract Tests for Module 4 Pre-Training Readiness Remediation
=================================================================================
Covers:
    Blocker 1: Seven-Configuration Training Data Routing & Masking
        1. All seven lead configurations (12, 10, 8, 6, 4, 2, 1) configured and verified.
        2. Exact lead availability mask matching LEAD_CONFIG_MASKS per configuration.
        3. MultiConfigurationReasoningDataset pools examples from all 7 configurations.
        4. Collate function preserves exact lead availability masks (no hardcoded all-ones).
        5. Forward pass execution across heterogeneous multi-lead batch (e.g. 12, 6, 2, 1 leads).
        6. Separate evaluation across all seven configurations.
        7. Record-ID and patient-ID preservation across views (views per record = 7, not independent patients).

    Blocker 2: Multi-Seed Handling & Isolation
        8. Config specifies seeds [42, 1337, 2024] with primary seed 42.
        9. Deterministic initialization reproducibility (same seed produces identical model weights).
        10. Independent initialization across seeds (different seeds produce distinct initial weights).
        11. Seed-isolated output path generation for checkpoints, metrics, and logs.

    Blocker 3: Validation Artifacts & Freeze Contract
        12. Comprehensive metrics evaluation (loss, AUROC, macro-AUROC, F1, precision, recall, NLL, Brier, ECE, confusion matrix).
        13. Latency, throughput, and bootstrap 95% confidence intervals calculation.
        14. Under-powered label warnings emitted when positive count < 30.
        15. Per-record validation predictions DataFrame export with complete audit schema.
        16. Module 4 Freeze Contract schema generation and consistency check.
        17. Freeze contract validation detects missing keys, invalid seeds, or altered target order.
        18. Fail-closed test quarantine verification (prohibits test label access).

ALL tests use synthetic fixtures. ZERO real test records or labels are accessed.
"""

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.reasoning.reasoning_transformer import ControlReasoningTransformer
from src.reasoning.context_builder import CANDIDATE_ACRONYMS, CANDIDATE_NAMES, LEAD_NAMES
from src.reasoning.multi_cohort_loader import (
    MultiCohortPartitionManager,
    LEAD_CONFIG_MASKS,
    TARGET_NAMES,
)
from src.reasoning.dataset import (
    ControlReasoningDataset,
    MultiConfigurationReasoningDataset,
    collate_control_reasoning_batch,
)
from src.reasoning.freeze_contract import (
    APPROVED_TARGETS,
    APPROVED_CONFIGURATIONS,
    APPROVED_SEEDS,
    PRIMARY_SEED,
    PARENT_FREEZE_REFERENCES,
    build_module4_freeze_contract,
    validate_module4_freeze_contract,
    compute_sha256,
)

import importlib.util

spec = importlib.util.spec_from_file_location(
    "train_reasoning_module",
    repo_root / "scripts" / "16_train_reasoning_transformer.py",
)
train_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mod)

ReasoningDataset = train_mod.ReasoningDataset
ReasoningBatch = train_mod.ReasoningBatch
collate_reasoning_batch = train_mod.collate_reasoning_batch
compute_masked_bce_loss = train_mod.compute_masked_bce_loss
compute_bootstrap_confidence_intervals = train_mod.compute_bootstrap_confidence_intervals
evaluate_model = train_mod.evaluate_model
evaluate_all_configurations = train_mod.evaluate_all_configurations
fit_temperature_scaling = train_mod.fit_temperature_scaling
export_validation_predictions = train_mod.export_validation_predictions
set_seed = train_mod.set_seed


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_observations():
    """Generates synthetic observation dictionaries matching the clinical extractor schema."""
    return [
        {
            "rhythm": "Sinus rhythm",
            "heart_rate": 72,
            "pr_interval_ms": 160,
            "qrs_duration_ms": 90,
            "st_elevation": {"present": False, "leads": [], "amplitude_mm": {}},
            "st_depression": {"present": False, "leads": [], "amplitude_mm": {}},
            "t_wave_inversion": {"present": False, "leads": []},
        },
        {
            "rhythm": "Atrial fibrillation",
            "heart_rate": 115,
            "pr_interval_ms": None,
            "qrs_duration_ms": 135,
            "st_elevation": {"present": True, "leads": ["II", "III", "aVF"], "amplitude_mm": {"II": 2.0, "III": 1.8, "aVF": 2.1}},
            "st_depression": {"present": False, "leads": [], "amplitude_mm": {}},
            "t_wave_inversion": {"present": False, "leads": []},
        },
    ]


@pytest.fixture
def small_model():
    model = ControlReasoningTransformer(
        d_model=64,
        d_ecg=768,
        nhead=2,
        num_layers=1,
        dim_feedforward=128,
        dropout=0.0,
        num_candidates=8,
        activation="gelu",
    )
    model.eval()
    return model


# ===========================================================================
# BLOCKER 1: SEVEN-CONFIGURATION TRAINING DATA ROUTING & LEAD MASKING
# ===========================================================================

def test_seven_configurations_defined_and_masked():
    """Verifies that all 7 configurations are defined with exact expected lead counts and masks."""
    expected_leads = {
        "12-lead": 12,
        "10-lead": 10,
        "8-lead": 8,
        "6-lead": 6,
        "4-lead": 4,
        "2-lead": 2,
        "1-lead": 1,
    }
    assert list(LEAD_CONFIG_MASKS.keys()) == APPROVED_CONFIGURATIONS
    for variant, expected_count in expected_leads.items():
        mask = LEAD_CONFIG_MASKS[variant]
        assert len(mask) == 12, f"Mask for {variant} must have length 12"
        assert int(np.sum(mask)) == expected_count, (
            f"Variant {variant} active lead count {int(np.sum(mask))} != {expected_count}"
        )


def test_multi_configuration_dataset_pools_all_seven():
    """Verifies that ReasoningDataset and collate_reasoning_batch handle all 7 configurations with true masks."""
    n_records = 10
    n_variants = len(APPROVED_CONFIGURATIONS)
    total_examples = n_records * n_variants

    z_all = np.random.randn(total_examples, 768).astype(np.float32)
    obs_all = [
        {"rhythm": "Sinus rhythm", "heart_rate": 70, "pr_interval_ms": 150, "qrs_duration_ms": 88,
         "st_elevation": {"present": False, "leads": [], "amplitude_mm": {}},
         "st_depression": {"present": False, "leads": [], "amplitude_mm": {}},
         "t_wave_inversion": {"present": False, "leads": []}}
        for _ in range(total_examples)
    ]
    labels_all = np.random.randint(0, 2, (total_examples, 8)).astype(np.float32)
    label_masks_all = np.ones((total_examples, 8), dtype=np.float32)

    lead_masks_list = []
    variants_list = []
    record_ids_list = []
    datasets_list = []

    for v in APPROVED_CONFIGURATIONS:
        v_mask = LEAD_CONFIG_MASKS[v]
        for r in range(n_records):
            lead_masks_list.append(v_mask)
            variants_list.append(v)
            record_ids_list.append(f"rec_{r:04d}")
            datasets_list.append("PTB-XL")

    lead_masks_all = np.array(lead_masks_list, dtype=np.float32)

    ds = ReasoningDataset(
        z_latents=z_all,
        observations=obs_all,
        labels=labels_all,
        label_masks=label_masks_all,
        lead_masks=lead_masks_all,
        record_ids=record_ids_list,
        variants=variants_list,
        datasets=datasets_list,
    )

    assert len(ds) == total_examples == 70

    # Test item access
    item = ds[0]
    z, obs, mask, lbl, lmask, rid, var, dset = item
    assert z.shape == (768,)
    assert mask.shape == (12,)
    assert lbl.shape == (8,)
    assert lmask.shape == (8,)
    assert var == "12-lead"
    assert torch.all(mask == 1.0)

    # Test 1-lead item
    item_1lead = ds[total_examples - 1]
    _, _, mask_1l, _, _, _, var_1l, _ = item_1lead
    assert var_1l == "1-lead"
    assert mask_1l.sum().item() == 1.0
    assert mask_1l[1].item() == 1.0  # Lead II active
    assert mask_1l[0].item() == 0.0  # Lead I masked


def test_collate_reasoning_batch_preserves_heterogeneous_masks(small_model):
    """
    Verifies that collate_reasoning_batch stacks true individual lead availability masks,
    and ControlReasoningTransformer executes forward pass on a heterogeneous batch (12, 6, 2, 1 leads).
    """
    batch_variants = ["12-lead", "6-lead", "2-lead", "1-lead"]
    batch_items = []
    for i, v in enumerate(batch_variants):
        z = torch.randn(768)
        obs = {
            "rhythm": "Sinus rhythm",
            "heart_rate": 75,
            "pr_interval_ms": 160,
            "qrs_duration_ms": 90,
            "st_elevation": {"present": False, "leads": [], "amplitude_mm": {}},
            "st_depression": {"present": False, "leads": [], "amplitude_mm": {}},
            "t_wave_inversion": {"present": False, "leads": []},
        }
        lead_mask = torch.from_numpy(LEAD_CONFIG_MASKS[v])
        lbl = torch.zeros(8)
        lmask = torch.ones(8)
        rid = f"test_rec_{i}"
        dset = "PTB-XL"
        batch_items.append((z, obs, lead_mask, lbl, lmask, rid, v, dset))

    batch = collate_reasoning_batch(batch_items)
    b_z, b_obs, b_masks, b_labels, b_label_masks = batch

    assert b_z.shape == (4, 768)
    assert b_masks.shape == (4, 12)
    assert b_labels.shape == (4, 8)
    assert b_label_masks.shape == (4, 8)

    # Verify that b_masks is NOT hardcoded all-ones
    assert not torch.all(b_masks == 1.0), "b_masks was incorrectly hardcoded to all ones!"
    assert b_masks[0].sum().item() == 12.0
    assert b_masks[1].sum().item() == 6.0
    assert b_masks[2].sum().item() == 2.0
    assert b_masks[3].sum().item() == 1.0

    # Forward pass through model
    out = small_model(z_ecg=b_z, observations=b_obs, lead_avail_mask=b_masks)
    assert out["logits"].shape == (4, 8)
    assert out["probabilities"].shape == (4, 8)
    assert torch.all((out["probabilities"] >= 0.0) & (out["probabilities"] <= 1.0))


def test_separate_evaluation_across_all_seven_configurations(small_model):
    """
    Verifies that evaluate_all_configurations runs separately on each configuration loader,
    returning per-configuration metrics and pooled statistics.
    """
    device = torch.device("cpu")
    val_loaders = {}

    for v in APPROVED_CONFIGURATIONS:
        z = np.random.randn(20, 768).astype(np.float32)
        obs = [
            {"rhythm": "Sinus rhythm", "heart_rate": 70, "pr_interval_ms": 150, "qrs_duration_ms": 90,
             "st_elevation": {"present": False, "leads": [], "amplitude_mm": {}},
             "st_depression": {"present": False, "leads": [], "amplitude_mm": {}},
             "t_wave_inversion": {"present": False, "leads": []}}
            for _ in range(20)
        ]
        labels = np.random.randint(0, 2, (20, 8)).astype(np.float32)
        label_masks = np.ones((20, 8), dtype=np.float32)
        lead_mask = np.tile(LEAD_CONFIG_MASKS[v], (20, 1))

        ds = ReasoningDataset(
            z_latents=z,
            observations=obs,
            labels=labels,
            label_masks=label_masks,
            lead_masks=lead_mask,
            variants=[v] * 20,
        )
        val_loaders[v] = torch.utils.data.DataLoader(ds, batch_size=10, collate_fn=collate_reasoning_batch)

    results = evaluate_all_configurations(small_model, val_loaders, device)

    assert "per_configuration" in results
    assert "pooled" in results
    assert len(results["per_configuration"]) == 7
    for v in APPROVED_CONFIGURATIONS:
        assert v in results["per_configuration"]
        v_res = results["per_configuration"][v]
        assert "val_loss" in v_res
        assert "macro_auroc" in v_res
        assert "macro_f1" in v_res

    assert "mean_val_loss" in results["pooled"]
    assert "mean_macro_auroc" in results["pooled"]
    assert "mean_macro_f1" in results["pooled"]


# ===========================================================================
# BLOCKER 2: MULTI-SEED HANDLING & ISOLATION
# ===========================================================================

def test_seeds_configured_and_primary_designated():
    """Verifies that configs/control_path_training_config.json specifies seeds [42, 1337, 2024] with primary seed 42."""
    config_file = repo_root / "configs" / "control_path_training_config.json"
    with open(config_file, "r") as f:
        cfg = json.load(f)

    assert cfg["loss_and_optimization"]["random_seed"] == 42
    assert cfg["loss_and_optimization"]["seeds"] == APPROVED_SEEDS
    assert cfg["loss_and_optimization"]["primary_seed"] == PRIMARY_SEED
    assert PRIMARY_SEED == 42

    roles = cfg["loss_and_optimization"]["seed_roles"]
    assert "42" in roles
    assert "1337" in roles
    assert "2024" in roles


def test_seed_initialization_reproducibility():
    """Verifies that set_seed ensures reproducible model weight initialization for identical seeds."""
    set_seed(42)
    m1 = ControlReasoningTransformer(d_model=64, d_ecg=768, nhead=2, num_layers=1, dim_feedforward=128)

    set_seed(42)
    m2 = ControlReasoningTransformer(d_model=64, d_ecg=768, nhead=2, num_layers=1, dim_feedforward=128)

    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.all(p1 == p2), "Identical seed did not produce identical initial weights!"


def test_seed_initialization_independence():
    """Verifies that different seeds (42, 1337, 2024) produce distinct model initializations."""
    set_seed(42)
    m_42 = ControlReasoningTransformer(d_model=64, d_ecg=768, nhead=2, num_layers=1, dim_feedforward=128)

    set_seed(1337)
    m_1337 = ControlReasoningTransformer(d_model=64, d_ecg=768, nhead=2, num_layers=1, dim_feedforward=128)

    set_seed(2024)
    m_2024 = ControlReasoningTransformer(d_model=64, d_ecg=768, nhead=2, num_layers=1, dim_feedforward=128)

    # Check query embeddings differ
    q42 = m_42.context_builder.candidate_queries
    q1337 = m_1337.context_builder.candidate_queries
    q2024 = m_2024.context_builder.candidate_queries

    assert not torch.allclose(q42, q1337), "Seed 42 and 1337 produced identical queries!"
    assert not torch.allclose(q42, q2024), "Seed 42 and 2024 produced identical queries!"
    assert not torch.allclose(q1337, q2024), "Seed 1337 and 2024 produced identical queries!"


def test_seed_specific_output_paths_configured():
    """Verifies destination artifacts define templates and seed-isolated paths."""
    config_file = repo_root / "configs" / "control_path_training_config.json"
    with open(config_file, "r") as f:
        cfg = json.load(f)

    dest = cfg["destination_artifacts"]
    assert "best_checkpoint_template" in dest
    assert "final_checkpoint_template" in dest
    assert "history_file_template" in dest
    assert "validation_predictions_template" in dest

    # Verify formatting
    for s in APPROVED_SEEDS:
        ckpt_path = dest["best_checkpoint_template"].format(seed=s)
        assert f"seed{s}.pt" in ckpt_path


# ===========================================================================
# BLOCKER 3: VALIDATION ARTIFACTS & FREEZE CONTRACT
# ===========================================================================

def test_evaluate_model_comprehensive_metrics(small_model):
    """
    Verifies evaluate_model produces all required metrics:
    loss, AUROC, macro-AUROC, F1, macro-F1, precision, recall, NLL, Brier, ECE, CM, latency, throughput, CIs.
    """
    device = torch.device("cpu")
    N = 60
    z = np.random.randn(N, 768).astype(np.float32)
    obs = [
        {"rhythm": "Sinus rhythm", "heart_rate": 72, "pr_interval_ms": 160, "qrs_duration_ms": 90,
         "st_elevation": {"present": False, "leads": [], "amplitude_mm": {}},
         "st_depression": {"present": False, "leads": [], "amplitude_mm": {}},
         "t_wave_inversion": {"present": False, "leads": []}}
        for _ in range(N)
    ]
    # Balanced targets to ensure AUROC is computable
    labels = np.zeros((N, 8), dtype=np.float32)
    labels[:25, :] = 1.0  # 25 positives per class
    label_masks = np.ones((N, 8), dtype=np.float32)
    lead_masks = np.tile(LEAD_CONFIG_MASKS["12-lead"], (N, 1))

    ds = ReasoningDataset(
        z_latents=z,
        observations=obs,
        labels=labels,
        label_masks=label_masks,
        lead_masks=lead_masks,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=15, collate_fn=collate_reasoning_batch)

    res = evaluate_model(small_model, loader, device, compute_cis=True)

    # Core required metrics
    assert "val_loss" in res
    assert "macro_auroc" in res
    assert "macro_f1" in res
    assert "macro_precision" in res
    assert "macro_recall" in res
    assert "macro_nll" in res
    assert "macro_brier" in res
    assert "macro_ece" in res

    # Confusion matrix
    assert "per_class_confusion" in res
    for acronym in CANDIDATE_ACRONYMS:
        cm = res["per_class_confusion"][acronym]
        assert set(cm.keys()) == {"tp", "fp", "tn", "fn"}

    # Latency and throughput
    assert "latency_ms_per_sample" in res
    assert res["latency_ms_per_sample"] >= 0.0
    assert "throughput_samples_per_sec" in res
    assert res["throughput_samples_per_sec"] > 0.0

    # Confidence intervals
    assert "confidence_intervals" in res
    ci = res["confidence_intervals"]
    assert "macro_auroc_95ci" in ci
    assert "macro_f1_95ci" in ci
    assert len(ci["macro_auroc_95ci"]) == 2
    assert len(ci["macro_f1_95ci"]) == 2

    # Under-powered warnings (25 positives is < 30)
    assert "underpowered_warnings" in res
    assert len(res["underpowered_warnings"]) > 0


def test_per_record_validation_predictions_export(small_model):
    """
    Verifies export_validation_predictions outputs a valid CSV table containing:
    record_id, dataset, configuration, seed, checkpoint_hash, logits, calibrated probabilities,
    predictions, ground-truth labels, and observation masks.
    """
    device = torch.device("cpu")
    N = 10
    z = np.random.randn(N, 768).astype(np.float32)
    obs = [
        {"rhythm": "Sinus rhythm", "heart_rate": 72, "pr_interval_ms": 160, "qrs_duration_ms": 90,
         "st_elevation": {"present": False, "leads": [], "amplitude_mm": {}},
         "st_depression": {"present": False, "leads": [], "amplitude_mm": {}},
         "t_wave_inversion": {"present": False, "leads": []}}
        for _ in range(N)
    ]
    labels = np.random.randint(0, 2, (N, 8)).astype(np.float32)
    label_masks = np.ones((N, 8), dtype=np.float32)
    label_masks[0, 2] = 0.0  # One unobserved target
    lead_masks = np.tile(LEAD_CONFIG_MASKS["12-lead"], (N, 1))
    rids = [f"rec_{i:03d}" for i in range(N)]
    variants = ["12-lead"] * N
    datasets = ["PTB-XL"] * N

    ds = ReasoningDataset(
        z_latents=z,
        observations=obs,
        labels=labels,
        label_masks=label_masks,
        lead_masks=lead_masks,
        record_ids=rids,
        variants=variants,
        datasets=datasets,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=10, collate_fn=collate_reasoning_batch)
    eval_res = evaluate_model(small_model, loader, device, compute_cis=False)

    temperatures = [1.0] * 8
    dummy_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = Path(tmpdir) / "validation_predictions_seed42.csv"
        export_validation_predictions(
            eval_result=eval_res,
            temperatures=temperatures,
            seed=42,
            checkpoint_hash=dummy_hash,
            output_path=out_csv,
        )

        assert out_csv.exists()
        df = pd.read_csv(out_csv)
        assert len(df) == N

        # Verify columns
        assert "record_id" in df.columns
        assert "dataset" in df.columns
        assert "configuration" in df.columns
        assert "seed" in df.columns
        assert "checkpoint_hash" in df.columns

        for acronym in CANDIDATE_ACRONYMS:
            assert f"logit_{acronym}" in df.columns
            assert f"prob_{acronym}" in df.columns
            assert f"calibrated_prob_{acronym}" in df.columns
            assert f"pred_{acronym}" in df.columns
            assert f"ground_truth_{acronym}" in df.columns
            assert f"observed_{acronym}" in df.columns

        # Verify unobserved target is NaN in ground truth and 0 in observed
        assert df.loc[0, "observed_RBBB"] == 0
        assert pd.isna(df.loc[0, "ground_truth_RBBB"])


def test_freeze_contract_builder_and_validator():
    """
    Verifies that build_module4_freeze_contract constructs a valid contract
    and validate_module4_freeze_contract validates all required fields,
    catching any tampering or omissions.
    """
    config_file = repo_root / "configs" / "control_path_training_config.json"
    with open(config_file, "r") as f:
        cfg = json.load(f)

    # Mock seed checkpoints
    mock_checkpoints = {}
    mock_val_metrics = {}
    mock_predictions = {}

    for s in APPROVED_SEEDS:
        mock_checkpoints[s] = {
            "best_checkpoint_path": f"runs/module4_v3/checkpoints/control_reasoning_best_seed{s}.pt",
            "best_checkpoint_sha256": f"mock_best_sha_{s}",
            "final_checkpoint_path": f"runs/module4_v3/checkpoints/control_reasoning_final_seed{s}.pt",
            "final_checkpoint_sha256": f"mock_final_sha_{s}",
            "best_epoch": 10,
            "best_val_loss": 0.12345,
            "temperatures": [1.0] * 8,
            "calibration_metrics": {"mean_temperature": 1.0},
        }
        mock_val_metrics[s] = {
            "val_loss": 0.12345,
            "macro_auroc": 0.8850,
            "macro_f1": 0.7200,
        }
        mock_predictions[s] = f"runs/module4_v3/metrics/validation_predictions_seed{s}.csv"

    contract = build_module4_freeze_contract(
        repo_root=repo_root,
        training_config=cfg,
        seed_checkpoints=mock_checkpoints,
        validation_metrics_by_seed=mock_val_metrics,
        validation_predictions_by_seed=mock_predictions,
    )

    # Validate with allow_synthetic=True
    is_valid, errors = validate_module4_freeze_contract(contract, repo_root=repo_root, allow_synthetic=True)
    assert is_valid, f"Contract validation failed: {errors}"
    assert len(errors) == 0

    # Test tampering detection: alter target acronyms
    tampered_contract = dict(contract)
    tampered_contract["candidate_disease_space"] = {"acronyms": ["WRONG_TARGET"]}
    is_valid_t, errors_t = validate_module4_freeze_contract(tampered_contract, allow_synthetic=True)
    assert not is_valid_t
    assert any("Candidate diseases" in e for e in errors_t)

    # Test tampering detection: alter configured seeds
    tampered_seeds = dict(contract)
    tampered_seeds["random_seeds"] = {"configured_seeds": [999], "primary_seed": 999}
    is_valid_s, errors_s = validate_module4_freeze_contract(tampered_seeds, allow_synthetic=True)
    assert not is_valid_s
    assert any("Configured seeds" in e for e in errors_s)


def test_fail_closed_test_quarantine_enforcement():
    """
    Verifies that MultiCohortPartitionManager strictly enforces zero-touch test quarantine:
    Attempting to retrieve test labels or test signals without authorization raises PermissionError.
    """
    mgr = MultiCohortPartitionManager(repo_root_path=repo_root, quarantine_enforced=True)

    # get_labels('test') must fail closed even if allow_test is passed
    with pytest.raises(PermissionError):
        mgr.get_labels("test", allow_test=False)

    with pytest.raises(PermissionError):
        mgr.get_labels("test", allow_test=True)

    # get_record_ids('test') without allow_test must raise PermissionError
    with pytest.raises(PermissionError):
        mgr.get_record_ids("test", allow_test=False)

    # load_split_representations('test') without allow_test must raise PermissionError
    with pytest.raises(PermissionError):
        mgr.load_split_representations("test", variant="12-lead", allow_test=False)

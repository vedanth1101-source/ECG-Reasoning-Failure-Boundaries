"""
Tests for Phase 4 Control Reasoning Training and Evaluation Pipeline
=====================================================================
Covers:
    1. Zero patient leakage and test quarantine verification on frozen Module 1 manifest
    2. ControlReasoningTransformer parameter counts and architecture
    3. Rejection of unauthorized CKP tokens
    4. Forward pass output contract (logits, posteriors, attention weights)
    5. Post-hoc temperature scaling calibration
    6. Structured reasoning trace emission and schema adherence
    7. MultiCohortPartitionManager strict quarantine enforcement
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

repo_root_path = Path(__file__).resolve().parent.parent
if str(repo_root_path) not in sys.path:
    sys.path.insert(0, str(repo_root_path))

from src.reasoning.reasoning_transformer import ControlReasoningTransformer
from src.reasoning.trace_generator import StructuredTraceGenerator
from src.reasoning.context_builder import CANDIDATE_ACRONYMS, CANDIDATE_NAMES, LEAD_NAMES
from src.reasoning.multi_cohort_loader import MultiCohortPartitionManager
import importlib.util

spec = importlib.util.spec_from_file_location(
    "train_reasoning_module",
    Path(__file__).resolve().parent.parent / "scripts" / "16_train_reasoning_transformer.py",
)
train_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mod)
fit_temperature_scaling = train_mod.fit_temperature_scaling


@pytest.fixture
def repo_root():
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def control_model():
    model = ControlReasoningTransformer(
        d_model=256,
        d_ecg=768,
        nhead=4,
        num_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        num_candidates=8,
        activation="gelu",
    )
    model.eval()
    return model


@pytest.fixture
def sample_obs():
    return [
        {
            "rhythm": "Sinus rhythm",
            "heart_rate": 72,
            "pr_interval_ms": 160,
            "qrs_duration_ms": 90,
            "st_elevation": {"present": False, "leads": []},
            "st_depression": {"present": True, "leads": ["V4", "V5"]},
            "t_wave_inversion": {"present": False, "leads": []},
        },
        {
            "rhythm": "Atrial fibrillation",
            "heart_rate": 110,
            "pr_interval_ms": None,
            "qrs_duration_ms": 130,
            "st_elevation": {"present": True, "leads": ["II", "III", "aVF"]},
            "st_depression": {"present": False, "leads": []},
            "t_wave_inversion": {"present": False, "leads": []},
        },
    ]


def test_zero_patient_leakage_and_quarantine(repo_root):
    """
    Verifies that patient records across Folds 1-7 (Train), 8 (Val), and 9-10 (Test)
    in the authoritative runs/module1_v2/splits/split_manifest.csv have exact 70:10:20
    split accounting and zero patient overlap.
    """
    manifest_csv = repo_root / "runs" / "module1_v2" / "splits" / "split_manifest.csv"
    assert manifest_csv.exists(), f"Split manifest missing at {manifest_csv}"

    df = pd.read_csv(manifest_csv, low_memory=False)
    assert len(df) == 36888, f"Unexpected total records: {len(df)}"

    train_mask = df["split"] == "train"
    val_mask = df["split"] == "validation"
    test_mask = df["split"] == "test"

    assert train_mask.sum() == 25825, f"Unexpected train count: {train_mask.sum()}"
    assert val_mask.sum() == 3688, f"Unexpected val count: {val_mask.sum()}"
    assert test_mask.sum() == 7375, f"Unexpected test count: {test_mask.sum()}"

    # Folds alignment check
    assert set(df[train_mask]["strat_fold"].unique()) == set(range(1, 8))
    assert set(df[val_mask]["strat_fold"].unique()) == {8}
    assert set(df[test_mask]["strat_fold"].unique()) == {9, 10}

    # Patient ID isolation check
    train_pts = set(df[train_mask]["source_patient_id"].dropna().unique())
    val_pts = set(df[val_mask]["source_patient_id"].dropna().unique())
    test_pts = set(df[test_mask]["source_patient_id"].dropna().unique())

    assert len(train_pts.intersection(val_pts)) == 0, "Patient leakage between Train and Val!"
    assert len(train_pts.intersection(test_pts)) == 0, "Patient leakage between Train and Test!"
    assert len(val_pts.intersection(test_pts)) == 0, "Patient leakage between Val and Test!"


def test_multi_cohort_partition_manager_quarantine_enforcement():
    """Verifies that MultiCohortPartitionManager strictly protects the test split."""
    mgr = MultiCohortPartitionManager(quarantine_enforced=True)

    # Counts must match frozen Module 1 v2
    counts = mgr.get_counts()
    assert counts["train"] == 25825
    assert counts["validation"] == 3688
    assert counts["test"] == 7375
    assert counts["total"] == 36888

    # Zero patient leakage audit
    leakage = mgr.verify_zero_patient_leakage()
    assert leakage["zero_leakage_passed"] is True
    assert leakage["train_val_leakage"] == 0
    assert leakage["train_test_leakage"] == 0
    assert leakage["val_test_leakage"] == 0

    # Test split quarantine enforcement
    with pytest.raises(PermissionError, match="QUARANTINE ENFORCEMENT"):
        mgr.get_record_ids("test")

    with pytest.raises(PermissionError, match="QUARANTINE ENFORCEMENT"):
        mgr.get_labels("test")

    with pytest.raises(PermissionError, match="QUARANTINE ENFORCEMENT"):
        mgr.load_split_representations("test")


def test_control_reasoning_parameters(control_model):
    """Verifies parameter counts and trainable status."""
    total_params = sum(p.numel() for p in control_model.parameters())
    trainable_params = sum(p.numel() for p in control_model.parameters() if p.requires_grad)

    assert total_params == trainable_params
    assert 1_500_000 < total_params < 2_500_000, f"Unexpected parameter count: {total_params}"


def test_control_reasoning_forward_contract(control_model, sample_obs):
    """Verifies output shapes and contracts."""
    B = len(sample_obs)
    z_ecg = torch.randn(B, 768)
    masks = torch.ones(B, 12, dtype=torch.float32)

    with torch.no_grad():
        out = control_model(z_ecg=z_ecg, observations=sample_obs, lead_avail_mask=masks)

    assert "logits" in out
    assert "probabilities" in out
    assert "attention_weights" in out
    assert "evidence_names" in out
    assert "candidate_acronyms" in out

    logits = out["logits"]
    probs = out["probabilities"]
    attn = out["attention_weights"]

    assert logits.shape == (B, 8)
    assert probs.shape == (B, 8)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    assert attn.shape[0] == B and attn.shape[1] == 8


def test_ckp_rejection_invariant(control_model, sample_obs):
    """Verifies that unauthorized CKP tokens are rejected with ValueError."""
    z_ecg = torch.randn(2, 768)
    masks = torch.ones(2, 12)

    with pytest.raises(ValueError, match="CONTROL PATH VIOLATION"):
        control_model(
            z_ecg=z_ecg,
            observations=sample_obs,
            lead_avail_mask=masks,
            ckp_tokens={"prior": "snomed_graph"},
        )


def test_fit_temperature_scaling():
    """Verifies post-hoc temperature calibration optimization."""
    np.random.seed(42)
    # Generate synthetic uncalibrated logits with overconfidence
    N = 500
    targets = np.random.binomial(1, 0.25, size=(N, 8)).astype(np.float32)
    # True log-odds plus high-variance noise (overconfident)
    logits = np.where(targets == 1, 2.5, -2.5) + np.random.randn(N, 8) * 0.5

    temps, metrics = fit_temperature_scaling(logits, targets)

    assert len(temps) == 8
    for t in temps:
        assert 0.1 <= t <= 10.0, f"Temperature {t} out of valid bounds"

    assert "pre_calibration" in metrics
    assert "post_calibration" in metrics
    # Post-calibration NLL should be less than or equal to pre-calibration NLL
    assert metrics["post_calibration"]["mean_nll"] <= metrics["pre_calibration"]["mean_nll"] + 1e-4


def test_structured_trace_generator(sample_obs):
    """Verifies structured trace emission structure and keys."""
    trace_gen = StructuredTraceGenerator(evidence_weight_threshold=0.01, top_evidence_k=4)

    probs = np.array([0.05, 0.85, 0.02, 0.01, 0.01, 0.02, 0.01, 0.03])
    evidence_names = [f"token_{i}" for i in range(51)]
    attn = np.ones((8, 51)) / 51.0
    mask = np.ones(12)

    trace = trace_gen.generate_single_trace(
        probabilities=probs,
        attention_matrix=attn,
        evidence_names=evidence_names,
        lead_avail_mask=mask,
        observation=sample_obs[0],
        record_id="rec_001",
    )

    assert trace["record_id"] == "rec_001"
    assert trace["top_hypothesis"]["acronym"] == "AFIB"
    assert "supporting_evidence" in trace
    assert "posterior_probabilities" in trace
    assert len(trace["posterior_probabilities"]) == 8

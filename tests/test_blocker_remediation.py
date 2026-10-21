"""
Regression and Verification Tests for Module 4 Gate 1 Blocker Remediation
==========================================================================
Verifies:
    1. Direct test record quarantine guards on get_signal(), get_representation(), and get_lead_mask().
    2. Fail-closed manifest loading with zero fallback to unredacted Module 1 manifest.
    3. Synthetic masked-input invariance under deterministic evaluation conditions.
    4. Positive control verifying sensitivity when active/observable leads are perturbed.
    5. Freeze contract source-code snapshot and protocol-content hashing completeness.
    6. Fail-closed behavior on missing or tampered freeze artifacts.
"""

import json
import sys
import tempfile
from pathlib import Path
import pytest
import torch
import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.reasoning.multi_cohort_loader import MultiCohortPartitionManager, LEAD_CONFIG_MASKS
from src.reasoning.reasoning_transformer import ControlReasoningTransformer
from src.reasoning.freeze_contract import (
    build_module4_freeze_contract,
    validate_module4_freeze_contract,
    REASONING_SOURCE_FILES,
    PREREGISTRATION_PROTOCOL_FILES,
    APPROVED_SEEDS,
    compute_sha256,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def synthetic_manifest_with_splits(tmp_path):
    """Creates a strictly synthetic test manifest with train, val, and test splits."""
    data = []
    # 5 Train records
    for i in range(5):
        data.append({
            "global_record_id": f"SYNTH_TR_{i}",
            "split": "train",
            "strat_fold": 1,
            "source_dataset": "SYNTH",
            "source_patient_id": f"P_TR_{i}",
            "source_record_id": i,
            "NORM": 1.0, "AFIB": 0.0, "RBBB": 0.0, "LBBB": 0.0,
            "SBRAD": 0.0, "STACH": 0.0, "1AVB": 0.0, "MI": 0.0,
        })
    # 3 Validation records
    for i in range(3):
        data.append({
            "global_record_id": f"SYNTH_VA_{i}",
            "split": "validation",
            "strat_fold": 8,
            "source_dataset": "SYNTH",
            "source_patient_id": f"P_VA_{i}",
            "source_record_id": 10 + i,
            "NORM": 0.0, "AFIB": 1.0, "RBBB": 0.0, "LBBB": 0.0,
            "SBRAD": 0.0, "STACH": 0.0, "1AVB": 0.0, "MI": 0.0,
        })
    # 3 Quarantined Test records (labels redacted as NaN)
    for i in range(3):
        data.append({
            "global_record_id": f"SYNTH_TE_{i}",
            "split": "test",
            "strat_fold": 9,
            "source_dataset": "SYNTH",
            "source_patient_id": f"P_TE_{i}",
            "source_record_id": 20 + i,
            "NORM": np.nan, "AFIB": np.nan, "RBBB": np.nan, "LBBB": np.nan,
            "SBRAD": np.nan, "STACH": np.nan, "1AVB": np.nan, "MI": np.nan,
        })

    csv_path = tmp_path / "split_manifest.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return csv_path


# -----------------------------------------------------------------------------
# Blocker 1 Tests: Direct Test-Data Access Paths Closed Fail-Closed
# -----------------------------------------------------------------------------

def test_direct_get_signal_quarantine_guard(synthetic_manifest_with_splits):
    """Verifies that get_signal() blocks direct access to test split records."""
    mgr = MultiCohortPartitionManager(
        repo_root_path=repo_root,
        manifest_path=synthetic_manifest_with_splits,
        quarantine_enforced=True,
    )

    # Calling get_signal directly with a test record ID must raise PermissionError
    test_rid = "SYNTH_TE_0"
    with pytest.raises(PermissionError, match="QUARANTINE ENFORCEMENT"):
        mgr.get_signal(test_rid, variant="12-lead")

    with pytest.raises(PermissionError, match="QUARANTINE ENFORCEMENT"):
        mgr.get_signal(test_rid, variant="4-lead")


def test_direct_get_representation_quarantine_guard(synthetic_manifest_with_splits):
    """Verifies that get_representation() blocks direct access to test split records."""
    mgr = MultiCohortPartitionManager(
        repo_root_path=repo_root,
        manifest_path=synthetic_manifest_with_splits,
        quarantine_enforced=True,
    )

    test_rid = "SYNTH_TE_1"
    with pytest.raises(PermissionError, match="QUARANTINE ENFORCEMENT"):
        mgr.get_representation(test_rid, variant="12-lead")

    with pytest.raises(PermissionError, match="QUARANTINE ENFORCEMENT"):
        mgr.get_representation(test_rid, variant="1-lead")


def test_direct_get_lead_mask_quarantine_guard(synthetic_manifest_with_splits):
    """Verifies that get_lead_mask() blocks direct access to test split records."""
    mgr = MultiCohortPartitionManager(
        repo_root_path=repo_root,
        manifest_path=synthetic_manifest_with_splits,
        quarantine_enforced=True,
    )

    test_rid = "SYNTH_TE_2"
    with pytest.raises(PermissionError, match="QUARANTINE ENFORCEMENT"):
        mgr.get_lead_mask(test_rid, variant="6-lead")


def test_missing_manifest_fails_closed_no_fallback(tmp_path):
    """Verifies that if Module 4 manifest is missing, manager fails closed with FileNotFoundError."""
    dummy_root = tmp_path / "empty_repo"
    dummy_root.mkdir(parents=True)

    # Must raise FileNotFoundError, never fallback to Module 1 manifest
    with pytest.raises(FileNotFoundError, match="FAIL-CLOSED"):
        MultiCohortPartitionManager(repo_root_path=dummy_root, manifest_path=None)


def test_production_manifest_test_records_denied():
    """Verifies that direct query for actual test records in Module 4 manifest raises PermissionError."""
    mgr = MultiCohortPartitionManager(repo_root_path=repo_root, quarantine_enforced=True)

    # Query known test record ID from the manifest metadata mapping
    test_rids = mgr._partitions["test"]
    assert len(test_rids) == 7375

    arbitrary_test_rid = test_rids[0]

    with pytest.raises(PermissionError, match="QUARANTINE ENFORCEMENT"):
        mgr.get_signal(arbitrary_test_rid)

    with pytest.raises(PermissionError, match="QUARANTINE ENFORCEMENT"):
        mgr.get_representation(arbitrary_test_rid)

    with pytest.raises(PermissionError, match="QUARANTINE ENFORCEMENT"):
        mgr.get_sample(arbitrary_test_rid)

    with pytest.raises(PermissionError, match="QUARANTINE ENFORCEMENT"):
        mgr.get_lead_mask(arbitrary_test_rid)


# -----------------------------------------------------------------------------
# Blocker 2 Tests: Synthetic Masked-Input Invariance and Positive Control
# -----------------------------------------------------------------------------

def test_masked_input_invariance_and_positive_control():
    """
    Synthetic regression test for masked-input invariance under deterministic evaluation conditions:
    1. Evaluates model on a 4-lead mask (leads 0, 1, 2, 6 active; leads 3, 4, 5, 7, 8, 9, 10, 11 unavailable).
    2. Perturbs values exclusively corresponding to unavailable leads (ST elevation, ST depression, T wave inversion).
    3. Confirms that output logits are bitwise/floating-point identical (diff == 0.0).
    4. Positive Control: Perturbs an ACTIVE lead (Lead I, index 0). Confirms output logits change significantly.
    """
    torch.manual_seed(42)
    model = ControlReasoningTransformer(d_model=128, d_ecg=768, num_candidates=8)
    model.eval()

    # 4-lead mask: I (0), II (1), III (2), V1 (6) active; others unobservable
    mask_4lead = torch.zeros(1, 12, dtype=torch.float32)
    mask_4lead[0, [0, 1, 2, 6]] = 1.0

    z_ecg = torch.randn(1, 768)

    # Baseline: Finding only on active Lead II
    obs_base = {
        "rhythm": "Sinus rhythm",
        "heart_rate": 72.0,
        "pr_interval_ms": 160.0,
        "qrs_duration_ms": 90.0,
        "st_elevation": {"present": True, "leads": ["II"]},
        "st_depression": {"present": False, "leads": []},
        "t_wave_inversion": {"present": False, "leads": []},
    }

    # Perturbed unavailable leads: Inject adversarial findings on unobservable channels
    obs_pert_unavailable = {
        "rhythm": "Sinus rhythm",
        "heart_rate": 72.0,
        "pr_interval_ms": 160.0,
        "qrs_duration_ms": 90.0,
        # aVR, V4, V6 are masked out in 4-lead
        "st_elevation": {"present": True, "leads": ["II", "aVR", "V4", "V6"]},
        # V2, V3 are masked out in 4-lead
        "st_depression": {"present": True, "leads": ["V2", "V3"]},
        # aVL, aVF, V5 are masked out in 4-lead
        "t_wave_inversion": {"present": True, "leads": ["aVL", "aVF", "V5"]},
    }

    # Positive control: Inject finding on an ACTIVE channel (Lead I is active in 4-lead)
    obs_pos_control = {
        "rhythm": "Sinus rhythm",
        "heart_rate": 72.0,
        "pr_interval_ms": 160.0,
        "qrs_duration_ms": 90.0,
        "st_elevation": {"present": True, "leads": ["II", "I"]},
        "st_depression": {"present": False, "leads": []},
        "t_wave_inversion": {"present": False, "leads": []},
    }

    with torch.no_grad():
        out_base = model(z_ecg, [obs_base], mask_4lead)
        out_pert = model(z_ecg, [obs_pert_unavailable], mask_4lead)
        out_pos = model(z_ecg, [obs_pos_control], mask_4lead)

    logits_base = out_base["logits"]
    logits_pert = out_pert["logits"]
    logits_pos = out_pos["logits"]

    max_diff_unavail = (logits_base - logits_pert).abs().max().item()
    max_diff_pos = (logits_base - logits_pos).abs().max().item()

    # Invariance property: Unavailable lead perturbation MUST produce 0.0 logit difference
    assert max_diff_unavail == 0.0, (
        f"Masked-input invariance violated: perturbing unavailable leads altered logits by {max_diff_unavail}"
    )

    # Positive control: Active lead perturbation MUST produce a non-zero shift in logits
    assert max_diff_pos > 1e-4, (
        f"Positive control failed: perturbing active Lead I produced insufficient logit change ({max_diff_pos})"
    )


# -----------------------------------------------------------------------------
# Blocker 3 Tests: Strengthened Freeze Contract Verification
# -----------------------------------------------------------------------------

def test_freeze_contract_source_code_and_protocol_snapshot():
    """
    Verifies that build_module4_freeze_contract hashes all required source code and protocol files,
    and validate_module4_freeze_contract fails closed if any file is missing or altered.
    """
    config_file = repo_root / "configs" / "control_path_training_config.json"
    with open(config_file, "r") as f:
        cfg = json.load(f)

    mock_ckpts = {
        s: {
            "best_checkpoint_path": f"runs/module4_v3/checkpoints/control_reasoning_best_seed{s}.pt",
            "best_checkpoint_sha256": f"mock_best_sha_{s}",
            "final_checkpoint_path": f"runs/module4_v3/checkpoints/control_reasoning_final_seed{s}.pt",
            "final_checkpoint_sha256": f"mock_final_sha_{s}",
            "best_epoch": 10,
            "best_val_loss": 0.12345,
            "temperatures": [1.0] * 8,
            "calibration_metrics": {"mean_temperature": 1.0},
        }
        for s in APPROVED_SEEDS
    }
    mock_val = {s: {"val_loss": 0.12, "macro_auroc": 0.88, "macro_f1": 0.72} for s in APPROVED_SEEDS}
    mock_preds = {s: f"runs/module4_v3/metrics/validation_predictions_seed{s}.csv" for s in APPROVED_SEEDS}

    contract = build_module4_freeze_contract(
        repo_root=repo_root,
        training_config=cfg,
        seed_checkpoints=mock_ckpts,
        validation_metrics_by_seed=mock_val,
        validation_predictions_by_seed=mock_preds,
    )

    # 1. Verify all 7 source files are present and hashed
    assert "source_code_integrity" in contract
    sources = contract["source_code_integrity"]
    for src_key in REASONING_SOURCE_FILES:
        assert src_key in sources
        assert sources[src_key]["status"] == "VERIFIED_PRESENT"
        assert len(sources[src_key]["sha256"]) == 64  # Valid SHA-256

    # 2. Verify all 6 protocol and configuration files are present and hashed
    assert "protocol_integrity" in contract["preregistration"]
    protocols = contract["preregistration"]["protocol_integrity"]
    for proto_key in PREREGISTRATION_PROTOCOL_FILES:
        assert proto_key in protocols
        assert protocols[proto_key]["status"] == "VERIFIED_PRESENT"
        assert len(protocols[proto_key]["sha256"]) == 64

    # 3. Contract validator passes with allow_synthetic=True
    is_valid, errors = validate_module4_freeze_contract(contract, repo_root=repo_root, allow_synthetic=True)
    assert is_valid, f"Validation failed: {errors}"
    assert len(errors) == 0

    # 4. Tampering detection: Alter a source file hash
    tampered_contract = dict(contract)
    tampered_contract["source_code_integrity"] = dict(contract["source_code_integrity"])
    tampered_contract["source_code_integrity"]["training_script"] = {
        "path": "scripts/16_train_reasoning_transformer.py",
        "sha256": "bad_hash_000000000000000000000000000000000000000000000000000000000000",
        "status": "VERIFIED_PRESENT",
    }
    # When repo_root is checked without allow_synthetic, hash mismatch must fail
    is_valid_t, errors_t = validate_module4_freeze_contract(tampered_contract, repo_root=repo_root, allow_synthetic=False)
    assert not is_valid_t
    assert any("Source code hash mismatch" in e for e in errors_t)

    # 5. Missing source file detection
    tampered_missing = dict(contract)
    tampered_missing["source_code_integrity"] = dict(contract["source_code_integrity"])
    del tampered_missing["source_code_integrity"]["context_builder"]
    is_valid_m, errors_m = validate_module4_freeze_contract(tampered_missing, repo_root=repo_root, allow_synthetic=True)
    assert not is_valid_m
    assert any("Missing source code integrity record for 'context_builder'" in e for e in errors_m)

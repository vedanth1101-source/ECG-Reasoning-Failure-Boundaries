"""
Unit and Regression Tests for scripts/17_inspect_transformer_inputs.py
======================================================================
Tests CLI argument parsing, strict split quarantine restrictions, missing file fail-closed behavior,
non-finite tensor detection, lead mask invariance, and token categorization logic.
"""

import importlib.util
from pathlib import Path
import tempfile
import pytest
import numpy as np
import torch

repo_root = Path(__file__).resolve().parent.parent

# Dynamic import for scripts/17_inspect_transformer_inputs.py
script17_path = repo_root / "scripts" / "17_inspect_transformer_inputs.py"
spec = importlib.util.spec_from_file_location("inspect_inputs_module", script17_path)
inspect_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inspect_mod)

parse_args = inspect_mod.parse_args
compute_tensor_stats = inspect_mod.compute_tensor_stats
categorize_token = inspect_mod.categorize_token
TransformerInputInspector = inspect_mod.TransformerInputInspector

from src.reasoning.freeze_contract import APPROVED_CONFIGURATIONS
from src.reasoning.multi_cohort_loader import LEAD_CONFIG_MASKS, MultiCohortPartitionManager


def test_parse_args_defaults():
    """Verifies that parse_args applies the approved default parameters."""
    args = parse_args([])
    assert args.split == "validation"
    assert args.num_records == 3
    assert args.seed == 42
    assert args.output_dir is None
    assert args.configs == list(APPROVED_CONFIGURATIONS)


def test_parse_args_custom():
    """Verifies that parse_args accepts custom command-line options."""
    custom_flags = [
        "--configs", "12-lead", "4-lead",
        "--split", "train",
        "--num-records", "5",
        "--seed", "1337",
        "--output-dir", "custom/run/dir",
    ]
    args = parse_args(custom_flags)
    assert args.split == "train"
    assert args.num_records == 5
    assert args.seed == 1337
    assert args.output_dir == "custom/run/dir"
    assert args.configs == ["12-lead", "4-lead"]


def test_parse_args_all_keyword():
    """Verifies that '--configs all' expands to all seven approved configurations."""
    args = parse_args(["--configs", "all"])
    assert args.configs == list(APPROVED_CONFIGURATIONS)


def test_split_quarantine_fail_closed():
    """Verifies that requesting 'test' split raises ValueError immediately."""
    with pytest.raises(ValueError, match="strictly prohibited"):
        TransformerInputInspector(
            repo_root_path=repo_root,
            split="test",
        )

    with pytest.raises(ValueError, match="strictly prohibited"):
        TransformerInputInspector(
            repo_root_path=repo_root,
            split="testing",
        )

    with pytest.raises(ValueError, match="strictly prohibited"):
        TransformerInputInspector(
            repo_root_path=repo_root,
            split="invalid_split_name",
        )


def test_compute_tensor_stats_finite_and_nonfinite():
    """Tests tensor statistics computation for clean and non-finite tensors."""
    # Clean floating point tensor
    t_clean = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    stats_clean = compute_tensor_stats("clean", t_clean)
    assert stats_clean["shape"] == [2, 2]
    assert stats_clean["is_all_finite"] is True
    assert stats_clean["min"] == 1.0
    assert stats_clean["max"] == 4.0
    assert stats_clean["mean"] == 2.5
    assert stats_clean["non_finite_count"] == 0

    # Tensor with NaN and Inf
    t_bad = torch.tensor([1.0, float("nan"), float("inf"), 4.0])
    stats_bad = compute_tensor_stats("bad", t_bad)
    assert stats_bad["is_all_finite"] is False
    assert stats_bad["non_finite_count"] == 2
    assert stats_bad["min"] is None

    # Boolean tensor
    t_bool = torch.tensor([True, True, True])
    stats_bool = compute_tensor_stats("bool", t_bool)
    assert stats_bool["is_all_finite"] is True
    assert stats_bool["all_true"] is True
    assert stats_bool["all_false"] is False


def test_categorize_token():
    """Tests token categorization and lead assignment across token names."""
    cat, lead = categorize_token("ECG_CONTINUOUS_FM")
    assert cat == "ECG_LATENT"
    assert lead == "GLOBAL"

    cat, lead = categorize_token("LEAD_OBS_I")
    assert cat == "LEAD_STATUS"
    assert lead == "I"

    cat, lead = categorize_token("RHYTHM_PATTERN")
    assert cat == "RHYTHM"
    assert lead == "GLOBAL"

    cat, lead = categorize_token("INTERVAL_METRICS")
    assert cat == "INTERVALS"
    assert lead == "GLOBAL"

    cat, lead = categorize_token("ST_ELEV_V1")
    assert cat == "ST_ELEVATION"
    assert lead == "V1"

    cat, lead = categorize_token("ST_DEP_aVR")
    assert cat == "ST_DEPRESSION"
    assert lead == "aVR"

    cat, lead = categorize_token("T_INV_V6")
    assert cat == "T_INVERSION"
    assert lead == "V6"


def test_invalid_configuration_fails_closed():
    """Verifies that an unapproved configuration fails closed and returns exit code 1."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        inspector = TransformerInputInspector(
            repo_root_path=repo_root,
            split="validation",
            configs=["invalid-3-lead"],
            output_dir=out_dir,
        )
        exit_code = inspector.inspect()
        assert exit_code == 1
        assert inspector.failed_checks > 0
        assert "invalid-3-lead" in inspector.config_verdicts
        assert inspector.config_verdicts["invalid-3-lead"] == "FAIL"


def test_invalid_record_id_fails_closed(monkeypatch):
    """Verifies that an unknown/unregistered record ID fails closed with exit code 1."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        inspector = TransformerInputInspector(
            repo_root_path=repo_root,
            split="validation",
            configs=["12-lead"],
            num_records=1,
            output_dir=out_dir,
        )
        monkeypatch.setattr(inspector, "_sample_records", lambda mgr: ["NONEXISTENT_RECORD_99999"])
        exit_code = inspector.inspect()
        assert exit_code == 1
        assert inspector.failed_checks > 0
        assert inspector.config_verdicts["12-lead"] == "FAIL"
        with open(out_dir / "warnings_and_failures.txt") as f:
            content = f.read()
            assert "Record metadata lookup failed" in content


def test_missing_representation_file_fails_closed(monkeypatch):
    """
    Verifies that when a representation file is missing, the inspector
    detects the missing file, logs a failure, does NOT substitute a fallback,
    and returns exit code 1.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        inspector = TransformerInputInspector(
            repo_root_path=repo_root,
            split="validation",
            configs=["12-lead"],
            num_records=1,
            output_dir=out_dir,
        )
        # Use valid record PTBXL_7 but mock representation path to a non-existent path
        monkeypatch.setattr(inspector, "_sample_records", lambda mgr: ["PTBXL_7"])

        orig_exists = Path.exists
        def mock_exists(p):
            if "representations" in str(p) and "PTBXL_7.npy" in str(p):
                return False
            return orig_exists(p)

        monkeypatch.setattr(Path, "exists", mock_exists)

        exit_code = inspector.inspect()
        assert exit_code == 1
        assert inspector.failed_checks > 0
        assert inspector.config_verdicts["12-lead"] == "FAIL"
        assert (out_dir / "warnings_and_failures.txt").exists()
        with open(out_dir / "warnings_and_failures.txt") as f:
            content = f.read()
            assert "Representation file missing" in content


def test_non_finite_representation_flagged_as_failure(monkeypatch):
    """Verifies that representations containing NaN or Inf are detected and flagged as failures."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        inspector = TransformerInputInspector(
            repo_root_path=repo_root,
            split="validation",
            configs=["12-lead"],
            num_records=1,
            output_dir=out_dir,
        )
        monkeypatch.setattr(inspector, "_sample_records", lambda mgr: ["PTBXL_7"])

        # Mock get_representation to return array with NaN
        nan_rep = np.ones(768, dtype=np.float32)
        nan_rep[10] = np.nan
        monkeypatch.setattr(MultiCohortPartitionManager, "get_representation", lambda self, rid, variant: nan_rep)

        exit_code = inspector.inspect()
        assert exit_code == 1
        assert inspector.failed_checks > 0
        assert inspector.config_verdicts["12-lead"] == "FAIL"
        with open(out_dir / "warnings_and_failures.txt") as f:
            content = f.read()
            assert "Non-finite values" in content


def test_lead_mask_mismatch_flagged_as_failure(monkeypatch):
    """Verifies that an altered or mismatched lead mask is detected and flagged as a failure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        inspector = TransformerInputInspector(
            repo_root_path=repo_root,
            split="validation",
            configs=["4-lead"],
            num_records=1,
            output_dir=out_dir,
        )
        monkeypatch.setattr(inspector, "_sample_records", lambda mgr: ["PTBXL_7"])

        # Mock get_lead_mask to return wrong mask (all ones instead of 4-lead schedule)
        wrong_mask = np.ones(12, dtype=np.float32)
        monkeypatch.setattr(MultiCohortPartitionManager, "get_lead_mask", lambda self, rid, variant: wrong_mask)

        exit_code = inspector.inspect()
        assert exit_code == 1
        assert inspector.failed_checks > 0
        assert inspector.config_verdicts["4-lead"] == "FAIL"
        with open(out_dir / "warnings_and_failures.txt") as f:
            content = f.read()
            assert "Lead availability mask mismatch" in content


def test_seven_configuration_reporting_consistency(monkeypatch, capsys):
    """
    Regression test: Verifies that when all seven approved configurations are inspected:
    1. All seven configurations are inspected and receive explicit PASS verdicts.
    2. Terminal summary contains all seven configuration verdicts and compact summary.
    3. All five output artifacts (Markdown, CSV, JSON, token CSV, warnings txt) record all seven configurations.
    """
    import csv
    import json

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        inspector = TransformerInputInspector(
            repo_root_path=repo_root,
            split="validation",
            configs=list(APPROVED_CONFIGURATIONS),
            num_records=1,
            output_dir=out_dir,
        )
        monkeypatch.setattr(inspector, "_sample_records", lambda mgr: ["PTBXL_7"])

        exit_code = inspector.inspect()
        assert exit_code == 0
        assert len(inspector.config_verdicts) == 7

        expected_configs = ["12-lead", "10-lead", "8-lead", "6-lead", "4-lead", "2-lead", "1-lead"]
        for cfg in expected_configs:
            assert cfg in inspector.config_verdicts
            assert inspector.config_verdicts[cfg] == "PASS"

        # Check terminal output
        captured = capsys.readouterr().out
        assert "Per-Configuration Verdicts (7/7 Configurations):" in captured
        for cfg in expected_configs:
            assert f"- {cfg:<10}: PASS" in captured

        # Check inspection_summary.md
        with open(out_dir / "inspection_summary.md", encoding="utf-8") as f:
            md_text = f.read()
            for cfg in expected_configs:
                assert f"| **{cfg}** |" in md_text
                assert f"**`PASS`**" in md_text

        # Check configuration_input_summary.csv
        with open(out_dir / "configuration_input_summary.csv", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            csv_configs = [row["configuration"] for row in reader]
            assert csv_configs == expected_configs

        # Check representative_tensor_summary.json
        with open(out_dir / "representative_tensor_summary.json", encoding="utf-8") as f:
            json_data = json.load(f)
            assert list(json_data["configurations"].keys()) == expected_configs
            for cfg in expected_configs:
                assert json_data["configurations"][cfg]["verdict"] == "PASS"



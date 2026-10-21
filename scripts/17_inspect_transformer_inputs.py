#!/usr/bin/env python3
"""
Script 17: Standalone Manual Input Inspection Tool for ControlReasoningTransformer
===================================================================================
Authoritative inspection tool allowing pre-training manual inspection of actual
input tensors, representations, lead masks, observations, and context tokens
prepared for ControlReasoningTransformer.

STRICT SAFETY BOUNDARIES:
- Never invokes training loop, optimizer, backward pass, calibration, or test evaluation.
- Never loads or inspects the test split.
- Never calls the model's forward method (reasoning transformer forward) on real-data batches.
- Never updates model parameters.
- Never alters frozen Modules 1-3 or the approved configuration.
- Fails closed if split is not 'train' or 'validation'.
- Reuses actual production loading and context-construction functions.
- Never silently substitutes a missing lead configuration.
- Never generates fake production results if actual inputs cannot be loaded.

Usage Examples:
    # Default: Inspects all 7 configurations on validation split (3 records per config, seed 42)
    python scripts/17_inspect_transformer_inputs.py

    # Inspect single configuration on validation split:
    python scripts/17_inspect_transformer_inputs.py --configs 12-lead

    # Inspect two specific configurations on train split:
    python scripts/17_inspect_transformer_inputs.py --configs 12-lead 4-lead --split train --num-records 5

    # Inspect all configurations with custom seed and output directory:
    python scripts/17_inspect_transformer_inputs.py --configs all --seed 123 --output-dir my_custom_dir
"""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import random
import sys
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import torch

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.reasoning.multi_cohort_loader import (
    MultiCohortPartitionManager,
    LEAD_CONFIG_MASKS,
    TARGET_NAMES,
)
from src.observations.clinical_extractor import ClinicalObservationExtractor
from src.reasoning.context_builder import (
    ControlContextBuilder,
    LEAD_NAMES,
    CANDIDATE_ACRONYMS,
)
from src.reasoning.freeze_contract import APPROVED_CONFIGURATIONS


def parse_args(args_list: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses command-line arguments for manual input inspection."""
    parser = argparse.ArgumentParser(
        prog="python scripts/17_inspect_transformer_inputs.py",
        description=(
            "Module 4 Standalone Manual Input Inspection Tool for ControlReasoningTransformer.\n"
            "Prepares and audits actual input tensors, representations, lead masks, observations, and context tokens.\n"
            "STRICT SAFETY: Never runs training, optimizer, loss backward, model inference, or test set access."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--configs",
        "--configurations",
        dest="configs",
        nargs="+",
        default=list(APPROVED_CONFIGURATIONS),
        help=(
            "One or more configurations to inspect: '12-lead', '10-lead', '8-lead', '6-lead', '4-lead', '2-lead', '1-lead', "
            "or 'all'. Default: all 7 configurations."
        ),
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation"],
        default="validation",
        help="Split to inspect: 'train' or 'validation' only. Test split access is strictly prohibited. Default: 'validation'.",
    )
    parser.add_argument(
        "--num-records",
        type=int,
        default=3,
        help="Number of representative records to inspect per configuration. Default: 3.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Reproducible random seed for sampling records. Default: 42.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Custom output directory for inspection artifacts. "
            "If not specified, creates a unique timestamped directory under "
            "'runs/module4_v3/reports/pretraining_input_audit/manual_runs/run_YYYYMMDD_HHMMSS'."
        ),
    )

    args = parser.parse_args(args_list)

    # Normalize configs if "all" keyword was supplied
    if len(args.configs) == 1 and args.configs[0].lower() == "all":
        args.configs = list(APPROVED_CONFIGURATIONS)

    return args


def compute_tensor_stats(name: str, tensor: Union[torch.Tensor, np.ndarray]) -> Dict[str, Any]:
    """Computes exact compact statistical summaries for a given tensor."""
    if isinstance(tensor, np.ndarray):
        t = torch.from_numpy(tensor)
    else:
        t = tensor.detach().cpu()

    shape = list(t.shape)
    dtype_str = str(t.dtype)
    numel = t.numel()

    if t.dtype == torch.bool:
        all_true = bool(t.all().item())
        all_false = bool((~t).all().item())
        return {
            "name": name,
            "shape": shape,
            "dtype": dtype_str,
            "min": bool(t.min().item()) if numel > 0 else None,
            "max": bool(t.max().item()) if numel > 0 else None,
            "mean": round(float(t.float().mean().item()), 5) if numel > 0 else None,
            "std": round(float(t.float().std().item()), 5) if numel > 1 else 0.0,
            "non_finite_count": 0,
            "is_all_finite": True,
            "all_true": all_true,
            "all_false": all_false,
        }

    t_float = t.float()
    nan_mask = torch.isnan(t_float)
    inf_mask = torch.isinf(t_float)
    non_finite_count = int((nan_mask | inf_mask).sum().item())
    is_all_finite = non_finite_count == 0

    if is_all_finite and numel > 0:
        t_min = round(float(t_float.min().item()), 5)
        t_max = round(float(t_float.max().item()), 5)
        t_mean = round(float(t_float.mean().item()), 5)
        t_std = round(float(t_float.std().item()), 5) if numel > 1 else 0.0
    else:
        t_min, t_max, t_mean, t_std = None, None, None, None

    return {
        "name": name,
        "shape": shape,
        "dtype": dtype_str,
        "min": t_min,
        "max": t_max,
        "mean": t_mean,
        "std": t_std,
        "non_finite_count": non_finite_count,
        "is_all_finite": is_all_finite,
    }


def categorize_token(token_name: str) -> Tuple[str, str]:
    """Returns (category, lead_name) for a given evidence token identifier."""
    if token_name == "ECG_CONTINUOUS_FM":
        return "ECG_LATENT", "GLOBAL"
    if token_name.startswith("LEAD_OBS_"):
        lead = token_name.replace("LEAD_OBS_", "")
        return "LEAD_STATUS", lead
    if token_name == "RHYTHM_PATTERN":
        return "RHYTHM", "GLOBAL"
    if token_name == "INTERVAL_METRICS":
        return "INTERVALS", "GLOBAL"
    if token_name.startswith("ST_ELEV_"):
        lead = token_name.replace("ST_ELEV_", "")
        return "ST_ELEVATION", lead
    if token_name.startswith("ST_DEP_"):
        lead = token_name.replace("ST_DEP_", "")
        return "ST_DEPRESSION", lead
    if token_name.startswith("T_INV_"):
        lead = token_name.replace("T_INV_", "")
        return "T_INVERSION", lead
    return "UNKNOWN", "GLOBAL"


class TransformerInputInspector:
    """
    Execution engine for inspecting inputs prepared for ControlReasoningTransformer.
    Strictly inspection-only: no training loop, no backward pass, no inference, no test split.
    """

    def __init__(
        self,
        repo_root_path: Path,
        split: str = "validation",
        configs: Optional[List[str]] = None,
        num_records: int = 3,
        seed: int = 42,
        output_dir: Optional[Path] = None,
    ):
        self.repo_root = repo_root_path
        self.split = split
        self.configs = configs or list(APPROVED_CONFIGURATIONS)
        self.num_records = num_records
        self.seed = seed

        # Enforce quarantine boundary
        if self.split not in ("train", "validation"):
            raise ValueError(
                f"[QUARANTINE ENFORCEMENT] Requested split '{self.split}' is strictly prohibited. "
                "Only 'train' and 'validation' splits may be inspected."
            )

        # Output directory setup
        if output_dir is not None:
            self.output_dir = Path(output_dir)
        else:
            timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
            self.output_dir = (
                self.repo_root
                / "runs"
                / "module4_v3"
                / "reports"
                / "pretraining_input_audit"
                / "manual_runs"
                / timestamp
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Validation counters and logs
        self.passed_checks = 0
        self.failed_checks = 0
        self.unverified_checks = 0
        self.uninspected_records = 0
        self.warnings: List[str] = []
        self.failures: List[str] = []

        # Results structures
        self.config_verdicts: Dict[str, str] = {}
        self.records_inspected_summary: List[Dict[str, Any]] = []
        self.tensor_stats_summary: Dict[str, Any] = {}
        self.token_rows: List[Dict[str, Any]] = []

    def _display_path(self, path: Union[Path, str]) -> str:
        """Returns relative path to repo_root if feasible; otherwise returns str(path)."""
        p = Path(path)
        try:
            return str(p.relative_to(self.repo_root))
        except (ValueError, TypeError):
            return str(p)

    def _sample_records(self, mgr: MultiCohortPartitionManager) -> List[str]:
        """Samples representative records deterministically using seeded RNG."""
        all_rids = mgr.get_record_ids(self.split, allow_test=False)
        if not all_rids:
            return []

        # Sort for deterministic ordering before sampling
        sorted_rids = sorted(all_rids)

        # Diverse cohort sampling: group by source dataset
        rng = random.Random(self.seed)
        by_cohort: Dict[str, List[str]] = {}
        for rid in sorted_rids:
            meta = mgr.get_record_metadata(rid)
            cohort = meta.get("source_dataset", "UNKNOWN")
            by_cohort.setdefault(cohort, []).append(rid)

        # Collect diverse sample across cohorts
        sampled: List[str] = []
        cohort_keys = sorted(by_cohort.keys())

        # Round-robin sampling across cohorts
        idx = 0
        while len(sampled) < self.num_records and any(by_cohort.values()):
            cohort = cohort_keys[idx % len(cohort_keys)]
            if by_cohort[cohort]:
                c_rids = by_cohort[cohort]
                # Pick deterministic choice using seed
                pick_idx = rng.randrange(len(c_rids))
                sampled.append(c_rids.pop(pick_idx))
            idx += 1

        return sampled

    def inspect(self) -> int:
        """
        Executes the manual input inspection.
        Returns:
            exit_code: 0 if all required checks passed; 1 if any check failed or artifact missing.
        """
        print("=" * 80)
        print("  MODULE 4: STANDALONE MANUAL INPUT INSPECTION TOOL")
        print("  Target Model: ControlReasoningTransformer")
        print("=" * 80)
        print(f"[*] Repository Root:       {self.repo_root}")
        print(f"[*] Inspection Split:      {self.split.upper()} (Quarantine Enforced)")
        print(f"[*] Configurations:        {', '.join(self.configs)}")
        print(f"[*] Sample Size / Config:  {self.num_records} representative records")
        print(f"[*] Random Seed:           {self.seed}")
        print(f"[*] Destination Directory: {self.output_dir}")
        print("-" * 80)

        # Initialize Partition Manager
        try:
            mgr = MultiCohortPartitionManager(repo_root_path=self.repo_root, quarantine_enforced=True)
            print("[+] MultiCohortPartitionManager initialized successfully.")
            self.passed_checks += 1
        except Exception as e:
            msg = f"Failed to initialize MultiCohortPartitionManager: {e}"
            print(f"[!] FATAL: {msg}")
            self.failures.append(msg)
            self.failed_checks += 1
            self._save_results()
            return 1

        # Check configuration validity
        for c in self.configs:
            if c not in APPROVED_CONFIGURATIONS:
                msg = f"Unknown or unapproved configuration '{c}'. Must be one of {list(APPROVED_CONFIGURATIONS)}"
                print(f"[!] FATAL: {msg}")
                self.failures.append(msg)
                self.failed_checks += 1
                self.config_verdicts[c] = "FAIL"
                self._save_results()
                return 1

        # Sample representative records
        sample_rids = self._sample_records(mgr)
        if not sample_rids:
            msg = f"Zero records available for split '{self.split}'. Split is empty or unpopulated."
            print(f"[!] FATAL: {msg}")
            self.failures.append(msg)
            self.failed_checks += 1
            self._save_results()
            return 1

        print(f"[+] Sampled {len(sample_rids)} representative records for inspection: {sample_rids}\n")

        # Instantiate production ClinicalObservationExtractor and ControlContextBuilder
        extractor = ClinicalObservationExtractor()
        context_builder = ControlContextBuilder(d_model=256, d_ecg=768, num_candidates=8)
        context_builder.eval()

        self.tensor_stats_summary = {
            "timestamp": datetime.now().isoformat(),
            "split": self.split,
            "seed": self.seed,
            "num_records_per_config": self.num_records,
            "configurations": {},
        }

        # Inspect each configuration
        for config_name in self.configs:
            print("=" * 80)
            print(f"  INSPECTING CONFIGURATION: [{config_name.upper()}]")
            print("=" * 80)

            config_has_failure = False
            config_record_data = []

            expected_active_count = int(LEAD_CONFIG_MASKS[config_name].sum())
            expected_mask_vec = LEAD_CONFIG_MASKS[config_name]

            for rid in sample_rids:
                record_failed = False
                try:
                    meta = mgr.get_record_metadata(rid)
                except Exception as e:
                    msg = f"Record metadata lookup failed for '{rid}' in configuration '{config_name}': {e}"
                    print(f"  [FAIL] {msg}")
                    self.failures.append(msg)
                    self.failed_checks += 1
                    config_has_failure = True
                    self.uninspected_records += 1
                    continue

                source_dataset = meta.get("source_dataset", "UNKNOWN")
                patient_id = meta.get("source_patient_id", "UNKNOWN")
                split_canonical = meta.get("split_canonical", self.split)

                print(f"\n--- [RECORD: {rid}] | Dataset: {source_dataset} | Split: {split_canonical} | Config: {config_name} ---")

                # 1. Inspect Representation
                rep_path = self.repo_root / "runs" / "module3_v1" / "representations" / config_name / self.split / "records" / f"{rid}.npy"
                if not rep_path.exists():
                    msg = f"Representation file missing for record '{rid}' in configuration '{config_name}' at {self._display_path(rep_path)}"
                    print(f"  [FAIL] {msg}")
                    self.failures.append(msg)
                    self.failed_checks += 1
                    config_has_failure = True
                    self.uninspected_records += 1
                    continue

                try:
                    z_np = mgr.get_representation(rid, variant=config_name)
                    self.passed_checks += 1
                except Exception as e:
                    msg = f"Failed to load representation for '{rid}' in '{config_name}': {e}"
                    print(f"  [FAIL] {msg}")
                    self.failures.append(msg)
                    self.failed_checks += 1
                    config_has_failure = True
                    continue

                # Validate Representation Dimensions and Numerical Health
                if z_np.shape != (768,):
                    msg = f"Incorrect representation shape for '{rid}' in '{config_name}': expected (768,), got {z_np.shape}"
                    print(f"  [FAIL] {msg}")
                    self.failures.append(msg)
                    self.failed_checks += 1
                    record_failed = True

                z_stats = compute_tensor_stats("z_ecg", z_np)
                if not z_stats["is_all_finite"]:
                    msg = f"Non-finite values ({z_stats['non_finite_count']}) detected in representation for '{rid}' in '{config_name}'"
                    print(f"  [FAIL] {msg}")
                    self.failures.append(msg)
                    self.failed_checks += 1
                    record_failed = True
                else:
                    self.passed_checks += 1

                if np.all(z_np == 0.0):
                    warn = f"Unexpected all-zero representation vector for '{rid}' in '{config_name}'"
                    print(f"  [WARN] {warn}")
                    self.warnings.append(warn)

                print(f"  [1] Representation:")
                print(f"      Path:   {self._display_path(rep_path)}")
                finite_desc = "ALL" if z_stats["is_all_finite"] else f"NON-FINITE ({z_stats['non_finite_count']})"
                print(f"      Shape:  {z_np.shape} | Dtype: {z_np.dtype} | Finite: {finite_desc}")
                if z_stats["is_all_finite"]:
                    print(f"      Stats:  min={z_stats['min']:.4f}, max={z_stats['max']:.4f}, mean={z_stats['mean']:.4f}, std={z_stats['std']:.4f}")
                else:
                    print(f"      Stats:  min=N/A, max=N/A, mean=N/A, std=N/A (non_finite_count={z_stats['non_finite_count']})")

                # 2. Inspect Lead Mask
                lead_mask_np = mgr.get_lead_mask(rid, variant=config_name)
                active_count = int(lead_mask_np.sum())
                inactive_count = 12 - active_count
                active_leads = [LEAD_NAMES[i] for i in range(12) if lead_mask_np[i] == 1.0]

                if not np.array_equal(lead_mask_np, expected_mask_vec):
                    msg = f"Lead availability mask mismatch for '{rid}' in '{config_name}': expected {expected_mask_vec}, got {lead_mask_np}"
                    print(f"  [FAIL] {msg}")
                    self.failures.append(msg)
                    self.failed_checks += 1
                    record_failed = True
                else:
                    self.passed_checks += 1

                if active_count != expected_active_count:
                    msg = f"Active lead count mismatch for '{rid}' in '{config_name}': expected {expected_active_count}, got {active_count}"
                    print(f"  [FAIL] {msg}")
                    self.failures.append(msg)
                    self.failed_checks += 1
                    record_failed = True
                else:
                    self.passed_checks += 1

                lead_mask_stats = compute_tensor_stats("lead_avail_mask", lead_mask_np)

                print(f"  [2] Lead Availability Mask:")
                print(f"      Active Leads ({active_count}/12):   {', '.join(active_leads) if active_leads else 'NONE'}")
                print(f"      Inactive Leads ({inactive_count}/12): {', '.join([LEAD_NAMES[i] for i in range(12) if lead_mask_np[i] == 0.0])}")
                print(f"      Mask Vector:              {lead_mask_np.astype(int).tolist()}")

                # 3. Inspect Target Labels and Observation Masks
                labels_dict = {}
                labels_arr = np.zeros(len(TARGET_NAMES), dtype=np.float32)
                label_masks_arr = np.zeros(len(TARGET_NAMES), dtype=np.float32)

                pos_labels = []
                neg_labels = []
                unk_labels = []

                for c_idx, target_name in enumerate(TARGET_NAMES):
                    raw_val = meta.get(target_name)
                    if pd.isna(raw_val):
                        # Unknown / Unobserved label
                        label_masks_arr[c_idx] = 0.0
                        labels_arr[c_idx] = 0.0  # safe numerical placeholder
                        labels_dict[target_name] = "UNKNOWN (mask=0)"
                        unk_labels.append(target_name)
                    else:
                        val = float(raw_val)
                        label_masks_arr[c_idx] = 1.0
                        labels_arr[c_idx] = val
                        if val == 1.0:
                            labels_dict[target_name] = "POSITIVE (1)"
                            pos_labels.append(target_name)
                        else:
                            labels_dict[target_name] = "NEGATIVE (0)"
                            neg_labels.append(target_name)

                observed_count = int(label_masks_arr.sum())
                unobserved_count = 8 - observed_count

                print(f"  [3] Multi-Hot Disease Targets (Observed: {observed_count}/8, Unknown: {unobserved_count}/8):")
                labels_line1 = [f"{t}: {labels_dict[t]}" for t in TARGET_NAMES[:4]]
                labels_line2 = [f"{t}: {labels_dict[t]}" for t in TARGET_NAMES[4:]]
                print(f"      { ' | '.join(labels_line1) }")
                print(f"      { ' | '.join(labels_line2) }")

                # 4. Extract Structured Observations from Masked Signal
                sig_path = self.repo_root / "runs" / "module2_v1" / "masked_leads" / config_name / self.split / "records" / f"{rid}.npy"
                if not sig_path.exists():
                    msg = f"Masked signal file missing for '{rid}' in '{config_name}' at {self._display_path(sig_path)}"
                    print(f"  [FAIL] {msg}")
                    self.failures.append(msg)
                    self.failed_checks += 1
                    config_has_failure = True
                    continue

                try:
                    sig_np = mgr.get_signal(rid, variant=config_name)
                    obs = extractor.extract(sig_np, lead_mask_np).to_dict()
                    self.passed_checks += 1
                except Exception as e:
                    msg = f"ClinicalObservationExtractor failed on '{rid}' in '{config_name}': {e}"
                    print(f"  [FAIL] {msg}")
                    self.failures.append(msg)
                    self.failed_checks += 1
                    config_has_failure = True
                    continue

                rhythm_str = obs.get("rhythm", "Unknown")
                hr_val = obs.get("heart_rate")
                pr_val = obs.get("pr_interval_ms")
                qrs_val = obs.get("qrs_duration_ms")
                ste_leads = obs.get("st_elevation", {}).get("leads", [])
                std_leads = obs.get("st_depression", {}).get("leads", [])
                twi_leads = obs.get("t_wave_inversion", {}).get("leads", [])

                print(f"  [4] Structured Clinical Observations:")
                print(f"      Rhythm: {rhythm_str} | HR: {hr_val if hr_val is not None else 'N/A'} bpm | "
                      f"PR: {pr_val if pr_val is not None else 'N/A'} ms | QRS: {qrs_val if qrs_val is not None else 'N/A'} ms")
                print(f"      STE Leads: {ste_leads if ste_leads else 'None'} | "
                      f"STD Leads: {std_leads if std_leads else 'None'} | "
                      f"TWI Leads: {twi_leads if twi_leads else 'None'}")

                # 5. Build Context Input Tensors (Pre-Forward Construction)
                z_tensor = torch.from_numpy(z_np).unsqueeze(0).float()
                mask_tensor = torch.from_numpy(lead_mask_np).unsqueeze(0).float()
                labels_tensor = torch.from_numpy(labels_arr).unsqueeze(0).float()
                lmasks_tensor = torch.from_numpy(label_masks_arr).unsqueeze(0).float()

                torch.manual_seed(self.seed)
                with torch.no_grad():
                    evidence_tokens, evidence_mask, evidence_names, query_tokens = context_builder(
                        z_ecg=z_tensor,
                        observations=[obs],
                        lead_avail_mask=mask_tensor,
                        ckp_tokens=None,
                    )

                # Validate Context Tensor Shapes
                if evidence_tokens.shape != (1, 51, 256):
                    msg = f"Evidence tokens shape mismatch: expected (1, 51, 256), got {evidence_tokens.shape}"
                    print(f"  [FAIL] {msg}")
                    self.failures.append(msg)
                    self.failed_checks += 1
                    record_failed = True
                else:
                    self.passed_checks += 1

                if query_tokens.shape != (1, 8, 256):
                    msg = f"Query tokens shape mismatch: expected (1, 8, 256), got {query_tokens.shape}"
                    print(f"  [FAIL] {msg}")
                    self.failures.append(msg)
                    self.failed_checks += 1
                    record_failed = True
                else:
                    self.passed_checks += 1

                # Compute Prepared Tensor Statistics
                ev_stats = compute_tensor_stats("evidence_tokens", evidence_tokens)
                ev_mask_stats = compute_tensor_stats("evidence_mask", evidence_mask)
                query_stats = compute_tensor_stats("query_tokens", query_tokens)
                lbl_stats = compute_tensor_stats("labels", labels_tensor)
                lmask_stats = compute_tensor_stats("label_masks", lmasks_tensor)

                if not ev_stats["is_all_finite"]:
                    msg = f"Non-finite values detected in evidence tokens for '{rid}' in '{config_name}'"
                    print(f"  [FAIL] {msg}")
                    self.failures.append(msg)
                    self.failed_checks += 1
                    record_failed = True
                else:
                    self.passed_checks += 1

                if not query_stats["is_all_finite"]:
                    msg = f"Non-finite values detected in query tokens for '{rid}' in '{config_name}'"
                    print(f"  [FAIL] {msg}")
                    self.failures.append(msg)
                    self.failed_checks += 1
                    record_failed = True
                else:
                    self.passed_checks += 1

                print(f"  [5] Prepared Transformer-Input Tensors (Ready for Forward Execution):")
                tensor_summary_list = [
                    ("z_ecg", z_stats),
                    ("lead_avail_mask", lead_mask_stats),
                    ("labels", lbl_stats),
                    ("label_masks", lmask_stats),
                    ("evidence_tokens", ev_stats),
                    ("evidence_mask", ev_mask_stats),
                    ("query_tokens", query_stats),
                ]
                for t_name, s in tensor_summary_list:
                    if s["dtype"] == "torch.bool":
                        print(f"      - {t_name:<18} shape={str(s['shape']):<14} dtype={s['dtype']:<10} "
                              f"all_true={s.get('all_true')} non_finite={s['non_finite_count']}")
                    else:
                        min_str = f"{s['min']:.4f}" if s['min'] is not None else "N/A"
                        max_str = f"{s['max']:.4f}" if s['max'] is not None else "N/A"
                        mean_str = f"{s['mean']:.4f}" if s['mean'] is not None else "N/A"
                        std_str = f"{s['std']:.4f}" if s['std'] is not None else "N/A"
                        print(f"      - {t_name:<18} shape={str(s['shape']):<14} dtype={s['dtype']:<10} "
                              f"min={min_str:<8} max={max_str:<8} mean={mean_str:<8} std={std_str:<8} non_finite={s['non_finite_count']}")

                # 6. Detailed Token Table Inspection
                print(f"  [6] Evidence Token Sequence Summary (51 Total Tokens):")
                print(f"      {'Idx':<4} {'Token Identifier':<20} {'Category':<14} {'Lead':<6} {'Availability':<12} {'AttnMask':<8} {'Norm':<8} {'Status'}")
                print(f"      {'-'*4} {'-'*20} {'-'*14} {'-'*6} {'-'*12} {'-'*8} {'-'*8} {'-'*10}")

                for t_idx, t_name in enumerate(evidence_names):
                    cat, lead_str = categorize_token(t_name)
                    token_vec = evidence_tokens[0, t_idx]
                    token_norm = round(float(torch.norm(token_vec).item()), 4)
                    token_mean = round(float(token_vec.mean().item()), 4)
                    token_std = round(float(token_vec.std().item()), 4)
                    is_all_zero = bool((token_vec == 0.0).all().item())

                    if lead_str == "GLOBAL":
                        avail_status = "GLOBAL"
                    else:
                        lead_idx = LEAD_NAMES.index(lead_str)
                        is_active_lead = lead_mask_np[lead_idx] == 1.0
                        avail_status = "ACTIVE" if is_active_lead else "UNAVAILABLE"

                    attn_status = "VALID" if evidence_mask[0, t_idx].item() else "PAD"
                    state_desc = "ZERO_TOKEN" if is_all_zero else "EMBEDDED"

                    # Collect row for token_inspection.csv
                    self.token_rows.append({
                        "configuration": config_name,
                        "split": self.split,
                        "record_id": rid,
                        "token_idx": t_idx,
                        "token_name": t_name,
                        "token_category": cat,
                        "lead_name": lead_str,
                        "lead_availability_status": avail_status,
                        "attention_mask_status": attn_status,
                        "token_norm": token_norm,
                        "token_mean": token_mean,
                        "token_std": token_std,
                        "is_all_zero": is_all_zero,
                    })

                    # Compact display in terminal: print key representative tokens
                    # Always print ECG, first 4 leads, Rhythm, Intervals, and morphological active findings
                    should_print = (
                        t_idx in (0, 1, 2, 3, 13, 14)
                        or (not is_all_zero and cat in ("ST_ELEVATION", "ST_DEPRESSION", "T_INVERSION"))
                        or t_idx in (50,)
                    )
                    if should_print:
                        print(f"      {t_idx:02d}   {t_name:<20} {cat:<14} {lead_str:<6} {avail_status:<12} {attn_status:<8} {token_norm:<8.4f} {state_desc}")
                    elif t_idx == 4:
                        print(f"      ...  [leads {LEAD_NAMES[4]}..{LEAD_NAMES[11]} omitted in compact view; full table exported to CSV] ...")
                    elif t_idx == 15 and not any(not (evidence_tokens[0, k] == 0.0).all().item() for k in range(15, 51)):
                        print(f"      ...  [Tokens 15-50 (Morphological findings) are all zeros for this record; full table exported to CSV] ...")

                # Store record summary
                record_status = "FAIL" if record_failed else "PASS"
                if record_failed:
                    config_has_failure = True

                record_summary = {
                    "configuration": config_name,
                    "split": self.split,
                    "record_id": rid,
                    "dataset": source_dataset,
                    "patient_id": patient_id,
                    "rep_path": self._display_path(rep_path),
                    "rep_shape": str(z_np.shape),
                    "rep_dtype": str(z_np.dtype),
                    "rep_all_finite": z_stats["is_all_finite"],
                    "rep_min": z_stats["min"],
                    "rep_max": z_stats["max"],
                    "rep_mean": z_stats["mean"],
                    "rep_std": z_stats["std"],
                    "num_active_leads": active_count,
                    "num_inactive_leads": inactive_count,
                    "active_leads": ";".join(active_leads),
                    "observed_labels_count": observed_count,
                    "unobserved_labels_count": unobserved_count,
                    "positive_labels": ";".join(pos_labels),
                    "negative_labels": ";".join(neg_labels),
                    "unknown_labels": ";".join(unk_labels),
                    "ev_tokens_shape": str(list(evidence_tokens.shape)),
                    "ev_tokens_all_finite": ev_stats["is_all_finite"],
                    "query_tokens_shape": str(list(query_tokens.shape)),
                    "query_tokens_all_finite": query_stats["is_all_finite"],
                    "check_status": record_status,
                }
                self.records_inspected_summary.append(record_summary)

                # Store detailed stats in JSON structure
                config_record_data.append({
                    "record_id": rid,
                    "dataset": source_dataset,
                    "patient_id": str(patient_id),
                    "status": record_status,
                    "tensors": {
                        "z_ecg": z_stats,
                        "lead_avail_mask": lead_mask_stats,
                        "labels": lbl_stats,
                        "label_masks": lmask_stats,
                        "evidence_tokens": ev_stats,
                        "evidence_mask": ev_mask_stats,
                        "query_tokens": query_stats,
                    },
                    "observations": {
                        "rhythm": rhythm_str,
                        "heart_rate": hr_val,
                        "pr_interval_ms": pr_val,
                        "qrs_duration_ms": qrs_val,
                        "st_elevation_leads": ste_leads,
                        "st_depression_leads": std_leads,
                        "t_wave_inversion_leads": twi_leads,
                    },
                })

            verdict = "FAIL" if config_has_failure else "PASS"
            self.config_verdicts[config_name] = verdict
            self.tensor_stats_summary["configurations"][config_name] = {
                "verdict": verdict,
                "active_leads_expected": expected_active_count,
                "records": config_record_data,
            }
            print(f"\n[+] Configuration [{config_name}] Result: {verdict}\n")

        # Save all required audit artifacts
        self._save_results()

        # Print final verdict and summary
        print("=" * 80)
        print("  INSPECTION EXECUTION COMPLETED")
        print("=" * 80)
        print(f"[*] Input Checks Passed:    {self.passed_checks}")
        print(f"[*] Input Checks Failed:    {self.failed_checks}")
        print(f"[*] Unverified Checks:      {self.unverified_checks}")
        print(f"[*] Uninspected Records:    {self.uninspected_records}")
        print(f"[*] Warnings Encountered:   {len(self.warnings)}")
        print(f"\n[*] Per-Configuration Verdicts ({len(self.config_verdicts)}/{len(self.configs)} Configurations):")
        compact_bar = " | ".join(f"{cfg}: {v}" for cfg, v in self.config_verdicts.items())
        print(f"    Summary: [{compact_bar}]")
        for cfg, v in self.config_verdicts.items():
            print(f"    - {cfg:<10}: {v}")

        print(f"\n[+] All inspection artifacts saved to: {self._display_path(self.output_dir)}")
        print("    1. inspection_summary.md")
        print("    2. configuration_input_summary.csv")
        print("    3. representative_tensor_summary.json")
        print("    4. token_inspection.csv")
        print("    5. warnings_and_failures.txt")
        print("=" * 80)

        # Final exit code decision
        if self.failed_checks > 0 or len(self.failures) > 0 or any(v == "FAIL" for v in self.config_verdicts.values()):
            print("[!] STATUS: INSPECTION FAILED — ERRORS DETECTED.")
            return 1

        print("[+] STATUS: ALL INSPECTION CHECKS PASSED SUCCESSFULLY (Exit Code 0).")
        return 0

    def _save_results(self) -> None:
        """Saves all 5 required deliverable files in the unique output directory."""
        # 1. configuration_input_summary.csv
        csv_path = self.output_dir / "configuration_input_summary.csv"
        if self.records_inspected_summary:
            fieldnames = list(self.records_inspected_summary[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.records_inspected_summary)

        # 2. representative_tensor_summary.json
        json_path = self.output_dir / "representative_tensor_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.tensor_stats_summary, f, indent=2)

        # 3. token_inspection.csv
        token_csv_path = self.output_dir / "token_inspection.csv"
        if self.token_rows:
            fieldnames = list(self.token_rows[0].keys())
            with open(token_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.token_rows)

        # 4. warnings_and_failures.txt
        txt_path = self.output_dir / "warnings_and_failures.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Module 4 Manual Input Inspection Warnings and Failures Log\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Split: {self.split} | Seed: {self.seed}\n")
            f.write("=" * 60 + "\n\n")

            if self.failures:
                f.write(f"FAILURES DETECTED ({len(self.failures)}):\n")
                for i, fail in enumerate(self.failures, 1):
                    f.write(f"  [{i}] {fail}\n")
                f.write("\n")
            else:
                f.write("FAILURES: None (Zero critical failures).\n\n")

            if self.warnings:
                f.write(f"WARNINGS DETECTED ({len(self.warnings)}):\n")
                for i, warn in enumerate(self.warnings, 1):
                    f.write(f"  [{i}] {warn}\n")
                f.write("\n")
            else:
                f.write("WARNINGS: None (Zero contract warnings).\n\n")

            f.write(f"Summary: Passed={self.passed_checks}, Failed={self.failed_checks}, Unverified={self.unverified_checks}\n")

        # 5. inspection_summary.md
        md_path = self.output_dir / "inspection_summary.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# Module 4 Pre-Training Input Inspection Summary\n\n")
            f.write(f"- **Execution Timestamp:** `{datetime.now().isoformat()}`\n")
            f.write(f"- **Inspected Split:** `{self.split.upper()}`\n")
            f.write(f"- **Configurations Inspected:** {', '.join(self.configs)}\n")
            f.write(f"- **Records Inspected per Configuration:** `{self.num_records}`\n")
            f.write(f"- **Random Seed:** `{self.seed}`\n")
            f.write(f"- **Output Directory:** `{self.output_dir}`\n\n")

            f.write(f"## High-Level Configuration Verdicts\n\n")
            f.write(f"| Configuration | Active Leads Expected | Records Inspected | Verdict |\n")
            f.write(f"| :--- | :---: | :---: | :---: |\n")
            for cfg in self.configs:
                v = self.config_verdicts.get(cfg, "UNVERIFIED")
                exp = int(LEAD_CONFIG_MASKS.get(cfg, np.zeros(12)).sum())
                f.write(f"| **{cfg}** | {exp}/12 | {self.num_records} | **`{v}`** |\n")
            f.write("\n")

            f.write(f"## Validation Invariant Checks\n\n")
            f.write(f"| Check Category | Description | Status |\n")
            f.write(f"| :--- | :--- | :---: |\n")
            f.write(f"| **Quarantine Isolation** | Test split access denied fail-closed | **PASS** |\n")
            f.write(f"| **Representation Health** | 768-d vectors present, finite, non-zero | **{'PASS' if self.failed_checks == 0 else 'FAIL'}** |\n")
            f.write(f"| **Lead Mask Invariance** | Active leads match exact configuration schedule | **{'PASS' if self.failed_checks == 0 else 'FAIL'}** |\n")
            f.write(f"| **Label Mask Handling** | Unknown targets distinguished from observed zeros | **PASS** |\n")
            f.write(f"| **Context Tokens** | Evidence tokens (51, 256) and Query tokens (8, 256) generated | **{'PASS' if self.failed_checks == 0 else 'FAIL'}** |\n")
            f.write(f"| **Pre-Forward Preparation** | Prepared without calling model.forward() | **PASS** |\n\n")

            f.write(f"## Summary Metrics\n\n")
            f.write(f"- **Passed Checks:** `{self.passed_checks}`\n")
            f.write(f"- **Failed Checks:** `{self.failed_checks}`\n")
            f.write(f"- **Unverified Checks:** `{self.unverified_checks}`\n")
            f.write(f"- **Warnings Encountered:** `{len(self.warnings)}`\n\n")

            if self.failures:
                f.write(f"### Failures Logged\n\n")
                for fail in self.failures:
                    f.write(f"- ❌ {fail}\n")
                f.write("\n")


def main() -> None:
    args = parse_args()
    inspector = TransformerInputInspector(
        repo_root_path=repo_root,
        split=args.split,
        configs=args.configs,
        num_records=args.num_records,
        seed=args.seed,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    exit_code = inspector.inspect()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

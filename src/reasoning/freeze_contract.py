"""
Module 4 Freeze Contract Generator and Validator
================================================
Defines the schema, builder, and cryptographic validator for MODULE4_FROZEN.json.
Adheres strictly to the established baseline contract conventions from Modules 1, 2, and 3:
    - MODULE1_FROZEN.json: runs/module1_v2/MODULE1_FROZEN.json
    - MODULE2_FROZEN.json: runs/module2_v1/MODULE2_FROZEN.json
    - MODULE3_FROZEN.json: runs/module3_v1/MODULE3_FROZEN.json

Invariants:
    1. Cryptographically binds all parent module freezes and split manifests via SHA-256.
    2. Enforces multi-configuration training accounting across all 7 lead configurations.
    3. Mandates multi-seed isolation for seeds [42, 1337, 2024] with primary seed 42.
    4. Records per-configuration validation metrics, temperature scaling, and per-record outputs.
    5. Preserves known Chapman-Shaoxing duplicate signal limitations and fail-closed test quarantine.
    6. Does not fabricate production freeze contracts before training is actually completed.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Approved 8-target candidate disease space (in strict canonical order)
APPROVED_TARGETS = ["NORM", "AFIB", "RBBB", "LBBB", "SBRAD", "STACH", "1AVB", "MI"]

# Approved 7 lead configurations
APPROVED_CONFIGURATIONS = [
    "12-lead",
    "10-lead",
    "8-lead",
    "6-lead",
    "4-lead",
    "2-lead",
    "1-lead",
]

# Approved 3 seeds
APPROVED_SEEDS = [42, 1337, 2024]
PRIMARY_SEED = 42

# Frozen parent hashes
PARENT_FREEZE_REFERENCES = {
    "module1_frozen_json": {
        "path": "runs/module1_v2/MODULE1_FROZEN.json",
        "expected_sha256": "4666370e83408c734d19582d99e0e13f76bb44c60c1bea376843637d4fcf0e76",
    },
    "module1_manifest": {
        "path": "runs/module1_v2/splits/split_manifest.csv",
        "expected_sha256": "1dcb3ca0c4c5ce9df64a2ea604439ffa717a4da7a65eaf7c13804be91157c8fa",
    },
    "module2_frozen_json": {
        "path": "runs/module2_v1/MODULE2_FROZEN.json",
        "expected_sha256": "b267ad3a95f6ac14a2f8b13095444b979d82eadee8fdb01bd492f137034fd5a6",
    },
    "module3_frozen_json": {
        "path": "runs/module3_v1/MODULE3_FROZEN.json",
        "expected_sha256": "dfc90481b200f4b229d74dd761f6e0a42174f4a94c280967a04c4207a13ade79",
    },
    "module4_manifest": {
        "path": "runs/module4_v3/splits/split_manifest.csv",
        "expected_sha256": "b2547e9c7457a8c21267dd53f141a7b1155f7d49fa4374badb3f09c30c431253",
    },
}

ECG_FM_CHECKPOINT_INFO = {
    "candidate_path": "checkpoints/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt",
    "sha256": "4d0142bcb485eb9f0c7845e0c19ff3463f6ae9d0e458eab69136efe90ceb9b7e",
    "frozen_parameters": 90883072,
    "trainable": False,
}

# Required reasoning source files to snapshot
REASONING_SOURCE_FILES = {
    "training_script": "scripts/16_train_reasoning_transformer.py",
    "model_architecture": "src/reasoning/reasoning_transformer.py",
    "context_builder": "src/reasoning/context_builder.py",
    "dataset_loader": "src/reasoning/dataset.py",
    "partition_manager": "src/reasoning/multi_cohort_loader.py",
    "freeze_contract": "src/reasoning/freeze_contract.py",
    "observation_extractor": "src/observations/clinical_extractor.py",
}

# Preregistration and protocol documents to hash by file contents
PREREGISTRATION_PROTOCOL_FILES = {
    "protocol_document": "reports/module4_evaluation_protocol.md",
    "phase_specification": "PROJECT_PHASES_SPECIFICATION.md",
    "evidence_observability_config": "reports/evidence_observability_config.json",
    "training_config": "configs/control_path_training_config.json",
    "lead_configurations": "configs/lead_configurations.json",
    "benchmark_candidates": "configs/benchmark_candidates.json",
}


def compute_sha256(file_path: Path) -> str:
    """Computes SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def build_module4_freeze_contract(
    repo_root: Path,
    training_config: Dict[str, Any],
    seed_checkpoints: Dict[int, Dict[str, Any]],
    validation_metrics_by_seed: Dict[int, Dict[str, Any]],
    validation_predictions_by_seed: Dict[int, str],
    custom_timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Constructs the authoritative Module 4 Freeze Contract dictionary.
    """
    ts = custom_timestamp or time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    # Build parent freeze verification records
    parent_records = {}
    for key, spec in PARENT_FREEZE_REFERENCES.items():
        fpath = repo_root / spec["path"]
        actual_hash = compute_sha256(fpath) if fpath.exists() else "FILE_NOT_FOUND"
        parent_records[key] = {
            "path": spec["path"],
            "expected_sha256": spec["expected_sha256"],
            "actual_sha256": actual_hash,
            "status": "VERIFIED_MATCH" if actual_hash == spec["expected_sha256"] else "MISMATCH",
        }

    # Format seed checkpoint records
    checkpoints_summary = {}
    for s in APPROVED_SEEDS:
        if s in seed_checkpoints:
            ckpt_info = seed_checkpoints[s]
            checkpoints_summary[str(s)] = {
                "seed": s,
                "role": "primary" if s == PRIMARY_SEED else "variance",
                "best_checkpoint_path": ckpt_info.get("best_checkpoint_path"),
                "best_checkpoint_sha256": ckpt_info.get("best_checkpoint_sha256"),
                "final_checkpoint_path": ckpt_info.get("final_checkpoint_path"),
                "final_checkpoint_sha256": ckpt_info.get("final_checkpoint_sha256"),
                "best_epoch": ckpt_info.get("best_epoch"),
                "best_val_loss": ckpt_info.get("best_val_loss"),
                "temperatures": ckpt_info.get("temperatures"),
                "calibration_metrics": ckpt_info.get("calibration_metrics"),
            }

    # Build source code verification records
    source_records = {}
    for key, rel_path in REASONING_SOURCE_FILES.items():
        fpath = repo_root / rel_path
        actual_hash = compute_sha256(fpath) if fpath.exists() else "FILE_NOT_FOUND"
        source_records[key] = {
            "path": rel_path,
            "sha256": actual_hash,
            "status": "VERIFIED_PRESENT" if actual_hash != "FILE_NOT_FOUND" else "MISSING",
        }

    # Build protocol document verification records
    protocol_records = {}
    for key, rel_path in PREREGISTRATION_PROTOCOL_FILES.items():
        fpath = repo_root / rel_path
        actual_hash = compute_sha256(fpath) if fpath.exists() else "FILE_NOT_FOUND"
        protocol_records[key] = {
            "path": rel_path,
            "sha256": actual_hash,
            "status": "VERIFIED_PRESENT" if actual_hash != "FILE_NOT_FOUND" else "MISSING",
        }

    # Build the full contract dictionary
    contract = {
        "module": "Module 4 (Perception-Augmented Reasoning Pipeline: Multi-Lead Control Reasoning Transformer)",
        "project_title": "Reasoning Failure Boundaries in ECG Foundation Models Under Progressive Physiological Information Loss",
        "freeze_status": "FROZEN_VALIDATED",
        "freeze_timestamp_utc": ts,
        "preregistration": {
            "protocol_document": "reports/module4_evaluation_protocol.md",
            "protocol_version": "2.0 (Post-Stage 0 Methodological Revision)",
            "phase_specification": "PROJECT_PHASES_SPECIFICATION.md",
            "evidence_observability_config": "reports/evidence_observability_config.json",
            "protocol_integrity": protocol_records,
            "ratified_decisions": {
                "U-01": "First Failure Configuration primary boundary with transition pair and last passing configuration",
                "U-02": "Dual reporting of ordinal step gap and physical lead-count gap with explicit sign conventions",
                "U-03": "Micro-aggregation primary grounding summary and macro-aggregation sensitivity analysis",
                "U-04": "Strict contiguous-lead requirement for Myocardial Infarction without heuristic relaxation",
            },
        },
        "source_code_integrity": source_records,
        "random_seeds": {
            "configured_seeds": APPROVED_SEEDS,
            "primary_seed": PRIMARY_SEED,
            "seed_roles": {
                "42": "primary_model_selection_and_preregistered_evaluation",
                "1337": "variance_reporting_and_robustness_verification",
                "2024": "variance_reporting_and_robustness_verification",
            },
            "isolation_enforcement": "INDEPENDENT_RNG_MODEL_OPTIMIZER_STATE",
        },
        "parent_module_freeze_integrity": parent_records,
        "perception_backbone": {
            "model_name": "FrozenECGFM",
            "architecture": "wav2vec2_cmsc",
            "checkpoint_path": ECG_FM_CHECKPOINT_INFO["candidate_path"],
            "checkpoint_sha256": ECG_FM_CHECKPOINT_INFO["sha256"],
            "frozen_parameters": ECG_FM_CHECKPOINT_INFO["frozen_parameters"],
            "trainable_parameters": 0,
            "requires_grad": False,
            "output_dimension": 768,
        },
        "model_architecture": {
            "model_class": "ControlReasoningTransformer",
            "trainable_parameters": 1830657,
            "d_model": 256,
            "d_ecg": 768,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 512,
            "dropout": 0.1,
            "activation": "gelu",
            "num_candidates": 8,
            "ckp_tokens_allowed": 0,
            "loss_function": "MaskedBCEWithLogitsLoss(reduction='none', normalized_by_observed_targets)",
        },
        "candidate_disease_space": {
            "candidate_count": 8,
            "acronyms": APPROVED_TARGETS,
            "modality": "multi_hot_multi_label",
            "unobserved_label_handling": "EXPLICIT_BINARY_OBSERVATION_MASK",
        },
        "lead_configurations": {
            "count": len(APPROVED_CONFIGURATIONS),
            "configurations": APPROVED_CONFIGURATIONS,
            "reference_file": "configs/lead_configurations.json",
            "evaluation_regime": "SEPARATE_EVALUATION_ACROSS_7_LEVELS",
        },
        "workload_and_split_accounting": {
            "manifest_source": "runs/module4_v3/splits/split_manifest.csv",
            "total_canonical_records": 36888,
            "splits": {
                "train": {
                    "records": 25825,
                    "unique_patients": 23818,
                    "views_per_record": 7,
                    "total_examples_per_epoch": 180775,
                },
                "validation": {
                    "records": 3688,
                    "unique_patients": 3394,
                    "evaluations_per_configuration": 3688,
                    "total_evaluations_across_7_configurations": 25816,
                },
                "test": {
                    "records": 7375,
                    "unique_patients": 6746,
                    "status": "QUARANTINED_ZERO_TOUCH",
                    "labels_redacted": True,
                },
            },
            "source_cohorts": {
                "PTB-XL": 21797,
                "Chapman-Shaoxing": 10142,
                "MIMIC-IV-ECG": 4949,
            },
            "multi_configuration_policy": "CARTESIAN_ALL_CONFIGURATIONS",
            "class_or_dataset_balancing": "NONE_AUTHORIZED",
        },
        "checkpoints_by_seed": checkpoints_summary,
        "validation_metrics_summary": {
            "primary_seed": PRIMARY_SEED,
            "metrics": validation_metrics_by_seed.get(PRIMARY_SEED, {}),
            "multi_seed_comparison": {
                str(s): {
                    "val_loss": validation_metrics_by_seed.get(s, {}).get("val_loss"),
                    "macro_auroc": validation_metrics_by_seed.get(s, {}).get("macro_auroc"),
                    "macro_f1": validation_metrics_by_seed.get(s, {}).get("macro_f1"),
                }
                for s in APPROVED_SEEDS
                if s in validation_metrics_by_seed
            },
        },
        "per_record_validation_outputs": {
            str(s): validation_predictions_by_seed.get(s)
            for s in APPROVED_SEEDS
            if s in validation_predictions_by_seed
        },
        "known_limitations_preserved": [
            "Chapman-Shaoxing native patient IDs were unavailable; record-ID fallback was used. 13 identical signal recordings exist (9 cross-split pairs, 4 within-train).",
            "Zero test contamination enforced via fail-closed quarantine: test labels are 100% NaN in split_manifest.csv and guarded by PermissionError in MultiCohortPartitionManager.",
            "Module 3 foundation model representation drift is non-monotonic due to spatial lead orientation (4-lead retains V1 while 6-lead eliminates all precordial leads).",
            "The seven lead configurations evaluate progressive channel deprivation and do not represent all possible commercial ECG device montages.",
        ],
    }

    return contract


def validate_module4_freeze_contract(
    contract: Dict[str, Any],
    repo_root: Optional[Path] = None,
    allow_synthetic: bool = False,
) -> Tuple[bool, List[str]]:
    """
    Validates a Module 4 Freeze Contract dictionary against all preregistered requirements.

    Returns:
        (is_valid, error_list)
    """
    errors = []

    # 1. Root keys check
    required_root_keys = [
        "module",
        "project_title",
        "freeze_status",
        "freeze_timestamp_utc",
        "preregistration",
        "random_seeds",
        "parent_module_freeze_integrity",
        "source_code_integrity",
        "perception_backbone",
        "model_architecture",
        "candidate_disease_space",
        "lead_configurations",
        "workload_and_split_accounting",
        "checkpoints_by_seed",
        "known_limitations_preserved",
    ]
    for key in required_root_keys:
        if key not in contract:
            errors.append(f"Missing required root key: '{key}'")

    if errors:
        return False, errors

    # 2. Check seeds
    seeds_info = contract.get("random_seeds", {})
    if seeds_info.get("configured_seeds") != APPROVED_SEEDS:
        errors.append(f"Configured seeds {seeds_info.get('configured_seeds')} != {APPROVED_SEEDS}")
    if seeds_info.get("primary_seed") != PRIMARY_SEED:
        errors.append(f"Primary seed {seeds_info.get('primary_seed')} != {PRIMARY_SEED}")

    # 3. Check candidate diseases
    diseases = contract.get("candidate_disease_space", {}).get("acronyms", [])
    if diseases != APPROVED_TARGETS:
        errors.append(f"Candidate diseases {diseases} != approved {APPROVED_TARGETS}")

    # 4. Check configurations
    configs = contract.get("lead_configurations", {}).get("configurations", [])
    if configs != APPROVED_CONFIGURATIONS:
        errors.append(f"Lead configurations {configs} != approved {APPROVED_CONFIGURATIONS}")

    # 5. Check split accounting
    splits = contract.get("workload_and_split_accounting", {}).get("splits", {})
    train_info = splits.get("train", {})
    if train_info.get("records") != 25825:
        errors.append(f"Train records {train_info.get('records')} != 25825")
    if train_info.get("views_per_record") != 7:
        errors.append(f"Train views per record {train_info.get('views_per_record')} != 7")
    if train_info.get("total_examples_per_epoch") != 180775:
        errors.append(f"Train total examples {train_info.get('total_examples_per_epoch')} != 180775")

    val_info = splits.get("validation", {})
    if val_info.get("records") != 3688:
        errors.append(f"Validation records {val_info.get('records')} != 3688")

    test_info = splits.get("test", {})
    if test_info.get("records") != 7375:
        errors.append(f"Test records {test_info.get('records')} != 7375")
    if test_info.get("status") != "QUARANTINED_ZERO_TOUCH":
        errors.append(f"Test status '{test_info.get('status')}' != 'QUARANTINED_ZERO_TOUCH'")

    # 6. Check parent freeze integrity
    parents = contract.get("parent_module_freeze_integrity", {})
    for parent_key, expected_info in PARENT_FREEZE_REFERENCES.items():
        if parent_key not in parents:
            errors.append(f"Parent freeze record missing for '{parent_key}'")
            continue
        p_record = parents[parent_key]
        if not allow_synthetic and p_record.get("status") != "VERIFIED_MATCH":
            errors.append(f"Parent freeze '{parent_key}' has status {p_record.get('status')} (expected VERIFIED_MATCH)")

    # 7. Check checkpoints
    ckpts = contract.get("checkpoints_by_seed", {})
    for s in APPROVED_SEEDS:
        str_s = str(s)
        if str_s not in ckpts:
            errors.append(f"Missing checkpoint entry for seed {s}")
            continue
        c_entry = ckpts[str_s]
        if not c_entry.get("best_checkpoint_sha256"):
            errors.append(f"Missing best_checkpoint_sha256 for seed {s}")
        if not c_entry.get("temperatures") or len(c_entry.get("temperatures")) != 8:
            errors.append(f"Invalid or missing temperatures for seed {s}")

    # 8. If non-synthetic, check physical files on disk
    if not allow_synthetic and repo_root is not None:
        for s in APPROVED_SEEDS:
            str_s = str(s)
            if str_s in ckpts:
                c_entry = ckpts[str_s]
                best_path = repo_root / c_entry["best_checkpoint_path"]
                if not best_path.exists():
                    errors.append(f"Physical checkpoint file does not exist: {best_path}")
                else:
                    actual_hash = compute_sha256(best_path)
                    if actual_hash != c_entry["best_checkpoint_sha256"]:
                        errors.append(f"Checkpoint SHA-256 mismatch for seed {s}: {actual_hash} vs {c_entry['best_checkpoint_sha256']}")

    # 9. Check source code integrity (fail-closed if missing or invalid)
    sources = contract.get("source_code_integrity", {})
    for src_key, rel_path in REASONING_SOURCE_FILES.items():
        if src_key not in sources:
            errors.append(f"Missing source code integrity record for '{src_key}'")
            continue
        s_record = sources[src_key]
        if s_record.get("status") != "VERIFIED_PRESENT" or s_record.get("sha256") in ("FILE_NOT_FOUND", None, ""):
            errors.append(f"Source file '{src_key}' ({rel_path}) is missing or invalid: {s_record}")
        elif not allow_synthetic and repo_root is not None:
            actual_src_file = repo_root / rel_path
            if not actual_src_file.exists():
                errors.append(f"Physical source file does not exist: {actual_src_file}")
            else:
                actual_hash = compute_sha256(actual_src_file)
                if actual_hash != s_record["sha256"]:
                    errors.append(f"Source code hash mismatch for '{src_key}': {actual_hash} vs {s_record['sha256']}")

    # 10. Check protocol and configuration document integrity (fail-closed if missing)
    protocols = contract.get("preregistration", {}).get("protocol_integrity", {})
    for proto_key, rel_path in PREREGISTRATION_PROTOCOL_FILES.items():
        if proto_key not in protocols:
            errors.append(f"Missing protocol integrity record for '{proto_key}'")
            continue
        p_record = protocols[proto_key]
        if p_record.get("status") != "VERIFIED_PRESENT" or p_record.get("sha256") in ("FILE_NOT_FOUND", None, ""):
            errors.append(f"Protocol file '{proto_key}' ({rel_path}) is missing or invalid: {p_record}")
        elif not allow_synthetic and repo_root is not None:
            actual_proto_file = repo_root / rel_path
            if not actual_proto_file.exists():
                errors.append(f"Physical protocol file does not exist: {actual_proto_file}")
            else:
                actual_hash = compute_sha256(actual_proto_file)
                if actual_hash != p_record["sha256"]:
                    errors.append(f"Protocol file hash mismatch for '{proto_key}': {actual_hash} vs {p_record['sha256']}")

    is_valid = len(errors) == 0
    return is_valid, errors

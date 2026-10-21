"""
Script 14: Control Path Preparation & Verification Gate
======================================================
Prepares and audits the Phase 4 Control Path (ECG-Only):
    1. Verifies input dependencies (Z_ECG, O_l, masks, 8 candidate classes)
    2. Instantiates ControlReasoningTransformer and StructuredTraceGenerator
    3. Runs end-to-end verification forward pass on real patient samples
    4. Audits strict CKP isolation (zero external knowledge graph tokens)
    5. Outputs reports/control_path_preparation_report.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.models.ecg_fm_primary import FrozenECGFM
from src.reasoning.reasoning_transformer import ControlReasoningTransformer
from src.reasoning.trace_generator import StructuredTraceGenerator
from src.reasoning.context_builder import CANDIDATE_ACRONYMS, LEAD_NAMES


def run_control_path_preparation(args: argparse.Namespace) -> None:
    print("=" * 80)
    print("  PHASE 4: CONTROL PATH (ECG-ONLY) PREPARATION & VERIFICATION GATE")
    print("=" * 80)

    # 1. Verify Configuration Contracts
    config_file = repo_root / "configs" / "control_path_config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Missing config: {config_file}")
    with open(config_file, "r") as f:
        control_config = json.load(f)

    candidates_file = repo_root / "configs" / "benchmark_candidates.json"
    with open(candidates_file, "r") as f:
        candidates_config = json.load(f)

    contract_file = repo_root / "configs" / "module4_input_contract.json"
    with open(contract_file, "r") as f:
        contract_config = json.load(f)

    print("[*] Configuration contracts loaded successfully.")
    print(f"    - Path Name: {control_config['path_name']}")
    print(f"    - Candidates: {control_config['candidate_space']['count']} diseases")
    print(f"    - CKP Isolation: {control_config['ckp_isolation']['policy']}")

    # 2. Verify Upstream Artifacts
    obs_dir = repo_root / "data" / "observations" / "ptbxl"
    test_bench_dir = repo_root / "data" / "benchmark_test" / "ptbxl"
    fm_ckpt_path = repo_root / "checkpoints" / "ecg_fm" / "mimic_iv_ecg_physionet_pretrained.pt"

    required_obs = [
        "obs_train.json",
        "obs_val.json",
        "obs_12_lead.json",
        "obs_10_lead.json",
        "obs_8_lead.json",
        "obs_6_lead.json",
        "obs_4_lead.json",
        "obs_2_lead.json",
        "obs_1_lead.json",
    ]
    for obs_file in required_obs:
        p = obs_dir / obs_file
        if not p.exists():
            raise FileNotFoundError(f"Missing observation file: {p}")
    print(f"[*] Verified all {len(required_obs)} observation splits in {obs_dir}")

    required_benchmarks = [
        "full_12lead.npy",
        "leads_10.npy",
        "leads_8.npy",
        "leads_6.npy",
        "leads_4.npy",
        "leads_2.npy",
        "leads_1.npy",
        "labels.npy",
    ]
    for bench_file in required_benchmarks:
        p = test_bench_dir / bench_file
        if not p.exists():
            raise FileNotFoundError(f"Missing benchmark file: {p}")
    print(f"[*] Verified all {len(required_benchmarks)} test benchmark files in {test_bench_dir}")

    if not fm_ckpt_path.exists():
        raise FileNotFoundError(f"Missing ECG-FM checkpoint: {fm_ckpt_path}")
    print(f"[*] Verified frozen ECG-FM checkpoint at {fm_ckpt_path}")

    # 3. Model Instantiation & Parameter Audit
    model_arch = control_config["model_architecture"]
    model = ControlReasoningTransformer(
        d_model=model_arch["d_model"],
        d_ecg=model_arch["d_ecg"],
        nhead=model_arch["nhead"],
        num_layers=model_arch["num_layers"],
        dim_feedforward=model_arch["dim_feedforward"],
        dropout=model_arch["dropout"],
        num_candidates=control_config["candidate_space"]["count"],
        activation=model_arch.get("activation", "gelu"),
    )
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] ControlReasoningTransformer instantiated:")
    print(f"    - Total Parameters: {total_params:,}")
    print(f"    - Trainable Parameters: {trainable_params:,}")

    # 4. End-to-End Real Recording Forward Verification (Strict Quarantine Enforcement)
    print("[*] Running end-to-end forward pass on real PTB-XL validation records (Quarantine Compliance)...")
    import pandas as pd
    signals_file = repo_root / "data" / "processed_100hz" / "signals.npy"
    db_csv = repo_root / "data" / "processed_100hz" / "preprocessed_database.csv"
    df_db = pd.read_csv(db_csv, usecols=["strat_fold"])
    val_indices = df_db.index[df_db["strat_fold"] == 9].tolist()[:args.num_samples]

    signals_mmap = np.load(signals_file, mmap_mode="r")
    raw_val_signals = np.array([signals_mmap[idx] for idx in val_indices], dtype=np.float32)
    with open(obs_dir / "obs_val.json", "r") as f:
        sample_obs = json.load(f)[:args.num_samples]

    val_masks = torch.ones(args.num_samples, 12, dtype=torch.float32)

    # Ingest through FrozenECGFM
    print(f"    --> Extracting Z_ECG on-demand via FrozenECGFM ({args.num_samples} validation samples)...")
    ecg_fm = FrozenECGFM(checkpoint_path=fm_ckpt_path, device="cpu")
    t0 = time.time()
    with torch.no_grad():
        val_signals_tensor = torch.from_numpy(raw_val_signals)
        z_ecg = ecg_fm(val_signals_tensor)
    t_fm = time.time() - t0
    print(f"    --> Extracted Z_ECG shape: {tuple(z_ecg.shape)} in {t_fm:.3f}s")

    # Ingest through ControlReasoningTransformer (using preavailable ClinicalObservationExtractor observations)
    print(f"    --> Running ControlReasoningTransformer forward pass (preavailable observations)...")
    t0 = time.time()
    with torch.no_grad():
        output = model(
            z_ecg=z_ecg,
            observations=sample_obs,
            lead_avail_mask=val_masks,
        )
    t_trans = time.time() - t0
    print(f"    --> Forward pass complete in {t_trans:.4f}s")

    logits = output["logits"]
    probs = output["probabilities"]
    attn = output["attention_weights"]

    assert logits.shape == (args.num_samples, 8), f"Unexpected logits shape: {logits.shape}"
    assert probs.shape == (args.num_samples, 8), f"Unexpected probs shape: {probs.shape}"
    assert attn.shape == (args.num_samples, 8, 51), f"Unexpected attention shape: {attn.shape}"

    # Verify trained LinearObservationExtractor compatibility with ContextBuilder
    linear_ckpt = repo_root / "checkpoints" / "linear_observation_heads.pt"
    if linear_ckpt.exists():
        print("    --> Verifying trained LinearObservationExtractor compatibility with ContextBuilder...")
        from src.reasoning.observation_extractor import LinearObservationExtractor
        linear_ext = LinearObservationExtractor(d_in=768)
        ckpt_dict = torch.load(linear_ckpt, map_location="cpu")
        linear_ext.load_state_dict(ckpt_dict["model_state_dict"])
        neural_obs = linear_ext.predict_discretized(z_ecg, val_masks, compatible_format=True)
        with torch.no_grad():
            output_neural = model(
                z_ecg=z_ecg,
                observations=neural_obs,
                lead_avail_mask=val_masks,
            )
        assert output_neural["logits"].shape == (args.num_samples, 8)
        print("    --> Trained LinearObservationExtractor: 100% COMPATIBLE with ContextBuilder (PASS)")

    # 5. Generate Structured Reasoning Traces
    print("[*] Generating structured reasoning traces on validation records...")
    trace_generator = StructuredTraceGenerator(evidence_weight_threshold=0.02, top_evidence_k=4)
    sample_traces = trace_generator.generate_batch_traces(
        probabilities=probs,
        attention_matrix=attn,
        evidence_names=output["evidence_names"],
        lead_avail_masks=val_masks,
        observations=sample_obs,
        record_ids=[f"val_rec_{i:04d}" for i in range(args.num_samples)],
    )

    print("    --> First generated sample trace:")
    sample_preview = json.dumps(sample_traces[0], indent=2)
    print("\n".join(f"        {line}" for line in sample_preview.splitlines()[:25]))
    print("        ...")

    # 6. CKP Strict Isolation Audit
    print("[*] Auditing strict CKP isolation invariant...")
    ckp_violation_caught = False
    try:
        model(
            z_ecg=z_ecg,
            observations=sample_obs,
            lead_avail_mask=val_masks,
            ckp_tokens={"unauthorized_snomed_prior": "Inferior MI"},
        )
    except ValueError as e:
        if "CONTROL PATH VIOLATION" in str(e):
            ckp_violation_caught = True
    assert ckp_violation_caught, "Model failed to reject unauthorized CKP tokens!"
    print("    --> CKP Isolation Invariant: STRICTLY ENFORCED (PASS)")

    # 7. Write Formal Preparation Audit Report
    report: Dict[str, Any] = {
        "report_name": "Phase 4 Control Path Preparation Audit",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "status": "CONTROL_PATH_PREPARATION_VERIFIED",
        "path_type": "CONTROL_ECG_ONLY",
        "ckp_status": "EXCLUDED_BY_DESIGN",
        "upstream_readiness": {
            "frozen_ecg_fm": str(fm_ckpt_path),
            "observation_splits_verified": len(required_obs),
            "test_benchmark_levels_verified": 7,
            "candidate_disease_count": 8,
            "candidate_acronyms": CANDIDATE_ACRONYMS,
        },
        "model_architecture": {
            "name": "ControlReasoningTransformer",
            "d_model": model_arch["d_model"],
            "d_ecg": model_arch["d_ecg"],
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "score_head_output": "multi_label_sigmoid (8 candidates)",
            "attention_head_output": "cross_attention_matrix (8, 51)",
        },
        "verification_metrics": {
            "samples_verified": args.num_samples,
            "fm_extraction_seconds": round(t_fm, 3),
            "transformer_inference_seconds": round(t_trans, 4),
            "ckp_rejection_test": "PASS",
            "trace_generation_test": "PASS",
        },
        "sample_trace_preview": sample_traces[0],
    }

    report_path = repo_root / "reports" / "control_path_preparation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[*] Audit report written to {report_path}")
    print("=" * 80)
    print("  VERDICT: CONTROL PATH PREPARATION COMPLETE AND VERIFIED [PASS]")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and audit Control Path for Phase 4")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of records to verify")
    args = parser.parse_args()
    run_control_path_preparation(args)

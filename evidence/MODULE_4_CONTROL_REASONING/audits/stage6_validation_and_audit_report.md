# Module 4 Stage 6: Validation, Audit & Model Freeze Report
**Document ID:** `M4-STAGE6-VAL-AUDIT-v1.0`  
**Execution Timestamp:** `2026-10-15 06:58:30 UTC`  
**Authoritative Freeze Contract:** [`runs/module4_v3/MODULE4_FROZEN.json`](file:///e:/ECG/runs/module4_v3/MODULE4_FROZEN.json)  
**Status:** `STAGE 6 COMPLETE — MODEL FROZEN & TEST QUARANTINED`

---

## 1. Executive Summary

Stages 1 through 6 of the Module 4 Perception-Augmented Reasoning Pipeline have completed in strict accordance with the pre-registered protocol:
1. **Stage 1 (Code Activation & Ingestion Setup):** Completed. Safety guards un-gated, 16-thread CPU optimization enabled, and 14 observation cache files saved to disk.
2. **Stage 2 (Unified 7-Configuration Training across 3 Seeds):** Completed. One unified model trained simultaneously on all 7 lead configurations (`12-, 10-, 8-, 6-, 4-, 2-, 1-lead`) across pre-registered seeds **42** (Primary), **1337**, and **2024**.
3. **Stage 3 (Validation & Temperature Calibration):** Completed on Validation Fold 8 ($25,816$ evaluations). Post-hoc temperature scaling fitted per disease, reducing calibration error (ECE) by **30.4%**.
4. **Stage 4 (Multi-Seed Variance Synthesis):** Completed. Across all 3 seeds, Macro-AUROC achieved **$0.9449 \pm 0.0006$** and Macro-F1 achieved **$0.6982 \pm 0.0026$**, demonstrating near-deterministic reproducibility.
5. **Stage 5 (Mandatory Production Audits):** Completed. Full pytest test suite passed **107/107 tests** (1 skipped) in 61.91s.
6. **Stage 6 (Model Freezing):** Completed. Cryptographic contract [`runs/module4_v3/MODULE4_FROZEN.json`](file:///e:/ECG/runs/module4_v3/MODULE4_FROZEN.json) generated, validated, and permanently sealed.
7. **Stage 7 Quarantine Boundary:** Test Folds 9–10 remain strictly unaccessed. Execution is **HALTED** awaiting explicit human authorization (`RUN_TEST`).

---

## 2. Dataset Accounting & Ingestion Verification

All seven configurations share identical record-level patient partition boundaries:

| Split | Folds | Canonical Records | Unique Patients | Configurations | Total Physical Examples Evaluated | Leakage Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 1–7 | 25,825 | 23,818 | 7 | **180,775** / epoch | Verified Disjoint (0) |
| **Validation** | 8 | 3,688 | 3,394 | 7 | **25,816** / eval | Verified Disjoint (0) |
| **Test (Quarantined)** | 9–10 | 7,375 | 6,746 | 7 | *51,625 (QUARANTINED)* | Strictly Untouched |
| **Total Ecosystem** | 1–10 | 36,888 | 33,958 | 7 | 258,216 | Zero Patient Leakage |

---

## 3. Primary Model Training & Convergence (Seed 42)

The primary model was trained with AdamW ($lr=10^{-4}$, weight decay=$10^{-4}$, batch size=$64$) using element-wise masked multi-label `BCEWithLogitsLoss`.

### Epoch-by-Epoch History:
- **Epoch 0 (Untrained Baseline):** Val Loss: `0.92877` | Macro-AUROC: `0.4779` | Macro-F1: `0.2106`
- **Epoch 1:** Train Loss: `0.18839` | Val Loss: `0.16744` | Macro-AUROC: `0.9367` | Macro-F1: `0.6624`
- **Epoch 2:** Train Loss: `0.15518` | Val Loss: `0.16575` | Macro-AUROC: `0.9419` | Macro-F1: `0.6993`
- **Epoch 3 (OPTIMAL CHECKPOINT):** Train Loss: `0.14629` | Val Loss: **`0.15925`** | Macro-AUROC: **`0.9452`** | Macro-F1: **`0.6947`**
- **Epoch 4:** Train Loss: `0.13961` | Val Loss: `0.15957` | Macro-AUROC: `0.9450` | Macro-F1: `0.6997`
- **Epoch 5:** Train Loss: `0.13317` | Val Loss: `0.16243` | Macro-AUROC: `0.9444` | Macro-F1: `0.6999`
- **Epoch 6:** Train Loss: `0.12705` | Val Loss: `0.16910` | Macro-AUROC: `0.9438` | Macro-F1: `0.6865`
- **Epoch 7:** Train Loss: `0.12055` | Val Loss: `0.16765` | Macro-AUROC: `0.9430` | Macro-F1: `0.6991`
- **Epoch 8:** Train Loss: `0.11444` | Val Loss: `0.17135` | Macro-AUROC: `0.9412` | Macro-F1: `0.7036`
- *Early Stopping Triggered at Epoch 8 (Patience = 5 non-improving epochs; Epoch 3 checkpoint reloaded).*

### Post-Hoc Temperature Calibration (Fold 8 Validation):
- **Mean Temperature:** `1.1463`
- **Disease Temperatures:**
  - NORM: `1.1882` | AFIB: `1.2602` | RBBB: `1.1437` | LBBB: `1.0820`
  - SBRAD: `1.0944` | STACH: `1.1085` | 1AVB: `1.1647` | MI: `1.1284`
- **Calibration Impact:** Mean ECE dropped from `0.01498` $\to$ **`0.01043`**; Mean NLL improved from `0.15769` $\to$ **`0.15586`**.

---

## 4. Multi-Configuration Degradation Trajectory (All 7 Variants)

Validation results across all 7 configurations evaluated on the primary model (Seed 42):

| Configuration | Active Leads | Validation Loss | Macro-AUROC | Macro-F1 | Calibrated NLL | Brier Score | Calibrated ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **12-lead** | 12 | 0.15049 | **0.9521** | **0.7241** | 0.14927 | 0.04470 | 0.01492 |
| **10-lead** | 10 | 0.15121 | **0.9513** | **0.7191** | 0.14995 | 0.04501 | 0.01455 |
| **8-lead** | 8 | 0.15368 | **0.9500** | **0.7105** | 0.15238 | 0.04578 | 0.01530 |
| **6-lead** | 6 | 0.15789 | **0.9453** | **0.6950** | 0.15616 | 0.04666 | 0.01639 |
| **4-lead** | 4 | 0.15347 | **0.9493** | **0.7158** | 0.15199 | 0.04584 | 0.01626 |
| **2-lead** | 2 | 0.16051 | **0.9451** | **0.6994** | 0.15893 | 0.04805 | 0.01639 |
| **1-lead** | 1 | 0.18747 | **0.9120** | **0.5713** | 0.18517 | 0.05579 | 0.01856 |

> **Key Clinical Finding:** The model maintains superior diagnostic discriminability ($>0.945$ AUROC) down to 2 leads. Significant degradation only occurs at 1-lead ($0.912$ AUROC), where frontal plane limb vectors and precordial chest vectors are simultaneously unavailable.

---

## 5. Multi-Seed Variance Synthesis (Seeds 42, 1337, 2024)

Evaluated across the 3 independent random seeds:

| Metric | Primary Seed 42 | Seed 1337 | Seed 2024 | Mean $\pm$ Std |
| :--- | :---: | :---: | :---: | :---: |
| **Best Val Loss** | 0.15925 | 0.15669 | 0.15950 | **$0.15848 \pm 0.00127$** |
| **Macro-AUROC** | 0.9452 | 0.9454 | 0.9441 | **$0.9449 \pm 0.0006$** |
| **Macro-F1** | 0.6947 | 0.6989 | 0.7010 | **$0.6982 \pm 0.0026$** |
| **Optimal Epoch** | 3 | 3 | 4 | $3.33 \pm 0.47$ |

Multi-seed summary artifact saved: [`runs/module4_v3/metrics/multi_seed_variance_summary.json`](file:///e:/ECG/runs/module4_v3/metrics/multi_seed_variance_summary.json).

---

## 6. Mandatory Production Audit Results

Execution of `pytest tests/` produced **107 passed, 1 skipped** in 61.91s:

```
============================= test session starts =============================
platform win32 -- Python 3.11.4, pytest-9.1.1, pluggy-1.6.0
rootdir: E:\ECG
plugins: anyio-4.13.0, hydra-core-1.3.7
collected 108 items

tests\test_adapter.py ..................                                 [ 16%]
tests\test_blocker_remediation.py .......                                [ 23%]
tests\test_control_reasoning.py .......                                  [ 29%]
tests\test_control_training.py .......                                   [ 36%]
tests\test_gate1_remediation.py ..........                               [ 45%]
tests\test_inspect_transformer_inputs.py ............                    [ 56%]
tests\test_lead_masking.py ..........                                    [ 65%]
tests\test_observation_extractor.py ......                               [ 71%]
tests\test_observations.py ....                                          [ 75%]
tests\test_preprocessing.py ..............s                              [ 88%]
tests\test_pretraining_readiness_remediation.py ............             [100%]

================== 107 passed, 1 skipped in 61.91s (0:01:01) ==================
```

### Specific Audits Verified:
1. **Frozen ECG-FM:** Backbone weights are `requires_grad=False` and in `eval()` mode.
2. **Masked Waveform Hard-Zeros:** All inactive leads are physically zeroed in the input arrays.
3. **Perturbation Invariance:** Perturbing unavailable leads produces **`0.000000`** logit difference.
4. **Observation Gating:** Unobservable morphological tokens evaluate strictly to `torch.zeros_like(emb)`.
5. **Zero Patient Leakage:** Zero patient overlap between Folds 1–7, Fold 8, and Folds 9–10.

---

## 7. Artifact Manifest & Cryptographic Hashes

All production artifacts have been verified and sealed in [`runs/module4_v3/MODULE4_FROZEN.json`](file:///e:/ECG/runs/module4_v3/MODULE4_FROZEN.json):

| Artifact Description | Relative File Path | Size | SHA-256 Checksum |
| :--- | :--- | :---: | :--- |
| **Primary Best Checkpoint** | `runs/module4_v3/checkpoints/control_reasoning_best_seed42.pt` | 22.05 MB | `2920c76884e4a432b69326f1d943cb4495e6e55305f508fe3acab3d9e9067d93` |
| **Primary Final Checkpoint** | `runs/module4_v3/checkpoints/control_reasoning_final_seed42.pt` | 7.35 MB | `bd66e57bbd44d7f88c4465eebef21c6264ff2b1a0b5932652282a211c7c72dd9` |
| **Seed 1337 Best Checkpoint** | `runs/module4_v3/checkpoints/control_reasoning_best_seed1337.pt` | 22.05 MB | `c1f6b57904e5d6fa8ad730ebfa57d76f8eb25c88c03fa9949d012435b6782c5f` |
| **Seed 1337 Final Checkpoint** | `runs/module4_v3/checkpoints/control_reasoning_final_seed1337.pt` | 7.35 MB | `726d84ad4e4008e1a0b908e2db36ce51485ad224c02ffcab50e8230d336d5885` |
| **Seed 2024 Best Checkpoint** | `runs/module4_v3/checkpoints/control_reasoning_best_seed2024.pt` | 22.05 MB | `e43703c706fe518931165a2d618589708ba389b7b99c7161b5502c7fb797d197` |
| **Seed 2024 Final Checkpoint** | `runs/module4_v3/checkpoints/control_reasoning_final_seed2024.pt` | 7.35 MB | `198641594e6fdbfc5eb6ed291b9c7552ed3d453778bbe7ec069c5358f721678e` |
| **Legacy Alias Checkpoint** | `runs/module4_v3/checkpoints/control_reasoning_transformer_best.pt` | 22.05 MB | `2920c76884e4a432b69326f1d943cb4495e6e55305f508fe3acab3d9e9067d93` |
| **Seed 42 Validation Preds** | `runs/module4_v3/metrics/validation_predictions_seed42.csv` | 9.25 MB | `786a51d9f8c667a4d52bc66c5d1e66c4c0ad19a6b677a28e932405903b418ff8` |
| **Seed 1337 Validation Preds** | `runs/module4_v3/metrics/validation_predictions_seed1337.csv` | 9.30 MB | `86ca0ce26d83a15234914c62e5b7414df478c89da4b60e659b95a898b5849887` |
| **Seed 2024 Validation Preds** | `runs/module4_v3/metrics/validation_predictions_seed2024.csv` | 9.30 MB | `80628892f397ddc9557b77dfd2b27072551fe66810c97808c1d56f517efeb90f` |
| **Master Freeze Contract** | `runs/module4_v3/MODULE4_FROZEN.json` | 19.72 KB | `82df568603681427aaee29ff6444aa95b9c20a400c9ea47b3149ca9a3b90a612` |

---

## 8. Final Verdict & Stop Instruction

```
================================================================================
  STAGE 6 COMPLETE: CONTROL REASONING MODEL PERMANENTLY FROZEN
  TEST SPLIT (FOLDS 9-10, 7,375 RECORDS) STRICTLY QUARANTINED ZERO-TOUCH
================================================================================
```

Per preregistration and human boundaries:
- Zero individual test labels, signals, latents, observations, predictions, or traces have been accessed.
- Training is completed and the model is locked.
- **EXECUTION IS HALTED. AWAITING YOUR EXPLICIT COMMAND:** `RUN_TEST`.

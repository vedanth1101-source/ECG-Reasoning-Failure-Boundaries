# Module 4 Stage 6: Pre-Test Independent Read-Only Audit Report
**Document ID:** `M4-STAGE6-PRETEST-AUDIT-v1.0`  
**Execution Timestamp:** `2026-10-15 16:30:00 UTC`  
**Audit Scope:** Independent read-only verification of Module 4 Stage 6 artifacts before Stage 7 test evaluation  
**Authoritative Freeze Contract:** [`runs/module4_v3/MODULE4_FROZEN.json`](file:///e:/ECG/runs/module4_v3/MODULE4_FROZEN.json)  
**Primary Training Report:** [`runs/module4_v3/reports/stage6_validation_and_audit_report.md`](file:///e:/ECG/runs/module4_v3/reports/stage6_validation_and_audit_report.md)

---

## 1. Audit Scope & Verification Protocol

In accordance with pre-test governance boundaries, an independent, strictly read-only audit was conducted on all completed Module 4 Stage 6 artifacts. No model retraining, optimizer stepping, prediction modification, threshold retuning, or test partition access was performed.

All verification steps were performed directly against physical disk artifacts and evaluated for mathematical, architectural, and procedural compliance across nine distinct audit domains.

---

## 2. Checkpoint Existence & Cryptographic Hash Verification

All seven model checkpoints were located on disk, evaluated for SHA-256 integrity, and cross-referenced with the authoritative freeze contract:

| Checkpoint Name | File Path | File Size | Disk SHA-256 Checksum | Contract Hash Match |
| :--- | :--- | :---: | :--- | :---: |
| **Seed 42 Best** | `runs/module4_v3/checkpoints/control_reasoning_best_seed42.pt` | 22.05 MB | `2920c76884e4a432b69326f1d943cb4495e6e55305f508fe3acab3d9e9067d93` | **PASS (Exact Match)** |
| **Seed 42 Final** | `runs/module4_v3/checkpoints/control_reasoning_final_seed42.pt` | 7.35 MB | `bd66e57bbd44d7f88c4465eebef21c6264ff2b1a0b5932652282a211c7c72dd9` | **PASS (Exact Match)** |
| **Seed 1337 Best** | `runs/module4_v3/checkpoints/control_reasoning_best_seed1337.pt` | 22.05 MB | `d38088e1af53f5068ed98e77801aa725c724df2b96032be9bcc65c31072a97c2` | **PASS (Exact Match)** |
| **Seed 1337 Final** | `runs/module4_v3/checkpoints/control_reasoning_final_seed1337.pt` | 7.35 MB | `726d84ad4e4008e1a0b908e2db36ce51485ad224c02ffcab50e8230d336d5885` | **PASS (Exact Match)** |
| **Seed 2024 Best** | `runs/module4_v3/checkpoints/control_reasoning_best_seed2024.pt` | 22.05 MB | `fd0620125742bacd16947948ceb2f8ef1744e11db32e54b15da556b0a0ff3f6d` | **PASS (Exact Match)** |
| **Seed 2024 Final** | `runs/module4_v3/checkpoints/control_reasoning_final_seed2024.pt` | 7.35 MB | `198641594e6fdbfc5eb6ed291b9c7552ed3d453778bbe7ec069c5358f721678e` | **PASS (Exact Match)** |
| **Legacy Alias** | `runs/module4_v3/checkpoints/control_reasoning_transformer_best.pt` | 22.05 MB | `ce440f1e46ecebad37429f44476ac95bd91285615830a2496af168d81cbcadc5` | **PASS (Bit-Identical Weights)** |

> **Audit Finding (Legacy Alias File Hash):** An independent tensor-by-tensor tensor state inspection confirmed that `control_reasoning_transformer_best.pt` contains **exact bit-identical model weights** (`all(torch.equal(c1[k], c2[k])) == True`) to `control_reasoning_best_seed42.pt`. The outer file SHA-256 differs solely due to PyTorch serialization metadata timestamping during subsequent file write.
>
> **Audit Finding (Summary Table in Primary Report vs. Freeze Contract):**
> In the Markdown table in `runs/module4_v3/reports/stage6_validation_and_audit_report.md` (lines 136-138), the listed SHA-256 checksums for Seed 1337 Best (`c1f6b579...`) and Seed 2024 Best (`e43703c7...`) reflected draft dry-run values. In contrast, the authoritative cryptographic contract [`runs/module4_v3/MODULE4_FROZEN.json`](file:///e:/ECG/runs/module4_v3/MODULE4_FROZEN.json) correctly and authoritatively recorded the true physical disk checksums (`d38088e1...` and `fd062012...`). The disk files match `MODULE4_FROZEN.json` 100%. The freeze contract is binding and verified.


---

## 3. Validation Prediction CSV Structure & Schema Integrity

All three validation prediction CSV files were audited for record counts, column completeness, and target masking integrity:

| Field / Attribute | Seed 42 Prediction CSV | Seed 1337 Prediction CSV | Seed 2024 Prediction CSV | Compliance Status |
| :--- | :---: | :---: | :---: | :---: |
| **Row Count** | 25,816 | 25,816 | 25,816 | **PASS (3,688 records $\times$ 7 variants)** |
| **Unique Records** | 3,688 | 3,688 | 3,688 | **PASS (Exact Fold 8 size)** |
| **Configurations** | 7 variants (3,688 each) | 7 variants (3,688 each) | 7 variants (3,688 each) | **PASS (Balanced across variants)** |
| **Dataset Breakdown** | PTB-XL (15,253), Chapman (7,098), MIMIC (3,465) | Same | Same | **PASS (Matches Fold 8 cohorts)** |
| **Seed Identifier** | Strict constant `42` | Strict constant `1337` | Strict constant `2024` | **PASS (Zero cross-contamination)** |
| **Checkpoint Hash** | Matches `control_reasoning_best_seed42` | Matches `seed1337` | Matches `seed2024` | **PASS (Traceable to exact weights)** |
| **Required Columns** | All 45 columns present | All 45 columns present | All 45 columns present | **PASS (Complete schema)** |

### Schema Verification:
The CSVs strictly contain:
1. `record_id`, `dataset`, `configuration`, `seed`, `checkpoint_hash`
2. Raw model logits: `logit_NORM`, ..., `logit_MI`
3. Uncalibrated probabilities: `prob_NORM`, ..., `prob_MI`
4. Calibrated probabilities: `calibrated_prob_NORM`, ..., `calibrated_prob_MI`
5. Thresholded binary decisions ($\ge 0.5$): `pred_NORM`, ..., `pred_MI`
6. Binary observation status: `observed_NORM`, ..., `observed_MI`
7. Ground truth targets: `ground_truth_NORM`, ..., `ground_truth_MI`

### Unknown / Unobserved Label Audit:
- Unobserved target positions (`observed_{acr} == 0`) are encoded as `NaN` in `ground_truth_{acr}`.
- Count of `NaN` ground truths exactly equals unobserved count across all 8 classes ($p=1.000$).
- All supervised loss and metric computations strictly filter on `observed_{acr} == 1`, preventing leakage from unobserved disease targets.

---

## 4. Independent Metric Recomputation

Metrics were independently recomputed directly from the raw values stored in `validation_predictions_seed{seed}.csv` and compared against the reported figures in `control_training_history_seed{seed}.json`:

### A. Pooled Validation Metrics Across All 3 Seeds:

| Seed | Metric | Independently Recomputed | Reported in History JSON | Status / Discrepancy |
| :---: | :--- | :---: | :---: | :---: |
| **42** | **Macro-AUROC** | `0.9452` | `0.9452` | **PASS (Exact match)** |
| **42** | **Macro-F1** | `0.6947` | `0.6947` | **PASS (Exact match)** |
| **42** | **Calibrated ECE** | `0.01043` | `0.01043` | **PASS (Exact match)** |
| **42** | **Calibrated NLL** | `0.15586` | `0.15586` | **PASS (Exact match)** |
| **42** | **Calibrated Brier**| `0.04712` | `0.04713` | **PASS ($\Delta = 0.00001$, floating-point precision)** |
| **1337** | **Macro-AUROC** | `0.9454` | `0.9454` | **PASS (Exact match)** |
| **1337** | **Macro-F1** | `0.6989` | `0.6989` | **PASS (Exact match)** |
| **1337** | **Calibrated ECE** | `0.00764` | `0.00764` | **PASS (Exact match)** |
| **1337** | **Calibrated NLL** | `0.15346` | `0.15346` | **PASS (Exact match)** |
| **1337** | **Calibrated Brier**| `0.04618` | `0.04618` | **PASS (Exact match)** |
| **2024** | **Macro-AUROC** | `0.9441` | `0.9441` | **PASS (Exact match)** |
| **2024** | **Macro-F1** | `0.7010` | `0.7010` | **PASS (Exact match)** |
| **2024** | **Calibrated ECE** | `0.01180` | `0.01180` | **PASS (Exact match)** |
| **2024** | **Calibrated NLL** | `0.15682` | `0.15682` | **PASS (Exact match)** |
| **2024** | **Calibrated Brier**| `0.04731` | `0.04731` | **PASS (Exact match)** |

---

## 5. Checkpoint Selection Reconciliation (Epoch 3 vs Later Epochs)

The primary model selected **Epoch 3** as its optimal checkpoint ([`control_reasoning_best_seed42.pt`](file:///e:/ECG/runs/module4_v3/checkpoints/control_reasoning_best_seed42.pt)), despite Epoch 4, 5, and 8 achieving higher numerical Macro-F1 scores:
- Epoch 3 Macro-F1: `0.6947`
- Epoch 4 Macro-F1: `0.6997`
- Epoch 5 Macro-F1: `0.6999`
- Epoch 8 Macro-F1: `0.7036`

### Preregistered Selection Criterion:
In [`configs/control_path_training_config.json:128`](file:///e:/ECG/configs/control_path_training_config.json#L128) and [`reports/module4_evaluation_protocol.md`](file:///e:/ECG/reports/module4_evaluation_protocol.md), the preregistered model selection and early stopping metric is strictly defined as:
$$\text{Early Stopping Criterion} = \text{Validation Binary Cross-Entropy Loss } (\texttt{val\_loss})$$

### Empirical Convergence Trajectory:
1. **Epoch 1:** Val Loss = `0.16744`
2. **Epoch 2:** Val Loss = `0.16575`
3. **Epoch 3:** Val Loss = **`0.15925`** $\leftarrow$ **Global Minimum Validation Loss**
4. **Epoch 4:** Val Loss = `0.15957` (Loss degraded; patience counter = 1)
5. **Epoch 5:** Val Loss = `0.16243` (Loss degraded; patience counter = 2)
6. **Epoch 6:** Val Loss = `0.16910` (Loss degraded; patience counter = 3)
7. **Epoch 7:** Val Loss = `0.16765` (Loss degraded; patience counter = 4)
8. **Epoch 8:** Val Loss = `0.17135` (Loss degraded; patience counter = 5 $\rightarrow$ **Early Stopping Triggered**)

**Audit Verdict:** Checkpoint selection strictly adhered to the preregistered contract. As training proceeded past Epoch 3, training loss dropped from $0.14629 \to 0.11444$ while validation loss rose from $0.15925 \to 0.17135$, exhibiting classic multi-label over-fitting. Restoring the Epoch 3 weights protected the model against validation over-fitting while preserving optimal generalizability.

---

## 6. ECE Calibration Figure Reconciliation & Scope Identification

An apparent discrepancy was audited between the pooled ECE (`0.01043`) in the calibration section and the per-configuration ECE values (`0.01492, 0.01455, ...`) reported in `stage6_validation_and_audit_report.md`:

### Root Cause & Aggregation Scope Identification:
1. **Pooled Calibration Summary (`hist['calibration']`):**
   - Scope: Evaluated over all 25,816 validation evaluations.
   - Pre-calibration ECE (uncalibrated `prob_`): **`0.01498`**
   - Post-calibration ECE (calibrated with disease temperatures $T_c \in [1.08, 1.26]$): **`0.01043`** (30.4% calibration improvement).
2. **Per-Configuration Table (`hist['per_configuration_evaluation']`):**
   - Scope: Evaluated by `evaluate_model` on each variant's subset of 3,688 records.
   - The reported values in the history JSON reflect **UNCALIBRATED (Raw Model) ECE**:
     - 12-lead Uncalibrated ECE: `0.01492` (Recomputed: `0.01491`)
     - 10-lead Uncalibrated ECE: `0.01455` (Recomputed: `0.01455`)
     - 8-lead Uncalibrated ECE:  `0.01530` (Recomputed: `0.01530`)
     - 6-lead Uncalibrated ECE:  `0.01639` (Recomputed: `0.01639`)
     - 4-lead Uncalibrated ECE:  `0.01626` (Recomputed: `0.01626`)
     - 2-lead Uncalibrated ECE:  `0.01639` (Recomputed: `0.01639`)
     - 1-lead Uncalibrated ECE:  `0.01856` (Recomputed: `0.01856`)
3. **True Calibrated Per-Configuration ECE (Independently Computed):**
   - When the fitted temperature scaling parameters are applied to each configuration's predictions, the post-calibrated ECE values are:
     - **12-lead Calibrated ECE:** **`0.01128`**
     - **10-lead Calibrated ECE:** **`0.01110`**
     - **8-lead Calibrated ECE:**  **`0.01118`**
     - **6-lead Calibrated ECE:**  **`0.01342`**
     - **4-lead Calibrated ECE:**  **`0.01361`**
     - **2-lead Calibrated ECE:**  **`0.01349`**
     - **1-lead Calibrated ECE:**  **`0.01416`**

**Audit Verdict:** Mathematical integrity is verified. The values reported in the per-configuration JSON reflect uncalibrated probabilities directly emitted by `evaluate_model(compute_cis=False)`. Post-calibration strictly improves calibration across all 7 configurations.

---

## 7. Operational & Statistical Evidence Verification

Every required evidence artifact was cross-examined against physical logs:

1. **Inference Latency & Throughput:**
   - Single-sample latency: **`0.887 ms`**
   - Inference throughput: **`1,127.88 samples / second`** (CPU, 16 threads)
   - Status: Verified present in `control_training_history_seed42.json`.
2. **Empirical 95% Bootstrap Confidence Intervals (200 Bootstraps):**
   - Macro-AUROC 95% CI: **`[0.9434, 0.9471]`**
   - Macro-F1 95% CI: **`[0.6884, 0.7005]`**
   - Status: Verified present in `control_training_history_seed42.json`.
3. **Per-Label Confusion Counts & Sample Support (Seed 42 Pooled):**

| Disease Target | Observed Records | Positives | Negatives | True Positives (TP) | False Positives (FP) | False Negatives (FN) | True Negatives (TN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NORM** | 25,725 | 7,896 | 17,829 | 6,159 | 2,078 | 1,737 | 15,751 |
| **AFIB** | 25,270 | 2,723 | 22,547 | 2,019 | 527 | 704 | 22,020 |
| **RBBB** | 23,912 | 1,169 | 22,743 | 664 | 144 | 505 | 22,599 |
| **LBBB** | 24,752 | 945 | 23,807 | 666 | 158 | 279 | 23,649 |
| **SBRAD** | 21,924 | 3,696 | 18,228 | 3,276 | 807 | 420 | 17,421 |
| **STACH** | 21,931 | 2,128 | 19,803 | 1,972 | 262 | 156 | 19,541 |
| **1AVB** | 22,407 | 987 | 21,420 | 276 | 279 | 711 | 21,141 |
| **MI** | 22,785 | 4,088 | 18,697 | 1,802 | 1,343 | 2,286 | 17,354 |

4. **Underpowered Label Handling:**
   - Lowest observed positive count is `LBBB` with **945 positive cases** (threshold is $\ge 30$).
   - `underpowered_warnings: []` is confirmed empty. Statistical power is sufficient across all 8 classes.
5. **Per-Dataset Generalization Breakdown (Seed 42):**
   - **PTB-XL ($N=15,253$):** Macro-AUROC = **`0.9367`** | Macro-F1 = **`0.6323`**
   - **Chapman-Shaoxing ($N=7,098$):** Macro-AUROC = **`0.9620`** | Macro-F1 = **`0.6178`**
   - **MIMIC-IV-ECG ($N=3,465$):** Macro-AUROC = **`0.9194`** | Macro-F1 = **`0.6700`**
6. **Full Test Suite Coverage:**
   - Production test execution verified: `pytest tests/` produced **107 passed, 1 skipped in 61.91s**.
   - Zero test failures or unexpected warnings.

---

## 8. Freeze Contract Completeness & Provenance Audit

[`runs/module4_v3/MODULE4_FROZEN.json`](file:///e:/ECG/runs/module4_v3/MODULE4_FROZEN.json) was audited across all required contract domains:

1. **Preregistration & Protocol Integrity:**
   - 6 protocol documents tracked under `preregistration.protocol_integrity`: `reports/module4_evaluation_protocol.md`, `PROJECT_PHASES_SPECIFICATION.md`, `reports/evidence_observability_config.json`, `configs/control_path_training_config.json`, `configs/lead_configurations.json`, and `configs/benchmark_candidates.json`. All 6 match disk SHA-256 digests (`Match=True`).
2. **Ratified Decision Records:**
   - Decisions U-01 (first failure boundary definition), U-02 (dual step & lead count gaps), U-03 (micro-aggregation grounding), and U-04 (contiguous-lead MI rule) are explicitly sealed under `preregistration.ratified_decisions`.
3. **Source Code Snapshots:**
   - 7 source code files tracked under `source_code_integrity`: `16_train_reasoning_transformer.py`, `reasoning_transformer.py`, `context_builder.py`, `dataset.py`, `multi_cohort_loader.py`, `freeze_contract.py`, and `clinical_extractor.py`. All 7 match disk SHA-256 digests (`Match=True`).
4. **Parent Module Frozen Integrity:**
   - Module 1 v2 frozen JSON & split manifest, Module 2 v1 frozen JSON, and Module 3 v1 frozen JSON are bound with verified matching SHA-256 checksums.
5. **Backbone & Model Architecture:**
   - ECG-FM backbone checkpoint path (`checkpoints/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt`) and SHA-256 (`4d0142bcb4...`) verified.
   - Transformer hyperparameters (d_model=256, d_ecg=768, nhead=4, layers=2, d_ff=512, candidates=8) sealed.
6. **Candidate Disease Space & Lead Configurations:**
   - 8 disease acronyms and unobserved label handling policy (`EXPLICIT_BINARY_OBSERVATION_MASK`) verified.
   - All 7 lead configurations (`12-, 10-, 8-, 6-, 4-, 2-, 1-lead`) verified.
7. **Checkpoints & Calibration Artifacts by Seed:**
   - Best and final checkpoint paths and exact SHA-256 hashes recorded for all 3 seeds (`42`, `1337`, `2024`).
   - Post-hoc temperature calibration vectors ($T_c \in \mathbb{R}^8$), mean temperatures, and pre/post calibration metrics (NLL, Brier, ECE per class) are fully embedded for each seed.
8. **Workload, Split Accounting & Decision Records:**
   - Exact canonical record counts (Train: 25,825; Validation: 3,688; Test: 7,375) and patient counts sealed.
   - Multi-configuration policy (`CARTESIAN_ALL_CONFIGURATIONS`) and class balancing policy (`NONE_AUTHORIZED`) sealed.
9. **Validation Prediction Output Binding:**
   - `per_record_validation_outputs` binds all three prediction files by relative path (`validation_predictions_seed{42,1337,2024}.csv`).
   - *Audit Finding (CSV Hashes):* While checkpoint hashes are recorded with full cryptographic SHA-256 digests, the prediction CSV SHA-256 checksums were bound by filepath rather than hash within the JSON schema. The independent audit verified their physical disk SHA-256 hashes directly:
     - `validation_predictions_seed42.csv`: `d14043973a98d6a98b4f03a35cad3e3391ed47ec4d9e8f822df880ce1e3687e8`
     - `validation_predictions_seed1337.csv`: `58b6451f9586e9ee0e2ff29de6af830aa606b2990e3f2ac51d20bbe37ec8ed57`
     - `validation_predictions_seed2024.csv`: `aa809925b8a0ea405c0bdf5b3f639484693a049f026a76f9eb5b2af2cb951210`


---

## 9. Test Partition Zero-Touch Quarantine Verification

The physical filesystem and execution logs were audited to verify that no test-set information was leaked:

1. **Split Quarantine Guard:** In [`src/reasoning/multi_cohort_loader.py:120-131`](file:///e:/ECG/src/reasoning/multi_cohort_loader.py#L120-L131), any call requesting test split records without `allow_test=True` immediately raises `PermissionError`.
2. **Training Script Ingestion:** In `scripts/16_train_reasoning_transformer.py`, lines 1050–1066 strictly pass `split="train"` and `split="validation"`. No call requesting `split="test"` exists in the training pipeline.
3. **Observation Cache Inspection:** In [`runs/module4_v3/observations/`](file:///e:/ECG/runs/module4_v3/observations/), exactly 14 files exist: 7 `obs_train_*.json` and 7 `obs_validation_*.json`. **Zero test observation files exist.**
4. **Metrics Directory Inspection:** In [`runs/module4_v3/metrics/`](file:///e:/ECG/runs/module4_v3/metrics/), only validation predictions and history reports exist. **Zero test predictions exist.**
5. **Traces Directory Inspection:** [`runs/module4_v3/traces/`](file:///e:/ECG/runs/module4_v3/traces/) contains **zero files**.
6. **Split Manifest Target Masking:** In [`runs/module4_v3/splits/split_manifest.csv`](file:///e:/ECG/runs/module4_v3/splits/split_manifest.csv), all 8 disease columns for all 7,375 test split records are strictly blank/`NaN`.

**Quarantine Verdict:** Test Folds 9–10 (7,375 records, 51,625 physical arrays across 7 configurations) have maintained **strict 100% zero-touch isolation**.

---

## 10. Audit Summary Table

| Audit Criterion | Verification Method | Outcome |
| :--- | :--- | :---: |
| 1. Checkpoint Hashes | SHA-256 cross-check against `MODULE4_FROZEN.json` | **PASS** |
| 2. 7 Configurations $\times$ 3 Seeds | Record count & configuration balance verification | **PASS** |
| 3. Metric Recomputation | Mathematical recomputation from raw prediction CSVs | **PASS** |
| 4. Prediction CSV Schema | Complete column presence & NaN masking validation | **PASS** |
| 5. Epoch 3 Selection Rationale | Validation BCE loss minimization reconciliation | **PASS** |
| 6. ECE Calibration Reconciliation | Scope isolation (uncalibrated per-config vs calibrated pooled) | **PASS** |
| 7. Statistical & Coverage Evidence | Latency, throughput, CIs, confusion matrices, test suite | **PASS** |
| 8. Freeze Contract Integrity | Comprehensive schema & code snapshot verification | **PASS** |
| 9. Zero Test Data Leakage | Provenance, logs, and directory quarantine verification | **PASS** |

---

## Final Verdict

```
READY FOR EXPLICIT RUN_TEST AUTHORIZATION
```

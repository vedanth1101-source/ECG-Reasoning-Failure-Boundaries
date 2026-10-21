# Module 4 — Stage 0 Read-Only Repository Inspection & Gate 0 Preparation Report

**Date:** 2026-10-14  
**Project:** Reasoning Failure Boundaries in ECG Foundation Models Under Progressive Physiological Information Loss  
**Phase:** Module 4 (Dual-Path Clinical Reasoning & Trace Generation) — Stage 0  
**Inspection Mode:** Strictly Read-Only (Zero model training, zero parameter updates, zero latent extraction, zero split creation, zero test label inspection)  
**Collision Verdict:** **PASS** (`runs/module4_v3/` does not exist; no naming collision or overwriting hazard)  
**Stage 0 Gate Status:** **STAGE 0 COMPLETE — PENDING USER AUTHORIZATION OF DECISIONS G0.1–G0.6**

---

## 1. Executive Summary & Verification Matrix

An exhaustive read-only inspection was executed across the repository, dataset manifests, split allocations, lead-configuration variants, foundation model checkpoints, observation extractors, cached representations, and audit histories.

All checks were executed directly against disk artifacts. Every finding is substantiated by concrete file paths, line numbers, hashes, and record counts.

| Item # | Determination Dimension | Status | Concrete Evidence / Source of Truth |
| :---: | :--- | :---: | :--- |
| **1** | Actual paths & record counts for all 7 variants across train, val, test | **PASS** | `runs/module2_v1/masked_leads/` (258,216 arrays); `runs/module3_v1/representations/` (258,216 vectors) |
| **2** | Canonical ECG record ID consistency across all 7 variants | **PASS** | `record_ids.json` matches row-for-row with `runs/module1_v2/splits/split_manifest.csv` |
| **3** | Split assignment consistency & mutual exclusion across all variants | **PASS** | $\text{Tr} \cap \text{Val} = 0$, $\text{Tr} \cap \text{Te} = 0$, $\text{Val} \cap \text{Te} = 0$; 36,888 unique records |
| **4** | Cohort counts, patient IDs, label mapping, duplicates, & leakage | **PASS\*** | 36,888 modeling records; 0 patient leakage across splits; `*`13 Chapman duplicate pairs preserved |
| **5** | Prior observation extractors & cached latents usage of val/test records | **FAIL** | `data/observations/ptbxl/observation_manifest.json` extracted on 2,198 Fold 10 test records |
| **6** | Prior exposure to PTB-XL Fold 9 and Fold 10 | **PARTIAL** | Frozen splits isolate Folds 9 & 10 in `test`, but legacy `scripts/14_prepare_control_path.py` read test files |
| **7** | ECG-FM checkpoint provenance, architecture, preprocessing, & audits | **PASS** | Checkpoint SHA-256 `4d0142bc...` matches official Hugging Face Git LFS pointer OID exactly |
| **8** | Conflicts between repository state and Module 4 baseline | **FAIL** | Legacy scripts/configs bind to outdated un-versioned `data/` instead of frozen `runs/` |

---

## 2. Comprehensive Findings for the Eight Required Determinations

### Determination 1: Actual Paths and Record Counts Across All Seven Lead Configurations
The seven fixed lead configurations defined by `configs/lead_configurations.json` are:
1. `12-lead` (C12_FULL: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6)
2. `10-lead` (C10_PREC_LIMB: I, II, aVR, aVL, V1, V2, V3, V4, V5, V6)
3. `8-lead` (C8_INDEPENDENT: I, II, V1, V2, V3, V4, V5, V6)
4. `6-lead` (C6_LIMB_6: I, II, III, aVR, aVL, aVF)
5. `4-lead` (C4_EINTHOVEN_SEPTAL: I, II, III, V1)
6. `2-lead` (C2_TELEMETRY: II, V1)
7. `1-lead` (C1_SINGLE_LEAD: II)

#### A. Module 2 Masked ECG Signals (`runs/module2_v1/masked_leads/`)
Every variant contains physically masked NumPy arrays of shape `(12, 1000)`, `float32`, standard units $\text{mV}$, $100\text{ Hz}$ sampling rate, in subdirectories `<variant>/<split>/records/<global_record_id>.npy`:

| Variant | Training Records (`train/records/`) | Validation Records (`val/records/`) | Test Records (`test/records/`) | Total Records per Variant |
| :--- | :---: | :---: | :---: | :---: |
| **12-lead** | 25,825 | 3,688 | 7,375 | 36,888 |
| **10-lead** | 25,825 | 3,688 | 7,375 | 36,888 |
| **8-lead** | 25,825 | 3,688 | 7,375 | 36,888 |
| **6-lead** | 25,825 | 3,688 | 7,375 | 36,888 |
| **4-lead** | 25,825 | 3,688 | 7,375 | 36,888 |
| **2-lead** | 25,825 | 3,688 | 7,375 | 36,888 |
| **1-lead** | 25,825 | 3,688 | 7,375 | 36,888 |
| **Total Physical Arrays** | **180,775** | **25,816** | **51,625** | **258,216** |

*Accompanying per-split manifests:* Each split folder `<variant>/<split>/` contains `labels.csv`, `lead_availability_mask.csv`, `metadata.csv`, and `observation_mask.csv`.

#### B. Module 3 Foundation Model Representations (`runs/module3_v1/representations/`)
Every variant directory contains both individual record vectors and consolidated matrix files:
- Individual files: `<variant>/<split>/records/<global_record_id>.npy` (vector shape `(768,)`, `float32`, exactly 3,200 bytes uniform).
- Consolidated matrices: `<variant>/<split>/representations.npy` (train: `(25825, 768)`, $79,334,528\text{ B}$; validation: `(3688, 768)`, $11,330,688\text{ B}$; test: `(7375, 768)`, $22,656,128\text{ B}$).
- Split manifests: `<variant>/<split>/extraction_manifest.csv` and `record_ids.json`.
- Total individual representation files: **258,216**. Total consolidated matrices: **21**. Total representation files: **258,237**.

---

### Determination 2: Canonical Record ID Consistency Across Variants
- **Evaluation:** Evaluated the ordered list of record IDs across all seven variants in Module 3 (`record_ids.json`) against the master partition manifest `runs/module1_v2/splits/split_manifest.csv`.
- **Findings:**
  - `train`: Exactly 25,825 records. All 7 variants have 100% identical record IDs in identical row order.
  - `validation`: Exactly 3,688 records. All 7 variants have 100% identical record IDs in identical row order.
  - `test`: Exactly 7,375 records. All 7 variants have 100% identical record IDs in identical row order.
- **Verdict:** **PASS**.

---

### Determination 3: Split Assignment Consistency and Mutual Exclusion
- **Evaluation:** Evaluated set intersections of `global_record_id` across `train`, `validation`, and `test` in `runs/module1_v2/splits/split_manifest.csv`:
  - $\text{Train} \cap \text{Validation} = \emptyset$ (count: 0)
  - $\text{Train} \cap \text{Test} = \emptyset$ (count: 0)
  - $\text{Validation} \cap \text{Test} = \emptyset$ (count: 0)
  - $\text{Total Union} = 25,825 + 3,688 + 7,375 = 36,888$ unique records.
- **Verdict:** **PASS**.

---

### Determination 4: Dataset Counts, Patient IDs, Label Mappings, Duplicates, and Leakage
Inspection of `runs/module1_v2/MODULE1_FROZEN.json` and `runs/module1_v2/splits/split_manifest.csv` revealed:

1. **Cohort Breakdown & Patient Uniqueness:**
   - **PTB-XL:** 21,797 records across 18,867 unique patients. Stratified patient-level partition: Folds 1–7 = train (15,260), Fold 8 = validation (2,179), Folds 9 & 10 = test (4,358). Patient leakage across splits: **0** (Train-Val: 0, Train-Test: 0, Val-Test: 0).
   - **Chapman-Shaoxing:** 10,142 records. Native patient IDs were missing in the raw repository; operational fallback `patient_id = record_id` was approved at Gate 0. Cross-split patient overlap under record ID: **0**.
   - **MIMIC-IV-ECG:** 4,949 records across 4,949 unique `subject_id`s (1:1 record-to-subject ratio). Cross-split patient overlap: **0**.
2. **Duplicate Records Audit:**
   - Raw signal hash scan across all 36,888 records identified 36,875 unique hashes.
   - PTB-XL duplicate signals: 0. MIMIC-IV-ECG duplicate signals: 0. Cross-cohort duplicate signals: 0.
   - Chapman-Shaoxing: Exactly 13 identical signal pairs exist under different record IDs (4 pairs within train, 9 pairs across splits). Confirmed and documented under the approved Gate 0 operational waiver.
3. **Quality Control & Missing Data:**
   - 69 records failed QC and were excluded from the modeling pool.
   - 89 Chapman entries had missing raw `.mat` files and were excluded from the inventory.
   - All 36,888 modeling records have complete physical `.npy` files and complete metadata rows. Missing data: **0.0%**.
4. **Target Prevalence & Observability (36,888 Eligible Records):**
   - Derived from `runs/module1_v2/labels/record_labels.csv` and `runs/module1_v2/labels/label_observation_mask.csv`:
     - `NORM`: 36,752 obs (99.6%) | 11,242 pos (30.6% of obs) | 25,510 neg | 136 unobs
     - `AFIB`: 36,010 obs (97.6%) | 3,893 pos (10.8% of obs) | 32,117 neg | 878 unobs
     - `RBBB`: 34,380 obs (93.2%) | 1,680 pos (4.9% of obs) | 32,700 neg | 2,508 unobs
     - `LBBB`: 35,422 obs (96.0%) | 1,372 pos (3.9% of obs) | 34,050 neg | 1,466 unobs
     - `SBRAD`: 31,354 obs (85.0%) | 5,285 pos (16.9% of obs) | 26,069 neg | 5,534 unobs
     - `STACH`: 31,350 obs (85.0%) | 3,082 pos (9.8% of obs) | 28,268 neg | 5,538 unobs
     - `1AVB`: 31,983 obs (86.7%) | 1,422 pos (4.4% of obs) | 30,561 neg | 4,905 unobs
     - `MI`: 32,484 obs (88.1%) | 5,877 pos (18.1% of obs) | 26,607 neg | 4,404 unobs
- **Verdict:** **PASS (with documented Chapman duplicate signal limitation)**.

---

### Determination 5: Prior Observation Extractors and Latent Caches
Inspection of legacy extraction scripts and artifacts revealed significant data exposure hazards in the un-versioned `data/` directory:

1. **Test-Set Observation Extraction:**
   - `data/observations/ptbxl/observation_manifest.json` explicitly states:
     `"total_test_records": 2198` (PTB-XL Fold 10 test records under legacy 80:10:10 split). Observation extraction was executed directly on test records.
2. **Under-Trained Linear Observation Heads:**
   - `checkpoints/linear_observation_heads.pt` (155 KB) was trained on only $N_{\text{train}} = 1,000$ legacy records and validated on $N_{\text{val}} = 300$ records (`reports/observation_extractor_audit.json`). ST-segment multi-lead accuracy was low (64.31%).
3. **ResNet Baseline Latent Pollution:**
   - `data/latents/ptbxl/latent_manifest.json` explicitly documents `z_12_lead.npy` through `z_1_lead.npy` as originating from a deprecated `1D-ResNet baseline + Bottleneck Adapter` marked `BASELINE_ABLATION_ONLY`.
4. **Outdated Split Cache:**
   - `data/latents/ptbxl/ecg_fm_train_val_manifest.json` cached 17,418 train and 2,183 val records for PTB-XL only under the legacy 80:10:10 split.
- **Verdict:** **FAIL**. Legacy `data/` caches contain test-set observation extractions, outdated split partitions, and ResNet baseline latents. They cannot be used as inputs for Module 4.

---

### Determination 6: Prior Exposure to PTB-XL Fold 9 and Fold 10
1. **Module 1 v2 Partitioning:**
   - In `runs/module1_v2/splits/split_manifest.csv`:
     - `strat_fold` 1–7 (15,260 records) $\to$ `train`
     - `strat_fold` 8 (2,179 records) $\to$ `validation`
     - `strat_fold` 9 (2,179 records) $\to$ `test`
     - `strat_fold` 10 (2,179 records) $\to$ `test`
   - Both Fold 9 and Fold 10 are quarantined in the final test split (4,358 total test records from PTB-XL).
2. **Legacy Preparation Code Exposure:**
   - In `scripts/14_prepare_control_path.py` (line 120), 5 test records from `data/benchmark_test/ptbxl/full_12lead.npy` were loaded for a smoke test.
- **Verdict:** **PARTIAL**. The frozen splits strictly isolate Folds 9 & 10 in `test`, but legacy scripts touched Fold 10. Module 4 must enforce absolute isolation of the test split during all preparation, training, hyperparameter selection, and calibration routines.

---

### Determination 7: ECG-FM Provenance, Preprocessing Compatibility, and Audits
1. **Model Checkpoint Provenance:**
   - Local checkpoint: `checkpoints/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt`
   - Size: 1,090,825,421 bytes (~1.02 GB).
   - SHA-256 Hash: `4d0142bcb485eb9f0c7845e0c19ff3463f6ae9d0e458eab69136efe90ceb9b7e`
   - Authentication: Exactly matches the Hugging Face Git LFS pointer OID from the official repository (`https://huggingface.co/wanglab/ecg-fm`).
2. **Architecture Reconciliation:**
   - Architecture: `wav2vec2_cmsc` (`Wav2Vec2CMSCModel`).
   - Total Parameters: 90,883,072 parameters (100% frozen, 0 trainable).
   - Configuration: 4 convolution downsampling blocks (kernel size 2, stride 2), 12 Transformer encoder layers, 768 hidden dimension, 12 attention heads, 3072 feedforward dimension.
   - Strict Load: PASS (0 missing keys, 0 unexpected keys).
3. **Preprocessing Compatibility:**
   - Input shape $(B, 12, 1000)$, `float32`, standard units $\text{mV}$, $100\text{ Hz}$ sampling rate.
   - Lead order strictly matches canonical 12-lead layout: `[I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]`.
4. **Existing Post-Freeze Audit Evidence:**
   - `runs/module3_v1/reports/module3_post_freeze_findings.csv` records 11 formal audit findings (F-01 through F-11).
   - Numerical drift calculations match saved summaries to within $< 10^{-15}$ precision.
   - Documented minor narrative correction: 4-lead cosine similarity (0.904078) is higher than 6-lead (0.877548) because 6-lead eliminates all precordial leads (V1–V6), whereas 4-lead retains V1 (non-nested spatial subspaces).
- **Verdict:** **PASS**.

---

### Determination 8: Repository vs Baseline Conflicts
The inspection identified six major conflicts between existing code in the repository and the frozen Module 1–3 baseline:

1. **Conflict C-01 (Legacy Data Binding):**
   - Existing reasoning datasets (`src/reasoning/dataset.py`, `src/reasoning/multi_cohort_loader.py`, and `scripts/16_train_reasoning_transformer.py` lines 270–298) load from `data/latents/ptbxl/` and `data/observations/ptbxl/` (17,418 records, PTB-XL only) rather than reading the 258,216 frozen representations from `runs/module3_v1/representations/` and masked signals from `runs/module2_v1/masked_leads/`.
2. **Conflict C-02 (Outdated Split Partitioning Configuration):**
   - `configs/control_path_training_config.json` lines 12–42 still specifies the legacy 80:10:10 partition (Folds 1–8 train, Fold 9 val, Fold 10 test; total 36,957 records), conflicting with the frozen 70:10:20 partition in `runs/module1_v2/MODULE1_FROZEN.json` (Folds 1–7 train, Fold 8 val, Folds 9–10 test; total 36,888 records).
3. **Conflict C-03 (Observation Extractor Scale & Supervision):**
   - `checkpoints/linear_observation_heads.pt` was trained on only 1,000 legacy records. It does not cover the full 25,825 training records of the 3-cohort ecosystem.
4. **Conflict C-04 (Observation Schema Discrepancy):**
   - `LinearObservationExtractor` outputs a nested dictionary structure (`{"rhythm": {"value": ...}}`), whereas `ControlContextBuilder` expects a flat dictionary (`{"rhythm": ...}`).
5. **Conflict C-05 (Specification Loss Ambiguity):**
   - `PROJECT_PHASES_SPECIFICATION.md` Equation 350 previously noted a Softmax single-label formulation, whereas the authoritative benchmark is multi-label binary cross-entropy (`nn.BCEWithLogitsLoss()`) across 8 independent disease targets.
6. **Conflict C-06 (Test-Set Quarantine Policy in Scripts):**
   - Preparation scripts (such as `scripts/14_prepare_control_path.py` line 120) sampled records from test benchmarks, violating the zero-touch test quarantine mandate.
- **Verdict:** **FAIL**. All six conflicts must be resolved under Gate 0 before Stage 1 implementation.

---

## 3. Unresolved Risks & Hazards

1. **Risk R-01 (Observation Extractor Dependency):** If `LinearObservationExtractor` is used without retraining, its low ST accuracy (64.31%) and toy sample size (1,000 records) could artificially degrade reasoning performance. Alternatively, using the deterministic rule-based `ClinicalObservationExtractor` directly on active leads $\mathbf{m}_l$ ensures 100% physiological ground truth for observable leads without relying on an under-trained linear probe.
2. **Risk R-02 (Premature Test Set Exposure):** Test set evaluation must not occur during training or temperature calibration. Test split data (51,625 arrays across 7 variants) must remain quarantined until final evaluation.
3. **Risk R-03 (Class Imbalance in Multi-Label Calibration):** Substantial class imbalance exists in the validation set (e.g., LBBB has 148 positives vs 3,540 negatives). Temperature scaling optimization must employ robust numerical optimization (e.g., Brent scalar minimization) to prevent degenerate temperatures.

---

## 4. The Exact Gate 0 Operational Decisions (G0.1–G0.6)

Before proceeding to Stage 1, please review and provide explicit decisions on the following six baseline requirements:

### Decision G0.1: Authoritative Input Data Source Binding
- **Option A (Recommended):** Bind Module 4 strictly to the frozen, versioned artifacts in `runs/module2_v1/masked_leads/` (masked signals and masks) and `runs/module3_v1/representations/` (pre-extracted 768-d representations), referencing the 36,888 records in `runs/module1_v2/splits/split_manifest.csv`. Permanently deprecate legacy `data/latents/ptbxl/` and `data/observations/ptbxl/`.
- **Option B:** Recompute representations dynamically on-the-fly from masked signals during training using frozen ECG-FM (high compute overhead).

### Decision G0.2: Structured Clinical Observation Extractor Architecture
- **Option A (Recommended):** Use deterministic rule-based clinical observations directly from active leads via `src/observations/clinical_extractor.py`, with masked leads strictly set to `UNAVAILABLE_MASKED`. This ensures 100% physiological ground truth for observable leads without relying on an under-trained linear probe.
- **Option B:** Retrain the linear observation heads (`src/reasoning/observation_extractor.py`) on all 25,825 training representations from Module 3 before feeding observations into the Reasoning Transformer.

### Decision G0.3: Test Split Absolute Quarantine Policy
- **Mandate:** Reaffirm absolute quarantine of the 7,375 test records per variant (51,625 total arrays). Zero test files may be accessed, read, loaded, or inspected during model development, hyperparameter selection, checkpoint saving, or temperature calibration. Test evaluation occurs exclusively in Stage 5 after all models and calibration parameters are frozen.
- **Decision:** Approve strict quarantine enforcement across all Module 4 scripts.

### Decision G0.4: Control Reasoning Transformer Training Scope
- **Option A (Recommended):** Train a single unified Control Reasoning Transformer across all 7 lead configurations by training on full 12-lead representations and evaluating across degraded leads (or using dynamic lead-availability masking during training).
- **Option B:** Train 7 separate independent Control Reasoning Transformers (one for each lead configuration variant).

### Decision G0.5: Multi-Label Loss and Calibration Protocol
- **Specification:** Ratify multi-label binary cross-entropy loss (`nn.BCEWithLogitsLoss()`) over 8 independent logits with sigmoid activation (superseding any historical mention of Softmax). Per-disease temperature scaling $T_i > 0$ for $i \in \{1, \dots, 8\}$ will be fitted strictly on the validation split (3,688 records) by minimizing class-wise binary negative log-likelihood.
- **Decision:** Approve this multi-label loss and validation-only calibration protocol.

### Decision G0.6: Versioned Target Output Directory Authorization
- **Specification:** Authorize `runs/module4_v3/` as the isolated, versioned destination directory for all Module 4 artifacts (checkpoints, training history, calibration parameters, reasoning traces, degradation curves, and freeze manifests).
- **Decision:** Authorize `runs/module4_v3/`.

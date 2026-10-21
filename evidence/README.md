# Electrocardiogram Foundation Model Reasoning Under Progressive Physiological Information Loss
## Comprehensive Evidence Archive and Cross-Module Technical Audit (Phase 1)

**Archive Root:** `evidence/`  
**Execution Phase:** Phase 1 — Comprehensive Evidence Collection, Audit Archive, and Module-Wise Documentation  
**Audit Completion Date:** 2026-10-16  
**Operating Principles:** Strictly Evidence-Driven • Zero Model Retraining • Zero Test-Set Unblinding • Strict Checkpoint Preservation  

---

## 1. Executive Summary

This archive establishes the authoritative, empirical, and reproducible evidence base for the multi-center research project: *“Reasoning Failure Boundaries in ECG Foundation Models Under Progressive Physiological Information Loss.”*

Covering all four project modules, this archive documents what each subsystem does, why specific design choices were adopted, what problems they solve, what empirical audits were conducted, what claims are verified or retracted, and what methodological limitations constrain downstream clinical conclusions.

The archive catalogs **86 distinct artifacts** (CSVs, JSON manifests, and technical audit documents), backed by:
- **36,888 eligible 12-lead ECG recordings** harmonized from three international cohorts (PTB-XL, Chapman-Shaoxing, MIMIC-IV-ECG).
- **258,216 degraded signal arrays** across seven canonical lead montages ($12 \rightarrow 10 \rightarrow 8 \rightarrow 6 \rightarrow 4 \rightarrow 2 \rightarrow 1$ lead).
- **258,216 foundation model representation vectors** ($\mathbb{R}^{768}$) extracted using the official, frozen ECG-FM model (McKeen et al., *JAMIA Open*, 2025; 90.88M parameters).
- **25,816 validation clinical reasoning trace dossiers** evaluated under a corrected four-tier evidence taxonomy across 3,688 validation records.

---

## 2. Cross-Module Pipeline Architecture and Data Flow

The research pipeline integrates four sequential modules into an end-to-end framework. Evaluating this complete pipeline provides insights that cannot be obtained by studying raw signals, static embeddings, or black-box classifiers in isolation.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ MODULE 1: Dataset Harmonization, Disease Annotation, and Split Accounting               │
│ - Input: Raw ECG recordings from PTB-XL, Chapman-Shaoxing, and MIMIC-IV-ECG            │
│ - Processing: ADC conversion, 100 Hz polyphase resampling, 10.0s windowing (12, 1000)  │
│ - Quality Control: 69 rejections (P_5 QC threshold); 36,888 eligible records retained  │
│ - Labels: 8 binary targets (NORM, AFIB, RBBB, LBBB, SBRAD, STACH, 1AVB, MI)            │
│ - Observation Masks: o_d in {0,1} to handle non-uniform cohort disease coverage        │
│ - Data Splits: 10-fold patient-clustered split (Train: 25,825; Val: 3,688; Test: 7,375)│
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Harmonized Signals (12, 1000) mV float32
                                            │ Observation Masks & Metadata
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ MODULE 2: Progressive Lead Degradation & Spatial Masking                               │
│ - Input: 36,888 harmonized 12-lead signal arrays                                       │
│ - Configurations: 7 canonical montages (12, 10, 8, 6, 4, 2, 1 lead)                    │
│ - Masking: Deterministic binary mask m in {0,1}^12; clean zeroing (x_deg = x * m)      │
│ - Constraint: Zero synthetic imputation or vector reconstruction                       │
│ - Output: 258,216 standardized input arrays of uniform shape (12, 1000)                │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Standardized Degraded Arrays (12, 1000)
                                            │ Lead Availability Masks m in {0,1}^12
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ MODULE 3: ECG-FM Pretrained Latent Representation Extraction                           │
│ - Backbone: Official Wav2Vec2CMSCModel (McKeen et al., 2025; 90,883,072 params)        │
│ - Parameters: 100% frozen (requires_grad = False; eval mode)                           │
│ - Extraction: 12 Transformer layers -> temporal mean-pooling over 62 frames            │
│ - Output: 258,216 compact representation vectors z in R^768 (3,200 bytes per record)    │
│ - Geometric Discovery: 4-lead representations retain higher cosine similarity to       │
│   12-lead (0.9041) than 6-lead (0.8775) due to preservation of septal Lead V1          │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Cached Latent Vectors z in R^768
                                            │ Lead Availability Masks m in {0,1}^12
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ MODULE 4: Control-Path Reasoning, Calibration, and Evidence Grounding Audits           │
│ - Classifier: Shared MLP (780 -> 384 -> 192 -> 8) conditioned on [z || m]              │
│ - Optimization: Masked BCE loss (zero loss for unobserved conditions)                  │
│ - Calibration: Validation-only per-disease temperature scaling (ECE reduced by >55%)   │
│ - Reasoning Traces: 25,816 validation traces audited under 4-tier evidence taxonomy    │
│ - Empirical Findings: True physiological contradictions are flat (~1.0%-1.5%); large   │
│   reported spikes were channel-unavailability violations from missing precordial leads │
│ - Governance Disclosures: Test evaluation launched via automated hook without logged   │
│   literal RUN_TEST token (authorization-unverified); Baseline B-2 QRS defect cataloged │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. High-Level Summary by Module

### Module 1: Dataset Harmonization and Disease Label Universe
- **Directory:** `evidence/MODULE_1_DATASET/`
- **Key Artifacts:** [dataset_inventory.csv](file:///e:/ECG/evidence/MODULE_1_DATASET/dataset_inventory/dataset_inventory.csv), [disease_label_dictionary.csv](file:///e:/ECG/evidence/MODULE_1_DATASET/disease_label_analysis/disease_label_dictionary.csv), [disease_distribution_by_cohort.csv](file:///e:/ECG/evidence/MODULE_1_DATASET/disease_label_analysis/disease_distribution_by_cohort.csv), [disease_distribution_by_split.csv](file:///e:/ECG/evidence/MODULE_1_DATASET/disease_label_analysis/disease_distribution_by_split.csv), [split_accounting.csv](file:///e:/ECG/evidence/MODULE_1_DATASET/split_analysis/split_accounting.csv).
- **Core Findings:** Reconciled data universe from 37,046 source records down to exactly **36,888 eligible modeling records** (after 89 missing Chapman files and 69 QC exclusions). Established 10-fold split with 33,958 patient clusters. Identified and cataloged **13 duplicate signal pairs** in Chapman-Shaoxing (including 9 cross-split pairs) resulting from unlinked hospital recording fallbacks.
- **Module Guide:** [MODULE_1_DATASET/README.md](file:///e:/ECG/evidence/MODULE_1_DATASET/README.md).

### Module 2: Progressive Lead Degradation
- **Directory:** `evidence/MODULE_2_LEAD_DEGRADATION/`
- **Key Artifacts:** [lead_configuration_matrix.csv](file:///e:/ECG/evidence/MODULE_2_LEAD_DEGRADATION/configuration_definitions/lead_configuration_matrix.csv), [lead_availability_evidence.csv](file:///e:/ECG/evidence/MODULE_2_LEAD_DEGRADATION/lead_masks/lead_availability_evidence.csv), [cross_configuration_alignment_audit.md](file:///e:/ECG/evidence/MODULE_2_LEAD_DEGRADATION/record_alignment/cross_configuration_alignment_audit.md), [signal_integrity_audit.md](file:///e:/ECG/evidence/MODULE_2_LEAD_DEGRADATION/signal_integrity/signal_integrity_audit.md).
- **Core Findings:** Established the authoritative 7-configuration ladder:
  - 12-lead (all 12 active)
  - 10-lead (III, aVF masked)
  - 8-lead (III, aVR, aVL, aVF masked)
  - 6-lead (V1–V6 masked; frontal limb only)
  - 4-lead (aVR, aVL, aVF, V2–V6 masked; active I, II, III, V1)
  - 2-lead (active II, V1)
  - 1-lead (active II)
  Verified exact mathematical zeroing ($X \cdot m$) with zero synthetic imputation across all 258,216 arrays.
- **Module Guide:** [MODULE_2_LEAD_DEGRADATION/README.md](file:///e:/ECG/evidence/MODULE_2_LEAD_DEGRADATION/README.md).

### Module 3: Pretrained ECG-FM Representations
- **Directory:** `evidence/MODULE_3_ECG_FM/`
- **Key Artifacts:** [checkpoint_provenance.md](file:///e:/ECG/evidence/MODULE_3_ECG_FM/checkpoint_provenance/checkpoint_provenance.md), [model_architecture_spec.md](file:///e:/ECG/evidence/MODULE_3_ECG_FM/architecture/model_architecture_spec.md), [representation_configuration_inventory.csv](file:///e:/ECG/evidence/MODULE_3_ECG_FM/representation_extraction/representation_configuration_inventory.csv), [representation_drift_summary.csv](file:///e:/ECG/evidence/MODULE_3_ECG_FM/representation_statistics/representation_drift_summary.csv), [representation_integrity_audit.md](file:///e:/ECG/evidence/MODULE_3_ECG_FM/representation_statistics/representation_integrity_audit.md).
- **Core Findings:** Authenticated official checkpoint (`mimic_iv_ecg_physionet_pretrained.pt`, SHA-256: `4d0142bcb485eb...`) with 90,883,072 parameters (100% frozen). Audited 258,216 representation vectors ($\mathbb{R}^{768}$, 3,200 bytes each) with 100% finiteness. Discovered **representation drift non-monotonicity**: 4-lead representations exhibit lower drift from 12-lead than 6-lead representations (cosine similarity 0.9041 vs 0.8775) because 4-lead preserves septal lead V1.
- **Module Guide:** [MODULE_3_ECG_FM/README.md](file:///e:/ECG/evidence/MODULE_3_ECG_FM/README.md).

### Module 4: Control-Path Reasoning and Evidence Audits
- **Directory:** `evidence/MODULE_4_CONTROL_REASONING/`
- **Key Artifacts:** [protocol_and_architecture_evidence.md](file:///e:/ECG/evidence/MODULE_4_CONTROL_REASONING/protocol/protocol_and_architecture_evidence.md), [calibration_evidence.md](file:///e:/ECG/evidence/MODULE_4_CONTROL_REASONING/calibration/calibration_evidence.md), [evaluation_provenance.md](file:///e:/ECG/evidence/MODULE_4_CONTROL_REASONING/test_evaluation_provenance/evaluation_provenance.md), [baseline_comparison_limitations.md](file:///e:/ECG/evidence/MODULE_4_CONTROL_REASONING/baseline_comparisons/baseline_comparison_limitations.md), [reasoning_trace_evidence_register.csv](file:///e:/ECG/evidence/MODULE_4_CONTROL_REASONING/reasoning_traces/reasoning_trace_evidence_register.csv), [module4_limitations.md](file:///e:/ECG/evidence/MODULE_4_CONTROL_REASONING/limitations/module4_limitations.md).
- **Core Findings:** Verified multi-seed validation macro-AUROC (~0.891). Confirmed validation-only temperature scaling reduced Expected Calibration Error (ECE) from 0.0482 to 0.0215 (>55% gain). Audited 25,816 validation reasoning traces under the corrected four-tier taxonomy, demonstrating that true physiological contradictions remain flat (~1.0%–1.5%) while reported spikes were channel-unavailability errors. Fully documented the **test evaluation provenance breach** (automated hook without literal `RUN_TEST` token) and the **fatal QRS window bug in Baseline B-2** (samples 0–50), formally retracting unsupported claims.
- **Module Guide:** [MODULE_4_CONTROL_REASONING/README.md](file:///e:/ECG/evidence/MODULE_4_CONTROL_REASONING/README.md).

---

## 4. Evidence Quality and Claim Classification Matrix

Every primary scientific claim is classified under strict evidentiary standards:

| Primary Research Claim | Evidentiary Status | Supporting Artifact / Verification Path | Key Limitation or Context |
| :--- | :---: | :--- | :--- |
| **Harmonized dataset contains 36,888 records** | **VERIFIED** | `dataset_inventory.csv` & `split_accounting.csv` | Exactly 36,888 records across 10 folds. |
| **Complete patient isolation in splits** | **PARTIALLY VERIFIED** | `patient_grouping_and_overlap_audit.md` | Verified in PTB/MIMIC; 9 cross-split duplicate pairs in Chapman. |
| **Zero synthetic imputation in lead masking** | **VERIFIED** | `lead_availability_evidence.csv` & `signal_integrity_audit.md` | 100% of masked leads strictly 0.000 mV. |
| **ECG-FM backbone is official & frozen** | **VERIFIED** | `checkpoint_provenance.md` & `model_architecture_spec.md` | SHA-256 matches Hugging Face LFS; 0 grad updates. |
| **4-lead latent drift is lower than 6-lead** | **VERIFIED** | `representation_drift_summary.csv` | Lead V1 provides critical septal anchor. |
| **Validation macro-AUROC ~0.891** | **VERIFIED** | `training_validation_metric_inventory.csv` | Multi-seed reproducibility verified. |
| **Temperature calibration reduces ECE >55%** | **VERIFIED** | `calibration_evidence.md` | Fitted strictly on validation split; 0 test leakage. |
| **Test set evaluated with valid gate** | **UNRESOLVED / DEFECT** | `evaluation_provenance.md` | Automated trigger without logged literal `RUN_TEST` token. |
| **B-3 outperforms Clinical Rules (B-2)** | **UNSUPPORTED / RETRACTED** | `baseline_comparison_limitations.md` | B-2 QRS window defect (0-50 samples) invalidates comparator. |
| **B-3 statistically superior to B-1** | **PARTIALLY VERIFIED** | `baseline_comparison_limitations.md` | Descriptive advantage observed; paired DeLong test missing CSVs. |
| **Severe physiological contradiction spikes** | **UNSUPPORTED / RETRACTED** | `reasoning_trace_evidence_register.csv` | Auditor defect; spikes were channel-unavailability violations. |
| **Anterior MI drives 6-lead contradictions** | **UNSUPPORTED / RETRACTED** | `validation_trace_documentation_correction.md` | 0 cases of MI; 100% were bundle branch blocks (RBBB/LBBB). |
| **Lead V1 causally reduces contradictions** | **UNSUPPORTED / RETRACTED** | `validation_trace_documentation_correction.md` | 6-lead and 4-lead have identical contradiction counts (43 vs 43). |

---

## 5. Reviewer Entry Points and Navigation Guide

For peer reviewers, auditors, and clinical researchers, we recommend navigating the archive as follows:

1. **Master Evidence Index & Catalog:**
   - [evidence_index.csv](file:///e:/ECG/evidence/evidence_index.csv): Itemized table of all 86 cataloged artifacts with hashes, categories, and verification verdicts.
   - [evidence_manifest.json](file:///e:/ECG/evidence/evidence_manifest.json): Machine-readable cryptographic manifest with exact byte sizes and SHA-256 checksums.
2. **Module Technical Guides:**
   - [Module 1 Technical README](file:///e:/ECG/evidence/MODULE_1_DATASET/README.md)
   - [Module 2 Technical README](file:///e:/ECG/evidence/MODULE_2_LEAD_DEGRADATION/README.md)
   - [Module 3 Technical README](file:///e:/ECG/evidence/MODULE_3_ECG_FM/README.md)
   - [Module 4 Technical README](file:///e:/ECG/evidence/MODULE_4_CONTROL_REASONING/README.md)
3. **Comprehensive Completion & Verification Report:**
   - [EVIDENCE_COLLECTION_COMPLETION_REPORT.md](file:///e:/ECG/evidence/EVIDENCE_COLLECTION_COMPLETION_REPORT.md): Final completion audit, safety compliance certificate, and discrepancy resolution log.

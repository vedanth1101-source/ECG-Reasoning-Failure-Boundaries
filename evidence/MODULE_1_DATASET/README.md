# Module 1: Multi-Cohort Dataset Standardization, Eight-Label Harmonization, and Stratified Partitioning

**Module Designation:** Module 1 v2  
**Status:** Permanently Frozen (`FROZEN`)  
**Freeze Timestamp:** 2026-10-12 12:22:51 UTC  
**Primary Freeze Manifest:** [`audits/MODULE1_FROZEN.json`](file:///e:/ECG/evidence/MODULE_1_DATASET/audits/MODULE1_FROZEN.json)  
**Authoritative Split Manifest:** [`runs/module1_v2/splits/split_manifest.csv`](file:///e:/ECG/runs/module1_v2/splits/split_manifest.csv)  
**Checksum Manifest:** [`runs/module1_v2/checksums_module1_manifest.sha256`](file:///e:/ECG/runs/module1_v2/checksums_module1_manifest.sha256) (SHA-256: `47bb4547d8a39f4a5ec1697f59dfe0a104b79811642c39115d73579fed09301c`)

---

## 1. Module Purpose & Scientific Question Addressed

### 1.1 Purpose
Module 1 creates a standardized, quality-controlled, multi-cohort clinical ECG foundation dataset spanning three major public databases: **PTB-XL v1.0.3**, **Chapman-Shaoxing v1.0.0**, and **MIMIC-IV-ECG v1.0.0**. It harmonizes divergent clinical vocabularies into an eight-disease tri-state multi-label schema, enforces physical voltage calibration, and partitions recordings into a patient-group-aware **70:10:20** stratified split.

### 1.2 Scientific Question Addressed
*Can clinical ECG signals and heterogeneous multi-label annotations from divergent hospital institutions be harmonized into a mathematically standardized tensor benchmark without injecting false-negative assumptions or cross-partition patient leakage?*

---

## 2. Input and Output Artifacts

### 2.1 Input Raw Sources
- **PTB-XL (PhysioNet v1.0.3):** 21,799 raw recordings (10-second 12-lead, 500 Hz/100 Hz, SCP-ECG statements, clinical patient IDs).
- **Chapman-Shaoxing (PhysioNet v1.0.0):** 10,247 raw recordings (10-second 12-lead, 500 Hz, SNOMED-CT codes, 89 missing files in public release).
- **MIMIC-IV-ECG (PhysioNet v1.0.0 Sampled Universe):** 5,000 raw recordings (10-second 12-lead, 500 Hz, GE Marquette 12SL machine statements, clinical subject IDs).

### 2.2 Standardized Output Artifacts
- **Physical Signal Tensors:** 36,888 binary NumPy arrays in `runs/module1_v2/dataset/` (shape `12 x 1000`, float32, physical mV, 100 Hz).
- **Global Split Manifest:** [`runs/module1_v2/splits/split_manifest.csv`](file:///e:/ECG/runs/module1_v2/splits/split_manifest.csv) (36,888 rows containing IDs, metadata, split assignments, and tri-state labels).
- **Observation Masks:** `runs/module1_v2/labels/label_observation_mask.csv` (boolean observation indicators for all 8 target diseases).
- **Label Matrix:** `runs/module1_v2/labels/record_labels.csv` (numerical label matrix: 1.0, 0.0, or NaN).

---

## 3. Key Processes and Design Rationale

1. **Uniform Tensor Contract (`12 x 1000`, 100 Hz, float32, mV):**  
   Standardizes diverse sampling rates ($100 - 500\text{ Hz}$) and unit calibrations into identical physical dimensions compatible with the downstream ECG-FM foundation model.
2. **P_5 Automated Signal Quality Control:**  
   Rejects non-physiological flatlines, severe amplifier saturation ($>15\text{ mV}$), and DC bias offsets. Excluded exactly 69 records across the catalog universe.
3. **Tri-State Multi-Label Mapping with Observation Masks:**  
   Replaces naive zero-filling with an explicit three-state model ($1.0 = \text{positive}$, $0.0 = \text{negative}$, $\text{NaN} = \text{unknown/unobserved}$). Protects model training against thousands of false-negative gradient signals.
4. **Patient-Group Iterative Stratification:**  
   Splits data into 70% Train (25,825), 10% Validation (3,688), and 20% Test (7,375) across 10 folds while grouping multi-recording patients into the same fold.

---

## 4. Evidence Supporting Implementation Correctness

All core assertions are backed by reproducible evidence tables in this archive:

| Evidence Dimension | Evidence File Path | Key Quantitative Finding |
| :--- | :--- | :--- |
| **Dataset Accounting** | [`dataset_inventory/dataset_inventory.csv`](file:///e:/ECG/evidence/MODULE_1_DATASET/dataset_inventory/dataset_inventory.csv) | 37,046 catalog $\rightarrow$ 89 missing excluded $\rightarrow$ 69 QC rejected $\rightarrow$ **36,888 eligible records**. |
| **Disease Label Dictionary** | [`disease_label_analysis/disease_label_dictionary.csv`](file:///e:/ECG/evidence/MODULE_1_DATASET/disease_label_analysis/disease_label_dictionary.csv) | Full mapping logic, source codes, and positive definitions across all 8 target diseases. |
| **Cohort Prevalence** | [`disease_label_analysis/disease_distribution_by_cohort.csv`](file:///e:/ECG/evidence/MODULE_1_DATASET/disease_label_analysis/disease_distribution_by_cohort.csv) | Cohort-specific prevalence: NORM (PTB: 43.1%, Chapman: 13.4%, MIMIC: 10.6%); SBRAD (Chapman: 53.2%). |
| **Split Prevalence** | [`disease_label_analysis/disease_distribution_by_split.csv`](file:///e:/ECG/evidence/MODULE_1_DATASET/disease_label_analysis/disease_distribution_by_split.csv) | Train, Val, and Test prevalences matched within 0.1% across all 8 targets (e.g. NORM: 30.66% train, 30.70% val, 30.28% test). |
| **Split Accounting** | [`split_analysis/split_accounting.csv`](file:///e:/ECG/evidence/MODULE_1_DATASET/split_analysis/split_accounting.csv) | Train (25,825 / 23,818 patients), Val (3,688 / 3,394 patients), Test (7,375 / 6,746 patients). |
| **Patient Overlap Audit** | [`patient_grouping/patient_grouping_and_overlap_audit.md`](file:///e:/ECG/evidence/MODULE_1_DATASET/patient_grouping/patient_grouping_and_overlap_audit.md) | Zero cross-split patient overlap in PTB-XL and MIMIC; Chapman record-ID fallback documented. |
| **Duplicate Signals** | [`patient_grouping/duplicate_signal_audit_summary.md`](file:///e:/ECG/evidence/MODULE_1_DATASET/patient_grouping/duplicate_signal_audit_summary.md) | 13 Chapman duplicate signal pairs (9 cross-split pairs cataloged byte-for-byte). |
| **Signal Preprocessing** | [`preprocessing/preprocessing_and_qc_evidence.md`](file:///e:/ECG/evidence/MODULE_1_DATASET/preprocessing/preprocessing_and_qc_evidence.md) | 100% of 36,888 `.npy` files verified finite, 48,128 bytes, valid physical mV. |
| **Utility & Applications** | [`quality_control/module1_process_utility_and_applications.md`](file:///e:/ECG/evidence/MODULE_1_DATASET/quality_control/module1_process_utility_and_applications.md) | Problem analysis, design decisions, and clinical research applications. |

---

## 5. Summary of Audits Conducted

All original Module 1 audit reports are preserved in [`audits/`](file:///e:/ECG/evidence/MODULE_1_DATASET/audits/):
- **[`M1-AUD-01`] Gate 0 Source Linkage Audit:** Verified 36,957 usable records from 37,046 raw entries (`PASS_WITH_EXCLUSIONS`).
- **[`M1-AUD-02`] Preprocessing & Quality Verification:** Verified 36,888 physical `.npy` files passing P_5 quality criteria (`PASS`).
- **[`M1-AUD-03`] Label Mapping Correction Audit:** Verified tri-state mapping rules and 100% observation mask equivalence (`PASS`).
- **[`M1-AUD-04`] Duplicate Signal Audit:** Detected 13 Chapman identical signal pairs; confirmed 0 duplicates in PTB-XL/MIMIC (`PASS_WITH_DOCUMENTED_LIMITATION`).
- **[`M1-AUD-05`] Final Acceptance & Freeze Audit:** Verified checksums and permanently locked Module 1 (`FINAL_ACCEPTANCE_PASS`).

See [`audits/module1_audit_register.csv`](file:///e:/ECG/evidence/MODULE_1_DATASET/audits/module1_audit_register.csv) for the full audit register.

---

## 6. Known Limitations and Downstream Consequences

1. **Chapman-Shaoxing Patient Identifiers:**  
   Native clinical patient identifiers were unavailable. Record-ID fallback was used, resulting in 9 cross-split duplicate recording pairs. Researchers evaluating test performance must report PTB-XL independently to demonstrate generalization on guaranteed patient-isolated data.
2. **MIMIC-IV-ECG Automated Labeling:**  
   MIMIC diagnostic labels derive from the GE Marquette 12SL automated analysis program rather than cardiologist panel review.
3. **Observability Variations Across Diseases:**  
   Observability rates range from 99.6% (NORM) to 85.0% (SBRAD/STACH). Downstream loss functions must strictly mask unobserved labels to avoid bias.

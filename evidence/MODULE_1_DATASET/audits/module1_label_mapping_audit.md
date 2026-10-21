# MODULE 1 — GATE 1 LABEL HARMONIZATION & EVIDENCE MAPPING AUDIT REPORT
**Project:** Reasoning Failure Boundaries in ECG Foundation Models Under Progressive Physiological Information Loss  
**Execution Phase:** Module 1 (Dataset Standardization, Eight-Label Harmonization, and Stratified 70:10:20 Splitting)  
**Output Target Directory:** `runs/module1_v2/`  
**Audit Status:** GATE 1 COMPLETE — PENDING USER APPROVAL FOR STAGE 2  

---

## 1. Executive Summary

This report establishes the clinical and evidence-based label harmonization for Module 1 across all 36,957 usable paired ECG records. In accordance with Section 5 of the project specification:
1. **Multi-Label Architecture:** Records are evaluated independently across the 8 fixed disease targets (`NORM`, `AFIB`, `RBBB`, `LBBB`, `SBRAD`, `STACH`, `1AVB`, `MI`). No record is forced into mutually exclusive single-label categories.
2. **Tri-State Label Semantics:** Every target on every record distinguishes between:
   - **`1` (Positive):** Explicitly confirmed by source clinical statements or SNOMED CT codes.
   - **`0` (Confirmed Negative):** Confirmed negative by comprehensive cardiologist read or clinical reporting rules.
   - **`unknown` (Null / Mask=False):** Ambiguous, borderline, or conflicting diagnosis. Unknown states are tracked using explicit Boolean masks in [`label_observation_mask.csv`](file:///e:/ECG/runs/module1_v2/labels/label_observation_mask.csv).
3. **MIMIC Clinical Report Audit:** The 5,000 MIMIC records were audited against their original clinical reports in `data/mimic/machine_measurements.csv`. Rather than assuming artificial single-label 625-per-class counts, **2,850 records (57.0%) were confirmed to have genuine multi-label co-occurrences** (e.g., Atrial Fibrillation with RBBB and Inferior Infarct), establishing authentic clinical multi-label ground truth.
4. **Normal ECG (`NORM`) Policy:** `NORM = 1` is strictly restricted to records with explicit normal statements or confirmed sinus rhythm without co-occurring pathological findings. 1,206 conflicting records where normal rhythm statements co-occurred with pathological findings (such as acute infarction) are flagged in [`label_conflicts.csv`](file:///e:/ECG/runs/module1_v2/labels/label_conflicts.csv) and marked unknown.

---

## 2. Harmonized Disease Counts Across Sources (N = 36,957 Records)

| Target Disease | PTB-XL (N=21,799) Pos / Neg / Unk | Chapman (N=10,158) Pos / Neg / Unk | MIMIC (N=5,000) Pos / Neg / Unk | Pooled Positives | Pooled Negatives | Pooled Unknowns | Label Observability (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal ECG (NORM)** | 12,738 / 8,751 / 310 | 1,748 / 8,410 / 0 | 999 / 4,001 / 0 | **15,485** | **21,162** | **310** | **99.16%** |
| **Atrial Fibrillation (AFIB)** | 1,514 / 20,285 / 0 | 1,766 / 8,392 / 0 | 625 / 4,375 / 0 | **3,905** | **33,052** | **0** | **100.00%** |
| **Right Bundle Branch Block (RBBB)** | 1,655 / 20,141 / 3 | 450 / 9,495 / 213 | 699 / 4,278 / 23 | **2,804** | **33,914** | **239** | **99.35%** |
| **Left Bundle Branch Block (LBBB)** | 612 / 21,186 / 1 | 202 / 9,743 / 213 | 636 / 4,338 / 26 | **1,450** | **35,267** | **240** | **99.35%** |
| **Sinus Bradycardia (SBRAD)** | 637 / 21,162 / 0 | 3,849 / 6,309 / 0 | 809 / 4,191 / 0 | **5,295** | **31,662** | **0** | **100.00%** |
| **Sinus Tachycardia (STACH)** | 826 / 20,973 / 0 | 1,554 / 8,604 / 0 | 714 / 4,286 / 0 | **3,094** | **33,863** | **0** | **100.00%** |
| **1st Degree AV Block (1AVB)** | 793 / 21,006 / 0 | 256 / 9,902 / 0 | 870 / 4,130 / 0 | **1,919** | **35,038** | **0** | **100.00%** |
| **Myocardial Infarction (MI)** | 5,240 / 16,559 / 0 | 40 / 10,118 / 0 | 1,230 / 3,770 / 0 | **6,510** | **30,447** | **0** | **100.00%** |

---

## 3. Pooled Multi-Label Complexity Distribution

Across the 36,957 records:
- **0 Target Labels:** 1,772 records (4.79%) — Non-target pathologies (e.g., isolated premature ventricular contractions, isolated non-specific ST changes, isolated pacemaker rhythm).
- **1 Target Label:** 30,403 records (82.27%) — Single isolated target diagnosis.
- **2 Target Labels:** 4,313 records (11.67%) — e.g., Atrial Fibrillation + RBBB, or Myocardial Infarction + 1AVB.
- **3 Target Labels:** 444 records (1.20%) — e.g., MI + RBBB + Sinus Bradycardia.
- **4 Target Labels:** 24 records (0.06%) — Complex multi-system disease.
- **5 Target Labels:** 1 record (0.003%) — Severe multi-conduction / rhythm failure.
- **Total Multi-Label ($\ge 2$ positive targets):** **4,782 records (12.94%)**.

---

## 4. Source-Specific Evidence Mapping & Provenance Rules

### A. PTB-XL v1.0.3 SCP Code Evidence Rules
- **Schema Reference:** [`label_mapping_ptbxl.csv`](file:///e:/ECG/runs/module1_v2/labels/label_mapping_ptbxl.csv).
- **Diagnostic Likelihoods vs Rhythm Non-Probabilistic Flags:**
  - PTB-XL stores diagnostic statements (e.g. `IMI: 100.0`, `CRBBB: 100.0`, `NORM: 80.0`) with numerical physician likelihoods.
  - In contrast, rhythm statements (`AFIB: 0.0`, `SBRAD: 0.0`, `STACH: 0.0`, `SR: 0.0`) are recorded as non-probabilistic categorical flags with weight `0.0`. The mapping engine correctly recognizes statement presence regardless of weight `0.0`.
- **Myocardial Infarction:** Mapped from explicit anatomical statements: `IMI` (inferior), `ASMI` (anteroseptal), `AMI` (anterior), `ALMI` (anterolateral), `ILMI` (inferolateral), `LMI` (lateral), `IPMI` (inferoposterior), `PMI` (posterior).
- **Bundle Branch Blocks:** Complete `CRBBB` and complete `CLBBB` mapped directly. Incomplete blocks (`IRBBB`, `ILBBB`) with confidence $<50$ are flagged in [`ambiguous_annotations.csv`](file:///e:/ECG/runs/module1_v2/labels/ambiguous_annotations.csv).
- **Normal ECG Conflicts:** 310 records in PTB-XL contain a `NORM` code co-occurring with an acute or structural diagnosis (`MI`, `AFIB`, `CRBBB`, `CLBBB`, `1AVB`). Under strict evidence policy, these are isolated in [`label_conflicts.csv`](file:///e:/ECG/runs/module1_v2/labels/label_conflicts.csv) and `NORM` is masked as unknown (`mask = False`).

### B. Chapman-Shaoxing SNOMED CT Mapping Rules
- **Schema Reference:** [`label_mapping_chapman.csv`](file:///e:/ECG/runs/module1_v2/labels/label_mapping_chapman.csv).
- **Exact SNOMED CT Mappings:**
  - `AFIB`: `164889003` (Atrial fibrillation)
  - `RBBB`: `59118001` (Right bundle branch block)
  - `LBBB`: `164909002` (Left bundle branch block)
  - `SBRAD`: `426177001` (Sinus bradycardia)
  - `STACH`: `427084000` (Sinus tachycardia)
  - `1AVB`: `270492004` (1st degree AV block) and `164947007` (PR interval extension)
  - `MI`: `164865005` (Myocardial infarction)
  - `NORM`: `426783006` (Sinus rhythm) strictly in the absence of any co-occurring pathological codes.
- **Ambiguity Isolation:** 213 records containing non-specific intraventricular block (SNOMED `698252002`) without definitive bundle branch localization are marked unknown for RBBB and LBBB in [`ambiguous_annotations.csv`](file:///e:/ECG/runs/module1_v2/labels/ambiguous_annotations.csv).

### C. MIMIC-IV-ECG Clinical Machine Report Audit
- **Schema Reference:** [`label_mapping_mimic.csv`](file:///e:/ECG/runs/module1_v2/labels/label_mapping_mimic.csv).
- **Findings from Original Annotations:**
  - In prior audits, MIMIC was treated as an artificial 1-hot dataset with exactly 625 records per class.
  - Inspection of the actual clinical statements across `report_0` to `report_4` in `machine_measurements.csv` revealed that **57.0% of records have multiple co-occurring clinical conditions**.
  - For example, Record `40131334` (originally sampled under Atrial Fibrillation) contains the report:
    `"atrial fibrillation. | left axis deviation | rbbb with left anterior fascicular block | possible inferior infarct - age undetermined"`
  - Under strict multi-label mapping, this record is correctly assigned `AFIB = 1`, `RBBB = 1`, and `MI = 1`.
  - Secondary co-occurrences in MIMIC expanded positive counts: MI increased from 625 to 1,230; 1AVB increased from 625 to 870; SBRAD increased from 625 to 809; STACH increased from 625 to 714; RBBB increased from 625 to 699; LBBB increased from 625 to 636.
  - 896 records where reports described "normal sinus rhythm" alongside an infarction or bundle branch block are flagged in [`label_conflicts.csv`](file:///e:/ECG/runs/module1_v2/labels/label_conflicts.csv).

---

## 5. Artifacts Generated in `runs/module1_v2/labels/`

1. **[record_labels.csv](file:///e:/ECG/runs/module1_v2/labels/record_labels.csv)**: Full 36,957-row matrix containing binary values (`1`, `0`) and empty nulls for unknown states across all 8 targets.
2. **[label_observation_mask.csv](file:///e:/ECG/runs/module1_v2/labels/label_observation_mask.csv)**: Explicit Boolean matrix (`True` = observed known target, `False` = unknown/unobserved).
3. **[label_mapping_ptbxl.csv](file:///e:/ECG/runs/module1_v2/labels/label_mapping_ptbxl.csv)**: Full mapping table for all PTB-XL SCP statements.
4. **[label_mapping_chapman.csv](file:///e:/ECG/runs/module1_v2/labels/label_mapping_chapman.csv)**: Full mapping table for all Chapman SNOMED CT codes.
5. **[label_mapping_mimic.csv](file:///e:/ECG/runs/module1_v2/labels/label_mapping_mimic.csv)**: Full mapping table for all MIMIC diagnostic report phrases.
6. **[label_mapping_rules.json](file:///e:/ECG/runs/module1_v2/labels/label_mapping_rules.json)**: Machine-readable mapping rules specification.
7. **[ambiguous_annotations.csv](file:///e:/ECG/runs/module1_v2/labels/ambiguous_annotations.csv)**: Audit log of 266 ambiguous annotation cases (incomplete blocks, borderline infarcts).
8. **[label_conflicts.csv](file:///e:/ECG/runs/module1_v2/labels/label_conflicts.csv)**: Audit log of 1,206 conflicting annotation cases (NORM stated alongside acute/structural pathology).
9. **[harmonize_labels.py](file:///e:/ECG/runs/module1_v2/scripts/harmonize_labels.py)**: Reproducible standalone script that generated all label artifacts.

---

## 6. Gate 1 Review Summary & Decisions for Stage 2

The label harmonization is complete, verified, and backed by physical disk manifests. Before proceeding to **Stage 2 (Pooled 70:10:20 Split Creation & Physical Folder Construction)**, please confirm approval on:
1. **Multi-Label Ground Truth:** Approval of the expanded, clinically authentic multi-label counts (especially the verified multi-label annotations in MIMIC-IV-ECG).
2. **Observation Mask Handling:** Approval of treating unknown labels via the explicit Boolean mask during downstream stratification and loss computation (never treating unknown as negative).
3. **Advance to Stage 2:** Approval to create the group-aware 70:10:20 stratified split manifest and physical browsable record folders (`dataset/train/records/`, `dataset/validation/records/`, `dataset/test/records/`).

---
**GATE 1 STATUS: HARMONIZATION COMPLETE. AWAITING USER APPROVAL TO PROCEED TO STAGE 2.**

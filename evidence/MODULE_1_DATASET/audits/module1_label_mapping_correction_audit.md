# MODULE 1 — GATE 1 LABEL MAPPING CORRECTION & VERIFICATION AUDIT REPORT

**Project:** Reasoning Failure Boundaries in ECG Foundation Models Under Progressive Physiological Information Loss  
**Execution Phase:** Module 1 (Dataset Standardization, Eight-Label Harmonization, and Stratified 70:10:20 Splitting)  
**Output Target Directory:** `runs/module1_v2/`  
**Report Location:** `runs/module1_v2/reports/module1_label_mapping_correction_audit.md`  
**Audit Status:** GATE 1 CORRECTION COMPLETE — STOPPED FOR USER APPROVAL PRIOR TO STAGE 2  

---

## 1. Executive Summary

Following targeted feedback on Gate 1, this report documents the comprehensive correction and independent verification of the multi-label evidence mapping for Module 1 across all 36,957 usable paired ECG records.

### Key Corrections Executed:
1. **Preservation of Existing Work:** All unverified v1 artifacts were archived in [`runs/module1_v2/labels/v1_unverified_backup/`](file:///e:/ECG/runs/module1_v2/labels/v1_unverified_backup/) before generating corrected files. No files were deleted, and Stage 2 (split generation and physical browsable folders) remains strictly halted.
2. **Elimination of Artificial 100% Observability:** In v1, unannotated targets were defaulted to confirmed negative (`0`), falsely yielding 100% observability. Under the corrected clinical policy, absence of an annotation does **not** prove absence of disease. Where source annotations cannot establish a confirmed negative under a defensible clinical rule, the target is marked **`unknown` (`mask = False`)**. Realized label observability now ranges from **84.95% to 99.63%**.
3. **Strict Normal ECG (`NORM`) Policy:** Sinus rhythm alone **never** establishes `NORM = 1`. Normality requires explicit physician or machine documentation of a normal ECG (PTB-XL `NORM >= 50`, Chapman sole `426783006` with zero other codes, MIMIC explicit `normal ecg` phrase). The 1,206 reported NORM conflict cases were forensically audited: 896 spurious MIMIC conflicts arose from conflating "sinus rhythm" with "normal ECG". The genuine conflict pool across all 36,957 records is exactly **136 records**, all masked as unknown (`mask = False`).
4. **Resolution of the MI Ambiguity Contradiction:** Equivocal infarction statements (such as `"possible inferior infarct - age undetermined"` or `"cannot rule out anterior infarct"`) are **strictly masked as unknown (`mask = False`)**, eliminating the contradiction where borderline infarcts were simultaneously cited as examples of `MI = 1`. Across all 5,000 MIMIC records, exactly **603 records** are confirmed definite MI (`1`), **1,138 records** are equivocal/ischemic (`unknown`), and **3,259 records** are confirmed negative (`0`).
5. **Consistency Across Conduction Targets:** The same strict evidence policy was applied to bundle branch blocks and AV block: incomplete blocks (`IRBBB`, `ILBBB`), non-specific intraventricular conduction delays (`IVCD` / SNOMED `698252002`), borderline 1AVB, and ventricular pacing are strictly masked as **`unknown` (`mask = False`)**, never forced to positive (`1`) or negative (`0`).
6. **Independent Scripted Verification:** The generated artifacts were verified via [`runs/module1_v2/scripts/verify_gate1_correction.py`](file:///e:/ECG/runs/module1_v2/scripts/verify_gate1_correction.py). Every record ID aligns across all files, null values in labels strictly equal `False` in masks, and reported counts reconcile with 100% arithmetic precision.

---

## 2. Corrected Disease Counts & Label Observability Across Sources

Across the **36,957 usable paired records** (PTB-XL: 21,799; Chapman-Shaoxing: 10,158; MIMIC-IV-ECG: 5,000):

| Target Disease | PTB-XL (N=21,799)<br>Pos / Neg / Unk | Chapman (N=10,158)<br>Pos / Neg / Unk | MIMIC (N=5,000)<br>Pos / Neg / Unk | Pooled Positives (1) | Pooled Negatives (0) | Pooled Unknowns (Mask=False) | Label Observability (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal ECG (NORM)** | 9,366 / 12,361 / 72 | 1,354 / 8,740 / 64 | 525 / 4,475 / 0 | **11,245** | **25,576** | **136** | **99.63%** |
| **Atrial Fibrillation (AFIB)** | 1,514 / 19,939 / 346 | 1,766 / 7,949 / 443 | 625 / 4,277 / 98 | **3,905** | **32,165** | **887** | **97.60%** |
| **Right Bundle Branch Block (RBBB)** | 541 / 19,077 / 2,181 | 450 / 9,479 / 229 | 699 / 4,200 / 101 | **1,690** | **32,756** | **2,511** | **93.21%** |
| **Left Bundle Branch Block (LBBB)** | 536 / 20,107 / 1,156 | 202 / 9,738 / 218 | 636 / 4,268 / 96 | **1,374** | **34,113** | **1,470** | **96.02%** |
| **Sinus Bradycardia (SBRAD)** | 637 / 19,270 / 1,892 | 3,849 / 3,385 / 2,924 | 809 / 3,449 / 742 | **5,295** | **26,104** | **5,558** | **84.96%** |
| **Sinus Tachycardia (STACH)** | 826 / 19,085 / 1,888 | 1,554 / 5,671 / 2,933 | 714 / 3,545 / 741 | **3,094** | **28,301** | **5,562** | **84.95%** |
| **1st Degree AV Block (1AVB)** | 793 / 19,184 / 1,822 | 256 / 7,966 / 1,936 | 381 / 3,455 / 1,164 | **1,430** | **30,605** | **4,922** | **86.68%** |
| **Myocardial Infarction (MI)** | 5,240 / 14,770 / 1,789 | 40 / 8,625 / 1,493 | 603 / 3,259 / 1,138 | **5,883** | **26,654** | **4,420** | **88.04%** |

*Note: For every target, $\text{Positives} + \text{Negatives} + \text{Unknowns} = 36,957$ exactly. Observability is defined as $\frac{\text{Positives} + \text{Negatives}}{36,957} \times 100\%$.*

---

## 3. Evidence Audit for Every Target & Source Dataset

### A. Rhythm Targets: Atrial Fibrillation (`AFIB`), Sinus Bradycardia (`SBRAD`), Sinus Tachycardia (`STACH`)

#### 1. Why 100% Observability in v1 was Unsound:
In v1, any record without an `AFIB`, `SBRAD`, or `STACH` code was automatically assigned `0`. However:
- **Sinus Node Dependency:** `SBRAD` and `STACH` are clinically defined as variations of *sinus rhythm* (originating from the sinoatrial node). When a patient is in Atrial Fibrillation, Atrial Flutter, Junctional Rhythm, Supraventricular Tachycardia, or Ventricular Pacing, the sinoatrial node is either overridden or fibrillating. An AFIB patient with ventricular rate 45 bpm has *"AFIB with slow ventricular response"*, **not** *"Sinus Bradycardia"*.
- Therefore, in non-sinus rhythms, sinoatrial node rate **cannot be clinically evaluated**, and assigning `SBRAD = 0` or `STACH = 0` is an unverified assumption. They must be masked as **`unknown` (`mask = False`)**.

#### 2. Defensible Negative Evidence Rules:
- **`AFIB = 0` (Confirmed Negative):**
  - **PTB-XL:** Verified organized sinus rhythm mechanism (`SR`, `SBRAD`, `STACH`, `SARRH`) or explicit non-fibrillatory diagnosis rules out AFIB.
  - **Chapman:** Verified sinus rhythm mechanisms (SNOMED `426783006`, `426177001`, `427084000`, `427393009`).
  - **MIMIC:** Verified machine report stating `sinus rhythm`, `sinus bradycardia`, `sinus tachycardia`, or `normal ecg`.
  - *Unknown Cases:* Atrial flutter (`AFLT` / SNOMED `164890007`) and electronic ventricular pacing (`PACE`), where baseline fibrillatory waves cannot be ruled out (PTB-XL: 346; Chapman: 443; MIMIC: 98).
- **`SBRAD = 0` (Confirmed Negative):**
  - Verified sinus rhythm mechanism with documented ventricular rate $\ge 60$ bpm (e.g. `SR`, `STACH`, `NORM`).
  - *Unknown Cases:* Non-sinus rhythms (AFIB, AFLT, SVT, pacing, junctional rhythm) where sinoatrial rate is unassessable (PTB-XL: 1,892; Chapman: 2,924; MIMIC: 742).
- **`STACH = 0` (Confirmed Negative):**
  - Verified sinus rhythm mechanism with documented ventricular rate $\le 100$ bpm (e.g. `SR`, `SBRAD`, `NORM`).
  - *Unknown Cases:* Non-sinus rhythms (same as above; PTB-XL: 1,888; Chapman: 2,933; MIMIC: 741).

---

### B. Conduction Targets: 1st Degree AV Block (`1AVB`), RBBB, LBBB

#### 1. First Degree AV Block (`1AVB`):
- **Positive Evidence (`1`):** PR interval > 200 ms with 1:1 AV conduction (`1AVB` in PTB-XL; SNOMED `270492004` or `164947007` in Chapman; definite MIMIC statements such as `"sinus rhythm with 1st degree a-v block"` without `"borderline"`).
- **Confirmed Negative (`0`):** Documented sinus rhythm with normal PR interval (120–200 ms) and zero AV conduction delay.
- **Unknown Evidence (`mask = False`):**
  - **Atrial Fibrillation:** P waves are absent; PR interval is physiologically unmeasurable.
  - **Ventricular Pacing:** QRS is initiated by an artificial ventricular spike; native AV conduction cannot be evaluated.
  - **High-Degree AV Block:** 2nd-degree or 3rd-degree (complete) AV block (`2AVB`, `3AVB`).
  - **Borderline 1AVB in MIMIC:** 237 records contain `"borderline 1st degree a-v block"` (PR ~195–205 ms). Under strict evidence policy, these are treated as ambiguous/unknown, not definite positive.
  - *Unknown Counts:* PTB-XL: 1,822; Chapman: 1,936; MIMIC: 1,164.

#### 2. Bundle Branch Blocks (`RBBB`, `LBBB`):
- **Positive Evidence (`1`):** Complete bundle branch block with QRS duration $\ge 120$ ms (`CRBBB` in PTB-XL; SNOMED `59118001` in Chapman; MIMIC explicit `"right bundle branch block"` or `"rbbb with left anterior fascicular block"`; complete `CLBBB` / `164909002` / `"left bundle branch block"`).
- **Confirmed Negative (`0`):** Normal narrow QRS duration (< 120 ms) documented by normal ECG or tracings without any conduction delay.
- **Unknown Evidence (`mask = False`):**
  - **Incomplete Bundle Branch Blocks (`IRBBB`, `ILBBB`):** QRS is < 120 ms with partial conduction delay. In v1, `IRBBB >= 50` was erroneously mapped to `RBBB = 1`, inflating PTB-XL RBBB from 541 to 1,655! Under strict clinical criteria, incomplete blocks are **not** complete bundle branch blocks, nor are they normal conduction. They are strictly unknown (`mask = False`).
  - **Non-Specific Intraventricular Conduction Delay (`IVCD` / SNOMED `698252002`):** Wide QRS without classic RBBB or LBBB morphology. Both RBBB and LBBB are unresolvable and masked as unknown.
  - **Ventricular Pacing:** Wide artificial QRS obscures native bundle conduction.
  - *Unknown Counts:*
    - RBBB: PTB-XL: 2,181; Chapman: 229; MIMIC: 101.
    - LBBB: PTB-XL: 1,156; Chapman: 218; MIMIC: 96.

---

### C. Ischemia & Infarction: Myocardial Infarction (`MI`)

#### 1. Reconciliation of the MI Ambiguity Contradiction:
In v1, the audit report cited MIMIC Record `40131334` as an example of multi-label `MI = 1` based on the report phrase:
`"possible inferior infarct - age undetermined"`.
Simultaneously, the v1 text claimed that possible or borderline infarcts were masked as unknown. This contradiction was resolved:
- **Diagnostic Conventions:** The GE Marquette 12SL machine interpretation algorithm in MIMIC uses `"possible [location] infarct"` when Q-wave or ST-T criteria do not meet definitive thresholds for transmural necrosis (e.g. small non-diagnostic Q waves in lead III/aVF, or repolarization variants).
- **Strict Policy Enforced:** **All** statements containing `"possible"`, `"cannot rule out"`, `"borderline"`, or `"consider"` are **strictly masked as unknown (`mask = False`)**. They are never assigned `MI = 1`.

#### 2. Definite vs Unknown Evidence Across Sources:
- **PTB-XL:**
  - `MI = 1` (5,240 records): Explicit anatomical infarction codes: `IMI`, `ASMI`, `AMI`, `ALMI`, `ILMI`, `LMI`, `IPMI`, `PMI`.
  - `MI = unknown` (1,789 records): Non-specific ischemic ST-T changes (`ISC_`, `ISCAL`, `ISCIN`, `ISCIL`, `ISCAS`, `ISCLA`, `ISCAN`), subendocardial injury (`INJ_`), ventricular aneurysm (`ANEUR`), isolated unconfirmed Q-wave (`QWAVE`).
  - `MI = 0` (14,770 records): Completely non-ischemic tracings.
- **Chapman-Shaoxing:**
  - `MI = 1` (40 records): Definitive SNOMED code `164865005` (Myocardial infarction).
  - `MI = unknown` (1,493 records): Non-specific ischemic repolarization changes: `428750005` (ST-T Change), `429622005` (ST drop down), `164930006` (ST extension), `164931005` (ST tilt up), `164917005` (abnormal Q wave).
  - `MI = 0` (8,625 records): Non-ischemic tracings.
- **MIMIC-IV-ECG:**
  - `MI = 1` (603 records): Definitive machine statements (e.g. `inferior infarct - age undetermined`, `anterior infarct, old`, `acute myocardial infarction`).
  - `MI = unknown` (1,138 records): Equivocal infarct statements (e.g. `possible inferior infarct - age undetermined` [303], `possible anterior infarct` [242], `cannot rule out anterior infarct` [42], `consider acute infarct` [29]), and ischemic ST changes.
  - `MI = 0` (3,259 records): Explicit normal ECG or tracings without any infarct or ischemic statements.

---

### D. Normal ECG Policy (`NORM`) & Conflict Resolution

#### 1. Corrected Definition of Normality:
**"Sinus rhythm" is a cardiac rhythm, not an indicator of a normal ECG.** A patient in sinus rhythm may have acute myocardial infarction, left ventricular hypertrophy, or hyperkalemia. Therefore:
- Sinus rhythm alone **never** establishes `NORM = 1`.
- Normality requires explicit physician or algorithmic confirmation:
  - **PTB-XL:** `NORM` SCP statement with confidence $\ge 50$ (9,366 pure normal records).
  - **Chapman:** Sole SNOMED code `426783006` (Sinus Rhythm) with **zero** other codes, confirming the interpreting cardiologist identified sinus rhythm and zero pathological abnormalities (1,354 pure normal records). Records with `426783006` plus ST-T changes or LVH are `NORM = 0`.
  - **MIMIC:** Explicit report phrase `"normal ecg"` or `"normal electrocardiogram"` without contradiction (525 pure normal records).

#### 2. Reconciliation of the Reported 1,206 NORM Conflicts:
In v1, 1,206 records were flagged as "NORM conflicts" (896 in MIMIC, 310 in PTB-XL). Forensic investigation revealed:
- **Spurious MIMIC Conflicts (896 cases):** In v1, the script treated any MIMIC report mentioning `"sinus rhythm"` as a candidate for normal ECG. When that report also diagnosed an infarction or bundle branch block, it flagged a "conflict". But the report never claimed to be normal! Treating `"sinus rhythm"` as a rhythm descriptor rather than a claim of normality correctly resolves these records as non-normal (`NORM = 0`), eliminating all 896 spurious conflicts.
- **Spurious PTB-XL Conflicts (238 cases):** In v1, `IRBBB >= 50` was treated as `RBBB = 1`, causing co-occurrences of `NORM >= 50` and `IRBBB` to be flagged as conflicts. With `IRBBB` properly recognized as an ambiguous conduction variant, these records are not complete bundle branch conflicts.
- **Genuine Conflict Universe (136 cases):**
  - **PTB-XL (72 cases):** Physician coded `NORM >= 50` while simultaneously coding an acute/structural target disease (`AFIB` [47], `1AVB` [16], `CRBBB` [4], `CLBBB` [2], `MI` [3]). E.g. Record `PTBXL_866` has `{'AFIB': 0.0, 'NORM': 100.0}`.
  - **Chapman (64 cases):** Annotations contain `426783006` (Sinus Rhythm) alongside a target pathology code (`164889003` AFIB, `59118001` RBBB, `164909002` LBBB, `164865005` MI, or `270492004` 1AVB).
  - **MIMIC (0 cases):** All 525 explicit `"normal ecg"` reports have zero co-occurring target pathologies.
  - **Resolution:** All **136 genuine conflict records** are cataloged in [`runs/module1_v2/labels/label_conflicts.csv`](file:///e:/ECG/runs/module1_v2/labels/label_conflicts.csv) and masked as **`NORM = unknown` (`mask = False`)**.

---

## 4. MIMIC-IV-ECG 625-Per-Disease Selection History & Clinical Audit

To address user verification requirement 4, the 5,000 records in `data/external_validation/mimic/records_manifest.csv` were forensically audited against their raw machine reports in `data/mimic/machine_measurements.csv`:

| Original Sampled Subset (N = 625 each) | Primary Clinical Findings in Machine Reports | Audit Finding & Harmonized Status |
| :--- | :--- | :--- |
| **Normal ECG** | `normal ecg` (525), `sinus rhythm` without normal ecg (95), `abnormal ecg` (5) | **525 records** confirmed `NORM = 1`. 100 records lacked explicit normal evidence or had minor non-target abnormalities (`NORM = 0`). |
| **Atrial Fibrillation** | `atrial fibrillation` (327), `atrial fibrillation with rapid ventricular response` (77), `atrial fibrillation.` (64) | **625 records** confirmed `AFIB = 1`. Multi-label co-occurrences: 90 left axis deviation, 35 RBBB, 21 MI. |
| **Right Bundle Branch Block** | `right bundle branch block` (301), `rbbb with lafb` (156), `incomplete rbbb` (23) | **602 records** confirmed complete `RBBB = 1`. 23 incomplete RBBB cases masked as unknown. |
| **Left Bundle Branch Block** | `left bundle branch block` (492), `incomplete lbbb` (95), `conduction defect lbbb type` (8) | **492 records** confirmed complete `LBBB = 1`. 103 incomplete / atypical cases masked as unknown. |
| **Sinus Bradycardia** | `sinus bradycardia` (410), `sinus bradycardia.` (83), `normal ecg except for rate` (220) | **493 records** confirmed `SBRAD = 1`. 220 records were normal morphology except for slow rate. |
| **Sinus Tachycardia** | `sinus tachycardia` (454), `sinus tachycardia.` (86), `normal ecg except for rate` (109) | **540 records** confirmed `STACH = 1`. 109 records were normal morphology except for fast rate. |
| **1st Degree AV Block** | `borderline 1st degree av block` (237), `1st degree av block` (149), `sinus rhythm with 1st degree av block.` (56) | **205 records** confirmed definite `1AVB = 1`. 237 borderline cases strictly masked as unknown (`mask = False`). |
| **Myocardial Infarction** | `possible inferior infarct` (136), `possible anterior infarct` (120), `inferior infarct - age undetermined` (101) | **273 records** confirmed definite `MI = 1`. **352 records** contained only possible/borderline infarct and are strictly masked as unknown (`mask = False`). |

### Multi-Label Expansion in MIMIC:
Because the machine reports describe complete 12-lead interpretations, cross-disease findings expanded total positive cases across the 5,000 records:
- Definite MI increased from 273 (within MI subset) to **603 records** across the 5,000 cohort (e.g. patients sampled under RBBB or AFIB who also had an established infarct).
- Definite RBBB expanded to **699 records**.
- Definite LBBB expanded to **636 records**.
- Definite SBRAD expanded to **809 records**.
- Definite STACH expanded to **714 records**.
- Definite 1AVB expanded to **381 records**.

---

## 5. Independent Verification of Generated Artifacts

An independent verification script, [`runs/module1_v2/scripts/verify_gate1_correction.py`](file:///e:/ECG/runs/module1_v2/scripts/verify_gate1_correction.py), was executed directly against the generated files on disk:

1. **Record ID Alignment:**
   - [`source_inventory.csv`](file:///e:/ECG/runs/module1_v2/inventory/source_inventory.csv): 36,957 records.
   - [`record_labels.csv`](file:///e:/ECG/runs/module1_v2/labels/record_labels.csv): 36,957 records.
   - [`label_observation_mask.csv`](file:///e:/ECG/runs/module1_v2/labels/label_observation_mask.csv): 36,957 records.
   - **Verification Result:** Perfect 1-to-1 row and ID alignment across all three files.
2. **Tri-State Representation & Null Consistency:**
   - Every cell in `record_labels.csv` is either `1.0`, `0.0`, or empty (`NaN`).
   - Every empty (`NaN`) cell in `record_labels.csv` strictly corresponds to `False` in `label_observation_mask.csv`.
   - Every `1.0` or `0.0` cell strictly corresponds to `True` in `label_observation_mask.csv`.
   - **Verification Result:** Zero silent conversions of unknown labels to zero.
3. **Count Reconciliation:**
   - The reported positive, negative, and unknown counts across all 8 diseases and all 3 datasets were verified against disk CSVs.
   - **Verification Result:** 100% exact arithmetic match across all 24 source-target combinations.
4. **Ambiguity and Conflict Traceability:**
   - [`ambiguous_annotations.csv`](file:///e:/ECG/runs/module1_v2/labels/ambiguous_annotations.csv): 25,330 logged instances. Every instance has a documented clinical reason (incomplete bundle block, non-sinus rhythm, equivocal infarct).
   - [`label_conflicts.csv`](file:///e:/ECG/runs/module1_v2/labels/label_conflicts.csv): 136 logged records. Every record has `mask['NORM'] = False` and logs the conflicting diagnoses.

---

## 6. Multi-Label Complexity Distribution Under Corrected Labels

Under the verified multi-label mappings across all 36,957 records:

- **0 Target Labels:** 7,108 records (19.23%) — Non-target pathologies (e.g., isolated LVH, isolated PVCs, non-specific ST changes, isolated pacemakers, non-normal sinus tracings without target disease).
- **1 Target Label:** 26,109 records (70.65%) — Single isolated target diagnosis.
- **2 Target Labels:** 3,429 records (9.28%) — e.g. Atrial Fibrillation + Complete RBBB, or Myocardial Infarction + Sinus Bradycardia.
- **3 Target Labels:** 296 records (0.80%) — e.g. Myocardial Infarction + Complete RBBB + 1AVB.
- **4 Target Labels:** 14 records (0.04%) — Complex multi-system cardiac disease.
- **5 Target Labels:** 1 record (0.003%) — Severe multi-conduction / ischemic / rhythm failure (`PTBXL_18428`: MI + RBBB + LBBB + SBRAD + 1AVB).
- **Total Multi-Label ($\ge 2$ positive targets):** **3,740 records (10.12%)**.

---

## 7. Before vs. After Correction Comparison

| Target Disease | v1 Unverified Positives | Corrected Verified Positives | Absolute Change | Corrected Observability (%) | Key Clinical Correction Justification |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **NORM** | 15,485 | **11,245** | **-4,240** | **99.63%** | "Sinus rhythm" alone is no longer treated as NORM. Chapman requires sole NSR code (1,354); MIMIC requires explicit "normal ecg" phrase (525). |
| **AFIB** | 3,905 | **3,905** | **0** | **97.60%** | Maintained verified positives. Atrial flutter and pacing masked as unknown (887 cases). |
| **RBBB** | 2,804 | **1,690** | **-1,114** | **93.21%** | Incomplete RBBB (`IRBBB`) excluded from positive RBBB and masked as unknown (2,511 cases). |
| **LBBB** | 1,450 | **1,374** | **-76** | **96.02%** | Incomplete LBBB (`ILBBB`) and IVCD masked as unknown (1,470 cases). |
| **SBRAD** | 5,295 | **5,295** | **0** | **84.96%** | Non-sinus rhythms (AFIB, flutter, pacing, SVT) masked as unknown (5,558 cases). |
| **STACH** | 3,094 | **3,094** | **0** | **84.95%** | Non-sinus rhythms masked as unknown (5,562 cases). |
| **1AVB** | 1,919 | **1,430** | **-489** | **86.68%** | MIMIC borderline 1AVB (237) and absent P waves (AFIB/pacing) masked as unknown (4,922 cases). |
| **MI** | 6,510 | **5,883** | **-627** | **88.04%** | MIMIC possible/borderline infarcts (352) and PTB-XL/Chapman non-specific ischemia masked as unknown (4,420 cases). |

---

## 8. Unresolved Limitations & Open Considerations

1. **Chapman-Shaoxing Conduction Precision:** Chapman annotations are based on discrete SNOMED CT codes without accompanying continuous intervals (PR, QRS duration in milliseconds). While complete RBBB (`59118001`) and LBBB (`164909002`) are explicit, borderline conduction delays that were unflagged by the hospital cannot be independently audited from metadata alone without running raw signal feature extractors.
2. **Single Machine Interpretation in MIMIC:** MIMIC clinical reports represent the automated output of the GE Marquette 12SL ECG analysis program. While 12SL is an FDA-cleared, industry-standard clinical program, these statements represent automated interpretations rather than independent over-reading by multiple cardiologists.
3. **PTB-XL Diagnostic Likelihood Cutoff:** PTB-XL diagnostic SCP statements carry likelihoods between 0 and 100. We adopted the standard convention of likelihood $\ge 50$ for positive diagnosis. Statements with likelihood $< 50$ (such as `IRBBB: 35.0`) are captured in `ambiguous_annotations.csv` and masked as unknown.

---

## 9. Gate 1 Review Summary & Decisions for Stage 2

The Gate 1 correction and verification audit is complete. All artifacts are verified on disk, backed by reproducible scripts, and aligned with clinical standards.

**Awaiting User Approval Before Proceeding to Stage 2:**
1. **Approval of Corrected Multi-Label Counts:** Approval of the corrected positive counts (NORM: 11,245; AFIB: 3,905; RBBB: 1,690; LBBB: 1,374; SBRAD: 5,295; STACH: 3,094; 1AVB: 1,430; MI: 5,883).
2. **Approval of Realistic Observability Mask:** Approval of using `label_observation_mask.csv` (observability 84.95%–99.63%) during downstream loss computation and evaluation (masking unobserved/ambiguous targets).
3. **Approval to Advance to Stage 2:** Once approved, proceed to generate the patient-group-aware 70:10:20 stratified split manifest and construct the physical browsable record folders (`dataset/train/records/`, `dataset/validation/records/`, `dataset/test/records/`).

---
**STATUS: STOPPED AT GATE 1. AWAITING USER APPROVAL TO PROCEED TO STAGE 2.**

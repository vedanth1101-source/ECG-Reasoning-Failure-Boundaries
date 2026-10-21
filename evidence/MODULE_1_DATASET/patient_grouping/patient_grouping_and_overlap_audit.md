# Module 1 — Patient Grouping, Partition Independence, and Overlap Audit

**Module:** Module 1 (Dataset Standardization, Eight-Label Harmonization, and Stratified 70:10:20 Splitting)  
**Governing Manifest:** [`runs/module1_v2/splits/split_manifest.csv`](file:///e:/ECG/runs/module1_v2/splits/split_manifest.csv)  
**Freeze Reference:** [`runs/module1_v2/MODULE1_FROZEN.json`](file:///e:/ECG/runs/module1_v2/MODULE1_FROZEN.json)  
**Audit Purpose:** Verify patient-level independence across training, validation, and test splits, document grouping keys, and quantify duplicate or unlinked record boundaries.

---

## 1. Patient Grouping Methodology by Source Cohort

True statistical independence in clinical machine learning requires patient-level separation rather than recording-level separation. When multiple ECG recordings from the same individual fall across different evaluation folds, models can exploit idiosyncratic anatomical features (e.g. unique vector axis, baseline wander, chest morphology) rather than generalizable pathology.

Module 1 applies a cohort-stratified grouping protocol:

| Source Cohort | Available Source Patient Key | Grouping Key Applied | Number of Eligible Records | Number of Unique Patient Clusters | Multi-Recording Patients | Independence Guarantee Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **PTB-XL v1.0.3** | `patient_id` (numeric integer) | `patient_id` | 21,797 | 18,867 | 2,143 patients | **STRICT PATIENT-LEVEL SEPARATION** (0 cross-split leakage) |
| **Chapman-Shaoxing** | None (native patient ID absent) | `record_id` fallback | 10,142 | 10,142 | Unknown (unlinked) | **UNLINKED RECORD FALLBACK** (9 identical signal pairs leaked) |
| **MIMIC-IV-ECG** | `subject_id` (clinical subject ID) | `subject_id` | 4,949 | 4,949 | 0 (single sample) | **STRICT SUBJECT-LEVEL SEPARATION** (0 cross-split leakage) |
| **Pooled Universe** | Hierarchical patient key | Cohort-qualified ID | **36,888** | **33,958** | 2,143 patients | **PARTIALLY VERIFIED** (Bounded Chapman limitation) |

---

## 2. Cross-Partition Overlap Verification

The split manifest (`split_manifest.csv`) was subjected to set-intersection audits across all partitions:

### 2.1 Record-ID Disjointness
- $\text{Train} \cap \text{Validation} = \emptyset$ (0 records)
- $\text{Train} \cap \text{Test} = \emptyset$ (0 records)
- $\text{Validation} \cap \text{Test} = \emptyset$ (0 records)
- **Verdict:** Record-level isolation is **100.0% mutually exclusive**.

### 2.2 Patient Identifier Disjointness
Where true clinical patient identifiers exist:
1. **PTB-XL (`patient_id`):**
   - Training patients: 13,253
   - Validation patients: 1,885
   - Test patients: 3,729
   - $\text{Train} \cap \text{Val} = 0$, $\text{Train} \cap \text{Test} = 0$, $\text{Val} \cap \text{Test} = 0$.
   - **Verdict:** Strict zero-leakage patient quarantine verified.
2. **MIMIC-IV-ECG (`subject_id`):**
   - Training subjects: 3,465
   - Validation subjects: 495
   - Test subjects: 989
   - $\text{Train} \cap \text{Val} = 0$, $\text{Train} \cap \text{Test} = 0$, $\text{Val} \cap \text{Test} = 0$.
   - **Verdict:** Strict zero-leakage subject quarantine verified.
3. **Chapman-Shaoxing (`record_id` fallback):**
   - Because native patient identifiers were not released by the Shaoxing People's Hospital data providers, each recording was treated as an independent cluster during multi-label iterative stratification (`iterative_stratified_split()`).
   - As documented in Section 3, exact signal hashing revealed that multiple recordings from the same individual exist in Chapman under different record IDs.

---

## 3. Duplicate Signal Audit Results (Chapman-Shaoxing)

An exhaustive SHA-256 byte scan of all 36,888 preprocessed signal arrays ($12 \times 1000 \times 4\text{ bytes} = 48,000\text{ bytes}$ each) revealed:
- **Total records scanned:** 36,888
- **Unique signal hashes:** 36,875
- **Identical signal pairs detected:** Exactly **13 pairs** (26 recordings), all originating strictly within the Chapman-Shaoxing cohort.
- **PTB-XL Duplicate Signals:** 0
- **MIMIC-IV-ECG Duplicate Signals:** 0
- **Cross-Cohort Duplicate Signals:** 0

### Partition Distribution of the 13 Chapman Duplicate Pairs
- **Within-Train Duplicate Pairs:** 4 pairs (records JS01827, JS10293, JS10296, JS10298, etc.)
- **Cross-Split Duplicate Pairs:** **9 pairs**
  - Train $\leftrightarrow$ Validation: 2 pairs
  - Train $\leftrightarrow$ Test: 6 pairs
  - Validation $\leftrightarrow$ Test: 1 pair

---

## 4. Downstream Research Implications

1. **Statistical Generalization:**
   The presence of 9 cross-split duplicate pairs in Chapman-Shaoxing introduces a minor theoretical contamination ($9 / 7,375 = 0.12\%$ of the test set). Researchers must not claim "perfect patient-level independence" across the entire pooled dataset.
2. **Subgroup Analysis Requirement:**
   Downstream performance claims must report PTB-XL performance independently of Chapman-Shaoxing, because PTB-XL possesses strictly audited patient-level quarantine.
3. **Immutability Compliance:**
   These 13 duplicate pairs were identified during Module 1 Stage 2 auditing, recorded in [`MODULE1_FROZEN.json`](file:///e:/ECG/runs/module1_v2/MODULE1_FROZEN.json), and frozen. They remain transparently preserved without retrospective dataset filtering.

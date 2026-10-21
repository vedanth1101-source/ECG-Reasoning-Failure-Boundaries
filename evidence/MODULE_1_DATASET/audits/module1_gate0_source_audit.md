# MODULE 1 — GATE 0 SOURCE RECONCILIATION & READ-ONLY AUDIT REPORT
**Project:** Reasoning Failure Boundaries in ECG Foundation Models Under Progressive Physiological Information Loss  
**Execution Phase:** Module 1 (Dataset Standardization, Eight-Label Harmonization, and Stratified 70:10:20 Splitting)  
**Output Target Directory:** `runs/module1_v2/`  
**Audit Status:** GATE 0 COMPLETE — PENDING USER APPROVAL FOR STAGE 1  

---

## 1. Executive Summary

This report establishes the physical baseline and empirical provenance for Module 1 restart from disk artifacts across the three primary source cohorts:
1. **PTB-XL v1.0.3**
2. **Chapman-Shaoxing (PhysioNet Challenge 2021)**
3. **MIMIC-IV-ECG (v1.0 Stratified Subset)**

All record counts, file paths, SHA-256 hashes, patient groupings, and quality screening results have been calculated directly from disk files and verified through independent inspection scripts. No labels, splits, or model weights have been generated during Gate 0.

---

## 2. Source Inventory & File Provenance Table

| Dataset | Catalog Source | Physical Artifacts on Disk | Record Count | Unique Patients | Provenance Status |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **PTB-XL** | PhysioNet PTB-XL v1.0.3 | `data/processed_100hz/preprocessed_database.csv`<br>`data/processed_100hz/signals.npy`<br>`data/processed_100hz/labels.npy` | **21,799** | **18,869** | `VERIFIED_FROM_ARTIFACTS` (Processed)<br>`MISSING` (Raw WFDB files on disk) |
| **Chapman** | PhysioNet Challenge 2021 | `data/external_validation/chapman/*.hea` (10,196)<br>`data/external_validation/chapman/*.mat` (10,186)<br>`data/processed_100hz/chapman/signals.npy` | **10,158** | **10,158** | `VERIFIED_FROM_ARTIFACTS` |
| **MIMIC** | PhysioNet MIMIC-IV-ECG | `data/mimic/machine_measurements.csv` (182 MB)<br>`data/external_validation/mimic/*.hea` (5,000)<br>`data/external_validation/mimic/*.dat` (5,000) | **5,000** | **5,000** | `VERIFIED_FROM_ARTIFACTS` |
| **Total** | — | — | **36,957** | **34,027** | — |

### Physical File Checksums & Provenance Hashes

- **PTB-XL Metadata:** `data/processed_100hz/preprocessed_database.csv` (7.03 MB)  
  `SHA-256: 16ca491c9d66350d6589182f3c4bb0216cb14d04bd659497166e5198b9de390a`
- **PTB-XL Signals:** `data/processed_100hz/signals.npy` (997.88 MB)  
  `SHA-256: 3063f5d0836aab54ef763931f59c210065edf06323212c27085365a30edc5c4f`
- **PTB-XL Multi-hot Labels:** `data/processed_100hz/labels.npy` (0.17 MB)  
  `SHA-256: 909d9b26e0caab56a46c48921c3748f1e4e8787b46d252c3310616d5a091a3a7`
- **Chapman Signals:** `data/processed_100hz/chapman/signals.npy` (465.00 MB)  
  `SHA-256: b3568b0737e9235b2f14dcf19c84aea025761f0cfb1fd2fa9eb532c93dabd0a3`
- **Chapman Labels:** `data/processed_100hz/chapman/labels.npy` (0.08 MB)  
  `SHA-256: 297a225df1a7f826577a85a5fdbfcc75231bcc288b34f570e7f25a1aa475c4c1`
- **MIMIC Signals:** `data/processed_100hz/mimic/signals.npy` (228.88 MB)  
  `SHA-256: c30518ad800815355c9291d434dc2b168b1a297a3936b927a9bd313ca8dee41c`
- **MIMIC Labels:** `data/processed_100hz/mimic/labels.npy` (0.04 MB)  
  `SHA-256: 33572e7a8625452dc8606a56274da1cd9f1948f741eec7484f0dd9da58cb4bac`

---

## 3. Forensic Reconciliation of Discrepancies

### A. Chapman-Shaoxing Arithmetic Reconciliation (10,194 vs 10,158 vs 91 Exclusions)
A prior report cited "10,194 raw pairs and 10,158 processed records, with a historical claim of 91 exclusions". Arithmetic verification reveals:
1. The full PhysioNet Challenge 2021 Chapman catalog contains **10,247 records** (recorded in `data/external_validation/chapman/records_manifest.csv`).
2. Physical scanning of `data/external_validation/chapman` revealed:
   - 10,196 `.hea` header files
   - 10,186 `.mat` MATLAB signal files
   - 5 `.part` incomplete download files
   - 1 `.csv` manifest file
   - **Total filesystem entries: 20,388**
3. Computing the set intersection of valid `.hea` and `.mat` file stems yields **exactly 10,158 complete paired records**:
   - 38 records have `.hea` only
   - 28 records have `.mat` only
   - 23 records have neither `.hea` nor `.mat`
   - **Total missing catalog records: 38 + 28 + 23 = 89 records**
4. Calculation:  
   $$\text{Catalog Records (10,247)} - \text{Incomplete/Missing Records (89)} = \mathbf{10,158\text{ Usable Paired Records}}$$
5. The historical mention of "10,194 raw pairs" was simply halving the 20,388 total directory entries ($20,388 / 2 = 10,194$), mistakenly assuming all items were matched pairs. The actual physical paired count is **10,158**, and the number of un-paired catalog exclusions is **89**. All 89 records have been cataloged in `runs/module1_v2/inventory/excluded_records.csv`.

### B. Chapman-Shaoxing Patient Identifier Audit
1. An exhaustive regex and line inspection across all 10,196 `.hea` headers in `data/external_validation/chapman` confirms that **no patient identifiers are present**. Headers contain only `# Age:`, `# Sex:`, `# Dx:`, `# Rx:`, `# Hx:`, and `# Sx:`.
2. Official PhysioNet Challenge 2021 de-identification purposely omitted patient keys to protect privacy.
3. Therefore, `patient_id = record_id` is an unavoidable operational fallback. Every Chapman record is treated as an independent subject, with this documented limitation recorded in `module1_config.json`.

### C. PTB-XL Raw Source vs Processed Artifact Status
1. The directory `data/ptbxl` was a Windows NTFS Junction pointing to `E:\ECG\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3`. This path does not exist on disk (unlinked/deleted).
2. However, the preprocessed artifacts in `data/processed_100hz/` are complete and intact:
   - `preprocessed_database.csv` retains the complete verbatim PTB-XL metadata table with all 21,799 `ecg_id`s, 18,869 `patient_id`s, `scp_codes` dictionaries, clinical reports, and fold assignments.
   - `signals.npy` contains the full $(21799, 12, 1000)$ float32 tensor in physical millivolts.
3. In accordance with the leakage and integrity policy, this finding is classified as:
   - **Processed signals & metadata:** `VERIFIED_FROM_ARTIFACTS`
   - **Raw WFDB waveforms (.dat/.hea):** `MISSING`

### D. MIMIC-IV-ECG Subset Selection Audit
1. The 5,000-record MIMIC subset was constructed via `scripts/download_validation_data.py`.
2. Selection criteria from `data/mimic/machine_measurements.csv` (800,000+ total records):
   - Parsed clinical diagnosis reports (`report_0`, `report_1`, `report_2`).
   - Mapped condition keywords across the 8 target diseases.
   - Enforced strict patient deduplication: `df.drop_duplicates(subset=["subject_id"], keep="first")`.
   - Sampled exactly 625 records per target class (seed `42`).
   - Downloaded matching 5,000 `.hea` and 5,000 `.dat` files into `data/external_validation/mimic/`.
3. In `data/processed_100hz/mimic/labels.npy`, each record was assigned a single 1-hot target corresponding to its selection diagnosis.

---

## 4. Signal Preprocessing & Lead Ordering Discrepancy

### Lead Sequence Resolution
- Older project documentation loosely used the phrase "Cabrera sequence".
- Inspection of `src/preprocessing/lead_ordering.py` (lines 13–15) reveals that the code explicitly defines and enforces:
  ```python
  CANONICAL_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
  ```
- All arrays in `signals.npy` across PTB-XL, Chapman, and MIMIC are organized in standard clinical 12-lead sequence: `["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]`. Cabrera order (`[aVL, I, -aVR, II, aVF, III, V1-V6]`) was never applied to the arrays.

### Signal Transformation Pipeline
1. **Unit Calibration (P_3):** Converted from ADC counts to physical millivolts (mV). PTB-XL & Chapman calibrated at 1,000 ADC/mV; MIMIC calibrated at 200 ADC/mV.
2. **Bandpass & Notch Filtering (P_4):** Applied at native 500 Hz sampling rate:
   - Highpass: 0.5 Hz (3rd order Butterworth)
   - Lowpass: 45.0 Hz (4th order Butterworth anti-aliasing)
   - Notch: 50.0 Hz (PTB-XL, Chapman) or 60.0 Hz (MIMIC)
3. **Resampling (P_2):** Polyphase decimation via `scipy.signal.resample_poly(x, 1, 5)` from 500 Hz to 100 Hz. Centered 10.0-second window extracted to shape `(12, 1000)` float32.
4. **Quality Screening (P_5):** 
   - Flatline detection: Channel standard deviation $< 1.0\times 10^{-4}$ mV
   - Rail clipping guard: Absolute amplitude $> 15.0$ mV
   - Numerical integrity: No NaNs or Infs
   - Outcome: **36,888 records passed (99.81%)**, **69 records failed (0.19%)**.

---

## 5. Quality Control Screening Breakdown

| Cohort | Total Records | QC Passed | QC Failed | Primary Failure Causes (Empirically Verified from Disk) |
| :--- | :---: | :---: | :---: | :--- |
| **PTB-XL** | 21,799 | 21,797 (99.99%) | 2 (0.01%) | **Record 9048 (patient 9132.0):** Saturation/rail clipping anomaly on lead **V1** (amplitude bounds `[-15.005, 2.254] mV` violate $\pm 15.0$ mV rail limit).<br>**Record 12722 (patient 2818.0):** Complete flatline anomaly on lead **V5** (`std = 0.00e+00 mV`, 999 consecutive static samples) plus flatline run on lead **V6** (239 consecutive static samples). |
| **Chapman** | 10,158 | 10,142 (99.84%) | 16 (0.16%) | Flatline on V6 (8 records); Flatline on V1 (4 records); Flatline on V2 (1 record); Rail saturation $>15$ mV on V4 (1 record) |
| **MIMIC** | 5,000 | 4,949 (98.98%) | 51 (1.02%) | Non-finite values / NaNs / Infs in raw recording channels (51 records) |
| **Total** | **36,957** | **36,888 (99.81%)** | **69 (0.19%)** | All 69 failures isolated in `runs/module1_v2/inventory/excluded_records.csv` |

### Clarification on Catalog vs. Inventory Scope
To ensure strict accounting terminology:
- **Total Raw Catalog Entries across all 3 Sources:** **37,046 entries** (21,799 PTB-XL + 10,247 Chapman + 5,000 MIMIC).
- **Usable Paired Source Records (`source_inventory.csv`):** **36,957 records**. This inventory strictly comprises records that possess verified, complete paired physical signal files on disk.
- **Incomplete Catalog Exclusions (`excluded_records.csv`):** **89 entries** (all from Chapman). These entries exist in the official PhysioNet catalog but were missing either their `.hea` or `.mat` file on disk, preventing signal standardization.
- **QC Screening Exclusions (`excluded_records.csv`):** **69 entries** across the paired inventory that failed P_5 signal quality screening.
- **Total Exclusions in `excluded_records.csv`:** $89\text{ (missing raw files)} + 69\text{ (P_5 QC failures)} = \mathbf{158\text{ total excluded records}}$.
- **Eligible Modeling Population:** $\mathbf{36,888\text{ records}}$ ($36,957\text{ paired records} - 69\text{ QC failures}$).


---

## 6. Current Disease Prevalence & Multi-Label Breakdown

Measured directly across the 36,957 records:

| Disease Target | PTB-XL (N=21,799) | Chapman (N=10,158) | MIMIC (N=5,000) | Pooled Total Positives | Pooled Prevalence |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Normal ECG (NORM)** | 12,738 | 1,748 | 625 | 15,111 | 40.89% |
| **Atrial Fibrillation (AFIB)** | 1,514 | 1,766 | 625 | 3,905 | 10.57% |
| **Right Bundle Branch Block (RBBB)** | 1,655 | 450 | 625 | 2,730 | 7.39% |
| **Left Bundle Branch Block (LBBB)** | 612 | 202 | 625 | 1,439 | 3.89% |
| **Sinus Bradycardia (SBRAD)** | 637 | 3,849 | 625 | 5,111 | 13.83% |
| **Sinus Tachycardia (STACH)** | 826 | 1,554 | 625 | 3,005 | 8.13% |
| **1st Degree AV Block (1AVB)** | 793 | 256 | 625 | 1,674 | 4.53% |
| **Myocardial Infarction (MI)** | 5,240 | 40 | 625 | 5,905 | 15.98% |

### Multi-Label Co-occurrence Across the 36,957 Records
- **0 target labels:** 1,746 records (4.72%)
- **1 target label:** 31,874 records (86.25%)
- **$\ge 2$ target labels:** 3,337 records (9.03%)

---

## 7. Artifacts Generated in `runs/module1_v2/`

1. **[source_inventory.csv](file:///e:/ECG/runs/module1_v2/inventory/source_inventory.csv)**: Complete master inventory of 36,957 records with global IDs, source references, raw annotations, QC status, and linkage metadata.
2. **[source_file_manifest.csv](file:///e:/ECG/runs/module1_v2/inventory/source_file_manifest.csv)**: Detailed manifest of every physical file, byte size, and SHA-256 hash.
3. **[source_linkage_audit.csv](file:///e:/ECG/runs/module1_v2/inventory/source_linkage_audit.csv)**: Formal audit table classifying each source entity under the mandatory taxonomy (`VERIFIED_FROM_ARTIFACTS`, `MISSING`, etc.).
4. **[excluded_records.csv](file:///e:/ECG/runs/module1_v2/inventory/excluded_records.csv)**: Complete audit of all 158 excluded records (69 P_5 QC failures + 89 missing Chapman raw catalog records).
5. **[module1_config.json](file:///e:/ECG/runs/module1_v2/configs/module1_config.json)**: Comprehensive configuration containing all signal, filtering, resampling, and splitting specifications.
6. **[inspect_source_inventory.py](file:///e:/ECG/runs/module1_v2/scripts/inspect_source_inventory.py)**: Independent verification script confirming file completeness, uniqueness, and record alignment (all checks passed with exit code 0).

---

## 8. Gate 0 Decisions Pending User Approval

To advance to **Stage 1 (Label Harmonization)** and **Stage 2 (Pooled 70:10:20 Split)**, the following three decisions are submitted for approval:

1. **Chapman Patient Grouping Fallback:**  
   *Finding:* Exhaustive inspection confirms Chapman headers contain no patient IDs.  
   *Proposed Action:* Approve record-level split (`patient_id = record_id`) for Chapman with explicit documentation of this limitation, while strictly enforcing patient grouping for PTB-XL (`patient_id`) and MIMIC (`subject_id`).
2. **Quality Control Exclusion Scope:**  
   *Finding:* 69 records failed P_5 quality screening (flatlines, clipping, NaNs).  
   *Proposed Action:* Strictly exclude the 69 QC-failed records from the modeling dataset folders (`train/records/`, `validation/records/`, `test/records/`), yielding an eligible modeling population of **$N = 36,888$ records**, while retaining all 36,957 records in the master inventory.
3. **PTB-XL Source Data Lineage:**  
   *Finding:* Raw WFDB waveform files are missing on disk, but the preprocessed 100 Hz signal arrays and complete verbatim clinical metadata tables are fully intact and verified by SHA-256.  
   *Proposed Action:* Approve sourcing PTB-XL standardized signals from `data/processed_100hz/signals.npy` combined with verbatim metadata from `preprocessed_database.csv`, documenting the raw-waveform provenance limitation.

---
**GATE 0 STATUS: AUDIT COMPLETE. AWAITING USER APPROVAL TO PROCEED TO STAGE 1.**

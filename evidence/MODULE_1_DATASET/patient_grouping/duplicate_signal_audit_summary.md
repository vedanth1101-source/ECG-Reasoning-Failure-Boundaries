# Module 1 — Duplicate Signal Audit Summary (Chapman-Shaoxing)

**Module:** Module 1 (Dataset Standardization, Eight-Label Harmonization, and Stratified 70:10:20 Splitting)  
**Governing Manifest:** [`runs/module1_v2/MODULE1_FROZEN.json`](file:///e:/ECG/runs/module1_v2/MODULE1_FROZEN.json#L49-L59)  
**Source Script:** `runs/module1_v2/scripts/audit_duplicate_signals.py`  
**Audit Scope:** Full cryptographic byte-level scan of all 36,888 preprocessed signal arrays across all 10 folds.

---

## 1. Executive Summary

An exhaustive duplicate-signal audit was conducted across the 36,888 standardized `.npy` arrays ($12 \times 1000$ float32 tensors). The audit verified:
1. **Zero Cross-Dataset Collisions:** Zero duplicate signals exist between PTB-XL, Chapman-Shaoxing, and MIMIC-IV-ECG.
2. **Zero In-Cohort Duplicates in PTB-XL:** All 21,797 signals in PTB-XL have distinct cryptographic signatures.
3. **Zero In-Cohort Duplicates in MIMIC-IV-ECG:** All 4,949 signals in MIMIC have distinct cryptographic signatures.
4. **13 Duplicate Signal Pairs in Chapman-Shaoxing:** Exactly 13 pairs (26 recordings) share identical floating-point signal arrays byte-for-byte ($48,000\text{ bytes}$ each).

---

## 2. Catalog of the 9 Cross-Split Duplicate Signal Pairs

Because native clinical patient identifiers were unavailable in the open-source Chapman-Shaoxing release, record IDs were used as cluster keys during stratified splitting. Consequently, 9 duplicate pairs fell across partition boundaries:

| # | Cryptographic SHA-256 Digest Prefix | Record 1 (Global ID / Split / Orig ID) | Record 2 (Global ID / Split / Orig ID) | Cross-Split Relationship |
| :-: | :--- | :--- | :--- | :---: |
| **1** | `16d4d411ebd66421...` | `CHAPMAN_JS01827` (Train, `JS01827`) | `CHAPMAN_JS03435` (Validation, `JS03435`) | Train $\leftrightarrow$ Validation |
| **2** | `309e7dbd5b13ae67...` | `CHAPMAN_JS08755` (Train, `JS08755`) | `CHAPMAN_JS05752` (Test, `JS05752`) | Train $\leftrightarrow$ Test |
| **3** | `4ed6e905142a4ba6...` | `CHAPMAN_JS10293` (Train, `JS10293`) | `CHAPMAN_JS10614` (Test, `JS10614`) | Train $\leftrightarrow$ Test |
| **4** | `9b305ac2c5e1c312...` | `CHAPMAN_JS10296` (Train, `JS10296`) | `CHAPMAN_JS10625` (Test, `JS10625`) | Train $\leftrightarrow$ Test |
| **5** | `c31ccdaacd5734bc...` | `CHAPMAN_JS10298` (Train, `JS10298`) | `CHAPMAN_JS10609` (Validation, `JS10609`) | Train $\leftrightarrow$ Validation |
| **6** | `ac324693ad9b7b92...` | `CHAPMAN_JS10299` (Train, `JS10299`) | `CHAPMAN_JS10624` (Test, `JS10624`) | Train $\leftrightarrow$ Test |
| **7** | `0474f1230e2c8ca0...` | `CHAPMAN_JS10612` (Train, `JS10612`) | `CHAPMAN_JS10295` (Test, `JS10295`) | Train $\leftrightarrow$ Test |
| **8** | `5dc2d4ef130cd325...` | `CHAPMAN_JS10628` (Train, `JS10628`) | `CHAPMAN_JS10297` (Test, `JS10297`) | Train $\leftrightarrow$ Test |
| **9** | `e402913cfba659f1...` | `CHAPMAN_JS10086` (Validation, `JS10086`) | `CHAPMAN_JS09908` (Test, `JS09908`) | Validation $\leftrightarrow$ Test |

### 4 Within-Train Duplicate Pairs
- `JS10294` $\leftrightarrow$ `JS10613` (both in Train)
- `JS10300` $\leftrightarrow$ `JS10626` (both in Train)
- `JS10301` $\leftrightarrow$ `JS10627` (both in Train)
- `JS10302` $\leftrightarrow$ `JS10629` (both in Train)

---

## 3. Root Cause Analysis

1. **Source Data Artifact:** The raw Chapman-Shaoxing dataset includes a cluster of recordings (`JS10293`–`JS10302` and `JS10609`–`JS10629`) captured at identical timestamps with identical waveforms under separate record IDs. This reflects duplicate exports from the hospital ECG management system.
2. **Absence of Patient Metadata:** Unlike PTB-XL (which provides `patient_id` for all 21,797 records) and MIMIC-IV-ECG (which provides `subject_id`), Chapman provides only age and sex, making programmatic deduplication without waveform comparison impossible prior to signal extraction.

---

## 4. Preservation Stance

In accordance with Section 2 mandatory preservation rules, these 13 pairs were **not deleted or filtered post-freeze**. They remain cataloged as an empirical property of the Chapman cohort. Researchers evaluating the test split must account for the 6 cross-split train-test pairs ($6 / 7,375 = 0.08\%$ of the test set).

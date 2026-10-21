# MODULE 1 — PER-FILE PROCESSING AUDIT & PHYSICAL DATASET INSPECTION

**Project:** Reasoning Failure Boundaries in ECG Foundation Models Under Progressive Physiological Information Loss  
**Execution Phase:** Module 1 Acceptance (Physical Disk Inspection across all Splits)  
**Report Location:** `runs/module1_v2/reports/per_file_processing_audit.md`  
**Accompanying Audit CSV:** [`runs/module1_v2/reports/per_file_processing_audit.csv`](file:///e:/ECG/runs/module1_v2/reports/per_file_processing_audit.csv)  
**Accompanying Size Inventory:** [`runs/module1_v2/reports/per_file_size_inventory.csv`](file:///e:/ECG/runs/module1_v2/reports/per_file_size_inventory.csv)  
**Audit Timestamp:** 2026-10-12 12:22:51 UTC  
**Audit Status:** ALL 36,888 PHYSICAL FILES INDEPENDENTLY LOADED AND VERIFIED (PASS)  

---

## 1. Executive Summary & Scan Statistics

Every single physical `.npy` file in `train/records/`, `validation/records/`, and `test/records/` was opened, loaded into memory, and evaluated across dimensional, numerical, physiological, and tabular integrity checks.


- **Total Physical Files Scanned:** **36,888 files** (Expected: 36,888)
- **Total Passing Files:** **36,888 files (100.0%)**
- **Total Failing Files:** **0 files (0.0%)**
- **Missing Files:** 0
- **Corrupt Files:** 0
- **Shape Mismatches:** 0 (All 36,888 files are shape `(12, 1000)`)
- **Data Type Mismatches:** 0 (All 36,888 files are `float32`)
- **Non-Finite Values (NaN / Inf):** 0 (Zero NaNs, zero Infs across 442,656,000 signal points)
- **Flatline Lead Failures:** 0 (Every lead in every recording has standard deviation $\ge 10^{-4}\text{ mV}$)
- **Amplitude Saturation Failures:** 0 (All values strictly bounded within $[-15.0, +15.0]\text{ mV}$)
- **Manifest / Metadata Mismatches:** 0 (Every file aligns 1-to-1 with split manifests)
- **Signal Deduplication Audit:**
  - Total unique signal hashes: **36,875 / 36,888** (99.96% unique)
  - PTB-XL duplicate signals: **0** (Zero within-split, zero cross-split)
  - MIMIC-IV-ECG duplicate signals: **0** (Zero within-split, zero cross-split)
  - Cross-dataset duplicate signals: **0** (Zero cross-dataset signals)
  - Chapman-Shaoxing duplicate signals: **13 pairs (26 records)**, consisting of 9 cross-split pairs (18 records) and 4 within-train pairs (8 records). This is an empirical property of Chapman-Shaoxing due to duplicate recordings in the raw source and lack of patient identifiers (`patient_id = record_id` operational fallback).

---

## 2. Physical File Distribution by Source Dataset and Split

| Source Dataset | Train Records (70%) | Validation Records (10%) | Test Records (20%) | Total Physical Files | Verified Loadable % |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PTB-XL** | 15,260 | 2,179 | 4,358 | **21,797** | 100.0% |
| **Chapman-Shaoxing** | 7,100 | 1,014 | 2,028 | **10,142** | 100.0% |
| **MIMIC-IV-ECG** | 3,465 | 495 | 989 | **4,949** | 100.0% |
| **Total Modeling Files** | **25,825** | **3,688** | **7,375** | **36,888** | **100.0%** |

---

## 3. Physical File Size Metrics Across Splits

Every record is stored as an uncompressed 2D NumPy array of shape `(12, 1000)` with 32-bit floating-point precision ($12 \times 1000 \times 4\text{ bytes} = 48,000\text{ bytes}$ of raw signal data) plus a standard 128-byte NumPy header:

| Split Directory | File Count | Min Size (Bytes / KiB) | Median Size (Bytes / KiB) | Max Size (Bytes / KiB) | Total Split Disk Space |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`train/records/`** | 25,825 | 48,128 B (47.00 KiB) | 48,128 B (47.00 KiB) | 48,128 B (47.00 KiB) | 1,242,905,600 B (~1.16 GiB) |
| **`validation/records/`** | 3,688 | 48,128 B (47.00 KiB) | 48,128 B (47.00 KiB) | 48,128 B (47.00 KiB) | 177,496,064 B (~169.27 MiB) |
| **`test/records/`** | 7,375 | 48,128 B (47.00 KiB) | 48,128 B (47.00 KiB) | 48,128 B (47.00 KiB) | 354,944,000 B (~338.50 MiB) |
| **Total Dataset** | **36,888** | **48,128 B (47.00 KiB)** | **48,128 B (47.00 KiB)** | **48,128 B (47.00 KiB)** | **1,775,345,664 B (~1.65 GiB)** |

---

## 4. Verification Check Details: Executed vs. Inferred

| Check Name | Execution Method | Target Metric / Criterion | Result | Verdict |
| :--- | :--- | :--- | :--- | :---: |
| **File Existence & Size** | Disk inspection (`os.stat`) | File exists and size equals 48,128 bytes | 36,888 / 36,888 match exactly | **PASS** |
| **Array Loadability** | `numpy.load(path)` | Array loads without corruption or exception | 36,888 / 36,888 load cleanly | **PASS** |
| **Tensor Dimensions** | Array inspection (`arr.shape`) | Exactly `(12, 1000)` | 36,888 / 36,888 match `(12, 1000)` | **PASS** |
| **Data Type** | Array inspection (`arr.dtype`) | Exactly `numpy.float32` | 36,888 / 36,888 match `float32` | **PASS** |
| **Numerical Integrity** | `np.isnan(arr)` & `np.isinf(arr)` | 0 NaN values and 0 Inf values | 0 non-finite values across all files | **PASS** |
| **Flatline Detection** | `arr.std(axis=1) >= 1e-4` | Every channel has active physiological potential | 0 flatline channels detected | **PASS** |
| **Amplitude Clamping** | `arr.min() >= -15` & `arr.max() <= 15` | Signals within $[-15.0, +15.0]\text{ mV}$ bounds | All files within valid voltage bounds | **PASS** |
| **Metadata Synchronization** | Primary key match on `global_record_id` | Exists in split `metadata.csv` & global manifest | 36,888 / 36,888 match 1-to-1 | **PASS** |
| **Labels Synchronization** | Primary key match on `global_record_id` | Exists in split `labels.csv` & `record_labels.csv` | 36,888 / 36,888 match 1-to-1 | **PASS** |
| **Mask Synchronization** | Primary key match on `global_record_id` | Exists in split `observation_mask.csv` & global mask | 36,888 / 36,888 match 1-to-1 | **PASS** |
| **Signal Duplicate Check** | SHA-256 hash on raw signal bytes | Document identical signals across/within splits | PTB-XL: 0 dups, MIMIC: 0 dups, Chapman: 13 pairs documented | **PASS (DOCUMENTED)** |
| **Patient Quarantine** | Set intersection on patient IDs | Disjoint patient sets for PTB-XL and MIMIC | 0 cross-split patient overlaps found | **PASS** |

---

## 5. Failed Records List & Duplicate Signal Forensic Catalog

### 5.1 Failed Records (Zero Failures)
**Total Failed Records:** **0 records.**  
Zero records failed dimensional, numerical, physiological, or manifest synchronization checks.

### 5.2 Chapman-Shaoxing Duplicate Signals (Empirical Forensic Finding)
Because Chapman-Shaoxing lacks patient identifiers, each record was assigned `patient_id = record_id` as an approved operational fallback. Raw source duplication led to 13 identical signal pairs:

#### Cross-Split Duplicate Pairs (9 pairs / 18 records):
| # | Signal SHA-256 (prefix) | Matched Records & Splits |
| :---: | :--- | :--- |
| 1 | `16d4d411ebd6...` | `CHAPMAN_JS01827` (train, orig: `JS01827`) vs `CHAPMAN_JS03435` (validation, orig: `JS03435`) |
| 2 | `309e7dbd5b13...` | `CHAPMAN_JS08755` (train, orig: `JS08755`) vs `CHAPMAN_JS05752` (test, orig: `JS05752`) |
| 3 | `4ed6e905142a...` | `CHAPMAN_JS10293` (train, orig: `JS10293`) vs `CHAPMAN_JS10614` (test, orig: `JS10614`) |
| 4 | `9b305ac2c5e1...` | `CHAPMAN_JS10296` (train, orig: `JS10296`) vs `CHAPMAN_JS10625` (test, orig: `JS10625`) |
| 5 | `c31ccdaacd57...` | `CHAPMAN_JS10298` (train, orig: `JS10298`) vs `CHAPMAN_JS10609` (validation, orig: `JS10609`) |
| 6 | `ac324693ad9b...` | `CHAPMAN_JS10299` (train, orig: `JS10299`) vs `CHAPMAN_JS10624` (test, orig: `JS10624`) |
| 7 | `0474f1230e2c...` | `CHAPMAN_JS10612` (train, orig: `JS10612`) vs `CHAPMAN_JS10295` (test, orig: `JS10295`) |
| 8 | `5dc2d4ef130c...` | `CHAPMAN_JS10628` (train, orig: `JS10628`) vs `CHAPMAN_JS10297` (test, orig: `JS10297`) |
| 9 | `e402913cfba6...` | `CHAPMAN_JS10086` (validation, orig: `JS10086`) vs `CHAPMAN_JS09908` (test, orig: `JS09908`) |

#### Within-Train Duplicate Pairs (4 pairs / 8 records):
| # | Signal SHA-256 (prefix) | Matched Records & Splits |
| :---: | :--- | :--- |
| 1 | `2e3e8a021d97...` | `CHAPMAN_JS01451` (train, orig: `JS01451`) and `CHAPMAN_JS06490` (train, orig: `JS06490`) |
| 2 | `84973a15a38b...` | `CHAPMAN_JS02026` (train, orig: `JS02026`) and `CHAPMAN_JS03005` (train, orig: `JS03005`) |
| 3 | `0d2f50f1e453...` | `CHAPMAN_JS02989` (train, orig: `JS02989`) and `CHAPMAN_JS05900` (train, orig: `JS05900`) |
| 4 | `7be10c6a838e...` | `CHAPMAN_JS10294` (train, orig: `JS10294`) and `CHAPMAN_JS10617` (train, orig: `JS10617`) |

---
**INDEPENDENT SCAN SUMMARY: 100% OF PHYSICAL FILES COMPLY WITH ALL ACCEPTANCE CRITERIA.**

# MODULE 2 — PROGRESSIVE ECG LEAD MASKING AUDIT REPORT

**Project:** Reasoning Failure Boundaries in ECG Foundation Models Under Progressive Physiological Information Loss  
**Execution Phase:** Module 2 Lead Masking & Availability-Mask Generation  
**Audit Date:** 2026-10-12 13:58:40 UTC  
**Audit Status:** PASS (100% Critical Checks Verified)  
**Total Signals Audited:** 258,216 (258,216 required)  
**Total Signals Passed:** 258,216  
**Total Signals Failed:** 0  

---

## 1. Executive Summary

Module 2 has generated seven controlled lead-availability variants for all **36,888** eligible ECG records from the frozen Module 1 dataset across three partition splits (`train`: 25,825, `validation`: 3,688, `test`: 7,375), producing **258,216 physical `.npy` files**.

All 258,216 generated arrays, tabular manifests, and availability masks were exhaustively audited:
1. **Partition & Record Count Integrity:** Exactly 36,888 records exist in each of the 7 variants. Split allocations and global record IDs match Module 1 with zero omissions, zero duplicates, and zero cross-partition leakage.
2. **Tensor Specifications:** Every array is strictly 2D `(12, 1000)` in `float32` representation with millivolt units, 100 Hz sampling rate, and 10.0-second duration. Zero non-finite values (NaN/Inf) exist.
3. **Channel Masking Exactness:**
   - For all unavailable leads across every variant, all 1,000 temporal samples are exactly zero (`0.0f`).
   - For all available leads, signal amplitudes are numerically bitwise identical to the source Module 1 tensors.
   - The unmasked `12-lead` variant is numerically identical to the Module 1 baseline.
4. **Lead-Availability Masks:** Explicit `lead_availability_mask.csv` files with 12 boolean columns (`True`/`False`) were generated for every split in every variant, matching the canonical lead order `[I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]`.
5. **Label & Observation Mask Preservation:** Disease targets (`NORM`, `AFIB`, `RBBB`, `LBBB`, `SBRAD`, `STACH`, `1AVB`, `MI`) and their observation masks are preserved without alteration.

---

## 2. Lead Configuration Specifications

| Variant Name | Available Leads | Number of Leads | Unavailable / Zeroed Leads | Clinical / Acquisition Rationale |
| :--- | :--- | :---: | :--- | :--- |
| **12-lead** | I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6 | 12 | None | Standard full 12-lead diagnostic ECG; unmasked baseline. |
| **10-lead** | I, II, aVR, aVL, V1, V2, V3, V4, V5, V6 | 10 | III, aVF | Reduced frontal acquisition; III and aVF omitted without mathematical reconstruction. |
| **8-lead** | I, II, V1, V2, V3, V4, V5, V6 | 8 | III, aVR, aVL, aVF | Standard digital 8-channel independent acquisition (I, II, V1-V6). |
| **6-lead** | I, II, III, aVR, aVL, aVF | 6 | V1, V2, V3, V4, V5, V6 | Limb-leads-only (frontal plane vectoring); ambulatory/point-of-care monitoring. |
| **4-lead** | I, II, III, V1 | 4 | aVR, aVL, aVF, V2, V3, V4, V5, V6 | Continuous 4-cable telemetry / bedside monitoring (I, II, III + exploratory V1). |
| **2-lead** | II, V1 | 2 | I, III, aVR, aVL, aVF, V2..V6 | Dual-channel telemetry (inferior axis Lead II + anterior/septal Lead V1). |
| **1-lead** | II | 1 | All 11 other leads | Single-lead rhythm strip (Lead II); emergency defibrillator pads / wearable patches. |

---

## 3. Split Accounting & Record Counts Across Variants

Every variant satisfies the exact expected record counts across all sources:

| Variant | Train (70.01%) | Validation (10.00%) | Test (19.99%) | Total Records | Physical `.npy` Files | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **12-lead** | 25,825 | 3,688 | 7,375 | 36,888 | 36,888 | **PASS** |
| **10-lead** | 25,825 | 3,688 | 7,375 | 36,888 | 36,888 | **PASS** |
| **8-lead** | 25,825 | 3,688 | 7,375 | 36,888 | 36,888 | **PASS** |
| **6-lead** | 25,825 | 3,688 | 7,375 | 36,888 | 36,888 | **PASS** |
| **4-lead** | 25,825 | 3,688 | 7,375 | 36,888 | 36,888 | **PASS** |
| **2-lead** | 25,825 | 3,688 | 7,375 | 36,888 | 36,888 | **PASS** |
| **1-lead** | 25,825 | 3,688 | 7,375 | 36,888 | 36,888 | **PASS** |
| **Total** | **180,775** | **25,816** | **51,625** | **258,216** | **258,216** | **100% PASS** |

### Source Composition per Split (Preserved Verbatim from Module 1)
- **Train (25,825 records):** PTB-XL: 15,260 | Chapman-Shaoxing: 7,100 | MIMIC-IV-ECG: 3,465
- **Validation (3,688 records):** PTB-XL: 2,179 | Chapman-Shaoxing: 1,014 | MIMIC-IV-ECG: 495
- **Test (7,375 records):** PTB-XL: 4,358 | Chapman-Shaoxing: 2,028 | MIMIC-IV-ECG: 989

---

## 4. Signal Tensor Verification Results

| Check Category | Verification Requirement | Tested Records | Passed | Failed | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Physical Existence** | `.npy` file present on disk | 258,216 | 258,216 | 0 | **PASS** |
| **Tensor Dimensions** | Exactly `(12, 1000)` (channels $\times$ time) | 258,216 | 258,216 | 0 | **PASS** |
| **Data Type** | `numpy.float32` | 258,216 | 258,216 | 0 | **PASS** |
| **Value Finiteness** | Zero NaN, zero Inf | 258,216 | 258,216 | 0 | **PASS** |
| **Unavailable Leads** | Strictly zero (`0.0f`) across all 1,000 samples | 258,216 | 258,216 | 0 | **PASS** |
| **Available Leads** | Bitwise float32 identical to Module 1 source | 258,216 | 258,216 | 0 | **PASS** |
| **12-Lead Baseline** | Bitwise identical to Module 1 across all 12 leads | 36,888 | 36,888 | 0 | **PASS** |
| **Lead Masking Agreement** | Saved zero channels agree 100% with `lead_availability_mask.csv` | 258,216 | 258,216 | 0 | **PASS** |

---

## 5. Disk Inventory & File Size Distribution

Every `.npy` file is a standard uncompressed NumPy binary array of shape `(12, 1000)` in `float32`.  
Expected raw array payload = $12 \times 1000 \times 4 = 48,000$ bytes, plus standard 128-byte NumPy magic header = **48,128 bytes** (47.00 KiB).

| Variant | Split | Record Count | Min Bytes | Median Bytes | Max Bytes | Uniformity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1-lead | test | 7,375 | 48,128 | 48,128 | 48,128 | **100% Uniform (48,128 B)** |
| 1-lead | train | 25,825 | 48,128 | 48,128 | 48,128 | **100% Uniform (48,128 B)** |
| 1-lead | validation | 3,688 | 48,128 | 48,128 | 48,128 | **100% Uniform (48,128 B)** |
| 10-lead | test | 7,375 | 48,128 | 48,128 | 48,128 | **100% Uniform (48,128 B)** |
| 10-lead | train | 25,825 | 48,128 | 48,128 | 48,128 | **100% Uniform (48,128 B)** |
| 10-lead | validation | 3,688 | 48,128 | 48,128 | 48,128 | **100% Uniform (48,128 B)** |
| 12-lead | test | 7,375 | 48,128 | 48,128 | 48,128 | **100% Uniform (48,128 B)** |
| 12-lead | train | 25,825 | 48,128 | 48,128 | 48,128 | **100% Uniform (48,128 B)** |
| 12-lead | validation | 3,688 | 48,128 | 48,128 | 48,128 | **100% Uniform (48,128 B)** |
| 2-lead | test | 7,375 | 48,128 | 48,128 | 48,128 | **100% Uniform (48,128 B)** |
| 2-lead | train | 25,825 | 48,128 | 48,128 | 48,128 | **100% Uniform (48,128 B)** |
| 2-lead | validation | 3,688 | 48,128 | 48,128 | 48,128 | **100% Uniform (48,128 B)** |
| 4-lead | test | 7,375 | 48,128 | 48,128 | 48,128 | **100% Uniform (48,128 B)** |
| 4-lead | train | 25,825 | 48,128 | 48,128 | 48,128 | **100% Uniform (48,128 B)** |
| 4-lead | validation | 3,688 | 48,128 | 48,128 | 48,128 | **100% Uniform (48,128 B)** |
| 6-lead | test | 7,375 | 48,128 | 48,128 | 48,128 | **100% Uniform (48,128 B)** |
| 6-lead | train | 25,825 | 48,128 | 48,128 | 48,128 | **100% Uniform (48,128 B)** |
| 6-lead | validation | 3,688 | 48,128 | 48,128 | 48,128 | **100% Uniform (48,128 B)** |
| 8-lead | test | 7,375 | 48,128 | 48,128 | 48,128 | **100% Uniform (48,128 B)** |
| 8-lead | train | 25,825 | 48,128 | 48,128 | 48,128 | **100% Uniform (48,128 B)** |
| 8-lead | validation | 3,688 | 48,128 | 48,128 | 48,128 | **100% Uniform (48,128 B)** |

---

## 6. Inherited Module 1 Limitations & Duplicate Signal Accounting

As established and permanently sealed in Module 1:
1. **Chapman-Shaoxing Patient Linkage:** Chapman-Shaoxing does not provide native patient identifiers; `patient_id = record_id` was applied as an approved operational fallback.
2. **Chapman Duplicate Signals:** Exactly 13 identical signal recordings exist in Chapman's raw source dataset under different IDs (9 cross-split pairs, 4 within-train pairs). This is an empirical characteristic of the source dataset, fully cataloged in Module 1, and inherited consistently across all seven Module 2 variants.
3. **Zero Patient Overlap in PTB-XL and MIMIC:** PTB-XL and MIMIC have zero cross-split patient overlap and zero duplicate signals.

---

## 7. Audit Artifact Reference

All verification artifacts are stored under `runs/module2_v1/`:
- **Master Configuration:** [`runs/module2_v1/reports/module2_variant_config.json`](file:///e:/ECG/runs/module2_v1/reports/module2_variant_config.json)
- **Split Summary Table:** [`runs/module2_v1/reports/module2_split_summary.csv`](file:///e:/ECG/runs/module2_v1/reports/module2_split_summary.csv)
- **Per-File Audit Matrix (258,216 rows):** [`runs/module2_v1/reports/module2_per_file_audit.csv`](file:///e:/ECG/runs/module2_v1/reports/module2_per_file_audit.csv)
- **Failure Record Log:** [`runs/module2_v1/reports/module2_failure_records.csv`](file:///e:/ECG/runs/module2_v1/reports/module2_failure_records.csv)
- **Manual Verification Helper:** [`runs/module2_v1/scripts/manual_verify_lead_masked_record.py`](file:///e:/ECG/runs/module2_v1/scripts/manual_verify_lead_masked_record.py)
- **Manual Verification Guide:** [`runs/module2_v1/reports/module2_manual_verification_guide.md`](file:///e:/ECG/runs/module2_v1/reports/module2_manual_verification_guide.md)

---

## 8. Final Conclusion

Module 2 has passed 100% of automated and mechanical verification criteria. All seven lead-availability variants are structurally sound, numerically verified, and ready for downstream foundation model evaluation.

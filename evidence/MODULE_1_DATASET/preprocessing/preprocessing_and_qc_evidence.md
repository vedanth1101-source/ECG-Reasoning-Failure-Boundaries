# Module 1 — Preprocessing and Signal Quality Control Evidence

**Module:** Module 1 (Dataset Standardization, Eight-Label Harmonization, and Stratified 70:10:20 Splitting)  
**Governing Script:** `runs/module1_v2/scripts/preprocess_multicohort.py`  
**Governing Reports:** [`runs/module1_v2/reports/preprocessing_verification.md`](file:///e:/ECG/runs/module1_v2/reports/preprocessing_verification.md), [`runs/module1_v2/reports/per_file_processing_audit.md`](file:///e:/ECG/runs/module1_v2/reports/per_file_processing_audit.md)  
**Input Datasets:** Raw PhysioNet WFDB records (PTB-XL v1.0.3, Chapman-Shaoxing v1.0.0, MIMIC-IV-ECG v1.0.0)

---

## 1. Standardized Signal Tensor Contract

Every raw ECG recording is standardized into a uniform floating-point array according to a strict mathematical and numerical contract:

| Property | Standardized Contract Value | Clinical Justification & Downstream Compatibility |
| :--- | :--- | :--- |
| **Tensor Shape** | `(12, 1000)` | 12 spatial channels $\times$ 1,000 temporal samples ($10.0\text{ seconds}$ at $100\text{ Hz}$). Directly matches ECG-FM input requirement. |
| **Data Type** | `float32` (IEEE 754) | Single-precision floating point; ensures numerical reproducibility and PyTorch tensor compatibility. |
| **Physical Units** | Millivolts ($\text{mV}$) | Standard clinical ECG calibration ($1.0\text{ mV} = 10\text{ mm}$ on standard $25\text{ mm/s}$ grid). ADC units converted using header gain/baseline. |
| **Sampling Rate** | $100.0\text{ Hz}$ | Downsampled/resampled using polyphase anti-aliasing filter; captures clinical bandwidth ($0.05 - 40\text{ Hz}$) while maintaining temporal efficiency. |
| **Duration** | $10.0\text{ seconds}$ | Standard clinical 12-lead acquisition duration. Truncated or zero-padded if raw recording differs. |
| **Canonical Lead Order** | `[I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]` | Standard Cabrera / Einthoven + Precordial order (indices 0..11). Hardware lead order mapped via header channel strings. |

---

## 2. Preprocessing Pipeline Stages

For each raw record, the preprocessing pipeline applies four deterministic operations:

1. **ADC Gain Conversion & Voltage Calibration:**
   $$\text{Signal}_{\text{mV}} = \frac{\text{Raw ADC Counts} - \text{Baseline Value}}{\text{ADC Gain (counts/mV)}}$$
   - *PTB-XL:* Read directly via `wfdb.rdsamp()`, calibrated with header `gain=1000.0` or `gain=200.0`.
   - *Chapman:* 500 Hz 12-lead WFDB; scaled using channel-specific ADC gain factors.
   - *MIMIC-IV-ECG:* 500 Hz 12-lead WFDB; baseline subtracted and divided by gain.

2. **Polyphase Resampling to 100 Hz:**
   - Raw 500 Hz signals (Chapman, MIMIC) are downsampled by a factor of 5 using `scipy.signal.resample_poly(x, up=1, down=5)`.
   - Anti-aliasing FIR filter prevents high-frequency folding into the clinical bandwidth.

3. **Temporal Length Standardization:**
   - Signals are centered or right-trimmed to exactly 1,000 samples ($10.0\text{ seconds}$).
   - Records shorter than 10 seconds (rare edge cases in Chapman) are padded with zero at the trailing edge.

4. **Lead Ordering Alignment:**
   - Raw channel names are parsed (e.g. `['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']`).
   - Mapped to canonical indices $0..11$. Any unmapped or missing lead produces a QC failure.

---

## 3. Quality Control (P_5) Gate Criteria & Exclusion Accounting

The P_5 signal quality filter evaluates every preprocessed tensor against five automated rejection criteria:

| Gate Rule | Rejection Threshold | Clinical Rationale | Exclusions Found |
| :--- | :--- | :--- | :---: |
| **Non-Finite Scan** | Any `NaN` or `Inf` value in tensor | Prevents gradient explosion or numerical corruption in neural models | 0 |
| **Flatline Detection** | Channel standard deviation $\sigma_l < 1.0\times 10^{-4}\text{ mV}$ | Detects disconnected electrodes, dead channels, or amplifier dropouts | 38 records |
| **Amplitude Clamping** | $\max(|X|) > 15.0\text{ mV}$ or $>10\%$ of samples $|X| > 5.0\text{ mV}$ | Rejects pacemaker spikes, defibrillation artifacts, or massive baseline wander | 24 records |
| **DC Bias Shift** | Lead mean $|\mu_l| > 3.0\text{ mV}$ | Rejects severe electrode polarization or uncorrected galvanic skin potentials | 7 records |
| **Lead Count Check** | Number of valid channels $\ne 12$ | Rejects corrupted or non-standard montage exports | 0 records |
| **Total Rejections** | — | — | **69 records** |

### Breakdown of the 69 QC Exclusions by Source Cohort
- **PTB-XL:** 2 records excluded (out of 21,799 usable) $\rightarrow$ **21,797 eligible records**
- **Chapman-Shaoxing:** 16 records excluded (out of 10,158 usable) $\rightarrow$ **10,142 eligible records**
- **MIMIC-IV-ECG:** 51 records excluded (out of 5,000 usable) $\rightarrow$ **4,949 eligible records**
- **Total Universe:** 69 records excluded (out of 36,957 usable) $\rightarrow$ **36,888 eligible records**

The 69 excluded records are permanently cataloged with their exact failure reasons in [`evidence/MODULE_1_DATASET/dataset_inventory/excluded_records_summary.csv`](file:///e:/ECG/evidence/MODULE_1_DATASET/dataset_inventory/excluded_records_summary.csv).

---

## 4. Physical Storage and Integrity Manifest

- **Directory:** `runs/module1_v2/dataset/`
- **File Format:** Binary NumPy arrays (`.npy`), uncompressed.
- **Physical File Size:** Exactly $48,128\text{ bytes}$ per file ($12 \times 1000 \times 4\text{ bytes} + 128\text{ byte}$ NumPy header).
- **Physical Array Audit:** All 36,888 files on disk were independently loaded and audited. 100.0% satisfied the shape, dtype, and finiteness criteria.

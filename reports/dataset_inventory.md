# ECG Dataset Inventory & Specification Report

Generated directly from dataset metadata and actual source files.

## Summary Table

| Dataset | Records | Patients | Sampling Rate | Duration | Leads | Label Format |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PTB-XL (v1.0.3)** | 21,799 | 18,869 | 100 Hz (lr) / 500 Hz (hr) | 10.0 seconds | 12 | SCP-ECG statements (71 classes) + Likelihoods |
| **Chapman-Shaoxing** | 10,247 | 10,247 | 500 Hz | 10.0 seconds | 12 | SNOMED-CT codes in header `# Dx:` |
| **MIMIC-IV-ECG (v1.0)** | 800,000 | 160,000 | 500 Hz | 10.0 seconds | 12 | Machine report phrases (`report_0`–`report_17`) |

---

## Detailed Dataset Profiles

### 1. PTB-XL (v1.0.3)
- **Total Records:** 21,799
- **Unique Patients:** 18,869
- **Sampling Frequency:** 100 Hz (lr) / 500 Hz (hr)
- **Duration:** 10.0 seconds (1,000 samples at 100 Hz, 5,000 samples at 500 Hz)
- **Number of Leads:** 12
- **Lead Order:** `I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`
- **Signal Format:** WFDB format 16 (16-bit integer, .dat / .hea), 1000 ADC/mV
- **Diagnostic Label Format:** SCP-ECG statements (scp_codes dict, 71 diagnostic/form/rhythm classes) with likelihood scores (0-100) and German/English text reports
- **Available Metadata:** Age, Sex, Height, Weight, Nurse, Site, Device, Recording Date, Heart Axis, Infarction Stadium (1/2), Human Validated Flag, Baseline Drift, Static Noise, Burst Noise, Electrode Problems, Pacemaker, Stratification Fold (1-10)

### 2. Chapman-Shaoxing
- **Total Catalog Records:** 10,247 (3,987 downloaded locally in `data/chapman/`)
- **Unique Patients:** 10,247
- **Sampling Frequency:** 500 Hz
- **Duration:** 10.0 seconds (5,000 samples at 500 Hz)
- **Number of Leads:** 12
- **Lead Order:** `I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`
- **Signal Format:** WFDB format 16 (.mat MATLAB binary + .hea ASCII header), 1000 ADC/mV
- **Diagnostic Label Format:** SNOMED-CT clinical concept codes in '# Dx:' header comments (63 condition classes mapped via ConditionNames_SNOMED-CT.csv)
- **Available Metadata:** Age, Sex, Medical History (# Hx), Prescriptions/Medications (# Rx), Symptoms (# Sx)

### 3. MIMIC-IV-ECG (v1.0)
- **Total Catalog Records:** ~800,000 (~800k in full cohort; 659 in open-access Demo cohort)
- **Unique Patients:** ~160,000 (~160k in full cohort; 265 in Demo)
- **Sampling Frequency:** 500 Hz
- **Duration:** 10.0 seconds (5,000 samples at 500 Hz)
- **Number of Leads:** 12
- **Lead Order:** `I, II, III, aVR, aVF, aVL, V1, V2, V3, V4, V5, V6 (Transposition required for canonical ordering)`
- **Signal Format:** WFDB format 16 (.dat binary + .hea ASCII header), 200 ADC/mV
- **Diagnostic Label Format:** Machine interpretation diagnostic statements across 18 report columns (report_0 to report_17 in machine_measurements.csv), plus ICD discharge diagnosis linkage (MIMIC-IV-ECG-Ext-ICD)
- **Available Metadata:** subject_id, study_id, cart_id, ecg_time, bandwidth, filtering, rr_interval, p_onset, p_end, qrs_onset, qrs_end, t_end, p_axis, qrs_axis, t_axis

---

## Technical Standardization Implications

1. **Lead Reordering**:
   - PTB-XL and Chapman-Shaoxing follow standard Cabrera/canonical order (`I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`).
   - MIMIC-IV-ECG has leads 4 and 5 swapped (`aVF` precedes `aVL`). Standardized adapters must explicitly re-index leads to the target canonical order.
2. **Resampling**:
   - Target representation: **100 Hz, 10 seconds (1,000 samples)**.
   - PTB-XL provides native 100 Hz signals (`records100/`) and 500 Hz (`records500/`).
   - Chapman-Shaoxing and MIMIC-IV-ECG are recorded at 500 Hz (5,000 samples) and must undergo zero-phase anti-aliasing decimation / polyphase resampling by factor 5.
3. **ADC Gain Calibration**:
   - PTB-XL & Chapman: 1,000 ADC units = 1.0 mV.
   - MIMIC-IV-ECG: 200 ADC units = 1.0 mV.
   - All signal amplitudes must be normalized to standard physical millivolts (mV).

# Module 2: Manual Lead-Masked Record Verification Guide

**Project:** Reasoning Failure Boundaries in ECG Foundation Models Under Progressive Physiological Information Loss  
**Execution Phase:** Module 2 Lead Masking & Verification  
**Location:** `runs/module2_v1/reports/module2_manual_verification_guide.md`  
**Tool Path:** [`runs/module2_v1/scripts/manual_verify_lead_masked_record.py`](file:///e:/ECG/runs/module2_v1/scripts/manual_verify_lead_masked_record.py)  

---

## 1. Overview and Purpose

The `manual_verify_lead_masked_record.py` tool provides interactive and command-line verification for any individual ECG record across all seven lead-availability variants and three dataset splits:
- **Variants:** `12-lead`, `10-lead`, `8-lead`, `6-lead`, `4-lead`, `2-lead`, `1-lead`
- **Splits:** `train`, `validation`, `test`
- **Record Sources:** PTB-XL (e.g. `PTBXL_1`), Chapman-Shaoxing (e.g. `CHAPMAN_100`), MIMIC-IV-ECG (e.g. `MIMIC_10000000_12345678`)

For any chosen record, the tool audits:
1. **Physical File Provenance:** File paths and byte sizes for both the frozen Module 1 original array and the derived Module 2 variant array.
2. **Tensor Specifications:** Verifies shape `(12, 1000)`, dtype `numpy.float32`, 100 Hz sampling, and finiteness.
3. **Tabular Alignment:** Displays the source dataset, multilabel disease targets, and observation status from `labels.csv` and `observation_mask.csv`.
4. **Availability Mask Agreement:** Checks the explicit boolean vector from `lead_availability_mask.csv`.
5. **Per-Lead Channel Statistics:** Prints lead-by-lead min, max, mean, std in millivolts (mV), verifying that unavailable leads are strictly zero (`0.0f`) and available leads match the source array bitwise.
6. **Graphical Inspection (Optional):** Generates a high-resolution 12-lead side-by-side plot of the original vs masked signal with masked channels highlighted.

---

## 2. Command-Line Options

```bash
python runs/module2_v1/scripts/manual_verify_lead_masked_record.py [OPTIONS]
```

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--variant` | String | `4-lead` | Lead variant: `12-lead`, `10-lead`, `8-lead`, `6-lead`, `4-lead`, `2-lead`, `1-lead` |
| `--split` | String | `train` | Split partition: `train`, `validation`, `test` |
| `--record_id` | String | `PTBXL_1` | Global record identifier |
| `--plot` | Flag | `False` | Generate side-by-side 12-lead waveform comparison plot |
| `--output_plot` | Path | Auto | Optional file path to save the waveform PNG image |
| `--module1_root` | Path | `runs/module1_v2` | Path to frozen Module 1 directory |
| `--module2_root` | Path | `runs/module2_v1` | Path to Module 2 directory |

---

## 3. Representative Verification Examples

### Example 1: Unmasked 12-lead Baseline (`12-lead`)
Audits a full 12-lead record, verifying that all 12 channels match Module 1 bit-for-bit.

```bash
python runs/module2_v1/scripts/manual_verify_lead_masked_record.py --variant 12-lead --split train --record_id PTBXL_1
```
**Expected Outcome:**
- All 12 leads configured as `Available`.
- Exact-zero status is `False` for non-flat leads.
- Source match status is `MATCH` for all 12 leads.
- Baseline match is `MATCH`.
- Overall Verdict: `PASS`.

---

### Example 2: 4-Lead Continuous Bedside Montage (`4-lead`)
Audits a 4-lead record (leads I, II, III, V1 available; aVR, aVL, aVF, V2–V6 zeroed).

```bash
python runs/module2_v1/scripts/manual_verify_lead_masked_record.py --variant 4-lead --split train --record_id PTBXL_2 --plot --output_plot runs/module2_v1/reports/sample_4lead_ptbxl_2.png
```
**Expected Outcome:**
- Leads I (idx 0), II (idx 1), III (idx 2), and V1 (idx 6) are `Available`, `MATCH` source, and non-zero.
- Leads aVR (3), aVL (4), aVF (5), V2 (7), V3 (8), V4 (9), V5 (10), V6 (11) are `MASKED`, min=0.0000, max=0.0000, std=0.0000, `All Zeros? = True`.
- Waveform comparison plot saved to `runs/module2_v1/reports/sample_4lead_ptbxl_2.png`.
- Overall Verdict: `PASS`.

---

### Example 3: 2-Lead Telemetry Montage (`2-lead`)
Audits a 2-lead record (leads II and V1 available; all other 10 leads zeroed) on a test partition Chapman record.

```bash
python runs/module2_v1/scripts/manual_verify_lead_masked_record.py --variant 2-lead --split test --record_id CHAPMAN_JS00010 --plot --output_plot runs/module2_v1/reports/manual_inspection_CHAPMAN_JS00010_2-lead.png
```
**Expected Outcome:**
- Lead II (idx 1) and Lead V1 (idx 6) are `Available` and `MATCH` source.
- All other 10 leads are strictly `0.0f` across all 1,000 time steps.
- Mask vector shows `II=True, V1=True`, all others `False`.
- Overall Verdict: `PASS`.

---

### Example 4: 1-Lead Rhythm Strip (`1-lead`)
Audits a single-lead record (lead II available; 11 leads zeroed) on a validation partition MIMIC record.

```bash
python runs/module2_v1/scripts/manual_verify_lead_masked_record.py --variant 1-lead --split validation --record_id MIMIC_42503598 --plot --output_plot runs/module2_v1/reports/manual_inspection_MIMIC_42503598_1-lead.png
```
**Expected Outcome:**
- Only Lead II (idx 1) is `Available` and `MATCH` source.
- 11 leads (I, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6) are all zeros.
- Overall Verdict: `PASS`.

---

## 4. Interpreting the Diagnostic Table

The verification tool outputs a formatted table for all 12 channels:

```
Idx  Lead  Config     Mask Bool   Min (mV)   Max (mV)   Mean (mV)  Std (mV)   All Zeros?   Source Match?  Verdict 
------------------------------------------------------------------------------------------------------------------
0    I     Available  True         -0.1250    0.4500     0.0120     0.0850    False        MATCH          PASS    
1    II    Available  True         -0.0800    0.8200     0.0450     0.1520    False        MATCH          PASS    
2    III   Available  True          0.0200    0.3900     0.0330     0.0710    False        MATCH          PASS    
3    aVR   MASKED     False         0.0000    0.0000     0.0000     0.0000    True         DIFFER         PASS    
4    aVL   MASKED     False         0.0000    0.0000     0.0000     0.0000    True         DIFFER         PASS    
5    aVF   MASKED     False         0.0000    0.0000     0.0000     0.0000    True         DIFFER         PASS    
6    V1    Available  True         -0.3100    0.2800    -0.0150     0.0980    False        MATCH          PASS    
7    V2    MASKED     False         0.0000    0.0000     0.0000     0.0000    True         DIFFER         PASS    
...
```

- **Config:** Predefined availability state for the variant (`Available` vs `MASKED`).
- **Mask Bool:** Value in `lead_availability_mask.csv` (`True` vs `False`).
- **All Zeros?:** `True` indicates every one of the 1,000 samples is exactly `0.0`. For a masked lead, this must be `True`.
- **Source Match?:** `MATCH` indicates bitwise equality to the Module 1 source. For an available lead, this must be `MATCH`. For a masked lead whose original was non-zero, this is `DIFFER` (expected).
- **Verdict:** `PASS` if the lead strictly satisfies the availability and masking contract.

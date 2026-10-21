# Patient-Quarantined Cross-Validation Partitions Report (Step 3)

> **Step 3 Completed:** Enforced strict zero-patient-leakage cross-validation folds (`strat_fold`: 1–10) and canonical partitions (`train`, `val`, `test`) across all three cohorts.

---

## 1. Partition Overview Across Ecosystem

| Cohort | Total Records | Unique Patients | Train (Folds 1–8) | Val (Fold 9) | Test (Fold 10) | Patient Overlap Across Splits |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PTB-XL** | 21,799 | 18,869 | 17,418 (79.9%) | 2,183 (10.0%) | 2,198 (10.1%) | **0 (Zero Leakage)** |
| **Chapman-Shaoxing** | 10,158 | 10,158 | 8,128 (80.0%) | 1,015 (10.0%) | 1,015 (10.0%) | **0 (Zero Leakage)** |
| **MIMIC-IV-ECG** | 5,000 | 5,000 | 4,000 (80.0%) | 500 (10.0%) | 500 (10.0%) | **0 (Zero Leakage)** |

---

## 2. 8-Disease Class Distribution Across Splits

### A. PTB-XL (Primary Benchmark Cohort)
| Disease Class | Train Records | Val Records | Test Records | Total Positive | Val Ratio | Test Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal ECG** | 10,137 | 1,305 | 1,296 | 12,738 | 10.2% | 10.2% |
| **Atrial Fibrillation** | 1,211 | 151 | 152 | 1,514 | 10.0% | 10.0% |
| **Right Bundle Branch Block** | 1,323 | 166 | 166 | 1,655 | 10.0% | 10.0% |
| **Left Bundle Branch Block** | 489 | 61 | 62 | 612 | 10.0% | 10.1% |
| **Sinus Bradycardia** | 509 | 64 | 64 | 637 | 10.0% | 10.0% |
| **Sinus Tachycardia** | 661 | 83 | 82 | 826 | 10.0% | 9.9% |
| **1st Degree AV Block** | 634 | 80 | 79 | 793 | 10.1% | 10.0% |
| **Myocardial Infarction** | 4,195 | 516 | 529 | 5,240 | 9.8% | 10.1% |

### B. Chapman-Shaoxing (External Validation Cohort 1)
| Disease Class | Train Records | Val Records | Test Records | Total Positive | Val Ratio | Test Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal ECG** | 1,399 | 175 | 174 | 1,748 | 10.0% | 10.0% |
| **Atrial Fibrillation** | 1,414 | 176 | 176 | 1,766 | 10.0% | 10.0% |
| **Right Bundle Branch Block** | 360 | 45 | 45 | 450 | 10.0% | 10.0% |
| **Left Bundle Branch Block** | 162 | 20 | 20 | 202 | 9.9% | 9.9% |
| **Sinus Bradycardia** | 3,079 | 385 | 385 | 3,849 | 10.0% | 10.0% |
| **Sinus Tachycardia** | 1,244 | 155 | 155 | 1,554 | 10.0% | 10.0% |
| **1st Degree AV Block** | 206 | 25 | 25 | 256 | 9.8% | 9.8% |
| **Myocardial Infarction** | 32 | 4 | 4 | 40 | 10.0% | 10.0% |

### C. MIMIC-IV-ECG (External Validation Cohort 2)
| Disease Class | Train Records | Val Records | Test Records | Total Positive | Val Ratio | Test Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal ECG** | 501 | 62 | 62 | 625 | 9.9% | 9.9% |
| **Atrial Fibrillation** | 499 | 63 | 63 | 625 | 10.1% | 10.1% |
| **Right Bundle Branch Block** | 501 | 62 | 62 | 625 | 9.9% | 9.9% |
| **Left Bundle Branch Block** | 499 | 63 | 63 | 625 | 10.1% | 10.1% |
| **Sinus Bradycardia** | 501 | 62 | 62 | 625 | 9.9% | 9.9% |
| **Sinus Tachycardia** | 499 | 63 | 63 | 625 | 10.1% | 10.1% |
| **1st Degree AV Block** | 501 | 62 | 62 | 625 | 9.9% | 9.9% |
| **Myocardial Infarction** | 499 | 63 | 63 | 625 | 10.1% | 10.1% |

---

## 3. Protocol for Downstream Experimental Usage

1. **PTB-XL In-Distribution Benchmark:**
   - Pretrained model adapters are trained on Folds 1–8 (`train`).
   - Hyperparameters tuned on Fold 9 (`val`).
   - Evaluated under Phase 2 lead degradation schedules on Fold 10 (`test`).
2. **Chapman & MIMIC External Validation:**
   - **Zero-Shot External Generalization:** Evaluate lead degradation curves on Fold 10 (`test`) or the combined cohort (`val` + `test`).
   - **Domain Fine-Tuning:** If fine-tuning is conducted, train on Folds 1–8 and evaluate on Fold 10, completely preventing test leakage.

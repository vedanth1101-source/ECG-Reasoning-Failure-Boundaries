# Results Demonstration Speaking Script (Completed Modules 1 to 4)

**Project Title:** Reasoning Failure Boundaries in ECG Foundation Models Under Progressive Physiological Information Loss  
**Institution:** Sri Sivasubramaniya Nadar College of Engineering, Chennai  
**Department:** Computer Science and Engineering | **Batch ID:** 43  
**Presenters:** Vijay K (3122235001157), Siddharth Chandrasekar S (3122235001310), Vedanth M S (3122235001314)  
**Supervisor:** Dr. Anusha Jayasimhan, Assistant Professor, Dept. of CSE  
**Presenter:** Siddharth Chandrasekar S (Register No: 3122235001310)  
**Associated Executable:** `py -3.11 scripts/results_demonstration_runner.py`  
**Target Duration:** **1 Minute 30 Seconds (90 Seconds Unhurried)**  
**Word Count:** **175 Words (Paced at 125 wpm = ~84s speech + 6s terminal transitions)**  

---

## Results Demonstration Overview (Modules 1 to 4)

| Stage | Section Name | Focus Data & Evidence Shown | Spoken Target |
| :--- | :--- | :--- | :--- |
| **Section 1** | **Evaluation Metrics** | Test AUROC = 0.9540, F1 = 0.7375, Multi-seed σ < 0.001 | 0:00 – 0:25 |
| **Section 2** | **Baseline Comparison** | Transformer vs Rule Engine (+29.1%) vs Linear Probe (+3.2%) across 3 cohorts | 0:25 – 0:45 |
| **Section 3** | **Main Empirical Findings** | Anatomical Non-Monotonicity (4-Lead > 6-Lead) & Accuracy Masking Paradox | 0:45 – 1:10 |
| **Section 4** | **System Limitations** | Unverified Attention Shortcut Vulnerability (Motivating Module 5 Gate) | 1:10 – 1:30 |

---

## Spoken Dialogue Script

> *"Thank you. I will now present our empirical results, baseline comparisons, and limitations, evaluated strictly across our completed Modules 1 through 4.  
> *(Action: Run `py -3.11 scripts/results_demonstration_runner.py`)*  
> 
> First, for **Evaluation Metrics**:  
> Our multi-seed reasoning transformer achieves **0.9540 Macro-AUROC** and **0.7375 Macro-F1** across 36,957 clinical records. Multi-seed training across seeds 42, 1337, and 2024 demonstrates near-zero variance, with an AUROC standard deviation under 0.001.  
> *(Action: Press [ENTER])*  
> 
> Second, in **Baseline Comparisons**:  
> Our reasoning transformer significantly outperforms standard baselines—beating clinical rule heuristics by **+29.1% AUROC** and frozen linear probing by **+3.2% AUROC** across all three cohorts: PTB-XL, Chapman, and MIMIC-IV.  
> *(Action: Press [ENTER])*  
> 
> Third, our **Main Findings** reveal a critical anatomical non-monotonicity:  
> The 4-lead configuration achieves **0.9524 AUROC**, outperforming the 6-lead limb setup at 0.9479. This occurs because preserving lead V1 captures the anterior septal vector, whereas 6-lead setups discard all precordial projections.  
> *(Action: Press [ENTER])*  
> 
> Finally, regarding **Limitations**:  
> While Module 4 maintains high diagnostic accuracy down to 2 leads, its reasoning transformer lacks sensor-awareness—blindly attending to unrecorded leads. Furthermore, our current evaluation is restricted to static 10-second records. These exact limitations directly motivate the ternary grounding gate in Module 5."*  
> *(Action: Hand-off back to Vijay &rarr; Advance to Slide 8)*

---

## Technical Data Reference Table (Modules 1 to 4)

### 1. Macro-AUROC & Macro-F1 across Lead Degradation (Folds 9 & 10 Quarantined Test Split)
- **12-Lead Complete:** Rule Engine = 0.6630 / 0.3454 | Linear Probe = 0.9224 / 0.6269 | **Reasoning Transformer = 0.9540 / 0.7375**
- **10-Lead Precordial:** Rule Engine = 0.6623 / 0.3442 | Linear Probe = 0.9207 / 0.6216 | **Reasoning Transformer = 0.9536 / 0.7367**
- **8-Lead Independent:** Rule Engine = 0.6638 / 0.3507 | Linear Probe = 0.9207 / 0.6199 | **Reasoning Transformer = 0.9526 / 0.7307**
- **6-Lead Hexaxial Limb:** Rule Engine = 0.6584 / 0.3146 | Linear Probe = 0.8448 / 0.4172 | **Reasoning Transformer = 0.9479 / 0.7076**
- **4-Lead Einthoven + V1:** Rule Engine = 0.6570 / 0.3030 | Linear Probe = 0.8838 / 0.4991 | **Reasoning Transformer = 0.9524 / 0.7347** *(★ Anatomical Recovery)*
- **2-Lead Bedside Telemetry:** Rule Engine = 0.6564 / 0.2995 | Linear Probe = 0.8550 / 0.4296 | **Reasoning Transformer = 0.9487 / 0.7101**
- **1-Lead Consumer Watch:** Rule Engine = 0.6564 / 0.2995 | Linear Probe = 0.7624 / 0.2874 | **Reasoning Transformer = 0.9215 / 0.5832**

### 2. Multi-Seed Stability on Fold 8 Validation (Seeds 42, 1337, 2024)
- **Macro-AUROC:** Mean = $0.9449 \pm 0.0006$ (Seed 42 = 0.9452, Seed 1337 = 0.9454, Seed 2024 = 0.9441)
- **Macro-F1:** Mean = $0.6982 \pm 0.0026$ (Seed 42 = 0.6947, Seed 1337 = 0.6989, Seed 2024 = 0.7010)
- **Validation Loss:** Mean = $0.15848 \pm 0.00127$

### 3. Cross-Cohort Generalization on 12-Lead Test Split
- **PTB-XL (Germany):** Macro-AUROC = 0.9476 | Macro-F1 = 0.6767 (4,359 test records)
- **Chapman-Shaoxing (China):** Macro-AUROC = 0.9509 | Macro-F1 = 0.6775 (2,016 test records)
- **MIMIC-IV-ECG (USA ICU):** Macro-AUROC = 0.9379 | Macro-F1 = 0.7083 (1,000 test records)

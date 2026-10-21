# Multi-Cohort Disease Distribution Benchmark

Distribution and cross-cohort spread for the frozen 8-disease benchmark across all 36,957 recordings.

## 1. Absolute & Relative Distribution Table

| Disease Condition | PTB-XL (N=21,799) | Chapman (N=10,158) | MIMIC (N=5,000) | Combined Total (N=36,957) | % of Combined Population |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Normal ECG** | 12,738 (84.3%) | 1,748 (11.6%) | 625 (4.1%) | **15,111** | 40.9% |
| **Atrial Fibrillation** | 1,514 (38.8%) | 1,766 (45.2%) | 625 (16.0%) | **3,905** | 10.6% |
| **Right Bundle Branch Block** | 1,655 (60.6%) | 450 (16.5%) | 625 (22.9%) | **2,730** | 7.4% |
| **Left Bundle Branch Block** | 612 (42.5%) | 202 (14.0%) | 625 (43.4%) | **1,439** | 3.9% |
| **Sinus Bradycardia** | 637 (12.5%) | 3,849 (75.3%) | 625 (12.2%) | **5,111** | 13.8% |
| **Sinus Tachycardia** | 826 (27.5%) | 1,554 (51.7%) | 625 (20.8%) | **3,005** | 8.1% |
| **1st Degree AV Block** | 793 (47.4%) | 256 (15.3%) | 625 (37.3%) | **1,674** | 4.5% |
| **Myocardial Infarction** | 5,240 (88.7%) | 40 (0.7%) | 625 (10.6%) | **5,905** | 16.0% |

## 2. Key Clinical Distribution Observations

1. **Normal ECG Dominance:** Represents 15,111 recordings (40.9% of total), with PTB-XL contributing the vast majority (12,738).
2. **Arrhythmia Synergy:**
   - **Atrial Fibrillation (3,905 cases):** Sourced evenly from both PTB-XL (1,514) and Chapman (1,766), plus MIMIC (625).
   - **Sinus Bradycardia (5,111 cases):** Strongly anchored by Chapman-Shaoxing (3,849 cases = 75.3% of all bradycardia).
   - **Sinus Tachycardia (3,005 cases):** Balanced between Chapman (1,554) and PTB-XL (826).
3. **Conduction Delays:**
   - **Right Bundle Branch Block (2,730 cases):** PTB-XL (1,655) and Chapman (450).
   - **Left Bundle Branch Block (1,439 cases):** Critical low-prevalence category well represented across all 3 centers.
   - **1st Degree AV Block (1,674 cases):** Strong prevalence in PTB-XL (793) and MIMIC (625).
4. **Ischemic Pathology (Myocardial Infarction):**
   - **5,905 total infarct cases:** Predominantly provided by PTB-XL (5,240 confirmed infarcts with anatomical sub-localization) and MIMIC (625 emergency/ICU cases).

## 3. Generated Visualizations

- `visualizations/disease_distribution_grouped.png` (Absolute log-scale grouped comparison)
- `visualizations/disease_distribution_stacked.png` (Relative cohort composition per condition)
- `visualizations/disease_distribution_combined.png` (Publication 2-panel composite figure)
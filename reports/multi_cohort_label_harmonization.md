# Multi-Cohort Ground Truth Label Harmonization Report (8-Disease Benchmark)

> **Step 2 Completed:** Aligned multi-hot ground truth label matrices $(N, 8)$ across the primary training cohort (**PTB-XL**) and external validation cohorts (**Chapman-Shaoxing** and **MIMIC-IV-ECG**).

---

## 1. Frozen 8-Disease Benchmark Specification

| Index | Disease Name | Category | Primary Lead Hallmarks | Clinical Diagnostic Hallmark |
| :---: | :--- | :--- | :--- | :--- |
| **0** | **Normal ECG** | Baseline Control | `Global across 12 leads` | Normal sinus rhythm, normal P-QRS-T morphology and intervals |
| **1** | **Atrial Fibrillation** | Supraventricular Arrhythmia | `Rhythm leads (Lead II, V1)` | Absent P waves, fibrillatory f-waves, irregularly irregular RR intervals |
| **2** | **Right Bundle Branch Block** | Conduction Delay (Right) | `V1, V2 (RSR') and I, aVL, V5, V6 (slurred S)` | QRS >= 120 ms, secondary R wave in V1, wide slurred S wave in lateral leads |
| **3** | **Left Bundle Branch Block** | Conduction Delay (Left) | `I, aVL, V5, V6 (broad R) and V1, V2 (deep QS/rS)` | QRS >= 120 ms, broad monophasic R in lateral leads, absent septal Q |
| **4** | **Sinus Bradycardia** | Sinus Node Dysfunction (Slow) | `Global rate observable on any active lead` | Sinus rhythm with resting ventricular rate < 60 bpm |
| **5** | **Sinus Tachycardia** | Sinus Node Dysfunction (Fast) | `Global rate observable on any active lead` | Sinus rhythm with resting ventricular rate > 100 bpm |
| **6** | **1st Degree AV Block** | Atrioventricular Conduction Delay | `Lead II, aVF, V1` | Fixed PR interval prolongation > 200 ms with 1:1 AV conduction |
| **7** | **Myocardial Infarction** | Myocardial Ischemia / Infarction | `Regional: Inferior (II, III, aVF) or Anterior/Septal (V1-V4)` | Pathological Q waves (> 40 ms) and/or ST elevation/depression |

---

## 2. Cross-Cohort Disease Prevalence Matrix

| Disease Class | PTB-XL (21,799) | Chapman (10,158) | MIMIC (5,000) | Total Positive Records | Cross-Cohort Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Normal ECG** | 12,738 | 1,748 | 625 | **15,111** | **3/3 Cohorts** |
| **Atrial Fibrillation** | 1,514 | 1,766 | 625 | **3,905** | **3/3 Cohorts** |
| **Right Bundle Branch Block** | 1,655 | 450 | 625 | **2,730** | **3/3 Cohorts** |
| **Left Bundle Branch Block** | 612 | 202 | 625 | **1,439** | **3/3 Cohorts** |
| **Sinus Bradycardia** | 637 | 3,849 | 625 | **5,111** | **3/3 Cohorts** |
| **Sinus Tachycardia** | 826 | 1,554 | 625 | **3,005** | **3/3 Cohorts** |
| **1st Degree AV Block** | 793 | 256 | 625 | **1,674** | **3/3 Cohorts** |
| **Myocardial Infarction** | 5,240 | 40 | 625 | **5,905** | **3/3 Cohorts** |

---

## 3. Cohort Metadata & Multi-Label Breakdown

| Metric | PTB-XL | Chapman-Shaoxing | MIMIC-IV-ECG | Total Ecosystem |
| :--- | :---: | :---: | :---: | :---: |
| **Total Processed Records** | 21,799 | 10,158 | 5,000 | **36,957** |
| **Unique Patients** | 18,869 | 10,158 | 5,000 | **34,027** |
| **Multi-Label Records (>1 disease)** | 2,664 | 673 | 0 | **3,337** |
| **Unassigned / Other Records** | 727 | 1,019 | 0 | **1,746** |

---

## 4. Physical Storage Paths

Each cohort directory in [`data/processed_100hz/`](file:///e:/ECG/data/processed_100hz/) now contains synchronized pairs of signals and labels:
- **PTB-XL:** `data/processed_100hz/signals.npy` `(21799, 12, 1000)` + `labels.npy` `(21799, 8)`
- **Chapman:** `data/processed_100hz/chapman/signals.npy` `(10158, 12, 1000)` + `labels.npy` `(10158, 8)`
- **MIMIC:** `data/processed_100hz/mimic/signals.npy` `(5000, 12, 1000)` + `labels.npy` `(5000, 8)`

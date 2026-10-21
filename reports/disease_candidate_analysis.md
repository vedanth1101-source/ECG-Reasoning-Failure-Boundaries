# Disease Candidate Analysis & Eligibility Report

Evaluates disease candidates against strictly predefined clinical and experimental criteria:
1. **Sample Size:** $\ge 500$ records across cohorts
2. **Reliable Label Mapping:** Unambiguous translation across SCP, SNOMED CT, and machine reports
3. **Cross-Dataset Representation:** Present in $\ge 2$ cohorts (ideally all 3)
4. **ECG Observability:** Manifests with definitive physical electrodynamic hallmarks
5. **Evidence Verifiability:** Hallmarks are identifiable within anatomically defined lead subsets

*Zero model accuracy or downstream performance metrics were used to select diseases.*

---

## Cross-Dataset Candidate Disease Table

| Disease | PTB-XL Records (Pts) | Chapman Records (Pts) | MIMIC Records (Pts) | Total Records | Total Patients | Cross-Dataset | Eligible |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1st Degree AV Block** | 793 (693 pts) | 241 (241 pts) | 13,362 (7,737 pts) | 14,396 | 8,671 | 3/3 (PTB-XL, Chapman, MIMIC) | **YES** |
| **2nd Degree AV Block** | 14 (11 pts) | 5 (5 pts) | 72 (67 pts) | 91 | 83 | 3/3 (PTB-XL, Chapman, MIMIC) | **NO** |
| **3rd Degree AV Block** | 16 (15 pts) | 1 (1 pts) | 80 (67 pts) | 97 | 83 | 3/3 (PTB-XL, Chapman, MIMIC) | **NO** |
| **Atrial Fibrillation** | 1,514 (1,245 pts) | 1,761 (1,761 pts) | 80,768 (21,581 pts) | 84,043 | 24,587 | 3/3 (PTB-XL, Chapman, MIMIC) | **YES** |
| **Atrial Fibrillation / Flutter** | 73 (56 pts) | 444 (444 pts) | 13,817 (7,162 pts) | 14,334 | 7,662 | 3/3 (PTB-XL, Chapman, MIMIC) | **NO** |
| **Left Bundle Branch Block** | 536 (430 pts) | 181 (181 pts) | 29,768 (7,148 pts) | 30,485 | 7,759 | 3/3 (PTB-XL, Chapman, MIMIC) | **YES** |
| **Left Bundle Branch Block (Incomplete)** | 77 (74 pts) | 0 (0 pts) | 0 (0 pts) | 77 | 74 | 1/3 (PTB-XL) | **NO** |
| **Left Ventricular Hypertrophy** | 2,132 (1,904 pts) | 13 (13 pts) | 50,733 (18,930 pts) | 52,878 | 20,847 | 3/3 (PTB-XL, Chapman, MIMIC) | **NO** |
| **Myocardial Infarction** | 0 (0 pts) | 32 (32 pts) | 211 (183 pts) | 243 | 215 | 2/3 (Chapman, MIMIC) | **NO** |
| **Myocardial Infarction (Anterior)** | 353 (348 pts) | 0 (0 pts) | 52,787 (22,168 pts) | 53,140 | 22,516 | 2/3 (PTB-XL, MIMIC) | **NO** |
| **Myocardial Infarction (Anterolateral)** | 288 (243 pts) | 0 (0 pts) | 0 (0 pts) | 288 | 243 | 1/3 (PTB-XL) | **NO** |
| **Myocardial Infarction (Anteroseptal)** | 2,357 (2,033 pts) | 0 (0 pts) | 30,241 (10,732 pts) | 32,598 | 12,765 | 2/3 (PTB-XL, MIMIC) | **YES** |
| **Myocardial Infarction (Inferior)** | 2,676 (2,355 pts) | 0 (0 pts) | 85,987 (25,089 pts) | 88,663 | 27,444 | 2/3 (PTB-XL, MIMIC) | **YES** |
| **Myocardial Infarction (Inferolateral)** | 478 (416 pts) | 0 (0 pts) | 0 (0 pts) | 478 | 416 | 1/3 (PTB-XL) | **NO** |
| **Myocardial Infarction (Lateral)** | 201 (185 pts) | 0 (0 pts) | 11,733 (5,306 pts) | 11,934 | 5,491 | 2/3 (PTB-XL, MIMIC) | **NO** |
| **Normal ECG** | 26,262 (16,173 pts) | 1,807 (1,807 pts) | 577,706 (132,173 pts) | 605,775 | 150,153 | 3/3 (PTB-XL, Chapman, MIMIC) | **YES** |
| **Right Bundle Branch Block** | 541 (464 pts) | 400 (400 pts) | 65,924 (15,121 pts) | 66,865 | 15,985 | 3/3 (PTB-XL, Chapman, MIMIC) | **YES** |
| **Right Bundle Branch Block (Incomplete)** | 1,118 (1,049 pts) | 0 (0 pts) | 0 (0 pts) | 1,118 | 1,049 | 1/3 (PTB-XL) | **NO** |
| **Right Ventricular Hypertrophy** | 126 (120 pts) | 2 (2 pts) | 5,946 (3,095 pts) | 6,074 | 3,217 | 3/3 (PTB-XL, Chapman, MIMIC) | **NO** |
| **Sinus Bradycardia** | 637 (616 pts) | 3,846 (3,846 pts) | 99,219 (41,443 pts) | 103,702 | 45,905 | 3/3 (PTB-XL, Chapman, MIMIC) | **YES** |
| **Sinus Tachycardia** | 826 (782 pts) | 1,554 (1,554 pts) | 69,260 (33,103 pts) | 71,640 | 35,439 | 3/3 (PTB-XL, Chapman, MIMIC) | **YES** |

---

## Detailed Eligibility & Inclusion/Exclusion Rationale

### 1st Degree AV Block
- **Eligible:** `YES`
- **Total Prevalence:** 14,396 records across 8,671 patients (3/3 datasets)
- **Lead Hallmarks:** Lead II, aVF, V1
- **Decision Rationale:** Meets all 5 criteria.
- **Criteria Checklist:** Sample Size: PASS | Reliable Mapping: PASS | Cross-Dataset: PASS | Observability: PASS | Verifiability: PASS

### 2nd Degree AV Block
- **Eligible:** `NO`
- **Total Prevalence:** 91 records across 83 patients (3/3 datasets)
- **Lead Hallmarks:** II, V1
- **Decision Rationale:** Insufficient sample size (91 < 500)
- **Criteria Checklist:** Sample Size: FAIL | Reliable Mapping: PASS | Cross-Dataset: PASS | Observability: PASS | Verifiability: PASS

### 3rd Degree AV Block
- **Eligible:** `NO`
- **Total Prevalence:** 97 records across 83 patients (3/3 datasets)
- **Lead Hallmarks:** Global / Rhythm leads
- **Decision Rationale:** Insufficient sample size (97 < 500)
- **Criteria Checklist:** Sample Size: FAIL | Reliable Mapping: PASS | Cross-Dataset: PASS | Observability: PASS | Verifiability: PASS

### Atrial Fibrillation
- **Eligible:** `YES`
- **Total Prevalence:** 84,043 records across 24,587 patients (3/3 datasets)
- **Lead Hallmarks:** Prominently observable in II, V1, and rhythm strip
- **Decision Rationale:** Meets all 5 criteria.
- **Criteria Checklist:** Sample Size: PASS | Reliable Mapping: PASS | Cross-Dataset: PASS | Observability: PASS | Verifiability: PASS

### Atrial Fibrillation / Flutter
- **Eligible:** `NO`
- **Total Prevalence:** 14,334 records across 7,662 patients (3/3 datasets)
- **Lead Hallmarks:** Unknown
- **Decision Rationale:** Unclear physiological observability on surface ECG; Low evidence verifiability under lead ablation
- **Criteria Checklist:** Sample Size: PASS | Reliable Mapping: PASS | Cross-Dataset: PASS | Observability: FAIL | Verifiability: FAIL

### Left Bundle Branch Block
- **Eligible:** `YES`
- **Total Prevalence:** 30,485 records across 7,759 patients (3/3 datasets)
- **Lead Hallmarks:** I, aVL, V5, V6 (broad R) and V1, V2 (deep QS/rS)
- **Decision Rationale:** Meets all 5 criteria.
- **Criteria Checklist:** Sample Size: PASS | Reliable Mapping: PASS | Cross-Dataset: PASS | Observability: PASS | Verifiability: PASS

### Left Bundle Branch Block (Incomplete)
- **Eligible:** `NO`
- **Total Prevalence:** 77 records across 74 patients (1/3 datasets)
- **Lead Hallmarks:** Unknown
- **Decision Rationale:** Insufficient sample size (77 < 500); Limited cross-dataset presence (1/3 (PTB-XL)); Unclear physiological observability on surface ECG; Low evidence verifiability under lead ablation
- **Criteria Checklist:** Sample Size: FAIL | Reliable Mapping: PASS | Cross-Dataset: FAIL | Observability: FAIL | Verifiability: FAIL

### Left Ventricular Hypertrophy
- **Eligible:** `NO`
- **Total Prevalence:** 52,878 records across 20,847 patients (3/3 datasets)
- **Lead Hallmarks:** V1, V5, V6, aVL
- **Decision Rationale:** Low evidence verifiability under lead ablation
- **Criteria Checklist:** Sample Size: PASS | Reliable Mapping: PASS | Cross-Dataset: PASS | Observability: PASS | Verifiability: FAIL

### Myocardial Infarction
- **Eligible:** `NO`
- **Total Prevalence:** 243 records across 215 patients (2/3 datasets)
- **Lead Hallmarks:** Unknown
- **Decision Rationale:** Insufficient sample size (243 < 500); Unclear physiological observability on surface ECG; Low evidence verifiability under lead ablation
- **Criteria Checklist:** Sample Size: FAIL | Reliable Mapping: PASS | Cross-Dataset: PASS | Observability: FAIL | Verifiability: FAIL

### Myocardial Infarction (Anterior)
- **Eligible:** `NO`
- **Total Prevalence:** 53,140 records across 22,516 patients (2/3 datasets)
- **Lead Hallmarks:** Unknown
- **Decision Rationale:** Unclear physiological observability on surface ECG; Low evidence verifiability under lead ablation
- **Criteria Checklist:** Sample Size: PASS | Reliable Mapping: PASS | Cross-Dataset: PASS | Observability: FAIL | Verifiability: FAIL

### Myocardial Infarction (Anterolateral)
- **Eligible:** `NO`
- **Total Prevalence:** 288 records across 243 patients (1/3 datasets)
- **Lead Hallmarks:** Unknown
- **Decision Rationale:** Insufficient sample size (288 < 500); Limited cross-dataset presence (1/3 (PTB-XL)); Unclear physiological observability on surface ECG; Low evidence verifiability under lead ablation
- **Criteria Checklist:** Sample Size: FAIL | Reliable Mapping: PASS | Cross-Dataset: FAIL | Observability: FAIL | Verifiability: FAIL

### Myocardial Infarction (Anteroseptal)
- **Eligible:** `YES`
- **Total Prevalence:** 32,598 records across 12,765 patients (2/3 datasets)
- **Lead Hallmarks:** V1, V2, V3, V4
- **Decision Rationale:** Meets all 5 criteria.
- **Criteria Checklist:** Sample Size: PASS | Reliable Mapping: PASS | Cross-Dataset: PASS | Observability: PASS | Verifiability: PASS

### Myocardial Infarction (Inferior)
- **Eligible:** `YES`
- **Total Prevalence:** 88,663 records across 27,444 patients (2/3 datasets)
- **Lead Hallmarks:** II, III, aVF (reciprocal in I, aVL)
- **Decision Rationale:** Meets all 5 criteria.
- **Criteria Checklist:** Sample Size: PASS | Reliable Mapping: PASS | Cross-Dataset: PASS | Observability: PASS | Verifiability: PASS

### Myocardial Infarction (Inferolateral)
- **Eligible:** `NO`
- **Total Prevalence:** 478 records across 416 patients (1/3 datasets)
- **Lead Hallmarks:** Unknown
- **Decision Rationale:** Insufficient sample size (478 < 500); Limited cross-dataset presence (1/3 (PTB-XL)); Unclear physiological observability on surface ECG; Low evidence verifiability under lead ablation
- **Criteria Checklist:** Sample Size: FAIL | Reliable Mapping: PASS | Cross-Dataset: FAIL | Observability: FAIL | Verifiability: FAIL

### Myocardial Infarction (Lateral)
- **Eligible:** `NO`
- **Total Prevalence:** 11,934 records across 5,491 patients (2/3 datasets)
- **Lead Hallmarks:** Unknown
- **Decision Rationale:** Unclear physiological observability on surface ECG; Low evidence verifiability under lead ablation
- **Criteria Checklist:** Sample Size: PASS | Reliable Mapping: PASS | Cross-Dataset: PASS | Observability: FAIL | Verifiability: FAIL

### Normal ECG
- **Eligible:** `YES`
- **Total Prevalence:** 605,775 records across 150,153 patients (3/3 datasets)
- **Lead Hallmarks:** Global across 12 leads
- **Decision Rationale:** Meets all 5 criteria.
- **Criteria Checklist:** Sample Size: PASS | Reliable Mapping: PASS | Cross-Dataset: PASS | Observability: PASS | Verifiability: PASS

### Right Bundle Branch Block
- **Eligible:** `YES`
- **Total Prevalence:** 66,865 records across 15,985 patients (3/3 datasets)
- **Lead Hallmarks:** V1, V2 (RSR') and I, aVL, V5, V6 (slurred S)
- **Decision Rationale:** Meets all 5 criteria.
- **Criteria Checklist:** Sample Size: PASS | Reliable Mapping: PASS | Cross-Dataset: PASS | Observability: PASS | Verifiability: PASS

### Right Bundle Branch Block (Incomplete)
- **Eligible:** `NO`
- **Total Prevalence:** 1,118 records across 1,049 patients (1/3 datasets)
- **Lead Hallmarks:** Unknown
- **Decision Rationale:** Limited cross-dataset presence (1/3 (PTB-XL)); Unclear physiological observability on surface ECG; Low evidence verifiability under lead ablation
- **Criteria Checklist:** Sample Size: PASS | Reliable Mapping: PASS | Cross-Dataset: FAIL | Observability: FAIL | Verifiability: FAIL

### Right Ventricular Hypertrophy
- **Eligible:** `NO`
- **Total Prevalence:** 6,074 records across 3,217 patients (3/3 datasets)
- **Lead Hallmarks:** Unknown
- **Decision Rationale:** Unclear physiological observability on surface ECG; Low evidence verifiability under lead ablation
- **Criteria Checklist:** Sample Size: PASS | Reliable Mapping: PASS | Cross-Dataset: PASS | Observability: FAIL | Verifiability: FAIL

### Sinus Bradycardia
- **Eligible:** `YES`
- **Total Prevalence:** 103,702 records across 45,905 patients (3/3 datasets)
- **Lead Hallmarks:** Global rate observable on any active lead
- **Decision Rationale:** Meets all 5 criteria.
- **Criteria Checklist:** Sample Size: PASS | Reliable Mapping: PASS | Cross-Dataset: PASS | Observability: PASS | Verifiability: PASS

### Sinus Tachycardia
- **Eligible:** `YES`
- **Total Prevalence:** 71,640 records across 35,439 patients (3/3 datasets)
- **Lead Hallmarks:** Global rate observable on any active lead
- **Decision Rationale:** Meets all 5 criteria.
- **Criteria Checklist:** Sample Size: PASS | Reliable Mapping: PASS | Cross-Dataset: PASS | Observability: PASS | Verifiability: PASS

---

## Proposed Final Frozen 6–7 Disease Set

The proposed frozen set provides balanced representation across distinct anatomical and electrophysiological mechanisms (Rhythm, Conduction, Ischemia, Rate), with clear lead-specific grounding boundaries essential for lead-ablation experiments:

| # | Disease Name | Category | Primary Lead Hallmarks | Total Records | Cross-Dataset | Rationale for Freezing |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Normal ECG** | Clinical Target | `Global across 12 leads` | 605,775 | 3/3 | Meets all 5 criteria. |
| **2** | **Sinus Bradycardia** | Clinical Target | `Global rate observable on any active lead` | 103,702 | 3/3 | Meets all 5 criteria. |
| **3** | **Myocardial Infarction (Inferior)** | Clinical Target | `II, III, aVF (reciprocal in I, aVL)` | 88,663 | 2/3 | Meets all 5 criteria. |
| **4** | **Atrial Fibrillation** | Clinical Target | `Prominently observable in II, V1, and rhythm strip` | 84,043 | 3/3 | Meets all 5 criteria. |
| **5** | **Sinus Tachycardia** | Clinical Target | `Global rate observable on any active lead` | 71,640 | 3/3 | Meets all 5 criteria. |
| **6** | **Right Bundle Branch Block** | Clinical Target | `V1, V2 (RSR') and I, aVL, V5, V6 (slurred S)` | 66,865 | 3/3 | Meets all 5 criteria. |
| **7** | **Myocardial Infarction (Anteroseptal)** | Clinical Target | `V1, V2, V3, V4` | 32,598 | 2/3 | Meets all 5 criteria. |

### Architectural Justification for Proposed Frozen Set
1. **Normal ECG (Negative Control)**: Baseline physiological control essential for specificity evaluation under lead degradation.
2. **Atrial Fibrillation (Supraventricular Arrhythmia)**: Hallmark irregular RR intervals and absent P waves observable globally across rhythm leads (Lead II, V1).
3. **Right Bundle Branch Block (Right Ventricular Conduction)**: Pathognomonic terminal conduction delay (rsR') strictly localized to anterior/septal leads (V1, V2) with slurred S wave in lateral leads (I, aVL, V5-V6). Tests localized evidence elimination when V1-V2 are masked.
4. **Left Bundle Branch Block (Left Ventricular Conduction)**: Broad monophasic R wave localized to lateral leads (I, aVL, V5-V6) and deep QS in V1-V2. Tests localized evidence elimination when lateral leads are masked.
5. **Sinus Bradycardia (Sinus Node Dysfunction - Slow Rate)**: Global rate disturbance (< 60 bpm) observable on any active lead; tests global reasoning stability under extreme lead reduction.
6. **Sinus Tachycardia (Sinus Node Dysfunction - Fast Rate)**: Global rate disturbance (> 100 bpm) observable on any active lead; complements Sinus Bradycardia across the physiological spectrum.
7. **1st Degree AV Block (Atrioventricular Conduction Delay)**: Fixed PR interval prolongation (> 200 ms) detectable in Lead II, aVF, and V1.
*(Alternative Ischemic Candidate: **Myocardial Infarction [Inferior / Anteroseptal]**, which provides regional ST-elevation and pathological Q-waves strictly confined to inferior [II, III, aVF] or septal [V1-V4] leads).*

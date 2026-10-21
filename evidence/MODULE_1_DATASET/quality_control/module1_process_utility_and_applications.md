# Module 1 — Process Utility, Research Contribution, and Applications

**Module:** Module 1 (Dataset Standardization, Eight-Label Harmonization, and Stratified 70:10:20 Splitting)  
**Deliverable Category:** Foundational Data Infrastructure  
**Governing Documents:** [`runs/module1_v2/MODULE1_FROZEN.json`](file:///e:/ECG/runs/module1_v2/MODULE1_FROZEN.json), [`runs/module1_v2/reports/module1_final_freeze_report.md`](file:///e:/ECG/runs/module1_v2/reports/module1_final_freeze_report.md)

---

## 1. Problem Addressed by Module 1

Clinical electrocardiography datasets suffer from severe multi-source heterogeneity:
1. **Conflicting Signal Formats:** Sampling rates vary ($100\text{ Hz}$, $500\text{ Hz}$, $1000\text{ Hz}$), units vary (ADC counts, microvolts, millivolts), durations differ, and lead orders differ across clinical device manufacturers.
2. **Taxonomic Incompatibility:** Annotation formats range from free-text physician interpretations, to hierarchical SCP-ECG statements (PTB-XL), to SNOMED-CT clinical concepts (Chapman), to automated proprietary interpretation engine statements (GE Marquette 12SL in MIMIC).
3. **Pervasive False Negatives from Naive Zero-Filling:** In multi-label medical ML, researchers frequently assume that an unmentioned pathology is absent, setting unobserved targets to 0. This injects thousands of false negative labels for conditions that were simply not evaluated or reported.
4. **Data Leakage from Patient Resampling:** Splitting records randomly when multiple ECGs exist for a single patient causes severe data leakage, overstating diagnostic generalization.

---

## 2. Research Purpose & Principal Design Decisions

Module 1 resolves these fundamental barriers through four core innovations:

### 2.1 Uniform Numerical Tensor Contract
- Standardizes all records to `(12, 1000)`, `float32`, physical `mV`, $100.0\text{ Hz}$, and canonical lead ordering.
- *Why necessary:* Directly aligns inputs with the pretrained ECG-FM foundation model (Module 3) and downstream reasoning transformer (Module 4), eliminating runtime preprocessing variations.

### 2.2 Tri-State Multi-Label Harmonization with Explicit Observation Masking
- Targets are modeled as tri-state values: $1.0$ (confirmed positive), $0.0$ (confirmed negative), and $\text{NaN}$ / `mask=False` (unobserved/unknown).
- *Why necessary:* Prevents unassessed conditions from being treated as confirmed negatives. Loss functions in Module 4 compute gradients strictly over observed targets ($\mathcal{L}_{\text{masked}}$).

### 2.3 Patient-Group-Aware Multi-Label Stratified Splitting
- Partitions the pooled 36,888 records into a strict **70:10:20** split (25,825 train, 3,688 validation, 7,375 test) using 10-fold iterative stratification while grouping by patient identifier.
- *Why necessary:* Preserves identical label prevalence across splits while preventing cross-split patient contamination.

### 2.4 Empirical Failure Cataloging
- Rather than discarding Chapman-Shaoxing due to missing patient IDs or hiding the 13 duplicate pairs, Module 1 explicitly catalogs them and establishes the exact boundary of statistical independence.

---

## 3. Downstream Dependencies

Module 1 outputs serve as the foundation for the entire research project:
- **Module 2 (Lead Degradation):** Directly consumes the standardized signals and split assignments to generate 7 masked montages ($36,888 \times 7 = 258,216$ arrays).
- **Module 3 (ECG-FM Representations):** Extracts 768-dimensional latent embeddings from Module 2 arrays, indexed by Module 1 global record IDs.
- **Module 4 (Control Reasoning Path):** Uses Module 1 label tensors and observation masks to train the reasoning transformer with masked BCE loss.

---

## 4. Practical & Clinical Research Applications

1. **Benchmarking ECG Foundation Models:** Provides a standardized, quality-controlled multi-cohort evaluation benchmark spanning 36,888 diverse clinical recordings.
2. **Tri-State Loss Evaluation:** Demonstrates how handling unobserved clinical targets properly prevents gradient distortion in multi-label learning.
3. **Multi-Cohort Robustness Testing:** Combines European (PTB-XL), Asian (Chapman-Shaoxing), and North American intensive care (MIMIC-IV) cohorts to test geographical and demographic transportability.

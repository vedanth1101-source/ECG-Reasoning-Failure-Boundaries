# Module 2: Progressive Lead Degradation — Evidence, Audits, and Architecture

**Module Identifier:** `MODULE_2_LEAD_DEGRADATION`  
**Authoritative Frozen Contract:** `runs/module2/MODULE2_FROZEN.json` (SHA-256: `6c71f5443fa00cb29ff37d5718a99478f654b9d5c4149fa531bfa4910cf972d3`)  
**Frozen Lead Degradation Config:** `runs/module2/config/module2_variant_config.json`  
**Verified Data Scale:** 36,888 eligible recordings $\times$ 7 configurations = 258,216 distinct input arrays.  

---

## 1. Module Purpose

Module 2 creates a strictly controlled, reproducible experimental framework for evaluating how the progressive loss of electrocardiographic leads affects foundation model representations and diagnostic reasoning. By applying deterministic binary lead-availability masks to the harmonized 12-lead ECG waveforms produced by Module 1, Module 2 simulates various reduced-lead clinical and ambulatory monitoring environments without altering waveform sampling rates, temporal alignment, patient grouping, or diagnostic disease labels.

---

## 2. Scientific Question Addressed

> *How does the progressive reduction of spatial observation channels (from standard 12-lead ECG down to single-lead monitoring) degrade physiological signal representations and clinical reasoning capacity, and what is the minimal lead montage required to preserve diagnostic accuracy across diverse cardiovascular pathologies?*

Standard diagnostic workflows rely on 12 simultaneous electrical projections to construct frontal and horizontal cardiac vector loops. However, point-of-care devices, consumer wearables, and emergency telemetry often operate with 1 to 6 leads. Module 2 standardizes the degradation continuum to quantify information loss systematically.

---

## 3. Input and Output Artifacts

### 3.1 Input Artifacts (from Module 1)
- **Harmonized Signal Arrays:** `data/processed/records/{record_id}.npy` (36,888 arrays).
  - Format: NumPy array of shape `(12, 1000)`, float32, physical units in mV, 100 Hz sampling rate, 10.0 s duration.
- **Master Dataset Split Manifest:** `data/splits/master_split_manifest.csv` (25,825 train, 3,688 validation, 7,375 test).
- **Disease Target Metadata:** `data/processed/disease_labels_multilabel.csv`.

### 3.2 Output Artifacts (Supplied to Modules 3 and 4)
- **Degraded Signal Arrays:** Standardized inputs $\mathbf{x}_{\text{masked}} \in \mathbb{R}^{12 \times 1000}$ for each of the 7 configurations.
- **Lead Availability Masks:** Binary vectors $\mathbf{m} \in \{0, 1\}^{12}$ indicating active versus zeroed channels.
- **Configuration Matrix Specification:** `evidence/MODULE_2_LEAD_DEGRADATION/configuration_definitions/lead_configuration_matrix.csv`.
- **Integrity & Availability Verification Register:** `evidence/MODULE_2_LEAD_DEGRADATION/lead_masks/lead_availability_evidence.csv`.

---

## 4. Key Processes and Design Rationale

### 4.1 Canonical Seven-Configuration Ladder
Module 2 specifies seven canonical configurations representing standardized physiological projections:

| Config Name | Active Leads Count | Active Lead Set | Masked Lead Set | Binary Mask Vector `[I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]` | Clinical / Physiological Rationale |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **12-lead** | 12 | All 12 leads | None | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]` | Full diagnostic reference standard covering all myocardial planes. |
| **10-lead** | 10 | I, II, aVR, aVL, V1–V6 | III, aVF | `[1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1]` | Inferior lead degradation (tests Einthoven limb redundancy). |
| **8-lead** | 8 | I, II, V1–V6 | III, aVR, aVL, aVF | `[1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]` | Bipolar limb leads I & II + full precordial horizontal plane. |
| **6-lead** | 6 | I, II, III, aVR, aVL, aVF | V1–V6 | `[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]` | Frontal limb plane only (complete loss of precordial/anterior data). |
| **4-lead** | 4 | I, II, III, V1 | aVR, aVL, aVF, V2–V6 | `[1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]` | Frontal bipolar limb + septal anchor (Lead V1 retained). |
| **2-lead** | 2 | II, V1 | I, III, aVR, aVL, aVF, V2–V6 | `[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]` | Primary rhythm strip (Lead II) + ventricular conduction anchor (Lead V1). |
| **1-lead** | 1 | II | All except Lead II | `[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]` | Single-lead ambulatory rhythm monitor (standard Einthoven II). |

> **Critical Note on Errata in Older Reports:** Early draft reports colloquially described 10-lead as "missing V5, V6" or 8-lead as "missing V1-V4". The authoritative frozen configuration registered in `MODULE2_FROZEN.json` and `module2_variant_config.json` confirms that 10-lead strictly masks leads `[III, aVF]` and 8-lead masks `[III, aVR, aVL, aVF]`. All downstream representations in Module 3 and models in Module 4 were evaluated strictly against this authoritative definition.

### 4.2 Exact Zeroing Without Synthetic Reconstruction
- **Mathematical Transformation:** For each channel $c \in \{0, \dots, 11\}$ and time sample $t \in \{0, \dots, 999\}$:
  $$\mathbf{x}_{\text{masked}}[c, t] = \mathbf{x}[c, t] \cdot m[c]$$
- **Design Rationale:** Missing channels are zeroed directly rather than filled with synthetic interpolations or autoregressive estimates. This ensures that downstream models are forced to evaluate true information deficits rather than reconstruction artifacts.
- **Preservation of Dimension:** Output tensors maintain the uniform shape `(12, 1000)` across all configurations. This allows the frozen foundation model backbone (Module 3) to process all variants through an invariant input layer.

### 4.3 Deterministic Record Alignment
Every single one of the 36,888 eligible records is degraded identically across the seven configurations, ensuring exact pairwise counterfactual comparisons without cross-cohort or cross-split sampling bias.

---

## 5. Evidence Supporting Implementation Correctness

The implementation correctness of Module 2 is verified by the following empirical evidence:
1. **Total Array Census:** $36,888 \times 7 = 258,216$ arrays processed with zero dropped or misaligned records.
2. **Signal Integrity and Finite Value Check:** All 258,216 arrays verified finite (zero `NaN`, zero `Inf`, zero out-of-range ADC overflow).
3. **Active Channel Conservation:** Verified that for all active channels ($m[c] = 1$), $\mathbf{x}_{\text{masked}}[c, :] \equiv \mathbf{x}_{\text{raw}}[c, :]$ to double-precision machine epsilon ($|\Delta| = 0.00000000$).
4. **Masked Channel Invariance:** Verified that for all masked channels ($m[c] = 0$), $\max |\mathbf{x}_{\text{masked}}[c, :]| = 0.00000000$.

---

## 6. Existing Audits and Test Results

| Audit ID | Document Name & Path | Scope and Checks Performed | Outcome / Verdict |
| :--- | :--- | :--- | :--- |
| **M2-AUD-01** | `audits/module2_lead_masking_audit.md` | Verification of binary mask application across all 258,216 arrays; validation of channel preservation and channel zeroing. | **PASSED** (100% compliance across all 7 configurations). |
| **M2-AUD-02** | `audits/module2_manual_verification_guide.md` | Spot-check inspection of individual degraded arrays, ADC scaling verification, and lead name alignment. | **PASSED** (Deterministic repeatability confirmed). |
| **M2-AUD-03** | `record_alignment/cross_configuration_alignment_audit.md` | Cross-configuration 1-to-1 record ID alignment, split manifest identity, and label invariance across all 7 tiers. | **PASSED** (Zero split drift; exactly 36,888 records per config). |
| **M2-AUD-04** | `signal_integrity/signal_integrity_audit.md` | Physical unit calibration (mV), float32 precision, absence of synthetic reconstruction, and finiteness scan. | **PASSED** (Strict numeric integrity verified). |

---

## 7. Practical and Research Applications

1. **Information-Theoretic Degradation Benchmarking:** Establishes the exact boundary conditions under which multi-lead foundation models fail when physiological projections are removed.
2. **Clinical Device Triage:** Identifies the minimum lead requirements for specific diagnostic tasks (e.g., distinguishing rhythm disorders like AFIB/SBRAD which survive in Lead II from ventricular conduction defects like RBBB/LBBB which require precordial Lead V1).
3. **Zero-Imputation Robustness Testing:** Serves as a standard testbed for evaluating deep learning architectures under missing sensor channels without relying on imputation models.

---

## 8. Known Defects and Methodological Limitations

1. **Clean Zeroing vs Real Lead-Off Noise:** Zero-masking simulates an ideal absence of signal, omitting high-impedance 50/60 Hz hum, amplifier saturation, or electrode motion artifacts found in real clinical disconnects.
2. **Vector Algebraic Redundancy:** Because frontal limb leads obey Einthoven's Law ($I + III = II$), masking Lead III does not fully eliminate Lead III information from a nonlinear deep model that observes Leads I and II.
3. **Precordial Plane Discontinuity:** The 6-lead configuration loses all precordial leads (horizontal plane), whereas the 4-lead configuration preserves Lead V1. Consequently, 4-lead representations retain higher cosine similarity to 12-lead than 6-lead representations.
4. **Non-Universal Hardware Montage:** Configurations represent experimental channel availability rather than commercial Holter/patch hardware topologies.

*Detailed analysis:* [module2_limitations.md](file:///e:/ECG/evidence/MODULE_2_LEAD_DEGRADATION/limitations/module2_limitations.md).

---

## 9. Evidence Gaps

1. **No Real-World Ambulatory Noise Overlay:** The archive contains zeroed waveforms, but does not benchmark additive electrode noise or baseline drift profiles.
2. **No Dynamic Lead Dropout:** Masks are static across the full 10-second duration; intermittent lead detachment (transient lead-off) was not evaluated.

---

## 10. Relevant Artifact Paths and Navigation Guide

- **Configuration Specification:** [lead_configuration_matrix.csv](file:///e:/ECG/evidence/MODULE_2_LEAD_DEGRADATION/configuration_definitions/lead_configuration_matrix.csv)
- **Lead Availability Evidence:** [lead_availability_evidence.csv](file:///e:/ECG/evidence/MODULE_2_LEAD_DEGRADATION/lead_masks/lead_availability_evidence.csv)
- **Cross-Config Alignment Report:** [cross_configuration_alignment_audit.md](file:///e:/ECG/evidence/MODULE_2_LEAD_DEGRADATION/record_alignment/cross_configuration_alignment_audit.md)
- **Signal Integrity Report:** [signal_integrity_audit.md](file:///e:/ECG/evidence/MODULE_2_LEAD_DEGRADATION/signal_integrity/signal_integrity_audit.md)
- **Process Utility and Downstream Applications:** [module2_process_utility_and_applications.md](file:///e:/ECG/evidence/MODULE_2_LEAD_DEGRADATION/reproducibility/module2_process_utility_and_applications.md)
- **Limitations Analysis:** [module2_limitations.md](file:///e:/ECG/evidence/MODULE_2_LEAD_DEGRADATION/limitations/module2_limitations.md)
- **Audit Register:** [module2_audit_register.csv](file:///e:/ECG/evidence/MODULE_2_LEAD_DEGRADATION/audits/module2_audit_register.csv)

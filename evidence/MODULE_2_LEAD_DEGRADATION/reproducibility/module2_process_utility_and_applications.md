# Module 2 — Process Utility, Research Contribution, and Applications

**Module:** Module 2 (Progressive ECG Lead Masking and Availability-Mask Generation)  
**Deliverable Category:** Controlled Sensor Information Deprivation Infrastructure  
**Governing Documents:** [`runs/module2_v1/MODULE2_FROZEN.json`](file:///e:/ECG/runs/module2_v1/MODULE2_FROZEN.json), [`runs/module2_v1/reports/module2_lead_masking_audit.md`](file:///e:/ECG/runs/module2_v1/reports/module2_lead_masking_audit.md)

---

## 1. Problem Addressed by Module 2

Most modern machine learning models for electrocardiography assume the availability of a standard, complete 12-lead acquisition. However, in real-world clinical and remote environments:
1. **Pervasive Channel Loss in Telemetry & Wearables:** Ambulatory monitors, smartwatches, continuous telemetry patches, and emergency defibrillator pads capture only 1, 2, or 4 leads.
2. **Confounding in Real-World Degraded Datasets:** Real-world single-lead or reduced-lead datasets typically come from different patient populations, different recording equipment, and different clinical environments than 12-lead diagnostic databases. Comparing a 12-lead model trained on hospitalized patients to a 1-lead model trained on ambulatory patients confounds channel loss with cohort shift.
3. **Flawed Synthetic Reconstructions:** Many prior pipelines attempt to synthetically reconstruct missing leads (e.g. estimating 12 leads from 1 or 2 leads via neural networks), which injects hallucinations and masks true failure boundaries.

---

## 2. Research Purpose & Principal Design Decisions

Module 2 provides a strictly controlled experimental framework for isolating physiological information loss:

### 2.1 Controlled Counterfactual Degradation
- By progressively zeroing channels from the exact same 36,888 quality-controlled recordings, Module 2 holds the patient, the clinical pathology, and the recording session strictly constant.
- *Research Contribution:* Any diagnostic failure or representation shift observed down the ladder ($12 \rightarrow 10 \rightarrow 8 \rightarrow 6 \rightarrow 4 \rightarrow 2 \rightarrow 1$) is causally attributable to the removal of spatial electrocardiographic vectors.

### 2.2 Anatomically Grounded Montages
The seven configurations model standard clinical acquisition hardware:
- **10-lead (omits III, aVF):** Models non-redundant frontal acquisitions.
- **8-lead (omits III, augmented leads):** Models the 8 physical recording channels of digital ECG carts.
- **6-lead (limb leads only):** Models emergency triage limb clamps without chest leads.
- **4-lead (I, II, III, V1):** Models 4-cable bedside telemetry monitoring with exploratory septal lead V1.
- **2-lead (II, V1):** Models dual-channel Holter and continuous telemetry monitoring.
- **1-lead (II):** Models single-channel rhythm strips, defibrillator pad views, and single-lead wearable patches.

### 2.3 Strict Zero-Masking Without Reconstruction
- Missing channels are set strictly to floating-point zero.
- *Why necessary:* Prevents generative hallucinations and tests whether deep foundation models can reason directly over incomplete sensor observations.

---

## 3. Downstream Dependencies

Module 2 artifacts feed directly into the subsequent phases:
- **Module 3 (ECG-FM Representations):** The 258,216 masked arrays serve as direct inputs to the frozen ECG-FM backbone, generating 258,216 representation vectors ($z \in \mathbb{R}^{768}$) to quantify representation drift.
- **Module 4 (Control Reasoning Path):** The reasoning transformer evaluates diagnostic performance and generates reasoning traces across all 7 configurations, testing where clinical reasoning fails.

---

## 4. Practical & Clinical Applications

1. **Failure Boundary Mapping:** Identifies the exact sensor threshold where clinical diagnostic capabilities collapse (e.g. rhythm diagnoses like AFIB/SBRAD/STACH persist down to 1-lead, whereas conduction and ischemia diagnoses collapse when precordial leads are lost).
2. **Clinical Device Deployment Guidance:** Informs hospital procurement and device manufacturers regarding whether a 4-lead or 2-lead telemetry montage is sufficient for specific monitoring targets.
3. **Foundation Model Robustness Auditing:** Tests whether modern self-supervised ECG foundation models maintain invariant semantic representations under sensor dropouts.

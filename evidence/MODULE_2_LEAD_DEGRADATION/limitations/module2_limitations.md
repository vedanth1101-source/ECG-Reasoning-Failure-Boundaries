# Module 2 — Progressive Lead Degradation: Known Limitations and Downstream Research Impacts

**Module Identifier:** `MODULE_2_LEAD_DEGRADATION`  
**Authoritative Frozen Contract:** `runs/module2/MODULE2_FROZEN.json`  
**Execution Date:** 2026-10-16  

---

## 1. Scope and Objective

This document catalogs the structural, physiological, and experimental limitations of Module 2 (Progressive Lead Degradation) based on verified repository artifacts and audit findings. It explains how these limitations constrain the clinical interpretability of downstream results in Module 3 (ECG-FM representations) and Module 4 (control-path reasoning).

---

## 2. Identified Technical and Methodological Limitations

### 2.1 Absence of Physical Cable Disconnection Artifacts (Clean Zeroing vs Real Noise)
- **Mechanism:** In Module 2, degraded leads are zeroed mathematically:
  $$\mathbf{x}_{\text{masked}}[c, t] = \mathbf{x}[c, t] \cdot m[c], \quad m[c] \in \{0, 1\}$$
- **Limitation:** In clinical environments, a detached or degraded lead rarely presents as clean 0.000 mV flatline. Physical disconnections introduce:
  - 50/60 Hz mains powerline hum from high impedance.
  - Rail-to-rail amplifier saturation.
  - High-amplitude motion artifacts and intermittent baseline wander.
- **Downstream Consequence:** Downstream models (ECG-FM backbone and reasoning classifier) evaluate an idealized absence of signal. Performance under physical lead detachment or poor electrode contact cannot be directly inferred from these zero-masked benchmarks.

### 2.2 Vector Electrocardiographic Redundancy and Einthoven Reconstruction
- **Mechanism:** Standard frontal limb leads are geometrically interdependent via Einthoven's Law and Goldberger's equations:
  $$\text{Lead II} = \text{Lead I} + \text{Lead III}$$
  $$\text{aVR} = -\frac{\text{Lead I} + \text{Lead II}}{2}, \quad \text{aVL} = \text{Lead I} - \frac{\text{Lead II}}{2}, \quad \text{aVF} = \text{Lead II} - \frac{\text{Lead I}}{2}$$
- **Limitation:** In configurations such as 10-lead (Lead III and aVF masked) or 8-lead (Lead III, aVR, aVL, aVF masked), Leads I and II remain active. A deep neural network with sufficient representation capacity can implicitly reconstruct the zeroed limb leads through linear combinations of the active channels.
- **Downstream Consequence:** Performance degradation in 10-lead and 8-lead configurations is minimal not necessarily because those leads lack diagnostic information, but because their information is algebraically redundant with Leads I and II.

### 2.3 Non-Uniform Precordial Spatial Coupling
- **Mechanism:** Precordial leads (V1–V6) record electrical activity in the horizontal plane across the anterior and lateral myocardium.
- **Limitation:** In the 4-lead configuration (`[I, II, III, V1]`), Lead V1 is preserved while V2–V6 are zeroed. V1 provides crucial septal and right ventricular depolarization information. Because ventricular activation propagates continuously across the anterior wall, preserving V1 maintains significant predictive grounding for anterior wall pathology (and bundle branch blocks), creating a non-linear drop-off when transitioning from 4-lead to 2-lead (`[II, V1]`) and 1-lead (`[II]`).
- **Downstream Consequence:** The degradation profile across configurations is non-monotonic with respect to lead count when comparing horizontal vs frontal planes. Specifically, the 6-lead configuration (which retains all 6 frontal limb leads but zeroes all 6 precordial leads) lacks horizontal plane information entirely, whereas the 4-lead configuration retains V1. This structural asymmetry explains why 4-lead representations retain higher cosine similarity to 12-lead (0.9041) than 6-lead representations (0.8775).

### 2.4 Academic Counterfactual Setting vs Clinical Hardware Standards
- **Mechanism:** The seven configurations (12, 10, 8, 6, 4, 2, 1 lead) were selected as a standardized experimental ladder for information-theoretic degradation analysis.
- **Limitation:** These seven configurations do not map directly to commercial ambulatory monitor electrode topologies (e.g., EASI 5-lead derivation, Frank XYZ vectorcardiography, or 3-channel Mason-Likar Holter montages).
- **Downstream Consequence:** Findings must be reported as evaluations of *channel availability under standard 12-lead placement*, not as validations of specific commercial wearable or telemetry hardware devices.

---

## 3. Impact on Research Conclusions

1. **Information Bound vs Diagnostic Failure:** A drop in diagnostic AUROC under degraded leads reflects the joint effect of lost physiological projections and the model's capacity to extract residual correlations from surviving leads.
2. **Reconstruction Independence:** Because Module 2 strictly enforces no synthetic reconstruction ($\mathbf{x}_{\text{masked}}[c, :] = 0$), downstream models must handle channel absence natively.
3. **Trace Availability Audits:** When evaluating reasoning traces in Module 4, any claim regarding morphology in zeroed channels (e.g., "ST elevation in Lead III" when Lead III is masked) represents an ungrounded hallucination or channel-unavailability violation, as confirmed in the corrected Module 4 validation audits.

---

## 4. Verification Evidence References

- `evidence/MODULE_2_LEAD_DEGRADATION/configuration_definitions/lead_configuration_matrix.csv`
- `evidence/MODULE_2_LEAD_DEGRADATION/lead_masks/lead_availability_evidence.csv`
- `evidence/MODULE_2_LEAD_DEGRADATION/signal_integrity/signal_integrity_audit.md`
- `evidence/MODULE_2_LEAD_DEGRADATION/audits/module2_lead_masking_audit.md`

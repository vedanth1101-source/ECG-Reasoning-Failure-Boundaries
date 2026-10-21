# MODULE 3 — POST-FREEZE CORRECTIONS ADDENDUM

**Project:** Reasoning Failure Boundaries in ECG Foundation Models Under Progressive Physiological Information Loss  
**Subject:** Formal Corrections and Interpretive Amendments to Module 3 Reports  
**Audit Date:** 2026-10-13  
**Status:** **ACTIVE CORRECTION ADDENDUM** (Preserving Immutable Freeze of `MODULE3_FROZEN.json`)  

---

## 1. Rationale & Governance Policy

In accordance with strict scientific research integrity protocols:
1. **The frozen representation outputs (`runs/module3_v1/representations/`) remain 100% unaltered, unregenerated, and bitwise frozen.**
2. **The original freeze record (`MODULE3_FROZEN.json`) is preserved as a permanent historical audit baseline.**
3. This Addendum provides targeted, versioned corrections to narrative descriptions, configuration wordings, and interpretive claims where post-freeze auditing identified inaccuracies or imprecise characterizations.
4. Subsequent modules (Module 4 onwards) must cite and adopt the corrected definitions specified in this document.

---

## 2. Erratum 1: Resolution of the Monotonicity Contradiction

### 2.1. The Inaccurate Claim
The Executive Summary and narrative overview of Module 3 previously stated:
> *"Progressive physiological lead loss produces monotonic cosine-similarity decay across the seven lead configurations."*

### 2.2. Mathematical Evidence & Contradiction
Recalculation and validation across all 36,888 records reveals the actual pooled mean cosine similarities relative to the 12-lead reference:
- **12-lead:** $1.000000$
- **10-lead:** $0.998051$
- **8-lead:** $0.997095$
- **6-lead:** $0.877548$
- **4-lead:** $0.904078$  *(Rises relative to 6-lead!)*
- **2-lead:** $0.876178$
- **1-lead:** $0.814373$

Because **$\text{cos\_sim}(\text{4-lead}) = 0.904078 > \text{cos\_sim}(\text{6-lead}) = 0.877548$**, the sequence of embeddings ordered purely by descending lead count is **not globally monotonic**.

### 2.3. Root Cause: Non-Nested Lead Subspaces
Monotonic embedding decay under channel masking only holds if each successive configuration is a strict mathematical subset of the preceding configuration ($\mathcal{C}_{k+1} \subset \mathcal{C}_k$). In the experimental design:
- `10-lead` ($\{\text{I, II, aVR, aVL, V1–V6}\}$) $\subset$ `12-lead` ($\{\text{I, II, III, aVR, aVL, aVF, V1–V6}\}$)
- `8-lead` ($\{\text{I, II, V1–V6}\}$) $\subset$ `10-lead` $\subset$ `12-lead`
- `6-lead` ($\{\text{I, II, III, aVR, aVL, aVF}\}$) $\subset$ `12-lead`, **but `6-lead` is NOT a subset of `8-lead`**, and **`4-lead` is NOT a subset of `6-lead`**!

Specifically:
- In `6-lead`, **all six precordial leads (V1–V6) are zeroed**, retaining only the 6 frontal limb leads.
- In `4-lead`, **precordial lead V1 is present** ($\{\text{I, II, III, V1}\}$). 

Because lead V1 provides horizontal-plane anterior dipole projection that is completely absent in the 6-lead limb montage, the foundation model's representation in the 4-lead configuration is closer to the 12-lead embedding ($0.9041$) than the 6-lead limb montage ($0.8775$).

### 2.4. Formal Corrective Statement
> **Corrected Interpretation:**
> Monotonic representation drift occurs within two distinct, nested physiological subspaces:
> 1. **Limb Lead Reduction Subspace (with full precordial coverage):** `12-lead` ($1.0000$) $\rightarrow$ `10-lead` ($0.9981$) $\rightarrow$ `8-lead` ($0.9971$). Cosine similarity decays monotonically with limb lead removal as long as horizontal-plane precordial leads (V1–V6) remain intact.
> 2. **Minimal Hybrid Subspace (with V1 retained):** `4-lead` ($0.9041$) $\rightarrow$ `2-lead` ($0.8762$) $\rightarrow$ `1-lead` ($0.8144$). Cosine similarity decays monotonically as leads are removed down to rhythm lead II.
> 3. **Cross-Subspace Non-Monotonicity:** The transition from `8-lead` to `6-lead` represents an abrupt shift from a precordial-dominant montage to a purely frontal-limb montage, inducing a severe representation drop ($0.9971 \rightarrow 0.8775$). Transitioning from `6-lead` to `4-lead` re-introduces anterior precordial lead V1, partially restoring embedding alignment ($0.8775 \rightarrow 0.9041$) despite having fewer total active leads.

---

## 3. Erratum 2: Correction of 8-Lead Configuration Description

### 3.1. The Ambiguous Text
In Section 4.3 of `module3_final_audit_report.md` and related notes, the text stated:
> *"The loss of limb leads III and aVF (10-lead) or further loss of aVR and aVL (8-lead) causes less than 0.3% cosine drift."*

### 3.2. Ambiguity & Correction
This phrasing describes the incremental change from 10-lead to 8-lead (where aVR and aVL are zeroed in addition to III and aVF). However, if read relative to the 12-lead baseline, it could be misinterpreted as claiming that the 8-lead configuration only excludes aVR and aVL.

### 3.3. Formal Corrective Statement
Relative to the unmasked 12-lead configuration, the actual lead subsets configured in Module 2 and extracted in Module 3 are:

| Configuration | Active Leads | Leads Zeroed Relative to 12-Lead Baseline | Leads Zeroed Relative to Preceding Step |
| :--- | :--- | :--- | :--- |
| **12-lead** | I, II, III, aVR, aVL, aVF, V1–V6 | None (0 leads) | Baseline |
| **10-lead** | I, II, aVR, aVL, V1–V6 | III, aVF (2 leads) | III, aVF |
| **8-lead** | I, II, V1–V6 | **III, aVF, aVR, aVL (4 leads)** | aVR, aVL |
| **6-lead** | I, II, III, aVR, aVL, aVF | V1, V2, V3, V4, V5, V6 (6 leads) | Swapped: limbs active, all precordials zeroed |
| **4-lead** | I, II, III, V1 | aVR, aVL, aVF, V2, V3, V4, V5, V6 (8 leads) | aVR, aVL, aVF zeroed, V1 restored |
| **2-lead** | II, V1 | I, III, aVR, aVL, aVF, V2, V3, V4, V5, V6 (10 leads) | I, III zeroed |
| **1-lead** | II | All except lead II (11 leads) | V1 zeroed |

---

## 4. Erratum 3: Precise Scientific Framing of Representation Drift

### 4.1. Claim Discipline
In certain descriptive headings and narrative text, phrases such as *"physiological information loss"* and *"reasoning failure boundaries"* could be conflated with the raw representation drift measurements.

### 4.2. Formal Corrective Framing
1. **Measured Quantity:** Module 3 measures **representation drift**—defined as the geometric displacement (cosine distance and normalized Euclidean distance) between frozen 768-dimensional latent vectors extracted from the pretrained ECG-FM encoder.
2. **Distinction from Clinical Accuracy:** Representation drift in a self-supervised foundation model indicates that the encoder produces altered contextual representations when specific input channels are zeroed. It does **not** prove that downstream diagnostic classification fails, nor does it measure information-theoretic entropy loss directly.
3. **Hypothesis for Module 4:** The observed embedding divergence provides a mechanistic hypothesis to test in Module 4: *Does the non-linear embedding shift observed in 6-lead and 1-lead configurations correspond to catastrophic degradation in downstream diagnostic performance for specific cardiac pathologies (e.g., anterior MI vs. inferior MI vs. rhythm disorders)?*

---

## 5. Adoption & Integration for Downstream Modules

All downstream research scripts and reports (Module 4 Probing and Module 5 Failure Boundary Analysis) must:
1. Reference the 7 lead configurations by their exact active lead sets.
2. Characterize the drift trajectory as a two-subspace system with cross-plane discontinuity at `6-lead` vs `4-lead`.
3. Evaluate downstream classification performance directly against the frozen representations without assuming that representation drift strictly predicts diagnostic failure.

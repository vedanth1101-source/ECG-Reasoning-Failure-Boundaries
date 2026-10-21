# Module 4 — Decision Register: Resolved Methodological Decisions

**Date:** 2026-10-14  
**Phase:** Module 4 Stage 0 Protocol Review -> Stage 1 Transition  
**Status:** **ALL DECISIONS FORMALLY RESOLVED & RATIFIED BY USER**  

This document records the user's explicit approvals and the authoritative verification of clinical criteria for decisions U-01 through U-04.

---

### Decision U-01: Canonical Representation of Boundary Points
* **User Decision:** **APPROVED — Option A (First Failure Configuration as Primary Boundary).**
* **Adopted Rule:**
  - The primary scalar failure boundary is the **First Failure Configuration** $c_{\text{fail}} = \text{first } c_k \in \mathcal{S}$ where $GQ(c_k) < \tau$ (e.g. `RFB = 6-lead`), explicitly accompanied in all reporting tables by the **Transition Pair** $(c_{k-1} \to c_k)$ (e.g. `8-lead -> 6-lead`) and the **Last Passing Configuration** $c_{\text{pass}}$ (e.g. `8-lead`).
  - In cases of subsequent non-monotonic recovery (e.g. 4-lead recovering over 6-lead), the initial failure point is reported alongside the sustained failure point and the flag `RECOVERY_DETECTED = True`.

---

### Decision U-02: Primary Headline Formulation for Decoupling Gap ($\Delta_{\text{decoupling}}$)
* **User Decision:** **APPROVED — Option A (Dual Reporting of Step Gap and Physical Lead Gap).**
* **Adopted Rule:**
  - **Ordinal Degradation Step Gap:** $\Delta_{\text{steps}} = k_{\text{DFB}} - k_{\text{RFB}}$ on integer scale $[-6, +6]$ across $\mathcal{S} = \langle 12, 10, 8, 6, 4, 2, 1 \rangle$.
  - **Physical Lead Count Gap:** $\Delta_{\text{leads}} = l_{\text{RFB}} - l_{\text{DFB}}$ (physical channel difference, e.g. $+4\text{ leads}$).
  - **Sign Convention:** $\Delta > 0$ denotes **Diagnostic Persistence Beyond Reasoning Failure** (the dangerous regime where diagnostic predictions remain accurate despite ungrounded reasoning); $\Delta < 0$ denotes diagnostic fragility; $\Delta = 0$ denotes synchronous failure.

---

### Decision U-03: Primary Aggregation Protocol for Grounding Quality ($GQ_l$)
* **User Decision:** **APPROVED — Option A (Micro-Aggregation Primary, Macro-Aggregation Sensitivity).**
* **Adopted Rule:**
  - **Primary Headline Metric:** **Micro-Aggregation** (summing $N_{\text{supported}}$ and $N_{\text{contradicted}}$ across all 7,375 test records for each lead configuration before division). This provides statistical stability and total immunity to record-level zero denominators.
  - **Secondary Sensitivity Metric:** **Macro-Aggregation** (mean and standard error of per-record $GQ$ across records with $\ge 1$ observable claim) reported in supplementary tables with 95% bootstrap confidence intervals.

---

### Decision U-04: Strict Electrophysiological Verification Rule for Myocardial Infarction
* **User Instruction:** Retain strict, predefined clinical evidence requirements and do not relax them merely to produce a supported or contradicted verdict. Verify contiguous-lead requirement and anatomical-territory mapping against approved project criteria.
* **Verification Findings:**
  1. **Anatomical Territories Codified:** [`src/reasoning/clinical_taxonomy.py`](file:///e:/ECG/src/reasoning/clinical_taxonomy.py#L24-L33) explicitly defines:
     ```python
     ANATOMICAL_TERRITORIES = {
         "inferior": ["II", "III", "aVF"],
         "septal": ["V1", "V2"],
         "anterior": ["V3", "V4"],
         "anteroseptal": ["V1", "V2", "V3", "V4"],
         "lateral": ["I", "aVL", "V5", "V6"],
         "high_lateral": ["I", "aVL"],
         "extensive_anterior": ["I", "aVL", "V1", "V2", "V3", "V4", "V5", "V6"],
     }
     ```
     Anchored in AHA/ACCF/HRS Recommendations (Surawicz et al., JACC 2009; Wagner et al., Circulation 2009).
  2. **Contiguous Lead Verification Rule Codified:** [`reports/evidence_observability_config.json`](file:///e:/ECG/reports/evidence_observability_config.json#L130-L145) explicitly mandates:
     - `measurement_requirement`: `st_elevation >= 0.10 mV (1 mm) in >= 2 contiguous anatomical leads` or `pathological Q waves (duration >= 40 ms or amplitude >= 25% of R wave)`.
     - `verification_rule`: `st_elevation.present and len(st_elevation.leads) >= 2 in contiguous territory`.
     - `unavailable_condition`: `no contiguous lead pairs active in any anatomical territory (Inferior: [II, III, aVF], Anterior: [V3, V4], Septal: [V1, V2], Lateral: [I, aVL, V5, V6])`.
  3. **Phase Specification Alignment:** [`PROJECT_PHASES_SPECIFICATION.md`](file:///e:/ECG/PROJECT_PHASES_SPECIFICATION.md#L335) confirms: `Inferior MI requires ST elevation in >= 2 contiguous inferior leads: II, III, aVF`.
* **Resolution:** **VERIFIED & RATIFIED — STRICT CRITERIA RETAINED.**
  - Under 2-lead telemetry (II, V1) and 1-lead ambulatory patch (II), zero contiguous anatomical lead pairs exist.
  - Therefore, any claim asserting localized ST elevation for Myocardial Infarction at 2-lead or 1-lead is strictly and reproducibly categorized as **`UNAVAILABLE`** ($100\%$ ungrounded), exposing the model's reliance on non-anatomical priors.
  - Zero rules were invented; existing project specifications are upheld in full.

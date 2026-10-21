# Module 4 v3 — Validation Reasoning-Trace Documentation Correction Report

**Remediation Date:** 2026-10-15  
**Governing Module:** Module 4 v3 (Multi-Lead Degradation and Reasoning Transformer)  
**Governing Authority:** [`runs/module4_v3/reports/validation_trace_consistency_review.md`](file:///e:/ECG/runs/module4_v3/reports/validation_trace_consistency_review.md)  
**Task Scope:** Strictly Documentation-Only Remediation & Evidence Correction  
**Primary Checkpoint:** `runs/module4_v3/checkpoints/control_reasoning_best_seed42.pt` (SHA-256: `2920c76884e4a432b69326f1d943cb4495e6e55305f508fe3acab3d9e9067d93`)  
**Underlying Artifacts Preserved:** All 25,816 validation trace dossiers, frozen checkpoints, split manifests, and source scripts remain strictly untouched.

---

## 1. Executive Summary & Objective

This report provides the authoritative documentation remediation for the Module 4 v3 validation reasoning-trace evidence. The preceding independent consistency review ([`validation_trace_consistency_review.md`](file:///e:/ECG/runs/module4_v3/reports/validation_trace_consistency_review.md)) identified several critical defects in the initial generation report ([`validation_trace_generation_report.md`](file:///e:/ECG/runs/module4_v3/reports/validation_trace_generation_report.md)):
1. **Conflation of Channel Unavailability with Contradiction:** Missing diagnostic leads were improperly classified as "Contradictory to available evidence" rather than "Required evidence unavailable" or "Insufficient evidence to assess".
2. **Unsupported Narrative Regarding Anterior MI in 6-Lead:** The initial report claimed anterior MI drove contradictions in 6-lead, whereas an exhaustive data audit proved that exactly 0 cases were anterior MI.
3. **Unsupported Causal Interpretation of V1 Grounding:** The initial report asserted that retaining septal lead V1 caused a drop in contradictions and explained superior 4-lead AUROC over 6-lead. Physical trace audit revealed that physiological contradictions were identical (43 vs. 43), and the apparent drop was an artifact of classification logic.
4. **Asymmetric Disease Evaluation:** The classification heuristic selectively checked lead feasibility for RBBB, LBBB, and MI, while completely omitting lead checks and contradiction checks for NORM, AFIB, and 1AVB.
5. **Lead Configuration Textual Errata:** The initial report's narrative description of lead masks in Section 3.2 conflicted with the physical masks in [`src/reasoning/multi_cohort_loader.py`](file:///e:/ECG/src/reasoning/multi_cohort_loader.py) and the frozen Module 2 specification ([`runs/module2_v1/reports/module2_variant_config.json`](file:///e:/ECG/runs/module2_v1/reports/module2_variant_config.json)).
6. **Breakdown Table Row Arithmetic Omission:** The 6-lead breakdown table omitted 2 traces that had lead unavailability only, causing the row sum to show 287 instead of the true total of 289.

### Remediation Protocol
In accordance with strict research preservation rules:
- No model retraining, fine-tuning, or weight modifications occurred.
- No trace regeneration or modification of the 25,816 JSON trace dossiers occurred.
- No underlying classification code (`scripts/18_generate_validation_reasoning_traces.py`, `src/reasoning/trace_generator.py`, or `src/reasoning/context_builder.py`) was altered.
- No test records, test labels, or test traces were accessed or unblinded.
- Previous reports are preserved intact as an auditable historical trail. This document and its accompanying CSV ([`validation_trace_corrected_findings.csv`](file:///e:/ECG/runs/module4_v3/reports/validation_trace_corrected_findings.csv)) serve as the authoritative corrected interpretation.

---

## 2. Mandatory Preservation Verification

Prior to preparing this documentation correction, all physical artifacts and freeze manifests were audited to confirm zero unauthorized modifications:

| Artifact Path | Expected Hash / Role | Verified Current Hash | Preservation Status |
| :--- | :--- | :--- | :---: |
| `runs/module4_v3/checkpoints/control_reasoning_best_seed42.pt` | `2920c76884e4a432b69326f1d943cb4495e6e55305f508fe3acab3d9e9067d93` | `2920c76884e4a432...` | **VERIFIED MATCH** |
| `runs/module4_v3/checkpoints/control_reasoning_best_seed1337.pt` | `d38088e1af53f5068ed98e77801aa725c724df2b96032be9bcc65c31072a97c2` | `d38088e1af53f506...` | **VERIFIED MATCH** |
| `runs/module4_v3/checkpoints/control_reasoning_best_seed2024.pt` | `fd0620125742bacd16947948ceb2f8ef1744e11db32e54b15da556b0a0ff3f6d` | `fd0620125742bacd...` | **VERIFIED MATCH** |
| `runs/module4_v3/splits/split_manifest.csv` | `b2547e9c7457a8c21267dd53f141a7b1155f7d49fa4374badb3f09c30c431253` | `b2547e9c7457a8c2...` | **VERIFIED MATCH** |
| `runs/module1_v2/MODULE1_FROZEN.json` | `4666370e83408c734d19582d99e0e13f76bb44c60c1bea376843637d4fcf0e76` | `4666370e83408c73...` | **VERIFIED MATCH** |
| `runs/module2_v1/MODULE2_FROZEN.json` | `b267ad3a95f6ac14a2f8b13095444b979d82eadee8fdb01bd492f137034fd5a6` | `b267ad3a95f6ac14...` | **VERIFIED MATCH** |
| `runs/module3_v1/MODULE3_FROZEN.json` | `dfc90481b200f4b229d74dd761f6e0a42174f4a94c280967a04c4207a13ade79` | `dfc90481b200f4b2...` | **VERIFIED MATCH** |
| `runs/module4_v3/MODULE4_FROZEN.json` | `aa61673ab46b1c15fe08fb1aa3e9d94f79aa8144f8a55f9c9809e280c4ad8cc1` | `aa61673ab46b1c15...` | **VERIFIED MATCH** |
| `runs/module4_v3/validation_traces/` (7 JSON files) | Exactly 25,816 validation trace dossiers; total size 112,198,331 bytes | Confirmed identical | **VERIFIED UNCHANGED** |

Zero test inference, individual record inspection, or label access occurred. Fail-closed split quarantine remained enforced.

---

## 3. Four-Tier Classification Framework for Evidence Consistency

To establish rigorous clinical epistemology, all reasoning-trace evaluations must explicitly distinguish between four mutually exclusive categories:

```
                                  EVIDENCE-CONSISTENCY TAXONOMY
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. Consistent with Available Evidence                                                           │
│    The model's diagnostic claim is supported by active observable criteria, has passed an       │
│    implemented contradiction check, and attends exclusively to physically measured leads.      │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. Physiologically Contradictory                                                                │
│    An active, implemented, defensible physiological rule demonstrates a direct conflict between │
│    the diagnostic claim and observable criteria (e.g. SBRAD claimed when HR >= 60 bpm).         │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. Required Evidence Unavailable                                                                │
│    The diagnostic claim requires a specific anatomical lead or physiological measurement that   │
│    was physically unmeasured/masked in that montage (e.g. RBBB claimed without Lead V1).        │
│    *An unmeasured channel cannot confirm or refute a disease; absence of evidence is not       │
│     evidence of absence or contradiction.*                                                      │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 4. Insufficient Evidence to Assess                                                              │
│    The trace generator lacks contradiction rules for that disease class (NORM, AFIB, 1AVB, MI), │
│    or cross-attention queries unmeasured lead tokens without a direct physiological conflict.   │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

> [!CAUTION]
> **Defect in Existing Classification Logic:**
> In `scripts/18_generate_validation_reasoning_traces.py` (lines 104–128), Category 3 ("Required Evidence Unavailable") was conflated with Category 2 ("Physiologically Contradictory"). Traces where required leads were absent (such as V1 for bundle branch blocks) were forced into the "Contradictory" count.
>
> In this remediation, retrospective reclassification of individual JSON trace dossiers is **not** performed, as modifying existing trace files would breach the non-destructive freeze contract. Instead, corrected aggregate counts and rates are established from the permitted audit evidence, while full individual-level re-tagging is scheduled for an approved future pipeline run.

---

## 4. Remediation of Review Findings (REV-VAL-01 to REV-VAL-07)

### REV-VAL-01: Channel-Loss Classification Conflation
- **Original Claim:** The initial generation report presented aggregate contradiction rates of **7.8% (6-lead)**, **18.9% (2-lead)**, and **22.7% (1-lead)** as reflecting model reasoning contradictions.
- **Audited Reality:** The underlying classification script set `classification = "Contradictory to available evidence"` whenever `disease_leads_ok == False`. Missing leads represent an absence of evidence, not physiological contradiction.
- **Corrected Aggregate Evidence:** When true physiological contradictions (HR bound conflicts) are separated from lead unavailability across all 25,816 validation evaluations:
  - **12-lead:** 42 physiological HR contradictions (1.14%)
  - **10-lead:** 37 physiological HR contradictions (1.00%)
  - **8-lead:** 38 physiological HR contradictions (1.03%)
  - **6-lead:** 43 physiological HR contradictions (1.17%) [plus 246 lead-unavailable traces]
  - **4-lead:** 43 physiological HR contradictions (1.17%) [0 lead-unavailable traces]
  - **2-lead:** 45 physiological HR contradictions (1.22%) [plus 652 lead-unavailable traces]
  - **1-lead:** 54 physiological HR contradictions (1.46%) [plus 782 lead-unavailable traces]
- **Correction Statement:** The true physiological contradiction rate is virtually flat across all configurations (~1.0% to 1.5%). The reported aggregate contradiction rates of 7.8%, 18.9%, and 22.7% are artifacts of classifying unavailable channels as contradictions and are **unsuitable for evaluating clinical contradiction**.

---

### REV-VAL-02: Retraction of Anterior MI Narrative in 6-Lead
- **Original Claim:** The initial report asserted: *"In 6-lead, in 244 cases, the model predicts an anterior myocardial infarction or conduction disturbance as its top hypothesis... attention frequently queries precordial tokens... generating ungrounded claims."*
- **Audited Reality:**
  1. An exhaustive inspection of all 289 contradictory traces in `traces_validation_6-lead.json` revealed **0 cases** where MI was the top hypothesis.
  2. The candidate acronym space contains only generic `MI`; no "Anterior MI" class exists in the model.
  3. In 6-lead, limb leads contain contiguous pairs (`II-III`, `III-aVF`, `II-aVF`, `I-aVL`). Therefore, `has_pair` evaluated to `True` for every MI trace in 6-lead. All 668 MI top-hypothesis traces in 6-lead were classified as "Insufficient evidence to assess".
  4. All 246 channel-loss cases were bundle branch blocks: **130 RBBB** and **116 LBBB**.
- **Correction Statement:** The claim that anterior MI drove contradictions in 6-lead is **completely retracted**. The 246 lead-unavailability cases were entirely conduction disturbances caused by the elimination of lead V1.

---

### REV-VAL-03: Retraction of the Septal V1 Grounding Hypothesis
- **Original Claim:** The initial report claimed: *"In 4-lead (I, II, III, V1), contradiction cases plummet from 289 back down to 43 (1.2%)... Preserving V1 provides an anchor in the horizontal septal plane, allowing the model to ground right-ventricular and septal conduction features. This explains the previously documented finding where 4-lead macro-AUROC (0.7598) exceeds 6-lead macro-AUROC (0.7410)."*
- **Audited Reality:**
  1. The genuine physiological contradiction count did **not** plummet; it was **43 in 6-lead** and **43 in 4-lead** (strictly identical).
  2. In 4-lead, lead V1 is present in the lead mask. Consequently, RBBB (144 traces) and LBBB (126 traces) had `disease_leads_ok == True`.
  3. Because contradiction rules for bundle branch blocks do not exist in the code, these traces were assigned to **"Insufficient evidence to assess"**. Not a single trace became "Consistent with available evidence".
  4. The difference in reported contradiction counts (289 vs. 43) was entirely a classification movement between "Contradictory" and "Insufficient", not clinical grounding.
- **Correction Statement:** The assertion that V1 retention established clinical grounding explaining AUROC superiority is **retracted**. It must be documented strictly as an **untested hypothesis**.

---

### REV-VAL-04: Asymmetric Disease Feasibility Checks
- **Original Claim:** The initial report treated the trace consistency classifications as a uniform assessment across the target disease space.
- **Audited Reality:**
  - The classification code applies ad-hoc lead feasibility checks selectively to **RBBB**, **LBBB** (requiring V1), and **MI** (requiring a contiguous pair), shunting them into "Contradictory" when leads are missing.
  - In contrast, **NORM**, **AFIB**, and **1AVB** are subjected to zero lead feasibility checks. Even on a single lead (Lead II), predicting NORM is classified as "Insufficient evidence to assess", despite claiming an entirely normal heart from a 1-lead rhythm strip.
  - When attention queries masked leads, NORM, AFIB, and 1AVB silently remain in "Insufficient evidence to assess", whereas SBRAD and STACH are demoted from "Consistent".
- **Correction Statement:** The existing classifier evaluates diseases asymmetrically. It does **not** provide a uniform or comprehensive evaluation across the eight target diseases.

---

### REV-VAL-05: Authoritative Lead Configuration Descriptions
- **Original Error:** Section 3.2 of the initial generation report incorrectly described the canonical lead masks for 10-lead, 8-lead, and 2-lead.
- **Authoritative Source Verification:** Audited against the frozen Module 2 specification ([`runs/module2_v1/reports/module2_variant_config.json`](file:///e:/ECG/runs/module2_v1/reports/module2_variant_config.json)) and Module 4 loader ([`src/reasoning/multi_cohort_loader.py:LEAD_CONFIG_MASKS`](file:///e:/ECG/src/reasoning/multi_cohort_loader.py#L40-L48)):

| Configuration | Active Count | Canonical Active Leads | Mask Vector $[I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]$ | Masked Leads | Corrected Narrative Description |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **12-lead** | 12 | I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]` | *(None)* | Standard diagnostic 12-lead ECG. |
| **10-lead** | 10 | I, II, aVR, aVL, V1, V2, V3, V4, V5, V6 | `[1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1]` | **III, aVF** | **Omits III and aVF** (derivable via Einthoven/Goldberger; report erroneously claimed V5/V6). |
| **8-lead** | 8 | I, II, V1, V2, V3, V4, V5, V6 | `[1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]` | **III, aVR, aVL, aVF** | **Omits augmented leads and III** (standard 8 independent channels; report erroneously claimed V5/V6). |
| **6-lead** | 6 | I, II, III, aVR, aVL, aVF | `[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]` | V1, V2, V3, V4, V5, V6 | Frontal limb leads only (eliminates all precordial leads). |
| **4-lead** | 4 | I, II, III, V1 | `[1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]` | aVR, aVL, aVF, V2, V3, V4, V5, V6 | Einthoven limb leads plus exploratory septal lead V1. |
| **2-lead** | 2 | **II, V1** | `[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]` | **I, III, aVR, aVL, aVF, V2-V6** | **Dual-channel telemetry (Lead II + Lead V1)** (report erroneously claimed I and II). |
| **1-lead** | 1 | II | `[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]` | I, III, aVR, aVL, aVF, V1-V6 | Single-lead rhythm strip / ambulatory telemetry. |

All physical JSON trace files already adhered to these canonical masks; the error was strictly textual in the previous report's narrative table.

---

### REV-VAL-06: 6-Lead Breakdown Accounting Reconciliation
- **Original Error:** The breakdown table in Section 6.2 of the initial report summed to 287 traces ($18 + 25 + 244$), omitting 2 traces from the column accounting against the reported total of 289.
- **Reconciliation Audit:** An audit of all 289 contradictory traces in `traces_validation_6-lead.json` revealed:
  1. **Physiological HR Contradiction Only (all attended tokens in active limb leads):** 18 traces
  2. **Physiological HR Contradiction + Unobservable Token Attended:** 25 traces
     *(Subtotal Physiological HR Contradictions: $18 + 25 = \mathbf{43}$ traces)*
  3. **Lead Unavailability (RBBB/LBBB missing V1) + Unobservable Token Attended:** 244 traces
  4. **Lead Unavailability Only (RBBB/LBBB missing V1, but attended tokens in limb leads):** 2 traces
     *(Subtotal Lead Unavailability Cases: $244 + 2 = \mathbf{246}$ traces)*
- **Reconciled Equation:**
  $$\text{Total Contradictory Reported} = 18 + 25 + 244 + 2 = 289$$
- **Correction Statement:** The 2 previously omitted traces belonged to category (4) ("Lead Unavailability Only"). The 289 traces are now fully and transparently reconciled without discrepancies.

---

### REV-VAL-07: Patient Unit Governance & Trace Provenance
- **Affirmed Trace Coverage:** Exactly **3,688 validation records** evaluated across **7 canonical configurations** = **25,816 total configuration evaluations**, with **0 generation failures**.
- **Patient Unit Separation:**
  The 3,688 validation records represent **3,394 unique patient clusters**:
  - **PTB-XL:** 2,179 records across 1,885 unique patients
  - **Chapman-Shaoxing:** 1,014 records across 1,014 unique patients
  - **MIMIC-IV-ECG:** 495 records across 495 unique patients
- **Governance Mandate:** The 25,816 evaluations represent 7 simulated sensor montages evaluated on the same 3,688 recordings (3,394 unique patients). They must **never** be described or reported as 25,816 independent patients.

---

## 5. Summary: Original Claims vs. Corrected Interpretations

| Feature / Domain | Original Claim in Generation Report | Corrected Authoritative Interpretation | Remediation Status |
| :--- | :--- | :--- | :---: |
| **Contradiction Rate in Degraded Leads** | 6-lead: 7.8%, 2-lead: 18.9%, 1-lead: 22.7% represented model reasoning contradictions. | Spikes were caused by classifying missing leads as contradictions. True physiological contradictions remain flat (~1.0%–1.5%). | **CORRECTED** |
| **6-Lead Contradiction Cause** | Driven by anterior MI and conduction block claims querying unobservable precordial leads. | 0 cases of anterior MI existed. All 246 channel-loss cases were bundle branch blocks (130 RBBB, 116 LBBB). | **RETRACTED & CORRECTED** |
| **4-Lead V1 Grounding** | V1 retention anchored septal grounding, reducing contradictions and explaining superior AUROC. | Untested hypothesis. Physiological contradictions were identical (43 vs. 43). The drop was a classification artifact. | **RETRACTED & CORRECTED** |
| **Disease Scope Uniformity** | Implied uniform evidence consistency evaluation across all eight candidate diseases. | Highly asymmetric: RBBB/LBBB/MI penalize missing leads as contradictions; NORM/AFIB/1AVB ignore missing leads. | **CORRECTED** |
| **10-Lead Mask** | Stated that V5 and V6 were masked. | Canonical mask preserves V5/V6; masks III and aVF (derivable frontal channels). | **CORRECTED** |
| **8-Lead Mask** | Stated that V5, V6, III, and aVF were masked. | Canonical mask preserves all precordials V1–V6; masks augmented leads and III. | **CORRECTED** |
| **2-Lead Montage** | Stated that active leads were I and II. | Canonical mask defines active leads as II and V1 (rhythm + septal QRS view). | **CORRECTED** |
| **6-Lead Breakdown Table** | Summed to 287 traces, leaving 2 traces unaccounted for. | Fully reconciled: 18 HR-only, 25 HR+unobservable, 244 unobservable+lead unavail, 2 lead unavail only = 289. | **RECONCILED** |
| **Patient Accounting** | 25,816 evaluations. | Affirmed 25,816 evaluations across 3,688 records from 3,394 unique patient clusters; never 25,816 patients. | **AFFIRMED** |

---

## 6. Corrected Interpretation of Performance and Trace Metrics

1. **Decoupling Trace Quality from Predictive Accuracy:**
   The classification defects identified in `classify_trace_quality()` affect the post-hoc interpretation of reasoning traces. They do **not** invalidate the diagnostic prediction metrics (e.g. macro-AUROC, macro-F1, calibration curves) recorded in `runs/module4_v3/metrics/validation_predictions_seed42.csv`. Predictive accuracy is driven by the frozen weights of Module 4 v3 and the underlying Module 3 latents, whereas trace classifications are descriptive rule-based labels.
2. **Epistemological Boundaries of Transformer Attention:**
   Cross-attention weights represent mathematical routing correlations between token representations within the transformer's latent space. They do **not** establish mechanistic faithfulness or proof of internal clinical reasoning. Generating a natural-language summary from attention weights reflects post-hoc attribution, not a validated causal trace.
3. **Status of Aggregate Rates:**
   Until a newly approved pipeline reclassifies individual trace records into the four-tier taxonomy, reported aggregate contradiction rates (7.8%, 18.9%, 22.7%) must be explicitly qualified as channel-loss artifacts, while the true physiological contradiction rate is cited as ~1.0% to 1.5%.

---

## 7. Final Verification Dashboard & Stop Condition

### 7.1 Compliance Verification Checklist
- [x] **Zero Test Contamination:** Zero test records accessed, loaded, or evaluated. Quarantine remained fail-closed.
- [x] **Trace File Immutability:** All 25,816 validation trace JSON dossiers in `runs/module4_v3/validation_traces/` remain identical in byte size and content.
- [x] **Code Immutability:** No changes were made to `scripts/18_generate_validation_reasoning_traces.py`, `src/reasoning/trace_generator.py`, or `src/reasoning/context_builder.py`.
- [x] **Weight & Contract Preservation:** All checkpoints, calibration parameters, split manifests, and Module 1–3 artifacts matched their frozen hashes.
- [x] **Deliverables Produced:**
  - Corrected Findings CSV: [`runs/module4_v3/reports/validation_trace_corrected_findings.csv`](file:///e:/ECG/runs/module4_v3/reports/validation_trace_corrected_findings.csv)
  - Corrected Documentation Report: [`runs/module4_v3/reports/validation_trace_documentation_correction.md`](file:///e:/ECG/runs/module4_v3/reports/validation_trace_documentation_correction.md)
- [x] **Previous Audits Preserved:** The original generation report and consistency review remain unaltered in `runs/module4_v3/reports/`.

### 7.2 Stop Condition
With the completion and physical verification of this documentation remediation report and its accompanying corrected findings CSV, all requirements of the task have been fulfilled. 

In strict adherence to instructions:
- No retraining or fine-tuning has occurred.
- No Module 4 v4 implementation has been initiated.
- No baseline comparisons or test evaluations have been conducted.

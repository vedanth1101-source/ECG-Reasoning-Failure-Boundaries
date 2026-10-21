# Module 4 — Downstream Evaluation Protocol & Grounding Failure Specification

**Document Version:** 2.0 (Post-Stage 0 Methodological Revision)  
**Date:** 2026-10-14  
**Project:** Reasoning Failure Boundaries in ECG Foundation Models Under Progressive Physiological Information Loss  
**Reference Document:** `PROJECT_PHASES_SPECIFICATION.md` (Phases 4, 5, and 6)  
**Evidence Configuration:** `reports/evidence_observability_config.json`  
**Status:** **PRE-REGISTERED & FROZEN PROTOCOL (Strict Pre-Test Quarantine)**  

---

## 1. Core Research Objective & Scientific Mandate

The primary research objective of this investigation is:
> **To evaluate how clinical reasoning and evidence-grounding degrade in ECG foundation models as physiological information is progressively removed across controlled lead subsets, and to identify the empirical Reasoning Failure Boundary (RFB) where clinical reasoning collapses into ungrounded speculation.**

### Diagnostic vs. Reasoning Decoupling Principle
Standard machine learning benchmarks evaluate models strictly by end-to-end diagnostic accuracy (e.g., AUROC or Macro-F1). However, in high-stakes clinical electrocardiography:
1. A foundation model can achieve a correct diagnostic classification purely by exploiting non-causal statistical dataset priors, demographic correlations, or global rhythm regularities (e.g., predicting *Myocardial Infarction* in an elderly patient based on heart rate or non-specific repolarization changes), while its internal reasoning attributes the diagnosis to absent or hallucinated anatomical waveforms.
2. In degraded lead environments (such as 2-lead inpatient telemetry or single-lead ambulatory patches), contiguous anatomical lead sets required to verify localized pathologies (such as inferior ST elevation in II, III, aVF, or anterior ST elevation in V1–V4) are physically absent.
3. Therefore, this project treats **clinical reasoning grounding** as the primary object of study, evaluating whether the model's generated claims and cross-attention weights remain anchored in physically observable leads.

---

## 2. Representation Drift Non-Monotonicity Invariant

In strict accordance with the **Module 3 Post-Freeze Audit Addendum** (Finding F-09 in `runs/module3_v1/reports/module3_post_freeze_findings.csv`), all downstream documentation, analysis, and interpretation adhere to the following invariant:

### Empirical Evidence
Mean cosine similarities between degraded lead representations and the full 12-lead reference across all 36,888 records are:
- **12-lead (C12_FULL):** $1.000000$ (Reference)
- **10-lead (C10_PREC_LIMB):** $0.998051$
- **8-lead (C8_INDEPENDENT):** $0.997095$
- **6-lead (C6_LIMB_6):** $0.877548$
- **4-lead (C4_EINTHOVEN_SEPTAL):** **$0.904078$** *(Higher than 6-lead by $+0.026529$)*
- **2-lead (C2_TELEMETRY):** $0.876178$
- **1-lead (C1_SINGLE_LEAD):** $0.814373$

### Theoretical and Anatomical Justification
The 7-step lead reduction ladder is **not a chain of strictly nested spatial subspaces**.
- The 6-lead configuration (C6_LIMB_6) retains all frontal hexaxial limb leads (I, II, III, aVR, aVL, aVF) but completely eliminates all six horizontal precordial chest leads (V1–V6), entirely removing the anterior-posterior electrical projection of the cardiac vector.
- The 4-lead configuration (C4_EINTHOVEN_SEPTAL) retains Einthoven's triangle (I, II, III) and a single precordial electrode at **V1**. Retaining V1 preserves horizontal septal vector information that is totally missing in the 6-lead configuration.
- **Mandate:** No report, manuscript section, or code comment shall claim that foundation model representation drift is "globally monotonic." Drift reflects anatomical subspace orientation and spatial lead geometry.

---

## 3. The Grounding Verification Logic Gate (Phase 5 Protocol)

Every reasoning trace emitted by the model is decomposed into individual verifiable clinical claims:
$$\mathcal{T} = \{ C_1, C_2, \dots, C_K \}$$
where each claim asserts a clinical finding $f_k$ supporting or refuting a candidate disease $H_i$.

### Verification Logic Rules
For each claim $C_k$, let $\mathcal{L}_{\text{req}}(f_k)$ be the clinically mandatory leads required to establish $f_k$ according to AHA/ACC guidelines (`reports/evidence_observability_config.json`), and let $\mathbf{m}_l \in \{0, 1\}^{12}$ be the active lead mask at degradation level $l$:

```
FOR each claim C_k in Trace:
    IF (any lead in L_req(f_k) has m_l[lead] == 0):
        ASSIGN Verdict = "UNAVAILABLE"
        RATIONALE: "Required sensor lead is masked. Claim relies on unobservable sensor data."

    ELSE:
        // All required leads are present in active mask
        IF (Measured signal features satisfy clinical rule threshold theta(f_k)):
            ASSIGN Verdict = "SUPPORTED"
            RATIONALE: "Clinical finding is physically confirmed in observed ECG waveform."
        ELSE:
            ASSIGN Verdict = "CONTRADICTED"
            RATIONALE: "Required leads are observable, but physical waveform refutes the claim."
```

### Mutually Exclusive Claim Statuses
| Active Mask Status | Measured Signal Features | Logic Gate Verdict | Epistemic Status |
| :--- | :--- | :---: | :--- |
| $\ge 1$ required lead masked ($\mathbf{m}_l[c] = 0$) | Any value (irrelevant) | **`UNAVAILABLE`** | **Ungrounded Claim / Relying on Masked Sensor** |
| All required leads active ($\mathbf{m}_l[c] = 1$) | Satisfies $\theta(f_k)$ | **`SUPPORTED`** | **Factually Grounded Claim** |
| All required leads active ($\mathbf{m}_l[c] = 1$) | Violates $\theta(f_k)$ | **`CONTRADICTED`** | **Factually Refuted Claim** |

*Critical Distinction:* A claim is NEVER marked `CONTRADICTED` merely because its required leads are absent. Missing sensor data is strictly categorized as `UNAVAILABLE`.

### 3.1 Strict Electrophysiological Evidence Verification for Myocardial Infarction (Decision U-04)
In strict compliance with user ratification of Decision U-04, clinical evidence requirements for Myocardial Infarction are maintained under rigid, predefined electrophysiological standards and are **never relaxed** to force a binary supported/contradicted verdict:

1. **Approved Anatomical Territories:**
   Anchored directly in `src/reasoning/clinical_taxonomy.py` (lines 24–33) and AHA/ACCF/HRS guidelines:
   - **Inferior Territory:** Leads `[II, III, aVF]`
   - **Septal Territory:** Leads `[V1, V2]`
   - **Anterior Territory:** Leads `[V3, V4]`
   - **Anteroseptal Territory:** Leads `[V1, V2, V3, V4]`
   - **Lateral Territory:** Leads `[I, aVL, V5, V6]`
   - **High Lateral Territory:** Leads `[I, aVL]`
   - **Extensive Anterior Territory:** Leads `[I, aVL, V1, V2, V3, V4, V5, V6]`

2. **Rigid Contiguous-Lead Rule:**
   - As codified in `reports/evidence_observability_config.json` (`CLAIM_MYOCARDIAL_INFARCTION`) and `PROJECT_PHASES_SPECIFICATION.md` (§4.2):
     $$\text{ST Elevation} \ge 0.10\,\text{mV}\;(1.0\,\text{mm}) \quad \text{in} \ge 2 \text{ contiguous anatomical leads}$$
     or pathological Q waves (duration $\ge 40\,\text{ms}$ or amplitude $\ge 25\%$ of R wave).
   - A claim asserting localized ST elevation or acute infarction in territory $\mathcal{T}$ requires that at least 2 contiguous leads from $\mathcal{T}$ be present in the active lead mask $\mathbf{m}_l$.
   - **Strict Non-Relaxation Mandate:** If fewer than 2 contiguous leads from territory $\mathcal{T}$ are active in $\mathbf{m}_l$, the evidence requirement is physically unobservable. The claim verdict is strictly and irrevocably assigned **`UNAVAILABLE`**. It is NEVER relaxed to single-lead criteria or assumed from non-contiguous leads.

3. **Behavior Across the 7 Lead Configurations:**
   - **12-lead (C12), 10-lead (C10), 8-lead (C8):** Contiguous pairs exist across inferior, anterior, septal, and lateral territories.
   - **6-lead (C6_LIMB_6):** Inferior (`[II, III, aVF]`) and high lateral (`[I, aVL]`) contiguous pairs exist. All precordial territories (V1–V6) are completely absent; anterior and septal claims are strictly `UNAVAILABLE`.
   - **4-lead (C4_EINTHOVEN_SEPTAL):** Einthoven leads `[I, II, III]` and `V1` active. Leads II and III form an inferior pair. Precordial contiguous pairs do not exist (only single lead V1); anterior and lateral claims are strictly `UNAVAILABLE`.
   - **2-lead (C2_TELEMETRY: II, V1):** Zero contiguous pairs exist in *any* anatomical territory. Any claim asserting localized ST-elevation MI is strictly `UNAVAILABLE` ($100\%$ ungrounded).
   - **1-lead (C1_SINGLE_LEAD: II):** Zero contiguous pairs exist. All localized MI claims are strictly `UNAVAILABLE`.
   - This strict electrophysiological gate prevents spurious grounding and correctly exposes whether the foundation model hallucinated ischemic features from single leads.

---

## 4. Formal Mathematical Metrics

### 4.1 Unit of Analysis & Counting Rules
- **One Generated Claim ($C_{r, k}$):** An assertion emitted in the reasoning trace for record $r$ declaring that finding $f_k$ supports or refutes candidate disease $H_i$. Claims are deduplicated per record at the finding/lead level to prevent repeated attention tokens from inflating counts.
- **One Required Evidence Criterion ($e_{i, j}$):** An atomic clinical condition specified in `evidence_observability_config.json` for disease $H_i$.
- **One Observable Criterion:** A required evidence criterion whose specified leads $\mathcal{L}_{\text{req}}(e)$ are all active in mask $\mathbf{m}_l$.
- **Mutually Exclusive Partition:** Every evaluated claim receives exactly one verdict:
  $$N_{\text{total\_claims}, l} = N_{\text{supported}, l} + N_{\text{contradicted}, l} + N_{\text{unavailable}, l}$$

### 4.2 The Five Core Grounding Metrics (At Lead Level $l$)

1. **Evidence Observability ($EO_l$):**
   $$EO_l = \frac{N_{\text{observable}, l}}{N_{\text{required}, l}} \in [0.0, 1.0]$$
   - *Interpretation:* Proportion of clinically mandatory evidence elements for a disease physically observable under active lead mask $\mathbf{m}_l$. $EO$ is a structural property of the lead subset and the diagnostic criteria; it does not measure disease presence.
   - *Edge Case:* If $N_{\text{required}} = 0$, $EO$ is `NOT_APPLICABLE`.

2. **Grounding Quality ($GQ_l$):**
   $$GQ_l = \frac{N_{\text{supported}, l}}{N_{\text{supported}, l} + N_{\text{contradicted}, l}} \in [0.0, 1.0]$$
   - *Interpretation:* The precision of verified diagnostic claims among all claims whose required leads are observable. Measures whether the model's assertions are factually corroborated when the sensors are present.
   - *Edge Case (Zero Denominator):* If $N_{\text{supported}, l} + N_{\text{contradicted}, l} = 0$ (meaning 100% of claims are `UNAVAILABLE`), $GQ_l$ is mathematically **`NOT_ESTIMABLE` / `UNDEFINED`**. It is never arbitrarily set to 0.0 or 1.0.

3. **True Grounding Ratio ($TGR_l$ — The Unified Score):**
   $$TGR_l = GQ_l \times EO_l = \frac{N_{\text{supported}, l} \cdot N_{\text{observable}, l}}{(N_{\text{supported}, l} + N_{\text{contradicted}, l}) \cdot N_{\text{required}, l}} \in [0.0, 1.0]$$
   - *Interpretation:* Unified metric penalizing both false assertions and missing anatomical visibility.
   - *Edge Case:* When $EO_l = 0$ (all required leads masked), $TGR_l = 0.0$ by continuous extension.

4. **Unavailable Ratio ($UR_l$ — The Hallucination Index):**
   $$UR_l = \frac{N_{\text{unavailable}, l}}{N_{\text{total\_claims}, l}} \in [0.0, 1.0]$$
   - *Interpretation:* Proportion of generated claims that rely on masked leads. This directly indexes model speculation in the absence of sensory input.
   - *Edge Case:* If $N_{\text{total\_claims}, l} = 0$, $UR_l$ is `NOT_ESTIMABLE`.

5. **Grounded Support Ratio ($GSR_l$):**
   $$GSR_l = \frac{N_{\text{supported}, l}}{N_{\text{total\_claims}, l}} \in [0.0, 1.0]$$
   - *Interpretation:* Fraction of total generated claims that are strictly verified as `SUPPORTED`.
   - *Relationship:* Since claims partition into three states: $GSR_l + \frac{N_{\text{contradicted}, l}}{N_{\text{total\_claims}, l}} + UR_l = 1.0$.

### 4.3 Aggregation Protocol
- **Micro-Aggregation (Primary):** Sum counts $N_{\text{supported}}, N_{\text{contradicted}}, N_{\text{unavailable}}$ across all test records at level $l$, then calculate ratios.
- **Macro-Aggregation (Secondary):** Calculate ratios per record and compute sample means and standard errors across records with valid denominators.
- **Confidence Intervals:** 95% empirical bootstrap confidence intervals (1,000 resamples clustered by patient).

---

## 5. Corrected Operational Failure Boundaries

### 5.1 The Reasoning Failure Boundary (RFB)
The experimental sequence is the ordered progression:
$$\mathcal{S} = \langle c_1, c_2, c_3, c_4, c_5, c_6, c_7 \rangle = \langle \text{12-lead}, \text{10-lead}, \text{8-lead}, \text{6-lead}, \text{4-lead}, \text{2-lead}, \text{1-lead} \rangle$$
where index $k \in \{1, 2, 3, 4, 5, 6, 7\}$.

#### Operational Definition
The Reasoning Failure Boundary identifies the transition along $\mathcal{S}$ where grounding quality drops below safety threshold $\tau$ (primary $\tau = 0.60$; sensitivity $\tau = 0.70$ and $\tau = 0.50$):

1. **First Failure Configuration ($c_{\text{RFB\_fail}}$):**
   $$c_{\text{RFB\_fail}}(\tau) = \text{first } c_k \in \mathcal{S} \quad \text{such that} \quad GQ(c_k) < \tau$$
2. **Last Acceptable Configuration ($c_{\text{RFB\_pass}}$):**
   The configuration immediately preceding the first failure ($c_{k-1}$).
3. **Canonical Reporting Standard:**
   - Primary report: The **Failure Transition Pair** $(c_{k-1} \to c_k)$ (e.g., $8\text{-lead} \to 6\text{-lead}$).
   - Discrete reporting: When cited as a single configuration, the **First Failure Configuration** $c_{\text{RFB\_fail}}$ is reported, as it designates entry into the clinically ungrounded regime.

#### Resolution of Boundary Cases:
- **Non-Monotonicity & Recovery:**
  If $GQ$ drops below $\tau$ at $c_k$ but recovers to $\ge \tau$ at a subsequent step $c_j$ ($j > k$, e.g. 4-lead recovering over 6-lead due to V1 retention):
  - Do NOT force monotonicity.
  - Report the **Initial Failure Transition** at $c_k$.
  - Set flag `RECOVERY_DETECTED = True` and document the recovery configuration $c_j$.
  - Report the **Sustained Failure Boundary** (the configuration after which $GQ$ never recovers).
  - Always publish the complete 7-element vector $\mathbf{GQ} = [GQ_{12}, GQ_{10}, GQ_8, GQ_6, GQ_4, GQ_2, GQ_1]$.
- **All Configurations Pass ($GQ(c_k) \ge \tau \;\forall k$):**
  Report: `RFB = ROBUST_ACROSS_ALL_LEVELS (< 1-lead)`.
- **Baseline Fails ($GQ(\text{12-lead}) < \tau$):**
  Report: `RFB = FAILED_AT_BASELINE (> 12-lead)`.
- **Undefined Values:**
  If $GQ(c_k)$ is `NOT_ESTIMABLE` at step $k$, the boundary search marks `INDETERMINATE_AT_STEP_k` and reports the lowest estimable configuration.

---

### 5.2 The Diagnostic Failure Boundary (DFB)

#### Operational Definition
Let $\text{MacroF1}_{12}$ be the full 12-lead reference performance on the test set.
The Retained Performance Ratio (RPR) at configuration $c_k$ is:
$$\text{RPR}(c_k) = \frac{\text{MacroF1}(c_k)}{\text{MacroF1}_{12}}$$
The Diagnostic Failure Boundary identifies the transition along $\mathcal{S}$ where retained performance drops below retention threshold $\gamma$ (primary $\gamma = 0.80$, i.e., $80\%$ retention):

1. **First Diagnostic Failure Configuration ($c_{\text{DFB\_fail}}$):**
   $$c_{\text{DFB\_fail}}(\gamma) = \text{first } c_k \in \mathcal{S} \quad \text{such that} \quad \text{RPR}(c_k) < \gamma$$
2. **Canonical Reporting Standard:**
   - Transition Pair: $(c_{k-1} \to c_k)$.
   - Discrete Configuration: First Failure Configuration $c_{\text{DFB\_fail}}$.

#### Resolution of Boundary Cases:
- **Zero or Undefined Reference ($\text{MacroF1}_{12} = 0$):**
  Report: `DFB = NOT_ESTIMABLE (Zero Baseline Macro-F1)`.
- **Non-Monotonicity:**
  If $\text{RPR}$ recovers at a lower lead configuration, flag `DIAGNOSTIC_RECOVERY_DETECTED = True`, report the initial failure transition, the recovery point, and the sustained failure transition.
- **All Configurations Retain $\ge 80\%$:**
  Report: `DFB = ROBUST_ACROSS_ALL_LEVELS (< 1-lead)`.
- **No Reduced-Lead Configuration Retains $80\%$:**
  Report: `DFB = 10-lead (Transition 12-lead -> 10-lead)`.
- **Uncertainty Reporting:**
  Report 95% bootstrap confidence intervals for $\text{MacroF1}(c_k)$ and $\text{RPR}(c_k)$.

---

### 5.3 Clarified Diagnostic-Reasoning Decoupling Gap

Lead configurations are discrete categorical conditions. To evaluate decoupling rigorously without numerical distortion, we define the gap using two complementary representations:

#### A. Ordinal Degradation Step Gap ($\Delta_{\text{steps}}$)
Let step index $k \in \{0, 1, 2, 3, 4, 5, 6\}$ correspond to $\langle 12, 10, 8, 6, 4, 2, 1 \rangle$.
Let $k_{\text{DFB}}$ and $k_{\text{RFB}}$ be the step indices of the first failure configurations:
$$\Delta_{\text{decoupling}}^{\text{steps}} = k_{\text{DFB}} - k_{\text{RFB}}$$

#### B. Physical Lead Count Gap ($\Delta_{\text{leads}}$)
Let $l_{\text{RFB}}, l_{\text{DFB}} \in \{12, 10, 8, 6, 4, 2, 1\}$ be the numerical lead counts of the first failure configurations:
$$\Delta_{\text{decoupling}}^{\text{leads}} = l_{\text{RFB}} - l_{\text{DFB}}$$

#### C. Sign Convention & Clinical Interpretation

| Decoupling State | Step Gap | Lead Gap | Clinical Interpretation |
| :--- | :---: | :---: | :--- |
| **Positive Gap ($\Delta > 0$)** | $k_{\text{DFB}} > k_{\text{RFB}}$ | $l_{\text{RFB}} > l_{\text{DFB}}$ | **Diagnostic Persistence Beyond Reasoning Failure (Dangerous Failure Regime).** Reasoning collapses earlier in the lead-removal ladder (at higher lead count), but diagnostic classification survives further down the ladder. The model makes accurate diagnostic predictions using ungrounded or absent evidence. |
| **Negative Gap ($\Delta < 0$)** | $k_{\text{DFB}} < k_{\text{RFB}}$ | $l_{\text{RFB}} < l_{\text{DFB}}$ | **Diagnostic Fragility despite Grounded Reasoning.** Diagnostic performance collapses while the few claims the model makes remain grounded. |
| **Zero Gap ($\Delta = 0$)** | $k_{\text{DFB}} = k_{\text{RFB}}$ | $l_{\text{RFB}} = l_{\text{DFB}}$ | **Synchronous Failure.** Diagnostic utility and reasoning grounding collapse at the exact same configuration. |
| **Undefined Gap** | `UNDEFINED` | `UNDEFINED` | Occurs when either RFB or DFB is `ROBUST` or `FAILED_AT_BASELINE`. Reported as a categorical descriptor (e.g. `RFB=6-lead, DFB=ROBUST -> PERSISTENT_DIAGNOSTIC_BEYOND_REASONING`). |

*Scientific Hypothesis Note:* The hypothesis that $\Delta_{\text{decoupling}} > 0$ under physiological information loss is an empirical hypothesis to be tested, not a presumed fact.

---

## 6. Pre-Registered Experimental Baselines

To contextualize the reasoning model, four baselines are pre-registered:

| Baseline ID | Architecture & Inputs | Output Type | Evaluated Metrics | Primary Purpose in Study |
| :--- | :--- | :---: | :--- | :--- |
| **B-1: FM Linear Probe** | Frozen ECG-FM ($Z_{\text{ECG}} \in \mathbb{R}^{768}$) $\to$ direct linear head | Diagnostic Logits Only | Macro AUROC, Macro-F1, Brier Score, ECE, DFB | Quantifies raw foundation representation power without structured reasoning. |
| **B-2: Clinical Rule Engine** | Deterministic AHA/ACC criteria on active leads $\mathbf{m}_l$ | Rule Verdicts Only | Diagnostic F1, EO, GQ, TGR, RFB, DFB | Determines heuristic baseline achievable from pure symbolic rules without representations. |
| **B-3: Control Path ($ECG_{\text{only}}$)** | Multimodal Control Reasoning Transformer ($Z_{\text{ECG}} + O_l$) | Logits + Structured Reasoning Trace | All diagnostic and grounding metrics, RFB, DFB, $\Delta_{\text{decoupling}}$, AURDC | Primary control model: evaluates reasoning without external ontologies. |
| **B-4: Experimental Path ($ECG + CKP$)** | Dual-Path Reasoning Transformer with SNOMED-CT priors | Logits + Grounded Reasoning Trace | All diagnostic and grounding metrics, RFB, DFB, $\Delta_{\text{decoupling}}$, AURDC | Evaluates whether ontological knowledge delays the Reasoning Failure Boundary ($\text{RFB}_{\text{exp}} < \text{RFB}_{\text{ctrl}}$). |

*Strict Reporting Rule:* Grounding metrics ($EO, GQ, TGR, UR, GSR, \text{RFB}$) cannot be computed for B-1 because it outputs no reasoning trace. No synthetic traces will be fabricated for B-1.

---

## 7. Multi-Lead Curve Metrics (AURDC and AUTDC)

### 7.1 Area Under the Reasoning Degradation Curve (AURDC)
The aggregate grounding resilience across the degradation sequence is computed via trapezoidal integration:
$$\text{AURDC} = \frac{1}{12 - 1} \sum_{k=1}^{6} \frac{GQ(c_k) + GQ(c_{k+1})}{2} \cdot (l_k - l_{k+1})$$
where:
- Lead counts: $l = [12, 10, 8, 6, 4, 2, 1]$.
- Step widths: $\Delta l_k = l_k - l_{k+1} \in [2, 2, 2, 2, 2, 1]$.
- Normalization: Denominator is $12 - 1 = 11$, scaling $\text{AURDC} \in [0.0, 1.0]$.
- **Operational Convention:** The numerical lead-count differences are an explicit integration convention over the experimental evaluation points. They do not imply that the configurations form a continuous or nested Euclidean subspace.

### 7.2 Area Under the True Grounding Curve (AUTDC)
Calculated identically using $TGR(c_k)$:
$$\text{AUTDC} = \frac{1}{12 - 1} \sum_{k=1}^{6} \frac{TGR(c_k) + TGR(c_{k+1})}{2} \cdot (l_k - l_{k+1})$$

### 7.3 Handling of Undefined Points
If $GQ(c_k)$ is `NOT_ESTIMABLE` at any configuration:
- Compute the partial integral over the well-defined sub-interval (e.g. $[12, 2]$) and normalize by $(12 - 2) = 10$.
- Document the sub-interval explicitly. Do not impute arbitrary values.

---

## 8. Formal Test-Set Quarantine and Data-Access Policy

```
+---------------------------------------------------------------------------------------------------+
| 1. TRAINING SPLIT (runs/module1_v2/ split == 'train', 25,825 records)                              |
|    - Permitted: Model parameter optimization, loss minimization, backward pass.                   |
|    - Prohibited: Evaluation metrics, threshold tuning, temperature calibration.                    |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| 2. VALIDATION SPLIT (runs/module1_v2/ split == 'validation', 3,688 records)                       |
|    - Permitted: Model selection, early stopping, hyperparameter tuning, temperature calibration,  |
|                 decision threshold optimization (tau, gamma).                                     |
|    - Prohibited: Gradient updates, backward passes.                                               |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| 3. PRE-TEST QUARANTINE GATE (Formal Checkpoint & Protocol Freeze)                                  |
|    - All model weights, calibration temperatures, thresholds, and evaluation code locked.         |
|    - SHA-256 manifests generated and frozen. Gate approval required before test access.           |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| 4. TEST SPLIT (runs/module1_v2/ split == 'test', 7,375 records x 7 variants = 51,625 arrays)      |
|    - Permitted: Strictly one-pass forward inference in eval mode (torch.no_grad).                 |
|    - Prohibited: Retraining, hyperparameter tuning, post-hoc calibration, iterative re-testing.   |
|    - Output Directory: runs/module4_v3/evaluation/                                                |
+---------------------------------------------------------------------------------------------------+
```

### Policy Rules
1. **Zero Test Contamination:** No test records may be loaded during model training, hyperparameter search, or temperature calibration.
2. **Single-Pass Execution:** Test set inference occurs exactly once after the experimental pipeline is locked.
3. **Immutability of Post-Test Code:** If any defect is discovered after test evaluation, the results cannot be overwritten. A new versioned run (`runs/module4_v4/`) must be declared and documented.

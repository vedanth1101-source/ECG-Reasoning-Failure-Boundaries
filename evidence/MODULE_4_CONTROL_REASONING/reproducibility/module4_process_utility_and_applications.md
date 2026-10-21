# Module 4 — Control-Path Reasoning: Process Utility, Design Rationale, and Research Applications

**Module Identifier:** `MODULE_4_CONTROL_REASONING`  
**Execution Phase:** Diagnostic Classification, Evidence Grounding, and Reasoning Audit  
**Execution Date:** 2026-10-16  

---

## 1. Problem Statement and Research Purpose

### 1.1 The Black-Box Failure Problem in Reduced-Lead AI
When standard 12-lead deep learning models are deployed in reduced-lead settings (such as 1-lead smartwatches or 2-lead ambulatory patches), they often continue to output high-confidence predictions without indicating whether the necessary anatomical projections were actually observed. For example, a model might predict "Anterior Myocardial Infarction" or "Right Bundle Branch Block" based on spurious correlations in Lead II, even when precordial leads V1–V4 are physically absent.

### 1.2 The Research Role of Module 4
Module 4 integrates the outputs of Modules 1, 2, and 3 into an end-to-end evaluative framework that couples diagnostic classification with auditable reasoning traces. It determines:
1. At what precise point along the lead degradation continuum ($12 \rightarrow 10 \rightarrow 8 \rightarrow 6 \rightarrow 4 \rightarrow 2 \rightarrow 1$) diagnostic discrimination fails for each specific disease.
2. Whether the model's textual explanations and evidence citations respect the anatomical constraints of the active lead channels.

---

## 2. Design Decisions and Methodological Rationale

### 2.1 Shared Unified Architecture Across All 7 Configurations
- **Decision:** Rather than training seven separate models (one for each lead subset), a single shared neural network processes all configurations, conditioned on the binary availability mask $\mathbf{m} \in \{0, 1\}^{12}$.
- **Rationale:** A shared network evaluates whether a single clinical reasoning system can dynamically re-weight its internal features based on sensor availability, mirroring an intelligent clinical decision-support assistant operating across heterogeneous hardware.

### 2.2 Masked Binary Cross-Entropy Loss
- **Decision:** The multi-label loss function operates strictly on observed disease indicators ($o_{i,d}=1$).
- **Rationale:** Because cohorts have differing annotation coverage (e.g., Chapman does not annotate MI; PTB-XL lacks certain arrhythmias), setting unobserved labels to zero would inject false negative signals. The masked loss preserves gradient purity.

### 2.3 Four-Tier Evidence Taxonomy for Reasoning Audits
- **Decision:** Trace evaluations were categorized into four mutually exclusive tiers:
  1. *Consistent with available evidence*
  2. *Physiologically contradictory* (e.g., claiming bradycardia when HR = 120 bpm)
  3. *Required evidence unavailable* (e.g., asserting V1 morphology when V1 is masked)
  4. *Insufficient evidence to assess*
- **Rationale:** Disentangling channel unavailability (missing sensors) from physiological contradiction (factually impossible assertions) is essential for diagnosing AI failure modes.

---

## 3. Practical and Research Applications

### 3.1 Establishing Empirical Safe Operating Envelopes
Module 4 provides the empirical evidence needed to define clinical operating envelopes for reduced-lead monitoring:
- **Rhythm Conditions (AFIB, SBRAD, STACH):** Maintain strong discrimination (>0.85 AUROC) down to single-lead monitoring (Lead II).
- **Conduction Blocks (RBBB, LBBB):** Require precordial lead V1; performance drops precipitously when transitioning from 4-lead to 2-lead and 1-lead.
- **Myocardial Infarction (MI):** Requires multi-plane limb and precordial coverage; fails under severe degradation.

### 3.2 Auditing AI Faithfulness and Hallucinations
The reasoning trace evaluation framework provides an open-source auditing tool for verifying whether multi-modal medical LLMs and clinical reasoning agents hallucinate findings from absent sensory channels.

---

## 4. End-to-End Pipeline Integration Map

```
┌────────────────────────────────────────────────────────┐
│ MODULE 1: Harmonized Data & Label Universe             │
│ - 36,888 records across PTB-XL, Chapman, MIMIC         │
│ - 8 canonical target diseases with observation masks   │
└────────────────────────────────────────────────────────┘
                           │
                           ▼ Harmonized Waveforms (12, 1000)
┌────────────────────────────────────────────────────────┐
│ MODULE 2: Progressive Lead Degradation                 │
│ - Deterministic binary masks across 7 configurations   │
│ - Clean zero-masking with zero synthetic imputation    │
└────────────────────────────────────────────────────────┘
                           │
                           ▼ Degraded Waveforms (258,216 arrays)
┌────────────────────────────────────────────────────────┐
│ MODULE 3: ECG-FM Representation Backbone               │
│ - Frozen Wav2Vec2CMSCModel (90.88M parameters)         │
│ - Cached latent embeddings in R^768                    │
└────────────────────────────────────────────────────────┘
                           │
                           ▼ Embeddings z in R^768 + Masks m in {0,1}^12
┌────────────────────────────────────────────────────────┐
│ MODULE 4: Control Reasoning & Evidence Evaluation       │
│ - Shared MLP Classifier (780 -> 384 -> 192 -> 8)       │
│ - Masked BCE Loss & Validation Temperature Scaling     │
│ - 25,816 Validation Reasoning Trace Dossiers           │
│ - Four-Tier Evidence Consistency Auditing              │
└────────────────────────────────────────────────────────┘
```

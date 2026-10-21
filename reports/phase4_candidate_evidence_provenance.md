# Phase 4 Forensic Audit: Candidate-Specific Evidence & Attention Head Provenance

**Audit Date:** October 11, 2026  
**Auditor:** Independent AI Pair-Programming System (Forensic Inspection Mode)  
**Target Architecture:** Phase 4 Control Reasoning Transformer (`src/reasoning/`)  
**Scope:** Forensic code-level investigation of Candidate-Specific Evidence $E_{l,i}$, the 51 Multimodal Evidence Tokens, Cross-Attention Mechanics, Supervision Paths, and Clinical Rule Coverage.  
**Operational Status:** Inspection-Only Audit. No models retrained, no checkpoints modified, no benchmarks rerun.

---

## 1. Executive Summary

This forensic audit investigates the mathematical, code-level, and clinical reality of candidate-specific evidence and the attention head in Phase 4.

### Key Audit Findings:

1. **Candidate-Specific Evidence $E_{l,i}$ Does Not Exist as an Input Tensor:**
   In the source code ([`src/reasoning/context_builder.py`](file:///e:/ECG/src/reasoning/context_builder.py)), the multimodal evidence sequence is **strictly candidate-agnostic** ($E_l \in \mathbb{R}^{B \times 51 \times 256}$). It is constructed from continuous latent embeddings $Z_{\text{ECG}}$, lead masks $m_l$, and discrete observations $O_l$ **before** and **independently of** any disease candidate. Candidate conditioning occurs exclusively *inside* the Transformer via Multihead Cross-Attention, where static learned query vectors $q_i \in \mathbb{R}^{256}$ query the shared evidence keys. Thus, candidate-specific evidence is a **latent, learned continuous mixture**, not an explicit clinical evidence mapping.

2. **The "50 Evidence Tokens" Correspond to 51 Fixed Structural Feature Embeddings:**
   The context builder constructs exactly **51 tokens** in $\mathbb{R}^{256}$:
   - 1 continuous foundation model embedding (`ECG_CONTINUOUS_FM`)
   - 12 lead geometry & availability tokens (`LEAD_OBS_I` .. `LEAD_OBS_V6`)
   - 1 rhythm classification token (`RHYTHM_PATTERN`)
   - 1 numerical interval token (`INTERVAL_METRICS`: normalized HR, PR, QRS)
   - 36 morphological abnormality tokens ($12 \times \text{ST elevation}$, $12 \times \text{ST depression}$, $12 \times \text{T-wave inversion}$)
   These tokens represent **continuous neural embeddings** indexed by categorical states or linear projections, rather than symbolic clinical facts or discrete token IDs.

3. **The Attention Head is an Unsupervised Cross-Attention Module:**
   The "attention/evidence head" is implemented as standard PyTorch `nn.MultiheadAttention(embed_dim=256, num_heads=4)`. It receives **zero direct supervision, zero evidence-ranking loss, and zero clinical annotation targets**. Attention weights receive gradients exclusively via backpropagation through the final disease classification loss (`nn.BCEWithLogitsLoss()`).

4. **Attention Weights Are Heuristic Exploratory Attributions, Not Proven Clinical Evidence:**
   In [`src/reasoning/trace_generator.py`](file:///e:/ECG/src/reasoning/trace_generator.py), tokens with the highest attention weights are sorted and labeled `"supporting_evidence"`. Because cross-attention does not enforce a `key_padding_mask` on inactive leads, **masked, unobservable channels still receive positive attention weights**. The trace generator flags them post-hoc as `is_observable_in_mask: false`. Attention weights reflect mathematical query-key dot products, not verified causal medical reasoning.

5. **Clinical Evidence Rules Exist in Configuration, But Are Not Executed by the Model:**
   [`reports/evidence_observability_config.json`](file:///e:/ECG/reports/evidence_observability_config.json) defines detailed clinical verification rules (required leads, signal quality, support/contradiction conditions) for all 8 candidates. However, **this rulebook is not integrated into model training or forward inference**. In [`src/reasoning/trace_generator.py`](file:///e:/ECG/src/reasoning/trace_generator.py), only a single heart-rate check for Sinus Bradycardia ($\ge 60$) and Sinus Tachycardia ($\le 100$) is executed. All other diagnostic criteria remain unexecuted by the code.

---

## 2. Forensic Audit A — Exact Definition of Candidate-Specific Evidence $E_{l,i}$

### 2.1 Mathematical & Code-Level Definition

In the formal mathematical specification, $E_{l,i}$ represents candidate-specific evidence for candidate disease $i \in \{1 \dots 8\}$ under lead configuration $l \in \{12, 10, 8, 6, 4, 2, 1\}$. 

A rigorous inspection of [`src/reasoning/context_builder.py`](file:///e:/ECG/src/reasoning/context_builder.py) and [`src/reasoning/reasoning_transformer.py`](file:///e:/ECG/src/reasoning/reasoning_transformer.py) reveals the exact implementation:

```
[Raw ECG X_l] ---> FrozenECGFM ---> Z_ECG in R^768 -------------------------\
[Mask m_l]    ---------------------------------------------------------------> ControlContextBuilder ---> evidence_tokens (B, 51, 256)
[Obs O_l]     -------------------------------------------------------------/        |
                                                                                    v
                                                                       nn.TransformerEncoder (2 layers)
                                                                                    |
                                                                                    v
                                                                           encoded_evidence (B, 51, 256)
                                                                                    |
Candidate Queries Q = [q_1, ..., q_8] in R^(8 x 256) -------------------------------> nn.MultiheadAttention (cross-attn)
                                                                                    |
                                                                                    +---> attn_weights A in R^(B x 8 x 51)
                                                                                    v
                                                                             attn_out in R^(B x 8 x 256)
                                                                                    |
                                                                               Query FFN + Norm
                                                                                    |
                                                                                    v
                                                                           queries_final in R^(B x 8 x 256)
                                                                                    |
                                                                                Score Head (Linear)
                                                                                    |
                                                                                    v
                                                                             logits in R^(B x 8)
```

#### Code Locations & Tensor Contracts:
- **Evidence Construction:** [`src/reasoning/context_builder.py:lines 108–287`](file:///e:/ECG/src/reasoning/context_builder.py#L108-L287) (`ControlContextBuilder.forward`).
  - Inputs:
    - `z_ecg`: `torch.Tensor` of shape `(B, 768)`, float32.
    - `observations`: `List[Dict[str, Any]]` of length $B$.
    - `lead_avail_mask`: `torch.Tensor` of shape `(B, 12)`, float32.
    - `ckp_tokens`: `None` (strictly enforced; non-null raises `ValueError`).
  - Output:
    - `evidence_tokens`: `torch.Tensor` of shape `(B, 51, 256)`, float32. **(Candidate-Agnostic)**.
    - `query_tokens`: `torch.Tensor` of shape `(B, 8, 256)`, float32.
- **Cross-Attention & Conditioning:** [`src/reasoning/reasoning_transformer.py:lines 132–148`](file:///e:/ECG/src/reasoning/reasoning_transformer.py#L132-L148) (`ControlReasoningTransformer.forward`).
  - Operation:
    $$\text{attn\_out}, A = \text{MultiheadAttention}(Q=\text{query\_tokens}, K=\text{encoded\_evidence}, V=\text{encoded\_evidence})$$
  - Tensor Shapes:
    - `attn_weights` ($A$): `(B, 8, 51)`, float32. Softmax-normalized across the 51 evidence tokens for each candidate.
    - `attn_out`: `(B, 8, 256)`, float32.
    - `queries_final`: `(B, 8, 256)`, float32.

#### Key Invariant Findings:
- **No Explicit Input Tensor $E_{l,i}$:** There is no tensor named `E_{l,i}` or candidate-specific evidence token list at the input stage.
- **Candidate Conditioning Mechanism:** The candidate identity $i$ enters *only* as a static learned query embedding vector $q_i \in \mathbb{R}^{256}$ stored in `self.candidate_queries = nn.Parameter(torch.randn(8, d_model) / sqrt(d_model))`.
- **Invariance across Candidates:** The evidence tokens $E_l$ do **not** change when the candidate changes. $E_l$ is identical for candidate 0 (NORM) and candidate 7 (MI).
- **Variance across Leads:** $E_l$ **does** change when the available leads change, because morphological abnormality embeddings corresponding to inactive leads are zeroed out (`b_elev.append(emb if is_active else torch.zeros_like(emb))`), and lead geometry tokens reflect $m_l$.

---

### 2.2 Source-to-Output Provenance

The 51 tokens in `evidence_tokens` originate from the following distinct computational sources:

| Token Slot Index | Canonical Identifier | Source Modality | Underlying Source Input | Mathematical Transformation | Clinical Interpretability |
|:---:|---|---|---|---|---|
| **0** | `ECG_CONTINUOUS_FM` | Continuous FM Representation | Raw ECG $X \in \mathbb{R}^{12 \times 1000}$ via `FrozenECGFM` | $Z_{\text{ECG}} \in \mathbb{R}^{768} \to \text{Linear}(768, 256) \to \text{LayerNorm} \to \text{GELU}$ | **Latent only**; uninterpretable continuous embedding. |
| **1 – 12** | `LEAD_OBS_I` .. `LEAD_OBS_V6` | Lead Availability / Geometry | Lead Availability Mask $m_l \in \{0, 1\}^{12}$ | $\text{Embedding}_{pos}(l) + \text{Embedding}_{status}(m_l)$ | **Geometric status**; indicates whether channel $l$ is active or masked. |
| **13** | `RHYTHM_PATTERN` | Discrete Structured Observation | `obs['rhythm']` from $O_l$ | String $\to \text{RHYTHM\_TO\_IDX} \to \text{Embedding}(6, 256)$ | **Categorical rhythm finding** (e.g., Sinus rhythm, AFib). |
| **14** | `INTERVAL_METRICS` | Discrete Structured Observation | `obs['heart_rate']`, `obs['pr_interval_ms']`, `obs['qrs_duration_ms']` | $\begin{bmatrix} (HR-75)/30 \\ (PR-160)/40 \\ (QRS-100)/25 \end{bmatrix} \to \text{Linear}(3, 256) \to \text{LN} \to \text{GELU}$ | **Standardized interval metrics**; continuous projection of 3 clinical numbers. |
| **15 – 26** | `ST_ELEV_I` .. `ST_ELEV_V6` | Discrete Structured Observation | `obs['st_elevation']['leads']` $\cap$ active leads | If lead $l$ elevated and active: $\text{Embedding}_{elev}(l)$; Else: $\mathbf{0} \in \mathbb{R}^{256}$ | **Lead-specific ST elevation**; indicates ischemic injury pattern. |
| **27 – 38** | `ST_DEP_I` .. `ST_DEP_V6` | Discrete Structured Observation | `obs['st_depression']['leads']` $\cap$ active leads | If lead $l$ depressed and active: $\text{Embedding}_{dep}(l)$; Else: $\mathbf{0} \in \mathbb{R}^{256}$ | **Lead-specific ST depression**; indicates subendocardial ischemia/strain. |
| **39 – 50** | `T_INV_I` .. `T_INV_V6` | Discrete Structured Observation | `obs['t_wave_inversion']['leads']` $\cap$ active leads | If lead $l$ inverted and active: $\text{Embedding}_{t}(l)$; Else: $\mathbf{0} \in \mathbb{R}^{256}$ | **Lead-specific T-wave inversion**; indicates reperfusion/ischemia/strain. |

---

### 2.3 Candidate-Specific Construction Across All 8 Candidates

The actual implementation for each of the eight disease candidates was audited against the source code:

| Candidate ID | Acronym | Clinical Disease Name | Input Evidence Tokens Provided | Candidate-Specific Operation | Explicit Candidate-to-Evidence Rule in Code? |
|:---:|:---:|---|---|---|:---:|
| **0** | `NORM` | Normal ECG | Shared 51 tokens ($E_l$) | $q_0 \in \mathbb{R}^{256}$ queries $K(E_l)$ via cross-attention | **NO** (Learned query only) |
| **1** | `AFIB` | Atrial Fibrillation | Shared 51 tokens ($E_l$) | $q_1 \in \mathbb{R}^{256}$ queries $K(E_l)$ via cross-attention | **NO** (Learned query only) |
| **2** | `RBBB` | Right Bundle Branch Block | Shared 51 tokens ($E_l$) | $q_2 \in \mathbb{R}^{256}$ queries $K(E_l)$ via cross-attention | **NO** (Learned query only) |
| **3** | `LBBB` | Left Bundle Branch Block | Shared 51 tokens ($E_l$) | $q_3 \in \mathbb{R}^{256}$ queries $K(E_l)$ via cross-attention | **NO** (Learned query only) |
| **4** | `SBRAD` | Sinus Bradycardia | Shared 51 tokens ($E_l$) | $q_4 \in \mathbb{R}^{256}$ queries $K(E_l)$ via cross-attention | **NO** (Learned query only) |
| **5** | `STACH` | Sinus Tachycardia | Shared 51 tokens ($E_l$) | $q_5 \in \mathbb{R}^{256}$ queries $K(E_l)$ via cross-attention | **NO** (Learned query only) |
| **6** | `1AVB` | First Degree AV Block | Shared 51 tokens ($E_l$) | $q_6 \in \mathbb{R}^{256}$ queries $K(E_l)$ via cross-attention | **NO** (Learned query only) |
| **7** | `MI` | Myocardial Infarction | Shared 51 tokens ($E_l$) | $q_7 \in \mathbb{R}^{256}$ queries $K(E_l)$ via cross-attention | **NO** (Learned query only) |

#### Finding:
The architecture contains **no explicit candidate-to-clinical-finding mapping**. There is no programmatic rule selecting rhythm or PR intervals for 1AVB, or ST elevation for MI. All candidates receive the exact same 51 input tokens. The candidate specificity exists solely in the 256-dimensional learned vector $q_i$.

---

## 3. Forensic Audit B — The Reported 50 Evidence Tokens

### 3.1 Verification of Count and Storage
- **Actual Token Count:** Exactly **51 tokens** ($1 \text{ continuous} + 50 \text{ discrete/structural}$).
- **Where Created:** [`src/reasoning/context_builder.py:lines 255–267`](file:///e:/ECG/src/reasoning/context_builder.py#L255-L267).
- **Representation Format:** Continuous tensors `torch.Tensor` of shape `(B, 51, 256)` with `dtype=torch.float32`.
- **Vocabulary Mapping Table:** Defined in `context_builder.py:lines 270–278` as a string list `evidence_token_names`.

### 3.2 Semantic Stability Analysis
1. **Positional Invariance:** The semantic meaning of each slot $j \in \{0 \dots 50\}$ is **strictly fixed and stable**. Position 0 is always the foundation model vector, Position 13 is always rhythm, Position 14 is always numerical intervals, and Position 15 is always ST elevation in Lead I.
2. **Behavior Under Lead Masking:** When Lead $l$ is masked in $m_l$:
   - Slot $1 + l$ (`LEAD_OBS_l`) embeds `mask_binary = 0`.
   - Slots $15 + l$ (`ST_ELEV_l`), $27 + l$ (`ST_DEP_l`), and $39 + l$ (`T_INV_l`) are assigned the zero vector $\mathbf{0} \in \mathbb{R}^{256}$.
3. **Learned vs. Symbolic Semantics:**
   While the *inputs* to these slots derive from explicit medical observations, the *tokens themselves* are passed through learned embedding matrices (`nn.Embedding(12, 256)`) and linear projections. Their internal 256-dimensional activations are latent neural representations, not human-readable symbolic facts.

---

## 4. Forensic Audit C — Attention / Evidence Head Architecture and Supervision

### 4.1 Exact Mechanism & Mathematical Formulation
The "attention head" is implemented in [`src/reasoning/reasoning_transformer.py:lines 71–76, 137–143`](file:///e:/ECG/src/reasoning/reasoning_transformer.py#L71-L76):

```python
self.cross_attention = nn.MultiheadAttention(
    embed_dim=256,
    num_heads=4,
    dropout=0.1,
    batch_first=True,
)
```

#### Tensor Mathematics:
For sample $b$ and candidate hypothesis $i \in \{1 \dots 8\}$:
1. Candidate Query: $Q_{b, i} = q_i W_Q \in \mathbb{R}^{1 \times 256}$.
2. Evidence Keys: $K_{b} = \text{encoded\_evidence}_b W_K \in \mathbb{R}^{51 \times 256}$.
3. Evidence Values: $V_{b} = \text{encoded\_evidence}_b W_V \in \mathbb{R}^{51 \times 256}$.
4. Multi-head Cross-Attention:
   $$A_{b, i, j} = \frac{1}{4} \sum_{h=1}^4 \text{Softmax}_j \left( \frac{Q_{b, i}^{(h)} (K_{b, j}^{(h)})^T}{\sqrt{64}} \right)$$
   $$\text{attn\_out}_{b, i} = \sum_{j=0}^{50} A_{b, i, j} V_{b, j}$$

#### Dimensions and Normalization:
- **Attention Tensor Shape:** `(B, 8, 51)`.
- **Normalization Axis:** Normalized along dimension 2 (the 51 evidence tokens) via Softmax.
- **Invariant:** For every candidate $i$, $\sum_{j=0}^{50} A_{b, i, j} = 1.0000 \pm 10^{-4}$ ([`tests/test_control_reasoning.py:lines 168–169`](file:///e:/ECG/tests/test_control_reasoning.py#L168-L169)).
- **Candidate Aggregation:** None. Attention weights are computed independently for each of the 8 candidates.

---

### 4.2 Training Loss and Supervision Analysis

Inspection of [`scripts/16_train_reasoning_transformer.py:lines 257, 302–310`](file:///e:/ECG/scripts/16_train_reasoning_transformer.py#L257):

```python
criterion = nn.BCEWithLogitsLoss()
...
output = model(z_ecg=b_z, observations=b_obs, lead_avail_mask=b_masks)
loss = criterion(output["logits"], b_labels)
loss.backward()
optimizer.step()
```

#### Critical Audit Finding: ZERO EVIDENCE SUPERVISION
- **Loss Terms:** The entire training script uses **only one loss term**: multi-label binary cross-entropy on final disease presence (`b_labels`).
- **Absence of Attention Targets:** There are no ground truth attention maps, no token-importance targets, and no auxiliary ranking losses.
- **Gradient Flow:** The attention head parameters ($W_Q, W_K, W_V, W_O$) receive gradients *only* through their contribution to `attn_out`, which updates `queries_final` to minimize classification error.
- **Scientific Implication:** The attention weights $A_{b, i, j}$ are **completely unsupervised**. The fact that attention weights sum to 1.0 is an axiomatic property of the Softmax function, not evidence of clinical comprehension.

---

### 4.3 Lead Masking Failure in Cross-Attention

In [`src/reasoning/reasoning_transformer.py:line 137`](file:///e:/ECG/src/reasoning/reasoning_transformer.py#L137):
```python
attn_out, attn_weights = self.cross_attention(
    query=query_tokens,
    key=encoded_evidence,
    value=encoded_evidence,
    need_weights=True,
    average_attn_weights=True,
)
```
- **Finding:** `key_padding_mask` is **omitted** (set to `None`).
- **Consequence:** In 2-lead or 1-lead evaluations, the masked leads are still present as zero vectors in `encoded_evidence`. Because Softmax is computed across all 51 positions without masking, **masked channels receive non-zero attention weights**.
- **Empirical Proof from Saved Traces ([`reports/sample_control_reasoning_traces.json:lines 592–607`](file:///e:/ECG/reports/sample_control_reasoning_traces.json#L592-L607)):**
  In the 2-lead test evaluation (`test_fold10_0000`, active leads: `['II', 'V1']`), the top hypothesis (RBBB) assigns attention weights:
  - `LEAD_OBS_V5`: weight **0.0267** (`is_observable_in_mask: false`)
  - `LEAD_OBS_V3`: weight **0.0262** (`is_observable_in_mask: false`)
  The model places attention on channels that are physically absent.

---

### 4.4 Trace Generation and Evidence Claim Verification

In [`src/reasoning/trace_generator.py:lines 65–96`](file:///e:/ECG/src/reasoning/trace_generator.py#L65-L96):
1. The generator extracts $A_{b, i^*, :}$ for the winning hypothesis $i^* = \arg\max P(H_i)$.
2. It sorts token indices by weight in descending order.
3. It packages the top 4–5 tokens into `"supporting_evidence"`.

#### Overstatement Audit:
- **Heuristic Labeling:** Naming these top-weighted tokens `"supporting_evidence"` is a developer-chosen naming convention. The model was never trained to ensure that high attention implies positive diagnostic support.
- **Trace Serialization vs. Clinical Grounding:** Successfully serializing tokens and attention values into JSON verifies software plumbing, **not clinical grounding**.

---

## 5. Forensic Audit D — Existing Clinical Evidence Rules

A comprehensive scan of the repository was conducted to locate existing clinical rule definitions.

### 5.1 Rules Located in `reports/evidence_observability_config.json`
This configuration file contains structured diagnostic rules for all 8 candidates:

| Disease Claim | Required Leads | Min Leads | Measurement Requirements | Contradiction Condition | Implementation Status in Code |
|---|---|:---:|---|---|:---:|
| **CLAIM_NORMAL_ECG** | `I, II, V1–V6` | 8 | HR 60–100, PR 120–200, QRS < 120, ST/T normal | Any abnormality detected; HR out of bounds; QRS $\ge 120$ | **Documented Only** (0% executed) |
| **CLAIM_AFIB** | `II, V1` | 1 | RR CV > 0.20, absence of discrete P waves | Regular sinus P waves; RR CV $\le 0.12$ | **Documented Only** (0% executed) |
| **CLAIM_RBBB** | `V1, V6` | 2 | QRS $\ge 120\text{ ms}$, rsR' in V1 or wide S in V6 | QRS < 100 ms; narrow R in V1 | **Documented Only** (0% executed) |
| **CLAIM_LBBB** | `I, V1, V6` | 2 | QRS $\ge 120\text{ ms}$, broad monophasic R in I/V6 | QRS < 100 ms; normal septal Q waves | **Documented Only** (0% executed) |
| **CLAIM_SINUS_BRADYCARDIA** | `II` | 1 | HR < 60 bpm, regular sinus P waves | HR $\ge 60\text{ bpm}$ | **Partially Executed** (HR check in trace generator) |
| **CLAIM_SINUS_TACHYCARDIA** | `II` | 1 | HR > 100 bpm, regular sinus P waves | HR $\le 100\text{ bpm}$ | **Partially Executed** (HR check in trace generator) |
| **CLAIM_FIRST_DEGREE_AV_BLOCK** | `II` | 1 | PR interval > 200 ms, 1:1 AV conduction | PR interval $\le 200\text{ ms}$ | **Documented Only** (0% executed) |
| **CLAIM_MYOCARDIAL_INFARCTION** | `I, II, III, aVF, aVL, V1–V4` | 2 | ST elevation $\ge 0.10\text{ mV}$ in $\ge 2$ contiguous leads | All contiguous leads active but ST dev < 0.05 mV | **Documented Only** (0% executed) |

### 5.2 Code Execution Gap:
- **Observation Extractor:** Extracts observations ($O_l$) using heuristic signal processing or linear heads.
- **Context Builder:** Embeds observations into continuous vectors.
- **Transformer:** Ingests vectors and outputs sigmoid logits.
- **Trace Generator:** Evaluates **only** lines 99–111 of `trace_generator.py` (checking if HR contradicts SBRAD or STACH).
- **Finding:** The remaining rules in `evidence_observability_config.json` (contiguous ST elevation for MI, QRS duration for bundle branch blocks, PR prolongation for 1AVB, normal multi-lead criteria) are **completely unexecuted in the live inference pipeline**.

---

## 6. Forensic Audit E — Control-Path Integrity and CKP Separation

### 6.1 Audit of CKP Isolation Invariant
- **Source Check:** Inspected [`src/reasoning/context_builder.py:lines 130–136`](file:///e:/ECG/src/reasoning/context_builder.py#L130-L136) and [`src/reasoning/reasoning_transformer.py:lines 121–123`](file:///e:/ECG/src/reasoning/reasoning_transformer.py#L121-L123).
- **Mechanism:**
  ```python
  if ckp_tokens is not None:
      raise ValueError("[CONTROL PATH VIOLATION] CKP tokens detected in Control Path.")
  ```
- **Automated Verification:** Verified by unit tests [`tests/test_control_reasoning.py:lines 97–106`](file:///e:/ECG/tests/test_control_reasoning.py#L97-L106) and [`tests/test_control_training.py:lines 123–133`](file:///e:/ECG/tests/test_control_training.py#L123-L133).
- **Audit Proof:** The Control Path strictly rejects external knowledge graph priors, SNOMED-CT ontological relations, or multi-modal CKP tokens.

---

## 7. Forensic Audit F — Suitability for Future Clinical Evidence Schema (Module 5)

Based on the forensic findings, the suitability of Phase 4 components for the future Module 5 Clinical Evidence Schema is evaluated below:

### 7.1 Component Suitability Evaluation

| Architectural Component | Can it serve as the Evidence Schema? | Can it serve as an Input Representation? | Can it rank evidence? | Defensible Next Step |
|---|:---:|:---:|:---:|---|
| **$E_{l,i}$ Representation** | **NO** (Not an explicit schema) | **YES** (51 token slots have stable semantics) | N/A | Retain 51-token structure as perception input; build discrete symbolic evidence layer on top. |
| **Attention Head** | **NO** (Unsupervised query-key matching) | N/A | **NO** (Heuristic attribution only) | Treat attention weights as exploratory saliency; do not use as certified clinical proof without explicit rule verification. |
| **`evidence_observability_config.json`** | **YES** (Strong clinical foundation) | N/A | **YES** (Defines explicit criteria) | Implement this JSON as an active deterministic verification layer in Module 5. |

### 7.2 What Module 5 Must Independently Verify
Module 5 cannot trust model attention weights to establish evidence validity. Module 5 must independently implement deterministic programmatic gates for:
1. **Contiguous Lead Criteria:** Verify that ST elevation occurs in $\ge 2$ anatomically contiguous leads (e.g., II, III, aVF for inferior; V1, V2 for septal) rather than isolated noisy channels.
2. **Mandatory Lead Requirements:** Enforce that RBBB/LBBB cannot be declared "supported" if V1 is masked, regardless of model confidence.
3. **Hard Contradiction Logic:** Programmatically flag contradictions when QRS $< 100\text{ ms}$ for bundle branch blocks, or PR $\le 200\text{ ms}$ for First Degree AV Block.
4. **Active Lead Key-Padding Masking:** Apply `key_padding_mask` in the cross-attention module so masked leads receive exactly 0 attention.

---

## 8. Final Verdicts & Justifications

### Primary Verdict: Candidate-Specific Evidence
# **LATENT OR CANDIDATE-CONDITIONED REPRESENTATION — CLINICAL MAPPING REQUIRED**

**Justification:**
Code inspection confirms that the multimodal evidence sequence is strictly candidate-agnostic at the input stage ($E_l \in \mathbb{R}^{B \times 51 \times 256}$). Candidate specificity occurs solely inside the cross-attention head via 8 learned static query vectors $q_i \in \mathbb{R}^{256}$. There is no explicit candidate-to-clinical-finding mapping, no candidate-specific token filtering, and no discrete symbolic evidence schema in the current implementation.

---

### Secondary Verdict: Attention / Evidence Head
# **UNSUPERVISED OR INDIRECT ATTENTION — EXPLORATORY ONLY**

**Justification:**
Inspection of [`scripts/16_train_reasoning_transformer.py`](file:///e:/ECG/scripts/16_train_reasoning_transformer.py) confirms that the model is trained strictly with multi-label binary cross-entropy loss (`nn.BCEWithLogitsLoss()`) on disease labels. The cross-attention module receives zero evidence-level supervision, zero ranking targets, and zero attention regularizers. Furthermore, because key padding masks are not applied to inactive leads, the attention head places non-zero weights on masked channels. The attention weights represent mathematical query-key dot products rather than validated causal clinical reasoning.

---

**Audit Complete. Standing by for user review and approval before proceeding to Module 5 design.**

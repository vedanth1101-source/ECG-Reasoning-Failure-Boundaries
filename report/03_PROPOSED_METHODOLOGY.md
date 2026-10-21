# CHAPTER 3: PROPOSED METHODOLOGY

---

## ARCHITECTURAL DESIGN OF THE PROPOSED SYSTEM

The proposed methodology is structured as an end-to-end, closed-loop, six-phase epistemological architecture designed to measure the degradation of clinical reasoning under progressive physiological information loss. Each phase consumes a strictly validated artifact from its immediate predecessor and emits an immutable, auditable output for downstream components. 

A central innovation of the architecture is the **independent verification bypass channel**: the binary lead-availability mask and the physical millivolt waveforms are routed directly from Phase 2 to Phase 5, completely bypassing the reasoning neural network. This architectural guarantee ensures that the Grounding Verification Logic Gate adjudicates claims against physical ground truth rather than relying on the model's subjective self-reports.

```
       +-------------------------------------------------------------------------------+
       |             PHASE 1: DATA AND GROUND TRUTH MANAGEMENT                         |
       |  Raw ECG Cohorts (PTB-XL, Chapman, MIMIC) -> Operations P1 to P5 ->           |
       |  Standardized Voltage Matrix X in R^{12 x T} (500 Hz / 100 Hz, mV Calibrated) |
       +-------------------------------------------------------------------------------+
                                              |
                                              v
       +-------------------------------------------------------------------------------+
       |             PHASE 2: CONTROLLED PHYSIOLOGICAL DEGRADATION                     |
       |  Deterministic Lead Mask m_l in {0,1}^12 (Levels: 12, 10, 8, 6, 4, 2, 1)      |
       |  Cabrera Lead Provenance DAG (Zero Generative Inpainting)                     |
       +-------------------------------------------------------------------------------+
                         /                                             \
        [Bypass Channel: m_l, X_l]                                      v
                       /             +-------------------------------------------------+
                      /              | PHASE 3: FOUNDATION MODEL ADAPTATION & DUAL REP |
                     /               | 1. Frozen ECG-FM (wav2vec2_cmsc) -> Z_ECG R^768 |
                    /                | 2. ClinicalObservationExtractor -> O_l (Rules)  |
                   /                 +-------------------------------------------------+
                  /                                           |
                 /                                            v
                /                    +-------------------------------------------------+
               /                     | PHASE 4: DUAL-PATH CLINICAL REASONING           |
              /                      | Context Construction:                           |
             /                       | - Control Context: [Z_ECG; O_l; H_i]            |
            /                        | - Experimental Context: [Z_ECG; CKP; O_l; H_i]  |
           /                         | Shared Reasoning Transformer ->                 |
          /                          | 1. Score Head (Calibrated Posterior P(H_i))     |
         /                           | 2. Attention Head (Attribution Matrix A)        |
        /                            | Output: Structured Clinical Reasoning Trace     |
       /                             +-------------------------------------------------+
      |                                                       |
      v                                                       v
+--------------------------------------------------------------------------------------+
|             PHASE 5: GROUNDING VERIFICATION LOGIC GATE                               |
| Decompose Trace into Claims gamma_k -> Query Expert-Ratified Evidence Schema         |
| Evaluate Physical Sensor Mask & Measured Thresholds:                                 |
| --> State: SUPPORTED | CONTRADICTED | UNAVAILABLE                                    |
+--------------------------------------------------------------------------------------+
                                              |
                                              v
+--------------------------------------------------------------------------------------+
|             PHASE 6: RESULTS COMPILATION AND DEGRADATION METRICS                     |
| Compute: Evidence Observability (EO_l), Grounding Quality (GQ_l), True Grounding     |
| Ratio (TGR_l) -> Construct Reasoning Degradation Curve (RDC) ->                      |
| Integrate AURDC (Trapezoidal Quadrature) -> Identify Reasoning Failure Boundary (RFB)|
+--------------------------------------------------------------------------------------+
```

---

## PHASE 1: DATA AND GROUND TRUTH MANAGEMENT

### Multi-cohort dataset ingestion (PTB-XL, Chapman-Shaoxing, MIMIC-IV-ECG)
To establish an authoritative and generalizable evaluation benchmark, Phase 1 ingests three massive, publicly accessible clinical ECG repositories:
1. **PTB-XL (Primary Development & Benchmarking Cohort):** Comprises 21,799 clinical 12-lead recordings of 10.0-second duration sampled at 500 Hz from 18,869 unique patients. Recordings are mapped to the SCP-ECG standard. Label analysis reveals 14,719 single-label records (67.52%), 2,456 multi-label records (11.27%), and 4,624 unmapped/other records (21.21%).
2. **Chapman-Shaoxing (External Validation Cohort 1):** Ingests 10,249 12-lead ECG recordings from distinct patient admissions at Chapman University and Shaoxing People's Hospital. Following data integrity audits, 91 records with missing leads or corrupted MAT headers were excluded, yielding a verified cohort of 10,158 records.
3. **MIMIC-IV-ECG (External Validation Cohort 2):** Derived from the massive MIMIC-IV clinical database (containing over 800,000 diagnostic studies across 161,352 ICU and emergency patients). To prevent class imbalance and computational saturation, a balanced, patient-stratified benchmark subset of **5,000 records from 5,000 unique patients** (zero duplicate `subject_id`) was constructed with seed-locked random sampling (~625 records per benchmark condition).

### Patient-stratified cohort partitioning and leakage prevention
To eliminate data leakage, partitioning is performed strictly at the patient level. When a patient has multiple sequential recordings taken over hours or days, allocating their recordings across training and testing sets causes deep models to memorize patient-specific ventricular geometry, resulting in artificially inflated performance estimates.

On PTB-XL, we adhere strictly to the recommended 10-fold patient-stratified split:
- **Training Cohort (Folds 1–8):** 17,418 recordings from 15,096 patients.
- **Validation Cohort (Fold 9):** 2,183 recordings from 1,893 patients (used exclusively for hyperparameter tuning, temperature calibration, and checkpoint selection).
- **Quarantined Test Cohort (Fold 10):** 2,198 recordings from 1,880 patients (isolated under strict lock until final evaluation).

### The five sequential preprocessing operations (P1 to P5)
Raw ECG recordings exhibit significant heterogeneity in sampling rates, lead sequences, ADC conversion baselines, and ambient electrical noise. To standardize all recordings without introducing phase distortions or artificial morphological shifts, Phase 1 executes five ordered mathematical operations:
$$\mathbf{X} = (P_5 \circ P_4 \circ P_3 \circ P_2 \circ P_1)(\mathbf{X}_{\text{raw}})$$

```
  Raw Input Signal (C, T_raw)
               |
               v
  +-----------------------------------------------------------------+
  | Operation P1: Canonical Lead Ordering & Transposition           |
  | Maps arbitrary channel permutations into canonical sequence:    |
  | [I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]             |
  +-----------------------------------------------------------------+
               |
               v
  +-----------------------------------------------------------------+
  | Operation P2: Polyphase Resampling & Temporal Windowing         |
  | Zero-phase polyphase resampling to f_s = 500 Hz / 100 Hz        |
  | Symmetrical midpoint crop or zero-pad -> Fixed shape (12, N)    |
  +-----------------------------------------------------------------+
               |
               v
  +-----------------------------------------------------------------+
  | Operation P3: ADC Calibration & Voltage Normalization           |
  | V_mV = (x_raw - b_ADC) / g_ADC; Clamped to [-15, +15] mV        |
  +-----------------------------------------------------------------+
               |
               v
  +-----------------------------------------------------------------+
  | Operation P4: Zero-Phase Digital Filtering (filtfilt @ 500 Hz)   |
  | - 0.5 Hz 3rd-order Butterworth HP (baseline wander)             |
  | - 50/60 Hz IIR Notch filter (mains power hum, Q = 30)           |
  | - 45.0 Hz Butterworth LP (high-frequency muscle tremor)         |
  +-----------------------------------------------------------------+
               |
               v
  +-----------------------------------------------------------------+
  | Operation P5: Quality Screening & Integrity Audit Gates         |
  | Dimensional check, NaN/Inf rejection, Flatline & Saturation     |
  +-----------------------------------------------------------------+
               |
               v
  Standardized Output Tensor X in R^{12 x T}
```

#### Operation P1: Canonical lead ordering and spatial alignment
Clinical ECG leads reflect distinct anatomical projection vectors. Operation P1 reads header metadata, normalizes lead nomenclatures (e.g., converting `lead1`, `I`, `MLI` to canonical `I`), and applies a permutation matrix $\mathbf{P} \in \{0, 1\}^{12 \times C}$:
$$\mathbf{X}_{\text{canonical}} = \mathbf{P} \mathbf{X}_{\text{raw}}$$
arranging the recording into the canonical 12-lead sequence:
$$\mathcal{L}_{\text{canonical}} = [\text{I}, \text{II}, \text{III}, \text{aVR}, \text{aVL}, \text{aVF}, \text{V1}, \text{V2}, \text{V3}, \text{V4}, \text{V5}, \text{V6}]$$
If any mandatory lead is absent or duplicated, the pipeline raises a strict `LeadOrderingError`.

#### Operation P2: Sampling-rate and duration standardization
To guarantee constant sample indices for temporal interval extraction, recordings are resampled to a standardized frequency ($f_s = 500\text{ Hz}$ for high-resolution morphological extraction, and $100\text{ Hz}$ for foundation model ingestion). 

Crucially, rather than utilizing standard FFT Fourier truncation (which induces edge wrap-around artifacts), Operation P2 employs polyphase filtering via `scipy.signal.resample_poly` with up-sampling factor $p$ and down-sampling factor $q$. Recordings are standardized to a 10.0-second duration: recordings longer than 10.0 seconds are cropped symmetrically around their temporal midpoint:
$$\text{start} = \lfloor (N_{\text{resampled}} - N_{\text{target}}) / 2 \rfloor, \quad \text{end} = \text{start} + N_{\text{target}}$$
while shorter recordings are symmetrically padded with zeros, recording exact padding sample counts in a validity mask.

#### Operation P3: Unit calibration and physical-voltage normalization
Digital integer counts stored by analog-to-digital converters (ADCs) depend on hardware amplifier gains ($g_{\text{ADC}}$) and baseline offsets ($b_{\text{ADC}}$). Operation P3 converts raw counts into physical millivolts:
$$V_{\text{mV}}[c, n] = \frac{x_{\text{raw}}[c, n] - b_{\text{ADC}}[c]}{g_{\text{ADC}}[c]}$$
Calibration parameters were formally audited:
- PTB-XL: Format 16, Gain = 1000.0 counts/mV, Baseline = 0.
- Chapman-Shaoxing: MATLAB float/int, Gain = 1000.0 counts/mV, Baseline = 0.
- MIMIC-IV-ECG: Format 16, Gain = 200.0 counts/mV, Baseline = 0.

To eliminate non-cardiac hardware saturation transients, voltage amplitudes outside physiological plausibility are clamped to $[-15.0, +15.0]\text{ mV}$. An audit across all 36,957 records (443,484,000 samples) revealed that this clamp affected only 3 records (0.0081%) and 6 individual samples (0.000001%), confirming that valid cardiac depolarizations are never distorted.

#### Operation P4: Zero-Phase digital filtering
Raw surface potentials contain baseline wander ($< 0.5\text{ Hz}$), mains interference ($50\text{ Hz} / 60\text{ Hz}$), and electromyographic muscle tremor ($> 40\text{ Hz}$). Operation P4 applies a three-stage cascade:
1. **High-Pass Filter:** 3rd-order Butterworth high-pass filter with cutoff $f_{\text{hp}} = 0.5\text{ Hz}$.
2. **Dataset-Aware Notch Filter:** 2nd-order IIR notch filter ($Q = 30$) applied at $50.0\text{ Hz}$ for PTB-XL and Chapman-Shaoxing (European/Asian mains) and $60.0\text{ Hz}$ for MIMIC-IV-ECG (US mains).
3. **Low-Pass Filter:** 4th-order Butterworth low-pass filter with cutoff $f_{\text{lp}} = 45.0\text{ Hz}$.

Crucially, all filtering is applied in a forward-backward bidirectional manner (`scipy.signal.filtfilt`) at the original acquisition frequency ($500\text{ Hz}$) *prior* to decimation. Bidirectional filtering achieves zero phase shift ($\phi(\omega) = 0$), preventing any artificial temporal displacement of QRS onsets, J-points, or T-wave peaks.

#### Operation P5: Quality screening and integrity audit
The preprocessed tensor passes through four automated quality verification gates:
- **Dimensional Gate:** Verifies exact shape $(12, N_{\text{target}})$.
- **Non-Finite Gate:** Rejects arrays containing `NaN`, `+Inf`, or `-Inf`.
- **Flatline Gate:** Detects disconnected electrode leads where channel variance $\sigma_c^2 < 10^{-8}\text{ mV}^2$ or where consecutive identical samples persist for $> 1.5\text{ seconds}$.
- **Saturation Gate:** Identifies amplifier rail clipping ($|V| \ge 15.0\text{ mV}$).

---

## PHASE 2: CONTROLLED PHYSIOLOGICAL DEGRADATION

### Mathematical formulation of the binary lead-availability mask
Phase 2 imposes controlled physiological degradation by systematically masking lead channels without altering the underlying temporal samples. Let $\mathbf{X} \in \mathbb{R}^{12 \times T}$ be the standardized voltage tensor. For lead level $l \in \Lambda = (12, 10, 8, 6, 4, 2, 1)$, a binary availability vector:
$$\mathbf{m}_l = [m_1, m_2, \dots, m_{12}]^\top \in \{0, 1\}^{12}, \quad \|\mathbf{m}_l\|_0 = l$$
is applied via:
$$\mathbf{X}_l = \mathbf{X} \odot (\mathbf{m}_l \mathbf{1}_T^\top)$$
Inactive channels ($m_j = 0$) are clamped strictly to $0.00000\text{ mV}$. Inactive channel leakage is continuously verified to equal exactly zero.

### The seven primary clinically and device-motivated lead configurations
Rather than applying arbitrary or combinatorial random masking, the seven degradation levels are authoritatively defined in `configs/lead_configurations.json` based on authentic clinical acquisition and device form-factors:

| Level | Config ID | Active Leads | Physical Count | Clinical & Device Motivation |
| :---: | :---: | :--- | :---: | :--- |
| **12** | `C12_FULL` | `I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6` | 12 | AHA/ACC/HRS standard 12-lead diagnostic electrocardiogram. |
| **10** | `C10_PREC_LIMB` | `I, II, aVR, aVL, V1, V2, V3, V4, V5, V6` | 10 | Simulates loss of redundant derived inferior leads (III and aVF). |
| **8** | `C8_INDEPENDENT`| `I, II, V1, V2, V3, V4, V5, V6` | 8 | The 8 mathematically independent leads physically acquired by modern digital carts. |
| **6** | `C6_LIMB_6` | `I, II, III, aVR, aVL, aVF` | 6 | Hexaxial limb reference system; handheld monitors (AliveCor KardiaMobile 6L). |
| **4** | `C4_EINTHOVEN_SEPT`| `I, II, III, V1` | 4 | Standard 4-electrode emergency transport cable (Einthoven limb + V1 chest). |
| **2** | `C2_TELEMETRY` | `II, V1` | 2 | Hospital inpatient telemetry (Lead II for rhythm, Lead V1 for QRS bundle branch block). |
| **1** | `C1_SINGLE_LEAD`| `II` | 1 | Single-channel ambulatory patch / wearable aligned along ventricular axis. |

### Cabrera lead provenance directed acyclic graph (DAG)
The twelve surface leads are not mutually independent. In Phase 2, we implement the Cabrera Lead Provenance Directed Acyclic Graph (DAG) to track physical signal dependencies:
```
             [RA, LA, LL Electrodes]
                    /      \
                   v        v
               Lead I      Lead II  (Directly Measured Parent Leads)
                 |   \    /   |
                 |    \  /    |
                 v     v      v
              Lead III  aVR  aVL  aVF  (Derived Limb Leads)
```

The mathematical relationships are formally defined:
$$\text{III} = \text{II} - \text{I}$$
$$\text{aVR} = -\frac{\text{I} + \text{II}}{2}$$
$$\text{aVL} = \text{I} - \frac{\text{II}}{2}$$
$$\text{aVF} = \text{II} - \frac{\text{I}}{2}$$

Under our provenance model, each lead channel is assigned one of four operational states:
1. `observed_measured`: Directly recorded from physical electrodes (I, II, V1–V6) and active.
2. `observed_derived`: Mathematically derived lead active with both parent leads (I and II) present.
3. `unavailable_derived`: Derived lead active in header, but parent lead I or II is masked.
4. `masked_lead`: Lead clamped to zero by the degradation mask.

*Provenance Invariant:* Derived leads ($\text{III}, \text{aVR}, \text{aVL}, \text{aVF}$) cannot count as independent physiological evidence if parents I or II are missing or already accounted for in the evidence tally.

### Zero-leakage enforcement and prohibition of generative inpainting
In accordance with Constraint C1, generative waveform inpainting, variational autoencoders, and generative adversarial networks are strictly prohibited. Inactive channels remain clamped to zero. This enforces authentic sensor scarcity and prevents synthetic hallucinations from contaminating the benchmark.

---

## PHASE 3: FOUNDATION MODEL ADAPTATION AND DUAL REPRESENTATION

Phase 3 extracts two complementary representations from the degraded recording: a continuous high-dimensional latent vector ($Z_{\text{ECG}}$) and discrete, deterministic clinical observations ($O_l$).

```
            +-----------------------------------------------+
            |    Degraded ECG Input X_masked (Level l)      |
            +-----------------------------------------------+
                                   |
                    +--------------+--------------+
                    |                             |
                    v                             v
    +-------------------------------+  +-------------------------------+
    | Pretrained Frozen ECG-FM      |  | Rule-Based Observation Engine |
    | (wav2vec2_cmsc, 90.8M params) |  | (Physical-unit mV signals)    |
    +-------------------------------+  +-------------------------------+
                    |                             |
                    v                             v
    +-------------------------------+  +-------------------------------+
    | Continuous Latent Vector      |  | Structured Observations O_l   |
    | Z_ECG in R^768                |  | (Rhythm, Intervals, ST, T)   |
    +-------------------------------+  +-------------------------------+
```

### Frozen foundation model backbone (ECG-FM wav2vec2_cmsc, 90.8M parameters)
To benchmark foundational representations without confounding representation adaptation with reasoning degradation, the backbone weights of **ECG-FM** are strictly frozen (`requires_grad = False`).

Model specifications:
- **Architecture:** Wav2Vec 2.0 with Contrastive Multi-Segment Coding (`wav2vec2_cmsc`).
- **Transformer Encoder:** 12 layers, 12 attention heads, hidden dimension $d_{\text{model}} = 768$, FFN dimension $3072$.
- **Convolutional Feature Extractor:** 4 stages of 1D convolutions: `[(256, 2, 2)] * 4` (downsampling factor of 16).
- **Total Parameters:** 90,883,072 parameters (0 trainable).

### Continuous latent vector extraction (Z_ECG in R^768)
The degraded recording $\mathbf{X}_l \in \mathbb{R}^{12 \times 1000}$ is processed by the frozen encoder with `features_only=True`. The temporal sequence of hidden states $\mathbf{H} \in \mathbb{R}^{B \times T' \times 768}$ emitted by the 12th Transformer layer is mean-pooled across time to produce the continuous latent vector:
$$\mathbf{Z}_{\text{ECG}} = \frac{1}{T'} \sum_{t=1}^{T'} \mathbf{H}[:, t, :] \in \mathbb{R}^{768}$$
Extracted latent vectors are cached to disk under SHA-256 manifests. Every vector was audited for numerical stability, verifying zero non-finite values (`NaN`/`Inf`).

### Cosine divergence representation sensitivity audit
To audit how the frozen foundation model's latent space responds to progressive lead deprivation, we compute the cosine distance between the full 12-lead representation $\mathbf{Z}_{12}$ and the degraded representation $\mathbf{Z}_l$:
$$D_l = 1 - \frac{\langle\mathbf{Z}_{12}, \mathbf{Z}_l\rangle}{\|\mathbf{Z}_{12}\|_2 \|\mathbf{Z}_l\|_2}$$

Empirical profiling on test recordings demonstrated that cosine distance increases systematically as leads are eliminated:
- 12-lead: $D_{12} = 0.0000$ (Cosine Similarity = 1.0000)
- 10-lead: $D_{10} = 0.0027$ (Cosine Similarity = 0.9973)
- 8-lead:  $D_{8}  = 0.0029$ (Cosine Similarity = 0.9971)
- 6-lead:  $D_{6}  = 0.1018$ (Cosine Similarity = 0.8982)
- 4-lead:  $D_{4}  = 0.1323$ (Cosine Similarity = 0.8677)
- 2-lead:  $D_{2}  = 0.1587$ (Cosine Similarity = 0.8413)
- 1-lead:  $D_{1}  = 0.2197$ (Cosine Similarity = 0.7803)

This monotonic increase in cosine distance confirms that the frozen backbone is responsive to physiological information loss rather than producing degenerate, invariant embeddings.

### Deterministic clinical observation extraction (O_l: rhythm, intervals, ST, T)
In parallel with latent embedding extraction, the `ClinicalObservationExtractor` extracts discrete physiological biomarkers strictly from observable channels:
1. **Cardiac Rate and Rhythm:** R-peaks are localized using a two-stage derivative thresholding detector on available leads. Heart rate is calculated via median R--R intervals:
   $$\text{HR} = \frac{60 \times f_s}{\text{median}(\Delta n_{\text{RR}})} \quad \text{beats/min}$$
2. **Conduction Intervals:** PR interval (P-wave onset to QRS onset) and QRS duration (Q-onset to S-offset) are measured in milliseconds.
3. **ST-Segment Deviation:** Evaluated at the J-point plus 80 ms ($t_J + 80\text{ ms}$) referenced against the isoelectric TP baseline:
   $$\Delta V_{\text{ST}, c} = V_c(t_J + 80\text{ ms}) - \overline{V}_{\text{TP}, c}$$
4. **T-Wave Morphology:** Peak amplitude and polarity (upright, inverted, biphasic) are determined for each active channel.

*Strict Observability Rule:* If a required lead is masked, the extractor marks that observation field as `UNAVAILABLE`. Missing values are never imputed.

---

## PHASE 4: DUAL-PATH CLINICAL REASONING AND TRACE GENERATION

### The 8-disease multi-label diagnostic candidate space
The diagnostic space is permanently frozen at **exactly 8 conditions** in multi-label, multi-hot format:
1. **Normal ECG (NORM):** Absence of pathological rhythm, conduction, or morphological abnormalities.
2. **Atrial Fibrillation (AFIB):** Absence of P waves, irregularly irregular ventricular response.
3. **Right Bundle Branch Block (RBBB):** QRS duration $\ge 120\text{ ms}$, rsR' pattern in lead V1.
4. **Left Bundle Branch Block (LBBB):** QRS duration $\ge 120\text{ ms}$, broad slurred R in leads I, aVL, V5, V6.
5. **Sinus Bradycardia (SBRAD):** Normal sinus P waves, regular rhythm, $\text{HR} < 60\text{ bpm}$.
6. **Sinus Tachycardia (STACH):** Normal sinus P waves, regular rhythm, $\text{HR} > 100\text{ bpm}$.
7. **First Degree AV Block (1AVB):** Regular rhythm, normal P-QRS relationship, $\text{PR interval} > 200\text{ ms}$.
8. **Myocardial Infarction (MI):** Acute or established ischemic injury (AMI, IMI, ASMI, ALMI).

### Clinical Knowledge Prior (CKP) construction from SNOMED-CT subgraphs
The Clinical Knowledge Prior (CKP) formalizes medical reasoning through a curated directed subgraph of SNOMED-CT. For each candidate disease $h_i \in \mathcal{H}$, the subgraph encodes:
- Required anatomical lead sets $\mathcal{L}_{\text{req}}(h_i)$
- Mandatory physiological findings (e.g., ST elevation $\ge 1.0\text{ mm}$ in $\ge 2$ contiguous leads)
- Reciprocal relationship constraints (e.g., ST depression in V1–V3 during posterior MI)

The relevant subgraph $\mathcal{G}_{l, i}$ is encoded via graph convolutional layers and projected into the latent dimension:
$$\mathbf{Z}^K_{l, i} = \mathbf{W}_K \phi(\mathcal{G}_{l, i}) + \mathbf{b}_K \in \mathbb{R}^{768}$$

### Control path (ECG-only) context construction
The Control Path represents the unaugmented baseline. Its input context token sequence comprises the latent ECG embedding, structured clinical observations, candidate evidence tokens, and the candidate disease hypothesis:
$$\mathcal{C}^{(0)}_{l, i} = [\mathbf{Z}_{\text{ECG}}; \mathcal{O}_l; \mathcal{E}_{l, i}; h_i]$$

### Experimental path (ECG + CKP) context construction and additive fusion
The Experimental Path evaluates the knowledge-augmented system. It incorporates the Clinical Knowledge Prior via additive latent fusion:
$$\text{CKP}_{l, i} = \mathbf{Z}_{\text{ECG}} + \mathbf{Z}^K_{l, i}$$
The resulting context token sequence is:
$$\mathcal{C}^{(1)}_{l, i} = [\mathbf{Z}_{\text{ECG}}; \text{CKP}_{l, i}; \mathcal{O}_l; \mathcal{E}_{l, i}; h_i]$$

Crucially, the two conditions differ by *exactly one knowledge-derived term*, ensuring that differences in reasoning are attributable solely to the ontological prior.

### Shared Reasoning Transformer and dual output heads
Both contexts are processed by a **Shared Reasoning Transformer** ($\text{Trf}_\varphi$):
$$\mathbf{U}^{(c)}_{l, i} = \text{Trf}_\varphi(\mathcal{C}^{(c)}_{l, i})$$

The transformer features two distinct output heads:
1. **Diagnostic Score Head:** Computes uncalibrated logits $s^{(c)}_{l, i} = \mathbf{w}_s^\top \mathbf{U}^{(c)}_{l, i} + b_s$. Independent posterior probabilities are obtained via multi-label temperature scaling:
   $$P(h_i \mid \mathbf{X}_l) = \sigma(s^{(c)}_{l, i} / \beta_{l, i})$$
2. **Evidence Attention Head:** Computes a normalized cross-attention attribution matrix $\mathbf{A} \in \mathbb{R}^{|\mathcal{E}| \times |\mathcal{H}|}$:
   $$\mathbf{A} = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}}\right)$$
   quantifying the attribution weight of each physiological evidence item toward candidate hypotheses.

### Structured clinical reasoning trace schema
Phase 4 emits an inspectable JSON reasoning trace:
```json
{
  "record_id": "PTBXL_7",
  "lead_configuration": "2-lead",
  "active_leads": ["II", "V1"],
  "top_hypothesis": { "disease": "MI", "confidence": 0.84 },
  "supporting_evidence": [
    { "claim": "ST elevation in II", "lead": "II", "weight": 0.68, "is_observable": true },
    { "claim": "ST elevation in III", "lead": "III", "weight": 0.72, "is_observable": false },
    { "claim": "ST elevation in aVF", "lead": "aVF", "weight": 0.70, "is_observable": false }
  ],
  "uncertainty": 0.16
}
```

---

## PHASE 5: GROUNDING VERIFICATION LOGIC GATE

Phase 5 evaluates whether the model's generated claims are genuinely justified by the physical sensor data available at lead level $l$.

```
                        PHASE 5: GROUNDING VERIFICATION
                        
  1. INPUTS FROM PHASE 4:
     A. Reasoning Trace (Claims gamma_k)
     B. Candidate Evidence E_j
     C. Lead Availability Mask m_l (from Phase 2 bypass)
                                   |
                                   v
  +-----------------------------------------------------------------------------------+
  | 2. DETERMINISTIC VERIFICATION PROCESS                                             |
  |                                                                                   |
  |   2.1 Query Evidence Schema S(gamma_k) = (meas, L_req, theta):                    |
  |       Extract required lead set L_req(gamma_k) and threshold rule theta.          |
  |                                                                                   |
  |   2.2 Lead Availability Gate:                                                     |
  |       Calculate observability fraction:                                           |
  |           o(gamma_k, l) = |L_req(gamma_k) \cap A_l| / |L_req(gamma_k)|             |
  |       - If o(gamma_k, l) == 0:                                                    |
  |           * VERDICT = UNAVAILABLE                                                 |
  |       - If o(gamma_k, l) > 0:                                                     |
  |           * Inspect observed physical signal in observable leads:                 |
  |             - Does measured value satisfy threshold rule theta?                   |
  |                 * YES ---> VERDICT = SUPPORTED                                    |
  |                 * NO  ---> VERDICT = CONTRADICTED                                 |
  +-----------------------------------------------------------------------------------+
                                   |
                                   v
  3. OUTPUTS:
     - Claim-Level Verdicts: { SUPPORTED, CONTRADICTED, UNAVAILABLE }
     - Evidence Observability (EO_l) and Grounding Quality (GQ_l)
```

### Deterministic claim extraction from reasoning traces
The reasoning trace is decomposed into a set of discrete, testable claims $\Gamma_l = \{\gamma_1, \gamma_2, \dots, \gamma_K\}$.

### Independent evidence schema specification
Each claim $\gamma$ is mapped to an independently defined Evidence Schema:
$$\mathcal{S}(\gamma) = (\text{meas}(\gamma), \mathcal{L}_{\text{req}}(\gamma), \theta(\gamma))$$
where $\text{meas}(\gamma)$ defines the physical measurement (e.g., ST amplitude at J+80ms), $\mathcal{L}_{\text{req}}(\gamma)$ is the mandatory anatomical lead set, and $\theta(\gamma)$ is the clinical threshold rule ratified by clinical advisors.

### The ternary logic gate (SUPPORTED, CONTRADICTED, UNAVAILABLE)
The observability fraction for claim $\gamma$ under active lead set $\mathcal{A}_l = \{c \mid \mathbf{m}_l[c] = 1\}$ is:
$$o(\gamma, l) = \frac{|\mathcal{L}_{\text{req}}(\gamma) \cap \mathcal{A}_l|}{|\mathcal{L}_{\text{req}}(\gamma)|} \in [0, 1]$$

The ternary assignment operates deterministically:
1. **`UNAVAILABLE`:** Assigned if $o(\gamma, l) = 0$. The physical sensors required to verify the claim are completely absent.
2. **`SUPPORTED`:** Assigned if $o(\gamma, l) > 0$ and the physical signal in active leads satisfies $\theta(\gamma)$.
3. **`CONTRADICTED`:** Assigned if $o(\gamma, l) > 0$ but the physical signal directly violates $\theta(\gamma)$.

### Handling derived limb lead provenance in claim verification
In strict accordance with our Cabrera DAG provenance model, if a claim asserts a finding in derived lead III, aVR, aVL, or aVF, the verifier checks whether the parent leads (I and II) are present. If either parent lead is masked, the derived lead is classified as `unavailable_derived`, and the claim is assigned the verdict `UNAVAILABLE`.

---

## PHASE 6: RESULTS COMPILATION AND DEGRADATION METRICS

Phase 6 compiles claim-level verification logs into formal mathematical indices.

### Evidence Observability (EO_l)
Measures the proportion of clinically required evidence that remains physically observable under mask $l$:
$$EO_l = \frac{N^{\text{obs}}_l}{N^{\text{req}}_l} = \frac{\sum_{\gamma \in \Gamma_l} |\mathcal{L}_{\text{req}}(\gamma) \cap \mathcal{A}_l|}{\sum_{\gamma \in \Gamma_l} |\mathcal{L}_{\text{req}}(\gamma)|}$$

### Grounding Quality (GQ_l)
Quantifies the precision of clinical reasoning against observable evidence, penalizing contradicted claims:
$$GQ_l = \frac{N^{\text{sup}}_l}{N^{\text{sup}}_l + N^{\text{con}}_l}$$

### True Grounding Ratio (TGR_l)
A unified metric penalizing both ungrounded claims and unobservable evidence:
$$TGR_l = GQ_l \times EO_l = \frac{N^{\text{sup}}_l \cdot N^{\text{obs}}_l}{(N^{\text{sup}}_l + N^{\text{con}}_l) \cdot N^{\text{req}}_l}$$

### Reasoning Degradation Curve (RDC)
Constructed by plotting Grounding Quality as a function of active lead count:
$$\text{RDC}(l) = \{(l, GQ_l) \mid l \in [12, 10, 8, 6, 4, 2, 1]\}$$

### Area Under the Reasoning Degradation Curve (AURDC) trapezoidal integration
To summarize reasoning resilience across the entire degradation continuum $(L_1, \dots, L_7) = (12, 10, 8, 6, 4, 2, 1)$, we apply weighted trapezoidal quadrature:
$$\text{AURDC} = \frac{1}{11} \sum_{i=1}^{6} \frac{GQ_{L_i} + GQ_{L_{i+1}}}{2} (L_i - L_{i+1})$$
The normalization factor $11 = 12 - 1$ bounds $\text{AURDC} \in [0.0, 1.0]$.

### Reasoning Failure Boundary (RFB) derivation at safety threshold
Given predefined safety threshold $\tau_{\text{safety}} = 0.60$, the Reasoning Failure Boundary is:
$$\text{RFB} = \min \{l \in \Lambda \mid GQ_l \ge \tau_{\text{safety}}\}$$
The paired improvement conferred by the Clinical Knowledge Prior is formalized as:
$$\Delta\text{AURDC} = \text{AURDC}_{\text{CKP}} - \text{AURDC}_{\text{Control}}$$
$$\Delta\text{RFB}_{\text{imp}} = \text{RFB}_{\text{Control}} - \text{RFB}_{\text{CKP}}$$
A positive $\Delta\text{RFB}_{\text{imp}}$ confirms that the knowledge prior preserves reliable, grounded reasoning under more severe sensor scarcity.

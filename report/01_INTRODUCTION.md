# CHAPTER 1: INTRODUCTION

---

## MOTIVATION

### Clinical ubiquity and foundational role of the 12-lead electrocardiogram
The standard 12-lead electrocardiogram (ECG) represents the primary, non-invasive diagnostic standard in contemporary cardiovascular medicine. First conceptualized in the early twentieth century by Willem Einthoven through the string galvanometer and subsequently expanded by Frank Wilson and Emanuel Goldberger, the multi-lead ECG captures the temporal propagation of cardiac action potentials across the myocardium from twelve distinct spatial projections. Clinically, these twelve channels provide two orthogonal, complementary perspectives of cardiac electrodynamics: the six limb leads (I, II, III, aVR, aVL, and aVF) project the mean electrical vector onto the frontal (vertical) plane of the thoracic cavity, while the six precordial chest leads (V1, V2, V3, V4, V5, and V6) project cardiac dipoles onto the horizontal (transverse) plane. 

Together, these anatomical perspectives enable cardiologists and emergency physicians to rapidly diagnose a vast spectrum of life-threatening cardiac conditions. These include acute regional myocardial infarction, ventricular and supraventricular arrhythmias, intraventricular conduction delays (such as right and left bundle branch blocks), atrioventricular (AV) dissociation, chamber enlargements, and electrolyte disturbances. The high temporal fidelity of standard surface electrocardiography—typically acquired between 250 Hz and 1000 Hz at physical calibrations of 10 mm/mV—allows clinicians to measure millisecond-level depolarization and repolarization milestones, notably the P wave, PR interval, QRS complex, ST segment, and T wave.

### The emergence of deep learning foundation models in cardiology
Over the past decade, the rapid expansion of curated, digital biomedical repositories—such as PhysioNet PTB-XL, the MIMIC-IV-ECG database, the Chapman-Shaoxing cohort, and the UK Biobank—has catalyzed a profound paradigm shift in automated electrocardiographic analysis. Early algorithmic methods relied predominantly on handcrafted feature engineering, morphological landmark detectors (such as the Pan-Tompkins QRS algorithm), and classical machine learning classifiers (e.g., support vector machines and random forests). These approaches were subsequently superseded by deep convolutional neural networks (1D-CNNs, ResNets) and recurrent neural networks (LSTMs, GRUs), which demonstrated cardiologist-level diagnostic discrimination across selected rhythm categories on high-quality, static 12-lead recordings.

More recently, the machine learning community has transitioned toward large-scale self-supervised learning and Transformer-based architectures, giving rise to *ECG Foundation Models* (ECG-FMs). Notable architectures such as the wav2vec2-based ECG-FM, K-MERL, and GEM leverage self-supervised masked autoencoding, contrastive predictive coding, and multi-segment contrastive learning across hundreds of thousands of unlabelled multi-channel recordings. These foundation models construct rich, general-purpose latent representations ($Z_{\text{ECG}} \in \mathbb{R}^{D}$) that capture subtle spatio-temporal correlations across cardiac cycles. When fine-tuned or probed on downstream diagnostic tasks, foundation models consistently achieve state-of-the-art discrimination, demonstrating strong multi-label area under the receiver operating characteristic (AUROC) scores across diverse pathology taxonomies.

### The reality of physiological sensor degradation in acute and ambulatory care
Despite their remarkable success in controlled laboratory benchmarks, deep learning foundation models encounter severe real-world operational challenges when deployed outside of static, hospital-based cardiology suites. In real-world clinical environments—such as emergency triage bays, pre-hospital emergency medical service (EMS) transport, intensive care units (ICUs), rural diagnostic clinics, and consumer ambulatory monitoring—acquiring a complete, clean, and artifact-free 12-lead ECG is frequently impossible. 

Clinical practitioners and wearable devices operate under conditions of *progressive physiological information loss*, characterized by:
1. **Electrode Detachment and Motion Artifacts:** Patient movement, diaphoresis, respiratory excursions, chest compressions during cardiopulmonary resuscitation (CPR), and emergency transport vibrations cause electrodes to detach or produce transient signal flatlines.
2. **Time-Critical Triage Constraints:** In acute trauma or out-of-hospital cardiac arrest, attaching ten distinct physical electrodes (four limb clamps and six precordial suction cups) consumes critical minutes. Paramedics routinely apply reduced cable configurations to obtain instantaneous rhythm feedback.
3. **Hardware and Form-Factor Limitations:** Ambulatory telemetry monitors and Holter devices typically deploy 2 to 5 electrodes (e.g., telemetry configurations utilizing Leads II and V1 for simultaneous rhythm and bundle branch analysis). Consumer wearable health devices (such as smartwatches and single-lead chest patches) capture only a single bipolar vector, corresponding to Lead I or Lead II. Handheld mobile screening monitors (such as the AliveCor KardiaMobile 6L) capture the six frontal limb leads without any precordial chest leads.

Under these real-world scenarios, sensor scarcity is not random independent missingness; it corresponds to the physical elimination of entire anatomical projection planes.

### The illusion of robustness: Diagnostic accuracy versus evidence grounding
Contemporary machine learning evaluation paradigms benchmark models predominantly on *final diagnostic label accuracy* under fixed lead subsets. When a foundation model is evaluated under missing-lead conditions, researchers typically report classification metrics: Macro-AUROC, Area Under the Precision-Recall Curve (AUPRC), sensitivity, specificity, and F1-score.

However, evaluating models strictly on output labels creates a perilous *illusion of clinical robustness*. Because deep foundation models possess immense representational capacity, they learn complex, non-physiological statistical shortcuts, demographic priors, and disease co-occurrence patterns present in training cohorts. For example, if a patient presenting with an acute inferior myocardial infarction has concurrent atrial fibrillation or sinus tachycardia, a deep model may accurately predict the label *"Inferior Myocardial Infarction"* even if the key diagnostic inferior leads (Leads II, III, and aVF) are completely masked. 

In this scenario, the model's diagnostic label matches the ground-truth annotation. Conventional benchmark frameworks reward this prediction as a true positive. However, from a physical and electrophysiological perspective, confirming an acute inferior infarction is strictly impossible without observing concordant ST-segment elevation across at least two contiguous inferior leads. The model's prediction is not an evidence-grounded deduction; it is an ungrounded guess supported by statistical correlation.

### The silent failure mode: Ungrounded clinical claims in automated healthcare
When deep learning foundation models are coupled with clinical reasoning layers or generative report generators (such as Large Language Models or reasoning transformers), this ungrounded behavior manifests as a severe, safety-critical failure mode: *hallucinated clinical reasoning*.

Under severe sensor loss, generative reasoning modules frequently produce structured explanations that cite physiological findings from disconnected sensors. For instance, a model processing a single-lead rhythm strip (Lead II) may output:
> *"Diagnosis: Acute Anterior Myocardial Infarction. Supporting Evidence: Observed 2.5 mm ST-segment elevation in precordial leads V2, V3, and V4 with pathological Q waves."*

In this instance, leads V2, V3, and V4 were completely absent from the input. The explanation is clinically plausible, aesthetically convincing, and linguistically fluent, yet it is physically fabricated. In an emergency room or ambulatory triage setting, an ungrounded claim of this nature creates catastrophic clinical hazards:
1. It misleads attending clinicians into assuming that a comprehensive precordial survey has already been conducted, delaying life-saving 12-lead acquisition or cardiac catheterization.
2. It conceals sensor failure or lead disconnection from the clinical operator.
3. It violates medical ethics and regulatory safety mandates, which demand that medical diagnostic software provide an auditable, verifiable chain of evidence derived strictly from observed patient physiology.

Current research lacks an empirical, formal methodology to measure the boundary where clinical reasoning ceases to be truthfully grounded in physical sensor evidence. This research is dedicated to resolving this fundamental deficiency.

---

## BACKGROUND

### Electrophysiological basis of surface electrocardiography
The surface electrocardiogram is a macroscopic spatiotemporal recording of the extracellular electrical potentials generated by the synchronized depolarization and repolarization of billions of cardiac myocytes. During each cardiac cycle, the sinoatrial (SA) node initiates an electrical impulse that traverses the atria (producing the P wave), delays at the atrioventricular (AV) node (forming the PR segment), propagates rapidly down the bundle of His, bundle branches, and Purkinje network to depolarize the ventricular myocardium (generating the high-amplitude QRS complex), and concludes with ventricular repolarization (producing the ST segment and T wave).

```
   Voltage (mV)
        ^
        |                  R
        |                 / \
        |                /   \
        |               /     \
        |        P     /       \            T
        |      _---_  /         \         _---_
        |     /     \/           \       /     \
        |----+-------Q-----------S------+-------+---> Time (ms)
        |    |<-PR ->|<-  QRS  ->|<-ST->|
        |    |       interval    |segment
```

The standard 12-lead arrangement provides specific anatomical viewpoints:
1. **Bipolar Limb Leads (I, II, III):** Formulated by Einthoven's triangle, where electrodes are placed on the right arm (RA), left arm (LA), and left leg (LL). Einthoven's Law dictates that:
   $$\text{Lead I} = \Phi_{\text{LA}} - \Phi_{\text{RA}}$$
   $$\text{Lead II} = \Phi_{\text{LL}} - \Phi_{\text{RA}}$$
   $$\text{Lead III} = \Phi_{\text{LL}} - \Phi_{\text{LA}} = \text{Lead II} - \text{Lead I}$$
2. **Augmented Unipolar Limb Leads (aVR, aVL, aVF):** Formulated by Emanuel Goldberger, utilizing Wilson's Central Terminal (WCT) with the exploring electrode disconnected from the central terminal:
   $$\text{aVR} = \Phi_{\text{RA}} - \frac{\Phi_{\text{LA}} + \Phi_{\text{LL}}}{2} = -\frac{\text{Lead I} + \text{Lead II}}{2}$$
   $$\text{aVL} = \Phi_{\text{LA}} - \frac{\Phi_{\text{RA}} + \Phi_{\text{LL}}}{2} = \text{Lead I} - \frac{\text{Lead II}}{2}$$
   $$\text{aVF} = \Phi_{\text{LL}} - \frac{\Phi_{\text{RA}} + \Phi_{\text{LA}}}{2} = \text{Lead II} - \frac{\text{Lead I}}{2}$$
3. **Unipolar Precordial Leads (V1–V6):** Positioned across specific intercostal spaces on the anterior and lateral chest wall, referenced against Wilson's Central Terminal $\Phi_{\text{WCT}} = \frac{1}{3}(\Phi_{\text{RA}} + \Phi_{\text{LA}} + \Phi_{\text{LL}})$. V1 and V2 observe the ventricular septum; V3 and V4 observe the anterior wall of the left ventricle; V5 and V6 observe the low lateral myocardium.

Crucially, of the 12 conventional leads, only eight channels are mathematically independent (two limb leads and six precordial leads). Any derived limb lead ($\text{III}, \text{aVR}, \text{aVL}, \text{aVF}$) is an algebraic linear combination of Leads I and II. Consequently, in an evidence-grounded verification framework, derived leads cannot be counted as independent physical evidence if their parent leads are missing or already accounted for.

### Foundation model architectures for sequential biomedical signals
In recent years, foundation models have transformed biomedical signal processing. Moving beyond supervised networks trained on single datasets, foundation models utilize self-supervised pre-training on vast multi-institutional databases to learn robust, transferrable representations.

In this research, we focus on the authoritative **ECG-FM** backbone (`wav2vec2_cmsc`), an architecture inspired by the Wav2Vec 2.0 framework adapted specifically for multi-channel electrocardiograms. The architecture comprises:
- **Convolutional Feature Encoder:** A multi-layer temporal convolutional network that downsamples high-frequency raw ECG voltage samples into localized temporal feature blocks.
- **Transformer Context Network:** A 12-layer bidirectional Transformer encoder with 12 self-attention heads, a hidden embedding dimension of $d_{\text{model}} = 768$, and a feed-forward network (FFN) dimension of $3072$, totaling 90.88 million parameters.
- **Pre-training Objective:** Contrastive Multi-Segment Coding (CMSC) combined with random lead and temporal masking ($p = 0.5$). CMSC encourages the model to recognize representations across adjacent cardiac segments while distinguishing them from negative distractor segments.

Crucially, because ECG-FM was exposed to random lead masking during pre-training, its internal latent embeddings exhibit an inherent degree of representational stability under missing channels. However, whether downstream clinical reasoning heads remain truthfully anchored to observable evidence as leads are progressively removed remains an unaddressed scientific question.

### Knowledge representation and clinical ontologies in medicine
Medical decision-making is inherently structured and hierarchical. Clinicians do not map raw sensory inputs directly to diagnostic categories in an uninterpretable black-box manner; rather, they extract intermediate physiological biomarkers and cross-reference them against established ontological knowledge.

The Systematized Nomenclature of Medicine -- Clinical Terms (**SNOMED-CT**) represents the world's most comprehensive, multilingual clinical healthcare terminology. SNOMED-CT models clinical knowledge as an ontological directed graph where nodes represent clinical concepts (e.g., `Inferior myocardial infarction`, `ST segment elevation`, `Right bundle branch block`) and directed edges represent formalized semantic relationships:
- `finding_site` (e.g., linking `Inferior myocardial infarction` to `Inferior wall of left ventricle`)
- `has_finding` (e.g., linking a condition to its mandatory electrographic signs)
- `requires_lead` (e.g., specifying that anterior findings require contiguous leads V1–V4)
- `reciprocal_to` (e.g., defining reciprocal ST depression in lateral leads during acute inferior STEMI)

By projecting SNOMED-CT subgraphs into computational vector spaces, artificial intelligence architectures can introduce structural inductive biases that constrain reasoning models to clinically valid paths, preventing arbitrary or biologically impossible inferences.

### Formal verification, evidence grounding, and AI safety in healthcare
Artificial Intelligence safety in high-stakes clinical domains requires formal verification mechanisms that separate predictive confidence from factual evidence grounding. Contemporary research in AI explainability has demonstrated that post-hoc interpretability methods (such as saliency maps, Grad-CAM, or unconstrained self-attention weights) frequently reflect model rationalization rather than faithful reasoning.

To guarantee clinical safety, verification must operate as an independent, deterministic gate external to the neural model. In formal logic, this requires evaluating each generated claim against a known physical observation mask. Rather than treating evidence evaluation as a binary decision (True vs. False), clinical safety demands a *ternary logic framework*:
- **`SUPPORTED`:** The required sensor leads are physically active, signal quality is verified, and the measured waveform satisfies clinical threshold criteria.
- **`CONTRADICTED`:** The required sensor leads are physically active and of sufficient quality, but the observed physiological waveform directly refutes the claim.
- **`UNAVAILABLE`:** The physical sensor channels required to evaluate the claim are absent from the input.

Distinguishing between `CONTRADICTED` and `UNAVAILABLE` is paramount: the absence of evidence must never be conflated with evidence of absence. A claim referencing an unavailable lead indicates that the system is reasoning beyond its observable sensor space.

---

## PROBLEM DEFINITION

### Mathematical formulation of progressive information loss
Let an acquired twelve-lead surface electrocardiogram recording be denoted as a standardized voltage-time matrix:
$$\mathbf{X} \in \mathbb{R}^{12 \times T}$$
where $12$ corresponds to the canonical sequence of clinical leads:
$$\mathcal{C} = [\text{I}, \text{II}, \text{III}, \text{aVR}, \text{aVL}, \text{aVF}, \text{V1}, \text{V2}, \text{V3}, \text{V4}, \text{V5}, \text{V6}]$$
and $T$ represents the fixed temporal sample length (e.g., $T = 5,000$ samples for a 10.0-second recording at $500\text{ Hz}$, or $T = 1,000$ samples at $100\text{ Hz}$).

We define a controlled, deterministic degradation schedule comprising seven ordered lead-availability levels:
$$\Lambda = (12, 10, 8, 6, 4, 2, 1)$$
where each scalar $l \in \Lambda$ defines the exact number of physically active sensor leads.

At any degradation stage $l \in \Lambda$, sensor availability is defined by a binary lead-availability mask vector:
$$\mathbf{m}_l = (m_1, m_2, \dots, m_{12})^\top \in \{0, 1\}^{12}, \quad \text{such that } \sum_{j=1}^{12} m_j = l$$
The degraded physiological observation presented to the computational system is formalized as the Hadamard (element-wise) product across the temporal dimension:
$$\mathbf{X}_l = \mathbf{X} \odot (\mathbf{m}_l \mathbf{1}_T^\top) \in \mathbb{R}^{12 \times T}$$
where $\mathbf{1}_T \in \mathbb{R}^T$ is a vector of ones. Inactive channels ($m_j = 0$) are strictly clamped to a zero constant vector $\mathbf{0}_T$.

### Input specifications
The proposed evaluation framework ingests four distinct, rigorously defined inputs:
1. **Multi-Lead ECG Voltage Tensor ($\mathbf{X}_l$):** The masked voltage-time matrix representing the physical signals that survived degradation stage $l$.
2. **Binary Lead-Availability Mask ($\mathbf{m}_l$):** The explicit 12-dimensional binary vector denoting active physical sensor observability. The mask is forwarded directly to both the reasoning engine and the independent verifier.
3. **Diagnostic Candidate Hypotheses Space ($\mathcal{H}$):** A closed, frozen space of eight clinically distinct cardiovascular conditions defined in multi-hot format:
   $$\mathcal{H} = \{h_1: \text{NORM}, h_2: \text{AFIB}, h_3: \text{RBBB}, h_4: \text{LBBB}, h_5: \text{SBRAD}, h_6: \text{STACH}, h_7: \text{1AVB}, h_8: \text{MI}\}$$
4. **Clinical Knowledge Prior Graph ($\mathcal{G}_{\text{CKP}}$):** A directed acyclic subgraph extracted from SNOMED-CT encoding verified physiological relations, required lead sets $\mathcal{L}_{\text{req}}$, and threshold criteria $\theta$ linking candidates $h_i \in \mathcal{H}$ to observable waveform features.

### Output specifications
For every input pair $(\mathbf{X}_l, \mathbf{m}_l)$, the system produces four auditable outputs:
1. **Calibrated Posterior Probability Distribution ($\hat{\mathbf{y}}_l$):** Multi-label posterior probabilities across all eight disease candidates:
   $$\hat{\mathbf{y}}_l = (\hat{y}_{l,1}, \hat{y}_{l,2}, \dots, \hat{y}_{l,8}) \in [0, 1]^8$$
   calibrated via independent per-lead temperature scaling parameters $\beta_l$.
2. **Structured Clinical Reasoning Trace ($\mathcal{T}_l$):** An inspectable trace comprising the top diagnostic hypothesis, associated confidence, predictive uncertainty ($1 - \text{confidence}$), and an ordered set of discrete clinical claims $\Gamma_l = \{\gamma_1, \gamma_2, \dots, \gamma_K\}$. Each claim $\gamma_k$ encapsulates an asserted finding $f_k$, contributing leads $\mathcal{L}_{\text{contrib}}(\gamma_k)$, and cross-attention weight $\alpha_k$.
3. **Ternary Grounding Verdicts ($\mathcal{V}_l$):** An independent verification mapping assigning each claim $\gamma_k \in \Gamma_l$ to an immutable state:
   $$\mathcal{V}(\gamma_k) \in \{\text{SUPPORTED}, \text{CONTRADICTED}, \text{UNAVAILABLE}\}$$
4. **Aggregate Degradation Indices:** Quantitative metrics compiling reasoning fidelity at level $l$, specifically Evidence Observability ($EO_l$), Grounding Quality ($GQ_l$), True Grounding Ratio ($TGR_l$), Area Under the Reasoning Degradation Curve (AURDC), and the Reasoning Failure Boundary (RFB).

### Operational and scientific constraints
To guarantee absolute experimental validity, eliminate confounding factors, and ensure reproducibility, the research operates under five strict constraints:
- **Constraint C1 (Strict Prohibition of Generative Inpainting):** Generative lead reconstruction, deep variational inpainting, autoencoding waveform synthesis, and mathematical lead estimation are strictly prohibited. Missing leads must remain missing ($0.00000$ amplitude). Generative reconstruction introduces synthetic artifacts that obscure authentic failure boundaries under sensor loss.
- **Constraint C2 (Morphology-Preserving Zero-Phase Preprocessing):** All digital filtering operations must execute in a zero-phase, bidirectional manner (`scipy.signal.filtfilt`) at $500\text{ Hz}$ prior to polyphase decimation. Net group delay must equal zero ($\phi(\omega) = 0$) to prevent phase shifts that artificially displace QRS boundaries, J-points, or ST segments.
- **Constraint C3 (Patient-Stratified Disjoint Partitioning):** All dataset splits (training, validation, held-out test) must enforce strict patient-level quarantine. Zero patient overlap is permitted across splits, preventing subject-specific cardiac anatomy from leaking into test evaluations.
- **Constraint C4 (Controlled Counterfactual Comparison):** The baseline Control Path (ECG-only) and experimental CKP Path must share identical input recordings, lead masks, frozen foundation model backbones, latent embeddings $Z_{\text{ECG}}$, discrete observations $O_l$, candidate hypotheses $\mathcal{H}$, and verification rules. Divergence is isolated strictly to CKP relational tokens.
- **Constraint C5 (Evidence-Limited Knowledge Integration):** The Clinical Knowledge Prior is strictly prohibited from asserting unobserved physiological findings or overriding the physical lead-availability mask. The CKP serves as an inductive structural constraint on attention, not a generator of synthetic evidence.

### Formal hypothesis and success criteria

```
+---------------------------------------------------------------------------------------------------+
|                                      CENTRAL HYPOTHESIS                                           |
| "A clinical diagnostic prediction is safe if and only if every physiological claim made in its    |
| reasoning trace can be validated against signals present in the available lead subset."            |
+---------------------------------------------------------------------------------------------------+
```

Let $GQ(l)$ represent the Grounding Quality measured at lead-availability level $l \in \Lambda$. We establish the formal clinical safety threshold as:
$$\tau_{\text{safety}} = 0.60$$
The **Reasoning Failure Boundary (RFB)** is defined as the lowest lead count at which the model satisfies the clinical safety threshold:
$$\text{RFB} = \min \{l \in \Lambda \mid GQ(l) \ge \tau_{\text{safety}}\}$$
A lower RFB indicates greater resilience to physiological sensor degradation. If no evaluated level satisfies the threshold, the RFB is designated as undefined ($\emptyset$).

The research establishes three mandatory, quantitative success criteria:
1. **Primary Metric (AURDC Resilience):** The knowledge-augmented system must achieve a normalized Area Under the Reasoning Degradation Curve of:
   $$\text{AURDC}_{\text{CKP}} \ge 0.80$$
   demonstrating sustained reasoning fidelity across the entire degradation continuum.
2. **Boundary Extension ($\Delta\text{RFB}$):** The Clinical Knowledge Prior must delay the Reasoning Failure Boundary by at least two physical leads relative to the unaugmented baseline:
   $$\Delta\text{RFB}_{\text{imp}} = \text{RFB}_{\text{Control}} - \text{RFB}_{\text{CKP}} \ge 2\text{ leads}$$
   (e.g., maintaining acceptable grounding at 2 leads if the baseline fails at 4 leads).
3. **Zero Hallucination Inflation:** Improvements in $GQ(l)$ and AURDC must be achieved without increasing the rate of ungrounded assertions or compromising diagnostic calibration across multi-cohort test partitions.

---

## ORGANIZATION OF THE REPORT

The remainder of this project report is structured into seven comprehensive chapters:

- **CHAPTER 2: LITERATURE SURVEY** provides an extensive, critical review of prior scholarship. It surveys contemporary literature across ECG foundation models, partial-lead acquisition and reconstruction methods, knowledge-enhanced neural architectures, AI factuality and medical hallucination detection, and safety verification envelopes. It systematically identifies key research gaps in existing evaluation paradigms and formalizes the project's research objectives.
- **CHAPTER 3: PROPOSED METHODOLOGY** provides the comprehensive mathematical, architectural, and algorithmic specification of the proposed 6-phase closed-loop reasoning architecture: Phase 1 (Data Standardization and Quality Gates), Phase 2 (Controlled Masking and Cabrera DAG Provenance), Phase 3 (Frozen Foundation Model Adaptation and Deterministic Observation Extraction), Phase 4 (Dual-Path Reasoning Transformer and CKP Fusion), Phase 5 (Independent Grounding Verification Logic Gate), and Phase 6 (Mathematical Metrics Compilation).
- **CHAPTER 4: EXPERIMENTAL RESULTS AND PERFORMANCE ANALYSIS** delivers the complete empirical evaluation of the framework across 51,625 test evaluations on the held-out PTB-XL, Chapman-Shaoxing, and MIMIC-IV-ECG cohorts. It details the computing ecosystem, analyzes representation sensitivity and cosine drift, evaluates multi-seed stability, investigates anatomical non-monotonicity (why 4 leads outperform 6 leads), and computes AURDC, RFB, and True Grounding Ratios.
- **CHAPTER 5: SOCIAL IMPACT AND SUSTAINABILITY** examines the broader societal, translational, and environmental dimensions of this work. It analyzes clinical safety in emergency triage, healthcare democratization in low-resource clinics, alignment with FDA SaMD and EU AI Act regulatory frameworks, patient privacy, and the computational energy efficiency of deploying frozen foundation models on edge devices.
- **CHAPTER 6: CONCLUSIONS AND FUTURE WORK** synthesizes the core findings of the project, details the limitations and threats to validity of the current study, and outlines concrete avenues for future exploration, including prospective clinical trials, dynamic active-sensing acquisition, and multi-modal cardiovascular foundation models.
- **CHAPTER 7: REFERENCES** lists all cited academic literature strictly formatted in APA style (7th edition) with verified LaTeX `\url{}` DOI hyperlinks.
- **CHAPTER 8: APPENDIX** provides supplementary reference material, including the 8-disease benchmark definition, authoritative lead configuration tables, Cabrera DAG mathematical derivations, hardware-software environment specifications, and representative reasoning trace dossiers.

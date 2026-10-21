# CHAPTER 2: LITERATURE SURVEY

---

## PAPERS RELATED TO THE PROBLEM IDENTIFIED

The challenge of deploying artificial intelligence safely for automated electrocardiographic interpretation under sensor degradation lies at the confluence of five major academic disciplines: (1) self-supervised foundation models for biomedical signals, (2) partial-lead ECG interpretation and waveform reconstruction, (3) knowledge-enhanced neural networks and clinical ontologies, (4) factuality, hallucination detection, and evidence grounding in clinical machine learning, and (5) formal safety verification, selective classification, and operational envelopes. This section provides an extensive, critical review of foundational and contemporary literature across these domains.

### Electrocardiogram foundation models and self-supervised representation learning
In clinical cardiology, automated ECG interpretation has historically relied on algorithmic waveform extraction and rule-based decision trees. The seminal work by Wagner et al. (2020) establishing the **PTB-XL** database—a publicly accessible, clinically balanced dataset of 21,799 clinical 12-lead recordings annotated with standardized SCP-ECG statements—fundamentally transformed the field by providing a standardized, reproducible benchmark for deep learning evaluation. 

Subsequently, Strodthoff et al. (2021) conducted an extensive benchmarking study on PTB-XL, comparing convolutional neural networks (1D-CNNs, ResNets) and recurrent architectures (LSTMs). Their work established baseline performance metrics across five diagnostic superclasses: Normal ECG (NORM), Myocardial Infarction (MI), ST/T Changes (STTC), Conduction Disturbance (CD), and Hypertrophy (HYP). Strodthoff et al. demonstrated that deep residual networks (specifically ResNet-1D-101) achieved Macro-AUROC scores exceeding $0.92$, matching certified cardiologists in distinguishing normal from abnormal recordings. However, the authors explicitly noted that these models operated as black boxes, lacking mechanistic traceability and exhibiting vulnerability to subtle distribution shifts.

In parallel, Ribeiro et al. (2020) demonstrated the clinical efficacy of deep neural networks on a massive telemedicine database of over 2.3 million 12-lead ECG recordings collected across 811 counties in Minas Gerais, Brazil. Their deep residual network achieved diagnostic F1-scores surpassing $80\%$ across six common cardiac arrhythmias (atrial fibrillation, first-degree AV block, complete left bundle branch block, complete right bundle branch block, sinus bradycardia, and sinus tachycardia), with specificities exceeding $99\%$. Despite these impressive results, the Ribeiro et al. architecture was trained and evaluated strictly on complete 12-lead inputs. When individual leads were obscured or corrupted by electrode contact noise, the network's diagnostic calibration degraded unpredictably.

Recognizing the limitations of supervised learning on static cohorts, recent research has pivoted toward *ECG Foundation Models* trained via self-supervised objectives. Inspired by breakthroughs in natural language processing (Devlin et al., 2019) and speech processing (Baevski et al., 2020), foundation models learn universal representations by solving pretext tasks over massive, unlabelled bio-signal repositories. 

The **ECG-FM** architecture represents a landmark milestone in this domain. Adapted from the Wav2Vec 2.0 framework (Baevski et al., 2020), ECG-FM combines a multi-layer temporal convolutional feature extractor with a 12-layer bidirectional Transformer context network (768 hidden dimension, 12 attention heads, 90.88M parameters). Pre-trained on PhysioNet's MIMIC-IV-ECG database (comprising over 800,000 recordings from intensive care admissions), ECG-FM utilizes Contrastive Multi-Segment Coding (CMSC) combined with random lead and temporal masking ($p = 0.5$). CMSC forces the Transformer encoder to optimize mutual information between contextualized representations of adjacent cardiac cycles while distinguishing them from temporal distractor segments. When adapted to downstream classification tasks, frozen ECG-FM representations achieve state-of-the-art discrimination while requiring minimal task-specific fine-tuning.

Building upon foundation model paradigms, Liu et al. (2025) introduced **K-MERL** (Knowledge-enhanced Multimodal ECG Representation Learning). K-MERL addresses arbitrary-lead inputs through a flexible masking mechanism coupled with structured medical report generation. By aligning ECG signal representations with clinical text embeddings using contrastive objectives, K-MERL demonstrated impressive zero-shot classification and partial-lead representation learning. However, K-MERL evaluated partial-lead performance predominantly through linear probing classification accuracy. It did not verify whether the generated text explanations or internal attention mechanisms were faithfully grounded in the physical leads available to the model during inference.

### Partial-lead ECG interpretation and generative waveform reconstruction
A substantial body of clinical and engineering literature has investigated reduced-lead electrocardiography, motivated by the practical need for portable, wearable, and emergency cardiac monitoring.

Early classical approaches explored mathematical lead synthesis based on Einthoven's triangle, Goldberger's equations, and Burger's lead field theory. Frank (1956) formulated the orthogonal 3-lead vectorcardiogram (VCG; leads X, Y, Z), demonstrating that cardiac electrical activity can be approximated as a time-varying three-dimensional equivalent electrical dipole. Numerous investigators (e.g., Nelwan et al., 2004; Dawson et al., 2009) developed linear transformation matrices (e.g., the Dower transform and the Kors quasi-orthogonal matrix) to synthesize a 12-lead ECG from 3-lead or 4-lead subsets. While linear reconstruction matrices adequately reconstruct QRS complexes under normal conduction, clinical studies (e.g., Atoui et al., 2010) revealed that linear synthesis fails catastrophically in regional pathologies, such as localized anterior or inferior myocardial infarctions, where non-dipolar electrical vectors dominate.

With the advent of deep generative modeling, researchers introduced deep autoencoders, Variational Autoencoders (VAEs), and Generative Adversarial Networks (GANs) to synthesize missing ECG leads. For example, Hao et al. (2021) and Li et al. (2023) developed conditional GANs (cGANs) to impute precordial leads V1–V6 from limb leads I and II. These models achieve low mean squared reconstruction error (RMSE) on synthetic test sets and generate visually realistic waveforms.

However, from an AI safety and clinical verification perspective, *generative lead inpainting introduces severe hazards*:
1. **Hallucination of Pathological Markers:** A generative network trained on statistical priors may artificially reconstruct an elevated ST-segment or pathological Q-wave in an unobserved precordial lead simply because the patient has risk factors or rhythm disturbances in the observable limb leads. 
2. **Confounding Authentic Information Scarcity:** If an evaluation framework permits synthetic waveform imputation, researchers cannot determine whether a correct diagnostic prediction stems from genuine physiological information present in the active sensors or from the generative network's hallucinated synthesis.

Consequently, modern safety paradigms demand that reduced-lead evaluation be conducted under *authentic sensor scarcity*, strictly prohibiting generative lead imputation.

### Knowledge-enhanced neural architectures and medical knowledge graphs
Deep neural networks are notoriously data-hungry and prone to learning spurious correlations. To mitigate these vulnerabilities, knowledge-enhanced artificial intelligence integrates structured, symbolic domain knowledge into neural architectures.

In the biomedical domain, clinical knowledge is codified within standardized ontologies, most notably the Unified Medical Language System (UMLS) and **SNOMED-CT** (Systematized Nomenclature of Medicine -- Clinical Terms). SNOMED-CT formalizes medical terminology as a directed knowledge graph where concepts (e.g., `Inferior Myocardial Infarction`, `ST Elevation`, `Prolonged PR Interval`) are interconnected via formalized semantic relationships (e.g., `has_finding`, `finding_site`, `requires_lead`).

Methodologically, integrating knowledge graphs into deep neural pipelines is achieved through two primary mechanisms:
- **Graph Neural Network (GNN) Embeddings:** Graph Convolutional Networks (GCNs; Kipf & Welling, 2017; Wu et al., 2019) or Graph Attention Networks (GATs; Veličković et al., 2018) propagate topological node features across ontology subgraphs, producing dense knowledge vectors ($\mathbf{Z}_K \in \mathbb{R}^D$). These knowledge embeddings are subsequently fused with latent signal representations through concatenation, gated cross-attention, or additive projection.
- **Relational Prompt Conditioning:** Structured clinical relationships are serialized into discrete relational tokens that serve as contextual memory prompts within shared Transformer attention blocks (e.g., Zhang et al., 2021; Yasunaga et al., 2021 in QA-GNN).

In clinical electrocardiography, integrating clinical knowledge priors (CKP) provides vital inductive biases. When physiological sensor information is degraded (e.g., in a 2-lead telemetry configuration), the knowledge prior restricts the reasoning transformer's hypothesis search space to clinically valid finding-disease relationships, preventing the model from associating arbitrary latent noise with implausible diagnoses. However, a critical research question remains: *Does introducing structured clinical knowledge improve genuine evidence grounding, or does it merely teach the model to generate more fluent, persuasive hallucinations?* This project is specifically designed to answer this question.

### Factuality, hallucination detection, and evidence grounding in clinical AI
As artificial intelligence transitions toward generative report generation and multi-modal clinical reasoning, the phenomena of *hallucination* and *unfaithful rationalization* have emerged as paramount safety concerns (Rajpurkar et al., 2022).

In natural language processing and clinical text generation, Jacovi and Goldberg (2020) established a vital distinction between *plausibility* and *faithfulness*:
- **Plausibility:** How convincing, fluent, and medically reasonable an explanation appears to a human clinician.
- **Faithfulness:** The degree to which an explanation accurately reflects the actual evidence and computational reasoning utilized by the underlying model.

Deep neural models frequently generate highly plausible explanations that are entirely unfaithful. In medical imaging and electrocardiography, Lakkaraju et al. (2020) demonstrated that saliency maps and post-hoc attention visualizations frequently suffer from confirmation bias: attention heatmaps highlight features that human evaluators expect to see, even when the model's underlying prediction was driven by non-physiological edge artifacts or background text markers.

In automated ECG interpretation, evidence grounding requires that every asserted claim be verified against physical signal reality. Current clinical explainability benchmarks (such as BLEU, ROUGE, or BERTScore on generated free-text reports) measure linguistic overlap against reference cardiologist reports. These metrics do not evaluate factual grounding: if a reference report describes an inferior infarction based on an original 12-lead ECG, an automated report generator evaluated on a degraded 2-lead input that recites the same text will achieve a near-perfect ROUGE score, despite having hallucinated the unobserved inferior leads!

Addressing this critical limitation requires a formal, claim-level evidence verification framework that unpacks generated reasoning traces into discrete, adjudicable claims and verifies each claim against the active sensor mask using deterministic electrophysiological criteria.

### Formal verification, calibration, and safety envelopes in medical machine learning
In safety-critical engineering domains—such as avionics, autonomous robotics, and radiation therapy—computational systems are not deployed without rigorously certified *operational safety envelopes*. A safety envelope defines the precise input boundaries and environmental conditions under which a system's failure probability remains bounded below an acceptable threshold.

In machine learning, safety certification has been explored through:
1. **Selective Classification (Classification with Abstention):** Pioneered by Geifman and El-Yaniv (2017, 2019), selective classification equips models with a selection function $g(x) \in \{0, 1\}$ that enables the network to abstain from predicting when its estimated confidence falls below an operating threshold $\theta$, thereby guaranteeing a predefined target risk level.
2. **Conformal Prediction:** Developed by Vovk et al. (2005) and popularized in medical imaging by Angelopoulos and Bates (2021), conformal prediction generates valid prediction sets that guarantee finite-sample marginal coverage (e.g., $95\%$ coverage) under arbitrary distribution shifts.
3. **Probability Calibration:** Guo et al. (2017) demonstrated that modern deep neural networks with batch normalization and high capacity are frequently poorly calibrated, exhibiting extreme overconfidence in incorrect predictions. Platt scaling and temperature scaling provide effective, computationally efficient mechanisms to align predicted probabilities with empirical accuracy.

While selective classification and conformal prediction provide mathematical guarantees on output label risk, they have not been extended to *reasoning validity under progressive structural deprivation*. They answer the question: *"Is the predicted label likely to be correct?"* They do not answer the question: *"Does the model possess sufficient observable physical sensor evidence to justify its clinical reasoning?"*

Formalizing the **Reasoning Failure Boundary (RFB)** across controlled lead-availability levels bridges this critical methodological divide.

---

## GAPS IDENTIFIED

Based on a systematic review of the academic literature, we identify five fundamental research gaps that undermine the clinical safety and evaluation of contemporary ECG foundation models:

```
+----------------------------------------------------------------------------------------------------+
|                                  IDENTIFIED RESEARCH GAPS IN ECG AI                                |
+----------------------------------------------------------------------------------------------------+
| Gap 1: Diagnostic Accuracy Bias          | Models evaluated strictly on label correctness,         |
|                                          | rewarding ungrounded guesses from dataset shortcuts.    |
| Gap 2: Confounding Generative Inpainting | Missing leads synthesized via GANs/VAEs, obscuring      |
|                                          | authentic sensor failure limits with fake evidence.     |
| Gap 3: Absence of Claim-Level Grounding  | Explanations evaluated via linguistic NLP metrics       |
|                                          | rather than physical sensor mask verification.          |
| Gap 4: Lack of Failure Boundary Curves   | Performance measured at static lead levels (e.g., 12 vs |
|                                          | 1) without mapping the continuous breakdown boundary.   |
| Gap 5: Unverified Role of Clinical Prior | Unclear whether structured ontologies (SNOMED-CT)       |
|                                          | anchor authentic evidence or induce persuasive guesses. |
+----------------------------------------------------------------------------------------------------+
```

### Gap 1: The diagnostic accuracy bias in contemporary ECG evaluation
Existing benchmark studies evaluate foundation models almost exclusively using classification accuracy, AUROC, and macro-F1 on static, fixed-lead cohorts. This paradigm creates a false sense of security by failing to detect when a model predicts the correct diagnosis through non-physiological shortcuts, demographic biases, or learned label co-occurrences while its clinical reasoning is ungrounded.

### Gap 2: Confounding effects of generative lead imputation under sensor scarcity
Prior literature on reduced-lead ECG interpretation has focused heavily on generative reconstruction (autoencoders, GANs, linear Dower transforms) to reconstruct missing channels prior to classification. This approach conflates signal synthesis with clinical reasoning, making it impossible to determine the true empirical limits of clinical AI under authentic information scarcity.

### Gap 3: Absence of claim-level evidence verification against physical sensor masks
Current interpretability methods evaluate model explanations using post-hoc attention visualizations or n-gram linguistic overlap (BLEU/ROUGE) against free-text cardiologist reports. No existing framework decomposes clinical reasoning traces into individual claims and deterministically audits them against the physical sensor mask using an immutable, expert-ratified evidence schema.

### Gap 4: Lack of continuous failure boundary characterization across progressive degradation
Most existing studies compare complete 12-lead acquisitions against extreme single-lead monitors in isolation. The literature lacks a continuous, fine-grained degradation ladder ($12 \to 10 \to 8 \to 6 \to 4 \to 2 \to 1$) capable of tracing the precise trajectory of reasoning breakdown, mapping non-monotonicities (such as anatomical vector preservation), and identifying the exact operating threshold where reasoning collapses.

### Gap 5: Insufficient investigation of structured clinical priors as non-generative reasoning anchors
While knowledge graphs (SNOMED-CT) have been explored for multi-modal report generation, their potential as non-generative inductive constraints to preserve evidence-grounded reasoning under severe sensor degradation has not been investigated in a controlled, counterfactual experimental design.

---

## RESEARCH OBJECTIVES

To overcome these identified research gaps, this project formulates four rigorous, sequential research objectives:

### Primary objective: Formalizing the reasoning failure boundary framework
To establish a rigorous mathematical and experimental framework that models progressive physiological information loss across standard ECG configurations containing 12, 10, 8, 6, 4, 2, and 1 available leads, and to formalize the **Reasoning Failure Boundary (RFB)** and the **Area Under the Reasoning Degradation Curve (AURDC)** as standardized metrics for clinical AI safety.

### Secondary objective: Development of the 6-phase closed-loop architecture
To design, implement, and audit a modular 6-phase processing architecture encompassing:
1. Multi-cohort data standardization across PTB-XL, Chapman-Shaoxing, and MIMIC-IV-ECG.
2. Controlled lead masking governed by a Cabrera lead provenance directed acyclic graph (DAG), strictly excluding generative inpainting.
3. Dual-representation extraction combining continuous 768-dimensional latent vectors from a frozen 90.8M-parameter foundation model (`wav2vec2_cmsc`) with discrete, rule-extracted clinical observations ($O_l$).
4. Dual-path clinical reasoning within a shared Reasoning Transformer.
5. Deterministic grounding verification.
6. Mathematical degradation compilation.

### Tertiary objective: Deterministic ternary grounding verification logic gate
To develop and validate an independent, deterministic grounding verification logic gate governed by an expert-ratified Evidence Schema that evaluates every generated clinical claim against observable physical signals, strictly categorizing each claim into an immutable ternary state: `SUPPORTED`, `CONTRADICTED`, or `UNAVAILABLE`.

### Quaternary objective: Empirical evaluation of the clinical knowledge prior on multi-cohort benchmarks
To conduct a controlled, counterfactual comparison between an unaugmented ECG-only Control Path and an Experimental Path augmented with a Clinical Knowledge Prior (CKP) derived from SNOMED-CT. The objective is to determine whether the CKP extends the Reasoning Failure Boundary by at least two leads ($\Delta\text{RFB} \ge 2$) and achieves $\text{AURDC} \ge 0.80$ without promoting ungrounded or hallucinated inference across 51,625 held-out evaluations.

---

### Comparative Analysis of Existing Literature vs. Proposed Framework

| Dimension / Feature | Strodthoff et al. (2021) | Ribeiro et al. (2020) | K-MERL (Liu et al., 2025) | Generative Imputation (Hao et al., 2021) | Proposed Framework (This Work) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Model Architecture** | 1D-ResNet / LSTM | 1D-ResNet (Supervised) | Multimodal Transformer | cGAN / Deep VAE | **Frozen ECG-FM (`wav2vec2_cmsc`)** |
| **Parameter Budget** | ~4M - 10M | ~2.5M | ~110M | ~15M | **90.88M (Backbone Frozen)** |
| **Lead Input Scope** | Fixed 12-lead | Fixed 12-lead | Arbitrary Leads | Imputes 12 from 2/4 | **7 Fixed Levels ($12 \to 1$)** |
| **Missing Lead Policy** | Zeroed / Not evaluated | Not evaluated | Zero-masked | Generative Inpainting | **Zero Clamping (No Synthesis)** |
| **Evaluation Metrics** | Label Macro-AUROC/F1 | Sensitivity / Specificity | Zero-shot AUROC / Text BLEU | Waveform RMSE / PSNR | **AURDC, RFB, EO, GQ, TGR** |
| **Evidence Verification** | None (Black Box) | None (Black Box) | Linguistic NLP overlap | Visual inspection | **Deterministic 3-State Gate** |
| **Clinical Ontology** | None | None | Free-text reports | None | **SNOMED-CT Directed Subgraphs** |
| **Failure Boundary Mapping**| Absent | Absent | Absent | Absent | **Formalized Mathematical RFB** |

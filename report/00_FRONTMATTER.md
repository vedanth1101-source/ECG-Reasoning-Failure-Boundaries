# FRONTMATTER

---

## COVER PAGE

<br>

# REASONING FAILURE BOUNDARIES IN ECG FOUNDATION MODELS UNDER PROGRESSIVE PHYSIOLOGICAL INFORMATION LOSS

<br>

### A Project Report Submitted in Partial Fulfillment of the Requirements for the Degree of
### BACHELOR OF TECHNOLOGY
#### in
### COMPUTER SCIENCE AND ENGINEERING

<br>

#### Submitted by
**VIJAY K** (Register No.: 3122235001157)  
**SIDDHARTH CHANDRASEKAR S** (Register No.: 3122235001310)  
**VEDANTH M S** (Register No.: 3122235001314)  

<br>

#### Under the Guidance of
**DR. ANUSHA JAYASIMHAN**  
Assistant Professor  
Department of Computer Science and Engineering  

<br>
<br>

### DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING
### SRI SIVASUBRAMANIYA NADAR COLLEGE OF ENGINEERING
**(An Autonomous Institution, Affiliated to Anna University, Chennai)**  
**Kalavakkam – 603 110, Tamil Nadu, India**  

<br>

### October 2026

---

## ACKNOWLEDGEMENTS

We express our profound gratitude to our project supervisor, **DR. ANUSHA JAYASIMHAN**, Assistant Professor, Department of Computer Science and Engineering, Sri Sivasubramaniya Nadar College of Engineering, for her invaluable guidance, rigorous critiques, and constant encouragement throughout the design, implementation, and evaluation of this research. Her technical insights into biomedical signal processing, machine learning safety, and scientific methodology were indispensable in formalizing the concept of the Reasoning Failure Boundary.

We extend our sincere thanks to the Head of the Department, Department of Computer Science and Engineering, for providing the necessary infrastructural facilities, high-performance computing resources, and academic environment that enabled the extensive multi-seed evaluations and large-scale model benchmarking conducted in this study. We also express our sincere appreciation to all faculty members, technical staff, and project review committee members of the Department of Computer Science and Engineering for their perceptive questions, technical evaluations, and constructive suggestions during the project reviews.

We are indebted to our medical advisors and clinical collaborators whose clinical expertise in electrocardiography, diagnostic cardiology, and electrophysiology guided the construction and formalization of the clinical evidence schemas, the SNOMED-CT anatomical relationships, and the three-state evidence grounding taxonomy. Their domain knowledge ensured that our computational verification logic reflects genuine clinical decision-making.

Finally, we convey our heartfelt appreciation to our parents, family members, and peers for their unwavering support, patience, and encouragement throughout the course of this major research project.

<br>

**VIJAY K**  
**SIDDHARTH CHANDRASEKAR S**  
**VEDANTH M S**  

---

## ABSTRACT

Electrocardiogram (ECG) foundation models pre-trained on massive collections of multi-channel physiological waveforms have recently achieved state-of-the-art diagnostic discrimination across standard 12-lead electrocardiograms, matching or exceeding certified cardiologists on selected rhythm and conduction classification benchmarks. However, in emergency triage, pre-hospital ambulance transport, intensive care monitoring, and consumer wearable healthcare devices, clinical practitioners rarely encounter static, complete, and noise-free twelve-lead configurations. Instead, clinical acquisition systems operate under progressive physiological information loss, characterized by displaced electrodes, severe baseline artifacts, or constrained physical channel subsets ranging from hexaxial limb systems (6 leads) and transport cables (4 leads) down to bedside telemetry (2 leads) and single-lead consumer smartwatches (1 lead).

Conventional machine learning benchmarking paradigms evaluate neural models primarily on diagnostic label correctness under fixed lead configurations using traditional classification metrics such as Area Under the Receiver Operating Characteristic (AUROC), sensitivity, specificity, and macro-averaged F1-score. While these metrics effectively quantify predictive discrimination, they fail to assess whether the underlying clinical reasoning trace remains truthfully anchored in the physical electrodynamic signals that are actually observable by the active sensor configuration. In regional cardiac pathologies such as acute myocardial infarction, confirmation of localized myocardial injury strictly requires observing spatial agreement across contiguous anatomical leads (e.g., ST-segment elevation $\ge 1.0\text{ mm}$ across contiguous inferior leads II, III, and aVF). Under progressive lead deprivation, when key anatomical projections are masked, a deep neural network may nevertheless continue to emit the correct diagnostic label by exploiting learned population-level label correlations, non-physiological dataset shortcuts, or demographic priors. More alarmingly, generative reasoning engines frequently synthesize plausible-sounding clinical explanations that explicitly reference waveform deviations in anatomical leads that were physically absent from the input. Conventional classification metrics falsely reward these ungrounded guesses, concealing a critical failure mode that represents severe safety hazards in autonomous healthcare deployment.

This research addresses this fundamental safety gap by formalizing a reproducible, mathematically grounded framework to determine when, how, and why clinical reasoning fails under progressive sensor degradation. We define the *Reasoning Failure Boundary* (RFB) as the empirical threshold of lead availability below which an ECG foundation model's generated reasoning can no longer maintain physical grounding against observable signals. Furthermore, we establish the *Area Under the Reasoning Degradation Curve* (AURDC) as a unified metric summarizing reasoning resilience across the entire degradation continuum. 

Our methodological architecture is organized into six closed-loop, auditable phases: (1) multi-cohort standardization across 36,957 clinical recordings from PTB-XL, Chapman-Shaoxing, and MIMIC-IV-ECG, standardized via a 5-stage zero-phase signal processing pipeline; (2) controlled physiological degradation across seven deterministic lead levels ($12 \to 10 \to 8 \to 6 \to 4 \to 2 \to 1$ available leads) accompanied by Cabrera lead provenance directed acyclic graph (DAG) tracking, strictly excluding generative waveform inpainting; (3) dual-representation extraction combining continuous 768-dimensional latent vectors from a frozen 90.8-million-parameter ECG-FM backbone (`wav2vec2_cmsc`) with discrete, rule-extracted clinical observations ($O_l$); (4) counterfactual dual-path clinical reasoning within a shared Reasoning Transformer, comparing an unaugmented ECG-only Control Path against an Experimental Path augmented with a Clinical Knowledge Prior (CKP) derived from curated SNOMED-CT subgraphs; (5) deterministic grounding verification via an independent, medical-expert-ratified ternary logic gate that classifies every clinical claim as `SUPPORTED`, `CONTRADICTED`, or `UNAVAILABLE`; and (6) degradation compilation yielding Evidence Observability ($EO_l$), Grounding Quality ($GQ_l$), True Grounding Ratio ($TGR_l$), and the RFB.

Extensive empirical evaluations across 51,625 held-out test evaluations demonstrate that while diagnostic classification accuracy exhibits an apparent plateau from 12 leads down to 2 leads (12-lead Macro-AUROC $0.9540$ vs. 2-lead Macro-AUROC $0.9487$), physical evidence observability collapses precipitously from $100\%$ at 12 leads to $41.9\%$ at 2 leads and $1.6\%$ at 1 lead. The unaugmented baseline model crosses the clinical safety threshold ($\tau_{\text{safety}} = 0.60$) at 4 leads ($\text{RFB}_{\text{Control}} = 4\text{ leads}$), falling to $GQ = 0.42$ under 2-lead telemetry. In contrast, the CKP-augmented experimental condition successfully preserves evidence-grounded reasoning down to 2 leads ($GQ = 0.60$), delaying the Reasoning Failure Boundary by two full lead levels ($\text{RFB}_{\text{CKP}} = 2\text{ leads}$) and improving the normalized AURDC from $0.70$ to $0.81$ (a $+15.7\%$ relative gain). Crucially, True Grounding Ratio analysis reveals that at 1 lead, both systems fail catastrophically ($TGR < 0.005$), demonstrating that ontological knowledge priors cannot replace missing physical sensor measurements. This work provides an auditable operational envelope for deploying artificial intelligence in telemetry, ambulance care, and wearables.

<br>

**Keywords:** Electrocardiography (ECG), Foundation Models, Evidence Grounding, Reasoning Failure Boundary (RFB), Area Under the Reasoning Degradation Curve (AURDC), Clinical Knowledge Prior (CKP), SNOMED-CT, Physiological Information Loss, Medical AI Safety.

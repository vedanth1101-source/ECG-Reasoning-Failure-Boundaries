# Final-Year Project Video Presentation: Speaker Cue Cards & Rehearsal Script

**Project Title:** Reasoning Failure Boundaries in ECG Foundation Models Under Progressive Physiological Information Loss  
**Batch ID:** 43 (`BID_43_Video.mp4`)  
**Institution:** Sri Sivasubramaniya Nadar (SSN) College of Engineering, Kalavakkam, Chennai  
**Degree:** B.Tech. in Computer Science and Engineering  
**Supervisor / Guide:** Dr. Anusha Jayasimhan, Assistant Professor, Department of CSE  
**Total Target Duration:** 8 minutes 30 seconds (Strict Window: 7m 00s – 10m 00s)  

---

## Quick Reference: Speaker Timeline & Hand-offs

| Timestamp | Section | Speaker | Visual on Screen |
| :--- | :--- | :--- | :--- |
| **0:00 – 0:30** | 1. Introduction | **Vijay K** | Title Slide + Webcams (All 3 in formal attire) |
| **0:30 – 1:30** | 2. Problem Statement & Objectives | **Vijay K** | Slide 2 (12-Lead vs Telemetry + Objectives) |
| **1:30 – 2:30** | 3. Related Work & Research Gap | **Vedanth M S** | Slide 3 (Literature Matrix + Core Gap) |
| **2:30 – 4:00** | 4. Methodology & Architecture | **Vedanth M S** | Slide 4 (6-Phase Pipeline Diagram) |
| **4:00 – 5:00** | 5. Functional Modules & Dataset | **Vedanth M S** | Slide 5 (Modules Table + Multi-Cohort Prevalence) |
| **5:00 – 7:00** | 6. Working Demonstration | **Siddharth S** | Screen Share: Interactive Terminal / Cockpit |
| **7:00 – 8:30** | 7. Results & Analysis | **Siddharth S** | Slide 7 (Degradation Curves & 51k Evaluation Table) |
| **8:30 – 9:00** | 8. Conclusion & Future Work | **Vijay K** | Slide 8 (Summary + Future Scope + SSN Details) |

---

## Detailed Section Scripts

---

### Section 1: Introduction
- **Speaker:** Vijay K (Register No: 3122235001157)
- **Duration:** 30 Seconds (Target Cumulative: 0:00 &rarr; 0:30)
- **On Screen:** Slide 1 (Project Title, SSN Logo, Team details, Dr. Anusha Jayasimhan). Webcams active.

#### Speaking Script:
> *"Good day, respected evaluators and professors.  
> We are Batch 43, presenting our final-year project titled: **'Reasoning Failure Boundaries in ECG Foundation Models Under Progressive Physiological Information Loss.'**  
> I am Vijay, joined by my teammates Siddharth Chandrasekar and Vedanth, from the Department of Computer Science and Engineering at Sri Sivasubramaniya Nadar College of Engineering, Chennai.  
> This research was carried out under the expert guidance and supervision of **Dr. Anusha Jayasimhan**, Assistant Professor in our department."*

*(Transition: 2-second pause &rarr; Advance to Slide 2)*

---

### Section 2: Problem Statement & Objectives
- **Speaker:** Vijay K (Register No: 3122235001157)
- **Duration:** 1 Minute (Target Cumulative: 0:30 &rarr; 1:30)
- **On Screen:** Slide 2 (The Clinical Safety Gap vs. The 4 Measurable Objectives)

#### Speaking Script:
> *"Recent deep foundation models pre-trained on physiological waveforms have achieved state-of-the-art diagnostic performance on standard 12-lead electrocardiograms, often rivaling certified cardiologists.  
> However, in acute emergency triage, mobile ambulance transport, intensive care monitoring, and consumer wearable healthcare devices, clinical acquisition systems operate under progressive physiological information loss. Electrodes frequently detach, baseline artifacts obscure signals, or physical channels are restricted down to 4 leads, 2-lead telemetry, or single-lead smartwatches.  
> Today's machine learning benchmarks evaluate models strictly on classification metrics like AUROC and F1-score. When critical anatomical leads are stripped away, deep neural networks continue to output the correct disease label by exploiting non-physiological statistical dataset shortcuts. Alarmingly, generative reasoning engines synthesize plausible-sounding clinical explanations that cite wave deviations in anatomical leads that were physically absent from the patient's recording! Conventional accuracy metrics falsely reward these ungrounded guesses, concealing a critical hazard in medical AI deployment.  
> To address this safety gap, our project establishes four measurable objectives:  
> First, mathematically formalize the **Reasoning Failure Boundary**, or **RFB**—the empirical lead threshold below which physical grounding collapses.  
> Second, establish the **Area Under the Reasoning Degradation Curve**, or **AURDC**, to benchmark reasoning resilience across the entire degradation continuum.  
> Third, construct a counterfactual dual-path architecture comparing an unaugmented ECG-only Control Path against an Experimental Path augmented with a Clinical Knowledge Prior derived from SNOMED-CT knowledge graphs.  
> And fourth, implement an independent, expert-ratified ternary Grounding Verification Gate that classifies every clinical claim as `SUPPORTED`, `CONTRADICTED`, or `UNAVAILABLE`."*

*(Transition: Hand-off to Vedanth &rarr; "I now invite Vedanth to explain our related work and methodology.")*

---

### Section 3: Related Work & Research Gap
- **Speaker:** Vedanth M S (Register No: 3122235001314)
- **Duration:** 1 Minute (Target Cumulative: 1:30 &rarr; 2:30)
- **On Screen:** Slide 3 (Literature Matrix + Core Gap)

#### Speaking Script:
> *"Thank you, Vijay.  
> In related literature, landmark benchmarks like PTB-XL by Strodthoff et al. established standard supervised baselines for 12-lead classification. Recently, foundation models such as ECG-FM by McKeen et al. and GEM by Lan et al. introduced 90-million-parameter self-supervised representations and vision-language explanations for physiological signals.  
> However, existing approaches exhibit two fundamental limitations:  
> First, existing benchmarks evaluate models exclusively on diagnostic label correctness under fixed lead configurations, completely blind to whether the clinical reasoning trace is physically anchored in observable signals.  
> Second, works that attempt to handle missing leads rely on generative waveform inpainting. In clinical cardiology, fabricating unobserved electrodynamic vectors is fundamentally unsafe—an unrecorded lead cannot be assumed or hallucinated during acute cardiac ischemia.  
> The research gap we address is the complete absence of a reproducible, mathematically grounded framework that audits evidence provenance without generative inpainting, tracking exactly when, how, and why foundation model reasoning fails as physical sensors are progressively lost."*

*(Transition: Advance to Slide 4)*

---

### Section 4: Methodology & Architecture
- **Speaker:** Vedanth M S (Register No: 3122235001314)
- **Duration:** 1½ Minutes (Target Cumulative: 2:30 &rarr; 4:00)
- **On Screen:** Slide 4 (6-Phase Pipeline Diagram: `phase_1_to_3_master_pipeline.png` & `Phase 4.png`)

#### Speaking Script:
> *"To resolve this gap, our methodology is structured into six closed-loop, auditable phases:  
> In **Phase 1: Multi-Cohort Harmonization**, raw signals across three international clinical cohorts undergo a 5-stage zero-phase signal processing pipeline—including zero-phase Butterworth bandpass filtering between 0.5 and 40 Hertz, baseline wander removal, and 100 Hertz resampling.  
> In **Phase 2: Controlled Physiological Degradation**, we systematically simulate sensor loss across seven deterministic clinical configurations: 12, 10, 8, 6, 4, 2, and 1 lead. Every active lead is tracked via a Cabrera Lead Provenance Directed Acyclic Graph, strictly enforcing Einthoven and Goldberger vector relationships while prohibiting waveform inpainting.  
> In **Phase 3: Dual-Representation Extraction**, each recording is mapped into continuous 768-dimensional latent vectors, $Z_{\text{ECG}}$, pooled from the 12th transformer layer of a frozen 90.8-million-parameter wav2vec2-CMSC foundation model, combined with discrete rule-extracted clinical observations. Freezing the foundation model preserves universal electrodynamic features while eliminating catastrophic forgetting.  
> In **Phase 4: Dual-Path Clinical Reasoning**, a shared Reasoning Transformer evaluates counterfactual hypotheses: an unaugmented Control Path using ECG signals alone, and an Experimental Path augmented with a Clinical Knowledge Prior, or CKP, constructed from curated SNOMED-CT subgraphs.  
> In **Phase 5: Deterministic Grounding Verification**, an expert-ratified ternary logic gate independently verifies every clinical claim against the active sensor mask as `SUPPORTED`, `CONTRADICTED`, or `UNAVAILABLE`.  
> Finally, in **Phase 6: Degradation Compilation**, we compute Evidence Observability, Grounding Quality, and the True Grounding Ratio to trace the Reasoning Degradation Curve and isolate the Reasoning Failure Boundary."*

*(Transition: Advance to Slide 5)*

---

### Section 5: Functional Modules & Dataset
- **Speaker:** Vedanth M S (Register No: 3122235001314)
- **Duration:** 1 Minute (Target Cumulative: 4:00 &rarr; 5:00)
- **On Screen:** Slide 5 (Module Responsibilities Table + Multi-Cohort Disease Distribution Chart)

#### Speaking Script:
> *"Our software architecture is implemented across six modular functional components: the Preprocessing Module, the Controlled Lead Masking Engine, the Dual Representation Extractor, the Transformer-based Clinical Reasoner, the Ternary Grounding Gate, and the Metric Compilation Module.  
> To guarantee true clinical generalization, our experimental benchmark pools **36,957 clinical recordings** across three geographically diverse cohorts:  
> - 21,799 records from PTB-XL in Germany,  
> - 10,158 records from Chapman-Shaoxing in China,  
> - and 5,000 records from the MIMIC-IV-ECG database in the United States.  
> We enforced strict patient-level stratified partitioning into ten folds: Folds 1 through 7 for training, Fold 8 for validation, and Folds 9 and 10 strictly quarantined for single-pass test evaluation, ensuring zero patient leakage. All records are harmonized across eight core diagnostic categories, including Normal ECG, Atrial Fibrillation, Right and Left Bundle Branch Blocks, Sinus Bradycardia, Sinus Tachycardia, 1st Degree AV Block, and Myocardial Infarction.  
> I now hand over to Siddharth for our working demonstration and results."*

*(Transition: Hand-off to Siddharth &rarr; Switch to Screen Share)*

---

### Section 6: Working Demonstration
- **Speaker:** Siddharth Chandrasekar S (Register No: 3122235001310)
- **Duration:** 1 Minute 50 Seconds (< 2 Minutes unhurried) | **Word Count:** 229 words (125 wpm)
- **On Screen:** Live Screen Recording running `py -3.11 scripts/pipeline_flow_terminal_runner.py`

#### Speaking Script:
> *"Thank you, Vedanth. I will now demonstrate our working pipeline live on the terminal.  
> *(Action: Run `py -3.11 scripts/pipeline_flow_terminal_runner.py`)*  
> 
> We load quarantined patient record **`PTBXL_26`**, presenting with acute Myocardial Infarction.  
> In **Module 1**, zero-phase filtering standardizes the raw signal into a calibrated 12-lead tensor of shape `(12, 1000)` at 100 Hertz.  
> *(Action: Press [ENTER])*  
> 
> In **Module 2**, our Cabrera graph applies controlled lead masking across clinical setups, enforcing strictly zero synthetic inpainting.  
> *(Action: Press [ENTER])*  
> 
> In **Module 3**, our frozen 90.8-million-parameter foundation model encodes these masked waveforms into continuous 768-dimensional latent vectors, $Z_{\text{ECG}}$.  
> *(Action: Press [ENTER])*  
> 
> In **Module 4**, our reasoning engine extracts structured observations—detecting sinus rhythm and anterior ST elevations—and evaluates an 8-class hypothesis space.  
> Our Score Head outputs Myocardial Infarction as the top diagnosis with 86.7% probability.  
> 
> Examining reasoning traces across three clinical variants:  
> Under **12-lead**, it attends to anterior lead V1 and inferior lead aVF.  
> Under **6-lead hexaxial limb**, all chest leads are masked. Yet the model STILL predicts Infarction with 85.2% confidence, blindly citing missing chest leads V4, V1, and V3!  
> Under **2-lead telemetry**, confidence remains 82.5%, scattering attention across active and absent channels.  
> *(Action: Press [ENTER])*  
> 
> This brings us to our teaser for **Module 5: The Deterministic Ternary Grounding Gate**.  
> Module 5 audits every cited finding into three outcomes:  
> **`[SUPPORTED]`**, if verified in an active lead;  
> **`[CONTRADICTED]`**, if physiologically invalid; and  
> **`[UNAVAILABLE]`**, exposing when the model hallucinates findings in leads that were never recorded.  
> This uncovers the exact Reasoning Failure Boundary."*

*(Transition: Switch back to Slides &rarr; Advance to Slide 7)*

---

### Section 7: Results & Baseline Comparison (Completed Modules 1 to 4)
- **Speaker:** Siddharth Chandrasekar S (Register No: 3122235001310)
- **Duration:** 1½ Minutes (90 Seconds Unhurried) | **Word Count:** 175 words (125 wpm)
- **On Screen:** Slide 7 or Live Screen running `py -3.11 scripts/results_demonstration_runner.py`

#### Speaking Script:
> *"Thank you. I will now present our empirical results, baseline comparisons, and limitations, evaluated strictly across our completed Modules 1 through 4.  
> *(Action: Run `py -3.11 scripts/results_demonstration_runner.py`)*  
> 
> First, for **Evaluation Metrics**:  
> Our multi-seed reasoning transformer achieves **0.9540 Macro-AUROC** and **0.7375 Macro-F1** across 36,957 clinical records. Multi-seed training across seeds 42, 1337, and 2024 demonstrates near-zero variance, with an AUROC standard deviation under 0.001.  
> *(Action: Press [ENTER])*  
> 
> Second, in **Baseline Comparisons**:  
> Our reasoning transformer significantly outperforms standard baselines—beating clinical rule heuristics by **+29.1% AUROC** and frozen linear probing by **+3.2% AUROC** across all three cohorts: PTB-XL, Chapman, and MIMIC-IV.  
> *(Action: Press [ENTER])*  
> 
> Third, our **Main Findings** reveal a critical anatomical non-monotonicity:  
> The 4-lead configuration achieves **0.9524 AUROC**, outperforming the 6-lead limb setup at 0.9479. This occurs because preserving lead V1 captures the anterior septal vector, whereas 6-lead setups discard all precordial projections.  
> *(Action: Press [ENTER])*  
> 
> Finally, regarding **Limitations**:  
> While Module 4 maintains high diagnostic accuracy down to 2 leads, its reasoning transformer lacks sensor-awareness—blindly attending to unrecorded leads. Furthermore, our current evaluation is restricted to static 10-second records. These exact limitations directly motivate the ternary grounding gate in Module 5."*

*(Transition: Hand-off back to Vijay &rarr; Advance to Slide 8)*

---

### Section 8: Conclusion & Future Work
- **Speaker:** Vijay K (Register No: 3122235001157)
- **Target Duration:** ~45 Seconds | **Word Count:** 101 words (135 wpm)
- **On Screen:** Slide 8 (3-Panel Synthesis: Established Foundations, Evidence vs Uncertainty Dilemma, Future Roadmap)

#### Speaking Script (Verbatim Version matching Slide 8 Infographic):
> *"To conclude, we have established the foundational data pipeline, the lead-degradation framework, the ECG-FM representation pipeline, and the ECG-only control model.*  
> *(Gesture to Panel 1: Established Foundations & Modules 1–4)*  
>  
> *We have also examined the integrity of our inputs and reasoning traces, identifying limitations that must be addressed before drawing stronger conclusions.*  
> *(Gesture to Panel 2: Sensor-Evidence Integrity Gap & Limitation Callout)*  
>  
> *Our broader goal is to help researchers understand when reduced-lead ECG models have sufficient evidence to support their predictions—and when they should express uncertainty instead.*  
> *(Gesture to Panel 2 Bottom: Regime A vs Regime B Decision Boundary)*  
>  
> *Our next steps are to develop the clinical knowledge-assisted path, improve the evidence-grounding checks, and conduct a controlled comparison under a properly defined evaluation protocol.*  
> *(Gesture to Panel 3: Future Roadmap & Next Steps 1, 2, 3)*  
>  
> *Ultimately, we aim to make ECG-based AI evaluation more transparent and evidence-aware.*  
>  
> *Thank you."*  
> *(Speaker bows / concludes presentation)*

---

## Technical Recording & Submission Checklist

- [ ] **Recording Tool:** OBS Studio (Record at 1080p, 60fps, MP4 format).
- [ ] **Audio Check:** Test each speaker's microphone before starting. Keep microphone distance at 6–8 inches to prevent clipping.
- [ ] **Visual Layout:**
  - Intro & Conclusion: Full-screen camera or large side-by-side webcams.
  - Slides: Full-screen presentation (Press F11 on `presentation/index.html`) with webcams in top-right corner.
  - Demo: Clean Windows Terminal running `py -3.11 scripts/demo_video_presentation.py` at 14pt font size.
- [ ] **Output Filename:** `Batch_43_Video.mp4` or `BID_43_Video.mp4`.
- [ ] **Final Check:** Watch complete video to ensure duration is strictly between 7 and 10 minutes.

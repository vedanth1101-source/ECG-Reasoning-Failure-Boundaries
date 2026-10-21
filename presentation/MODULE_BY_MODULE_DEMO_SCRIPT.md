# Module-by-Module Live Demonstration Speaking Script

**Project Title:** Reasoning Failure Boundaries in ECG Foundation Models Under Progressive Physiological Information Loss  
**Institution:** Sri Sivasubramaniya Nadar College of Engineering, Chennai  
**Department:** Computer Science and Engineering | **Batch ID:** 43  
**Presenters:** Vijay K (3122235001157), Siddharth Chandrasekar S (3122235001310), Vedanth M S (3122235001314)  
**Supervisor:** Dr. Anusha Jayasimhan, Assistant Professor, Dept. of CSE  
**Presenter for Demo:** Siddharth Chandrasekar S  
**Associated Executable:** `py -3.11 scripts/pipeline_flow_terminal_runner.py`  
**Execution Modes:**
- **Interactive Presenter Mode:** `py -3.11 scripts/pipeline_flow_terminal_runner.py` *(Press [ENTER] to advance between modules)*
- **Rehearsal / Automated Mode:** `py -3.11 scripts/pipeline_flow_terminal_runner.py --auto --delay 1.0`

---

## Script Architecture Overview

| Stage | Section Name | Focus Data Structure / Representation | Target Time |
| :--- | :--- | :--- | :--- |
| **Intro** | Demo Launch & Cohort Loading | Patient `PTBXL_26` (Quarantined Test Split) | 0:00 – 0:15 |
| **Module 1** | Ingestion & Zero-Phase Filtering | `torch.Tensor(shape=[12, 1000], dtype=torch.float32)` | 0:15 – 0:35 |
| **Module 2** | Controlled Lead Masking (Cabrera DAG) | `torch.Tensor(shape=[7, 12, 1000])` & Binary Masks | 0:35 – 0:55 |
| **Module 3** | Foundation Model Latents ($Z_{\text{ECG}}$) | Continuous `torch.Tensor(shape=[7, 768])` & Cosine Drift | 0:55 – 1:15 |
| **Module 4** | Clinical Reasoning Engine | Observations $O_l$, Hypotheses ($8 \times 256$), Score Head & Traces | 1:15 – 2:00 |
| **Teaser 5** | Module 5 Ternary Grounding Preview | 3 Ternary States: `SUPPORTED`, `CONTRADICTED`, `UNAVAILABLE` | 2:00 – 2:30 |
| **Summary** | Pipeline Summary & Slide 7 Transition | End-to-End Representation Flow Handoff | 2:30 – 2:45 |

---

## The 2-Minute Quick Read Card (< 120 Seconds Unhurried)

- **Target Duration:** 1 Minute 50 Seconds  
- **Word Count:** 229 words  
- **Speaking Pace:** 125 words per minute (relaxed, authoritative, zero rushing)

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
> *(Action: Switch to Slides &rarr; Advance to Slide 7)*

---

## Detailed Section-by-Section Spoken Dialogue

### [0:00 – 0:15] Demo Launch & Record Ingestion

**Visual on Screen:**  
The presenter runs `py -3.11 scripts/pipeline_flow_terminal_runner.py`. The colored header banner renders, displaying the institution, batch details, and research title.

**Presenter Action:**  
Start the terminal script, look at the audience/evaluators, and begin speaking in a calm, confident tone.

> *"Thank you, Vedanth. I will now present our end-to-end working system live in the terminal environment.  
> As you can see on screen, our pipeline representation runner ingests real clinical ECG records from our quarantined test split. For this live demonstration, we load patient record **`PTBXL_26`**, a patient presenting with an acute, transmural Myocardial Infarction.  
> Let us advance into Module 1 to observe how this raw signal is standardized."*

*(Presenter Action: Press [ENTER])*

---

### [0:15 – 0:35] Module 1: Data Ingestion & 5-Stage Zero-Phase Preprocessing

**Visual on Screen:**  
- Animated spinner: *Connecting to PTB-XL, Chapman-Shaoxing, and MIMIC-IV-ECG Data Lakes...*  
- Progress bar: *Ingesting 36,957 raw records...*  
- Text breakdown of the 5 preprocessing stages ($P_1$ to $P_5$).  
- Blue Output Representation Box showing `torch.Tensor(shape=[12, 1000], dtype=torch.float32)` with numerical voltage slices and extrema.

**Presenter Spoken Words:**  
> *"In **Module 1**, our challenge is handling raw, heterogeneous signals across three distinct clinical cohorts. As shown on screen, the pipeline executes five mathematical preprocessing stages:  
> 1. Unit calibration converting machine ADC counts into millivolts clamped between minus 15 and plus 15 mV;  
> 2. Spatial alignment into standard Cabrera lead sequence;  
> 3. Zero-phase forward-backward Butterworth filtering from 0.5 to 45 Hertz with notch filtering to eliminate baseline wander and AC mains hum without phase distortion;  
> 4. Polyphase downsampling to exactly 100 Hertz, yielding a uniform 10-second temporal window of 1,000 samples;  
> 5. Quality screening to reject rails and flatlines.  
> 
> Look at the cyan representation box: our output is a standardized 12-lead physiological tensor of dimension `12 by 1000`. The numerical slice proves that all physical channels are finite, centered, and physically calibrated in millivolts."*

*(Presenter Action: Press [ENTER])*

---

### [0:35 – 0:55] Module 2: Controlled Lead Masking (7 Cabrera Variations)

**Visual on Screen:**  
- Animated spinner: *Mounting Cabrera Lead Provenance Directed Acyclic Graph (DAG)...*  
- Progress bar: *Generating 7 Variations per File across 258,699 instances...*  
- Output Representation Box displaying the batch tensor `torch.Tensor(shape=[7, 12, 1000])` and the 7 binary availability masks $M \in \{0, 1\}^{12}$ from 12-lead standard down to single-lead smartwatch.

**Presenter Spoken Words:**  
> *"In **Module 2**, we simulate real-world emergency medicine and wearable monitoring where clinicians lose leads.  
> Rather than arbitrary lead removal, we introduce controlled degradation across seven clinically meaningful levels: complete 12-lead, 10-lead precordial, 8-lead independent, 6-lead hexaxial limb, 4-lead Einthoven plus $V_1$, 2-lead bedside telemetry, and 1-lead consumer smartwatch.  
> 
> Mathematically, this is executed as a Hadamard channel-wise product: $X_{\text{masked}} = X \odot M$, where $M$ is our binary provenance mask.  
> As emphasized in red on screen: **we enforce strictly zero generative inpainting**. Imputing missing channels with diffusion or GAN models creates dangerous hallucinated waves; our framework preserves the ground-truth absence of sensors across all 258,699 generated instances."*

*(Presenter Action: Press [ENTER])*

---

### [0:55 – 1:15] Module 3: Foundation Model Adaptation & Latents ($Z_{\text{ECG}}$)

**Visual on Screen:**  
- Animated spinner: *Mounting Frozen ECG-FM Backbone (wav2vec2-CMSC, 90.88M Parameters, 12 Layers)...*  
- Progress bar: *Extracting Latent Representations into $\mathbb{R}^{768}$...*  
- Output Representation Box: `torch.Tensor(shape=[7, 768], dtype=torch.float32)` with cosine similarity drift metrics and the latent energy spectrum sparkline.

**Presenter Spoken Words:**  
> *"In **Module 3**, we project these degraded waveforms into high-dimensional semantic space.  
> We mount our frozen 90.88-million parameter `wav2vec2-CMSC` foundation model. By freezing all 12 transformer encoder layers, we prevent catastrophic forgetting and maintain an uncorrupted electrodynamic feature extractor.  
> 
> As displayed in the representation box, temporal mean pooling at the twelfth layer yields a continuous 768-dimensional latent vector, $Z_{\text{ECG}}$, for each variation.  
> Notice the cosine similarity drift metric: from 12 leads down to 1 lead, cosine similarity degrades from 1.000 to 0.892. Crucially, notice the gold recovery star on screen: the 4-lead configuration achieves a higher cosine similarity of 0.967 compared to the 6-lead configuration at 0.952, proving that preserving the horizontal septal lead $V_1$ recovers more anatomical information than retaining six limb leads."*

*(Presenter Action: Press [ENTER])*

---

### [1:15 – 2:00] Module 4: Clinical Reasoning Engine (Observations, Hypotheses, Score Head & Traces)

**Visual on Screen:**  
- **Component 4A Box:** Structured observations $O_l$ (HR: 74.2 BPM, PR: 162 ms, ST elevations in V1–V4, reciprocal depression in aVF).  
- **Component 4B Box:** Hypothesis space `torch.Tensor(shape=[8, 256])` across 8 pathologies.  
- **Component 4C Progress Bars:** Multi-seed training across Seeds 42, 123, 2024 with validation AUROC/F1, highlighting optimal checkpoint selection.  
- **Component 4D Box:** Score Head logits `torch.Tensor(shape=[8])` and posterior probabilities ($P(\text{MI}) = 86.7\%$).  
- **Component 4E Trace Boxes:** Detailed attention weight tables across the **3 clinical demonstration variants**:
  1. *Variant 1: 12-Lead Complete Standard*
  2. *Variant 2: 6-Lead Hexaxial Limb Only*
  3. *Variant 3: 2-Lead Bedside Telemetry*

**Presenter Spoken Words:**  
> *"Now we enter **Module 4: The Clinical Reasoning Engine**.  
> In Component 4A, we extract rule-grounded clinical observations, $O_l$. For this patient, our algorithmic feature extractor detects a normal rhythm of 74.2 BPM, normal intervals, but marked pathological ST elevation in anterior leads V1 through V4, and reciprocal ST depression in lead aVF.  
> In Component 4B, we project our 8 disease target classes into a 256-dimensional query space.  
> In Component 4C, we train our reasoning transformer across multiple random seeds, selecting the optimal Seed 42 checkpoint which achieves 0.9521 validation AUROC on Fold 8.  
> 
> Now, examine Component 4D—the Output Score Head:  
> Passing evidence cross-attention embeddings into our classification head produces eight output logits. Converting these via sigmoid yields Myocardial Infarction as the dominant hypothesis with an 86.7% posterior probability.  
> 
> In Component 4E, we examine how the model constructs its internal reasoning traces across **three representative variants**:  
> Under **Variant 1—Complete 12-Lead Standard**, the model attends heavily to anterior chest lead V1 and reciprocal lead aVF, appropriately linking anatomical ST elevations to inferior depression.  
> 
> Now, look at **Variant 2—6-Lead Hexaxial Limb Only**:  
> In this scenario, all precordial chest leads V1 through V6 are completely disconnected from the patient. Yet look at the model's Score Head: it STILL predicts Myocardial Infarction with 85.2% confidence! Furthermore, examine its reasoning trace: its top attended tokens are lead V4, lead V1, and lead V3. The model is actively citing chest leads that do not exist in the physical input! At this stage in Module 4, the transformer outputs these attention weights blindly.  
> 
> Finally, under **Variant 3—2-Lead Bedside Telemetry**, the model maintains 82.5% confidence, scattering its attention across both active leads and absent frontal channels."*

*(Presenter Action: Press [ENTER])*

---

### [2:00 – 2:30] Teaser: Module 5 (Deterministic Ternary Grounding Gate)

**Visual on Screen:**  
- Header banner: `[TEASER: MODULE 5] DETERMINISTIC TERNARY GROUNDING VERIFICATION GATE`.  
- Animated spinner: *Mounting Expert-Ratified Deterministic Ternary Logic Gate...*  
- Representation Box: The 3 Ternary Verification Outcomes defined with clinical criteria:
  1. `[SUPPORTED]` (Green)
  2. `[CONTRADICTED]` (Yellow)
  3. `[UNAVAILABLE]` (Red)
- Comparative Audit Preview Table across the 3 demonstrated variants.

**Presenter Spoken Words:**  
> *"This critical safety vulnerability leads directly into our preview of **Module 5: The Deterministic Ternary Grounding Gate**.  
> Conventional AI papers stop at diagnostic accuracy. If an AI correctly guesses 'Myocardial Infarction', it is declared successful—even if its explanation is completely fabricated.  
> 
> Module 5 changes this paradigm by implementing an independent, expert-ratified verification gate that audits every cited evidence token against the active sensor mask into **three deterministic ternary outcomes**:  
> 1. **`[SUPPORTED]`**, where the claimed electrodynamic finding is physically observable in an active sensor lead—such as ST elevation in V1 on a complete 12-lead setup;  
> 2. **`[CONTRADICTED]`**, where a claim directly violates physiological reality—such as claiming Sinus Bradycardia when the patient's measured heart rate is 74 BPM; and  
> 3. **`[UNAVAILABLE]`**, which catches the model citing leads that were physically masked or disconnected—such as citing chest lead V4 during 6-lead pre-hospital transport.  
> 
> Look at the audit preview table on screen:  
> On 12 leads, the model is 100% supported. But on 6 leads, despite 85.2% diagnostic accuracy, the grounding gate flags an immediate **UNAVAILABLE** safety violation.  
> This ternary audit unlocks the **Reasoning Failure Boundary (RFB)**, identifying the exact sensor threshold where medical AI ungrounds."*

---

### [2:30 – 2:45] Pipeline Summary & Transition to Slide 7

**Visual on Screen:**  
The clean Pipeline Summary Table displaying Modules 1 to 4 and Module 5 (Teaser), summarizing inputs, architectures, and exact tensor representations.

**Presenter Spoken Words:**  
> *"As summarized in our final pipeline table:  
> We have traced patient data from raw multi-cohort signals in Module 1, through controlled physical lead masking in Module 2, continuous latent representation in Module 3, multi-seed clinical reasoning in Module 4, and deterministic ternary verification in Module 5.  
> 
> With our live pipeline established, let us now return to our slides to analyze the large-scale benchmark results across 51,625 evaluations on our quarantined test split."*

*(Presenter Action: Switch display back to Presentation Slides &rarr; Advance to Slide 7)*

---

## Technical Reference Sheet for Presenter Q&A

If evaluators ask technical questions during or immediately following this demonstration, use these exact parameters:

| Question / Topic | Ground-Truth Architectural Detail |
| :--- | :--- |
| **Why 100 Hz instead of 500 Hz?** | Nyquist frequency is 50 Hz, perfectly capturing clinical ECG diagnostic bandwidth (0.05–45 Hz) while bounding tensor size to $(12, 1000)$ for efficient transformer attention. |
| **Why freeze the ECG Foundation Model?** | Fine-tuning 90.88M parameters on small downstream cohorts causes catastrophic representation collapse; freezing preserves universal electrodynamic features. |
| **How is the Score Head constructed?** | Two-layer projection: `Linear(256 -> 128) -> LayerNorm -> GELU -> Linear(128 -> 1)` with cross-attention evidence aggregation. |
| **What is the significance of the 3 Ternary States?** | Binary evaluation (True/False) conflates physical sensor absence with medical diagnostic inaccuracy. Ternary logic isolates sensor-dependent hallucination (`UNAVAILABLE`) from physiological error (`CONTRADICTED`). |
| **Why did 4-Lead outperform 6-Lead in Cosine Drift?** | The 6-lead limb setup only projects onto the frontal vertical plane. Adding lead $V_1$ introduces the horizontal transverse vector, capturing anterior septal depolarization. |

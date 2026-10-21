You are editing my LaTeX thesis. Make ONLY the changes below. Do not change any other wording, numbers, citations, equations, figures or chapter/section numbering. Keep the existing LaTeX style (\chapter, \section, table environments, \caption, \label, \ref, \cite). Where a table is removed, delete its whole table environment and update every \ref to it. Where I give replacement text, convert it to LaTeX faithfully: escape % as \%, keep $...$ maths, turn `code` into \texttt{...}, and keep each citation number [n] pointing to the same reference it points to now (use the existing \cite key for that reference). After editing, the List of Tables must regenerate from the captions.

=====================================================================
PART A. CHAPTER 2 (Literature Survey)
=====================================================================
A1. Delete Table 2.1 "Identified Research Gaps and Proposed Research Objectives" (the whole table).
A2. Delete the sentence "Table 2.1 relates the gaps to the objectives." at the end of the paragraph just before it. Chapter 2 then has no tables.

=====================================================================
PART B. CHAPTER 3 (Proposed Methodology)
=====================================================================
Keep Table 3.1 (Implementation Status of System Components), Table 3.2 (Benchmark Conditions) and the lead-availability configurations table. Keep Figure 3.1 and Algorithm 3.1. Remove Tables 3.3, 3.5 and 3.6 as follows.

B1. Section 3.3.3 Signal Preprocessing - delete Table 3.3 "Preprocessing Parameters".
  - Replace "applies five steps in a fixed order, with the parameters listed in Table 3.3." with "applies five steps in a fixed order."
  - Replace "a high-pass filter removes baseline wander, a notch filter removes power-line interference at 50 Hz for PTB-XL and Chapman-Shaoxing or 60 Hz for MIMIC-IV-ECG, and a low-pass filter removes high-frequency noise." with "a third-order Butterworth high-pass filter at 0.5 Hz removes baseline wander, an IIR notch filter (Q = 30) removes power-line interference at 50 Hz for PTB-XL and Chapman-Shaoxing or 60 Hz for MIMIC-IV-ECG, and a fourth-order Butterworth low-pass filter at 45 Hz removes high-frequency noise."
  - Replace "resampled from 500 Hz to 100 Hz with polyphase filtering," with "resampled from 500 Hz to 100 Hz with polyphase filtering (scipy.signal.resample_poly, downsampling factor 5),"
  - Replace "a flat channel," with "a flat channel (standard deviation below 10^-4 mV),"

B2. Section 3.3.6 Foundation-Model Representation Extraction - delete Table 3.5 "ECG-FM Pretraining Conditions and Project Input Conditions".
  - Replace "differ from the conditions under which it is used in this project, as shown in Table 3.5." with "differ from the conditions under which it is used in this project."
  - Directly after the sentence "The project presents 10-second recordings at 100 Hz in millivolts without normalisation." insert: "Pretraining also masked leads at random with a probability of 0.5 per lead, whereas the project removes leads only in the fixed configurations of Table 3.3."

B3. Section 3.3.11 - delete Table 3.6 "Comparison Models" and replace the sentence "The protocol defines four comparison models, listed in Table 3.6 with their current status." with this paragraph:
"The protocol defines four comparison models. The linear probe on the frozen foundation-model representation (B-1) has been evaluated for classification only; its evaluation script is not in the repository and its per-record predictions were not saved, so comparisons with it are descriptive. The result of the rule engine that applies the evidence criteria directly (B-2) has been withdrawn, because the QRS limitation of Section 3.3.7 made its bundle branch block rules unsatisfiable. The Control Reasoning Transformer (B-3) is implemented and evaluated for classification, but its grounding has not yet been evaluated. Knowledge-augmented reasoning with the Clinical Knowledge Prior (B-4) is planned."

B4. Renumber: the lead-availability configurations table (currently Table 3.4 "Lead-Availability Configurations") becomes Table 3.3. Update every reference to it in Chapter 3 (it is referred to as "Table 3.4" in Sections 3.3.5 and 3.3.7; the reference inside Table 3.5 disappears with that table). Chapter 3 then has exactly three tables: 3.1, 3.2, 3.3.

=====================================================================
PART C. CHAPTER 4 (Experimental Results and Performance Analysis): 16 tables -> 10 tables
=====================================================================
Final tables, in this order: 4.1 Source Datasets Used in the Project; 4.2 Recordings and Patients in Each Partition; 4.3 Positive Labels / Known Labels (Prevalence) in Each Partition; 4.4 Hardware, Software and Training Specifications; 4.5 Experiments Conducted; 4.6 Validation Performance by Seed; 4.7 Test Performance of the Control Reasoning Transformer by Configuration; 4.8 Test F1 per Condition at Threshold 0.5; 4.9 Test Performance of the Linear Probe (B-1) and the Control Reasoning Transformer (B-3); 4.10 Mean Cosine Similarity to the 12-Lead Representation.
Renumbering of kept tables (old -> new): 4.6->4.5, 4.7->4.6, 4.9->4.7, 4.11->4.8, 4.13->4.9, 4.14->4.10. Update every in-text reference accordingly (e.g. "(Table 4.9)" in Section 4.4.2 becomes "(Table 4.7)"; "(Table 4.7)" in Section 4.4.3 becomes "(Table 4.6)").

C1. Section 4.2 - merge old Table 4.5 (Training Settings) into old Table 4.4. Replace both tables with this single table, keeping the existing introductory paragraph above it:

**Table 4.4 Hardware, Software and Training Specifications**

| Item | Specification | Evidence |
|---|---|---|
| Processor and accelerator used for training | Not recorded in the available experiment artefacts. The Stage 6 report states that 16-thread CPU execution was enabled for Module 4; the training script enables this setting only when it runs on the CPU | `runs/module4_v3/reports/stage6_validation_and_audit_report.md`; `scripts/16_train_reasoning_transformer.py` |
| Foundation-model feature extraction | CPU; the extraction script fixes the device to the CPU, and the logs record batch size 32 with 8 threads per worker | `runs/module3_v1/scripts/extract_foundation_representations.py`; `runs/module3_v1/logs/` |
| Inference for validation traces | CPU, 16-thread worker pool | `runs/module4_v3/reports/validation_trace_generation_report.md` |
| Memory (RAM, accelerator memory) | Not recorded in the available experiment artefacts | — |
| Operating system | Windows (test session platform `win32`) | Stage 6 report, test-suite output |
| Python | 3.11.4; pytest 9.1.1 | Stage 6 report, test-suite output |
| Library versions used for the runs | Not recorded in the available experiment artefacts. The Python 3.11.4 installation on the project machine had PyTorch 2.14.0 (CPU build) when this report was written, but this is not evidence of the version used for the runs | Checked on 19 October 2026 |
| Foundation model | ECG-FM [7], `wav2vec2_cmsc`, 12 encoder layers, width 768; 90,883,072 parameters, all frozen; checkpoint `checkpoints/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt` (SHA-256 beginning `4d0142bc`) | `configs/control_path_training_config.json`; Module 4 freeze record |
| Control Reasoning Transformer | 1,830,657 trainable parameters | `configs/control_path_training_config.json`; checkpoints |
| Loss | Binary cross-entropy over known labels (Section 3.3.8) | `scripts/16_train_reasoning_transformer.py` |
| Optimiser | AdamW; learning rate $10^{-4}$; weight decay $10^{-4}$; betas (0.9, 0.999) | Checkpoint optimiser state |
| Batch size | 64 | Checkpoint step count; script default; Stage 6 report |
| Learning-rate schedule | Cosine annealing over 15 epochs to $10^{-6}$, no warm-up | Learning-rate history in `control_training_history_seed*.json` |
| Maximum epochs | 15 | Script default; schedule length |
| Early stopping | Validation loss; patience 5 epochs; minimum improvement $5 \times 10^{-4}$ | Training configuration; script |
| Examples per epoch | 180,775 (25,825 recordings × 7 configurations) | Stage 6 report; step count |
| Seeds | 42 (primary), 1337, 2024 | Training histories |
| Best epoch / last epoch | 3 / 8 for every seed | Training histories |
| Training time | 5,799 s (seed 42), 5,763 s (seed 1337), 5,973 s (seed 2024) | Training histories |
| Calibration | Per-condition temperature fitted on the validation partition [26] | Training histories; checkpoints |

Then delete old Table 4.5, and directly below the merged table put the existing paragraph that begins "The training settings are listed in Table 4.5. They are the settings that the training run actually used," but change its first sentence to: "The training settings in Table 4.4 are those that the training run actually used, established from the saved checkpoints and training histories." the rest of that paragraph is unchanged.

C2. Section 4.4.2 Calibration - delete old Table 4.8 "Calibration on the Pooled Validation Partition, Before and After Temperature Scaling" and its "Source:" line. The paragraph(s) between the 4.4.2 heading and 4.4.3 must read as this single paragraph:
"The temperatures were fitted per condition on the pooled validation partition (25,816 recording–configuration examples). The fitted temperatures lie between 0.93 and 1.26, with a mean of 1.15 for seed 42, 1.13 for seed 1337 and 1.10 for seed 2024. Most are slightly above 1, which means that the model was slightly overconfident and that calibration softened its probabilities. On the pooled validation partition, temperature scaling reduced the mean ECE from 0.01498 to 0.01043 for seed 42, from 0.01193 to 0.00764 for seed 1337 and from 0.01373 to 0.01180 for seed 2024, a reduction of between 14% and 36%. The NLL and Brier score changed only slightly; for seed 42 they fell from 0.15769 to 0.15586 and from 0.04741 to 0.04713 (source: `calibration` field of the training histories). These validation figures are optimistic, because the temperatures were fitted on the same data on which they are measured. The test partition gives an out-of-sample check: with the seed-42 temperatures, the test ECE ranges from 0.0105 in the 12-lead configuration to 0.0132 in the single-lead configuration (Table 4.7). This shows that one set of temperatures, fitted on all configurations together, remains usable when leads are removed, with a small rise in calibration error. The test values before calibration were not saved, so the effect of calibration on the test partition cannot be shown."

C3. Section 4.4.4 Per-Condition Performance - delete old Table 4.10 "Test AUROC per Condition"; keep old Table 4.11 as Table 4.8 and change its source line to "Source: runs/module4_v3/metrics/test_evaluation_summary.json."
  - The first paragraph must read: "Table 4.8 gives the per-condition F1 of the Control Reasoning Transformer on the test partition, and Figure 4.2 shows the same values. The number of known labels and positives behind each condition is given in Table 4.3; these denominators are the same in every configuration."
  - The paragraph after Figure 4.2 must read: "The per-condition AUROC, from the same file, shows the same picture. Except for myocardial infarction, every condition kept an AUROC of at least 0.917 down to two leads; with lead II alone, the AUROC ranged from 0.8011 for myocardial infarction to 0.9940 for sinus tachycardia. The conditions fall into three groups."
  - The "Conduction blocks" paragraph must read: "**Conduction blocks.** Right bundle branch block depends on chest lead V1 in a clear way. Its F1 is 0.7855 with all leads, falls to 0.6725 in the 6-lead configuration, which has no chest leads, recovers to 0.7785 and 0.7707 in the 4- and 2-lead configurations, which keep V1, and collapses to 0.1266 with lead II alone, where its recall is 0.074 and its AUROC 0.8408, against 0.9642 with all leads. Left bundle branch block degrades less: its F1 is between 0.748 and 0.799 down to two leads and falls to 0.4565 with one lead, where its AUROC is 0.9441. This pattern is consistent with the project schema, which assigns V1 and V6 to right bundle branch block (Table 3.2), but it does not show that the model uses those leads for the right reasons; establishing that is the purpose of the grounding gate."

C4. Section 4.4.5 Performance by Cohort - delete old Table 4.12 "Test Performance by Cohort" and its source line. The section must read:
"With all leads, the test macro-AUROC was 0.9476 for PTB-XL (4,358 recordings), 0.9509 for Chapman-Shaoxing (2,028) and 0.9379 for MIMIC-IV-ECG (989), and the corresponding macro-F1 values were 0.6767, 0.6775 and 0.7083. With lead II alone, the macro-AUROC was 0.9163, 0.9260 and 0.8981 (source: `dataset_breakdown` field of the test summary)."
"The model performs at a similar level on all three cohorts, and the configuration pattern of Section 4.4.3, including the advantage of the 4-lead over the 6-lead configuration, appears in each of them (4-lead and 6-lead macro-AUROC: 0.9431 and 0.9406 for PTB-XL, 0.9618 and 0.9488 for Chapman-Shaoxing, 0.9367 and 0.9318 for MIMIC-IV-ECG). The macro-F1 of each cohort is lower than the macro-F1 of the pooled test partition. A likely reason is that some conditions have few positives in a single cohort, which lowers the cohort's per-condition F1; the project's post-evaluation audit records, for example, only 8 positive myocardial infarction recordings among the Chapman-Shaoxing test recordings. The MIMIC-IV-ECG results measure agreement with machine-generated statements rather than with clinicians, so they are not directly comparable with the other two cohorts [8]. Because the three cohorts were pooled and split together, none of these results is an external validation on an unseen cohort. Nor are they comparable with published PTB-XL benchmarks [6], which use PTB-XL's own folds and label sets."

C5. Section 4.4.8 Clinical Observation Extractor - delete old Table 4.15 "Observation-Extractor Validation Against Machine Measurements" and its source line. The first paragraph ends "...The study did not record a code hash of its own." (delete "Table 4.15 gives its results."), and is followed by this new paragraph:
"Against the reference values, heart rate had a mean absolute error of 3.70 bpm (median 0.96 bpm; Pearson r = 0.866), the PR interval 21.57 ms (median 14.0 ms; r = 0.581) and the QRS duration 28.92 ms (median 26.0 ms; r = 0.255). Rhythm detection reached an F1 of 0.917 for sinus bradycardia, 0.681 for sinus tachycardia (sensitivity 0.517) and 0.475 for atrial fibrillation (sensitivity 0.633, precision 0.380, from 19 true and 31 false positives). The ST-segment and T-wave rules were not validated. The study labels the heart-rate measurement "clinically validated", although it compared it only with machine measurements (source: `reports/clinical_observation_validation.json`)."

C6. Section 4.4.9 Discussion - delete old Table 4.16 "Availability of Results at the End of the First Half". The last paragraph must read:
"The test results are a single seed of a single authorisation-unverified run, and they should be confirmed by an authorised, separately versioned evaluation before any confirmatory claim is made. Section 4.3.1 lists the parts of the research programme for which no results exist yet."

C7. Section 4.4.7 - replace "(Table 3.5)" with "(Section 3.3.6)", because Table 3.5 is removed in Part B.

=====================================================================
PART D. CHAPTER 5 (Social Impact & Sustainability): must fit two pages
=====================================================================
Replace the whole body of Chapter 5 (keep the chapter title "SOCIAL IMPACT & SUSTAINABILITY") with exactly two sections, 5.1 SOCIAL IMPACT and 5.2 SUSTAINABILITY, and this text. Chapter 5 has no tables.

## 5.1 SOCIAL IMPACT

Most ECGs are interpreted automatically, and clinicians may accept computer statements without checking them [4]. Lower-cost devices that record fewer leads may widen access to cardiac care [8], and single-lead monitors are already used for rhythm assessment [5]. A model developed on 12-lead recordings may therefore be applied to incomplete recordings without any warning that the evidence for some diagnoses is missing. This project studies that risk under controlled conditions, using seven zero-masked lead configurations and a frozen encoder. Its results show that the number of leads is a poor guide to reliability: the 4-lead configuration with V1 outperformed the 6-lead limb configuration, and the F1 score for right bundle branch block fell to 0.13 with lead II alone. Findings of this kind may help developers test models against stated lead configurations, and may in future help tell clinicians when an automated statement rests on leads that were not recorded.

These benefits are prospective. The project is a research evaluation, not a diagnostic tool, and it has not been clinically validated. Its test results are descriptive measurements from one seed of an authorisation-unverified run. The proposed grounding gate, with its SUPPORTED, CONTRADICTED and UNAVAILABLE states, and the Reasoning Failure Boundary have not been implemented, and the attention-based traces are not validated explanations [12, 13]. The datasets are public PhysioNet releases whose de-identification is documented by their providers; the project adds no privacy measures of its own and claims no ethics approval or regulatory compliance. Because the cohorts were pooled, and the MIMIC-IV-ECG labels are machine-generated [25], the results are not an external validation.

## 5.2 SUSTAINABILITY

The project reuses the pretrained ECG-FM encoder, with 90,883,072 parameters, without retraining it [7]. Only the Control Reasoning Transformer, with 1,830,657 parameters, was trained. The encoder's representations were computed once on the CPU for all 258,216 recording–configuration pairs and stored in about 0.83 GB, so the three training runs, of 5,763 to 5,973 seconds each, needed no passes through the encoder. A single model was trained for all seven configurations instead of one per configuration. Freeze records, split manifests and SHA-256 hashes made the Chapter 4 results checkable against their sources, although software versions and training logs were not recorded. No energy or carbon use was measured, so no savings are claimed.

=====================================================================
PART E. CHAPTER 6 (Conclusions and Future Work): must fit one page
=====================================================================
Replace the whole body of Chapter 6 with the three paragraphs below. Use NO subsections (remove 6.1, 6.2 and 6.3 headings); this matches the sample thesis, whose Chapter 6 is one block of text. Keep the chapter title "CONCLUSIONS AND FUTURE WORK". Chapter 6 has no tables.

In conclusion, this project examined whether the diagnoses of an ECG foundation model remain supported by the evidence left after leads are removed. A frozen ECG-FM encoder [7] and a Control Reasoning Transformer were evaluated on eight benchmark conditions across seven zero-masked lead configurations. The test macro-AUROC stayed between 0.9479 and 0.9540 down to two leads and fell to 0.9215 with lead II alone, and performance depended on which leads remained rather than how many.

These are descriptive results from one seed of an authorisation-unverified test run, and they do not establish clinical suitability or evidence grounding. The grounding gate, its metrics, the Reasoning Failure Boundary, the DFB and the AURDC are not yet implemented. The observation extractor cannot measure wide QRS complexes, morphology or Q waves, and lead removal is simulated rather than recorded.

As future work, the extractor and evidence schema will be repaired, the grounding gate implemented and the Reasoning Failure Boundary measured in an authorised, versioned evaluation. The SNOMED CT-based Clinical Knowledge Prior [16] will then be added, and the system validated on independent cohorts.

=====================================================================
FINAL CHECKS (report back to me)
=====================================================================
1. List of Tables shows exactly: 3.1, 3.2, 3.3 (Chapter 3) and 4.1-4.10 (Chapter 4); no Table 2.x, 5.x or 6.x.
2. No "??" from broken \ref after compiling.
3. Chapter 5 fits on two pages and Chapter 6 on one page; if Chapter 6 still overflows, tell me instead of deleting content.
4. No citation number or reference entry changed; the bibliography still has 26 entries.

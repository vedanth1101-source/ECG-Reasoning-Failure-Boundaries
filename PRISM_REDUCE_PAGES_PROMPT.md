You are editing my LaTeX thesis "Reasoning Failure Boundaries in ECG Foundation Models Under Progressive Physiological Information Loss". Goal: shorten the report by about 10 pages by removing repetition and secondary detail. Make ONLY the edits listed below.

FORMAT RULES (do not break these):
- Keep the mandatory seven-chapter structure and every existing chapter and section heading exactly as it is:
  Chapter 1 Introduction (1.1 Motivation, 1.2 Background, 1.3 Problem Definition, 1.4 Organization of the Report);
  Chapter 2 Literature Survey (2.1 Papers Related to the Problem Identified, 2.2 Gaps Identified, 2.3 Research Objectives);
  Chapter 3 Proposed Methodology (3.1, 3.2, 3.3 and 3.3.1-3.3.12 unchanged);
  Chapter 4 Experimental Results and Performance Analysis (4.1 Dataset Description, 4.2 Ecosystem (Hardware & Software Specifications), 4.3 Experiments Conducted, 4.4 Performance Analysis, with their existing subsections);
  Chapter 5 Social Impact & Sustainability; Chapter 6 Conclusions and Future Work; Chapter 7 References.
- Do not change the document class, fonts, font size, line spacing, margins, heading styles, table styles, caption style, page numbering, front matter or bibliography style.
- Do not touch Chapters 5, 6 and 7 (References). Do not add, remove or renumber any reference; every citation number [n] must still point to the same reference, and all 26 references must remain cited.
- Do not change any number, result, table value or equation except where an edit below removes it. Do not add new claims.
- Convert my plain text to LaTeX faithfully: escape % as \%, keep $...$ maths, turn `code` into \texttt{...}, keep [n] citations as the existing \cite for that reference.
- If you cannot find a passage exactly (wording may differ slightly in LaTeX), find the closest matching passage in the named section; if none exists, skip that edit and tell me.


==================== CHAPTER 1 ====================

1.1  [Section 1.1 Motivation] REPLACE:
"Methods for reconstructing missing leads exist, but synthesised waveforms are estimates rather than measurements, and a recent evaluation of deep-learning reconstruction found that reconstructed leads regress towards population averages rather than reflecting the individual patient [11]."
WITH:
"Reconstructed leads would be estimates rather than measurements [11]."

1.2  [Section 1.2 Background] REPLACE:
"In this project, criteria of this kind are represented in an evidence schema that groups leads by anatomical territory, for example leads II, III and aVF for the inferior wall, requires a QRS duration of at least 120 ms for complete bundle branch block, and accepts ST elevation of at least 0.10 mV in two or more contiguous leads, or pathological Q waves, as evidence for an infarction claim; the schema is described in Chapter 3."
WITH:
"In this project, criteria of this kind are represented in an evidence schema described in Chapter 3."

1.3  [Section 1.3 Problem Definition] REPLACE:
"The project's internal evaluation protocol (version 2.0, dated 14 October 2026), which describes itself as pre-registered and whose hash was recorded in the project's Module 4 freeze record on 15 October 2026, uses a threshold of 0.60, with 0.50 and 0.70 as sensitivity values, and reports the transition from the last acceptable to the first failing configuration. The protocol has not been registered with any external registry."
WITH:
"The project's internal evaluation protocol, which is hash-verified but not externally registered (Section 3.3.12), uses a threshold of 0.60, with 0.50 and 0.70 as sensitivity values, and reports the transition from the last acceptable to the first failing configuration."

1.4  [1.4 Organization of the Report: replace all paragraphs of this section with the single paragraph below]
NEW TEXT:
This report is organised into seven chapters. Chapter 1 presents the motivation, background, problem definition and organisation of the report. Chapter 2 reviews the literature on deep learning for ECG analysis, ECG foundation models, reduced-lead analysis, explainability and evidence verification, identifies the research gaps and states the research objectives. Chapter 3 describes the proposed methodology: the dataset pipeline, preprocessing, the seven lead-removal configurations, the frozen foundation model, the observation extractor and the Control Reasoning Transformer, together with the grounding gate, grounding metrics and Clinical Knowledge Prior, which are specified but not yet implemented. Chapter 4 describes the datasets, the hardware and software environment and the experiments conducted, and analyses the results obtained so far with their limitations. Chapter 5 discusses social impact and sustainability, Chapter 6 presents the conclusions and future work, and Chapter 7 lists the references.

==================== CHAPTER 2 ====================

2.1  [Section 2.1 (opening paragraph, before 'Standardised Criteria for ECG Interpretation')] DELETE this text:
"This section reviews the literature in the order in which it bears on the research problem: the clinical criteria that define ECG evidence, automated interpretation from rule-based programs to foundation models, reduced-lead analysis and reconstruction, and the evaluation of explanations and claims. For each group of studies, the discussion notes what was established and what remains open for the present problem."

2.2  [Section 2.1 Standardised Criteria] DELETE this text:
"These documents provide the clinical basis for lead-specific diagnostic criteria."

2.3  [Section 2.1 Computer-Based Interpretation Programs] DELETE this text:
"The review points to the need for human verification, a need that is greater for learned models whose outputs are not derived from explicit measurements."

2.4  [Section 2.1 Faithfulness of Attribution] REPLACE:
"They also reported that a ResNet-50 retrained with 90% of the input removed still reached 63.53% ImageNet accuracy, compared with 76.68% on clean data, and argued that degradation measured without retraining may reflect a shift in the input distribution rather than the loss of information."
WITH:
"They also argued that degradation measured without retraining may reflect a shift in the input distribution rather than the loss of information."

2.5  [Section 2.1 Claim Verification] REPLACE:
"; the best system reached 31.87% accuracy when the correct evidence was also required [17]"
WITH:
"[17]"

2.6  [2.3 Research Objectives: replace the paragraph beginning 'These objectives are at different stages.' (the paragraph after the numbered list of six objectives) with]
NEW TEXT:
These objectives are at different stages. In the first half, the pipeline of Objective 1 has been implemented, Objective 2 has produced preliminary representation and classification results (Chapter 4), and for Objective 3 the observation extractor has been implemented and the evidence requirements specified, although the extractor has known limitations. The grounding states and the Reasoning Failure Boundary of Objectives 4 and 6 are defined in the project's internal evaluation protocol (version 2.0, dated 14 October 2026), which labels itself pre-registered and whose hash was recorded in the Module 4 freeze record on 15 October 2026; it has not been registered externally. Objectives 4 to 6, including the knowledge-augmented path of Objective 5, have not yet been implemented. No objective includes clinical validation or deployment, and the threshold in Objective 6 is an analysis parameter rather than a clinical standard.

==================== CHAPTER 3 ====================

3.1  [Section 3.1 Architecture Design] REPLACE:
"The separation of the modules is deliberate. Because lead removal happens once, in Module 2, the same degraded recording is seen by the foundation model, the observation extractor and, in future, the grounding gate. Because the mask travels with the recording, the reasoning model and the gate can both distinguish a lead that shows no abnormality from a lead that was never recorded. And because the foundation model is frozen, any change in its output across configurations can be attributed to the removed leads rather than to retraining."
WITH:
"Because lead removal happens once, in Module 2, the foundation model, the observation extractor and, in future, the grounding gate all see the same degraded recording, and the mask travels with it, so a lead that shows no abnormality can be distinguished from a lead that was never recorded."

3.2  [Section 3.1 Architecture Design] DELETE this text:
"The table distinguishes components that are implemented and tested, components that are implemented but not yet fully evaluated, components that are defined in the evaluation protocol but not implemented, and proposed future work."

3.3  [Section 3.2 Control Reasoning Transformer Architecture] REPLACE:
"One limitation of the current implementation was identified while preparing this chapter. The observation extractor labels rhythms as "Sinus Bradycardia", "Sinus Tachycardia", "Atrial Fibrillation / Irregular rhythm" and "Indeterminate rhythm", whereas the rhythm vocabulary of the context builder uses different spellings ("Sinus bradycardia", "Sinus tachycardia", "Atrial fibrillation"). Only "Sinus rhythm" matches,"
WITH:
"One limitation was identified while preparing this chapter: the rhythm labels written by the observation extractor are spelled differently from the rhythm vocabulary of the context builder. Only "Sinus rhythm" matches,"

3.4  [Section 3.3.1 Dataset Acquisition and Provenance] DELETE this text:
"Some earlier project documents described the source as the Challenge 2020 data. That description is incorrect, because the public Challenge 2020 training set contains no Chapman-Shaoxing records, and it has been corrected in the project documentation."

3.5  [Section 3.3.4 Data Partitioning and Leakage Prevention] REPLACE:
"The Module 1 freeze record lists 13 pairs of identical Chapman-Shaoxing signals stored under different record identifiers. A recomputation of the signal hashes for this chapter confirmed the 13 pairs and showed that their labels agree. Nine pairs fall in different partitions: six link the training and test partitions, two the training and validation partitions, and one the validation and test partitions. Seven test records therefore have exact copies in data used for training or model selection. This is a confirmed, if small, leakage, which is disclosed with the results."
WITH:
"The Module 1 freeze record lists 13 pairs of identical Chapman-Shaoxing signals stored under different record identifiers; seven test records have exact copies in training or validation data, a small but confirmed leakage described in Section 4.1.2."

3.6  [Section 3.3.6 Foundation-Model Representation Extraction] DELETE this text:
"The YAML configuration distributed with the checkpoint lists a larger 24-layer configuration, which does not match the stored weights; the weights were treated as authoritative."

3.7  [Section 3.3.7 Clinical Observation Extraction] REPLACE:
"Unlike the original algorithm, the implementation uses a fixed threshold rather than adaptive thresholds and runs at 100 Hz rather than the 200 Hz of the original design, so the detection performance reported for the original algorithm does not apply to it."
WITH:
"Unlike the original, it uses a fixed threshold and runs at 100 Hz, so the detection performance reported for the original algorithm does not apply to it."

3.8  [Section 3.3.7 Clinical Observation Extraction] REPLACE:
"The extractor's unit tests check its output format, mask handling and determinism, but not the accuracy of its measurements. The only agreement study in the repository compares the extractor with the machine measurements of 250 MIMIC-IV-ECG recordings. It was run before the Module 4 freeze record was written, but on the same extractor code, whose file hash matches the freeze record; its results are examined in Chapter 4."
WITH:
"Its unit tests do not check measurement accuracy; its agreement with machine measurements on 250 MIMIC-IV-ECG recordings, obtained with the frozen extractor code, is examined in Section 4.4.8."

3.9  [Section 3.3.8 Training, Calibration and Reasoning Traces] REPLACE:
"These are the settings applied by the training script and confirmed by the saved run (Section 4.2). The training configuration file lists a batch size of 32, 20 epochs and two warm-up epochs, but the script does not read these values from it."
WITH:
"These settings are confirmed by the saved run; the training configuration file lists different values that the script does not apply (Section 4.2)."

3.10  [Section 3.3.8 Training, Calibration and Reasoning Traces] REPLACE:
"Three properties of the current traces limit their use as claims. First, the traced hypothesis is chosen by highest probability even when it falls below the decision threshold of 0.5. This happens in 3,641 of the 25,816 validation traces (14.1%), so some traces describe a diagnosis that the model does not predict. Second, only one hypothesis is traced per recording, although 3,130 validation traces (12.1%) come from recordings with more than one positive prediction. Third, the selected tokens are not statements about findings. Most are lead-status tokens, which encode only a lead's position and whether it is active, and a token's name does not record whether a finding was present. The procedure for converting traces into verifiable clinical claims has not yet been defined, so the current traces cannot yet be passed to a deterministic claim-level gate. The trace-quality labels written by the validation-trace script ("consistent", "contradictory", "insufficient") use ad hoc rules that differ from the evidence schema. They are not outputs of the grounding gate."
WITH:
"The current traces are not yet usable as claims: each traces only the single highest-probability hypothesis, even when it falls below the decision threshold (14.1% of the 25,816 validation traces) or when the recording has several positive predictions (12.1%), and most selected tokens encode only a lead's position and status rather than a finding. The procedure for converting traces into verifiable clinical claims has not yet been defined, and the trace-quality labels written by the validation-trace script come from ad hoc rules, not from the grounding gate."

3.11  [Section 3.3.9 Evidence States and the Grounding Gate] REPLACE:
"The UNAVAILABLE state is the project's own adaptation of this distinction to ECG claims. It is determined by lead availability, not by a dataset label, and it is not equivalent to FEVER's NotEnoughInfo category."
WITH:
"The UNAVAILABLE state is the project's own adaptation of this distinction: it is determined by lead availability and is not equivalent to FEVER's NotEnoughInfo category."

3.12  [Section 3.3.10 Clinical Knowledge Prior] REPLACE:
"Two constraints are already fixed. First, the benchmark configuration requires the knowledge-augmented path to evaluate exactly the same eight candidate diagnoses as the control path, without filtering or re-ranking them, so that any difference between the paths reflects reasoning and not a smaller candidate set. Second, the knowledge-augmented path appears in the evaluation protocol as a planned comparison with the control path (baseline B-4). The CKP has not been implemented, and no comparison with the control path is reported in this first-half report."
WITH:
"The knowledge-augmented path must score exactly the same eight candidate diagnoses as the control path, without filtering or re-ranking them, and it appears in the protocol only as a planned comparison (B-4). The CKP has not been implemented, and no such comparison is reported here."

3.13  [Section 3.3.11 Evaluation Metrics and the Reasoning Failure Boundary] REPLACE:
"In that case the protocol requires the initial failure to be reported, the recovery to be flagged and documented, and a sustained failure boundary, after which GQ never recovers, to be reported as well. If every configuration passes, the boundary is reported as robust across all levels; if the 12-lead configuration already fails, it is reported as failed at baseline; and if GQ is not estimable at some configuration, the search records that step as indeterminate and reports the lowest estimable configuration."
WITH:
"In that case the initial failure is reported, the recovery is flagged, and the sustained boundary after which GQ never recovers is reported as well; the protocol also defines robust, failed-at-baseline and indeterminate outcomes."

3.14  [Section 3.3.11 Evaluation Metrics and the Reasoning Failure Boundary] REPLACE:
"When GQ is not estimable at some configuration, the protocol integrates over the well-defined sub-interval and normalises by its length. It does not specify what to do when the undefined point lies inside the sequence rather than at its end. The protocol also asks for bootstrap confidence intervals for macro-F1 and the retained performance ratio, but the DFB, RPR and their intervals are not yet implemented in the evaluation scripts, and no paired procedure for comparing configurations is defined."
WITH:
"The protocol does not define how to handle a non-estimable GQ inside the sequence, and the DFB, RPR, their bootstrap intervals and any paired comparison of configurations are not yet implemented."

3.15  [3.3.9 Evidence States and the Grounding Gate: replace the sentence ending 'must be resolved before implementation:', the bulleted list after it and the sentence 'These are design decisions for the researcher; none of them is resolved in this report.' with the single paragraph below]
NEW TEXT:
A review of the schema for this chapter found further issues that must be resolved before implementation: the required leads of left bundle branch block, sinus bradycardia, sinus tachycardia, first-degree atrioventricular block and myocardial infarction disagree with their unavailability rules; some verification and contradiction rules leave cases undecided (a QRS duration between 100 and 120 ms satisfies neither rule for bundle branch block); the atrial fibrillation rule is satisfied whenever V1 is active; P-wave, QRS-morphology, Q-wave and signal-quality measurements are not produced by the extractor; the normal-ECG QRS requirement is always met because of the QRS limit; and the protocol does not say how to treat missing measurements, claims with several findings or alternative criteria, such as ST elevation or Q waves for infarction. These are design decisions for the researcher; none of them is resolved in this report.

3.16  [3.3.12 Reproducibility and Evaluation Governance: replace the second paragraph (beginning 'The protocol requires the test partition to be evaluated exactly once') with]
NEW TEXT:
The protocol requires the test partition to be evaluated once, after explicit authorisation by the investigator. The project's provenance statement (`runs/module4_v3/reports/evaluation_provenance_statement.md`) records that this condition was not met, so the test evaluation, completed on 15 October 2026 at 16:29 UTC, is classified as authorisation-unverified; Section 4.3.2 gives the details. The test results in Chapter 4 are therefore descriptive measurements, not a confirmatory pre-specified evaluation, and any authorised re-evaluation must be a new, separately versioned run.

==================== CHAPTER 4 ====================

4.1  [Section 4.1.1 Source Datasets and Versions] DELETE this text:
"The subset was sampled to contain roughly equal numbers of recordings for the eight conditions. Because the labels were later re-derived as multi-label, three-state labels, the subset is no longer exactly balanced."

4.2  [Section 4.1.1 Source Datasets and Versions] DELETE this text:
"The PTB-XL publication describes an earlier release with different counts, so the counts here are taken from release 1.0.3 and the repository."

4.3  [Section 4.1.3 Label Distribution] DELETE this text:
"The project's Stage 7 report gives 754 positive atrial fibrillation recordings in the test partition. The saved test summary, from which that report was written, gives 788, and the project's own post-evaluation audit traced the smaller figure to a transcription error. The value 788 is used here."

4.4  [Section 4.3.2 Evaluation Provenance] REPLACE:
"The comparison-model results of experiment E9 were produced after the test evaluation. According to its file timestamp, the comparison file was written at 16:52 UTC on 15 October 2026, 23 minutes after the test summary. The script that produced it (`evaluate_baselines.py`) is stored in the development environment's scratch directory, outside the repository."
WITH:
"The comparison-model results of experiment E9 were produced 23 minutes after the test summary, by a script (`evaluate_baselines.py`) stored outside the repository."

4.5  [Section 4.3.2 Evaluation Provenance] DELETE this text:
"The saved test outputs were not re-run, repaired or replaced for this report."

4.6  [Section 4.3.2 Evaluation Provenance] REPLACE:
"besides the atrial fibrillation count noted in Section 4.1.3, it lists"
WITH:
"it gives 754 instead of 788 positive atrial fibrillation recordings, lists"

4.7  [Section 4.3.3 Evaluation Measures (first paragraph)] REPLACE:
"The measures follow Section 3.3.11. For each condition, the area under the receiver operating characteristic curve (AUROC) and the F1 score are computed over recordings with a known label. The F1 score uses a fixed threshold of 0.5 on the calibrated probabilities. Macro-averages are unweighted means over the eight conditions. Calibration is measured by the Brier score, the negative log-likelihood (NLL) and the expected calibration error (ECE) with 10 equal-width bins [26], each computed per condition and averaged over the eight conditions. The 95% confidence intervals of the test results come from 200 bootstrap resamples of test recordings. These resamples are not grouped by patient, and they are fewer than the 1,000 patient-clustered resamples that the protocol specifies for grounding metrics. Because some PTB-XL patients contribute several recordings, the intervals may be somewhat narrower than patient-level intervals would be."
WITH:
"The measures are those of Section 3.3.11: per-condition AUROC and F1 at a threshold of 0.5 on calibrated probabilities over recordings with a known label, their unweighted macro-averages, and the Brier score, negative log-likelihood (NLL) and expected calibration error (ECE) with 10 equal-width bins [26]. The 95% intervals come from 200 record-level bootstrap resamples, not the 1,000 patient-clustered resamples that the protocol specifies for grounding metrics, so they may be somewhat narrow."

4.8  [Section 4.4.1 Validation Performance and Seed Variability] DELETE this text:
"The ordering of the configurations is the same for every seed: the 12-, 10-, 8- and 4-lead configurations form a group with macro-AUROC between 0.948 and 0.953, the 6- and 2-lead configurations are slightly lower, and the single-lead configuration is clearly lowest."

4.9  [Section 4.4.4 Per-Condition Performance] DELETE this text:
"A threshold chosen on the validation partition might give a higher F1, but no such threshold was pre-specified, and choosing one now from the test results would not be legitimate."

4.10  [Section 4.4.6 Comparison Models] REPLACE:
"This comparison is descriptive and must not be read as evidence that the observation tokens or the lead-availability tokens cause the improvement. The two models differ in several ways at once: the transformer has more capacity, it was trained on all seven configurations whereas the probe saw only 12-lead representations, it receives observation and lead-availability tokens, and its probabilities are calibrated. Under lead removal, the probe is applied to representations unlike any it was fitted on, so part of its decline reflects that mismatch. Separating these factors would need ablation experiments, which have not been run. In addition, the probe's per-recording predictions were not saved, so paired confidence intervals for the differences cannot be computed, and the script that produced these results is outside the repository and was run after the test evaluation (Section 4.3.2)."
WITH:
"This comparison is descriptive. The two models differ at once in capacity, in training (the probe saw only 12-lead representations), in their observation and lead-availability inputs and in calibration, so the gap cannot be attributed to any one factor without ablations, which have not been run. The probe's per-recording predictions were not saved, so no paired intervals are available (Section 4.3.2)."

4.11  [Section 4.4.6 Comparison Models] REPLACE:
"The other two comparison models are reported only for completeness. The class-prevalence model (B-0) has a macro-AUROC of 0.5 and a macro-F1 of 0 by construction. The rule-engine model (B-2), which applies threshold rules to the extracted observations, has a macro-F1 between 0.2995 and 0.3507 across configurations, but its result has been withdrawn as a comparison because two defects make it uninformative. It never predicts either bundle branch block, so its F1 for both is 0, because the extracted QRS duration never reaches the 120 ms that its rules require (Section 4.4.8). Its atrial fibrillation rule looks for the text "Atrial fibrillation", whereas the extractor writes "Atrial Fibrillation / Irregular rhythm", so the main branch of that rule never matches; its 168 true and 203 false atrial fibrillation detections come only from a secondary rule for irregular rhythms above 100 beats per minute."
WITH:
"The class-prevalence model (B-0) has a macro-AUROC of 0.5 and a macro-F1 of 0 by construction. The rule-engine model (B-2) is withdrawn as a comparison: it never predicts bundle branch block, because the extracted QRS duration never reaches 120 ms (Section 4.4.8), and the main branch of its atrial fibrillation rule never matches the extractor's label "Atrial Fibrillation / Irregular rhythm"."

4.12  [4.2 Ecosystem: replace the paragraph directly below Table 4.4 (beginning 'The training settings in Table 4.4 are those that the training run actually used') with]
NEW TEXT:
The training settings in Table 4.4 are those that the run actually used. The training configuration file specifies a batch size of 32, at most 20 epochs and two warm-up epochs, but the training script uses its own defaults of 64, 15 and none. The saved run confirms this: the seed-42 optimiser state shows 2,825 steps per epoch, which is 180,775 examples in batches of 64, and the recorded learning rates follow a 15-epoch cosine schedule from $10^{-4}$ to $10^{-6}$ without warm-up.

4.13  [4.4.7 Representation Stability: replace the paragraph beginning 'Two cautions affect the interpretation of these results.' with]
NEW TEXT:
Two cautions apply. Zero-masking changes the input distribution as well as the information available, and the two effects are hard to separate [20]. In addition, ECG-FM is used outside its pretraining conditions (Section 3.3.6) [7], and record-level overlap between its pretraining data and the project's test recordings has not been checked.

4.14  [4.4.8 Clinical Observation Extractor: replace the last paragraph (beginning 'These limitations matter for interpreting the classification results.') with]
NEW TEXT:
The Control Reasoning Transformer therefore receives capped QRS values and a merged rhythm token, so the observation tokens cannot be assumed to contribute the information they were designed to provide. The same limits will matter more for the grounding gate: with the current extractor, a bundle branch block claim could never be supported by a measured QRS duration (Section 3.3.9).

4.15  [4.4.9 Discussion: replace the second paragraph (beginning 'These findings answer only the diagnostic half') with]
NEW TEXT:
These findings answer only the diagnostic half of the research question. A model can keep a high AUROC on a single lead by relying on features that a cardiologist would not accept as evidence, and attention-based traces cannot settle this [13]; measuring it requires the grounding gate.

4.16  [4.4.9 Discussion: replace the third paragraph (beginning 'Several factors other than the reasoning model') with]
NEW TEXT:
Several factors other than the reasoning model could explain parts of the results: the cohorts' different labelling processes, the encoder's non-native inputs and possible pretraining overlap, zero-masking, the observation extractor's limitations and the duplicated recordings. None of them invalidates the results as descriptive measurements, but together they limit how far the results can be generalised.

==================== CHAPTER 4, TABLE 4.5 ====================
Table 4.5 "Experiments Conducted": delete the "Purpose" column (header and every cell). Keep all rows and the other columns (No., Experiment, Evidence artefact, Status, Key caveat) unchanged.

==================== FINAL CHECKS (report back) ====================
1. All chapter and section headings listed above are still present and numbered as before.
2. List of Tables unchanged (3.1-3.3, 4.1-4.10). List of Figures unchanged: keep every figure, including Figures 4.1 and 4.2; do not delete, move or resize any figure.
3. No "??" from broken \ref or \cite after compiling.
4. Report the new total page count and how many pages each chapter now takes.
5. Chapters 5, 6 and 7 and all front matter are unchanged.

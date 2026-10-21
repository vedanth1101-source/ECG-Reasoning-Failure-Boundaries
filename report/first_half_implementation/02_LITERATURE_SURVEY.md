# CHAPTER 2

# LITERATURE SURVEY

## 2.1 PAPERS RELATED TO THE PROBLEM IDENTIFIED

This section reviews the literature in the order in which it bears on the research problem: the clinical criteria that define ECG evidence, automated interpretation from rule-based programs to foundation models, reduced-lead analysis and reconstruction, and the evaluation of explanations and claims. For each group of studies, the discussion notes what was established and what remains open for the present problem.

**Standardised Criteria for ECG Interpretation**

The AHA/ACCF/HRS recommendations for ECG standardisation and interpretation are the clinical reference for the evidence an ECG diagnosis requires. Part I covers acquisition and lead systems [1], and later parts define the measurement criteria behind diagnostic statements. Part III addresses intraventricular conduction disturbances, including right and left bundle branch block [2], and Part VI addresses the ECG diagnosis of acute ischaemia and infarction [3]. These documents provide the clinical basis for lead-specific diagnostic criteria. The present project asks how criteria of this kind can be applied when some of the leads they refer to are missing.

**Computer-Based Interpretation Programs**

Automated ECG interpretation predates deep learning by several decades. Schläpfer and Wellens reviewed computer-interpreted ECGs and reported that the majority of ECGs are now analysed automatically, that limitations in diagnostic accuracy have persisted despite algorithmic improvements, that inexperienced physicians may accept automated statements without recognising errors, and that over-reading by an experienced reader remains recommended [4]. Such programs rest on measurement algorithms such as the Pan–Tompkins QRS detector [14], so their statements can be traced to measurements. The review points to the need for human verification, a need that is greater for learned models whose outputs are not derived from explicit measurements.

**Deep Neural Networks for ECG Classification**

Hannun et al. trained a deep neural network on 91,232 single-lead ambulatory ECGs from 53,549 patients to classify 12 rhythm classes, and reported an average F1 score of 0.837 compared with 0.780 for the average cardiologist on a test set annotated by a cardiologist committee [5]. Strodthoff et al. benchmarked deep time-series classifiers on PTB-XL, a public dataset of 12-lead ECGs annotated by cardiologists [18], and found that ResNet- and Inception-based convolutional networks performed best [6]. These studies establish that deep networks reach high classification performance on both single-lead and 12-lead recordings. Both evaluate predicted against reference labels; apart from an exploratory interpretability analysis by Strodthoff et al., neither, as reported, examines whether the waveform evidence for an individual prediction is present in the input.

**Self-Supervised Pretraining and Random Lead Masking**

Oh et al. observed that earlier pretrained ECG models could not be fine-tuned on an arbitrary lead set unless pretrained on the same set, and proposed random lead masking, which removes leads at random during pretraining, together with objectives that learn local and global representations; the method was evaluated on arrhythmia classification and patient identification [9]. This approach makes a pretrained model usable with partial inputs. Accepting a reduced lead set is, however, different from showing that a prediction made from it is supported by the evidence it contains, and the study assesses only the former.

**ECG Foundation Models**

In a study using 1.5 million 12-lead ECGs, McKeen et al. presented ECG-FM, an open transformer-based ECG foundation model pretrained with a hybrid self-supervised method that combines wav2vec 2.0 signal masking, the contrastive multi-segment coding objective of CLOCS and random lead masking [7]. The model was evaluated on multi-label interpretation and on prediction of reduced left ventricular ejection fraction in a public MIMIC-IV-ECG benchmark. Fine-tuned ECG-FM models reached, for example, an area under the curve of 0.996 for atrial fibrillation, the frozen pretrained encoder showed competitive linear-probing performance, and the authors also analysed latent-space structure and attention-based saliency. ECG-FM is the encoder used in this project. Its published evaluation concerns task performance and saliency inspection, not whether claims made by a reasoning stage built on its representations can be verified from the available leads.

**Knowledge-Enhanced ECG Representation Learning**

Liu et al. proposed K-MERL, which uses large language models to extract structured knowledge from free-text ECG reports and a lead-aware encoder with dynamic lead masking so that arbitrary lead combinations can be used as input; it reported an average improvement of 16% in AUC over existing methods in partial-lead zero-shot classification [19]. This is the reviewed work closest to the present project, since it combines clinical knowledge with variable lead inputs. Its knowledge, however, serves representation learning and classification; as reported, it is not used to check whether an individual diagnostic claim is supported by measurements in the available leads.

**Reduced-Lead Classification and Lead Selection**

The PhysioNet/Computing in Cardiology Challenge 2021, described by Reyna et al., sourced 131,149 recordings from ten international sources and asked teams to classify 30 conditions from several lead combinations. Performance differed less across lead combinations than across test databases, indicating that generalisation was a larger challenge than the choice of leads [8]. Lai et al. introduced a lead-selection stage within a deep model trained on CPSC 2018 data, identified an optimal four-lead subset of II, aVR, V1 and V4, and found that it significantly outperformed the full 12-lead input on validation and external test data, which they attribute to redundancy among leads [10]. These studies show that reduced lead sets can retain substantial classification performance and that a single score on one dataset is an incomplete description of a model. Their outcome measure, however, is classification performance rather than whether the criteria for each predicted diagnosis remain observable in the recorded leads.

**Reconstruction of Missing Leads**

A second line of research reconstructs missing leads from the remaining ones. Presacan et al. used a generative adversarial network to reconstruct 12-lead ECGs from lead I, or leads I and II, in PTB-XL. The reconstructed precordial amplitudes showed reduced variability and bias, a regression-to-the-mean effect that led the authors to conclude that such reconstruction is unsuitable for clinical use [11]. Reconstruction thus replaces measurements with estimates, and its evaluation measures waveform fidelity rather than whether diagnostic claims are supported by recorded evidence.

**Explainability in Clinical Machine Learning**

In a review of explainable AI in cardiology, Salih et al. found that 37% of the reviewed studies evaluated explanations against published literature, 11% used clinicians as domain experts, 11% used proxies or statistical analysis, and 43% did not assess their explanations at all [15]. Ghassemi et al. argued that current explainability methods are unlikely to give reliable patient-level explanations and recommended rigorous internal and external validation of models instead [12]. These works agree that explanations require evaluation, but neither provides a procedure for checking ECG explanations against the signal evidence that remains after leads are removed.

**Faithfulness of Attribution and Attention-Based Explanations**

Jain and Wallace found that attention weights were often uncorrelated with gradient-based measures of feature importance and that very different attention distributions could yield the same predictions [13]. Hooker et al. evaluated feature-attribution methods by removing increasing fractions of the input features ranked most important and retraining the model, a procedure they called ROAR, and found that many attribution methods ranked features no better than a random assignment [20]. They also reported that a ResNet-50 retrained with 90% of the input removed still reached 63.53% ImageNet accuracy, compared with 76.68% on clean data, and argued that degradation measured without retraining may reflect a shift in the input distribution rather than the loss of information. These studies concern language and image models and transfer to ECG models only by analogy, but they indicate that attention maps should not be treated as verified evidence without an independent check, and that performance under removed input can mix information loss with distribution shift. The ECG-FM authors themselves present their attention-based saliency maps as a heuristic for relative input importance [7], and the attention-based traces produced in the present project require the same caution.

**Claim Verification and Structured Clinical Knowledge**

Thorne et al. introduced FEVER, a dataset of 185,445 claims labelled by annotators as Supported, Refuted or NotEnoughInfo, with evidence sentences recorded for the first two labels; the best system reached 31.87% accuracy when the correct evidence was also required [17]. The dataset concerns text, but it establishes the practice of separating claims that evidence supports, claims that evidence refutes, and claims for which the available evidence is insufficient. In clinical informatics, SNOMED International describes how SNOMED CT-encoded knowledge can be combined with health records in clinical decision support [16]. In the material reviewed, this guidance does not supply lead-level ECG measurement criteria, which come from the clinical recommendations discussed earlier. Combining the two is an adaptation proposed by the present project and not yet implemented.

## 2.2 GAPS IDENTIFIED

Comparing the reviewed studies reveals the following gaps. They are stated with respect to the methods and results reported in the literature reviewed here, not as claims that no study anywhere has addressed them.

**Reduced-Lead Evaluation Measures Classification Rather Than Evidence**

The reduced-lead studies reviewed evaluate classification across lead subsets [8, 10], but neither tests whether the criteria for a predicted diagnosis remain observable in the leads used. Since a four-lead subset can outperform twelve leads [10], high reduced-lead performance cannot be taken as evidence that the defining, lead-specific findings were observed.

**Representation Stability Is Not Related to the Availability of Diagnostic Evidence**

Random lead masking, ECG-FM and K-MERL produce representations from partial inputs and evaluate them by downstream performance [7, 9, 19]. As reported, none of these studies relates the stability of a representation under lead removal to whether the evidence for particular diagnoses is still present, so a representation that changes little is not, on this evidence, an indication that the claims built on it remain supported.

**Unavailable Evidence Is Not Distinguished from Contradicting Evidence in ECG Explanations**

Text-based fact verification separates claims that evidence refutes from claims for which evidence is insufficient [17]. The ECG explanation reviewed, attention-based saliency for ECG-FM [7], indicates the relative importance of input regions but does not classify a diagnostic claim as unassessable when its required leads are absent. Reconstruction instead estimates the missing leads, with estimates that regress towards population averages [11]. None of the reviewed ECG approaches represents the state in which a claim can be neither supported nor contradicted because the necessary evidence was not recorded.

**Progressive Evaluation Is Limited to Performance or Attribution Quality**

The 2021 Challenge evaluated several lead combinations [8], and progressive removal of input features has been used to evaluate attribution methods in image classification [20]. These evaluations track classification accuracy or attribution quality. No reviewed study follows the grounding of individual claims across an ordered sequence of lead configurations, so the point at which claims stop being supported has not been characterised.

**Measurements, Lead Availability and Clinical Knowledge Are Not Combined for Claim Checking**

Clinical recommendations define diagnostic criteria for conduction disturbances and for ischaemia and infarction [2, 3], SNOMED CT-encoded knowledge is used in decision support [16], and a knowledge-enhanced ECG model uses clinical knowledge to improve representations [19]. In the reviewed work these elements appear separately; none of the reviewed ECG systems combines measured physiological findings, the current lead-availability state and structured clinical knowledge to check whether each diagnostic claim is supported, contradicted or unassessable.

**Explanations Are Rarely Validated Against Measurable Evidence**

The review of explainable AI in cardiology reports that many explanations are not evaluated at all [15], and the faithfulness literature cautions against treating attention or attribution as verified evidence [13, 20]. The reviewed studies do not validate ECG explanations against physiological measurements taken from the leads that remain after lead removal.

In summary, the reviewed literature shows how well ECG models classify with full and reduced inputs and how difficult explanations are to evaluate, but not whether the claims such models make remain supported by the evidence in the available leads. The research problem that follows is to characterise, across a controlled sequence of lead configurations, the relationship between lead availability, model representations, classification behaviour and the grounding of individual diagnostic claims, while keeping unavailable evidence distinct from contradicting evidence.

## 2.3 RESEARCH OBJECTIVES

The overall objective of this project is to investigate whether the diagnostic claims of a reasoning system built on a frozen ECG foundation model remain grounded in observable physiological evidence as ECG leads are progressively removed. The following objectives are derived from the gaps identified above.

1. To establish a controlled evaluation pipeline in which each ECG recording is presented at seven lead-availability levels, from 12 leads to 1 lead, with missing leads removed rather than reconstructed and with the dependence of derived limb leads on leads I and II recorded explicitly.
2. To analyse the relationship between lead availability, the representations produced by a frozen ECG foundation model and the classification behaviour of a reasoning model built on those representations.
3. To identify the physiological observations and lead requirements needed to assess claims for the eight benchmark conditions, and to extract those observations only from the available leads.
4. To design a grounding framework that classifies each clinical claim as SUPPORTED, CONTRADICTED or UNAVAILABLE according to the measured evidence and the current lead-availability state.
5. To investigate the use of structured clinical knowledge, in the form of a Clinical Knowledge Prior derived from SNOMED CT, for linking diagnostic claims to the findings required to assess them.
6. To define an evaluation procedure that identifies the Reasoning Failure Boundary, the first configuration in the order 12, 10, 8, 6, 4, 2 and 1 leads at which grounding quality falls below a threshold chosen by the project, that reports the sensitivity of this boundary to the choice of threshold, and that reports any later recovery of grounding quality separately instead of assuming a monotonic decline, since the configurations are not nested subsets of one another.

These objectives are at different stages. In the first half of the project, the pipeline in Objective 1 has been implemented, and Objective 2 has produced preliminary representation and classification results, which are presented with their limitations in Chapter 4. For Objective 3, the observation extractor has been implemented and the evidence requirements specified, although the extractor has known limitations. The grounding states and the Reasoning Failure Boundary of Objectives 4 and 6 are defined in the project's internal evaluation protocol (version 2.0, dated 14 October 2026), which describes itself as pre-registered and whose cryptographic hash was recorded in the project's Module 4 freeze record on 15 October 2026; the protocol has not been registered with any external registry. The knowledge-augmented path of Objective 5 appears in that protocol only as a planned comparison. None of Objectives 4 to 6 has yet been implemented, and together they form the main work of the second half. No objective includes clinical validation or deployment, and the threshold in Objective 6 is an analysis parameter rather than a clinical standard. Table 2.1 relates the gaps to the objectives.

**Table 2.1 Identified Research Gaps and Proposed Research Objectives**

| Category | Identified Gaps | Research Objectives |
|---|---|---|
| Evaluation under lead removal | Reduced-lead studies measure classification; progressive evaluations track performance or attribution quality rather than evidence grounding | **1.** Controlled seven-level lead-removal pipeline without reconstruction. **6.** Boundary-based evaluation of grounding across the ordered configurations |
| Representations and evidence | Representation stability under partial input is not related to the availability of diagnostic evidence | **2.** Analysis of lead availability, foundation-model representations and classification behaviour |
| Evidence states | ECG explanation methods do not separate unavailable evidence from contradicting evidence | **4.** Grounding framework with SUPPORTED, CONTRADICTED and UNAVAILABLE states |
| Measurements and clinical knowledge | Measured findings, lead availability and structured clinical knowledge are not combined to check individual claims | **3.** Lead requirements and observation extraction for the eight conditions. **5.** Investigation of a SNOMED CT-based Clinical Knowledge Prior |
| Explanation validation | Explanations are rarely validated against measurable evidence that remains after lead removal | **4.** and **6.** Claim-level grounding and its evaluation across lead configurations |


<!-- CITATION MAP (not rendered). Global numbering continues from earlier chapters; numbers shared with earlier chapters are unchanged. Keys refer to REFERENCES.bib / REFERENCE_LIBRARY.md.
[1] KLIGFIELD2007AHA
[2] SURAWICZ2009AHA
[3] WAGNER2009AHA
[4] SCHLAPFER2017COMPUTER
[5] HANNUN2019CARDIOLOGIST
[6] STRODTHOFF2021PTBXL
[7] MCKEEN2025ECGFM
[8] REYNA2022ISSUES
[9] OH2022LEADAGNOSTIC
[10] LAI2021LEADSELECTION
[11] PRESACAN2025RECON
[12] GHASSEMI2021FALSEHOPE
[13] JAIN2019ATTENTION
[14] PAN1985QRS
[15] SALIH2024XAICARDIO
[16] SNOMED2026CDSGUIDE
[17] THORNE2018FEVER
[18] WAGNER2020PTBXL
[19] LIU2025ARBITRARY
[20] HOOKER2019ROAR
-->

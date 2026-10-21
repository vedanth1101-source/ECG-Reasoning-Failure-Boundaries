# CHAPTER 5

# SOCIAL IMPACT & SUSTAINABILITY

## 5.1 SOCIAL IMPACT

Most ECGs are interpreted automatically, and clinicians may accept computer statements without checking them [4]. Lower-cost devices that record fewer leads may widen access to cardiac care [8], and single-lead monitors are already used for rhythm assessment [5]. A model developed on 12-lead recordings may therefore be applied to incomplete recordings without any warning that the evidence for some diagnoses is missing. This project studies that risk under controlled conditions, using seven zero-masked lead configurations and a frozen encoder. Its results show that the number of leads is a poor guide to reliability: the 4-lead configuration with V1 outperformed the 6-lead limb configuration, and the F1 score for right bundle branch block fell to 0.13 with lead II alone. Findings of this kind may help developers test models against stated lead configurations, and may in future help tell clinicians when an automated statement rests on leads that were not recorded.

These benefits are prospective. The project is a research evaluation, not a diagnostic tool, and it has not been clinically validated. Its test results are descriptive measurements from one seed of an authorisation-unverified run. The proposed grounding gate, with its SUPPORTED, CONTRADICTED and UNAVAILABLE states, and the Reasoning Failure Boundary have not been implemented, and the attention-based traces are not validated explanations [12, 13]. The datasets are public PhysioNet releases whose de-identification is documented by their providers; the project adds no privacy measures of its own and claims no ethics approval or regulatory compliance. Because the cohorts were pooled, and the MIMIC-IV-ECG labels are machine-generated [25], the results are not an external validation.

## 5.2 SUSTAINABILITY

The project reuses the pretrained ECG-FM encoder, with 90,883,072 parameters, without retraining it [7]. Only the Control Reasoning Transformer, with 1,830,657 parameters, was trained. The encoder's representations were computed once on the CPU for all 258,216 recording–configuration pairs and stored in about 0.83 GB, so the three training runs, of 5,763 to 5,973 seconds each, needed no passes through the encoder. A single model was trained for all seven configurations instead of one per configuration. Freeze records, split manifests and SHA-256 hashes made the Chapter 4 results checkable against their sources, although software versions and training logs were not recorded. No energy or carbon use was measured, so no savings are claimed.


<!-- CITATION MAP (not rendered). Global numbering continues from earlier chapters; numbers shared with earlier chapters are unchanged. Keys refer to REFERENCES.bib / REFERENCE_LIBRARY.md.
[4] SCHLAPFER2017COMPUTER
[5] HANNUN2019CARDIOLOGIST
[7] MCKEEN2025ECGFM
[8] REYNA2022ISSUES
[12] GHASSEMI2021FALSEHOPE
[13] JAIN2019ATTENTION
[25] GOW2023MIMICIVECG
-->

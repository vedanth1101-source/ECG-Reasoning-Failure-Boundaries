# REFERENCE LIBRARY — Verified Academic Sources

**Project:** Reasoning Failure Boundaries in ECG Foundation Models Under Progressive Physiological Information Loss
**Department:** Computer Science and Engineering, SSN College of Engineering — 2026–2027
**Built:** 2026-10-19 · **Companion files:** `RESEARCH_LIBRARY.bib` (all 79 entries, same citation keys), `REFERENCES.bib` (report set only), `CITATION_MATRIX.md`
**Scope:** Academic sources only. Evidence about the project's own code, data and results lives in `PROJECT_CONTEXT.md`. A citation in this file never shows that the project's code works or that it produced a particular result.

> **Report reference set (locked 2026-10-19).** This file is the **research library**, not the report's reference list. The report cites a fixed set of **26 sources**: 20 already cited in Chapters 1–2 and 6 reserved for Chapters 3–4. The set and the chapter-by-chapter use of each source are listed in `CITATION_MATRIX.md` §A–B. The other 53 entries are kept here for future work and must not appear in Chapter 7 unless they are formally added to the report set.

---

## How sources were selected and verified

**Selection.** I picked sources for the claims the thesis will actually make (listed in `PROJECT_CONTEXT.md` §A, §F–§H), not for ECG in general. Primary research is preferred for methods and results. Dataset descriptors and official PhysioNet records are used for dataset facts, clinical standards and consensus documents for ECG criteria, and reviews only for describing the research landscape.

**Verification, done on 2026-10-19:**

1. **Crossref REST API** for every journal and proceedings DOI: title, subtitle, authors, venue, volume, issue, pages or article number, year.
2. **DataCite API** for PhysioNet dataset DOIs (10.13026/…) and arXiv DOIs (10.48550/arXiv.…).
3. **PubMed E-utilities** for page ranges that Crossref lacks (Circulation e-pages), for full author lists where Crossref's list was incomplete, and for the Donnelly 2006 record (no DOI).
4. **Official proceedings pages** for papers without DOIs: PMLR (ICML, CHIL), NeurIPS, ICLR. I read the volume, page and title metadata from the pages themselves.
5. **GitHub API** and the official web pages for software and documentation (ECG-FM, fairseq-signals, the SNOMED CT CDS Guide).
6. **Resolution check.** All 64 DOIs in `REFERENCES.bib` resolve through the doi.org handle API, and all 11 URLs return a page. `REFERENCES.bib` also passed a structural validation: 76 entries, unique keys, balanced braces, required fields present, well-formed DOIs and years.

**Status labels**

- **Verified** — peer-reviewed; metadata confirmed against at least one authoritative registry.
- **Verified (preprint)** — the metadata is confirmed, but the work is not peer-reviewed. Use it only as supporting context.
- **Verified (official documentation/software)** — an official dataset record, standard, or code repository.
- No entry is labelled *Unverified*. Sources I could not verify were left out (see the excluded list at the end).

**Limits of this verification.** I checked each source's metadata and what it reports (from abstracts, proceedings pages, and, for ECG-FM, the paper text). I did **not** read every full text. Before a later chapter quotes a specific number or finding from a source, check it against that source's full text.

**Formatting.** Entries are in the IEEE-like numbered style the sample thesis uses (natbib `square, numbers`). Author lists longer than six are shortened to "et al." here; `REFERENCES.bib` keeps up to ten names and then "and others".

---

## Findings from verification that affect the thesis

1. **ECG-FM's pretraining input contract differs from the project's.** According to McKeen et al. (arXiv 2408.05178v2, the version of `[MCKEEN2025ECGFM]` I read), ECG-FM was pretrained on waveforms resampled to **500 Hz**, cut into **5 s** segments, **z-score normalised**, with random lead masking at **p = 0.5**. The project feeds the frozen model **100 Hz, 10 s, unnormalised millivolt** input (`PROJECT_CONTEXT.md` §E.1). The thesis has to state this as a limitation. No cited source shows that ECG-FM representations remain valid under the project's input contract.
2. **The backbone's pretraining data overlap the project's cohorts.** ECG-FM's README and paper list MIMIC-IV-ECG and PhysioNet 2021 as pretraining data, and PhysioNet 2021 includes PTB-XL and Chapman. That pretraining was self-supervised (unlabelled). Even so, the project's test records may have been seen by the backbone during pretraining, and Chapter 4/5 must disclose this.
3. **Random lead masking in pretraining** (ECG-FM README: W2V+CMSC+RLM; `[OH2022LEADAGNOSTIC]`) is a published, plausible explanation for why ECG-FM embeddings change little when the derived limb leads are removed. Treat it as a hypothesis, not as a measured cause.
4. **Errors in the existing `report/07_REFERENCES.md`** (not modified; found while checking):

| Existing entry | Problem found | Replacement in this library |
|---|---|---|
| Atoui, Fayn & Rubel 2010, *Computers in Cardiology* 37, doi 10.1109/CIC.2010.5681532 | DOI does not resolve | `[ATOUI2010RECON]` (IEEE TITB 14(3), 2010) |
| Dawson, Yang & Balasubramanian 2009, doi 10.1016/j.jelectrocard.2009.07.009 | DOI resolves to a different paper (Nikus et al., acute coronary syndromes review) | Removed; use `[NELWAN2004RECON]`, `[PRESACAN2025RECON]` |
| Hao, Chen & Lu 2021, IEEE TBME, doi 10.1109/TBME.2020.3045231 | DOI does not resolve | Removed |
| Lakkaraju et al. 2020, doi 10.48550/arXiv.2003.06578 | DOI resolves to an unrelated mathematics preprint (Stokes flow) | Removed (not needed) |
| Li, Zhang & Wang 2023, CBM 165:107412 | DOI resolves to an unrelated endoscopy paper | Removed |
| Liu et al. 2025, EMNLP Findings, doi …findings-emnlp.482 | DOI resolves to "EventRelBench"; correct DOI is …findings-emnlp.385 | `[LIU2025ARBITRARY]` |
| Nelwan et al. 2004, doi 10.1016/j.jelectrocard.2004.08.061 | DOI resolves to a different paper (prehospital diversion); authors and title also differ | `[NELWAN2004RECON]` (doi 10.1016/j.jelectrocard.2003.10.004) |
| Zheng et al. 2020, "PhysioNet", doi 10.13026/gfy7-fc21 | DOI does not resolve; author "Daniasekry" should be Danioko; venue is *Scientific Data* | `[ZHENG2020CHAPMAN]` |
| Frank 1956, "dipolar vectorcardiography" | Registered title is "…System For Spatial Vectorcardiography" | Not included (not needed) |
| Zhang & Fink 2025 (water-network leak detection) | Irrelevant to the project | Removed |
| ECG-FM | Missing although it is the backbone | `[MCKEEN2025ECGFM]` |

---

## Index by category

| # | Category | Keys |
|---|---|---|
| 1 | ECG fundamentals and clinical interpretation | KLIGFIELD2007AHA, MASON2007AHA, SURAWICZ2009AHA, RAUTAHARJU2009AHA, WAGNER2009AHA, THYGESEN2018UDMI, GOLDBERGER1942AUGMENTED, PAN1985QRS, WILLEMS1991CSE, SCHLAPFER2017COMPUTER, DREW2004MONITORING, SANDAU2017MONITORING |
| 2 | Deep learning for ECG analysis (incl. robustness evaluation) | HANNUN2019CARDIOLOGIST, RIBEIRO2020AUTOMATIC, STRODTHOFF2021PTBXL, ATTIA2019AF, SOMANI2021REVIEW, ZHANG2021INTERPRETABLE, VASWANI2017ATTENTION, GUO2017CALIBRATION, PEREZALDAY2020CINC, REYNA2022ISSUES, HENDRYCKS2019CORRUPTIONS, HOOKER2019ROAR |
| 3 | ECG foundation models | BOMMASANI2021FM, BAEVSKI2020WAV2VEC, KIYASSEH2021CLOCS, OH2022LEADAGNOSTIC, MCKEEN2025ECGFM, ECGFM2024REPO, FAIRSEQSIGNALS, MEHARI2022SSL, NA2024STMEM, LIU2024MERL, LIU2025ARBITRARY, HAN2024FMREVIEW, COPPOLA2024HUBERT |
| 4 | Reduced-lead ECG and missing channels | REYNA2021WILLTWO, LAI2021LEADSELECTION, CHO2020SIXLEAD, ATTIA2022SMARTWATCH, BARRETT2014PATCH, NELWAN2004RECON, ATOUI2010RECON, PRESACAN2025RECON, GRADOWSKI2025LEADINFO |
| 5 | Explainability and evidence-grounded medical AI | JUNG2023XAISR, AMANN2020XAI, GHASSEMI2021FALSEHOPE, SALIH2024XAICARDIO, RUDIN2019STOP, JACOVI2020FAITHFUL, JAIN2019ATTENTION, WIEGREFFE2019ATTENTION, ADEBAYO2018SANITY, JI2023HALLUCINATION, SINGHAL2023LLMCLIN, RAJPURKAR2022AI, KELLY2019CHALLENGES, THORNE2018FEVER, WADDEN2020SCIFACT |
| 6 | Clinical knowledge and decision support | SNOMED2026CDSGUIDE, DONNELLY2006SNOMED, BODENREIDER2004UMLS, SUTTON2020CDSS |
| 7 | Datasets | WAGNER2020PTBXL, WAGNER2022PTBXLDATA, ZHENG2020CHAPMAN, ZHENG2022CHAPMANDATA, CINC2020DATA, CINC2021DATA, GOW2023MIMICIVECG, JOHNSON2023MIMICIV, GOLDBERGER2000PHYSIONET, POLLARD2026PHYSIONET, SECHIDIS2011STRAT |
| 8 | Supplementary: reporting and sustainability | COLLINS2024TRIPODAI, SCHWARTZ2020GREENAI, STRUBELL2019ENERGY |

Totals: **79 sources** (POLLARD2026PHYSIONET added 2026-10-19). 67 are peer-reviewed and verified (HENDRYCKS2019CORRUPTIONS and HOOKER2019ROAR were added on 2026-10-19 for the robustness discussion in Chapter 2; both are general machine-learning works, not ECG studies), 4 are preprints (BOMMASANI2021FM, HAN2024FMREVIEW, COPPOLA2024HUBERT, GRADOWSKI2025LEADINFO), and 8 are official datasets, documentation or software. Category 8 is a supplementary group added for Chapters 4–5; it is outside the seven requested categories.

## 1. ECG fundamentals and clinical interpretation

### `[KLIGFIELD2007AHA]`

- **Entry:** P. Kligfield et al., “Recommendations for the Standardization and Interpretation of the Electrocardiogram: Part I: The Electrocardiogram and Its Technology,” *Circulation*, vol. 115, no. 10, pp. 1306–1324, 2007. doi: [10.1161/CIRCULATIONAHA.106.180200](https://doi.org/10.1161/CIRCULATIONAHA.106.180200).
- **Source type:** Peer-reviewed clinical recommendation (AHA/ACCF/HRS)
- **Verification status:** Verified
- **Contribution:** Part I of the AHA/ACCF/HRS recommendations: ECG acquisition, lead systems and technology, including the standard 12-lead system and the derivation of limb leads.
- **Supports (thesis use):** Standard 12-lead ECG definition; Einthoven limb leads and augmented leads; acquisition standards (Ch. 1.2, 3).
- **Caution:** Does not address machine-learning models.

### `[MASON2007AHA]`

- **Entry:** J. W. Mason et al., “Recommendations for the Standardization and Interpretation of the Electrocardiogram: Part II: Electrocardiography Diagnostic Statement List,” *Circulation*, vol. 115, no. 10, pp. 1325–1332, 2007. doi: [10.1161/CIRCULATIONAHA.106.180201](https://doi.org/10.1161/CIRCULATIONAHA.106.180201).
- **Source type:** Peer-reviewed clinical recommendation
- **Verification status:** Verified
- **Contribution:** Part II: the standardised list of ECG diagnostic statements.
- **Supports (thesis use):** That diagnostic ECG statements are standardised; mapping of label vocabularies (Ch. 3, 4.1).
- **Caution:** Statement list, not diagnostic criteria thresholds.

### `[SURAWICZ2009AHA]`

- **Entry:** B. Surawicz et al., “AHA/ACCF/HRS Recommendations for the Standardization and Interpretation of the Electrocardiogram: Part III: Intraventricular Conduction Disturbances,” *Circulation*, vol. 119, no. 10, pp. e235–e240, 2009. doi: [10.1161/CIRCULATIONAHA.108.191095](https://doi.org/10.1161/CIRCULATIONAHA.108.191095). PMID: 19228822.
- **Source type:** Peer-reviewed clinical recommendation
- **Verification status:** Verified
- **Contribution:** Part III: intraventricular conduction disturbances, including criteria for complete RBBB and LBBB (QRS >= 120 ms, lead-specific morphology).
- **Supports (thesis use):** Lead-specific criteria for RBBB/LBBB used in the evidence schema; why V1/V6/I are needed (Ch. 1.3, 3).
- **Verification update:** **2026-10-19 check:** the full text (ahajournals.org and the JACC copy, doi 10.1016/j.jacc.2008.12.013) returned HTTP 403, and there is no open abstract on Europe PMC. Only metadata and title are verified. Chapters 1–2 therefore cite it only for the existence of Part III criteria for intraventricular conduction disturbances. The QRS ≥ 120 ms threshold is attributed to the project's evidence schema. A manual full-text check is recommended before Chapter 3 cites specific criteria.

### `[RAUTAHARJU2009AHA]`

- **Entry:** P. M. Rautaharju et al., “AHA/ACCF/HRS Recommendations for the Standardization and Interpretation of the Electrocardiogram: Part IV: The ST Segment, T and U Waves, and the QT Interval,” *Circulation*, vol. 119, no. 10, pp. e241–e250, 2009. doi: [10.1161/CIRCULATIONAHA.108.191096](https://doi.org/10.1161/CIRCULATIONAHA.108.191096). PMID: 19228821.
- **Source type:** Peer-reviewed clinical recommendation
- **Verification status:** Verified
- **Contribution:** Part IV: the ST segment, T and U waves, and the QT interval.
- **Supports (thesis use):** Definitions of ST/T measurements used by the observation extractor (Ch. 3).

### `[WAGNER2009AHA]`

- **Entry:** G. S. Wagner et al., “AHA/ACCF/HRS Recommendations for the Standardization and Interpretation of the Electrocardiogram: Part VI: Acute Ischemia/Infarction,” *Circulation*, vol. 119, no. 10, pp. e262–e270, 2009. doi: [10.1161/CIRCULATIONAHA.108.191098](https://doi.org/10.1161/CIRCULATIONAHA.108.191098). PMID: 19228819.
- **Source type:** Peer-reviewed clinical recommendation
- **Verification status:** Verified
- **Contribution:** Part VI: acute ischaemia/infarction, including contiguous-lead groupings and ST-elevation thresholds.
- **Supports (thesis use):** Territory/contiguous-lead rule for MI claims; why MI evidence becomes unobservable with few leads (Ch. 1.3, 3).
- **Verification update:** **2026-10-19 check:** the full text (ahajournals.org and JACC) returned HTTP 403, and there is no open abstract; the Fourth Universal Definition of MI was also not openly accessible. Only metadata and title are verified. Chapters 1–2 cite it only for the existence of Part VI criteria for acute ischaemia/infarction. Lead territories and the ≥ 0.10 mV / two-contiguous-lead rule are attributed to the project's evidence schema. A manual full-text check is recommended.

### `[THYGESEN2018UDMI]`

- **Entry:** K. Thygesen et al., “Fourth Universal Definition of Myocardial Infarction (2018),” *Circulation*, vol. 138, no. 20, pp. e618–e651, 2018. doi: [10.1161/CIR.0000000000000617](https://doi.org/10.1161/CIR.0000000000000617). PMID: 30571511.
- **Source type:** Peer-reviewed consensus document (ESC/ACC/AHA/WHF)
- **Verification status:** Verified
- **Contribution:** Fourth Universal Definition of Myocardial Infarction; ECG criteria for ischaemia in two contiguous leads. Also published in Eur Heart J 40(3):237-269 (doi:10.1093/eurheartj/ehy462).
- **Supports (thesis use):** MI ECG criteria and the fact that MI is not diagnosed from the ECG alone (Ch. 1.2, 3, 5).
- **Caution:** MI definition requires biomarkers/clinical context; ECG-only labels are surrogates. Cite one version only.

### `[GOLDBERGER1942AUGMENTED]`

- **Entry:** E. Goldberger, “A simple, indifferent, electrocardiographic electrode of zero potential and a technique of obtaining augmented, unipolar, extremity leads,” *American Heart Journal*, vol. 23, no. 4, pp. 483–492, 1942. doi: [10.1016/S0002-8703(42)90293-X](https://doi.org/10.1016/S0002-8703(42)90293-X).
- **Source type:** Peer-reviewed original research (historical)
- **Verification status:** Verified
- **Contribution:** Original description of the augmented unipolar limb leads (aVR, aVL, aVF).
- **Supports (thesis use):** Origin of augmented leads and their dependence on limb electrodes (Ch. 1.2, 3 lead provenance).
- **Caution:** Historical; pair with KLIGFIELD2007AHA for current practice.

### `[PAN1985QRS]`

- **Entry:** J. Pan and W. J. Tompkins, “A Real-Time QRS Detection Algorithm,” *IEEE Transactions on Biomedical Engineering*, vol. BME-32, no. 3, pp. 230–236, 1985. doi: [10.1109/TBME.1985.325532](https://doi.org/10.1109/TBME.1985.325532).
- **Source type:** Peer-reviewed original research
- **Verification status:** Verified
- **Contribution:** Real-time QRS detection algorithm (band-pass, derivative, squaring, moving-window integration, adaptive thresholds).
- **Supports (thesis use):** Basis of the heart-rate/R-peak step of the observation extractor (Ch. 3).
- **Caution:** Designed for 200 Hz single-channel signals; project runs at 100 Hz -- do not claim equivalence.
- **Verification update:** **2026-10-19 full-text check** (copy on a University of Oxford course page): band-pass filter (5–15 Hz design), derivative, squaring, moving-window integration (150 ms at 200 samples/s), adaptive thresholds and T-wave discrimination; 99.3% of QRS complexes detected on MIT/BIH. **Mismatch:** the project's extractor (`src/observations/clinical_extractor.py:213–227`) uses a simplified variant (5–15 Hz Butterworth band-pass, derivative, squaring, 100 ms window, a fixed mean + 1.5 SD threshold) at 100 Hz. Describe it as "Pan–Tompkins-style", not as the original algorithm.

### `[WILLEMS1991CSE]`

- **Entry:** J. L. Willems et al., “The Diagnostic Performance of Computer Programs for the Interpretation of Electrocardiograms,” *New England Journal of Medicine*, vol. 325, no. 25, pp. 1767–1773, 1991. doi: [10.1056/NEJM199112193252503](https://doi.org/10.1056/NEJM199112193252503).
- **Source type:** Peer-reviewed original research
- **Verification status:** Verified
- **Contribution:** CSE study comparing computer ECG interpretation programs against cardiologists on a validated database.
- **Supports (thesis use):** Long history and limits of automated ECG interpretation (Ch. 1.1, 2).

### `[SCHLAPFER2017COMPUTER]`

- **Entry:** J. Schläpfer and H. J. Wellens, “Computer-Interpreted Electrocardiograms,” *Journal of the American College of Cardiology*, vol. 70, no. 9, pp. 1183–1192, 2017. doi: [10.1016/j.jacc.2017.07.723](https://doi.org/10.1016/j.jacc.2017.07.723).
- **Source type:** Peer-reviewed review
- **Verification status:** Verified
- **Contribution:** Review of benefits and limitations of computer-interpreted ECGs, including over-reliance by clinicians.
- **Supports (thesis use):** Motivation: automated ECG statements can be wrong and over-trusted (Ch. 1.1, 5).

### `[DREW2004MONITORING]`

- **Entry:** B. J. Drew et al., “Practice Standards for Electrocardiographic Monitoring in Hospital Settings,” *Circulation*, vol. 110, no. 17, pp. 2721–2746, 2004. doi: [10.1161/01.CIR.0000145144.56673.59](https://doi.org/10.1161/01.CIR.0000145144.56673.59).
- **Source type:** Peer-reviewed practice standard (AHA scientific statement)
- **Verification status:** Verified
- **Contribution:** Practice standards for in-hospital ECG monitoring, including choice of monitoring leads (e.g. V1 for wide-complex rhythms, lead II/III for P waves).
- **Supports (thesis use):** Clinical motivation for reduced-lead (telemetry-like) configurations such as II + V1 (Ch. 1.1, 3).
- **Caution:** Describes monitoring practice, not diagnostic equivalence of reduced leads.

### `[SANDAU2017MONITORING]`

- **Entry:** K. E. Sandau et al., “Update to Practice Standards for Electrocardiographic Monitoring in Hospital Settings: A Scientific Statement From the American Heart Association,” *Circulation*, vol. 136, no. 19, pp. e273–e344, 2017. doi: [10.1161/CIR.0000000000000527](https://doi.org/10.1161/CIR.0000000000000527). PMID: 28974521.
- **Source type:** Peer-reviewed practice standard (AHA scientific statement)
- **Verification status:** Verified
- **Contribution:** 2017 update of the hospital ECG-monitoring practice standards.
- **Supports (thesis use):** Current monitoring practice with limited lead sets (Ch. 1.1, 3).


## 2. Deep learning for ECG analysis

### `[HANNUN2019CARDIOLOGIST]`

- **Entry:** A. Y. Hannun et al., “Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network,” *Nature Medicine*, vol. 25, no. 1, pp. 65–69, 2019. doi: [10.1038/s41591-018-0268-3](https://doi.org/10.1038/s41591-018-0268-3). PMID: 30617320.
- **Source type:** Peer-reviewed original research
- **Verification status:** Verified
- **Contribution:** Deep CNN for 12-class rhythm classification from single-lead ambulatory ECG, compared with cardiologists.
- **Supports (thesis use):** Deep learning on single-lead ECG for rhythm tasks (Ch. 1.2, 2).
- **Caution:** Rhythm classes only; single-lead success for rhythm does not extend to morphology-dependent diagnoses.

### `[RIBEIRO2020AUTOMATIC]`

- **Entry:** A. H. Ribeiro et al., “Automatic diagnosis of the 12-lead ECG using a deep neural network,” *Nature Communications*, vol. 11, no. 1, art. 1760, 2020. doi: [10.1038/s41467-020-15432-4](https://doi.org/10.1038/s41467-020-15432-4).
- **Source type:** Peer-reviewed original research
- **Verification status:** Verified
- **Contribution:** Deep residual network trained on >2 million 12-lead ECGs (CODE study) for six abnormalities.
- **Supports (thesis use):** Large-scale supervised 12-lead deep learning (Ch. 1.2, 2).

### `[STRODTHOFF2021PTBXL]`

- **Entry:** N. Strodthoff, P. Wagner, T. Schaeffter, and W. Samek, “Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL,” *IEEE Journal of Biomedical and Health Informatics*, vol. 25, no. 5, pp. 1519–1528, 2021. doi: [10.1109/JBHI.2020.3022989](https://doi.org/10.1109/JBHI.2020.3022989).
- **Source type:** Peer-reviewed original research (benchmark)
- **Verification status:** Verified
- **Contribution:** Benchmarks of deep architectures on PTB-XL tasks, with evaluation protocol based on recommended folds.
- **Supports (thesis use):** PTB-XL benchmarking practice; deep learning baselines (Ch. 2, 4).

### `[ATTIA2019AF]`

- **Entry:** Z. I. Attia et al., “An artificial intelligence-enabled ECG algorithm for the identification of patients with atrial fibrillation during sinus rhythm: a retrospective analysis of outcome prediction,” *The Lancet*, vol. 394, no. 10201, pp. 861–867, 2019. doi: [10.1016/S0140-6736(19)31721-0](https://doi.org/10.1016/S0140-6736(19)31721-0).
- **Source type:** Peer-reviewed original research
- **Verification status:** Verified
- **Contribution:** AI-ECG identifies patients with atrial fibrillation from ECGs recorded in sinus rhythm.
- **Supports (thesis use):** Deep models can extract information not visible to clinicians -- relevant to why accuracy alone does not reveal what evidence a model uses (Ch. 1.1, 2).
- **Caution:** Different task (risk prediction); do not use as evidence about lead removal.

### `[SOMANI2021REVIEW]`

- **Entry:** S. Somani et al., “Deep learning and the electrocardiogram: review of the current state-of-the-art,” *EP Europace*, vol. 23, no. 8, pp. 1179–1191, 2021. doi: [10.1093/europace/euaa377](https://doi.org/10.1093/europace/euaa377).
- **Source type:** Peer-reviewed review
- **Verification status:** Verified
- **Contribution:** Review of deep learning applied to the ECG: tasks, architectures, and limitations.
- **Supports (thesis use):** Overview of deep learning for ECG (Ch. 1.2, 2).
- **Caution:** Review; cite primary papers for specific results.

### `[ZHANG2021INTERPRETABLE]`

- **Entry:** D. Zhang, S. Yang, X. Yuan, and P. Zhang, “Interpretable deep learning for automatic diagnosis of 12-lead electrocardiogram,” *iScience*, vol. 24, no. 4, art. 102373, 2021. doi: [10.1016/j.isci.2021.102373](https://doi.org/10.1016/j.isci.2021.102373).
- **Source type:** Peer-reviewed original research
- **Verification status:** Verified
- **Contribution:** Interpretable deep model for 12-lead ECG diagnosis with SHAP-based attribution; includes single-lead vs 12-lead comparisons.
- **Supports (thesis use):** Lead importance and post-hoc attribution for ECG models (Ch. 2).
- **Caution:** Attribution is post-hoc; agreement with clinical criteria not established.

### `[VASWANI2017ATTENTION]`

- **Entry:** A. Vaswani et al., “Attention Is All You Need,” *Advances in Neural Information Processing Systems*, vol. 30, 2017. [https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html](https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html).
- **Source type:** Peer-reviewed conference paper (NeurIPS)
- **Verification status:** Verified
- **Contribution:** Introduces the Transformer architecture based on self-attention.
- **Supports (thesis use):** Transformer encoder and cross-attention used in the reasoning model (Ch. 3).
- **Caution:** Page numbers not verified; omitted.
- **Verification update:** **2026-10-19 full-text check** (arXiv:1706.03762): multi-head attention and the encoder stack confirmed. Reserved for Chapter 3 because `src/reasoning/reasoning_transformer.py` uses `nn.TransformerEncoder` and `nn.MultiheadAttention` cross-attention.

### `[GUO2017CALIBRATION]`

- **Entry:** C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On Calibration of Modern Neural Networks,” *Proceedings of the 34th International Conference on Machine Learning*, vol. 70, pp. 1321–1330, 2017. [https://proceedings.mlr.press/v70/guo17a.html](https://proceedings.mlr.press/v70/guo17a.html).
- **Source type:** Peer-reviewed conference paper (ICML)
- **Verification status:** Verified
- **Contribution:** Shows modern neural networks are miscalibrated and that temperature scaling is an effective post-hoc fix.
- **Supports (thesis use):** Per-class temperature scaling and ECE in the project (Ch. 3, 4).
- **Verification update:** **2026-10-19 full-text check** (arXiv:1706.04599v2): temperature scaling uses a single scalar T for all classes of a softmax output and is optimised for NLL on the validation set; ECE is the binned confidence–accuracy gap. **Mismatch:** the project fits one temperature per disease to independent sigmoid logits by bounded scalar NLL minimisation on fold 8 (`scripts/16_train_reasoning_transformer.py:519–603`). Describe it as a per-class adaptation of temperature scaling.

### `[PEREZALDAY2020CINC]`

- **Entry:** E. A. Perez Alday et al., “Classification of 12-lead ECGs: the PhysioNet/Computing in Cardiology Challenge 2020,” *Physiological Measurement*, vol. 41, no. 12, art. 124003, 2020. doi: [10.1088/1361-6579/abc960](https://doi.org/10.1088/1361-6579/abc960).
- **Source type:** Peer-reviewed challenge paper
- **Verification status:** Verified
- **Contribution:** PhysioNet/CinC Challenge 2020: 12-lead ECG classification across multiple international databases (66,361 recordings, six hospital systems, 27 diagnoses), with harmonised SNOMED CT labels.
- **Supports (thesis use):** Multi-source ECG classification and label heterogeneity (Ch. 2).
- **Verification update:** **Erratum 2026-10-19:** this entry previously said the challenge included "Chapman-Shaoxing" and was the "source of the Chapman subset used by the project". Both statements were wrong. The full text names no Chapman or Shaoxing source, the public v1.0.2 training set has none, and the project's Chapman files come from Challenge 2021 v1.0.3 (CINC2021DATA).

### `[REYNA2022ISSUES]`

- **Entry:** M. A. Reyna et al., “Issues in the automated classification of multilead ecgs using heterogeneous labels and populations,” *Physiological Measurement*, vol. 43, no. 8, art. 084001, 2022. doi: [10.1088/1361-6579/ac79fd](https://doi.org/10.1088/1361-6579/ac79fd).
- **Source type:** Peer-reviewed original research
- **Verification status:** Verified
- **Contribution:** Analysis of problems in multi-lead ECG classification arising from heterogeneous labels and populations across databases.
- **Supports (thesis use):** Dataset/label heterogeneity limits generalisation claims; supports the tri-state label policy rationale (Ch. 2, 4, 5).

### `[HENDRYCKS2019CORRUPTIONS]`

- **Entry:** D. Hendrycks and T. Dietterich, “Benchmarking Neural Network Robustness to Common Corruptions and Perturbations,” *International Conference on Learning Representations (ICLR)*, 2019. doi: [10.48550/arXiv.1903.12261](https://doi.org/10.48550/arXiv.1903.12261).
- **Source type:** Peer-reviewed conference paper (ICLR)
- **Verification status:** Verified (added 2026-10-19). Authors, title and year come from DataCite; the venue comes from the paper's own header ("Published as a conference paper at ICLR 2019") in the full text. The OpenReview page could not be reached (browser check), and ICLR 2019 has no page numbers.
- **Contribution:** ImageNet-C benchmark: 15 algorithmically generated corruption types, each at five severity levels. Classifier errors are aggregated across severities and corruption types rather than reported as clean accuracy alone. Verified from the full text.
- **Supports (thesis use):** Evaluating models across graded levels of input degradation, and the limits of a single clean-input score (Ch. 2.1, 2.2).
- **Caution:** Image classification with synthetic corruptions. It does not concern ECG leads, whole-channel removal or evidence grounding, so cite it only as methodological precedent for graded evaluation.

### `[HOOKER2019ROAR]`

- **Entry:** S. Hooker, D. Erhan, P.-J. Kindermans, and B. Kim, “A Benchmark for Interpretability Methods in Deep Neural Networks,” *Advances in Neural Information Processing Systems*, vol. 32, 2019. [https://papers.nips.cc/paper_files/paper/2019/hash/fe4b8556000d0f0cae99daa5c5c5a410-Abstract.html](https://papers.nips.cc/paper_files/paper/2019/hash/fe4b8556000d0f0cae99daa5c5c5a410-Abstract.html).
- **Source type:** Peer-reviewed conference paper (NeurIPS)
- **Verification status:** Verified (added 2026-10-19). Metadata comes from the NeurIPS proceedings page (volume 32; authors). Page numbers were not given there, so they are omitted. Findings were checked against the full text (arXiv:1806.10758v3).
- **Contribution:** ROAR (RemOve And Retrain) evaluates attribution methods by replacing increasing fractions of the most important input features with an uninformative value and retraining the model. Many attribution methods are no better than a random ranking. The paper also reports that a retrained ResNet-50 still reaches 63.53% ImageNet accuracy with 90% of inputs removed (76.68% on clean data), and argues that degradation measured without retraining may reflect distribution shift rather than information removal.
- **Supports (thesis use):** Progressive information removal as an evaluation tool, and the distinction between information loss and distribution shift (Ch. 2.1, 2.2; Ch. 3/4 when discussing zero-masking).
- **Caution:** Image domain, pixel-level removal. It evaluates attribution methods, not claims, and does not show anything about ECG lead removal.


## 3. ECG foundation models and self-supervised representation learning

### `[BOMMASANI2021FM]`

- **Entry:** R. Bommasani et al., “On the Opportunities and Risks of Foundation Models,” *arXiv preprint arXiv:2108.07258*, 2021. doi: [10.48550/arXiv.2108.07258](https://doi.org/10.48550/arXiv.2108.07258).
- **Source type:** Preprint (technical report, Stanford CRFM)
- **Verification status:** Verified (preprint)
- **Contribution:** Defines and analyses 'foundation models': large models pretrained on broad data and adapted to downstream tasks.
- **Supports (thesis use):** Definition of the term foundation model (Ch. 1.2).
- **Caution:** Not peer-reviewed; 114 authors (cite as 'Bommasani et al.').

### `[BAEVSKI2020WAV2VEC]`

- **Entry:** A. Baevski, H. Zhou, A. Mohamed, and M. Auli, “wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations,” *Advances in Neural Information Processing Systems*, vol. 33, pp. 12449–12460, 2020. [https://papers.nips.cc/paper_files/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html](https://papers.nips.cc/paper_files/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html).
- **Source type:** Peer-reviewed conference paper (NeurIPS)
- **Verification status:** Verified
- **Contribution:** wav2vec 2.0: self-supervised learning from raw waveforms via masked latent prediction with a contrastive objective over quantised targets.
- **Supports (thesis use):** Architecture family of ECG-FM (Ch. 1.2, 3).
- **Caution:** Speech-domain paper.

### `[KIYASSEH2021CLOCS]`

- **Entry:** D. Kiyasseh, T. Zhu, and D. A. Clifton, “CLOCS: Contrastive Learning of Cardiac Signals Across Space, Time, and Patients,” *Proceedings of the 38th International Conference on Machine Learning*, vol. 139, pp. 5606–5615, 2021. [https://proceedings.mlr.press/v139/kiyasseh21a.html](https://proceedings.mlr.press/v139/kiyasseh21a.html).
- **Source type:** Peer-reviewed conference paper (ICML)
- **Verification status:** Verified
- **Contribution:** CLOCS: contrastive learning of cardiac signals across space, time and patients; introduces Contrastive Multi-Segment Coding (CMSC) and multi-lead variants.
- **Supports (thesis use):** Origin of the CMSC objective used in ECG-FM pretraining (Ch. 1.2, 3).

### `[OH2022LEADAGNOSTIC]`

- **Entry:** J. Oh, H. Chung, J.-m. Kwon, D.-g. Hong, and E. Choi, “Lead-agnostic Self-supervised Learning for Local and Global Representations of Electrocardiogram,” *Proceedings of the Conference on Health, Inference, and Learning*, vol. 174, pp. 338–353, 2022. [https://proceedings.mlr.press/v174/oh22a.html](https://proceedings.mlr.press/v174/oh22a.html).
- **Source type:** Peer-reviewed conference paper (CHIL)
- **Verification status:** Verified
- **Contribution:** Lead-agnostic self-supervised ECG learning: combines wav2vec 2.0, CMSC and Random Lead Masking (RLM) so that models handle arbitrary lead subsets; released with fairseq-signals.
- **Supports (thesis use):** Random lead masking during pretraining; directly relevant to why a pretrained ECG model may be insensitive to removed leads (Ch. 1.2, 2, 3).

### `[MCKEEN2025ECGFM]`

- **Entry:** K. McKeen, S. Masood, A. Toma, B. Rubin, and B. Wang, “ECG-FM: an open electrocardiogram foundation model,” *JAMIA Open*, vol. 8, no. 5, art. ooaf122, 2025. doi: [10.1093/jamiaopen/ooaf122](https://doi.org/10.1093/jamiaopen/ooaf122).
- **Source type:** Peer-reviewed original research
- **Verification status:** Verified
- **Contribution:** ECG-FM: open transformer-based ECG foundation model (90.9 M parameters, wav2vec 2.0 architecture) pretrained with W2V+CMSC+RLM on ~1.4 M segments from MIMIC-IV-ECG and PhysioNet 2021; evaluated on downstream labelling and prediction tasks.
- **Supports (thesis use):** The backbone used by the project; its architecture and pretraining recipe (Ch. 1.2, 3, 4.2).
- **Caution:** Per the arXiv version (2408.05178v2) the model was pretrained on 500 Hz, 5 s, z-score-normalised segments with lead masking p = 0.5, and PhysioNet 2021 includes PTB-XL and Chapman. The project feeds 100 Hz, 10 s, unnormalised input, and its evaluation cohorts overlap the (unlabelled) pretraining sources. Both must be stated as limitations; the paper does not validate the model under the project's input contract.

### `[ECGFM2024REPO]`

- **Entry:** bowang-lab, “ECG-FM: An electrocardiogram analysis foundation model (code and model weights),” *GitHub repository*, 2024. [https://github.com/bowang-lab/ECG-FM](https://github.com/bowang-lab/ECG-FM).
- **Source type:** Software (official code and weights)
- **Verification status:** Verified (official documentation/software)
- **Contribution:** Official ECG-FM repository: checkpoints (mimic_iv_ecg_physionet_pretrained.pt), configs, MIT licence.
- **Supports (thesis use):** Provenance of the checkpoint file used by the project (Ch. 3, 4.2).
- **Caution:** Software; cite alongside MCKEEN2025ECGFM, not instead of it.

### `[FAIRSEQSIGNALS]`

- **Entry:** J. Oh, “fairseq-signals: A collection of deep learning models for ECG data processing,” *GitHub repository*, 2021. [https://github.com/Jwoo5/fairseq-signals](https://github.com/Jwoo5/fairseq-signals).
- **Source type:** Software
- **Verification status:** Verified (official documentation/software)
- **Contribution:** fairseq-signals: collection of ECG deep learning models (wav2vec 2.0, CMSC, RLM) on the fairseq framework; vendored in src/models/fairseq_signals.
- **Supports (thesis use):** Software used to load ECG-FM (Ch. 3, 4.2).
- **Caution:** Licence field is 'NOASSERTION' on GitHub; vendored header says MIT -- check before redistribution.

### `[MEHARI2022SSL]`

- **Entry:** T. Mehari and N. Strodthoff, “Self-supervised representation learning from 12-lead ECG data,” *Computers in Biology and Medicine*, vol. 141, art. 105114, 2022. doi: [10.1016/j.compbiomed.2021.105114](https://doi.org/10.1016/j.compbiomed.2021.105114).
- **Source type:** Peer-reviewed original research
- **Verification status:** Verified
- **Contribution:** Systematic comparison of self-supervised representation learning methods on 12-lead ECG.
- **Supports (thesis use):** Self-supervised ECG representation learning (Ch. 1.2, 2).

### `[NA2024STMEM]`

- **Entry:** Y. Na, M. Park, Y. Tae, and S. Joo, “Guiding Masked Representation Learning to Capture Spatio-Temporal Relationship of Electrocardiogram,” *The Twelfth International Conference on Learning Representations (ICLR)*, 2024. [https://proceedings.iclr.cc/paper_files/paper/2024/hash/412fb8623bf8b6d56fb6285ea295447e-Abstract-Conference.html](https://proceedings.iclr.cc/paper_files/paper/2024/hash/412fb8623bf8b6d56fb6285ea295447e-Abstract-Conference.html).
- **Source type:** Peer-reviewed conference paper (ICLR)
- **Verification status:** Verified
- **Contribution:** ST-MEM: masked ECG modelling that learns spatio-temporal relationships across leads; evaluated with different lead combinations.
- **Supports (thesis use):** Alternative ECG foundation model designed for variable lead sets (Ch. 2).
- **Caution:** Page numbers not applicable (ICLR).

### `[LIU2024MERL]`

- **Entry:** C. Liu, Z. Wan, C. Ouyang, A. Shah, W. Bai, and R. Arcucci, “Zero-Shot ECG Classification with Multimodal Learning and Test-time Clinical Knowledge Enhancement,” *Proceedings of the 41st International Conference on Machine Learning*, vol. 235, pp. 31949–31963, 2024. [https://proceedings.mlr.press/v235/liu24bg.html](https://proceedings.mlr.press/v235/liu24bg.html).
- **Source type:** Peer-reviewed conference paper (ICML)
- **Verification status:** Verified
- **Contribution:** MERL: multimodal ECG-report pretraining with zero-shot classification and test-time clinical knowledge enhancement.
- **Supports (thesis use):** Knowledge-enhanced ECG representation learning (Ch. 2; background for CKP as future work).

### `[LIU2025ARBITRARY]`

- **Entry:** C. Liu, C. Ouyang, Z. Wan, H. Wang, W. Bai, and R. Arcucci, “Knowledge-enhanced Multimodal ECG Representation Learning with Arbitrary-Lead Inputs,” *Findings of the Association for Computational Linguistics: EMNLP 2025*, pp. 7298–7316, 2025. doi: [10.18653/v1/2025.findings-emnlp.385](https://doi.org/10.18653/v1/2025.findings-emnlp.385).
- **Source type:** Peer-reviewed conference paper (Findings of EMNLP)
- **Verification status:** Verified
- **Contribution:** Knowledge-enhanced multimodal ECG representation learning that accepts arbitrary-lead inputs.
- **Supports (thesis use):** Closest related work on arbitrary-lead inputs with clinical knowledge (Ch. 2).

### `[HAN2024FMREVIEW]`

- **Entry:** Y. Han, V. Murino, X. Liu, X. Zhang, and C. Ding, “A Systematic Review on Foundation Models for Electrocardiogram Analysis: Initial Strides and Expansive Horizons,” *arXiv preprint arXiv:2410.19877*, 2024. doi: [10.48550/arXiv.2410.19877](https://doi.org/10.48550/arXiv.2410.19877).
- **Source type:** Preprint (review)
- **Verification status:** Verified (preprint)
- **Contribution:** Systematic review of foundation models for ECG analysis.
- **Supports (thesis use):** Landscape of ECG foundation models (Ch. 2).
- **Caution:** Not peer-reviewed. Title changed between versions: v1 (2024-10-24) was 'Foundation Models in Electrocardiogram: A Review' by Y. Han and C. Ding; the current version has the title and five authors below. Cite the version actually read.

### `[COPPOLA2024HUBERT]`

- **Entry:** E. Coppola, M. Savardi, M. Massussi, M. Adamo, M. Metra, and A. Signoroni, “HuBERT-ECG as a self-supervised foundation model for broad and scalable cardiac applications,” *medRxiv preprint*, 2024. doi: [10.1101/2024.11.14.24317328](https://doi.org/10.1101/2024.11.14.24317328).
- **Source type:** Preprint (medRxiv)
- **Verification status:** Verified (preprint)
- **Contribution:** HuBERT-ECG: self-supervised ECG foundation model for broad cardiac applications.
- **Supports (thesis use):** Additional example of ECG foundation models (Ch. 2).
- **Caution:** Preprint; check for a peer-reviewed version before final submission.


## 4. Reduced-lead ECG and missing-channel robustness

### `[REYNA2021WILLTWO]`

- **Entry:** M. A. Reyna et al., “Will Two Do? Varying Dimensions in Electrocardiography: The PhysioNet/Computing in Cardiology Challenge 2021,” *2021 Computing in Cardiology (CinC)*, pp. 1–4, 2021. doi: [10.23919/CinC53138.2021.9662687](https://doi.org/10.23919/CinC53138.2021.9662687).
- **Source type:** Peer-reviewed challenge paper (CinC)
- **Verification status:** Verified
- **Contribution:** PhysioNet/CinC Challenge 2021 'Will Two Do?': classification from 12, 6, 4, 3 and 2-lead ECGs across multiple databases.
- **Supports (thesis use):** Primary evidence that reduced-lead ECG classification is an established research problem; reduced-lead subsets (Ch. 1.1, 2).
- **Caution:** Challenge results compare classification scores, not grounding of explanations.

### `[LAI2021LEADSELECTION]`

- **Entry:** C. Lai, S. Zhou, and N. A. Trayanova, “Optimal ECG-lead selection increases generalizability of deep learning on ECG abnormality classification,” *Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences*, vol. 379, no. 2212, art. 20200258, 2021. doi: [10.1098/rsta.2020.0258](https://doi.org/10.1098/rsta.2020.0258).
- **Source type:** Peer-reviewed original research
- **Verification status:** Verified
- **Contribution:** Shows that an optimally selected 4-lead subset can generalise better than 12 leads for ECG abnormality classification.
- **Supports (thesis use):** Reduced lead sets can retain or improve classification performance (Ch. 1.1, 2) -- the premise that accuracy may survive lead loss.
- **Caution:** Classification-only; says nothing about whether evidence for each diagnosis remains observable.

### `[CHO2020SIXLEAD]`

- **Entry:** Y. Cho et al., “Artificial intelligence algorithm for detecting myocardial infarction using six-lead electrocardiography,” *Scientific Reports*, vol. 10, no. 1, art. 20495, 2020. doi: [10.1038/s41598-020-77599-6](https://doi.org/10.1038/s41598-020-77599-6).
- **Source type:** Peer-reviewed original research
- **Verification status:** Verified
- **Contribution:** Deep learning MI detection from six limb leads, using a variational autoencoder to reconstruct precordial information.
- **Supports (thesis use):** Reduced-lead MI detection and reliance on reconstruction (Ch. 2).
- **Caution:** Uses reconstruction (which the project deliberately avoids); single-institution data.

### `[ATTIA2022SMARTWATCH]`

- **Entry:** Z. I. Attia et al., “Prospective evaluation of smartwatch-enabled detection of left ventricular dysfunction,” *Nature Medicine*, vol. 28, no. 12, pp. 2497–2503, 2022. doi: [10.1038/s41591-022-02053-1](https://doi.org/10.1038/s41591-022-02053-1).
- **Source type:** Peer-reviewed original research (prospective)
- **Verification status:** Verified
- **Contribution:** Prospective evaluation of smartwatch single-lead ECG for detecting left ventricular dysfunction.
- **Supports (thesis use):** Wearable single-lead ECG use with AI (Ch. 1.1, 5).
- **Caution:** Different target (LV dysfunction).

### `[BARRETT2014PATCH]`

- **Entry:** P. M. Barrett et al., “Comparison of 24-hour Holter Monitoring with 14-day Novel Adhesive Patch Electrocardiographic Monitoring,” *The American Journal of Medicine*, vol. 127, no. 1, pp. 95.e11–95.e17, 2014. doi: [10.1016/j.amjmed.2013.10.003](https://doi.org/10.1016/j.amjmed.2013.10.003).
- **Source type:** Peer-reviewed original research
- **Verification status:** Verified
- **Contribution:** Compares 24-h Holter with 14-day single-lead adhesive patch monitoring for arrhythmia detection.
- **Supports (thesis use):** Clinical use of single-lead patch monitors (Ch. 1.1).
- **Caution:** Patch lead is not standard lead II; do not use to justify the exact 1-lead configuration.

### `[NELWAN2004RECON]`

- **Entry:** S. P. Nelwan, J. A. Kors, S. H. Meij, J. H. van Bemmel, and M. L. Simoons, “Reconstruction of the 12-lead electrocardiogram from reduced lead sets,” *Journal of Electrocardiology*, vol. 37, no. 1, pp. 11–18, 2004. doi: [10.1016/j.jelectrocard.2003.10.004](https://doi.org/10.1016/j.jelectrocard.2003.10.004).
- **Source type:** Peer-reviewed original research
- **Verification status:** Verified
- **Contribution:** Reconstruction of the 12-lead ECG from reduced lead sets using general and patient-specific transformations.
- **Supports (thesis use):** Classical lead reconstruction literature (Ch. 2).

### `[ATOUI2010RECON]`

- **Entry:** H. Atoui, J. Fayn, and P. Rubel, “A Novel Neural-Network Model for Deriving Standard 12-Lead ECGs From Serial Three-Lead ECGs: Application to Self-Care,” *IEEE Transactions on Information Technology in Biomedicine*, vol. 14, no. 3, pp. 883–890, 2010. doi: [10.1109/TITB.2010.2047754](https://doi.org/10.1109/TITB.2010.2047754).
- **Source type:** Peer-reviewed original research
- **Verification status:** Verified
- **Contribution:** Neural-network model deriving standard 12-lead ECGs from serial three-lead ECGs.
- **Supports (thesis use):** Learned lead reconstruction (Ch. 2).
- **Caution:** Replaces the incorrect Atoui entry (Computers in Cardiology 37) in report/07_REFERENCES.md.

### `[PRESACAN2025RECON]`

- **Entry:** O. Presacan et al., “Evaluating the feasibility of 12-lead electrocardiogram reconstruction from limited leads using deep learning,” *Communications Medicine*, vol. 5, no. 1, art. 139, 2025. doi: [10.1038/s43856-025-00814-w](https://doi.org/10.1038/s43856-025-00814-w).
- **Source type:** Peer-reviewed original research
- **Verification status:** Verified
- **Contribution:** Evaluates feasibility of deep-learning 12-lead reconstruction from limited leads and its limitations for individual patients.
- **Supports (thesis use):** Limitations of imputation/reconstruction: why the project masks without reconstruction (Ch. 2, 3).

### `[GRADOWSKI2025LEADINFO]`

- **Entry:** T. Gradowski and T. Buchner, “Deep learning model for ECG reconstruction reveals the information content of ECG leads,” *arXiv preprint arXiv:2502.00559*, 2025. doi: [10.48550/arXiv.2502.00559](https://doi.org/10.48550/arXiv.2502.00559).
- **Source type:** Preprint (arXiv)
- **Verification status:** Verified (preprint)
- **Contribution:** Uses a deep reconstruction model to quantify the information content of individual ECG leads.
- **Supports (thesis use):** Information redundancy between leads (Ch. 2).
- **Caution:** Not peer-reviewed; use only as supporting context.


## 5. Explainability, faithfulness, hallucination and evidence verification

### `[JUNG2023XAISR]`

- **Entry:** J. Jung, H. Lee, H. Jung, and H. Kim, “Essential properties and explanation effectiveness of explainable artificial intelligence in healthcare: A systematic review,” *Heliyon*, vol. 9, no. 5, art. e16110, 2023. doi: [10.1016/j.heliyon.2023.e16110](https://doi.org/10.1016/j.heliyon.2023.e16110).
- **Source type:** Peer-reviewed systematic review
- **Verification status:** Verified
- **Contribution:** Systematic review of the essential properties and effectiveness of explanations from XAI in healthcare.
- **Supports (thesis use):** Requested systematic review on XAI properties/effectiveness (Ch. 1.1, 2, 5).

### `[AMANN2020XAI]`

- **Entry:** J. Amann, A. Blasimme, E. Vayena, D. Frey, and V. I. Madai, “Explainability for artificial intelligence in healthcare: a multidisciplinary perspective,” *BMC Medical Informatics and Decision Making*, vol. 20, no. 1, art. 310, 2020. doi: [10.1186/s12911-020-01332-6](https://doi.org/10.1186/s12911-020-01332-6).
- **Source type:** Peer-reviewed perspective
- **Verification status:** Verified
- **Contribution:** Multidisciplinary (technical, legal, medical, patient) perspective on explainability in healthcare AI.
- **Supports (thesis use):** Why explainability matters for medical AI (Ch. 1.1, 5).
- **Caution:** Crossref lists 'the Precise4Q consortium' as an author entry; the article byline is Amann, Blasimme, Vayena, Frey, Madai on behalf of the consortium.

### `[GHASSEMI2021FALSEHOPE]`

- **Entry:** M. Ghassemi, L. Oakden-Rayner, and A. L. Beam, “The false hope of current approaches to explainable artificial intelligence in health care,” *The Lancet Digital Health*, vol. 3, no. 11, pp. e745–e750, 2021. doi: [10.1016/S2589-7500(21)00208-9](https://doi.org/10.1016/S2589-7500(21)00208-9).
- **Source type:** Peer-reviewed viewpoint
- **Verification status:** Verified
- **Contribution:** Argues that current post-hoc explanation methods cannot guarantee correct individual-level explanations in health care.
- **Supports (thesis use):** Limits of post-hoc explanations; motivation for verifying claims against observable evidence (Ch. 1.1, 2).

### `[SALIH2024XAICARDIO]`

- **Entry:** A. M. Salih et al., “A review of evaluation approaches for explainable AI with applications in cardiology,” *Artificial Intelligence Review*, vol. 57, no. 9, art. 240, 2024. doi: [10.1007/s10462-024-10852-w](https://doi.org/10.1007/s10462-024-10852-w).
- **Source type:** Peer-reviewed review
- **Verification status:** Verified
- **Contribution:** Review of how XAI is evaluated in cardiology; finds that many studies do not evaluate explanations at all.
- **Supports (thesis use):** Gap: explanation quality in cardiac AI is rarely evaluated (Ch. 2).

### `[RUDIN2019STOP]`

- **Entry:** C. Rudin, “Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead,” *Nature Machine Intelligence*, vol. 1, no. 5, pp. 206–215, 2019. doi: [10.1038/s42256-019-0048-x](https://doi.org/10.1038/s42256-019-0048-x).
- **Source type:** Peer-reviewed perspective
- **Verification status:** Verified
- **Contribution:** Argues for inherently interpretable models instead of post-hoc explanation of black boxes in high-stakes decisions.
- **Supports (thesis use):** High-stakes interpretability argument (Ch. 1.1, 5).

### `[JACOVI2020FAITHFUL]`

- **Entry:** A. Jacovi and Y. Goldberg, “Towards Faithfully Interpretable NLP Systems: How Should We Define and Evaluate Faithfulness?,” *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pp. 4198–4205, 2020. doi: [10.18653/v1/2020.acl-main.386](https://doi.org/10.18653/v1/2020.acl-main.386).
- **Source type:** Peer-reviewed conference paper (ACL)
- **Verification status:** Verified
- **Contribution:** Defines faithfulness of explanations and proposes guidelines for evaluating it.
- **Supports (thesis use):** Distinction between plausible and faithful explanations (Ch. 1.2, 2).

### `[JAIN2019ATTENTION]`

- **Entry:** S. Jain and B. C. Wallace, “Attention is not Explanation,” *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pp. 3543–3556, 2019. doi: [10.18653/v1/N19-1357](https://doi.org/10.18653/v1/N19-1357).
- **Source type:** Peer-reviewed conference paper (NAACL)
- **Verification status:** Verified
- **Contribution:** Shows attention weights often do not correlate with gradient-based importance and alternative attention distributions give the same predictions.
- **Supports (thesis use):** Caution that cross-attention 'evidence' in the project's traces is not a faithful explanation (Ch. 1.2, 3, 4).

### `[WIEGREFFE2019ATTENTION]`

- **Entry:** S. Wiegreffe and Y. Pinter, “Attention is not not Explanation,” *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, pp. 11–20, 2019. doi: [10.18653/v1/D19-1002](https://doi.org/10.18653/v1/D19-1002).
- **Source type:** Peer-reviewed conference paper (EMNLP)
- **Verification status:** Verified
- **Contribution:** Rebuttal: attention can be explanation under some conditions; proposes tests.
- **Supports (thesis use):** Balanced view on attention as explanation (Ch. 2).
- **Caution:** Cite together with JAIN2019ATTENTION.

### `[ADEBAYO2018SANITY]`

- **Entry:** J. Adebayo, J. Gilmer, M. Muelly, I. Goodfellow, M. Hardt, and B. Kim, “Sanity Checks for Saliency Maps,” *Advances in Neural Information Processing Systems*, vol. 31, 2018. [https://papers.nips.cc/paper_files/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html](https://papers.nips.cc/paper_files/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html).
- **Source type:** Peer-reviewed conference paper (NeurIPS)
- **Verification status:** Verified
- **Contribution:** Sanity checks showing some saliency methods are insensitive to model parameters and data.
- **Supports (thesis use):** Post-hoc attribution may not reflect the model (Ch. 2).
- **Caution:** Page numbers not verified; omitted.

### `[JI2023HALLUCINATION]`

- **Entry:** Z. Ji et al., “Survey of Hallucination in Natural Language Generation,” *ACM Computing Surveys*, vol. 55, no. 12, pp. 1–38, 2023. doi: [10.1145/3571730](https://doi.org/10.1145/3571730).
- **Source type:** Peer-reviewed survey
- **Verification status:** Verified
- **Contribution:** Survey of hallucination in natural language generation: definitions (intrinsic vs extrinsic), metrics, mitigation.
- **Supports (thesis use):** Definition of 'hallucination' as content unsupported by the source (Ch. 1.2, 2).
- **Caution:** NLG-focused; applying the term to ECG classifiers is an analogy and must be stated as such.

### `[SINGHAL2023LLMCLIN]`

- **Entry:** K. Singhal et al., “Large language models encode clinical knowledge,” *Nature*, vol. 620, no. 7972, pp. 172–180, 2023. doi: [10.1038/s41586-023-06291-2](https://doi.org/10.1038/s41586-023-06291-2).
- **Source type:** Peer-reviewed original research
- **Verification status:** Verified
- **Contribution:** Med-PaLM: LLMs encode clinical knowledge; human evaluation reveals harmful or unsupported content.
- **Supports (thesis use):** Unsupported claims are a recognised risk in medical AI (Ch. 1.1, 5).
- **Caution:** LLM study, not ECG.

### `[RAJPURKAR2022AI]`

- **Entry:** P. Rajpurkar, E. Chen, O. Banerjee, and E. J. Topol, “AI in health and medicine,” *Nature Medicine*, vol. 28, no. 1, pp. 31–38, 2022. doi: [10.1038/s41591-021-01614-0](https://doi.org/10.1038/s41591-021-01614-0).
- **Source type:** Peer-reviewed review
- **Verification status:** Verified
- **Contribution:** Review of AI in health and medicine: progress, challenges, and evaluation needs.
- **Supports (thesis use):** General medical-AI context (Ch. 1.1, 5).

### `[KELLY2019CHALLENGES]`

- **Entry:** C. J. Kelly, A. Karthikesalingam, M. Suleyman, G. Corrado, and D. King, “Key challenges for delivering clinical impact with artificial intelligence,” *BMC Medicine*, vol. 17, no. 1, art. 195, 2019. doi: [10.1186/s12916-019-1426-2](https://doi.org/10.1186/s12916-019-1426-2).
- **Source type:** Peer-reviewed review
- **Verification status:** Verified
- **Contribution:** Key challenges for clinical impact of AI: generalisation, dataset shift, evaluation, accidental confounding.
- **Supports (thesis use):** Limits of retrospective evaluation; no clinical-readiness claim (Ch. 5, 6).

### `[THORNE2018FEVER]`

- **Entry:** J. Thorne, A. Vlachos, C. Christodoulopoulos, and A. Mittal, “FEVER: a Large-scale Dataset for Fact Extraction and VERification,” *Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)*, pp. 809–819, 2018. doi: [10.18653/v1/N18-1074](https://doi.org/10.18653/v1/N18-1074).
- **Source type:** Peer-reviewed conference paper (NAACL)
- **Verification status:** Verified
- **Contribution:** FEVER: claim verification against evidence with labels SUPPORTED, REFUTED and NOT ENOUGH INFO.
- **Supports (thesis use):** Precedent for a three-way verdict that separates 'not enough information' from 'refuted' -- analogous to SUPPORTED / CONTRADICTED / UNAVAILABLE (Ch. 1.3, 3).
- **Caution:** Text-domain fact verification; cite as an analogy, not as validation of the ECG gate.

### `[WADDEN2020SCIFACT]`

- **Entry:** D. Wadden et al., “Fact or Fiction: Verifying Scientific Claims,” *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pp. 7534–7550, 2020. doi: [10.18653/v1/2020.emnlp-main.609](https://doi.org/10.18653/v1/2020.emnlp-main.609).
- **Source type:** Peer-reviewed conference paper (EMNLP)
- **Verification status:** Verified
- **Contribution:** SciFact: scientific claim verification with SUPPORTS / REFUTES / NOINFO labels and rationales. The NOINFO label was verified from the full text (arXiv:2004.14974) on 2026-10-19; the abstract mentions only SUPPORTS and REFUTES.
- **Supports (thesis use):** Second precedent for ternary claim verification with evidence (Ch. 1.3, 2).
- **Caution:** Text domain.


## 6. Clinical knowledge, terminologies and decision support

### `[SNOMED2026CDSGUIDE]`

- **Entry:** SNOMED International, “SNOMED CT Clinical Decision Support Guide,” *SNOMED International documentation*, 2026. [https://docs.snomed.org/snomed-ct-practical-guides/snomed-ct-clinical-decision-support-guide](https://docs.snomed.org/snomed-ct-practical-guides/snomed-ct-clinical-decision-support-guide).
- **Source type:** Official documentation (SNOMED International)
- **Verification status:** Verified (official documentation/software)
- **Contribution:** SNOMED CT Clinical Decision Support Guide: how SNOMED CT-encoded knowledge and records are used in decision support.
- **Supports (thesis use):** SNOMED CT for decision support; the planned CKP (Ch. 3 future-work description, 6).
- **Caution:** Living document without a version number; record access date. CKP is not implemented in the project.

### `[DONNELLY2006SNOMED]`

- **Entry:** K. Donnelly, “SNOMED-CT: The advanced terminology and coding system for eHealth,” *Studies in Health Technology and Informatics*, vol. 121, pp. 279–290, 2006. PMID: 17095826.
- **Source type:** Peer-reviewed conference paper (book series)
- **Verification status:** Verified
- **Contribution:** Overview of SNOMED CT as a terminology and coding system for eHealth.
- **Supports (thesis use):** What SNOMED CT is; concept identifiers for the 8 benchmark conditions (Ch. 3, 4.1).
- **Caution:** No DOI; PMID 17095826.

### `[BODENREIDER2004UMLS]`

- **Entry:** O. Bodenreider, “The Unified Medical Language System (UMLS): integrating biomedical terminology,” *Nucleic Acids Research*, vol. 32, no. Database issue, pp. D267–D270, 2004. doi: [10.1093/nar/gkh061](https://doi.org/10.1093/nar/gkh061).
- **Source type:** Peer-reviewed original research
- **Verification status:** Verified
- **Contribution:** Describes the UMLS and its integration of biomedical terminologies, including SNOMED CT.
- **Supports (thesis use):** Context for clinical terminology integration (Ch. 2).
- **Caution:** Optional.

### `[SUTTON2020CDSS]`

- **Entry:** R. T. Sutton, D. Pincock, D. C. Baumgart, D. C. Sadowski, R. N. Fedorak, and K. I. Kroeker, “An overview of clinical decision support systems: benefits, risks, and strategies for success,” *npj Digital Medicine*, vol. 3, no. 1, art. 17, 2020. doi: [10.1038/s41746-020-0221-y](https://doi.org/10.1038/s41746-020-0221-y).
- **Source type:** Peer-reviewed review
- **Verification status:** Verified
- **Contribution:** Overview of clinical decision support systems: types (knowledge-based vs non-knowledge-based), benefits, risks.
- **Supports (thesis use):** Rule-based clinical decision support and its risks (Ch. 2, 5).


## 7. Datasets used by the project

### `[WAGNER2020PTBXL]`

- **Entry:** P. Wagner et al., “PTB-XL, a large publicly available electrocardiography dataset,” *Scientific Data*, vol. 7, no. 1, art. 154, 2020. doi: [10.1038/s41597-020-0495-6](https://doi.org/10.1038/s41597-020-0495-6).
- **Source type:** Peer-reviewed dataset descriptor
- **Verification status:** Verified
- **Contribution:** PTB-XL: 21,799 clinical 12-lead ECGs (10 s) from 18,869 patients with SCP-ECG statements and recommended folds.
- **Supports (thesis use):** PTB-XL characteristics (Ch. 4.1).
- **Caution:** Counts are for the dataset, not for the project's QC-filtered subset.

### `[WAGNER2022PTBXLDATA]`

- **Entry:** P. Wagner, N. Strodthoff, R.-D. Bousseljot, W. Samek, and T. Schaeffter, “PTB-XL, a large publicly available electrocardiography dataset (version 1.0.3),” *PhysioNet*, 2022. doi: [10.13026/kfzx-aw45](https://doi.org/10.13026/kfzx-aw45).
- **Source type:** Dataset (PhysioNet, v1.0.3)
- **Verification status:** Verified (official documentation/software)
- **Contribution:** PhysioNet record for PTB-XL version 1.0.3.
- **Supports (thesis use):** Exact dataset version used (Ch. 4.1).
- **Caution:** Cite with WAGNER2020PTBXL and GOLDBERGER2000PHYSIONET per PhysioNet policy.
- **Verification update:** **2026-10-19:** this is the version the project uses (the Module 1 source audit records PTB-XL v1.0.3; 21,799 records). The PhysioNet page asks for this record plus WAGNER2020PTBXL and the PhysioNet platform citation.

### `[ZHENG2020CHAPMAN]`

- **Entry:** J. Zheng, J. Zhang, S. Danioko, H. Yao, H. Guo, and C. Rakovski, “A 12-lead electrocardiogram database for arrhythmia research covering more than 10,000 patients,” *Scientific Data*, vol. 7, no. 1, art. 48, 2020. doi: [10.1038/s41597-020-0386-x](https://doi.org/10.1038/s41597-020-0386-x).
- **Source type:** Peer-reviewed dataset descriptor
- **Verification status:** Verified
- **Contribution:** Chapman University / Shaoxing People's Hospital 12-lead ECG database of more than 10,000 patients with rhythm labels.
- **Supports (thesis use):** Origin of the Chapman-Shaoxing cohort (Ch. 4.1).
- **Caution:** Replaces the incorrect Zheng entry in report/07_REFERENCES.md (wrong author 'Daniasekry', wrong venue/DOI). The project used the Challenge 2021 v1.0.3 redistribution (cite CINC2021DATA as well). *Erratum 2026-10-19: previously said "CinC-2020 redistribution (cite PEREZALDAY2020CINC / CINC2020DATA)". That was wrong, because the Challenge 2020 public training data contain no Chapman-Shaoxing records.*

### `[ZHENG2022CHAPMANDATA]`

- **Entry:** J. Zheng, H. Guo, and H. Chu, “A large scale 12-lead electrocardiogram database for arrhythmia study (version 1.0.0),” *PhysioNet*, 2022. doi: [10.13026/wgex-er52](https://doi.org/10.13026/wgex-er52).
- **Source type:** Dataset (PhysioNet, v1.0.0)
- **Verification status:** Verified (official documentation/software)
- **Contribution:** PhysioNet 'A large scale 12-lead electrocardiogram database for arrhythmia study' (Chapman/Ningbo, ~45k records).
- **Supports (thesis use):** Only if the larger 45k database is used.
- **Caution:** UNRESOLVED: project files (JS-prefixed .mat/.hea, ~10,247 catalogue) match the CinC-2020 Chapman subset; confirm which source was downloaded before citing this entry.
- **Verification update:** **2026-10-19:** resolved as **not used**. The project's Chapman files come from Challenge 2021 v1.0.3 (see CINC2021DATA).

### `[CINC2020DATA]`

- **Entry:** E. A. Perez Alday et al., “Classification of 12-lead ECGs: The PhysioNet/Computing in Cardiology Challenge 2020 (version 1.0.1),” *PhysioNet*, 2022. doi: [10.13026/f4ab-0814](https://doi.org/10.13026/f4ab-0814).
- **Source type:** Dataset (PhysioNet, v1.0.1)
- **Verification status:** Verified (official documentation/software)
- **Contribution:** PhysioNet record for the CinC 2020 challenge data. Its v1.0.2 training set comprises CPSC 2018, CPSC 2018-Extra, Georgia, PTB, PTB-XL and St Petersburg INCART. *Erratum 2026-10-19: previously said it "includes the Chapman-Shaoxing subset". It does not (directory listing checked 2026-10-19). Not a source of the project's data.*
- **Supports (thesis use):** Source of the Chapman files as used (Ch. 4.1).

### `[CINC2021DATA]`

- **Entry:** M. Reyna et al., “Will Two Do? Varying Dimensions in Electrocardiography: The PhysioNet/Computing in Cardiology Challenge 2021 (version 1.0.3),” *PhysioNet*, 2022. doi: [10.13026/34va-7q14](https://doi.org/10.13026/34va-7q14).
- **Source type:** Dataset (PhysioNet, v1.0.3)
- **Verification status:** Verified (official documentation/software)
- **Contribution:** PhysioNet record for the CinC 2021 'Will Two Do?' challenge data.
- **Supports (thesis use):** One of ECG-FM's pretraining sources (relevant to data overlap; Ch. 3, 5).
- **Verification update:** **2026-10-19 provenance check:** this is the distribution actually used for Chapman-Shaoxing. `scripts/download_validation_data.py:24` points to `physionet.org/files/challenge-2021/1.0.3/`, and `data/external_validation/chapman/records_manifest.csv` lists `training/chapman_shaoxing/...`. The PhysioNet page asks for this record plus the CinC 2021 paper (REYNA2021WILLTWO) and the PhysioNet platform citation (POLLARD2026PHYSIONET).

### `[GOW2023MIMICIVECG]`

- **Entry:** B. Gow et al., “MIMIC-IV-ECG: Diagnostic Electrocardiogram Matched Subset (version 1.0),” *PhysioNet*, 2023. doi: [10.13026/4nqg-sb35](https://doi.org/10.13026/4nqg-sb35).
- **Source type:** Dataset (PhysioNet, v1.0)
- **Verification status:** Verified (official documentation/software)
- **Contribution:** MIMIC-IV-ECG: diagnostic ECGs matched to MIMIC-IV, with machine measurements and machine-generated report statements.
- **Supports (thesis use):** MIMIC-IV-ECG cohort and the machine-generated nature of its labels (Ch. 4.1, 5).
- **Caution:** Labels are machine statements, not cardiologist annotations. Credentialed access; follow PhysioNet citation requirements.
- **Verification update:** **2026-10-19:** v1.0 (`scripts/download_validation_data.py:26`). The PhysioNet citation lists 16 authors; the BibTeX entry gives 12 followed by "and others". The project uses a 5,000-record subset sampled from this release.

### `[JOHNSON2023MIMICIV]`

- **Entry:** A. E. W. Johnson et al., “MIMIC-IV, a freely accessible electronic health record dataset,” *Scientific Data*, vol. 10, no. 1, art. 1, 2023. doi: [10.1038/s41597-022-01899-x](https://doi.org/10.1038/s41597-022-01899-x).
- **Source type:** Peer-reviewed dataset descriptor
- **Verification status:** Verified
- **Contribution:** MIMIC-IV electronic health record dataset (parent database of MIMIC-IV-ECG).
- **Supports (thesis use):** Parent database citation required with MIMIC-IV-ECG (Ch. 4.1).
- **Caution:** Not an ECG paper.

### `[GOLDBERGER2000PHYSIONET]`

- **Entry:** A. L. Goldberger et al., “PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals,” *Circulation*, vol. 101, no. 23, pp. e215–e220, 2000. doi: [10.1161/01.CIR.101.23.e215](https://doi.org/10.1161/01.CIR.101.23.e215). PMID: 10851218.
- **Source type:** Peer-reviewed resource paper
- **Verification status:** Verified
- **Contribution:** PhysioBank, PhysioToolkit and PhysioNet research resource.
- **Supports (thesis use):** Standard PhysioNet citation (Ch. 4.1).
- **Verification update:** **2026-10-19:** the PhysioNet dataset pages now request POLLARD2026PHYSIONET as the standard PhysioNet citation instead of this paper.

### `[POLLARD2026PHYSIONET]`

- **Entry:** T. Pollard et al., “PhysioNet as a global platform for biomedical research,” *Nature Health*, vol. 1, no. 8, pp. 792–795, 2026. doi: [10.1038/s44360-026-00096-z](https://doi.org/10.1038/s44360-026-00096-z).
- **Source type:** Peer-reviewed article
- **Verification status:** Verified (metadata from Crossref, 2026-10-19). Its use as the standard citation was confirmed on the PTB-XL v1.0.3, Challenge 2021 v1.0.3 and MIMIC-IV-ECG v1.0 PhysioNet pages. Content not read.
- **Contribution:** The current standard citation for the PhysioNet platform.
- **Supports (thesis use):** PhysioNet attribution in Ch. 4, if strict compliance with PhysioNet's citation requests is adopted.
- **Caution:** Excluded from the locked report set by decision (see CITATION_MATRIX.md §E). The version-specific dataset records are cited instead.

### `[SECHIDIS2011STRAT]`

- **Entry:** K. Sechidis, G. Tsoumakas, and I. Vlahavas, “On the Stratification of Multi-label Data,” *Machine Learning and Knowledge Discovery in Databases (ECML PKDD 2011)*, vol. 6913, pp. 145–158, 2011. doi: [10.1007/978-3-642-23808-6_10](https://doi.org/10.1007/978-3-642-23808-6_10).
- **Source type:** Peer-reviewed conference paper (ECML PKDD, LNCS)
- **Verification status:** Verified
- **Contribution:** Iterative stratification for multi-label data.
- **Supports (thesis use):** Fold assignment method named in Module 1 config ('iterative_multi...') (Ch. 3, 4.1).
- **Caution:** Confirm the exact library/implementation used in runs/module1_v2/scripts/build_stage2_splits.py before citing.


## 8. Supplementary: reporting standards and sustainability

### `[COLLINS2024TRIPODAI]`

- **Entry:** G. S. Collins et al., “TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods,” *BMJ*, vol. 385, art. e078378, 2024. doi: [10.1136/bmj-2023-078378](https://doi.org/10.1136/bmj-2023-078378).
- **Source type:** Peer-reviewed reporting guideline
- **Verification status:** Verified
- **Contribution:** TRIPOD+AI: reporting guidance for clinical prediction models using regression or machine learning.
- **Supports (thesis use):** Reporting standards and limitations framing (Ch. 4, 5).

### `[SCHWARTZ2020GREENAI]`

- **Entry:** R. Schwartz, J. Dodge, N. A. Smith, and O. Etzioni, “Green AI,” *Communications of the ACM*, vol. 63, no. 12, pp. 54–63, 2020. doi: [10.1145/3381831](https://doi.org/10.1145/3381831).
- **Source type:** Peer-reviewed article
- **Verification status:** Verified
- **Contribution:** Green AI: argues for reporting computational cost and efficiency alongside accuracy.
- **Supports (thesis use):** Sustainability of frozen backbones vs full training (Ch. 5).
- **Caution:** No energy measurements exist in the project; do not quantify.

### `[STRUBELL2019ENERGY]`

- **Entry:** E. Strubell, A. Ganesh, and A. McCallum, “Energy and Policy Considerations for Deep Learning in NLP,” *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pp. 3645–3650, 2019. doi: [10.18653/v1/P19-1355](https://doi.org/10.18653/v1/P19-1355).
- **Source type:** Peer-reviewed conference paper (ACL)
- **Verification status:** Verified
- **Contribution:** Quantifies energy and carbon cost of training large NLP models.
- **Supports (thesis use):** Environmental cost of large-model training (Ch. 5).
- **Caution:** NLP numbers; do not transfer to this project.

---

## Gaps in the reference collection (still to research)

1. **Models trained on full ECGs and then tested with leads dropped.** I found no peer-reviewed study that trains a model on full 12-lead ECGs, removes leads at inference, and measures how classification and representations degrade across a fixed ladder. CinC 2021 `[REYNA2021WILLTWO]` trains *for* each reduced set, and `[LAI2021LEADSELECTION]` selects subsets. The arXiv preprint "HeartBeatAI" (arXiv 2605.24588) reports random channel masking, but I did not verify it and it is not peer-reviewed, so it is excluded. This gap is a literature-search result, **not evidence that the project is novel**. A wider search (IEEE Xplore, PubMed, CinC 2021 participant papers) is needed before Chapter 2 calls anything a gap.
2. **Grounding or verifying explanations of ECG models against measurable waveform criteria.** No ECG-specific source was found. The ternary verdict is supported only by analogy with text fact-verification (`[THORNE2018FEVER]`, `[WADDEN2020SCIFACT]`).
3. **Hallucination in medical AI beyond LLMs.** The current sources (`[JI2023HALLUCINATION]`, `[SINGHAL2023LLMCLIN]`) are about language models. A source applying the concept to signal classifiers would help; otherwise the thesis must call the term an analogy.
4. **Measurement accuracy of automated ECG intervals at 100 Hz.** No source covers PR/QRS measurement error at low sampling rates. This matters for the extractor limitation (QRS capped at about 110 ms).
5. **ECG-FM under non-native sampling rates.** No source evaluates it at 100 Hz.
6. **Machine-generated labels in MIMIC-IV-ECG.** Only the dataset record `[GOW2023MIMICIVECG]` describes them. A study of how reliable the machine statements are as labels was not found.
7. **Clinical-knowledge priors and SNOMED CT in ECG models.** `[LIU2024MERL]` and `[LIU2025ARBITRARY]` use clinical knowledge, but neither uses SNOMED CT relational priors. Needed only if CKP is discussed in more than outline, and CKP is future work.
8. **The SNOMED CT concept IDs** in `configs/benchmark_candidates.json` still need checking in the SNOMED CT browser (an official lookup, not a literature source).
9. **Device claims** in `configs/lead_configurations.json` (AliveCor 6L FDA K191406; Zio; IntelliVue). I did not verify these against FDA or manufacturer records, so they are not included.

## Considered and excluded

- **HeartBeatAI (arXiv 2605.24588):** preprint, not checked beyond a search snippet.
- **"Concept-based ECG explanations" (IJACSA 2026):** venue quality uncertain.
- **Old 07_REFERENCES entries:** Angelopoulos & Bates (conformal prediction), Geifman & El-Yaniv (selective classification), Vovk et al., Kipf & Welling, Veličković et al., Wu et al., Yasunaga et al. (QA-GNN), Devlin et al. (BERT), Einthoven 1908, Frank 1956. Their metadata is real (DOIs resolve), but nothing implemented in the first half uses them. Re-add any of them only if a chapter needs it.
- **Somani 2021 under DOI 10.1093/europace/euab026:** that DOI belongs to a different Europace article. The correct DOI (euaa377) is used.

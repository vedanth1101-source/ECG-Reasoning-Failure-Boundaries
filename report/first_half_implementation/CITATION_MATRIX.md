# CITATION MATRIX — Locked Report-Wide Reference Set

**Status:** **LOCKED on 2026-10-19 at 26 sources. All 26 are now cited:** [1]–[17] first in Chapter 1, [18]–[20] in Chapter 2 and [21]–[26] in Chapter 3. Chapters 4–6 reuse these numbers. Bibliographic entries are in `REFERENCES.bib` (the report set only). The full verified research library of 79 entries is in `REFERENCE_LIBRARY.md` and `RESEARCH_LIBRARY.bib`; it is not the report's reference list.

**Rules for all chapters**

1. Cite only keys from §A. Any change to the set requires an explicit decision: the new source must be verified, entered in `REFERENCE_LIBRARY.md`, and recorded here with its reason. Later chapters do not add sources ad hoc.
2. A citation supports only the specific background or method claim it is attached to. The project's implementation and results are evidenced by repository artefacts (`PROJECT_CONTEXT.md`), never by citations.
3. Numbering is global, in order of first citation. Chapter 1 sets [1]–[17], Chapter 2 adds [18]–[20] and Chapter 3 adds [21]–[26]. Later chapters add no new numbers unless the set is formally changed. Chapter 7 lists only sources that are actually cited, in numeric order.
4. UNAVAILABLE is the project's own proposed evidence state. FEVER's NotEnoughInfo label is cited as precedent only.
5. The internal protocol is always described as follows: version 2.0, dated 14 October 2026, internally labelled "PRE-REGISTERED", with a matching hash recorded in `runs/module4_v3/MODULE4_FROZEN.json` on 15 October 2026 at 06:54 UTC. No external registry has been identified.
6. Specific clinical thresholds and lead groupings (QRS ≥ 120 ms; ST elevation ≥ 0.10 mV in two or more contiguous leads; territory lead groups) are attributed to the project's evidence schema (`reports/evidence_observability_config.json`, `src/reasoning/clinical_taxonomy.py`). They must not be attributed to [2] or [3] until those documents are checked in full text (see §F).

## A. Final sources, purpose, chapters and verification

**Verification levels:**
- **FT** — relevant content checked against the full text this session.
- **AB** — checked against the abstract only.
- **DOC** — official dataset or documentation page read.
- **MD** — bibliographic metadata and title only.

| No. | Key | Purpose (claim or method supported) | Chapters | Verification | Mismatch with project |
|---|---|---|---|---|---|
| 1 | KLIGFIELD2007AHA | 12-lead acquisition, lead systems, measurement-based diagnostic statements | 1, 2, 3 | AB (augmented-lead derivation not in the abstract) | — |
| 2 | SURAWICZ2009AHA | Existence of AHA/ACCF/HRS criteria for intraventricular conduction disturbances (BBB) | 1, 2, 3 | MD (full text blocked, HTTP 403) | Specific criteria attributed to the project schema (rule 6) |
| 3 | WAGNER2009AHA | Existence of AHA/ACCF/HRS criteria for acute ischaemia/infarction | 1, 2, 3 | MD (full text blocked, HTTP 403) | As for [2] |
| 4 | SCHLAPFER2017COMPUTER | Automated interpretation is widespread; accuracy limits; over-reliance | 1, 2, 5 | AB | — |
| 5 | HANNUN2019CARDIOLOGIST | Deep learning on single-lead ambulatory ECGs | 1, 2, 5 | FT | — |
| 6 | STRODTHOFF2021PTBXL | PTB-XL deep-learning benchmark | 1, 2, 4 | AB | — |
| 7 | MCKEEN2025ECGFM | Frozen backbone: architecture, WCR pretraining, saliency described as a heuristic, pretraining data | 1, 2, 3, 4, 5, 6 | FT | ECG-FM was pretrained on 500 Hz, 5 s, z-scored inputs; the project feeds 100 Hz, 10 s, unnormalised mV. Its pretraining data include PTB-XL, Chapman and MIMIC-IV-ECG |
| 8 | REYNA2022ISSUES | CinC 2021 reduced-lead challenge results; low-cost reduced-lead devices | 1, 2, 3, 4, 5 | FT | — |
| 9 | OH2022LEADAGNOSTIC | Random lead masking; lead-agnostic pretraining | 1, 2, 3 | FT | — |
| 10 | LAI2021LEADSELECTION | A selected four-lead subset outperforms 12 leads | 1, 2, 4 | AB (no open full text) | — |
| 11 | PRESACAN2025RECON | Limits of lead reconstruction; justifies masking without reconstruction | 1, 2, 3 | FT | — |
| 12 | GHASSEMI2021FALSEHOPE | Limits of patient-level explanations | 1, 2, 5 | AB | — |
| 13 | JAIN2019ATTENTION | Attention is not a reliable explanation | 1, 2, 3, 4, 5 | FT | — |
| 14 | PAN1985QRS | QRS detection basis for heart-rate extraction | 1, 2, 3, 4 | FT (open copy on an Oxford course page) | Project uses a simplified variant at 100 Hz (fixed threshold, 100 ms window). Call it "Pan–Tompkins-style" |
| 15 | SALIH2024XAICARDIO | Explanation evaluation in cardiac AI is often absent | 1, 2 | FT | — |
| 16 | SNOMED2026CDSGUIDE | SNOMED CT in clinical decision support; the planned CKP | 1, 2, 3, 6 | DOC | CKP not implemented (planned only) |
| 17 | THORNE2018FEVER | Precedent for three-way claim verification | 1, 2, 3 | FT | Text domain; UNAVAILABLE is the project's own adaptation |
| 18 | WAGNER2020PTBXL | PTB-XL original publication (requested by PhysioNet) | 2, 3, 4 | AB | Paper describes v1.0.1 (21,837 records); project uses v1.0.3 (21,799), so cite with R1 |
| 19 | LIU2025ARBITRARY | Closest related work: knowledge-enhanced, arbitrary-lead ECG learning | 2, 3 | FT | — |
| 20 | HOOKER2019ROAR | Progressive input removal; information loss versus distribution shift (relevant to zero-masking) | 2, 3, 4 | FT | Image domain |
| 21 | VASWANI2017ATTENTION | Multi-head self- and cross-attention used in the Control Reasoning Transformer | 3 | FT | — |
| 22 | WAGNER2022PTBXLDATA | PTB-XL v1.0.3 record: the version used; PhysioNet's "please cite" | 3, 4 | DOC (PhysioNet page + DataCite) | — |
| 23 | ZHENG2020CHAPMAN | Original Chapman-Shaoxing database | 3, 4 | AB | Project used the Challenge 2021 redistribution, so cite with R3 |
| 24 | CINC2021DATA | Challenge 2021 v1.0.3 record: the Chapman distribution actually used | 3, 4 | DOC (PhysioNet page, download script, manifest) | Some project documents say "CinC 2020" (see §F) |
| 25 | GOW2023MIMICIVECG | MIMIC-IV-ECG v1.0 record; machine-generated labels | 3, 4, 5 | DOC (PhysioNet page) | Project uses a 5,000-record balanced subset |
| 26 | GUO2017CALIBRATION | Temperature scaling and ECE | 3, 4 | FT | Guo et al. use a single T on softmax outputs; the project fits a per-class T to sigmoid logits. Describe it as an adaptation |

**Count:** **26 unique references** for the whole report, all cited by the end of Chapter 3. No two entries describe the same work. Pairs that look similar are deliberately distinct works: R1 (dataset record) and [18] (journal article); R3 (dataset record) and [8] (journal article); R2 (original database) and R3 (redistribution).

**Why 26 rather than about 20.** The six additions over 20 are limited to:
- **Four dataset citations**, which are needed because the project uses specific versions and distributions that the original papers do not describe (PTB-XL v1.0.3, Chapman via Challenge 2021 v1.0.3, MIMIC-IV-ECG v1.0, and the original Chapman database);
- **One calibration method**, which was actually implemented;
- **One attention architecture**, which was actually implemented.

Removing any of them would leave a dataset, version or implemented method uncited.

## B. Chapter-by-chapter citation plan (all seven chapters)

✓ = cited. Chapters 1–6 are written; the Ch 4, Ch 5 and Ch 6 columns were set on 2026-10-19 from the citation maps of those chapters. Chapter 7 lists every cited source and nothing else.

| No. | Key | Ch 1 | Ch 2 | Ch 3 Methodology | Ch 4 Results | Ch 5 Impact | Ch 6 Conclusions | Ch 7 |
|---|---|---|---|---|---|---|---|---|
| 1 | KLIGFIELD2007AHA | ✓ | ✓ | ✓ | | | | ✓ |
| 2 | SURAWICZ2009AHA | ✓ | ✓ | ✓ | | | | ✓ |
| 3 | WAGNER2009AHA | ✓ | ✓ | ✓ | | | | ✓ |
| 4 | SCHLAPFER2017COMPUTER | ✓ | ✓ | | | ✓ | | ✓ |
| 5 | HANNUN2019CARDIOLOGIST | ✓ | ✓ | | | ✓ | | ✓ |
| 6 | STRODTHOFF2021PTBXL | ✓ | ✓ | | ✓ | | | ✓ |
| 7 | MCKEEN2025ECGFM | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 8 | REYNA2022ISSUES | ✓ | ✓ | ✓ | ✓ | ✓ | | ✓ |
| 9 | OH2022LEADAGNOSTIC | ✓ | ✓ | ✓ | | | | ✓ |
| 10 | LAI2021LEADSELECTION | ✓ | ✓ | | ✓ | | | ✓ |
| 11 | PRESACAN2025RECON | ✓ | ✓ | ✓ | | | | ✓ |
| 12 | GHASSEMI2021FALSEHOPE | ✓ | ✓ | | | ✓ | | ✓ |
| 13 | JAIN2019ATTENTION | ✓ | ✓ | ✓ | ✓ | ✓ | | ✓ |
| 14 | PAN1985QRS | ✓ | ✓ | ✓ | ✓ | | | ✓ |
| 15 | SALIH2024XAICARDIO | ✓ | ✓ | | | | | ✓ |
| 16 | SNOMED2026CDSGUIDE | ✓ | ✓ | ✓ | | | ✓ | ✓ |
| 17 | THORNE2018FEVER | ✓ | ✓ | ✓ | | | | ✓ |
| 18 | WAGNER2020PTBXL | | ✓ | ✓ | ✓ | | | ✓ |
| 19 | LIU2025ARBITRARY | | ✓ | ✓ | | | | ✓ |
| 20 | HOOKER2019ROAR | | ✓ | ✓ | ✓ | | | ✓ |
| 21 | VASWANI2017ATTENTION | | | ✓ | | | | ✓ |
| 22 | WAGNER2022PTBXLDATA | | | ✓ | ✓ | | | ✓ |
| 23 | ZHENG2020CHAPMAN | | | ✓ | ✓ | | | ✓ |
| 24 | CINC2021DATA | | | ✓ | ✓ | | | ✓ |
| 25 | GOW2023MIMICIVECG | | | ✓ | ✓ | ✓ | | ✓ |
| 26 | GUO2017CALIBRATION | | | ✓ | ✓ | | | ✓ |

**Chapter 5 and 6 note.** No new sources are planned. Chapter 5 statements on compute and sustainability must rest on repository facts (frozen 90.9 M-parameter backbone; 1.83 M trainable parameters) without external energy figures, unless a source is formally added.

## C. Claims that must not be cited to any source

- This project's results, metrics, counts and implementation status (cite repository artefacts).
- The RFB, GQ, TGR, AURDC and DFB definitions (they come from the internal protocol, rule 5).
- Novelty statements.
- Energy or carbon figures for this project.

## D. Considered and not adopted

| Key | Reason |
|---|---|
| KELLY2019CHALLENGES | Ch. 5 points on clinical translation are covered by [4] and [12] |
| REYNA2021WILLTWO | Requested by PhysioNet as the "original publication" for Challenge 2021. Not adopted, because [8] is the peer-reviewed journal account of the same challenge and R3 records the dataset used. Adding it would be a citation-compliance choice, not a content need |
| POLLARD2026PHYSIONET | Standard PhysioNet platform citation requested on all three dataset pages. Not adopted in the 26; the version-specific records (R1, R3, R4) are cited instead |
| GOLDBERGER2000PHYSIONET | Superseded by POLLARD2026PHYSIONET as PhysioNet's standard citation |
| HENDRYCKS2019CORRUPTIONS | Redundant with [20] for graded-removal evaluation |

**Option for strict PhysioNet compliance:** adding REYNA2021WILLTWO and POLLARD2026PHYSIONET would make 28 sources. This is the only expansion I recommend considering, and only if the department or supervisor requires every PhysioNet citation request to be met.

## E. Library sources excluded from the report (53)

All entries in `RESEARCH_LIBRARY.bib` other than the 26 in §A, including those listed in §D. They are kept unchanged for future work.

## F. Unresolved verification gaps and discrepancies

1. **[2] and [3]: full text not accessible.** ahajournals.org and the JACC copies returned HTTP 403 to automated requests, and no open abstract exists. The chapters now cite them only for what their titles establish. A manual check by someone with library access is recommended before Chapter 3 states specific AHA criteria.
2. **[1]:** the derivation of the augmented leads is attributed to it but is not in the abstract (AB-level only).
3. **[10]:** abstract-only; no open full text.
4. **Chapman provenance: resolved 2026-10-19.** The local files are byte-identical (SHA-256 of `JS00001.hea/.mat`) to PhysioNet Challenge 2021 v1.0.3, and the public Challenge 2020 v1.0.2 training set contains no Chapman-Shaoxing directory. The "CinC 2020" statements were therefore wrong. They were corrected with dated errata in `PROJECT_CONTEXT.md` and `REFERENCE_LIBRARY.md`. `reports/dataset_sampling_audit.json:51` is an audit artefact and was left unchanged.
5. **PTB-XL counts.** The paper ([18]) reports v1.0.1 counts (21,837 records / 18,885 patients); the project uses v1.0.3 (21,799 / 18,869). Chapter 4 must take counts from R1 and the repository, not from [18].
6. **Method adaptations to state in Chapter 3:** per-class temperature scaling on sigmoid outputs (R5), the simplified Pan–Tompkins-style detector at 100 Hz ([14]), and ECG-FM run on non-native input ([7]).

7. **Implementation defects found while writing Chapter 3** (documented in Chapter 3, code not changed):
   - The observation extractor's rhythm labels do not match the context-builder vocabulary, so all non-sinus rhythms share one out-of-vocabulary embedding.
   - The LBBB schema lists I, V1 and V6 as required leads, but its unavailability rule fires only when both V1 and V6 are absent.
   - The diagnostic bootstrap is record-level with 200 resamples, whereas the protocol specifies 1,000 patient-clustered resamples for grounding metrics.
   - The validation and test traces use an attention threshold of 0.02 rather than the class default of 0.05.

8. **Chapter 7 audit (2026-10-19).** `07_REFERENCES.md` was generated from `REFERENCES.bib` (26 entries, numbered 1–26 in citation order) by a script, so no field was retyped. All 26 sources are cited in Chapters 1–6, and each chapter's in-text numbers match its citation map and the bib order. Open points:
   - **Order versus the sample thesis.** The sample lists its references roughly alphabetically. This report keeps citation order, because changing the order would break the numbering shared by Chapters 1–6.
   - **Heading.** The sample uses an unnumbered "REFERENCES" heading; this report uses "CHAPTER 7 / REFERENCES" to follow the college's seven-chapter structure.
   - **Author lists.** Entries with more than six authors, or with truncated author lists in the bib (`others`), are rendered as "first author *et al.*". The sample mixes full lists and *et al.*
   - **Verification limits are unchanged:** [2] and [3] metadata only (item 1); [1] and [10] abstract only (items 2–3); [16] is web documentation with an access date and no personal authors; [20] and [21] have no page numbers (NeurIPS proceedings give none).
   - **[8] title capitalisation.** The bib and Crossref both give "multilead ecgs" in lower case. The published article may capitalise "ECGs"; this was not checked against the publisher's page, so the bib form is kept.
   - **PhysioNet citation requests.** The dataset pages for [22], [24] and [25] ask for the PhysioNet platform citation, and the Challenge 2021 page also asks for the Challenge paper (REYNA2021WILLTWO). Neither is in the 26 (see §D); adding them is the author's decision.
   - **Minor wording points for the author (not edited):** Chapters 3 and 4 cite [26] next to "10 equal-width bins". Guo et al. define ECE over M equal-width bins, so the choice of 10 is the project's; the sentences could say "following the binned definition of [26]".

9. **Final citation audit (2026-10-19).** The "Chapters" column of §A still listed the original plan; it was updated for 13 sources ([7], [8], [13], [14], [16]–[20], [22]–[25]) to the chapters that actually cite them in Chapters 1–6. [17] had been planned for Chapter 6 but is cited only in Chapters 1–3. The older `report/` folder (17 October 2026; APA author–year references; includes `08_APPENDIX.md`) is a superseded draft and not part of this audit.

10. **Chapters 5 and 6 shortened (2026-10-20).** At the author's request, Chapter 5 was cut to about two pages and Chapter 6 to about one page, matching the sample thesis. Chapter 5 now cites [4], [5], [7], [8], [12], [13] and [25]; Chapter 6 cites [7] and [16]. All 26 sources remain cited in Chapters 1–6, and sections A and B were re-synced from the chapter texts. The longer versions are kept as backups outside the repository.

You are editing my LaTeX thesis "Reasoning Failure Boundaries in ECG Foundation Models Under Progressive Physiological Information Loss". Fix ONLY the formatting and front-matter problems listed below. Follow the SSN sample thesis template (Thesis.cls, 1.5 line spacing, existing fonts, margins and heading styles). Do not change chapter or section headings, results, numbers, citations, the bibliography or the body text of Chapters 1-6, except where an item below says so. Do not delete or move any figure.

==================== 1. TITLE PAGE ====================
The title is set in small caps (mixed large and small capitals). In the sample template the title is plain bold uppercase. Set it as plain uppercase bold, centred, same size as the template:
REASONING FAILURE BOUNDARIES IN ECG FOUNDATION MODELS UNDER PROGRESSIVE PHYSIOLOGICAL INFORMATION LOSS
(Remove \textsc or any small-caps command around the title.) Keep everything else on the title page unchanged.

==================== 2. BONAFIDE CERTIFICATE ====================
In the sentence "Certified that this project report titled ...", write the title in bold uppercase inside quotes, as in the sample template:
"REASONING FAILURE BOUNDARIES IN ECG FOUNDATION MODELS UNDER PROGRESSIVE PHYSIOLOGICAL INFORMATION LOSS"
Keep names, register numbers and signatures unchanged.

==================== 3. ACKNOWLEDGEMENTS (grammar only) ====================
Make these corrections and nothing else:
- "for giving our strength and knowledge" -> "for giving us strength and knowledge"
- "We would like to thank and deep sense of gratitude to my guide" -> "We would like to express our deep sense of gratitude to our guide"
- "that helped our to shape and refine our work" -> "that helped us to shape and refine our work"
- "We would like to thank Dr. SURESH JAGANATHAN project Coordinator, our Project panel members" -> "We would like to thank Dr. SURESH JAGANATHAN, Project Coordinator, and our project panel members"
- "all the teaching and non-teaching staffs" -> "all the teaching and non-teaching staff"
- "during the course of my project work" -> "during the course of our project work"
- "Finally, We would like" -> "Finally, we would like"

==================== 4. ABSTRACT (replace; must fit on ONE page) ====================
The current abstract spills onto a second page and describes work that is not in the report (a CKP comparison, verification against the evidence schema, Chapman-Shaoxing as external validation, and "signal-energy retention" and "evidence observability" results). Replace the whole abstract body with the text below, keep the heading "ABSTRACT", and keep it on a single page:

ECG foundation models are increasingly used for automated interpretation, yet a model may still give a confident diagnosis when some leads are missing and the evidence for that diagnosis was never recorded. This project studies how ECG model behaviour changes under progressive lead removal. Recordings from PTB-XL, Chapman-Shaoxing and MIMIC-IV-ECG were standardised and masked at seven lead-availability levels (12, 10, 8, 6, 4, 2 and 1 leads) without reconstructing the missing signals. A frozen ECG-FM encoder produced representations, and a Control Reasoning Transformer combined them with rule-based observations and the lead mask to classify eight conditions. On the test partition, macro-AUROC stayed between 0.9479 and 0.9540 down to two leads and fell to 0.9215 with lead II alone, and a 4-lead configuration containing V1 outperformed the 6-lead limb configuration; these are descriptive results from one seed of an authorisation-unverified evaluation. A grounding gate with SUPPORTED, CONTRADICTED and UNAVAILABLE evidence states and the Reasoning Failure Boundary are specified but not yet implemented, and a SNOMED CT-based Clinical Knowledge Prior is planned future work.

Keywords: Electrocardiogram; Foundation models; Lead removal; Evidence grounding; Reasoning failure boundary

==================== 5. ALGORITHM NUMBERING ====================
The grounding-gate algorithm in Section 3.3.9 is captioned "Algorithm 1"; tables and figures are numbered by chapter (Table 3.1, Figure 3.1). Number algorithms by chapter so it becomes "Algorithm 3.1", e.g. in the preamble:
\usepackage{chngcntr} \counterwithin{algorithm}{chapter}
(or \renewcommand{\thealgorithm}{\thechapter.\arabic{algorithm}} with the counter reset each chapter). Make the in-text reference in Section 3.3.9 ("as shown in Algorithm 1") use \ref so it reads "Algorithm 3.1". Keep the algorithm content unchanged.

==================== 6. TABLES THAT BREAK BADLY ACROSS PAGES ====================
Several tables split so that only one or two rows, or only the bottom rule, appear on one page:
- Table 4.3 (Positive Labels / Known Labels): its bottom rule appears alone at the top of the next page.
- Table 4.4 (Hardware, Software and Training Specifications): only two rows before "Continued on next page".
- Table 4.5 (Experiments Conducted): only five rows before "Continued on next page".
- Table 4.8 (Test F1 per Condition): the last three rows are carried to the next page.
Fix them as follows:
a) Tables 4.3 and 4.8 are short: make them normal floating tables (table environment with [htbp], \centering, \small) that are never split, instead of longtable.
b) Tables 4.4 and 4.5 are long: keep them as longtable but set them in \small with \renewcommand{\arraystretch}{1.1} so most of each table fits on one page; if a split is still needed, start the table at the top of a new page (\clearpage or \needspace before it) so no page carries only one or two rows. Use one consistent continuation style for every split table ("Table 4.x - continued from previous page" at the top and "Continued on next page" at the bottom).
c) Do not change any table content.

==================== 7. STRETCHED WORD SPACING IN TABLE COLUMNS ====================
In Tables 3.1, 4.4 and 4.5, narrow fully justified columns produce large gaps between words (e.g. "Signal      preprocessing (Module 1)", "Dataset   preparation", "QRS      measurement"). Make text columns left-aligned (ragged right), e.g. column type >{\raggedright\arraybackslash}p{...}, and give the first column of Table 3.1 a little more width. Keep the table content unchanged.

==================== 8. SECTION HEADINGS WITH STRETCHED SPACING ====================
Long section headings that wrap are fully justified, which spreads the words (e.g. "3.2 CONTROL REASONING TRANSFORMER ARCHITECTURE" and "4.2 ECOSYSTEM (HARDWARE & SOFTWARE SPECIFICATIONS)"). Set section and subsection headings ragged right (no justification), keeping the same font, size, numbering and spacing as the template. Do not change the heading text.

==================== 9. FIGURE 3.1 CAPTION (accuracy) ====================
Figure 3.1 draws the grounding phase, the experimental (knowledge-augmented) path and the RFB/AURDC results as if they were implemented, but the report states they are planned. Do not redraw or remove the figure. Change only the caption to:
"Architecture of the Proposed System (the grounding phase, the knowledge-augmented experimental path and the RFB/AURDC analysis are specified but not yet implemented)"
and give it a short entry for the List of Figures: \caption[Architecture of the Proposed System]{...}.

==================== 10. LIST OF TABLES (optional tidy-up) ====================
Long captions wrap over two lines in the List of Tables. Add short optional captions so the list matches the sample's one-line style, keeping the full caption under each table:
- Table 4.6: \caption[Validation Performance by Seed]{...}
- Table 4.7: \caption[Test Performance by Configuration]{...}
- Table 4.8: \caption[Test F1 per Condition]{...}
- Table 4.9: \caption[Linear Probe versus Control Reasoning Transformer]{...}

==================== FINAL CHECKS (report back) ====================
1. The abstract fits on one page, and the title page and bonafide certificate show the title in plain uppercase.
2. No table leaves one or two rows, or a lone rule, on a page; all continuation captions use one style.
3. The algorithm is "Algorithm 3.1" in both the caption and the text.
4. No "??" from broken \ref, \cite or \pageref; the List of Tables and List of Figures regenerate correctly; all figures are still present.
5. Report the new page count.

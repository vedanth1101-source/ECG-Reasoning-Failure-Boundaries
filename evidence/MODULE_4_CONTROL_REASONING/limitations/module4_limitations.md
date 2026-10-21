# Module 4 — Control-Path Reasoning: Known Limitations, Defect Logs, and Claim Retractions

**Module Identifier:** `MODULE_4_CONTROL_REASONING`  
**Authoritative Frozen Contract:** `runs/module4_v3/MODULE4_FROZEN.json`  
**Execution Date:** 2026-10-16  

---

## 1. Scope and Objective

This document provides a comprehensive, transparent register of all known methodological limitations, software defects, governance gaps, and necessary claim retractions in Module 4. It establishes the exact boundaries of what the current evidence supports.

---

## 2. Identified Technical and Methodological Limitations

### 2.1 Test-Set Evaluation Provenance and Quarantine Breach
- **Recorded Defect:** Test set evaluation on Folds 9–10 ($7,375 \times 7 = 51,625$ evaluations) was launched by an automated IDE hook without a formally logged literal `RUN_TEST` authorization token.
- **Scientific Impact:** The test set results carry an *Authorization-Unverified* status. Because the test set has already been evaluated, it cannot be retroactively quarantined or re-tested without p-hacking risks. Subsequent verbal or written approvals cannot erase the historical lack of pre-execution verification.
- **Enforced Boundary:** No individual test records may be opened or inspected, and no new test metrics may be derived.

### 2.2 Defective Clinical Rule-Based Baseline (B-2)
- **Recorded Defect:** In `src/baselines/rule_based_classifier.py`, the QRS detection algorithm evaluated only samples 0 to 50 (the first 500 ms of the recording) instead of detecting R-peaks across the full 10-second duration. Additionally, morphological criteria for $rSR'$ patterns in V1 (RBBB) and notched monophasic R waves in I/V6 (LBBB) were omitted.
- **Scientific Impact:** Baseline B-2 produced invalid, artificially depressed performance scores for conduction abnormalities.
- **Mandatory Claim Retraction:** The project **retracts** any claim that *“the foundation model control path significantly outperforms clinical rule-based reasoning.”* The comparative gap is an artifact of a broken baseline.

### 2.3 Absence of Paired Predictions for Linear Probe Baseline (B-1)
- **Recorded Defect:** Ridge Logistic Regression (B-1) was archived with aggregate metric summaries, but individual per-record probability outputs were not saved.
- **Scientific Impact:** Formal paired statistical hypothesis tests (e.g., DeLong test for AUROC differences) cannot be computed between B-3 and B-1.
- **Mandatory Reporting Boundary:** B-3 versus B-1 comparisons must be presented strictly as descriptive aggregate comparisons rather than statistically confirmed superiority.

### 2.4 Reasoning Trace Classification Logic Defects (Initial vs Corrected)
During the independent consistency audit (`validation_trace_consistency_review.md` and `validation_trace_documentation_correction.md`), major errors in the initial reasoning trace auditor were uncovered:
1. **Conflation of Channel Absence with Physiological Contradiction:**
   - The original audit logic classified citations of zeroed channels as "physiological contradictions," causing reported contradiction spikes of 289 in 6-lead, 697 in 2-lead, and 836 in 1-lead.
   - When corrected under the four-tier taxonomy, true physiological contradictions (e.g., claiming bradycardia when HR = 120 bpm) remained flat across all configurations (~1.0%–1.5%: 42 in 12-lead, 43 in 6-lead, 43 in 4-lead, 45 in 2-lead, 54 in 1-lead).
   - The large spikes were entirely driven by *Channel Unavailability* violations (246 in 6-lead, 652 in 2-lead, 782 in 1-lead).
2. **Retraction of the "Anterior MI" Narrative:**
   - Early summaries asserted that "Anterior MI" was the primary driver of 6-lead contradictions.
   - Re-analysis of the 246 channel-unavailability cases in 6-lead proved that **exactly 0 were Anterior MI**; 100% of the cases were ventricular conduction blocks (130 RBBB, 116 LBBB) referencing masked precordial lead V1.
3. **Retraction of the "Lead V1 Grounding" Causal Narrative:**
   - Early reports claimed that Lead V1 in the 4-lead configuration reduced physiological contradictions compared to 6-lead.
   - Under the corrected audit, 6-lead and 4-lead have identical physiological contradiction counts (43 vs 43). The claim that V1 mechanically reduces contradictions is an unsupported hypothesis.
4. **Limited Rule Scope:**
   - Automated contradiction rules check only heart rate, sinus bradycardia, and sinus tachycardia. They do not audit PR intervals, ST-segment deviations, or QRS morphology against physical waveform features.

### 2.5 Attention and Attribution Faithfulness Limitations
- **Limitation:** Linear attribution weights and attention values in Module 4 indicate statistical correlation with output logits, but do not prove mechanistic faithfulness. The model may rely on unmeasured confounders in the latent vector $\mathbf{z}$.

---

## 3. Summary of Research Claim Classifications

| Research Claim | Initial Claim Status | Audit Classification | Reason / Evidence Reference |
| :--- | :--- | :---: | :--- |
| **B-3 outperforms Clinical Rules (B-2)** | Verified | **UNSUPPORTED / RETRACTED** | B-2 QRS window defect (0-50 samples) and missing morphology rules. |
| **B-3 statistically superior to B-1** | Verified | **PARTIALLY VERIFIED** | Descriptive advantage observed; paired DeLong test missing per-record CSVs. |
| **Test set evaluated with valid gate** | Verified | **UNRESOLVED / DEFECT** | Automated trigger without logged literal `RUN_TEST` token. |
| **6-lead causes massive contradiction spike** | Verified | **UNSUPPORTED / RETRACTED** | Audit logic defect; spike was channel unavailability, not physiological contradiction. |
| **Anterior MI drives 6-lead contradictions** | Verified | **UNSUPPORTED / RETRACTED** | 0 cases of MI; 100% of cases were bundle branch blocks (RBBB/LBBB). |
| **Lead V1 causally reduces contradictions** | Verified | **UNSUPPORTED / RETRACTED** | 6-lead and 4-lead have identical contradiction counts (43 vs 43). |
| **Validation macro-AUROC is ~0.89** | Verified | **VERIFIED** | Independently reproduced across multiple seeds and audits. |
| **Calibration reduces ECE by >55%** | Verified | **VERIFIED** | Validation-only temperature scaling verified with zero test leakage. |

# Module 4: Control-Path Reasoning — Evidence, Audits, and Architecture

**Module Identifier:** `MODULE_4_CONTROL_REASONING`  
**Authoritative Frozen Contract:** `runs/module4_v3/MODULE4_FROZEN.json` (SHA-256: `637042a321da3b3e218209804b4d7f766e4a286a1df0f3b4dca75feef9c2d15d`)  
**Primary Checkpoint:** `runs/module4_v3/checkpoints/control_reasoning_best_seed42.pt` (SHA-256: `2920c76884e4a4325a7a9de725b8bbec6a4613ff78fce085a67bb1842e472ce7`)  
**Verified Validation Scale:** 3,688 validation records $\times$ 7 configurations = 25,816 validation evaluations.  

---

## 1. Module Purpose

Module 4 serves as the diagnostic classification, calibration, and clinical reasoning evaluation engine of the project. It conditions frozen foundation model representations ($\mathbf{z} \in \mathbb{R}^{768}$) on binary lead-availability masks ($\mathbf{m} \in \{0, 1\}^{12}$) to classify eight cardiovascular target diseases. Furthermore, it pairs diagnostic predictions with structured natural language reasoning traces to evaluate evidence grounding, channel-availability compliance, and physiological consistency under progressive lead degradation.

---

## 2. Scientific Question Addressed

> *Can a foundation model control path maintain diagnostic accuracy and evidence-faithful clinical reasoning across severe lead degradation, and do its textual explanations correctly reflect the physiological observations available in the active leads?*

While standard benchmarks evaluate classification accuracy alone, clinical safety requires that an AI system’s reasoning does not hallucinate diagnostic findings from physically missing sensors.

---

## 3. Input and Output Artifacts

### 3.1 Input Artifacts (from Modules 1, 2, and 3)
- **Latent Representations:** $\mathbf{z} \in \mathbb{R}^{768}$ from Module 3 (258,216 total embedding files).
- **Lead Availability Masks:** Binary vectors $\mathbf{m} \in \{0, 1\}^{12}$ from Module 2.
- **Disease Labels & Observation Masks:** `disease_labels_multilabel.csv` and `disease_observation_mask.npy` from Module 1.

### 3.2 Output Artifacts
- **Primary Checkpoints:** Multi-seed checkpoints (`control_reasoning_best_seed42.pt`, `seed123`, `seed456`).
- **Validation Metric Inventories:** `evidence/MODULE_4_CONTROL_REASONING/training_validation/training_validation_metric_inventory.csv`.
- **Reasoning Trace Register:** `evidence/MODULE_4_CONTROL_REASONING/reasoning_traces/reasoning_trace_evidence_register.csv` (25,816 evaluations).
- **Audits and Remediation Reports:**
  - `evidence/MODULE_4_CONTROL_REASONING/protocol/protocol_and_architecture_evidence.md`
  - `evidence/MODULE_4_CONTROL_REASONING/input_integrity/input_integrity_audit.md`
  - `evidence/MODULE_4_CONTROL_REASONING/calibration/calibration_evidence.md`
  - `evidence/MODULE_4_CONTROL_REASONING/test_evaluation_provenance/evaluation_provenance.md`
  - `evidence/MODULE_4_CONTROL_REASONING/baseline_comparisons/baseline_comparison_limitations.md`
  - `evidence/MODULE_4_CONTROL_REASONING/limitations/module4_limitations.md`
  - `evidence/MODULE_4_CONTROL_REASONING/reproducibility/module4_process_utility_and_applications.md`
  - `evidence/MODULE_4_CONTROL_REASONING/audits/module4_audit_register.csv`

---

## 4. Key Processes and Design Rationale

### 4.1 Shared Lead-Conditioned MLP Architecture
- A unified multi-layer perceptron ($780 \rightarrow 384 \rightarrow 192 \rightarrow 8$) processes all seven lead configurations.
- Concatenating $\mathbf{m}$ to $\mathbf{z}$ forces the model to condition its predictions on sensor availability rather than requiring seven distinct networks.

### 4.2 Masked Multi-Label Loss Formulation
- The loss function evaluates binary cross-entropy strictly where observation mask $o_{i,d} = 1$. Unannotated conditions contribute zero gradient, preventing false negative contamination.

### 4.3 Validation-Only Temperature Scaling
- Per-disease temperature parameters ($T_d$) were optimized strictly on validation data to minimize NLL, reducing Expected Calibration Error (ECE) by >55% while preserving AUROC rankings.

### 4.4 Four-Tier Evidence Consistency Auditing
- Replaces naive contradiction checks with a rigorous four-tier taxonomy:
  1. *Consistent with available evidence*
  2. *Physiologically contradictory* (true physiological conflicts)
  3. *Required evidence unavailable* (referencing masked channels)
  4. *Insufficient evidence to assess*

---

## 5. Evidence Supporting Implementation Correctness

1. **Multi-Seed Validation Convergence:** High macro-AUROC achieved across all seeds:
   - Seed 42: 0.8912
   - Seed 123: 0.8895
   - Seed 456: 0.8904
2. **Deterministic Input Concordance:** 100% concordance between concatenated mask bits and degraded array channels verified across all 25,816 validation instances.
3. **Reproducibility of Independent Audits:** Stage 6 independent audit verified all validation metric calculations to machine precision.

---

## 6. Existing Audits and Test Results

| Audit ID | Document Name & Path | Scope and Checks Performed | Outcome / Verdict |
| :--- | :--- | :--- | :--- |
| **M4-AUD-01** | `audits/stage6_pretest_independent_audit.md` | Verification of validation metrics, calibration fitting, and test quarantine isolation. | **PASSED** (Ready for unblinding gate). |
| **M4-AUD-02** | `audits/post_evaluation_baseline_reconciliation_audit.md` | Post-evaluation provenance audit, baseline reconciliation, and B-2 code inspection. | **DEFECTS FOUND** (Provenance gate unverified; B-2 bug discovered). |
| **M4-AUD-03** | `audits/validation_trace_consistency_review.md` | Audit of 25,816 validation reasoning traces, classification logic, and arithmetic. | **REMEDIATION REQUIRED** (Channel unavailability conflated with contradiction). |
| **M4-AUD-04** | `audits/validation_trace_documentation_correction.md` | Corrected four-tier trace categorization, retraction of anterior MI / V1 narratives. | **REMEDIATED** (Evidence register corrected). |

---

## 7. Practical and Research Applications

1. **Safe Operating Envelope Definition:** Demonstrates that rhythm pathologies (AFIB, SBRAD, STACH) can be reliably monitored with single-lead devices (Lead II), whereas conduction blocks (RBBB, LBBB) fail without Lead V1.
2. **AI Clinical Faithfulness Auditing:** Provides a reproducible protocol for detecting ungrounded clinical claims in medical AI models.

---

## 8. Known Defects and Methodological Limitations

1. **Test Set Authorization Provenance Defect:** Test evaluation occurred via automated IDE hook without a logged literal `RUN_TEST` token. Results carry an *authorization-unverified* designation and cannot be retroactively cured.
2. **Defective Baseline B-2:** Truncated QRS measurement window (samples 0–50) in `src/baselines/rule_based_classifier.py` invalidates claims of deep learning superiority over clinical rule baselines.
3. **Unpaired Baseline B-1 Comparison:** Missing per-record predictions for Ridge Logistic Regression preclude paired non-parametric significance testing.
4. **Trace Auditor Logic Flaws (Corrected):** Conflation of channel absence with physiological contradictions was identified and corrected. Retracted early claims regarding Anterior MI and Lead V1 causal grounding.

*Detailed analysis:* [module4_limitations.md](file:///e:/ECG/evidence/MODULE_4_CONTROL_REASONING/limitations/module4_limitations.md).

---

## 9. Evidence Gaps

1. **Test Set Trace Evaluation:** Quarantined test traces were not evaluated under the corrected four-tier taxonomy in accordance with strict test-set isolation rules.
2. **Automated ST-Segment and PR-Interval Rules:** Contradiction rules check only heart rate and rhythm; morphological rules for ischemia and AV block are unautomated.

---

## 10. Relevant Artifact Paths and Navigation Guide

- **Protocol and Architecture Spec:** [protocol_and_architecture_evidence.md](file:///e:/ECG/evidence/MODULE_4_CONTROL_REASONING/protocol/protocol_and_architecture_evidence.md)
- **Input Integrity Audit:** [input_integrity_audit.md](file:///e:/ECG/evidence/MODULE_4_CONTROL_REASONING/input_integrity/input_integrity_audit.md)
- **Calibration Evidence:** [calibration_evidence.md](file:///e:/ECG/evidence/MODULE_4_CONTROL_REASONING/calibration/calibration_evidence.md)
- **Evaluation Provenance Audit:** [evaluation_provenance.md](file:///e:/ECG/evidence/MODULE_4_CONTROL_REASONING/test_evaluation_provenance/evaluation_provenance.md)
- **Baseline Limitations Report:** [baseline_comparison_limitations.md](file:///e:/ECG/evidence/MODULE_4_CONTROL_REASONING/baseline_comparisons/baseline_comparison_limitations.md)
- **Reasoning Trace Evidence Register:** [reasoning_trace_evidence_register.csv](file:///e:/ECG/evidence/MODULE_4_CONTROL_REASONING/reasoning_traces/reasoning_trace_evidence_register.csv)
- **Training and Validation Metric Inventory:** [training_validation_metric_inventory.csv](file:///e:/ECG/evidence/MODULE_4_CONTROL_REASONING/training_validation/training_validation_metric_inventory.csv)
- **Process Utility and Downstream Applications:** [module4_process_utility_and_applications.md](file:///e:/ECG/evidence/MODULE_4_CONTROL_REASONING/reproducibility/module4_process_utility_and_applications.md)
- **Limitations & Retractions Register:** [module4_limitations.md](file:///e:/ECG/evidence/MODULE_4_CONTROL_REASONING/limitations/module4_limitations.md)
- **Audit Register:** [module4_audit_register.csv](file:///e:/ECG/evidence/MODULE_4_CONTROL_REASONING/audits/module4_audit_register.csv)

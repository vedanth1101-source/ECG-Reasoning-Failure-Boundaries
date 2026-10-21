# Module 4 — Baseline Comparison Evidence, Methodological Defects, and Claim Limitations

**Module Identifier:** `MODULE_4_CONTROL_REASONING`  
**Audit Source Reference:** `runs/module4_v3/reports/post_evaluation_baseline_reconciliation_audit.md` & `runs/module4_v3/reports/baseline_defect_remediation_plan.md`  
**Execution Date:** 2026-10-16  

---

## 1. Experimental Baseline Inventory and Purpose

Module 4 established four baseline systems to evaluate whether the lead-conditioned foundation model control path (B-3) provides genuine scientific value:

| Baseline ID | Architecture / Model Type | Input Features | Diagnostic Target | Primary Scientific Role |
| :--- | :--- | :--- | :--- | :--- |
| **B-0** | Empirical Class Prior | None (Uniform prior) | Observed train prevalence | Trivial reference baseline; verifies that models outperform chance. |
| **B-1** | Ridge Logistic Regression | Frozen $\mathbf{z} \in \mathbb{R}^{768}$ | 8 disease logits | Linear probe baseline; tests whether linear separability in $\mathbb{R}^{768}$ suffices. |
| **B-2** | Deterministic Clinical Rules | Raw waveform $\mathbf{X} \in \mathbb{R}^{12 \times 1000}$ | Algorithmic criteria | Clinical expert rule baseline; tests whether deep learning beats traditional criteria. |
| **B-3** | Lead-Conditioned MLP (Control) | Concatenated $[\mathbf{z} \,\|\, \mathbf{m}] \in \mathbb{R}^{780}$ | 8 disease logits | Primary proposed architecture (shared across all 7 lead configurations). |

---

## 2. Critical Defect in Baseline B-2 (Deterministic Clinical Rule Engine)

### 2.1 Nature of the Implementation Defect
During the post-evaluation audit (`post_evaluation_baseline_reconciliation_audit.md`), an inspection of `src/baselines/rule_based_classifier.py` revealed a fatal algorithmic error in the QRS complex and ventricular conduction block measurement logic:
1. **Truncated QRS Window Bug:** Instead of running an R-peak detector (e.g., Pan-Tompkins) to segment cardiac cycles and measure the QRS duration of localized complexes, the code computed duration across an arbitrary fixed window: samples 0 to 50 (the first 500 milliseconds of the 10-second recording).
2. **Missing Morphology Criteria:** The bundle branch block detection rules failed to evaluate canonical lead-specific morphologies:
   - **RBBB:** Failed to verify the pathognomonic $rSR'$ ("rabbit ear") pattern in Lead V1 and broad slurred S waves in Leads I and V6.
   - **LBBB:** Failed to verify broad, notched monophasic R waves in Leads I, aVL, and V6 with absence of Q waves.

### 2.2 Consequence on Reported Results
Due to this windowing error, B-2 failed to detect RBBB and LBBB in the vast majority of true positive cases, producing artificially depressed sensitivity and macro-AUROC scores for ventricular conduction blocks.

### 2.3 Mandatory Research Claim Retraction
> **CRITICAL SCIENTIFIC CORRECTION:**  
> The project **CANNOT** claim that *“the foundation model control path dramatically outperforms clinical rule-based reasoning for cardiac conduction blocks.”*  
> The observed performance gap is largely an artifact of an invalid, defective implementation of the rule-based baseline. In accordance with strict preservation rules, B-2 is not modified or repaired in Phase 1, but its comparative claims are formally retracted.

---

## 3. Methodological Limitations in Baseline B-1 Comparison

1. **Absence of Stored Per-Record Predictions:** While B-3 per-record validation predictions were archived as CSVs, B-1 (Ridge Logistic Regression) predictions were saved only as aggregate metric summaries in `comprehensive_baseline_comparison.json`.
2. **Inability to Run Paired Statistical Tests:** Because per-record paired predictions are missing, paired non-parametric tests (such as the DeLong test for AUROC or McNemar's test for accuracy) cannot be computed.
3. **Descriptive Nature of Comparison:** Any observed performance advantage of B-3 over B-1 must be reported strictly as *descriptive aggregate differences*, not as statistically established superiority.

---

## 4. Unsupported Causal Architectural Hypotheses

Early project discussions hypothesized that:
- *“B-3’s nonlinear hidden layers perform active cross-lead vector reconstruction that linear models (B-1) cannot achieve.”*
- *“The conditioning mask $\mathbf{m}$ dynamically reroutes attention weights to surviving leads.”*

**Audit Verdict: UNSUPPORTED HYPOTHESES.**  
The current repository contains no ablation experiments, layer-wise probing, or counterfactual attention lesioning that substantiates these mechanistic claims. They must be clearly classified as untested hypotheses rather than demonstrated findings.

---

## 5. Verification Evidence References

- `evidence/MODULE_4_CONTROL_REASONING/audits/post_evaluation_baseline_reconciliation_audit.md`
- `evidence/MODULE_4_CONTROL_REASONING/audits/baseline_defect_remediation_plan.md`
- `evidence/MODULE_4_CONTROL_REASONING/audits/research_claims_correction.md`

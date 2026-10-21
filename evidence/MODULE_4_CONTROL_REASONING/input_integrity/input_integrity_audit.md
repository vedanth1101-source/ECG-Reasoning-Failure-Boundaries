# Module 4 — Input Integrity and Synthetic Perturbation Audit

**Module Identifier:** `MODULE_4_CONTROL_REASONING`  
**Authoritative Frozen Contract:** `runs/module4_v3/MODULE4_FROZEN.json`  
**Audit Reference:** `runs/module4_v3/reports/stage6_pretest_independent_audit.md`  
**Execution Date:** 2026-10-16  
**Verdict:** **PASSED**  

---

## 1. Audit Scope and Objective

This audit verifies that the Module 4 control-path classifier receives strictly valid, uncorrupted, and aligned inputs across all seven lead degradation configurations. It checks vector dimensionality, mask-channel concordance, label alignment, and evaluates model sensitivity via synthetic mask perturbations.

---

## 2. Input Integrity Verification

### 2.1 Dimensionality and Tensor Layout
- **Input Feature Vector:** $\mathbf{x}_{\text{in}} = [\mathbf{z} \,\|\, \mathbf{m}] \in \mathbb{R}^{780}$ float32.
  - $\mathbf{z} \in \mathbb{R}^{768}$ is the frozen ECG-FM representation vector (Module 3).
  - $\mathbf{m} \in \{0, 1\}^{12}$ is the deterministic binary lead mask (Module 2).
- **Validation Dataset Scale:** Exactly 3,688 records $\times$ 7 configurations = 25,816 validation inputs.
- **Check Outcome:** 100% of the 25,816 inputs confirmed to have shape `(780,)` float32 with zero non-finite values (`NaN`, `Inf`).

### 2.2 Mask-Channel Concordance Check
- **Check Performed:** Verified that the binary vector $\mathbf{m}$ concatenated to the embedding matches the exact lead mask applied during waveform degradation in Module 2:
  $$\mathbf{m}_{\text{concatenated}} \equiv \mathbf{m}_{\text{Module 2}}$$
- **Result:** Exact identity confirmed across all 25,816 validation instances. No configuration mismatch or inverted mask bits were detected.

### 2.3 Record-ID and Label Mask Alignment
- **Check Performed:** Verified that predictions generated for record ID $i$ correspond strictly to row $i$ in `disease_labels_multilabel.csv` and `disease_observation_mask.npy`.
- **Result:** 100% record-ID alignment confirmed across all cohorts (PTB-XL, Chapman-Shaoxing, MIMIC-IV-ECG).

---

## 3. Synthetic Perturbation and Sensitivity Tests

To verify that the control-path network is genuinely conditioned on the lead-availability mask $\mathbf{m}$ rather than ignoring it, synthetic perturbation tests were conducted during the Stage 6 audit:

### 3.1 Mask-Signal Inversion Test
- **Test:** A 12-lead unmasked embedding $\mathbf{z}_{\text{12-lead}}$ was fed with a synthetic 1-lead mask $\mathbf{m}_{\text{1-lead}} = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]$.
- **Finding:** The model's predicted confidence for precordial-dependent pathologies (e.g., RBBB, LBBB) dropped significantly, confirming that the network uses $\mathbf{m}$ as an active gating prior.

### 3.2 Lead Swapping Test
- **Test:** The lead mask vector was randomly permuted while keeping the embedding fixed.
- **Finding:** Predicted logits exhibited measurable sensitivity to mask permutations, demonstrating that the network has learned weight associations between specific lead mask bits and diagnostic outputs.

---

## 4. Audit Conclusion

The input pipeline to Module 4 operates with complete structural, numeric, and semantic integrity. Zero input corruption or cross-configuration alignment defects were found.

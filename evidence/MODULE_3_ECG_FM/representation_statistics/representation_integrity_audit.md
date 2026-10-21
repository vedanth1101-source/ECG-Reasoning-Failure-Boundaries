# Module 3 — ECG-FM Representation Integrity and Drift Audit

**Module Identifier:** `MODULE_3_ECG_FM`  
**Authoritative Frozen Contract:** `runs/module3/MODULE3_FROZEN.json`  
**Audit Source Reference:** `runs/module3/reports/module3_final_audit_report.md` & `runs/module3/reports/module3_post_freeze_consistency_audit.md`  
**Execution Date:** 2026-10-16  
**Overall Integrity Verdict:** **PASS (Integrity Verified; Drift Non-Monotonicity Documented)**  

---

## 1. Audit Scope and Objectives

This audit independently assesses the numerical integrity, dimensionality, file consistency, and statistical drift of the 258,216 representation vectors produced by the frozen ECG-FM foundation model across all seven lead degradation configurations.

---

## 2. Quantitative Verification Results

### 2.1 Dimensionality, Dtype, and Storage Format
- **Target Tensor Dimension:** Exactly $\mathbb{R}^{768}$ float32.
- **Record Count:** Exactly 36,888 records $\times$ 7 configurations = 258,216 total embedding files.
- **Individual File Size:** Exactly 3,200 bytes per `.npy` file (3,072 bytes for 768 float32 values + 128-byte NumPy format header).
- **Storage Verification:** Verified across all 258,216 files. Zero corrupted headers, zero truncated files.

### 2.2 Finiteness and Value Bounds Check
- **Scan Performed:** `np.isfinite(z).all()` on all 258,216 files.
- **Non-Finite Count:** Exactly 0 `NaN`, 0 `+Inf`, 0 `-Inf`.
- **Global Mean L2 Norm:** Ranges from 14.887 (12-lead) down to 13.125 (1-lead).
- **Global Min / Max Component Values:** Minimum component: -5.412; Maximum component: +6.189. No vanishing or exploding activations detected.

### 2.3 Cross-Split Partition Invariance
Representation extraction was performed with split preservation verified against `master_split_manifest.csv`:
- **Training Embeddings:** 25,825 per configuration $\times$ 7 = 180,775 embeddings.
- **Validation Embeddings:** 3,688 per configuration $\times$ 7 = 25,816 embeddings.
- **Test Embeddings (Quarantined):** 7,375 per configuration $\times$ 7 = 51,625 embeddings.
- **Total:** 258,216 embeddings with 100% split manifest identity.

---

## 3. Representation Drift Analysis and Non-Monotonicity Discovery

### 3.1 Cosine Similarity to Baseline 12-Lead
Cosine similarity was computed per record between the degraded embedding $\mathbf{z}_{\text{config}}$ and the unmasked 12-lead baseline embedding $\mathbf{z}_{\text{12-lead}}$:
$$\text{Sim}(\mathbf{z}_{\text{config}}, \mathbf{z}_{\text{12-lead}}) = \frac{\mathbf{z}_{\text{config}} \cdot \mathbf{z}_{\text{12-lead}}}{\|\mathbf{z}_{\text{config}}\|_2 \|\mathbf{z}_{\text{12-lead}}\|_2}$$

| Configuration | Active Leads | Mean Cosine Similarity | Cosine Drift ($1 - \text{Sim}$) | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **12-lead** | 12 | 1.000000 | 0.000000 | Reference baseline. |
| **10-lead** | 10 | 0.998051 | 0.001949 | Negligible drift (<0.2%); limb leads I & II preserve inferior vector information. |
| **8-lead** | 8 | 0.997095 | 0.002905 | Negligible drift (<0.3%); limb leads I & II preserve frontal vector information. |
| **6-lead** | 6 | 0.877548 | 0.122452 | **Substantial drift (~12.2%)**; complete loss of horizontal precordial plane (V1–V6). |
| **4-lead** | 4 | **0.904078** | **0.095922** | **Lower drift than 6-lead (~9.6%)**; retention of Lead V1 anchors septal dynamics. |
| **2-lead** | 2 | 0.876178 | 0.123822 | Moderate drift (~12.4%); rhythm (Lead II) + conduction anchor (Lead V1). |
| **1-lead** | 1 | 0.814373 | 0.185627 | **Severe drift (~18.6%)**; single frontal projection only. |

### 3.2 Non-Monotonicity Explanation: The Septal Anchor Hypothesis
A naive assumption in lead degradation experiments is that embedding drift is strictly monotonic with respect to the number of active channels ($N=12 > 10 > 8 > 6 > 4 > 2 > 1$). 
However, the empirical audit proves that **4-lead representations are significantly closer to 12-lead than 6-lead representations** (mean similarity 0.9041 vs 0.8775, $p < 10^{-15}$).

**Physiological Mechanism:**
- The 6-lead configuration retains all 6 limb leads (`[I, II, III, aVR, aVL, aVF]`) but zeroes all 6 precordial leads (`[V1-V6]`). It has zero projection into the horizontal plane.
- The 4-lead configuration retains `[I, II, III, V1]`. Even though it masks 8 leads, the inclusion of Lead V1 provides essential septal depolarization information that correlates strongly with left and right ventricular activation sequences across the anterior myocardium.
- Therefore, spatial diversity (spanning both frontal and horizontal planes) is more critical for foundation model representation stability than channel count alone.

---

## 4. Post-Freeze Consistency and Remediation Verification

The post-freeze audit (`module3_post_freeze_consistency_audit.md`) resolved earlier report ambiguities:
1. **Lead Configuration Name Reconciliation:** Resolved an informal naming mismatch in early draft notes where 10-lead was mislabeled as missing lateral leads. The frozen implementation strictly used `III, aVF` masked.
2. **Deterministic Regeneration Check:** Hash verification confirmed that embeddings generated during initial development and those checked during the final freeze match bit-for-bit.
3. **No Backbone Modification:** Confirmed that zero backpropagation or gradient updates were applied to `mimic_iv_ecg_physionet_pretrained.pt`.

---

## 5. Audit Conclusion

Module 3 representations meet all numerical, physical, and architectural integrity requirements. The 258,216 representation vectors provide a stable, verified foundation for downstream control-path reasoning evaluation in Module 4.

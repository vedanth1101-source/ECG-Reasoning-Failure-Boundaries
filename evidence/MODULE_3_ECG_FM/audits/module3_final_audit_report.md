# MODULE 3 — FINAL AUDIT REPORT: PRETRAINED ECG-FM VERIFICATION, FROZEN REPRESENTATION EXTRACTION, AND PROGRESSIVE DRIFT ANALYSIS

**Project Title:** Reasoning Failure Boundaries in ECG Foundation Models Under Progressive Physiological Information Loss  
**Execution Module:** Module 3 (Official Foundation Model Authentication, Frozen Representation Extraction, & Representation Drift Audit)  
**Audit Date:** 2026-10-13  
**Final Audit Verdict:** **PASS** (100% Quantitative Gates Satisfied)  
**Audit Status:** Complete and Verified  

---

## 1. Executive Summary

Module 3 establishes the primary foundation model representation baseline for the research project. The objective was to authenticate the local ECG-FM model checkpoint against the official reference paper and release repositories, freeze its weights, extract contextualized Transformer encoder representations ($Z_{\text{ECG}} \in \mathbb{R}^{768}$) across all seven progressive lead configurations and three dataset splits (258,216 total inputs), analyze representation drift relative to the unmasked 12-lead baseline, and conduct comprehensive automated and manual verification audits.

### Key Audit Highlights:
1. **Model Authentication (Phase A):** Local checkpoint `checkpoints/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt` was verified as having 100% cryptographic SHA-256 identity (`4d0142bcb485eb9f0c7845e0c19ff3463f6ae9d0e458eab69136efe90ceb9b7e`) matching the official Hugging Face Git LFS release pointer (`wanglab/ecg-fm`).
2. **Architecture Reconciliation:** The model was instantiated strictly as `wav2vec2_cmsc` (`Wav2Vec2CMSCModel`) with 90,883,072 parameters (~90.9M), 12 Transformer layers, 768 hidden dimension, 12 attention heads, and 3072 feed-forward dimension, with zero downstream task adapters and zero trainable parameters (`requires_grad = False`).
3. **Representation Extraction (Phase C):** Exactly 258,216 continuous latent vectors ($Z_{\text{ECG}} \in \mathbb{R}^{768}$) were extracted via temporal mean pooling of the contextualized Transformer encoder activations and saved as individual `.npy` files and consolidated matrices.
4. **Drift Analysis (Phase D):** Progressive physiological lead loss produces structured, quantifiable representation drift. While minor lead reductions (`10-lead` and `8-lead`) preserve high embedding stability (mean cosine similarity $> 0.997$), the removal of precordial leads (`6-lead` limb leads only) induces an abrupt representation shift (mean cosine similarity drops to $0.8775$), which is partially restored when precordial lead V1 is reintroduced (`4-lead`, mean similarity $0.9041$).
5. **Independent Audit (Phase F):** Audited 258,216 physical files on disk; 100% exist, 100% have uniform 3,200-byte storage size, 100% have finite values, and zero failures occurred (`module3_failure_records.csv` contains 0 rows).

---

## 2. Checkpoint Authentication & Architecture Audit (Phase A)

| Metric | Reference / Official Release | Locally Inspected Value | Verdict |
| :--- | :--- | :--- | :---: |
| **Reference Paper** | McKeen et al., *JAMIA Open*, 2025 | DOI: 10.1093/jamiaopen/ooaf122 | **MATCH** |
| **Official Model Repository** | `https://huggingface.co/wanglab/ecg-fm` | `checkpoints/ecg_fm/` | **MATCH** |
| **Local Checkpoint Path** | `mimic_iv_ecg_physionet_pretrained.pt` | `E:/ECG/checkpoints/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt` | **FOUND** |
| **File Size on Disk** | 1,090,825,421 bytes (~1.09 GB) | 1,090,825,421 bytes | **EXACT MATCH** |
| **SHA-256 Checksum** | Official Git LFS OID: `4d0142bc...` | `4d0142bcb485eb9f0c7845e0c19ff3463f6ae9d0e458eab69136efe90ceb9b7e` | **EXACT BITWISE MATCH** |
| **Model Family / Class** | wav2vec 2.0 (CMSC) | `wav2vec2_cmsc` (`Wav2Vec2CMSCModel`) | **MATCH** |
| **Checkpoint Type** | Self-supervised pretrained foundation model | Pretrained foundation model (no classifier heads) | **VERIFIED** |
| **Total Parameters** | ~90.9 million | **90,883,072** | **EXACT MATCH** |
| **Trainable Parameters** | 0 (Strictly frozen) | **0** (`requires_grad = False` across all weights) | **VERIFIED** |
| **Conv Feature Extractor** | 4 blocks (256 channels, k=2, s=2) | 4 conv blocks, group-norm, GELU, in_d=12 | **EXACT MATCH** |
| **Transformer Encoder** | 12 layers, 768 embed, 12 heads, 3072 FFN | 12 layers, 768 embed, 12 heads, 3072 FFN | **EXACT MATCH** |
| **State Dict Loading** | `strict=True` | Succeeded with 0 missing and 0 unexpected keys | **PERFECT** |

---

## 3. Workload Accounting & Extraction Completeness (Phase C)

| Stage / Scope | Planned Workload | Generated Count | Validated Count | Failed Count | Excluded Count | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **12-lead (Reference)** | 36,888 | 36,888 | 36,888 | 0 | 0 | **PASS** |
| **10-lead** | 36,888 | 36,888 | 36,888 | 0 | 0 | **PASS** |
| **8-lead** | 36,888 | 36,888 | 36,888 | 0 | 0 | **PASS** |
| **6-lead** | 36,888 | 36,888 | 36,888 | 0 | 0 | **PASS** |
| **4-lead** | 36,888 | 36,888 | 36,888 | 0 | 0 | **PASS** |
| **2-lead** | 36,888 | 36,888 | 36,888 | 0 | 0 | **PASS** |
| **1-lead** | 36,888 | 36,888 | 36,888 | 0 | 0 | **PASS** |
| **Total Workload** | **258,216** | **258,216** | **258,216** | **0** | **0** | **100% COMPLETE** |

### Split Breakdown per Variant:
- **Train Split:** 25,825 records $\times$ 7 variants = 180,775 representations
- **Validation Split:** 3,688 records $\times$ 7 variants = 25,816 representations
- **Test Split:** 7,375 records $\times$ 7 variants = 51,625 representations
- **Grand Total:** 258,216 representations

---

## 4. Representation Drift Quantitative Findings (Phase D)

Representation drift measures the mathematical distance between reduced-lead embeddings and the unmasked 12-lead reference embedding for the exact same patient ECG:

$$\text{cos\_sim}(z_{12}, z_{\text{var}}) = \frac{z_{12} \cdot z_{\text{var}}}{\|z_{12}\|_2 \cdot \|z_{\text{var}}\|_2}, \quad \text{cos\_dist} = 1 - \text{cos\_sim}, \quad \text{norm\_L2} = \frac{\|z_{12} - z_{\text{var}}\|_2}{\|z_{12}\|_2}$$

### 4.1. Pooled Drift Summary Across All 36,888 Records

| Lead Variant | Available Leads | Cosine Sim (Mean $\pm$ SD) | Cosine Sim (Median) | Cosine Distance (Mean) | Euclidean L2 Distance (Mean) | Normalized L2 Distance (Mean) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **12-lead** | I, II, III, aVR, aVL, aVF, V1–V6 | $1.000000 \pm 0.000000$ | $1.000000$ | $0.000000$ | $0.00$ | $0.0000$ |
| **10-lead** | I, II, aVR, aVL, V1–V6 | $0.998051 \pm 0.002085$ | $0.998645$ | $0.001949$ | $20.76$ | $0.0572$ |
| **8-lead** | I, II, V1–V6 | $0.997095 \pm 0.002668$ | $0.997850$ | $0.002905$ | $25.76$ | $0.0710$ |
| **6-lead** | I, II, III, aVR, aVL, aVF | $0.877548 \pm 0.062141$ | $0.890583$ | $0.122452$ | $174.32$ | $0.4802$ |
| **4-lead** | I, II, III, V1 | $0.904078 \pm 0.051987$ | $0.915447$ | $0.095922$ | $153.82$ | $0.4237$ |
| **2-lead** | II, V1 | $0.876178 \pm 0.062087$ | $0.887742$ | $0.123822$ | $175.31$ | $0.4829$ |
| **1-lead** | II | $0.814373 \pm 0.084938$ | $0.829456$ | $0.185627$ | $215.63$ | $0.5942$ |

### 4.2. $7 \times 7$ Pairwise Cosine Similarity Matrix (Pooled Across Splits)

| Reference Variant | 12-lead | 10-lead | 8-lead | 6-lead | 4-lead | 2-lead | 1-lead |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **12-lead** | **1.0000** | 0.9981 | 0.9971 | 0.8775 | 0.9041 | 0.8762 | 0.8144 |
| **10-lead** | 0.9981 | **1.0000** | 0.9982 | 0.8743 | 0.9015 | 0.8749 | 0.8124 |
| **8-lead** | 0.9971 | 0.9982 | **1.0000** | 0.8706 | 0.8986 | 0.8727 | 0.8085 |
| **6-lead** | 0.8775 | 0.8743 | 0.8706 | **1.0000** | 0.9443 | 0.9096 | 0.9026 |
| **4-lead** | 0.9041 | 0.9015 | 0.8986 | 0.9443 | **1.0000** | 0.9613 | 0.8760 |
| **2-lead** | 0.8762 | 0.8749 | 0.8727 | 0.9096 | 0.9613 | **1.0000** | 0.8923 |
| **1-lead** | 0.8144 | 0.8124 | 0.8085 | 0.9026 | 0.8760 | 0.8923 | **1.0000** |

### 4.3. Key Scientific Insights on Physiological Information Loss:
1. **The Precordial Lead Cluster (12-, 10-, 8-lead):** Preserving all precordial leads (V1–V6) maintains high stability with the 12-lead baseline ($> 0.997$ cosine similarity). The loss of limb leads III and aVF (10-lead) or further loss of aVR and aVL (8-lead) causes less than 0.3% cosine drift.
2. **The Precordial Information Cliff (6-lead):** When all precordial leads (V1–V6) are zeroed, leaving only the 6 limb leads, representation similarity drops precipitously from $0.9971$ to $0.8775$ (a 42-fold increase in cosine distance from $0.0029$ to $0.1225$). This quantifies the disproportionate diagnostic/spatial information carried by the horizontal plane (precordial) leads.
3. **The Single-Precordial Recovery Effect (4-lead):** Reintroducing a single precordial lead (V1) in the 4-lead configuration increases mean cosine similarity back up to $0.9041$ (higher than 6-lead limb only!), demonstrating that horizontal-plane anterior information provides essential orthogonal spatial data to the frontal limb leads.
4. **Single-Lead Degradation (1-lead):** Lead II alone exhibits substantial representation drift (mean cosine distance $0.1856$, normalized L2 distance $0.5942$).

---

## 5. Independent Verification Audit Findings (Phase F)

The independent verification script (`verify_module3_representations.py`) performed a comprehensive audit across all 258,216 representations:

- **Record Count & Partition Completeness:** All 21 (variant, split) partitions match expected counts exactly (180,775 train, 25,816 validation, 51,625 test).
- **Physical File Existence & Sizing:** All 258,216 individual `.npy` files exist in their designated directory structure. Every single file is exactly **3,200 bytes** on disk (zero file size anomalies).
- **Tensor Integrity:** All vectors are `(768,)` `float32`. 100% of values are finite. Zero NaN, zero Inf, zero negative or zero norms.
- **Matrix Consistency:** Consolidated matrices (`representations.npy`) match individual vector files and `record_ids.json` lists row-for-row.
- **Failures Logged:** `module3_failure_records.csv` contains **0 failure records**.

---

## 6. Documented Limitations & Scientific Caveats

In accordance with mandatory research integrity guidelines, the following known limitations are preserved and carried forward:
1. **Representation Drift vs. Diagnostic Performance:** Representation drift measures the geometric divergence of latent embeddings in $\mathbb{R}^{768}$. It indicates representation sensitivity to missing input channels; it does **not** by itself establish clinical misdiagnosis, diagnostic error, or causal reasoning failure. Downstream clinical diagnostic failure boundaries must be evaluated in subsequent modules.
2. **Chapman-Shaoxing Signal Duplication:** 13 identical signal pairs were identified in Module 1 (9 cross-split pairs and 4 within-train pairs). This signal-level leakage persists identically across all 7 masked variants.
3. **MIMIC-IV-ECG Algorithmic Labels:** MIMIC annotations are derived from GE Marquette 12SL automated interpretations rather than adjudicated expert cardiologist panels.
4. **Experimental Lead Configurations:** The 7 lead subsets evaluated (`12-`, `10-`, `8-`, `6-`, `4-`, `2-`, `1-lead`) are controlled experimental configurations designed to study systematic channel removal, and do not represent every commercial wearable or clinical reduced-lead montage.
5. **Observation Mask Integrity:** Unknown clinical labels remain strictly unobserved (`NaN`) with false observation masks. No imputation or negative assumptions were applied.

---

## 7. Audit Decision & Gate Status

- **Module 3 Status:** **PASS**
- **Readiness for Freeze:** All required scripts, representations, reports, summaries, and verification guides have been produced, verified, and sealed.

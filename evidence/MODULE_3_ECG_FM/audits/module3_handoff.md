# MODULE 3 — SYSTEM HANDOFF & SPECIFICATION

**Project:** Reasoning Failure Boundaries in ECG Foundation Models Under Progressive Physiological Information Loss  
**Component:** Module 3 (Official Foundation Model Authentication, Frozen Representation Extraction, & Progressive Representation Drift Audit)  
**Date:** 2026-10-13  
**Status:** **FROZEN & IMMUTABLE**  

---

## 1. Foundation Model Identity & Cryptographic Authentication

### 1.1. Verification Verdict
The local checkpoint located at:
`checkpoints/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt`
was **conclusively authenticated as the official, unaltered pretrained ECG-FM foundation model** from the cited publication:
- **Reference Publication:** K. McKeen, S. Masood, A. Toma, B. Rubin, and B. Wang, *"ECG-FM: An Open Electrocardiogram Foundation Model,"* *JAMIA Open*, vol. 8, no. 5, Art. ooaf122, 2025. [DOI: 10.1093/jamiaopen/ooaf122](https://doi.org/10.1093/jamiaopen/ooaf122).
- **Official Model Weight Repository:** [https://huggingface.co/wanglab/ecg-fm](https://huggingface.co/wanglab/ecg-fm).
- **Official GitHub Codebase:** [https://github.com/bowang-lab/ECG-FM/](https://github.com/bowang-lab/ECG-FM/).

### 1.2. Cryptographic and Architectural Evidence
1. **SHA-256 Checksum:** `4d0142bcb485eb9f0c7845e0c19ff3463f6ae9d0e458eab69136efe90ceb9b7e`
   - Perfectly matches the official Hugging Face Git LFS pointer OID byte-for-byte.
2. **File Size:** Exactly `1,090,825,421` bytes (~1.09 GB).
3. **Architecture:** Reconciled as `wav2vec2_cmsc` (`Wav2Vec2CMSCModel`) with:
   - Exactly **90,883,072 parameters** (~90.9M).
   - 4 convolutional feature extractor blocks with `in_d = 12`.
   - 12 Transformer encoder layers, 768 hidden dimension, 12 attention heads, 3072 feed-forward dimension.
   - Zero missing and zero unexpected keys loaded with `strict=True`.
4. **Checkpoint Classification:** Confirmed as the **self-supervised pretrained foundation model** (`mimic_iv_ecg_physionet_pretrained.pt`) rather than the downstream-finetuned model (`mimic_iv_ecg_finetuned.pt`), preserving experimental fidelity for foundation representation probing.

---

## 2. Representation Extraction Methodology

### 2.1. Model Freezing
- All model parameters were frozen (`param.requires_grad = False`).
- Evaluated strictly in `model.eval()` mode under `torch.no_grad()`.
- Zero trainable adapters, bottleneck projections, or head networks were introduced.

### 2.2. Tensor Input & Output Contracts
- **Input Contract:** Tensor of shape `(B, 12, 1000)` in `float32` representation with millivolt units at 100.0 Hz sampling rate (10.0 seconds).
- **Extraction Layer:** Contextualized Transformer encoder sequence representations via `model(source=x, mask=False, features_only=True)['x']`.
- **Temporal Pooling:** Temporal mean pooling over sequence dimension yields continuous latent representation $Z_{\text{ECG}} \in \mathbb{R}^{768}$.
- **Storage Format:** 
  - Individual files: `runs/module3_v1/representations/<variant>/<split>/records/<global_record_id>.npy` (each exactly 3,200 bytes).
  - Consolidated matrices: `runs/module3_v1/representations/<variant>/<split>/representations.npy` (shape `(N, 768)`).
  - Manifests: `record_ids.json` and `extraction_manifest.csv`.

---

## 3. Workload Accounting & Validation

- **Expected Workload:** 36,888 records $\times$ 7 lead variants = **258,216 representations**.
- **Successfully Extracted:** **258,216** (100.0%).
- **Validated by Independent Audit:** **258,216** (100.0%).
- **Failed Records:** **0**.
- **Excluded Records in Module 3:** **0** (carries forward the 69 QC exclusions from Module 1).

### Distribution by Partition:
- **Train:** 25,825 records $\times$ 7 variants = 180,775 representations
- **Validation:** 3,688 records $\times$ 7 variants = 25,816 representations
- **Test:** 7,375 records $\times$ 7 variants = 51,625 representations

---

## 4. Key Scientific Findings: Progressive Representation Drift

Comparing reduced-lead embeddings directly with the unmasked 12-lead baseline of the exact same ECG reveals structured, non-linear representation drift:

| Variant | Configuration | Mean Cosine Sim | Median Cosine Sim | Cosine Distance | Norm L2 Distance | Characterization |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **12-lead** | 12 leads unmasked | $1.000000$ | $1.000000$ | $0.000000$ | $0.0000$ | Reference Baseline |
| **10-lead** | III, aVF zeroed | $0.998051$ | $0.998645$ | $0.001949$ | $0.0572$ | Highly Stable ($>0.998$) |
| **8-lead** | III, aVF, aVR, aVL zeroed | $0.997095$ | $0.997850$ | $0.002905$ | $0.0710$ | Highly Stable ($>0.997$) |
| **6-lead** | Precordials V1–V6 zeroed | $0.877548$ | $0.890583$ | $0.122452$ | $0.4802$ | Precordial Cliff (Drift spike) |
| **4-lead** | I, II, III, V1 | $0.904078$ | $0.915447$ | $0.095922$ | $0.4237$ | Single-Precordial Recovery |
| **2-lead** | II, V1 | $0.876178$ | $0.887742$ | $0.123822$ | $0.4829$ | Severe Drift |
| **1-lead** | II only | $0.814373$ | $0.829456$ | $0.185627$ | $0.5942$ | Maximal Degradation |

### Core Observations:
1. **Precordial Information Dominance:** Zeroing limb leads while preserving precordial leads (`8-lead`: I, II, V1-V6) preserves 99.7% cosine similarity to the 12-lead embedding. Conversely, zeroing precordial leads (`6-lead`: I, II, III, aVR, aVL, aVF) causes an abrupt 42-fold increase in cosine distance ($0.0029 \rightarrow 0.1225$).
2. **Reintroduction of V1 Restores Orthogonal Plane Info:** The 4-lead variant (I, II, III, V1) yields higher similarity to the 12-lead embedding ($0.9041$) than the 6-lead variant ($0.8775$), demonstrating the high information value of anterior precordial lead V1.
3. **Monotonicity Within Subspaces:** Within the limb lead reduction trajectory (`12-lead` $\rightarrow$ `10-lead` $\rightarrow$ `8-lead`) and the minimal lead trajectory (`4-lead` $\rightarrow$ `2-lead` $\rightarrow$ `1-lead`), representation drift increases monotonically with lead removal.

---

## 5. Artifact & Directory Inventory

The complete Module 3 artifacts are organized under `runs/module3_v1/`:

```text
runs/module3_v1/
├── MODULE3_FROZEN.json                         # Formal cryptographic freeze marker
├── checksums_module3_manifest.sha256           # Bitwise SHA-256 seal covering key artifacts
├── representations/                            # 258,216 representation vectors & matrices
│   ├── 12-lead/ {train, validation, test}/
│   ├── 10-lead/ {train, validation, test}/
│   ├── 8-lead/  {train, validation, test}/
│   ├── 6-lead/  {train, validation, test}/
│   ├── 4-lead/  {train, validation, test}/
│   ├── 2-lead/  {train, validation, test}/
│   └── 1-lead/  {train, validation, test}/
├── scripts/
│   ├── verify_ecg_fm_checkpoint.py             # Phase A checkpoint authentication
│   ├── run_module3_pilot.py                    # Phase C4 pilot determinism check
│   ├── extract_foundation_representations.py   # Primary representation extraction engine
│   ├── extract_parallel_coordinator.py         # Multi-worker extraction orchestrator
│   ├── analyze_representation_drift.py         # Drift metrics & pairwise similarity suite
│   ├── verify_module3_representations.py       # Independent 258k file auditor
│   └── manual_verify_representation.py         # CLI inspection utility
├── reports/
│   ├── model_checkpoint_verification.md        # Checkpoint authentication report
│   ├── model_checkpoint_verification.json      # Checkpoint verification machine audit
│   ├── module3_inference_audit.md              # Representation audit report
│   ├── module3_per_file_audit.csv              # Per-file 258,216-row audit table
│   ├── module3_split_summary.csv               # Split-level summary counts
│   ├── module3_file_size_audit.csv             # Physical file size audit
│   ├── module3_failure_records.csv             # Failure log (0 failures)
│   ├── module3_representation_drift_per_record.csv # Full per-record drift metrics (258,216 rows)
│   ├── module3_cosine_similarity_summary.csv   # Aggregate drift summary by variant/split
│   ├── module3_pairwise_similarity_matrix.csv  # 7x7 pairwise cosine similarity matrix
│   ├── module3_source_summary.csv              # Source-stratified drift summary
│   ├── module3_disease_stratified_drift.csv    # Disease-stratified drift summary
│   ├── module3_manual_verification_guide.md    # Manual verification guide
│   ├── module3_final_audit_report.md           # Master audit report
│   └── module3_handoff.md                      # This handoff specification
└── logs/                                       # Extraction process logs
```

---

## 6. Input Immutability & Downstream Integrity

- **Module 1 Intact:** `runs/module1_v2/MODULE1_FROZEN.json` manifest hash `47bb4547d8a39f4a5ec1697f59dfe0a104b79811642c39115d73579fed09301c` remains 100% verified and untouched.
- **Module 2 Intact:** `runs/module2_v1/MODULE2_FROZEN.json` manifest hash `c7c2190198be54fc0ac01e9ceac96d3eb88333c93f74c9fa54bb5767eee47298` remains 100% verified and untouched.
- **Module 3 Sealed:** Module 3 is officially frozen and ready for downstream diagnostic probing and reasoning failure boundary analysis.

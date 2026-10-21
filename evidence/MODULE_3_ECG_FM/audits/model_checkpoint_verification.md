# MODULE 3 — PHASE A: ECG-FM CHECKPOINT VERIFICATION REPORT

**Project:** Reasoning Failure Boundaries in ECG Foundation Models Under Progressive Physiological Information Loss  
**Execution Phase:** Phase A (Foundation Model Checkpoint Authentication & Architecture Audit)  
**Verification Date:** 2026-10-13 03:11:20 UTC  
**Verification Verdict:** **PASS**  

---

## 1. Official Reference Publication & Repositories

- **Title:** *ECG-FM: An Open Electrocardiogram Foundation Model*
- **Authors:** K. McKeen, S. Masood, A. Toma, B. Rubin, and B. Wang
- **Journal Citation:** *JAMIA Open*, vol. 8, no. 5, Art. ooaf122, 2025.
- **DOI:** [https://doi.org/10.1093/jamiaopen/ooaf122](https://doi.org/10.1093/jamiaopen/ooaf122)
- **Official GitHub Repository:** [https://github.com/bowang-lab/ECG-FM/](https://github.com/bowang-lab/ECG-FM/)
- **Official Hugging Face Weights Repository:** [https://huggingface.co/wanglab/ecg-fm](https://huggingface.co/wanglab/ecg-fm)

---

## 2. Checkpoint Identity & Cryptographic Provenance

| Verification Metric | Expected Reference Value | Locally Inspected Value | Status |
| :--- | :--- | :--- | :---: |
| **Local Checkpoint Path** | `checkpoints/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt` | `E:/ECG/checkpoints/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt` | **FOUND** |
| **Local File Size** | 1,090,825,421 bytes (~1.09 GB) | 1,090,825,421 bytes | **MATCH** |
| **SHA-256 Checksum** | Official Hugging Face Git LFS OID | `4d0142bcb485eb9f0c7845e0c19ff3463f6ae9d0e458eab69136efe90ceb9b7e` | **MATCH** |
| **Hugging Face LFS Pointer** | `sha256:4d0142bcb485eb9f0c7845e0c19ff3463f6ae9d0e458eab69136efe90ceb9b7e` | `sha256:4d0142bcb485eb9f0c7845e0c19ff3463f6ae9d0e458eab69136efe90ceb9b7e` | **100% CRYPTOGRAPHIC IDENTITY** |
| **Model Configuration** | `mimic_iv_ecg_physionet_pretrained.yaml` | SHA-256: `710f5fdc54c31fd88c82d3ec1ad9c39c82a7848201fa6afd84074d5b71b726ab` | **MATCH** |

---

## 3. Checkpoint Classification: Pretrained Foundation Model vs. Downstream Finetuned

The official ECG-FM repository provides two checkpoints:
1. `mimic_iv_ecg_physionet_pretrained.pt` (the self-supervised pretrained foundation model)
2. `mimic_iv_ecg_finetuned.pt` (the model fine-tuned for downstream diagnostic classification)

### Provenance & Classification Evidence:
- **Selected File:** `mimic_iv_ecg_physionet_pretrained.pt`
- **Model Architecture Class:** `wav2vec2_cmsc` (`Wav2Vec2CMSCModel`)
- **State Dictionary Structure:** Contains the complete self-supervised pretraining task state, optimizer history, criterion, and CMSC contrastive heads. It does **not** contain downstream classification adapter heads.
- **Classification Verdict:** **PRETRAINED FOUNDATION MODEL** (Confirmed matching experimental protocol).

---

## 4. Architecture Reconciliation Against Paper Specifications

| Architectural Parameter | Publication Specification | Checkpoint / Instantiated Model | Reconciliation Status |
| :--- | :---: | :---: | :---: |
| **Model Family / Class** | wav2vec 2.0 (CMSC) | `wav2vec2_cmsc` (`Wav2Vec2CMSCModel`) | **EXACT MATCH** |
| **Total Parameters** | ~90.9 million | **90,883,072** (90.88M) | **EXACT MATCH** |
| **Trainable Parameters** | 0 (Strictly frozen for Module 3) | **0** (All `requires_grad = False`) | **EXACT MATCH** |
| **Convolutional Feature Extractor** | 4 blocks, (256, 2, 2) $\times$ 4 | 4 conv layers (k=2, s=2), group-norm, GELU | **EXACT MATCH** |
| **Input Signal Channels** | 12 leads | `in_d = 12` | **EXACT MATCH** |
| **Transformer Encoder Layers** | 12 layers | 12 Transformer layers | **EXACT MATCH** |
| **Hidden Embedding Dimension** | 768 | 768 | **EXACT MATCH** |
| **Self-Attention Heads** | 12 heads | 12 heads (64 dim / head) | **EXACT MATCH** |
| **Feed-Forward Dimension** | 3072 | 3072 | **EXACT MATCH** |
| **State Dictionary Matching** | Strict load (no missing / unexpected keys) | `strict=True` load succeeded with 0 errors | **EXACT MATCH** |

---

## 5. Inference Interface & Representation Extraction Protocol

- **Input Contract:** Tensor of shape `(B, 12, 1000)` in `float32` representation with millivolt units at 100.0 Hz sampling rate (10.0 seconds).
- **Lead Ordering:** Canonical order `[I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]`.
- **Extraction Layer:** Contextualized Transformer encoder representations via `model(source=x, mask=False, features_only=True)['x']`.
- **Temporal Pooling:** Temporal mean pooling over frame sequence dimension: Z_ECG = mean(X_seq, dim=1) in R^768.
- **Output Representation:** `(B, 768)` in `torch.float32`.
- **Finiteness & Stability:** Verified on sample zero and random inputs. Zero non-finite values.

---

## 6. Audit Decision & Gate Status

- **Status:** **PASS**
- **Conclusion:** The local checkpoint `checkpoints/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt` is authenticated conclusively as the official, unaltered pretrained ECG-FM model from McKeen et al. (2025). Phase A verification criteria are fully satisfied. Phase B and Phase C may proceed.

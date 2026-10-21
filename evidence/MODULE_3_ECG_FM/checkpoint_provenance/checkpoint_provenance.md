# Module 3 — ECG-FM Foundation Model Checkpoint Provenance

**Module Identifier:** `MODULE_3_ECG_FM`  
**Checkpoint Filename:** `mimic_iv_ecg_physionet_pretrained.pt`  
**Relative Checkpoint Path:** `checkpoints/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt`  
**Audit Source Reference:** `runs/module3/reports/model_checkpoint_verification.md`  

---

## 1. Official Model Attribution and Source Publication

- **Model Title:** *ECG-FM: An Open Electrocardiogram Foundation Model*
- **Primary Authors:** K. McKeen, S. Masood, A. Toma, B. Rubin, and B. Wang (Wang Lab, University of Toronto / Vector Institute)
- **Peer-Reviewed Citation:** *JAMIA Open*, vol. 8, no. 5, Art. ooaf122, 2025.
- **Publication DOI:** [https://doi.org/10.1093/jamiaopen/ooaf122](https://doi.org/10.1093/jamiaopen/ooaf122)
- **Official GitHub Codebase:** [https://github.com/bowang-lab/ECG-FM/](https://github.com/bowang-lab/ECG-FM/)
- **Hugging Face Model Repository:** [https://huggingface.co/wanglab/ecg-fm](https://huggingface.co/wanglab/ecg-fm)

---

## 2. Cryptographic Checksum and Storage Verification

| Provenance Property | Reference Hugging Face LFS OID | Local Repository Artifact | Verification Result |
| :--- | :--- | :--- | :---: |
| **Local Path** | `checkpoints/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt` | `E:/ECG/checkpoints/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt` | **VERIFIED** |
| **File Size (Bytes)** | 1,090,825,421 bytes | 1,090,825,421 bytes (~1.09 GB) | **EXACT MATCH** |
| **SHA-256 Hash** | `4d0142bcb485eb9f0c7845e0c19ff3463f6ae9d0e458eab69136efe90ceb9b7e` | `4d0142bcb485eb9f0c7845e0c19ff3463f6ae9d0e458eab69136efe90ceb9b7e` | **EXACT MATCH** |
| **Companion Config** | `mimic_iv_ecg_physionet_pretrained.yaml` | SHA-256: `710f5fdc54c31fd88c82d3ec1ad9c39c82a7848201fa6afd84074d5b71b726ab` | **VERIFIED** |

---

## 3. Foundation Model vs Downstream Fine-Tuned Checkpoint Authentication

The Wang Lab distribution provides two distinct checkpoint artifacts:
1. `mimic_iv_ecg_physionet_pretrained.pt` (Self-supervised CMSC contrastive pretraining checkpoint)
2. `mimic_iv_ecg_finetuned.pt` (Supervised diagnostic fine-tuned checkpoint)

### Authentication Evidence:
- **Selected File:** `mimic_iv_ecg_physionet_pretrained.pt`
- **Class Instantiation:** `Wav2Vec2CMSCModel` from `fairseq` / `ecg_fm` architecture.
- **State Dictionary Inspection:**
  - Contains contrastive projection heads, cross-modal CMSC loss buffers, and Transformer encoder parameters.
  - Contains **zero** downstream task-specific linear heads or diagnostic classifier weights.
  - Parameter weights are completely untuned for the 8 downstream target diseases.
- **Frozen Protocol Enforcement:** During representation extraction across all 258,216 input arrays, the backbone parameters were strictly frozen (`requires_grad = False`, `model.eval()`).

---

## 4. State Dictionary Loading & Integrity Verification

- **Loading Mode:** PyTorch `strict=True` state dictionary loading.
- **Missing Keys:** Exactly 0.
- **Unexpected Keys:** Exactly 0.
- **Device Placement:** CUDA execution with deterministic precision (`torch.float32`).
- **Reproducibility Verdict:** Authenticated conclusively as the unaltered official pretrained foundation model.

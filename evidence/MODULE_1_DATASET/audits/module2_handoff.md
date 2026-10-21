# MODULE 2 HANDOFF SPECIFICATION

**Project:** Reasoning Failure Boundaries in ECG Foundation Models Under Progressive Physiological Information Loss  
**Handoff From:** Module 1 (Dataset Standardization, Eight-Label Harmonization, and Stratified 70:10:20 Splitting)  
**Handoff To:** Module 2 (Progressive Physiological Lead Masking & Artifact Generation)  
**Handoff Location:** `runs/module1_v2/reports/module2_handoff.md`  
**Dataset Freeze Status:** MODULE 1 FROZEN (Immutable)  

---

## 1. Frozen Dataset Interface & Paths

Module 2 must consume the standardized, quality-controlled, split-partitioned ECG dataset directly from the following immutable paths:

| Split Partition | Physical Records Directory | Tabular Metadata Manifest | Multilabel Ground Truth | Observation Mask | Record Count |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Training** | `runs/module1_v2/dataset/train/records/` | `runs/module1_v2/dataset/train/metadata.csv` | `runs/module1_v2/dataset/train/labels.csv` | `runs/module1_v2/dataset/train/observation_mask.csv` | **25,825** |
| **Validation** | `runs/module1_v2/dataset/validation/records/` | `runs/module1_v2/dataset/validation/metadata.csv` | `runs/module1_v2/dataset/validation/labels.csv` | `runs/module1_v2/dataset/validation/observation_mask.csv` | **3,688** |
| **Testing** | `runs/module1_v2/dataset/test/records/` | `runs/module1_v2/dataset/test/metadata.csv` | `runs/module1_v2/dataset/test/labels.csv` | `runs/module1_v2/dataset/test/observation_mask.csv` | **7,375** |
| **Global Manifest** | `runs/module1_v2/splits/split_manifest.csv` | Contains full metadata, patient grouping, folds (1–10), and split assignments. | | | **36,888** |

---

## 2. Signal Tensor Specifications

Every file `<global_record_id>.npy` in the `records/` directories complies with the following strict contract:
- **Tensor Shape:** `(12, 1000)` (2D NumPy array, channels $\times$ time).
- **Data Type:** `numpy.float32`.
- **Physical Voltage Units:** Millivolts (mV). No arbitrary normalization has been applied; physical scale is preserved.
- **Sampling Frequency:** Exactly **100.0 Hz** (10 milliseconds per sample).
- **Temporal Duration:** Exactly **10.0 seconds** (1,000 time steps per lead).
- **Canonical Lead Order (Index 0 to 11):**
  ```python
  CANONICAL_LEADS = [
      "I",   # Index 0: Limb Lead I
      "II",  # Index 1: Limb Lead II
      "III", # Index 2: Limb Lead III
      "aVR", # Index 3: Augmented Unipolar Right
      "aVL", # Index 4: Augmented Unipolar Left
      "aVF", # Index 5: Augmented Unipolar Foot
      "V1",  # Index 6: Precordial V1 (Septal)
      "V2",  # Index 7: Precordial V2 (Septal)
      "V3",  # Index 8: Precordial V3 (Anterior)
      "V4",  # Index 9: Precordial V4 (Anterior)
      "V5",  # Index 10: Precordial V5 (Lateral)
      "V6"   # Index 11: Precordial V6 (Lateral)
  ]
  ```

---

## 3. Multi-Label Disease Targets & Observation Mask Contract

The eight disease targets must be evaluated independently:
1. `NORM`: Normal ECG
2. `AFIB`: Atrial Fibrillation
3. `RBBB`: Complete Right Bundle Branch Block
4. `LBBB`: Complete Left Bundle Branch Block
5. `SBRAD`: Sinus Bradycardia
6. `STACH`: Sinus Tachycardia
7. `1AVB`: 1st Degree AV Block
8. `MI`: Myocardial Infarction

### Mandatory Observation Mask Usage in Loss Computation:
Because clinical annotations cannot always confirm disease absence (e.g. non-sinus rhythms for SBRAD/STACH/1AVB, or equivocal infarct statements for MI), `observation_mask.csv` provides a Boolean mask (`True` = observed known state, `False` = unobserved / ambiguous).

Downstream loss functions and metrics **must mask out unknown targets**:
```python
# Supervised Binary Cross-Entropy with Observation Masking:
def masked_bce_loss(logits, labels, observation_mask):
    # logits: (B, 8), labels: (B, 8), observation_mask: (B, 8) [bool]
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    masked_loss = (bce * observation_mask.float()).sum() / observation_mask.float().sum().clamp(min=1.0)
    return masked_loss
```
**CRITICAL RULE:** Never impute unknown targets as negative (`0.0`). Doing so will penalize foundation model representations on clinically unassessable targets.

---

## 4. Preprocessing Lineage & Known Caveats

1. **Filtering Lineage:** Signals were filtered at native 500 Hz using a zero-phase 0.5 Hz highpass, dataset-aware 50/60 Hz notch filter, and 45 Hz lowpass before polyphase decimation (`resample_poly`) to 100 Hz.
2. **Chapman Patient Separation Fallback & Duplicate Signals:** Chapman-Shaoxing does not provide native patient identifiers; `patient_id = record_id` was applied as an approved operational fallback. Downstream evaluation on Chapman represents record-level rather than confirmed patient-level separation. Furthermore, an exhaustive SHA-256 byte-level signal audit identified 13 identical recording pairs in Chapman (9 pairs across splits, 4 pairs within train) originating from duplicate entries in the raw source dataset. Zero duplicate signals exist in PTB-XL, MIMIC-IV-ECG, or across datasets.
3. **MIMIC Automated Interpretation:** MIMIC labels originate from the GE Marquette 12SL automated algorithm.
4. **Data Normalization Constraint:** Any learned normalization parameters (e.g. per-lead $z$-score mean and standard deviation) **must be fit exclusively on the `train` partition (25,825 records)** and applied frozen to `validation` and `test`.

---

## 5. Next Steps for Module 2

Module 2 may now proceed with:
1. Designing the structured physiological lead masking configurations (e.g. single limb leads, precordial-only, lead-subset ablations).
2. Applying lead masking to the frozen records.
3. Generating the standardized masked dataset artifacts under a dedicated output directory (e.g. `runs/module2/`).
4. Preserving Module 1 files as read-only inputs.

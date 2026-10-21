# Module 3 — ECG-FM Representation Extraction: Process Utility, Design Rationale, and Research Applications

**Module Identifier:** `MODULE_3_ECG_FM`  
**Execution Phase:** Pretrained Latent Representation Pipeline  
**Execution Date:** 2026-10-16  

---

## 1. Problem Statement and Research Purpose

### 1.1 The High-Dimensional Waveform Bottleneck
Raw 12-lead electrocardiograms are high-dimensional spatio-temporal time series ($12 \text{ channels} \times 1000 \text{ samples} = 12,000$ floating-point values per 10-second recording). Training downstream clinical reasoning architectures, tabular classifiers, or multi-modal LLM reasoning pipelines directly from raw waveform arrays requires massive computational infrastructure, introduces gradient instability across long temporal horizons, and risks severe overfitting when training on limited target disease cohorts.

### 1.2 The Research Role of Module 3
Module 3 solves this by leveraging a state-of-the-art, open-access foundation model (ECG-FM; McKeen et al., *JAMIA Open*, 2025) pretrained via self-supervised contrastive learning on hundreds of thousands of diverse clinical ECGs. Module 3 maps each degraded ECG waveform from Module 2 into a compact, semantically dense representation vector $\mathbf{z} \in \mathbb{R}^{768}$. This allows the project to rigorously evaluate how foundation model representations degrade under progressive lead loss, while providing a standardized latent embedding layer for downstream reasoning models in Module 4.

---

## 2. Design Decisions and Methodological Rationale

### 2.1 Complete Backbone Freezing (`requires_grad = False`)
- **Decision:** The 90.88M parameter `Wav2Vec2CMSCModel` backbone was evaluated strictly in inference mode (`eval()`) with no gradient updates or fine-tuning.
- **Rationale:** If the backbone were fine-tuned separately on each degraded lead configuration, the model could compensate for missing channels by memorizing dataset-specific correlations or altering its internal feature detectors. Strict freezing isolates the *intrinsic diagnostic robustness* of the foundation model's self-supervised representations under out-of-distribution lead loss.

### 2.2 Temporal Mean Pooling over Contextual Frames
- **Decision:** The contextual Transformer output sequence $\mathbf{X}_{\text{seq}} \in \mathbb{R}^{B \times 62 \times 768}$ was pooled across the temporal sequence dimension ($T=62$) using uniform mean reduction:
  $$\mathbf{z}_{\text{ECG}} = \frac{1}{62} \sum_{t=1}^{62} \mathbf{x}_{\text{seq}}[t] \in \mathbb{R}^{768}$$
- **Rationale:** Mean pooling summarizes the global electrophysiological state of the 10-second recording into a fixed-width vector. It provides a robust, permutation-invariant summary of rhythm, conduction, and repolarization suitable for multi-label classification and vector similarity auditing.

### 2.3 Direct Zero-Mask Input Feeding
- **Decision:** Degraded waveforms with zeroed channels were fed directly into the model's 12-channel convolutional input layer without channel pruning or dynamic architecture resizing.
- **Rationale:** Commercial and clinical foundation models are typically deployed as fixed neural artifacts. Direct zero-masking evaluates how the network's convolutional receptive fields and self-attention heads tolerate uninformative (flatline) inputs without retraining the front-end weights.

---

## 3. Practical and Downstream Applications

### 3.1 Efficient Downstream Model Probing
By pre-extracting and caching all 258,216 representations to disk as compact `.npy` files (3,200 bytes each, total disk footprint ~826 MB), Module 4 can train, calibrate, and evaluate multi-seed control-path classifiers and baseline architectures in minutes rather than days of heavy GPU cluster compute.

### 3.2 Representation Geometry and Latent Manifold Probing
Module 3 establishes quantitative benchmarks for latent manifold drift. Researchers can measure cosine distance, Mahalanobis distance, or singular value decomposition (SVD) spectrum changes to study how deep neural networks represent multi-channel sensory loss.

### 3.3 Diagnostic Triage and Feature Attribution
By comparing latent drift across lead subsets (e.g., comparing 6-lead frontal limb leads vs 4-lead frontal+septal), researchers can quantify the specific information contribution of individual leads (such as Lead V1) to the foundation model's embedding space.

---

## 4. Upstream Inputs and Downstream Dependencies

```
[Module 1: Harmonized Signals (12, 1000)]
                     │
                     ▼
[Module 2: 7 Degraded Lead Variants (12, 1000)]
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ MODULE 3: ECG-FM Representation Pipeline                    │
│ - Checkpoint: mimic_iv_ecg_physionet_pretrained.pt          │
│ - Architecture: Wav2Vec2CMSCModel (90.88M params, frozen)  │
│ - Temporal Pooling: mean(dim=1)                             │
│ - Output: 258,216 vectors in R^768                          │
└─────────────────────────────────────────────────────────────┘
                     │
                     ▼
[Module 4: Control-Path Classifier & Reasoning Engine]
  - Input: z_ecg in R^768 + Lead Mask m in {0,1}^12
  - Output: 8 Target Disease Logits & Reasoning Traces
```

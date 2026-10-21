# Module 3 — ECG-FM Architecture and Parameter Specification

**Module Identifier:** `MODULE_3_ECG_FM`  
**Model Architecture Family:** wav2vec 2.0 with Contrastive Multi-Segment Coding (CMSC)  
**Implementation Class:** `Wav2Vec2CMSCModel`  
**Reference Document:** McKeen et al., *JAMIA Open*, 2025  

---

## 1. Architectural Summary

ECG-FM adapts the self-supervised wav2vec 2.0 temporal convolutional and Transformer encoder architecture to multi-lead electrocardiography. It maps continuous 12-channel voltage signals into high-dimensional latent representations capturing temporal wave morphology, inter-lead spatial correlations, and cardiac cycle rhythms.

---

## 2. Detailed Layer Specifications

```
Input ECG Tensor (B, 12, 1000) [100 Hz, 10s, in mV]
       │
       ▼
┌───────────────────────────────────────────────────────────┐
│ 1D Convolutional Feature Encoder (4 Conv Blocks)          │
│ - Conv1D: channels=12 -> 256, kernel=2, stride=2, GELU    │
│ - Conv1D: channels=256 -> 256, kernel=2, stride=2, GELU   │
│ - Conv1D: channels=256 -> 256, kernel=2, stride=2, GELU   │
│ - Conv1D: channels=256 -> 256, kernel=2, stride=2, GELU   │
│ Temporal downsampling factor: 2^4 = 16 (1000 -> 62 frames)│
└───────────────────────────────────────────────────────────┘
       │
       ▼ Latent Feature Representations (B, 62, 256)
┌───────────────────────────────────────────────────────────┐
│ Linear Feature Projection                                 │
│ - Dense Projection: 256 -> 768                            │
│ - Positional Embedding: Convolutional relative position   │
│ - LayerNorm & Dropout (p=0.1)                             │
└───────────────────────────────────────────────────────────┘
       │
       ▼ Projected Tokens (B, 62, 768)
┌───────────────────────────────────────────────────────────┐
│ Contextual Transformer Encoder (12 Layers)                │
│ - Hidden Dimension: 768                                   │
│ - Multi-Head Attention: 12 heads (64 dim / head)          │
│ - Feed-Forward Intermediate Dim: 3072                     │
│ - Activation: GELU                                        │
│ - Residual Connections & Pre-LayerNorm                    │
└───────────────────────────────────────────────────────────┘
       │
       ▼ Contextual Sequence Tensors (B, 62, 768)
┌───────────────────────────────────────────────────────────┐
│ Temporal Pooling (Mean Reduction over 62 frames)          │
│ - Operation: mean(dim=1)                                  │
└───────────────────────────────────────────────────────────┘
       │
       ▼
Output Latent Vector Z_ECG in R^768 (torch.float32)
```

---

## 3. Parameter Accounting and Verification

| Subsystem Component | Layer Specification | Parameter Count | Trainable in Module 3 | Verification Method |
| :--- | :--- | :---: | :---: | :---: |
| **Convolutional Front-End** | 4 conv blocks (256 filters, k=2, s=2) + GroupNorm | 398,592 | 0 (`eval()`) | State dict tensor tally |
| **Feature Projection** | Dense projection (256 -> 768) + ConvPos | 632,064 | 0 (`eval()`) | State dict tensor tally |
| **Transformer Encoder** | 12 layers $\times$ (Self-Attention + FFN 3072) | 85,054,464 | 0 (`eval()`) | State dict tensor tally |
| **LayerNorm Buffers & Projections** | Pre/Post LayerNorms and Layer Scaling | 4,797,952 | 0 (`eval()`) | State dict tensor tally |
| **Total Model Parameters** | `Wav2Vec2CMSCModel` Backbone | **90,883,072** | **0** | Exact PyTorch count |

---

## 4. Input-Output Compatibility Contract

1. **Input Shape:** Uniform `(B, 12, 1000)` where $B$ is batch size, 12 is lead channels, 1000 is samples (10 s at 100 Hz).
2. **Channel Padding/Masking:** Degraded leads from Module 2 are supplied with masked channels explicitly set to 0.000 mV float32. The input layer channel dimension remains invariant at 12 channels.
3. **Extraction Call:**
   ```python
   with torch.no_grad():
       out = model(source=x_masked, mask=False, features_only=True)
       z_seq = out['x']  # Shape: (B, 62, 768)
       z_ecg = torch.mean(z_seq, dim=1)  # Shape: (B, 768)
   ```
4. **Storage Format:** Stored as float32 NumPy arrays of dimension `(768,)`.
   - File size: Exactly 3,200 bytes per record (3,072 data bytes + 128 bytes NumPy `.npy` header).

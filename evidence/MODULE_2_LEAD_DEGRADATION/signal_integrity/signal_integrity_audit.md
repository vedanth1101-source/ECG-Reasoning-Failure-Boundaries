# Module 2 — Signal Integrity and Masking Verification Audit

**Module:** Module 2 (Progressive ECG Lead Masking and Availability-Mask Generation)  
**Governing Report:** [`runs/module2_v1/reports/module2_lead_masking_audit.md`](file:///e:/ECG/runs/module2_v1/reports/module2_lead_masking_audit.md)  
**Master Configuration:** [`runs/module2_v1/reports/module2_variant_config.json`](file:///e:/ECG/runs/module2_v1/reports/module2_variant_config.json)  
**Audit Scope:** Full numerical audit of all 258,216 physical arrays on disk.

---

## 1. Zeroing Formula and Tensor Preservation

Module 2 applies a strict element-wise binary masking formula:

$$X_{\text{masked}}[l, t] = X[l, t] \cdot m[l], \quad \forall l \in \{0 \dots 11\}, \; t \in \{0 \dots 999\}$$

Where:
- $X \in \mathbb{R}^{12 \times 1000}$ is the unmasked Module 1 signal array (in physical mV).
- $m \in \{0, 1\}^{12}$ is the canonical configuration availability mask vector.
- $X_{\text{masked}} \in \mathbb{R}^{12 \times 1000}$ is the masked tensor.

### Core Invariance Guarantees
1. **Preserved Tensor Dimensions:** All arrays maintain shape `(12, 1000)` and `float32` precision. Channel dimensions are never collapsed or dropped, ensuring direct compatibility with the 12-channel convolutional front-end of the ECG-FM foundation model.
2. **Exact Numerical Preservation of Active Channels:** For all channels where $m[l] = 1$, the signal samples are numerically identical to the unmasked Module 1 source:
   $$\max_{t} |X_{\text{masked}}[l, t] - X[l, t]| = 0.0, \quad \forall l \text{ s.t. } m[l] = 1$$
3. **Exact Floating-Point Zeroing of Masked Channels:** For all channels where $m[l] = 0$, every sample is set to exact IEEE 754 zero:
   $$X_{\text{masked}}[l, t] \equiv 0.0, \quad \forall t \in \{0 \dots 999\}, \; \forall l \text{ s.t. } m[l] = 0$$
4. **Non-Reconstruction Policy:** No synthetic estimation, interpolation, regression, or neural reconstruction is performed on missing leads. Omitted channels remain strictly zeroed.

---

## 2. Independent Audit Results Across 258,216 Files

The automated integrity audit scanned all 258,216 physical `.npy` files across all 7 configuration directories:

| Quality Dimension | Verification Standard | Executed Audit Result | Verdict |
| :--- | :--- | :--- | :---: |
| **Physical File Count** | 36,888 files per configuration $\times 7 = 258,216$ | Exactly 258,216 files present on disk | **PASS** |
| **Uniform File Size** | Exactly 48,128 bytes per `.npy` file | 258,216 / 258,216 files match 48,128 bytes | **PASS** |
| **Finiteness Check** | 0 NaN, 0 Inf across all $3.098\times 10^9$ data points | 100.0% finite (0 non-finite values) | **PASS** |
| **Active Lead Integrity** | Maximal absolute difference vs. Module 1 source $= 0.0$ | $\max(|X_{\text{masked}} - X|) = 0.0$ on active channels | **PASS** |
| **Masked Lead Integrity** | Maximum absolute value on masked channels $= 0.0$ | $\max(|X_{\text{masked}}|) = 0.0$ on masked channels | **PASS** |
| **Cross-Configuration Alignment** | Record IDs and split partitions identical to Module 1 | 100.0% row-for-row match | **PASS** |

Zero corruptions, zero leaks, and zero non-finite values were detected across the entire 258,216-array universe.

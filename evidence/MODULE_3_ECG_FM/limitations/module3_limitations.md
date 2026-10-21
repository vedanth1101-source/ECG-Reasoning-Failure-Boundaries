# Module 3 — Pretrained ECG-FM Representations: Known Limitations and Downstream Research Impacts

**Module Identifier:** `MODULE_3_ECG_FM`  
**Authoritative Frozen Contract:** `runs/module3/MODULE3_FROZEN.json`  
**Execution Date:** 2026-10-16  

---

## 1. Scope and Objective

This document details the architectural, statistical, and clinical limitations of using frozen ECG-FM representations for lead degradation analysis. It explains why representation stability cannot be equated directly with clinical diagnostic accuracy.

---

## 2. Identified Technical and Methodological Limitations

### 2.1 Information Loss via Global Temporal Mean Pooling
- **Mechanism:** The contextual Transformer encoder produces an output tensor $\mathbf{X}_{\text{seq}} \in \mathbb{R}^{B \times 62 \times 768}$, where 62 represents the temporal sequence of downsampled convolutional frames (~160 ms per frame). Module 3 collapses this sequence to a single vector $\mathbf{z} \in \mathbb{R}^{768}$ via temporal mean pooling:
  $$\mathbf{z} = \frac{1}{62} \sum_{t=1}^{62} \mathbf{x}_{\text{seq}}[t]$$
- **Limitation:** Mean pooling aggregates transient electrophysiological phenomena across the 10-second window. Paroxysmal events (such as isolated premature ventricular contractions, transient sinus pauses, or intermittent bundle branch block beats) have their latent activations attenuated by the surrounding normal beats.
- **Downstream Consequence:** Module 4 models operating on $\mathbf{z} \in \mathbb{R}^{768}$ lack fine-grained temporal resolution to locate specific beat-by-beat anomalies or measure exact millisecond durations of the PR interval, QRS complex, or QT interval.

### 2.2 Latent Cosine Drift is Not a Direct Measure of Clinical Diagnostic Failure
- **Mechanism:** Module 3 measures representation drift using cosine similarity $1 - \cos(\mathbf{z}_{\text{deg}}, \mathbf{z}_{\text{12-lead}})$.
- **Limitation:** In high-dimensional spaces ($\mathbb{R}^{768}$), high cosine similarity (e.g., 0.98 in 10-lead) indicates that the global direction of the latent vector is largely preserved. However, diagnostic classification boundaries for critical conditions (such as acute myocardial infarction or subtle conduction blocks) may depend on small sub-manifolds or specific low-variance orthogonal dimensions. A drift of only 0.002 can correspond to crossing a critical clinical decision boundary. Conversely, a large drift (e.g., 0.12 in 6-lead) might reflect the benign disappearance of baseline precordial noise without losing the primary rhythm classification.
- **Downstream Consequence:** Researchers must avoid concluding that a degraded configuration is clinically equivalent to 12-lead merely because its embedding cosine similarity is above 0.95. Diagnostic claims must be validated using calibrated clinical predictions in Module 4.

### 2.3 Out-of-Distribution Zero-Mask Artifacts in Convolutional Layers
- **Mechanism:** Degraded channels are set to constant 0.000 mV. The 4-layer 1D convolutional front-end processes all 12 channels simultaneously across multi-channel kernels.
- **Limitation:** The foundation model was pretrained primarily on full 12-lead clinical ECGs with active biological waveforms. Flatline channels introduce artificial zero-variance vectors into the front-end GroupNorm layers and attention mechanisms that were absent during self-supervised pretraining.
- **Downstream Consequence:** Some representation drift may reflect out-of-distribution neural activation artifacts rather than purely physiological information loss.

### 2.4 Pretraining Cohort Demographics and Precordial Weighting
- **Mechanism:** The checkpoint was pretrained on MIMIC-IV-ECG, an emergency and intensive care cohort from Beth Israel Deaconess Medical Center.
- **Limitation:** The self-supervised objective (CMSC) learned representations optimized for the hospital's specific patient demographic and lead placement conventions. Precordial leads V1–V6 carry substantial mutual information about left ventricular hypertrophy and coronary artery disease in this population, which may explain the sharp representation drop when all six precordial leads are removed in the 6-lead configuration.

---

## 3. Impact on Downstream Research Conclusions

1. **Representation Stability $\neq$ Diagnostic Equivalence:** Embedding metrics provide an upper bound on information preservation, but diagnostic performance must be assessed by empirical task-specific evaluation.
2. **Support for Non-Monotonic Hypotheses:** The finding that 4-lead representations retain higher cosine similarity to 12-lead than 6-lead representations confirms that horizontal plane information (Lead V1) is structurally vital to the foundation model's embedding manifold.

---

## 4. Verification Evidence References

- `evidence/MODULE_3_ECG_FM/representation_extraction/representation_configuration_inventory.csv`
- `evidence/MODULE_3_ECG_FM/representation_statistics/representation_drift_summary.csv`
- `evidence/MODULE_3_ECG_FM/representation_statistics/representation_integrity_audit.md`
- `evidence/MODULE_3_ECG_FM/audits/module3_final_audit_report.md`

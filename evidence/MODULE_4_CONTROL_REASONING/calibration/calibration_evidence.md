# Module 4 — Probability Calibration Evidence and Temperature Scaling Audit

**Module Identifier:** `MODULE_4_CONTROL_REASONING`  
**Authoritative Frozen Contract:** `runs/module4_v3/MODULE4_FROZEN.json`  
**Method:** Per-Disease Post-Hoc Temperature Scaling (Validation Split Only)  
**Execution Date:** 2026-10-16  

---

## 1. Calibration Objective and Clinical Rationale

Deep neural networks trained with cross-entropy loss frequently output overconfident probabilities, especially under covariate shift or sensor degradation. In clinical cardiology, uncalibrated risk probabilities undermine triage decision-making. 

Module 4 implements per-disease post-hoc temperature scaling:
$$\hat{p}_{i, d} = \sigma\left(\frac{z_{i, d}}{T_d}\right)$$
where $z_{i, d}$ is the raw uncalibrated logit for record $i$ and disease $d$, and $T_d > 0$ is the learned temperature parameter.

---

## 2. Validation-Only Protocol and Zero-Leakage Guarantee

- **Fitting Split:** Folds 7–8 (Validation split, 3,688 records $\times$ 7 configurations = 25,816 evaluations).
- **Optimization Objective:** Minimization of Negative Log-Likelihood (NLL) over observed labels ($o_{i,d} = 1$) via L-BFGS-B optimization.
- **Strict Quarantine Rule:** Temperature parameters were fitted **strictly on validation predictions**. The test set (Folds 9–10) was never accessed or used to fit, tune, or select temperatures.

---

## 3. Calibrated Parameters and Performance Improvements

### 3.1 Learned Optimal Temperature Parameters ($T_d$)

| Target Disease | Optimal Temperature $T_d$ | Uncalibrated Brier Score | Calibrated Brier Score | Uncalibrated ECE | Calibrated ECE | ECE Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **NORM** | 1.142 | 0.0912 | 0.0884 | 0.0421 | 0.0185 | -56.1% |
| **AFIB** | 1.085 | 0.0415 | 0.0398 | 0.0384 | 0.0162 | -57.8% |
| **RBBB** | 1.112 | 0.0321 | 0.0309 | 0.0452 | 0.0198 | -56.2% |
| **LBBB** | 1.094 | 0.0245 | 0.0238 | 0.0391 | 0.0174 | -55.5% |
| **SBRAD** | 1.168 | 0.0389 | 0.0371 | 0.0512 | 0.0224 | -56.3% |
| **STACH** | 1.135 | 0.0442 | 0.0426 | 0.0489 | 0.0211 | -56.9% |
| **1AVB** | 1.204 | 0.0287 | 0.0274 | 0.0584 | 0.0261 | -55.3% |
| **MI** | 1.248 | 0.0512 | 0.0489 | 0.0621 | 0.0298 | -52.0% |
| **Mean** | **1.148** | **0.0440** | **0.0424** | **0.0482** | **0.0215** | **-55.4%** |

*Note on Denominators:* ECE (Expected Calibration Error) was calculated with 10 equal-width probability bins using strictly observed labels ($o_{i,d}=1$).

---

## 4. Preservation of Discrimination Metrics

Because temperature scaling applies a strictly monotonic positive transformation ($T_d > 0$), the ranking order of predicted probabilities is invariant:
$$\text{AUROC}(\sigma(z / T)) \equiv \text{AUROC}(\sigma(z))$$
$$\text{AUPRC}(\sigma(z / T)) \equiv \text{AUPRC}(\sigma(z))$$
The validation macro-AUROC (0.8912) and per-disease AUROCs remained unchanged, while calibration error was reduced by over 55%.

---

## 5. Audit Verdict

**PASSED.** Temperature scaling was implemented correctly without test set leakage and achieved substantial calibration gains across all eight target diseases.

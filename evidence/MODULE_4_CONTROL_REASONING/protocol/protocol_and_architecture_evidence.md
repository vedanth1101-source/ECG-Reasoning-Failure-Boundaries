# Module 4 — Protocol and Architecture Evidence

**Module Identifier:** `MODULE_4_CONTROL_REASONING`  
**Authoritative Frozen Contract:** `runs/module4_v3/MODULE4_FROZEN.json`  
**Execution Phase:** Architecture, Loss Function, and Training Protocol Verification  
**Primary Checkpoint:** `runs/module4_v3/checkpoints/control_reasoning_best_seed42.pt` (SHA-256: `2920c76884e4a4325a7a9de725b8bbec6a4613ff78fce085a67bb1842e472ce7`)  

---

## 1. Experimental Protocol and Control-Path Architecture

### 1.1 Purpose of the Control Path
The Module 4 control-path architecture evaluates the diagnostic classification and reasoning capabilities of foundation model representations across degraded lead configurations. A central design constraint is that a single unified model processes all seven lead configurations (`12, 10, 8, 6, 4, 2, 1 lead`). This prevents configuration-specific models from memorizing different lead combinations and tests whether a single system can adapt its reasoning as leads are progressively removed.

### 1.2 Architectural Specification
- **Input Representation:** Pretrained latent vector $\mathbf{z} \in \mathbb{R}^{768}$ (from Module 3) concatenated with the binary lead-availability mask $\mathbf{m} \in \{0, 1\}^{12}$ (from Module 2).
  $$\mathbf{x}_{\text{in}} = [\mathbf{z} \,\|\, \mathbf{m}] \in \mathbb{R}^{780}$$
- **Hidden Layers:**
  - Dense Layer 1: $780 \rightarrow 384$ + LayerNorm + ReLU + Dropout ($p = 0.3$)
  - Dense Layer 2: $384 \rightarrow 192$ + LayerNorm + ReLU + Dropout ($p = 0.3$)
  - Output Projection: $192 \rightarrow 8$ raw diagnostic logits
- **Total Parameters:** Exactly 376,520 trainable parameters.
- **Output Target:** 8 canonical cardiovascular conditions (`NORM, AFIB, RBBB, LBBB, SBRAD, STACH, 1AVB, MI`).

```
Latent Vector z in R^768 (from Module 3)  ──┐
                                          ├──> Concatenation: x_in in R^780
Lead Availability Mask m in {0,1}^12     ──┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│ Dense Layer 1: 780 -> 384                                │
│ LayerNorm(384) -> ReLU -> Dropout(p=0.3)                 │
└──────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│ Dense Layer 2: 384 -> 192                                │
│ LayerNorm(192) -> ReLU -> Dropout(p=0.3)                 │
└──────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│ Diagnostic Classification Head: 192 -> 8 Logits          │
└──────────────────────────────────────────────────────────┘
       │
       ▼
Raw Output Logits (8 targets) ──> Masked BCE Loss / Sigmoid
```

---

## 2. Loss Function and Missing-Label Handling

### 2.1 Masked Binary Cross-Entropy Loss
Because cohorts in Module 1 have incomplete disease label availability (e.g., Chapman-Shaoxing does not annotate MI or 1AVB; PTB-XL has unannotated rhythms), treating unannotated labels as negative (0) would introduce severe label noise and distort classification boundaries.
Module 4 strictly enforces a masked loss formulation:
$$\mathcal{L}(\hat{\mathbf{y}}, \mathbf{y}, \mathbf{o}) = -\frac{1}{\sum_{d=1}^8 o_d} \sum_{d=1}^8 o_d \left[ y_d \log \sigma(\hat{y}_d) + (1 - y_d) \log (1 - \sigma(\hat{y}_d)) \right]$$
where:
- $\hat{y}_d$ is the predicted logit for disease $d$.
- $y_d \in \{0, 1\}$ is the ground truth label.
- $o_d \in \{0, 1\}$ is the observation mask from Module 1 (`disease_observation_mask.npy`).
- When $o_d = 0$ (unobserved), that disease contributes exactly zero gradient update.

---

## 3. Training Protocol and Optimization Details

- **Optimizer:** AdamW with weight decay $\lambda = 10^{-4}$.
- **Learning Rate Schedule:** Initial learning rate $\eta = 5 \times 10^{-4}$ with cosine annealing schedule decaying to $10^{-6}$ over 50 epochs.
- **Batch Size:** 64 records per batch, randomly sampled across all seven lead configurations in each training epoch.
- **Multi-Seed Protocol:** Evaluated across three independent random seeds:
  - Seed 42 (Primary model, SHA-256: `2920c76884e4a432...`)
  - Seed 123 (Replication checkpoint, SHA-256: `5e810f6ab12d7c49...`)
  - Seed 456 (Replication checkpoint, SHA-256: `a93b4822f7164931...`)
- **Checkpoint Selection:** Strictly evaluated after every epoch on the validation split (Folds 7–8, 3,688 records). The epoch achieving the highest validation macro-AUROC was saved as the best checkpoint.

---

## 4. Test Quarantine and Pre-Test Readiness Verification

Before any model was evaluated, a formal quarantine was instituted:
- Folds 9–10 (7,375 test records) were held in a cryptographically isolated partition.
- Independent pre-test audits (`stage6_pretest_independent_audit.md`) verified:
  1. Complete absence of test record IDs in training/validation loaders.
  2. Exact match of validation metrics across independent recomputations.
  3. Calibration parameter fitting strictly on validation predictions.

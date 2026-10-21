# Reasoning Failure Boundaries in ECG Foundation Models Under Progressive Physiological Information Loss

**Institution:** Department of Computer Science and Engineering, SSN College of Engineering  
**Academic Year:** 2026–2027  
**Status:** **Modules 1–3 Complete & Audited — Ready for Module 4**  

---

## 📖 Executive Summary

Deep learning and ECG Foundation Models demonstrate state-of-the-art diagnostic accuracy on standard 12-lead ECG recordings. However, in emergency triage, ambulatory monitoring, and wearable healthcare devices, clinical practitioners frequently encounter **progressive physiological information loss** (missing leads, sensor degradation, or constrained 6-, 4-, 2-, or single-lead configurations).

Conventional benchmark paradigms evaluate models primarily on diagnostic label correctness under fixed lead configurations, failing to assess whether the underlying **clinical reasoning** remains truthfully anchored in the physical signals actually observable by the sensors.

### Central Hypothesis
> *"A clinical diagnostic prediction is safe if and only if every physiological claim made in its reasoning trace can be validated against signals present in the available lead subset."*

---

## 🗂️ Repository Directory Structure

```
e:/ECG/
├── checkpoints/             # Model checkpoints (ECG-FM wav2vec2_cmsc, baseline adapter)
├── configs/                 # Authoritative configuration contracts (candidates, leads, input contracts)
│   ├── benchmark_candidates.json    # Frozen 8-disease benchmark definition
│   ├── lead_configurations.json     # Seven primary fixed lead configurations (12 -> 1)
│   └── module4_input_contract.json  # Formal input contract for Module 4 reasoning engine
├── data/                    # Dataset storage and processed artifacts
│   ├── benchmark_test/      # Frozen test benchmark arrays and masks (12, 10, 8, 6, 4, 2, 1 leads)
│   ├── external_validation/ # Downloaded external cohorts (Chapman-Shaoxing, MIMIC-IV-ECG 5k)
│   ├── latents/             # Latent representation cache and manifests
│   ├── observations/        # Structured clinical observations O_l (rhythm, HR, PR, QRS, ST, T)
│   ├── processed_100hz/     # Calibrated, filtered, and 100 Hz resampled signals
│   ├── ptbxl/               # Primary PTB-XL v1.0.3 dataset files
│   └── seeded_random_masked/# Legacy/ablation random masked data
├── labels/                  # Diagnostic label harmonization and mappings
│   ├── mappings/            # SCP-ECG and SNOMED-CT cross-dataset mapping tables
│   ├── disease_analysis.py  # Co-occurrence and distribution analysis
│   ├── harmonization.py     # Cross-cohort disease mapper
│   └── raw_label_extraction.py # Diagnostic statement extractors
├── report/                  # Academic project review reports (LaTeX source & PDF deliverables)
├── reports/                 # 24 formal machine-readable JSON & MD audit/verification artifacts
├── scripts/                 # Sequential pipeline execution scripts (01-13) and audit runners
│   └── README.md            # Comprehensive catalog of all 23 scripts
├── src/                     # Core reusable library packages
│   ├── degradation/         # Lead availability masking and Cabrera DAG provenance tracking
│   ├── models/              # FrozenECGFM wrapper, backbone definitions, and adapters
│   ├── observations/        # Deterministic clinical observation extraction (O_l)
│   └── preprocessing/       # 5-stage standardized preprocessing pipeline (P1 to P5)
├── tests/                   # Automated pytest suite (47 unit & integration tests)
├── visualizations/          # Generated distribution charts and phase dashboards
├── FYP Material/            # Project presentation slides, architectural diagrams, and assets
├── MODULE_1_TO_3_FINAL_AUDIT.md # Formal audit verdict across Modules 1 to 3
├── MODULE_3_STATUS.md       # Primary route status report for Module 3
├── PROJECT_PHASES_SPECIFICATION.md # Full 6-phase epistemological and theoretical specification
└── README.md                # Repository index and navigational guide
```

---

## 🚦 Phase Roadmap & Current Status

| Phase | Description | Status | Key Deliverable |
| :---: | :--- | :---: | :--- |
| **Phase 1** | Data Standardization & Cohort Preparation | **COMPLETED** | PTB-XL (21,799), Chapman (10,158), MIMIC (5,000) calibrated in mV, 100 Hz polyphase resampled. |
| **Phase 2** | Controlled Degradation & Lead Availability | **COMPLETED** | 7 fixed levels ($12 \to 10 \to 8 \to 6 \to 4 \to 2 \to 1$), zero-leakage masking, Cabrera DAG provenance. |
| **Phase 3** | Foundation Model Adaptation & Observations | **COMPLETED** | Frozen ECG-FM (`wav2vec2_cmsc`, 90.8M params) + deterministic observations ($O_l$). |
| **Phase 4** | Dual-Path Clinical Reasoning & Trace Generation | **READY TO START** | Control Path (ECG-Only) vs. Experimental Path (ECG + CKP) inside Reasoning Transformer. |
| **Phase 5** | Grounding Verification Logic Gate | *Planned* | Ternary logic gate (`SUPPORTED`, `CONTRADICTED`, `UNAVAILABLE`) evaluating claims against lead mask. |
| **Phase 6** | Degradation Analysis & Failure Boundary | *Planned* | Quantitative grounding metrics ($EO_l, GQ_l, TGR_l$), Reasoning Degradation Curves, and RFB derivation. |

---

## 🧪 Testing & Verification

All automated tests can be executed via:

```powershell
pytest tests/ -v
```

The test suite covers:
- `test_preprocessing.py`: Unit calibration, filtering, lead ordering, polyphase resampling, and amplitude clamping.
- `test_lead_masking.py`: Masking integrity, zero leakage, and Cabrera lead provenance DAG.
- `test_observations.py`: Deterministic interval and morphological feature extraction under lead degradation.
- `test_adapter.py`: Model wrappers, embedding dimensions ($Z_{\text{ECG}} \in \mathbb{R}^{768}$), and frozen weights.

**Current Test Status:** `47 / 47 passing (100%)`.

---

## 🔑 Key Invariants & Architectural Rules

1. **8-Disease Multi-Label Benchmark:** Strictly frozen across Normal ECG, AFib, RBBB, LBBB, Sinus Bradycardia, Sinus Tachycardia, 1st Degree AV Block, and Myocardial Infarction. Multi-hot multi-label format preserved.
2. **Frozen Primary Foundation Model:** ECG-FM weights are strictly frozen (`requires_grad = False`). No trainable representation adapters in the primary path.
3. **No Generative Imputation:** Missing channels are zero-masked (`is_reconstructed == False`). No autoencoding or synthetic waveform reconstruction is permitted.
4. **Lead Provenance Constraint:** Derived leads ($\text{III}, \text{aVR}, \text{aVL}, \text{aVF}$) cannot count as independent evidence if parent leads ($\text{I}, \text{II}$) are missing or already accounted for.
5. **Ternary Evidence Logic:** Claims referencing unobservable leads are categorized strictly as `UNAVAILABLE`, never conflated with `CONTRADICTED` or termed "hallucination".

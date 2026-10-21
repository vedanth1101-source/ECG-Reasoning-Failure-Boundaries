# PROJECT CONTEXT — Verified Repository Audit for the First-Half Implementation Report

**Project:** Reasoning Failure Boundaries in ECG Foundation Models Under Progressive Physiological Information Loss
**Department / Institution:** Computer Science and Engineering, SSN College of Engineering — Academic year 2026–2027
**Purpose of this document:** Evidence-checked working context for the later chapter-writing prompts. It is not a report chapter.
**Audit date:** 2026-10-19 (system date checked). Repository root: `E:\ECG` (not a git repository, so there is no commit history to anchor versions).
**Supersedes:** the earlier `PROJECT_CONTEXT.md` in this folder (written 2026-10-19 18:08, before this audit). Section L lists the claims in that version that this audit could not support.

---

## 0. How to read this document

Every important claim carries one evidence label:

| Label | Meaning |
|---|---|
| **Verified** | Checked this session against source code, data, a checkpoint, a recomputation, or a command I ran and read. |
| **Partially verified** | Some supporting evidence exists, but not for the full claim. |
| **Documented only** | Stated in a project report/JSON; no independent implementation or recomputation was done in this audit. |
| **Planned** | Described as future work, or not implemented. |
| **Unverified** | Not enough evidence either way. |
| **Contradictory** | Sources disagree. |

Commands run this session (all read-only on project data except a temp script under `%TEMP%`):

- `python -m pytest -q -p no:cacheprovider` (Python 3.11.4, torch 2.14.0+cpu) → **108 collected, 107 passed, 1 skipped** (skip reason: `tests/test_preprocessing.py:218` "Local PTB-XL dataset directory not found").
- Recomputation of macro-AUROC/F1 from `runs/module4_v3/metrics/test_predictions_seed42.csv` (12-, 6-, 4-, 1-lead) → matches `test_evaluation_summary.json` to 4 decimals.
- Recount of `runs/module4_v3/split_manifest.csv` (records, splits, folds, patients, leakage).
- Memory-mapped shape/dtype checks of `data/processed_100hz/**/signals.npy` and `labels.npy`.
- `torch.load` inspection of the ECG-FM checkpoint and all reasoning checkpoints (parameter counts, layer counts, embedded metadata).
- SHA-256 re-hash of every file listed in `runs/module4_v3/MODULE4_FROZEN.json`.
- `nvidia-smi` (hardware present now).

Nothing was trained, downloaded, or re-evaluated.

---

## A. Project Overview

### A.1 Research problem (from project documents; framing, not a result)

The project asks whether an ECG model that keeps a high diagnostic score as leads are removed is still basing its predictions on physiological evidence that is actually observable in the remaining leads. The hypothesis is that diagnostic accuracy can stay high after the evidence a clinician would need (for example, ST elevation in contiguous anterior leads for MI, or V1/V6 morphology for bundle-branch block) has been physically removed. The project calls the point where reasoning stops being grounded the **Reasoning Failure Boundary (RFB)**. Source: `reports/module4_evaluation_protocol.md` §1, §5; `PROJECT_PHASES_SPECIFICATION.md`.

The protocol itself says (`reports/module4_evaluation_protocol.md` §5.3): "The hypothesis that Δ_decoupling > 0 under physiological information loss is an empirical hypothesis to be tested, not a presumed fact." The report must keep this framing.

### A.2 Intended contributions (status in brackets)

1. A fixed seven-level, non-generative lead-removal ladder (12→10→8→6→4→2→1) with zero-leakage masking. **[Verified — implemented and tested]**
2. A multi-cohort, patient-disjoint benchmark of 8 conditions across PTB-XL, Chapman-Shaoxing and a MIMIC-IV-ECG subset. **[Verified — implemented]**
3. Frozen ECG-FM representations for all seven configurations, with representation-drift measurement. **[Verified/Documented]**
4. A "Control" reasoning transformer (ECG-FM latent + rule-based observations + lead-availability tokens → 8 posteriors + cross-attention "traces"). **[Verified — trained, evaluated]**
5. An "Experimental" path that adds a SNOMED-CT Clinical Knowledge Prior (CKP). **[Contradictory/Preliminary — see G.5; no CKP code remains in the repository]**
6. A ternary grounding gate (`SUPPORTED` / `CONTRADICTED` / `UNAVAILABLE`) and grounding metrics (EO, GQ, TGR, UR, GSR, AURDC, RFB, DFB, Δ). **[Planned — specified and pre-registered, not implemented in `src/`]**

### A.3 Scope of the first-half report

Modules 1–3 are frozen. Module 4 (Control path only) is trained and has produced test-set diagnostic metrics, but with a provenance caveat (H.5). Phases 5–6 (grounding gate, RFB derivation) and the CKP comparison have no results. The report therefore covers infrastructure, representation analysis, and the diagnostic degradation curve. It cannot claim an RFB value or any CKP effect.

---

## B. Repository Map

Paths are relative to the repository root. File and line counts were checked this session.

| Path | Purpose | Notes |
|---|---|---|
| `src/preprocessing/` | P1–P5 standardisation: `unit_calibration.py`, `lead_ordering.py`, `filtering.py`, `sampling_standardization.py`, `quality_screening.py`, `pipeline.py`, `dataset_loader.py` | 7 modules |
| `src/degradation/lead_masking.py` | Masks, schedules, **and** lead provenance (`LEAD_PROVENANCE`, `get_lead_observability_status`) | There is **no** `lead_provenance.py`; the old context document listed one |
| `src/models/` | `ecg_fm_primary.py` (FrozenECGFM), `adapter.py`, `backbone_wrapper.py`, `adapted_ecg_model.py`, `dataset.py` | `resnet1d.py` is **not** in `src/models/` (old context listed it) |
| `src/models/fairseq_signals/` | Vendored third-party `fairseq-signals` library (MIT header), 146 `.py` files | Not project-authored; exclude from LOC claims |
| `src/observations/clinical_extractor.py` | Deterministic HR / rhythm / PR / QRS / ST / T extraction | QRS window defect, see G.1 |
| `src/reasoning/` | `context_builder.py`, `reasoning_transformer.py`, `trace_generator.py`, `schema_validator.py`, `clinical_taxonomy.py`, `observation_extractor.py`, `multi_cohort_loader.py`, `dataset.py`, `freeze_contract.py` | Control path only; CKP rejected by design |
| `configs/` | `lead_configurations.json`, `benchmark_candidates.json`, `control_path_config.json`, `control_path_training_config.json`, `module4_input_contract.json` | Hash-frozen in `MODULE4_FROZEN.json` |
| `scripts/` | 39 `.py` (numbered pipeline 01–18, audits, demo/visual generators) + `README.md` | Several demo scripts print **hard-coded illustrative numbers** (G.6) |
| `tests/` | 11 test files, 108 tests | Verified run this session |
| `labels/` | `raw_label_extraction.py`, `harmonization.py`, `disease_analysis.py`, `mappings/` | Early (v1) label work |
| `data/processed_100hz/` | Preprocessed arrays: PTB-XL (root), `chapman/`, `mimic/` + CSV databases, summaries | v1 processing outputs (Sep 2026) |
| `data/benchmark_test/` | Older frozen test benchmark (per-cohort masked arrays, incl. `leads_6_precordial`) | Pre-Module-4 artifact |
| `data/chapman/`, `data/external_validation/` | Raw Chapman WFDB `.hea/.mat` | 20,388 files in `data/chapman/` (≈10,194 record pairs) |
| `data/ptbxl/` | Contains only an empty nested `data/ptbxl` | **Raw PTB-XL is not on disk** (hence the skipped test) |
| `data/mimic/` | `machine_measurements.csv` + data dictionary only | Raw MIMIC waveforms not in this folder |
| `runs/module1_v2/` | Frozen Module 1: manifests, tri-state labels, splits, per-record dataset | `MODULE1_FROZEN.json` |
| `runs/module2_v1/` | Frozen Module 2: masked arrays, audits | `MODULE2_FROZEN.json` |
| `runs/module3_v1/` | Frozen Module 3: 768-d representations per configuration, drift reports | `MODULE3_FROZEN.json` |
| `runs/module4_v3/` | Module 4 Control path: split manifest, checkpoints (3 seeds), metrics, observations, validation traces, audit reports | `MODULE4_FROZEN.json` |
| `checkpoints/` | ECG-FM checkpoint + YAML; older reasoning checkpoints; baseline adapter; linear observation heads | See E.3 |
| `reports/` | 45 audit/protocol artifacts (JSON/MD) | Includes the pre-registered protocol |
| `visualizations/` | 22 PNG + 4 PDF | Some are concept/roadmap graphics, not results |
| `presentation/` | Demo cockpit HTML, slides, speaker cue cards | Not evidence |
| `report/` | Existing report: `00_FRONTMATTER.md` … `08_APPENDIX.md`, `r2_reviewreport.tex` (+ backup), `antigravity report.pdf` | Review-2 report; do not modify |
| `report/first_half_implementation/` | This workspace: `PROJECT_CONTEXT.md`, plus `01_INTRODUCTION.md` from an earlier session | That intro was written from the superseded context and needs re-checking (L, item 24) |
| Root files | `PROJECT_PHASES_SPECIFICATION.md`, `MODULE_1_TO_3_FINAL_AUDIT.md`, `MODULE_3_STATUS.md`, `README.md`, `run_audit.py`, scratch txt files, loose PNGs | Root PNGs `01_dashboard_main.png` … `06_redoc_documentation.png` show a "customer"/Swagger dashboard unrelated to this project |
| Outside the repo | `C:\Users\VEDANTH\Downloads\UCS2717-ProjectReport-LaTeX\…\ug-project-report.{pdf,tex}` (sample thesis); `C:\Users\VEDANTH\Downloads\project_report.md` | Neither file is in the repository; both were found and read |

**Code size (Verified, this session):** 250 `.py` files / 58,114 lines in total, **including** the vendored `fairseq_signals` (146 files). Excluding it: 104 `.py` files / 30,205 lines (`src`, `scripts`, `tests`, `labels`, `runs/*/scripts`, root scripts). Project-authored `src/` alone is 27 files / 5,388 lines. Older figures ("225 files, ~43,335 lines", "38 scripts") in `project_report.md` do not match the current tree.

---

## C. Verified Implementation Inventory

| # | Subsystem | Purpose | Relevant files | Evidence status | Completion | Validation evidence | Limitations & dependencies | First-half classification |
|---|---|---|---|---|---|---|---|---|
| 1 | Preprocessing (P1–P5) | ADC→mV, canonical lead order, 0.5 Hz HP (order 3) + 50/60 Hz notch (Q=30) + 45 Hz LP (order 4) at native 500 Hz via `filtfilt`, `resample_poly` 500→100 Hz, centre 10 s window, QC | `src/preprocessing/*.py`; `runs/module1_v2/configs/module1_config.json` | **Verified** (code + config) | Complete, frozen | `tests/test_preprocessing.py` passing (1 skipped, no raw PTB-XL); `reports/preprocessing_audit.json`, `adc_conversion_audit.json`, `clamp_audit.json` (documented only) | ±15 mV clamp is applied in calibration (`np.clip`) **and** QC flags saturation; flatline threshold is **1.5 s** (old context said 1.0 s); 69 records rejected | Completed work |
| 2 | Lead masking | X ⊙ m_l, 7 fixed configs from JSON, zero imputation, `is_reconstructed=False` | `src/degradation/lead_masking.py`; `configs/lead_configurations.json`; `runs/module2_v1/` | **Verified** | Complete, frozen | `tests/test_lead_masking.py` passing; `reports/masking_integrity_report.json` (max inactive leakage 0.0; documented) | Masked leads are zeros, which the downstream model cannot tell apart from a flat signal except through mask tokens | Completed work |
| 3 | Lead provenance | Marks III, aVR, aVL, aVF as derived from I and II; status `observed_measured` / `observed_derived` / `unavailable_derived` / `masked_lead` | `lead_masking.py:38–174` | **Verified** (code) | Implemented as a lookup table, not a graph library | `tests/test_lead_masking.py`; `reports/lead_provenance_audit.json` (documented) | No configuration contains a derived lead without both parents, so `unavailable_derived` never occurs in the 7-level ladder (inferred from config) | Completed (supporting) |
| 4 | Label harmonisation v2 | Tri-state labels (1 / 0 / unknown→masked) per cohort | `runs/module1_v2/labels/label_mapping_rules.json`, `label_observation_mask.csv`, `scripts/harmonize_labels.py` | **Verified** (rules read; masks present in manifest) | Complete, frozen | Manifest has NaN for unknown labels (G/H) | MIMIC labels come from **machine report text**, not cardiologists; Chapman has no patient IDs | Completed work |
| 5 | Patient-disjoint splits | 10 folds, iterative multilabel stratification 70/10/20: folds 1–7 train, 8 val, 9–10 test | `runs/module4_v3/split_manifest.csv` (= `splits/split_manifest.csv`, identical hash) | **Verified** (recount) | Complete | 0 patient overlap train/val/test (recount) | Chapman patient = record ID; 13 identical Chapman signal pairs (9 cross-split) per `MODULE1_FROZEN.json` (documented) | Completed work |
| 6 | Frozen ECG-FM encoder | `wav2vec2_cmsc`, mean-pooled final encoder output → Z_ECG ∈ ℝ⁷⁶⁸ | `src/models/ecg_fm_primary.py`; `checkpoints/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt` | **Verified** (checkpoint inspected) | Complete, frozen | 90,883,072 params, 12 layers; strict loading; `tests/test_adapter.py` | Fed 100 Hz / 1,000-sample input; the pretraining sampling rate was not confirmed (L, item 12); the YAML disagrees with the checkpoint | Completed work |
| 7 | Representation extraction & drift | 7 × 36,888 = 258,216 vectors; cosine/L2 drift vs 12-lead | `runs/module3_v1/` | **Partially verified** (summary CSV read; file count documented only) | Complete, frozen | `module3_cosine_similarity_summary.csv`; `MODULE3_FROZEN.json` | Drift is not diagnostic accuracy, and it is not monotone (4-lead > 6-lead) | Completed work |
| 8 | Clinical observation extractor | HR (Pan–Tompkins-style), rhythm, PR, QRS, ST elev/dep, T inversion on active leads | `src/observations/clinical_extractor.py` | **Verified** (code) / validation **documented only** | Implemented | `reports/clinical_observation_validation.json` (250 MIMIC records vs machine measurements) | **QRS cannot exceed 110 ms** (±60 ms window at 100 Hz) → QRS ≥ 120 ms is never observable; PR/QRS are heuristic | Preliminary (validated for HR only) |
| 9 | Control Reasoning Transformer | 51 evidence tokens (1 ECG-FM + 12 lead-availability + rhythm + intervals + 3×12 ST/T) → 2-layer encoder → 8 candidate queries cross-attend → sigmoid head | `src/reasoning/context_builder.py`, `reasoning_transformer.py`; `scripts/16_train_reasoning_transformer.py` | **Verified** (code + checkpoint) | Trained (3 seeds), frozen | `tests/test_control_reasoning.py`, `test_control_training.py`; training histories; 1,830,657 trainable params | Rejects any CKP input (`ValueError`); one shared model trained on all 7 configurations | Completed work |
| 10 | Calibration | Per-class temperature scaling fitted on validation | `scripts/16_…`; `control_training_history_seed42.json` | **Verified** (artifact) | Done | Pre/post ECE 0.01498 → 0.01043 (validation, seed 42) | Validation-only fit, as specified | Completed work |
| 11 | Test-set evaluation | 7,375 records × 7 configs = 51,625 evaluations | `scripts/17_generate_reasoning_traces.py`; `runs/module4_v3/metrics/test_*` | **Verified** (recomputed) | Done, but **provenance AUTHORIZATION-UNVERIFIED** | Recomputation matches the JSON | Pre-registered "RUN_TEST" gate was not satisfied; the run crashed once and was relaunched (H.5) | Preliminary findings (with caveat) |
| 12 | Structured traces | JSON dossiers: posteriors, top hypothesis, top-attention tokens as "supporting evidence", observability flags | `src/reasoning/trace_generator.py`, `schema_validator.py`; `scripts/18_…` | **Verified** (counts) | 25,816 validation traces; 700 sample test traces | `runs/module4_v3/validation_traces/*.json` (7 × 3,688); `reports/sample_test_reasoning_traces.json` (100/config) | "Evidence" means attention-selected tokens, not clinical claims; the quality-classification heuristic was found asymmetric and later corrected in docs (G.4) | Preliminary |
| 13 | Baselines | B-1 linear probe on Z_ECG; B-2 rule engine | `runs/module4_v3/metrics/comprehensive_baseline_comparison.json` | **Partially verified** (JSON values) | B-1 descriptive only; B-2 **withdrawn** | `research_claims_correction.md` | B-1 per-record predictions not saved; baseline script lives in an IDE scratch folder **outside the repo** | Preliminary (B-1); do not report B-2 |
| 14 | CKP / Experimental path | SNOMED-CT prior tokens | `checkpoints/reasoning_transformer_experimental.pt` only | **Contradictory** | No code in repo | Checkpoint metadata: 500 train / 150 val PTB-XL samples, val AUROC 0.7493 (control pilot 0.7522) | Encoder code and `test_ckp_builder.py`, `test_dual_path_reasoning.py` existed (pytest cache lists them) but have since been **removed** | Future work |
| 15 | Grounding gate | SUPPORTED / CONTRADICTED / UNAVAILABLE per claim | `reports/evidence_observability_config.json`; protocol §3 | **Planned** (spec only) | Not implemented in `src/` or `scripts/` (grep) | — | Only demo/visual scripts print these labels, with hard-coded numbers | Future work |
| 16 | Grounding metrics & RFB | EO, GQ, TGR, UR, GSR, AURDC, AUTDC, RFB, DFB, Δ | `reports/module4_evaluation_protocol.md` §4–§7 | **Planned** (pre-registered) | Not implemented | — | Depends on 15 | Future work |
| 17 | Legacy ResNet-1D baseline adapter | Early task-trained baseline | `checkpoints/baseline_resnet1d_adapter.pt`; `src/models/adapter.py` | **Verified** (checkpoint metadata) | 1 epoch, val AUROC 0.7628 (CPU) | `reports/baseline_training_metrics.json` | Superseded; old drift numbers came from this model | Exclude, or mention as history |

---

## D. Dataset and Preprocessing Inventory

### D.1 Sources

| Cohort | Version (documented) | Native format (documented) | Role (project) | Labels source | Access | On disk |
|---|---|---|---|---|---|---|
| PTB-XL | v1.0.3 (`preprocessing_summary.json`) | 500 Hz / 100 Hz WFDB, 1000 ADC/mV | Train/val/test pool | SCP-ECG statements (cardiologist-annotated, with likelihoods) | PhysioNet, open (licence should be stated from the PhysioNet page in the literature stage; **Unverified** here) | Processed arrays only; **raw not present** |
| Chapman-Shaoxing | PhysioNet/Computing in Cardiology **Challenge 2021 v1.0.3**, `training/chapman_shaoxing/` (verified 2026-10-19: `scripts/download_validation_data.py:24`; `data/external_validation/chapman/records_manifest.csv`; local `JS00001.hea/.mat` byte-identical (SHA-256) to the Challenge 2021 v1.0.3 files). *Erratum: this row previously read "PhysioNet/CinC 2020" after `dataset_sampling_audit.json`. That is incorrect, because the Challenge 2020 v1.0.2 training set contains no Chapman-Shaoxing directory.* | 500 Hz `.mat/.hea`, SNOMED codes in `# Dx` | Train/val/test pool | SNOMED-CT codes | PhysioNet, open (**Unverified** licence) | Raw `.hea/.mat` (~10,194 pairs) + processed |
| MIMIC-IV-ECG | v1.0 (documented) | 500 Hz, 200 ADC/mV (documented) | Train/val/test pool (balanced 5,000 subset, seed 42) | **Machine-generated report statements** (`report_0…17`) | PhysioNet; access terms to be stated from source (**Unverified**) | Processed arrays + `machine_measurements.csv` only |

All three cohorts are pooled and split by fold. None is held out as a purely "external" cohort in Module 4, although several older documents call Chapman "external validation" and MIMIC "external evaluation" (**Contradictory** with the Module 4 manifest, where all three appear in train, val and test).

### D.2 Record accounting

| Stage | PTB-XL | Chapman | MIMIC | Total | Status / evidence |
|---|---|---|---|---|---|
| Catalog universe | — | — | — | 37,046 | Documented (`MODULE1_FROZEN.json`) |
| Missing Chapman files | — | 89 | — | 89 | Documented |
| Usable paired inventory = processed arrays | 21,799 | 10,158 | 5,000 | **36,957** | **Verified** (array shapes `(N,12,1000)` float32) |
| Rejected by P5 QC | 2 | 16 | 51 | 69 | Documented (`*/preprocessing_summary.json`); consistent with the manifest recount |
| **Eligible modelling records (Module 1 v2 → Modules 2–4)** | **21,797** | **10,142** | **4,949** | **36,888** | **Verified** (manifest recount) |

MIMIC `signals.npy` still contains 46 records with non-finite values. These are among the rejected records and are absent from the manifest (**Verified** that they exist in the array; exclusion inferred from counts).

Chapman source counts disagree across documents: 10,247 (`reports/dataset_inventory.md`), 10,249 (`dataset_sampling_audit.json`), 10,158 processed. The Chapman-Shaoxing subset of Challenge 2021 v1.0.3 is the source (see the D.1 erratum); its catalogue size of 10,247 is recorded in `data/external_validation/chapman/records_manifest.csv` per the Module 1 source audit. *Erratum 2026-10-19: this sentence previously referred to a "CinC-2020 Chapman subset", which does not exist in the public Challenge 2020 release.* `reports/dataset_sampling_audit.json:51` still says "PhysioNet/CinC 2020". It is an audit artefact and was not edited; treat that field as incorrect. `dataset_inventory.md` says 3,987 files were "downloaded locally", but `data/chapman/` holds about 10,194 pairs (**Contradictory**; that document is out of date).

MIMIC pool size: 800,035 records / 161,352 subjects in `dataset_sampling_audit.json` vs "~800,000 / ~160,000" in `dataset_inventory.md` (**Documented only**).

### D.3 Splits (Module 4 manifest — **Verified** by recount)

| Split | Folds | Records | Unique patients (source_dataset + patient_id) | PTB-XL / Chapman / MIMIC (test) |
|---|---|---|---|---|
| Train | 1–7 | 25,825 | 23,818 | — |
| Validation | 8 | 3,688 | 3,394 | — |
| Test | 9–10 | 7,375 | 6,746 | 4,358 / 2,028 / 989 |
| **Total** | | **36,888** | **33,958** | |

Fold sizes are 3,687–3,690. Patient overlap between all split pairs is 0. Per-cohort unique patients: PTB-XL 18,867, Chapman 10,142, MIMIC 4,949. Test labels are blanked (NaN) in the manifest as a quarantine measure. Test ground truth is only in the prediction CSV.

An **older split scheme** also exists: PTB-XL native folds 1–8/9/10 in `data/processed_100hz/splits_summary.json` (train 17,418 / val 2,183 / test 2,198), and `project_report.md` gives 17,441 / 2,183 / 2,175. These belong to Module 1 v1 and are **not** the splits behind the Module 4 results (**Contradictory** if presented as current).

### D.4 Labels and distributions

Benchmark (`configs/benchmark_candidates.json`, **Verified**): NORM 426783006, AFIB 164889003, RBBB 59118001, LBBB 164909002, SBRAD 426177001, STACH 427084000, 1AVB 270492004, MI 164865005. Multi-label, independent sigmoid scoring, "retrieval filtering" by CKP prohibited. Whether these SNOMED IDs are the correct concept codes should be checked against SNOMED CT during the literature stage (**Unverified**).

Tri-state labelling (Module 1 v2, **Verified** from rules file): 1 = positive by explicit evidence; 0 = defensible negative; unknown = masked out of loss and metrics. Examples: incomplete RBBB/IVCD → RBBB unknown; atrial flutter → AFIB unknown.

Positive counts per split in the Module 4 data (**Verified**, positives / observed labels):

| Class | Train | Validation | Test |
|---|---|---|---|
| NORM | 7,889 / 25,730 | 1,128 / 3,675 | 2,225 / 7,347 |
| AFIB | 2,716 / 25,212 | 389 / 3,610 | 788 / 7,188 |
| RBBB | 1,177 / 24,094 | 167 / 3,416 | 336 / 6,870 |
| LBBB | 963 / 24,797 | 135 / 3,536 | 274 / 7,089 |
| SBRAD | 3,701 / 21,982 | 528 / 3,132 | 1,056 / 6,240 |
| STACH | 2,156 / 21,980 | 304 / 3,133 | 622 / 6,237 |
| 1AVB | 995 / 22,414 | 141 / 3,201 | 286 / 6,368 |
| MI | 4,097 / 22,712 | 584 / 3,255 | 1,196 / 6,517 |

Per-class test metrics therefore use **different denominators** (6,237–7,347 records), and this has to be stated next to every per-class table.

Older label versions (do **not** present as current):
- v1 processed `labels.npy` (36,957 records, binary): positives NORM 12,738, AFIB 3,905 pooled, …; 0/1/≥2-label records 1,746 / 31,874 / 3,337. **Verified** for the v1 arrays. These are the numbers the superseded context presented as "the benchmark", but they are not the labels used in Module 4.
- `dataset_sampling_audit.json` PTB-XL: NORM 9,514, RBBB 541, MI 5,467 (another mapping; documented only).
- `MODULE1_FROZEN.json` prevalence over observed labels: NORM 30.59 %, AFIB 10.81 %, RBBB 4.89 %, LBBB 3.87 %, SBRAD 16.86 %, STACH 9.83 %, 1AVB 4.45 %, MI 18.09 % (documented).
- The MIMIC subset was sampled as 625 per class with single labels in v1, but v2 re-derived multi-label tri-state labels from report text, so MIMIC is no longer exactly balanced in Module 4 (**Verified**: v2 rows with two positives exist).

### D.5 Preprocessing parameters (**Verified** from `module1_config.json` and code)

| Stage | Parameter |
|---|---|
| P3 calibration | ADC → mV using header gain/baseline; clip to [−15, +15] mV |
| P1 ordering | I, II, III, aVR, aVL, aVF, V1–V6 (MIMIC aVF/aVL swap corrected; documented) |
| P4 filtering (at native 500 Hz) | Butterworth HP 0.5 Hz order 3; IIR notch Q=30 at 50 Hz (PTB-XL, Chapman) / 60 Hz (MIMIC); Butterworth LP 45 Hz order 4; all zero-phase `filtfilt` |
| P2 resampling | `scipy.signal.resample_poly(up=1, down=5)`, 500→100 Hz; centre 10 s window → 1,000 samples |
| P5 QC | non-finite, flat channel (std < 1e-4 mV), flatline run ≥ 1.5 s, saturation beyond ±15 mV |
| Units / dtype | mV, float32, shape (12, 1000) |
| Scaling | `scaling_method="none"` (no per-lead z-scoring) |

**Terminology issue:** the code and documents call this order the "Cabrera sequence". The Cabrera display order is aVL, I, −aVR, II, aVF, III; what is implemented is the **standard** order. Use "standard 12-lead order" in the report (**Contradictory** terminology; recommended correction).

---

## E. Model and Architecture Inventory

### E.1 ECG-FM (perception backbone) — **Verified** from the checkpoint

| Property | Value | Evidence |
|---|---|---|
| Architecture | `wav2vec2_cmsc` (fairseq-signals) | `ckpt['cfg']['model']['_name']` |
| Encoder | 12 layers, d=768, 12 heads, FFN 3,072 | checkpoint cfg + key scan (`encoder.layers.0…11`) |
| Conv feature extractor | `[(256, 2, 2)] * 4`, in_d = 12 | checkpoint cfg |
| Parameters | 90,883,072 (all frozen in pipeline) | summed state dict; `requires_grad=False` in `ecg_fm_primary.py:83` |
| Checkpoint SHA-256 | `4d0142bc…9b7e`, matches `MODULE4_FROZEN.json` | re-hash this session |
| Input used by project | (B, 12, 1000), mV, 100 Hz, 10 s, no normalisation | `MODULE3_FROZEN.json`; `ecg_fm_primary.py` |
| Output | mean over time of final encoder features → 768-d | `ecg_fm_primary.py:149–153` |
| Reference | McKeen et al., "ECG-FM: An Open Electrocardiogram Foundation Model", JAMIA Open 8(5) ooaf122, 2025 (as recorded in `MODULE3_FROZEN.json`) | bibliographic details **Unverified** |

**Discrepancy:** `checkpoints/ecg_fm/mimic_iv_ecg_physionet_pretrained.yaml` says `encoder_layers: 24`, `embed_dim 1024`, `heads 16`, `ffn 4096` (a larger configuration) and records pretraining with `random_leads_masking`, `mask_leads_prob 0.5`. The binary checkpoint is the ground truth for architecture. Whether the YAML's lead-masking pretraining settings apply to this checkpoint is **Unverified**. If they do, it matters for interpretation, because the backbone would have been pretrained with random lead dropout.

### E.2 Control Reasoning Transformer — **Verified**

| Property | Value |
|---|---|
| Inputs | Z_ECG (768) projected to 256; observation tokens; 12 lead-availability tokens; 8 learnable candidate queries |
| Evidence tokens | 51 = 1 ECG + 12 LEAD_OBS + 1 RHYTHM + 1 INTERVALS + 12 ST_ELEV + 12 ST_DEP + 12 T_INV (`context_builder.py:252–278`) |
| Encoder | `nn.TransformerEncoder`, 2 layers, d_model 256, 4 heads, FFN 512, GELU, pre-norm, dropout 0.1 |
| Reasoning step | Multi-head cross-attention (queries = 8 candidates, keys/values = encoded evidence), residual + FFN |
| Output | 8 logits → sigmoid; averaged cross-attention weights (8 × 51) |
| Trainable parameters | 1,830,657 (checkpoint sum; matches config) |
| Loss | Masked `BCEWithLogitsLoss` over observed labels (`scripts/16_…:63–77`) |
| Optimiser | AdamW, lr 1e-4, wd 1e-4, batch 32, cosine schedule with 2 warm-up epochs, max 20 epochs, early stop on val BCE (patience 5, Δ 5e-4) |
| Data regime | One shared model trained on all 7 configurations (180,775 examples/epoch); no class balancing |
| Seeds | 42 (primary), 1337, 2024 |
| Seed-42 run | best epoch 3, best val loss 0.15925, train time 5,799 s (documented in history JSON) |
| Primary checkpoint | `runs/module4_v3/checkpoints/control_reasoning_best_seed42.pt`, SHA-256 `2920c768…7d93` (re-hash matches) |

**Contradiction:** `research_claims_correction.md` §3.3 says B-3 used "Focal loss (γ=2.0)". The training script and config use masked BCE. Treat focal loss as an error in that document.

### E.3 Checkpoint inventory (**Verified** by `torch.load`)

| File | Params | Embedded metadata | Use in report |
|---|---|---|---|
| `checkpoints/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt` (1,090,825,421 B) | 90.9 M | official pretrained | backbone |
| `runs/module4_v3/checkpoints/control_reasoning_best_seed{42,1337,2024}.pt` | 1.83 M | seed-42: epoch 3, val AUROC 0.9452 | primary results |
| `runs/module4_v3/checkpoints/control_reasoning_final_seed*.pt` | 1.83 M | final weights | supporting |
| `runs/module4_v3/checkpoints/control_reasoning_transformer_best.pt` | — | legacy alias | — |
| `checkpoints/control_reasoning_transformer_best.pt` / `_final.pt` | 1.83 M | earlier run (config v1.0, 2026-10-10): val AUROC 0.9242, epoch 10 / final epoch 15 | superseded |
| `checkpoints/reasoning_transformer_control.pt` | 1.90 M | `mode: control`, 500/150 PTB-XL samples, val AUROC 0.7522 | pilot only |
| `checkpoints/reasoning_transformer_experimental.pt` | 1.90 M | `mode: experimental`, includes `ckp_encoder.*` weights, 500/150 samples, val AUROC 0.7493 | pilot only, no code |
| `checkpoints/linear_observation_heads.pt` | 37,681 | probe accuracies (rhythm 0.85, rate 0.84, PR 0.877, QRS 1.0, ST 0.643) | latent-probing side experiment (`reports/latent_observation_probing_results.json`) |
| `checkpoints/baseline_resnet1d_adapter.pt` | — | 1 epoch, val AUROC 0.7628 | legacy |

**Hardware:** `nvidia-smi` shows an NVIDIA GeForce RTX 4050 Laptop GPU (6 GB) on this machine today. The only Python with torch is 3.11.4 with **torch 2.14.0+cpu**, and no artifact records the training device. Whether Module 4 trained on GPU or CPU is **Unverified**. There is no `requirements.txt`/environment file. Library versions present today: numpy 2.4.2, scipy 1.17.1, scikit-learn 1.8.0, pandas 2.3.3, wfdb 4.3.1, neurokit2 0.2.13, omegaconf 2.3.1, hydra 1.3.7, matplotlib 3.10.8. These are today's versions, not necessarily the versions used for training.

---

## F. Lead-Degradation Protocol

### F.1 Configurations (**Verified**, `configs/lead_configurations.json`; used by `get_stepwise_schedule()`)

| Level | ID | Active leads | Masked | Config "type" field |
|---|---|---|---|---|
| 12 | C12_FULL | I, II, III, aVR, aVL, aVF, V1–V6 | — | device-supported |
| 10 | C10_PREC_LIMB | I, II, aVR, aVL, V1–V6 | III, aVF | clinically-motivated simulated |
| 8 | C8_INDEPENDENT | I, II, V1–V6 | III, aVR, aVL, aVF | clinically-motivated simulated |
| 6 | C6_LIMB_6 | I, II, III, aVR, aVL, aVF | V1–V6 | device-supported |
| 4 | C4_EINTHOVEN_SEPTAL | I, II, III, V1 | aVR, aVL, aVF, V2–V6 | clinically-motivated simulated |
| 2 | C2_TELEMETRY | II, V1 | 10 others | clinically-motivated simulated ("telemetry-like") |
| 1 | C1_SINGLE_LEAD | II | 11 others | clinically-motivated simulated ("ambulatory-like") |

Key properties for the report:
- The ladder is **not nested**: 6-lead drops all precordials, which 8-lead and 4-lead keep (in part). The protocol (§2) explicitly forbids calling drift monotonic.
- 10-lead and 8-lead remove only derived leads. Because III, aVR, aVL and aVF are linear combinations of I and II, **no electrical information is physically lost** at those levels. The information is only hidden from a model that cannot derive it. The report should state this directly, because it explains the near-flat 12→8 results.
- Device associations (AliveCor KardiaMobile 6L FDA K191406; Philips IntelliVue; "Zio XT") are **Documented only**. Commercial single-lead patches do not record standard lead II, so "ambulatory-like" is the safer wording.
- `lead_masking.py` also defines an "anatomical" schedule (including `6_lead_precordial`, `2_lead_orthogonal`) and `data/benchmark_test/` contains `leads_6_precordial`. These are **not** part of the Module 4 evaluation.

### F.2 Masking and derived-lead handling (**Verified**)

- Operator: X_masked = X ⊙ (m_l ⊗ 1ᵀ), numpy and torch paths; inactive channels exactly 0.0; active channels unchanged; `is_reconstructed` always False.
- Provenance: I, II, V1–V6 "measured"; III = II − I, aVR = −(I+II)/2, aVL = I − II/2, aVF = II − I/2. These identities hold for ideal Einthoven/Goldberger leads. Whether the dataset carts actually derived these channels is an assumption, not a measurement.
- The reasoning model receives the 12-dim mask as tokens, so it can in principle learn which leads are absent.
- Validation: `tests/test_lead_masking.py` (passing); `reports/masking_integrity_report.json`, `reports/lead_provenance_audit.json`, `runs/module2_v1/reports/module2_lead_masking_audit.md` (documented).

---

## G. Reasoning and Grounding Pipeline

### G.1 Clinical observation extraction (O_l)

- Implemented in `src/observations/clinical_extractor.py` (**Verified**). It outputs HR, rhythm label, PR, QRS, and per-lead ST elevation (J+60 ms ≥ 0.10 mV), ST depression (≤ −0.05 mV) and T inversion (< −0.10 mV), on active leads only.
- Validation (`reports/clinical_observation_validation.json`, **Documented only**): 250 MIMIC records against machine measurements. HR MAE 3.70 bpm (median 0.96, r 0.8664); PR MAE 21.57 ms (r 0.581); QRS MAE 28.92 ms (r 0.2551); AFIB detection sensitivity 0.633 / precision 0.38 / F1 0.475; SBRAD F1 0.917; STACH F1 0.681. ST/T rules are labelled "heuristic", without validation numbers.
- **Defect (Verified by reading `clinical_extractor.py:270–295`):** the QRS search window is ±60 ms, so at 100 Hz the largest measurable QRS is about 110 ms. Any rule that needs QRS ≥ 120 ms (RBBB, LBBB) can never fire. This is why the B-2 rule baseline produced F1 = 0 for RBBB/LBBB and was withdrawn. The same capped values are fed to the reasoning model as observation tokens.
- The JSON label "CLINICALLY_VALIDATED" for HR overstates what was done: agreement was measured against device measurements on 250 records, not against clinicians. The reference device is called "Philips PageWriter" in this JSON, while `MODULE3_FROZEN.json` says MIMIC labels come from "GE Marquette 12SL" (**Contradictory**).

### G.2 Context and reasoning

Covered in E.2 (**Verified**). No external knowledge is used. `context_builder.py:131` and `reasoning_transformer.py` raise errors if CKP tokens are passed.

### G.3 Reasoning traces

- `StructuredTraceGenerator` (**Verified**) takes the top-probability hypothesis and lists evidence tokens whose cross-attention weight passes a threshold. Each is tagged with associated leads and `is_observable_in_mask`. The trace also carries observation summaries and an "interpretation_audit" quality class.
- Counts (**Verified**): 25,816 validation traces (3,688 × 7); 700 sample test traces (100 per configuration). All 51,625 test predictions are in the CSV.
- **What a "claim" is right now:** an attention-selected token, not a clinical proposition with a measurable criterion. The pre-registered gate (G.5) needs explicit claims with thresholds θ(f_k). Turning attention tokens into claims is an unspecified step.
- Every trace carries this disclaimer: "Cross-attention reflects routing correlation … does not prove causal mechanistic faithfulness."

### G.4 Audit-derived descriptive statistic (computed during this audit; not a project result)

Share of attention-selected "supporting evidence" tokens that point at masked leads, over the 25,816 validation traces:

| Config | 12 | 10 | 8 | 6 | 4 | 2 | 1 |
|---|---|---|---|---|---|---|---|
| Supporting tokens referencing masked leads | 0.0 % | 25.7 % | 43.3 % | 52.3 % | 61.7 % | 72.5 % | 78.1 % |
| Fraction of the 12 leads masked | 0 % | 16.7 % | 33.3 % | 50 % | 66.7 % | 83.3 % | 91.7 % |

So the model does attend to tokens of absent leads, but at rates near or below what uniform attention over lead-specific tokens would produce. On its own this neither shows nor rules out "hallucinated" reasoning, so it must not be presented as an RFB finding.

Trace quality classes and their corrections: `validation_trace_findings.csv` originally counted channel unavailability as "contradiction". `validation_trace_documentation_correction.md` / `validation_trace_corrected_findings.csv` retract that. Genuine HR contradictions are flat at about 1.0–1.5 % across configurations, and the heuristic was asymmetric across diseases (**Documented**, consistent with the counts I reproduced: 6-lead 289, 2-lead 697, 1-lead 836 "Contradictory").

### G.5 Grounding gate and CKP

- **Gate:** the specification in `reports/evidence_observability_config.json` (claims with `required_leads`, `measurement_requirement`, `unavailable_condition`) and protocol §3 is pre-registered and hash-frozen. **No implementation** exists in `src/` or the pipeline scripts. Status: **Planned**. The MI rule (≥ 0.10 mV ST elevation in ≥ 2 contiguous leads of one territory; otherwise UNAVAILABLE) is specified, and so is a territory map in `src/reasoning/clinical_taxonomy.py`.
- **CKP:** **Contradictory.** The Review-2 report and `project_report.md` describe a dual-path system. In the repository today: (a) the Control model rejects CKP; (b) there is no CKP encoder, SNOMED graph, triple store, or experimental training mode in any `.py` file; (c) `checkpoints/reasoning_transformer_experimental.pt` contains `ckp_encoder.*` weights from a 500-sample pilot (val AUROC 0.7493 vs 0.7522 for the matching control pilot), so the pilot shows no benefit; (d) the pytest cache lists `tests/test_ckp_builder.py` (6 tests) and `tests/test_dual_path_reasoning.py` (6 tests), which are no longer present. Only the protocol's B-4 row and the "path invariance" clause in `benchmark_candidates.json` survive. **Decision (user, 2026-10-19):** CKP appears in the first-half report only as planned future work (Chapter 3 design intent and Chapter 6 roadmap). It is not described as implemented, and the pilot checkpoint is not reported.

---

## H. Metrics and Experimental Evidence

### H.1 Diagnostic metrics on the test split (B-3 Control, seed 42, calibrated)

Source: `runs/module4_v3/metrics/test_evaluation_summary.json`. Macro values for 12/6/4/1-lead were **recomputed** from the CSV and match. Other cells are **Documented** in the same JSON. Macro = unweighted mean over the 8 classes, each on its own observed-label subset. F1 uses a 0.5 threshold on calibrated probabilities. CIs come from 200 bootstrap resamples; whether those resamples were patient-clustered is **Unverified** (the protocol pre-registers 1,000 patient-clustered resamples for grounding metrics).

| Config | Macro-AUROC [95 % CI] | Macro-F1 [95 % CI] | ECE | Brier | NLL |
|---|---|---|---|---|---|
| 12 | 0.9540 [0.9514, 0.9572] | 0.7375 [0.7295, 0.7478] | 0.01047 | 0.04309 | 0.14367 |
| 10 | 0.9536 [0.9509, 0.9566] | 0.7367 [0.7278, 0.7483] | 0.01071 | 0.04330 | 0.14418 |
| 8 | 0.9526 [0.9502, 0.9558] | 0.7307 [0.7228, 0.7421] | 0.01100 | 0.04381 | 0.14591 |
| 6 | 0.9479 [0.9450, 0.9510] | 0.7076 [0.6987, 0.7209] | 0.01219 | 0.04612 | 0.15259 |
| 4 | 0.9524 [0.9499, 0.9553] | 0.7347 [0.7253, 0.7461] | 0.01247 | 0.04408 | 0.14637 |
| 2 | 0.9487 [0.9455, 0.9519] | 0.7101 [0.7000, 0.7218] | 0.01247 | 0.04666 | 0.15322 |
| 1 | 0.9215 [0.9175, 0.9251] | 0.5832 [0.5710, 0.5983] | 0.01319 | 0.05397 | 0.17846 |

(The superseded context had slightly different Brier/NLL values for 10/6/2/1-lead; the values above are copied from the JSON.)

Per-class highlights (JSON): at 12-lead, MI AUROC 0.8475 / F1 0.550 and 1AVB F1 0.423 are the weakest. At 1-lead, RBBB recall drops to 0.074 (F1 0.127) and LBBB F1 to 0.457, while rate-based classes hold (STACH F1 0.906, SBRAD 0.836). Per-cohort 12-lead macro-AUROC: PTB-XL 0.9476, Chapman 0.9509, MIMIC 0.9379.

DFB, using the pre-registered rule (RPR = F1/F1₁₂ < 0.80): the 1-lead RPR is 0.5832/0.7375 = 0.791 and every other level is ≥ 0.959. So DFB = 1-lead, transition 2→1. This is my arithmetic from verified numbers, and it is subject to the provenance caveat in H.5.

### H.2 Validation results

Seed 42 per configuration (`control_training_history_seed42.json`, documented): 12-lead AUROC 0.9521 / F1 0.7241 … 1-lead 0.912 / 0.571.
Multi-seed (`multi_seed_variance_summary.json`): macro-AUROC 0.9449 ± 0.0006, macro-F1 0.6982 ± 0.0026, best val loss 0.15848 ± 0.00127. These are **validation values pooled over all 7 configurations** (25,816 rows per seed), not test values. The superseded context did not say this, and they must not be set beside the test table as if they were comparable.

### H.3 Representation drift (ECG-FM, Module 3)

Pooled mean cosine similarity to the 12-lead embedding over 36,888 records (`MODULE3_FROZEN.json`; per-split rows in `runs/module3_v1/reports/module3_cosine_similarity_summary.csv` agree to 3 decimals): 10-lead 0.998051, 8-lead 0.997095, 6-lead 0.877548, 4-lead 0.904078, 2-lead 0.876178, 1-lead 0.814373 (**Partially verified**).

These numbers measure embedding geometry only, not diagnostic accuracy or reasoning quality (MODULE3 limitation note). Other drift numbers in the repository come from different models or samples and must not be mixed in: `reports/representation_sensitivity_report.json` (legacy ResNet adapter, 100 records, D₁ = 0.00368), `MODULE_3_STATUS.md` (one record, D₁ = 0.2197), `project_report.md` (0.9003 / 0.9182 / 0.8820 / 0.8535, origin unknown).

### H.4 Baselines

B-1 linear probe on Z_ECG (test, JSON): 12-lead macro-AUROC 0.9224, macro-F1 0.6269. F1 by config 0.6269 / 0.6216 / 0.6199 / 0.4172 / 0.4991 / 0.4296 / 0.2874 (12→1). Comparison with B-3 is **descriptive only**: no per-record B-1 predictions exist, so there are no paired CIs, and the script is outside the repo. B-1 and B-3 also differ in capacity, training augmentation over 7 configurations, and inputs, so the gap cannot be attributed to observation or availability tokens (CLM-03). B-2 rule engine: **withdrawn** (CLM-02).

### H.5 Test-set provenance (critical for wording)

`runs/module4_v3/reports/evaluation_provenance_statement.md` records **"EVALUATION PROVENANCE STATUS: AUTHORIZATION-UNVERIFIED"**. The pre-registered `RUN_TEST` gate was not satisfied: an automated IDE hook triggered execution, the first run crashed on a calibration-binning bug, and it was relaunched after a code fix. The project's own rule is that these results are "descriptive empirical reference points" and not a confirmatory pre-registered benchmark. The report must say so wherever test numbers appear.

### H.6 Grounding metrics (Planned; definitions as pre-registered in `reports/module4_evaluation_protocol.md` §4–§7)

- EO_l = N_observable / N_required (NOT_APPLICABLE if N_required = 0)
- GQ_l = N_supported / (N_supported + N_contradicted) (NOT_ESTIMABLE if the denominator is 0)
- TGR_l = GQ_l × EO_l; UR_l = N_unavailable / N_total; GSR_l = N_supported / N_total
- Aggregation: micro (primary), macro (secondary); 95 % CI from 1,000 patient-clustered bootstrap resamples
- **RFB:** first c_k in ⟨12,10,8,6,4,2,1⟩ with GQ(c_k) < τ, τ = 0.60 (sensitivity τ = 0.5 and 0.7), reported with the transition pair; recovery flag if GQ climbs back above τ; special cases ROBUST / FAILED_AT_BASELINE / INDETERMINATE
- **DFB:** first c_k with MacroF1(c_k)/MacroF1₁₂ < 0.80
- **Δ:** Δ_steps = k_DFB − k_RFB; Δ_leads = l_RFB − l_DFB; Δ > 0 = "diagnostic persistence beyond reasoning failure"
- **AURDC** = (1/11) Σ_{k=1..6} ½(GQ_k + GQ_{k+1})(l_k − l_{k+1}); AUTDC is the same with TGR

The superseded context and `project_report.md` give RFB = min{l : GQ_l ≥ τ} and an un-normalised AURDC. These **contradict** the pre-registered protocol, which is the hash-frozen authority (L, item 7).

Methodological observation (recommendation, not a finding): GQ leaves out UNAVAILABLE claims, so a model that only makes unobservable claims has an undefined GQ rather than a low one. As defined, the RFB measures *contradiction* among observable claims, while EO/UR/TGR capture *unobservability*. The report should explain why GQ, rather than TGR or UR, drives the boundary, or flag the choice as open.

### H.7 Tests

108 tests in 11 files, 107 passed and 1 skipped in this session (H.0 command). The pytest cache from an earlier run lists 120 node IDs, including 12 CKP/dual-path tests whose files have since been removed. "47/47 tests" (`MODULE_1_TO_3_FINAL_AUDIT.md`, `project_report.md`) is an older snapshot. Coverage was not measured.

### H.8 Plots

`visualizations/` holds degradation-unrelated concept graphics (conclusion roadmap, "sentence" visuals), disease-distribution plots, phase dashboards, Z_ECG sample plots, cosine-similarity variants and demo screenshots. Which data fed each plot was **not** checked. Before any figure goes into the report, trace it to its generating script (e.g. `scripts/plot_cosine_similarity.py`, `scripts/08_plot_disease_distribution.py`) and confirm the data version (v1 vs Module 4). `FYP Material/Degradation_Curve.png` and `Curve_Comparision.png` are of unknown provenance (**Unverified**).

---

## I. Research and Literature Checklist

### I.1 References currently in `report/07_REFERENCES.md` (33 entries)

None were checked against a live database in this audit. Labels: **Known** = a work I recognise as existing with the cited venue (details and DOI still to be checked); **Check** = details look doubtful or unfamiliar and must be verified before use; **Irrelevant** = out of scope.

| Reference | Status | Note |
|---|---|---|
| Angelopoulos & Bates 2021 (conformal prediction, arXiv) | Known | Only needed if conformal prediction is discussed |
| Atoui, Fayn & Rubel 2010, *Computers in Cardiology* 37 | **Check** | Venue/pages uncertain; the authors also published in IEEE TITB 2010 |
| Baevski et al. 2020, wav2vec 2.0, NeurIPS | Known | |
| Dawson, Yang & Balasubramanian 2009, J Electrocardiol 42(6) | **Check** | Title/author list doubtful |
| Devlin et al. 2019, BERT | Known | Marginal relevance |
| Einthoven 1908 | Known | |
| Frank 1956, Circulation | Known | |
| Geifman & El-Yaniv 2017; 2019 | Known | |
| Goldberger et al. 2000, PhysioNet | Known | |
| Goldberger E. 1942, augmented leads | Known | |
| Guo et al. 2017, calibration | Known | Supports temperature scaling |
| Hao, Chen & Lu 2021, IEEE TBME 68(8) | **Check** | Unfamiliar; possibly wrong |
| Jacovi & Goldberg 2020, ACL | Known | |
| Johnson et al. 2023, MIMIC-IV, Sci Data | Known | This is the EHR dataset paper; the **MIMIC-IV-ECG** dataset needs its own citation |
| Kipf & Welling 2017; Veličković 2018; Wu 2019 | Known | Only relevant if a graph-based CKP is described |
| Kligfield et al. 2007 (AHA/ACC/HRS Part I) | Known | |
| Lakkaraju et al. 2020 | Known | |
| Li, Zhang & Wang 2023, CBM 165:107412 | **Check** | Unfamiliar |
| Liu et al. 2025, EMNLP Findings (knowledge-enhanced arbitrary-lead ECG) | **Check** | Plausible and highly relevant; verify |
| Mason et al. 2007 (Part II) | Known | |
| Nelwan et al. 2004, J Electrocardiol | Known (details to check) | |
| Pan & Tompkins 1985 | Known | |
| Rajpurkar et al. 2022, Nat Med | Known | |
| Ribeiro et al. 2020 (Nat Commun) | Known | |
| Schwartz et al. 2020, Green AI | Known | |
| Strodthoff et al. 2021, JBHI | Known | |
| Vovk et al. 2005 | Known | |
| Wagner et al. 2020, PTB-XL, Sci Data | Known | |
| Willems et al. 1991, NEJM | Known | |
| Yasunaga et al. 2021, QA-GNN | Known | |
| Zhang & Fink 2025 (water-network leak detection) | **Irrelevant** | Remove |
| Zheng et al. 2020, PhysioNet DOI 10.13026/gfy7-fc21 | **Check** | Author list ("Daniasekry") and venue look wrong; Zheng et al. 2020 is a *Scientific Data* paper; the DOI may belong to a different PhysioNet dataset |

### I.2 Missing references the report will need (to be found and verified later — do not cite from memory)

- [ ] **ECG-FM** (McKeen et al., JAMIA Open 2025, per `MODULE3_FROZEN.json`): the backbone itself, currently uncited. Also confirm its pretraining sampling rate and segment length (L, item 12).
- [ ] **fairseq-signals** library and **CMSC** / CLOCS contrastive ECG pretraining (Kiyasseh et al.).
- [ ] **MIMIC-IV-ECG** dataset (PhysioNet), including how its machine statements were generated.
- [ ] PhysioNet/CinC Challenge 2020 and 2021 (2021 is specifically about reduced-lead ECG classification) and the Chapman-Shaoxing data paper.
- [ ] Other ECG foundation / self-supervised models for related work (e.g. ST-MEM, MERL, HuBERT-ECG, ECG-JEPA). Verify each one exists.
- [ ] Reduced-lead and missing-lead robustness studies, and lead-reconstruction literature (replacing the "Check" items above with verified ones).
- [ ] Attention as explanation: Jain & Wallace 2019; Wiegreffe & Pinter 2019. Essential, because the traces rely on attention.
- [ ] Hallucination and grounding in medical AI; evidence verification and faithfulness metrics.
- [ ] Clinical criteria actually used in configs: AHA/ACCF/HRS 2009 recommendations (Surawicz et al.; Wagner et al., both named in `reports/module4_unresolved_decisions.md`); Fourth Universal Definition of MI (Thygesen et al. 2018) for ST criteria; PR > 200 ms, QRS ≥ 120 ms.
- [ ] SNOMED CT (official source), and any ECG ontology if CKP is described.
- [ ] Iterative multilabel stratification (Sechidis et al. 2011).
- [ ] Trustworthy-AI-in-healthcare evaluation guidance (e.g. TRIPOD-AI, CONSORT-AI / SPIRIT-AI, DECIDE-AI), for the limitations and social-impact chapters.
- [ ] Device claims (AliveCor 6L FDA K191406; Zio) — use regulatory or primary sources, or drop them.

### I.3 Style

The sample uses numbered square-bracket citations (natbib `square, numbers, sort&compress`) and an IEEE-like `thebibliography`. `07_REFERENCES.md` currently uses APA author-year. It will need converting.

---

## J. Sample Thesis Structure (Verified from `ug-project-report.tex` + `chapter1–6.tex` + PDF TOC)

**Location:** `C:\Users\VEDANTH\Downloads\UCS2717-ProjectReport-LaTeX\UCS2717-ProjectReport-LaTeX\` (the duplicate folder "… (1)" has an identical `.tex`). It is not in the repository.

**Front matter (already done by the student; do not recreate):** title page, bonafide certificate, Tamil abstract placeholder, acknowledgements, abstract, TOC, list of tables, list of figures. Roman page numbers.

**Main matter, as in the sample (6 numbered chapters + unnumbered REFERENCES):**

| Ch. | Title (sample, uppercase) | Sections (sample) |
|---|---|---|
| 1 | INTRODUCTION | 3–4 unnumbered intro paragraphs, then 1.1 MOTIVATION, 1.2 BACKGROUND, 1.3 PROBLEM DEFINITION, 1.4 ORGANIZATION OF THE REPORT |
| 2 | LITERATURE SURVEY | 2.1 PAPERS RELATED TO THE PROBLEM IDENTIFIED (2.1.1–2.1.7, one per paper/approach, one paragraph each), 2.2 RESEARCH GAPS, 2.3 RESEARCH OBJECTIVES (+ Table 2.1 "Identified Research Gaps and Proposed Research Objectives") |
| 3 | PROPOSED METHODOLOGY | 3.1 ARCHITECTURE DESIGN (architecture figure), 3.2 core-model architecture section, 3.3 MODULES (3.3.1–3.3.8, sub-subsections, equations, pseudocode via `algorithm`/`algpseudocode`) |
| 4 | EXPERIMENTAL RESULTS AND PERFORMANCE ANALYSIS | 4.1 DATASET DESCRIPTION (4.1.1 Composition, 4.1.2 Distribution, 4.1.3 Feature Representation), 4.2 ECOSYSTEM (4.2.1 Hardware Configuration, 4.2.2 Software Environment), 4.3 EXPERIMENTS CONDUCTED (4.3.x), 4.4 PERFORMANCE ANALYSIS (4.4.x, ending with "Complete Tabulation of Results") |
| 5 | SOCIAL IMPACT AND SUSTAINABILITY | 5.1 SOCIAL IMPACT, 5.2 SUSTAINABILITY |
| 6 | CONCLUSION AND FUTURE WORK | No sections; 3 paragraphs |
| — | REFERENCES | Unnumbered (`\addtotoc{REFERENCES}`), numbered bibitems |

Conventions: Thesis.cls, A4, 14 pt, oneside, Times (pslatex), 1.5 spacing in main matter; section titles in capitals (some subsections in title case); tables captioned above (`\caption` before `tabular`); figures with labels referenced as `Fig~\ref{}`; formal third-person academic tone, with a few "we" in the acknowledgements. The sample's own 1.4 text says "Chapter 7 presents a list of all the references", but its TOC has References unnumbered. The template is inconsistent on this point.

**Proposed mapping for this project's first-half report:**

| Sample section | This project (first half) |
|---|---|
| 1 intro paragraphs | ECG in practice; reduced-lead acquisition; foundation models; accuracy-vs-grounding problem; first-half status |
| 1.1–1.4 | Motivation / ECG lead physiology and FM background / formal RFB problem statement / organisation |
| 2.1.x | One subsection per verified paper cluster (ECG FMs incl. ECG-FM; PTB-XL benchmarks; reduced-lead diagnosis and CinC 2021; lead reconstruction; attention faithfulness; knowledge-enhanced ECG; calibration) |
| 2.2 / 2.3 | Gaps → objectives table (Table 2.1 style), with first-half vs second-half objectives marked |
| 3.1 | End-to-end architecture figure (Modules 1–6), marking implemented vs planned blocks |
| 3.2 | Control Reasoning Transformer architecture (E.2) |
| 3.3.x | 3.3.1 Data ingestion & label harmonisation; 3.3.2 Preprocessing P1–P5; 3.3.3 Lead degradation & provenance; 3.3.4 ECG-FM representation; 3.3.5 Observation extraction; 3.3.6 Context builder & reasoning transformer; 3.3.7 Trace generation; 3.3.8 Grounding gate & RFB metrics (specification only, marked planned); optional 3.3.9 CKP (planned design) |
| 4.1.1–4.1.3 | Cohorts & accounting (D.1–D.2); split and label distribution (D.3–D.4); signal and feature representation (D.5, E.1) |
| 4.2.1–4.2.2 | Hardware (only what can be verified; flag training device) / software (versions with caveat) |
| 4.3.x | Preprocessing/QC; masking integrity; representation drift; observation-extractor agreement; Control model training (3 seeds); test degradation evaluation; trace generation; test suite |
| 4.4.x | Degradation curve; per-class; per-cohort; calibration; B-1 descriptive comparison; drift vs performance (non-monotonicity); trace observability (with caveats); complete tabulation |
| 5.1 / 5.2 | Safety of reduced-lead AI and over-trust; equity; frozen-backbone compute footprint (avoid unmeasured energy numbers) |
| 6 | First-half summary, limitations, second-half roadmap |
| REFERENCES | Unnumbered, numeric |

The superseded context stated a "mandatory" 7-chapter structure with "Chapter 7: References" and exclusively 4.1–4.4. The sample does not show that (it has 6 numbered chapters, References unnumbered, and subsections inside 4.1–4.4). If the department mandated that structure separately, the user has to confirm it (L, item 21).

---

## K. Verified Numerical Facts

| # | Fact | Value | Status | Evidence |
|---|---|---|---|---|
| 1 | Processed records (v1 arrays) | 36,957 = 21,799 + 10,158 + 5,000; each (N,12,1000) float32 | Verified | `data/processed_100hz/**/signals.npy` |
| 2 | QC-rejected | 69 = 2 + 16 + 51 | Documented (consistent) | `*/preprocessing_summary.json` |
| 3 | Modelling records | 36,888 = 21,797 + 10,142 + 4,949 | Verified | `runs/module4_v3/split_manifest.csv` |
| 4 | Splits | 25,825 / 3,688 / 7,375 (folds 1–7 / 8 / 9–10) | Verified | manifest |
| 5 | Unique patients | 23,818 / 3,394 / 6,746; total 33,958 | Verified | manifest recount |
| 6 | Patient leakage | 0 for all split pairs | Verified | manifest recount |
| 7 | Test by cohort | 4,358 / 2,028 / 989 | Verified | manifest, CSV |
| 8 | Test evaluations | 51,625 = 7,375 × 7 | Verified | CSV rows |
| 9 | Signal spec | 100 Hz, 10 s, 1,000 samples, mV, float32 | Verified | arrays + config |
| 10 | Filters | HP 0.5 Hz (o3), notch 50/60 Hz Q30, LP 45 Hz (o4), at 500 Hz | Verified | `module1_config.json`, `filtering.py` |
| 11 | ECG-FM | 90,883,072 params; 12 layers; 768; 12 heads; FFN 3,072 | Verified | checkpoint |
| 12 | Control model params | 1,830,657 | Verified | checkpoint |
| 13 | Evidence tokens | 51 | Verified | `context_builder.py` |
| 14 | Training examples/epoch | 180,775 = 25,825 × 7 | Verified (arithmetic) + documented | config |
| 15 | Test macro-AUROC 12→1 | 0.9540, 0.9536, 0.9526, 0.9479, 0.9524, 0.9487, 0.9215 | Verified (4 recomputed, 3 from JSON) | H.1 |
| 16 | Test macro-F1 12→1 | 0.7375, 0.7367, 0.7307, 0.7076, 0.7347, 0.7101, 0.5832 | Verified (as above) | H.1 |
| 17 | Validation multi-seed | AUROC 0.9449 ± 0.0006; F1 0.6982 ± 0.0026 (pooled over 7 configs) | Documented | `multi_seed_variance_summary.json` |
| 18 | ECG-FM cosine sim 10→1 | 0.998051, 0.997095, 0.877548, 0.904078, 0.876178, 0.814373 | Partially verified | `MODULE3_FROZEN.json` + CSV |
| 19 | Representation vectors | 258,216 = 36,888 × 7 | Documented (arithmetic checks) | `MODULE3_FROZEN.json` |
| 20 | Validation traces | 25,816 (7 × 3,688) | Verified | JSON lengths |
| 21 | Sample test traces | 700 (7 × 100) | Verified | JSON |
| 22 | Tests | 108 collected; 107 passed, 1 skipped | Verified (run this session) | pytest |
| 23 | HR extractor agreement | MAE 3.70 bpm, r = 0.8664 (n = 250, vs machine) | Documented | `clinical_observation_validation.json` |
| 24 | Max measurable QRS | ≈110 ms | Verified (code reading) | `clinical_extractor.py:279–285` |
| 25 | B-1 linear probe (test, 12-lead) | AUROC 0.9224, F1 0.6269 | Documented (JSON) | `comprehensive_baseline_comparison.json` |
| 26 | Calibration (val, seed 42) | mean ECE 0.01498 → 0.01043 | Documented | history JSON |
| 27 | CKP pilot | 500/150 PTB-XL; val AUROC 0.7493 vs control 0.7522 | Verified (checkpoint metadata) | `checkpoints/reasoning_transformer_*.pt` |
| 28 | Code size | 250 .py / 58,114 lines incl. vendored; 104 / 30,205 excl.; `src/` own 27 / 5,388 | Verified | `find`/`wc` |
| 29 | Scripts / tests / reports / visuals | 39 .py scripts; 11 test files; 45 report files; 22 PNG + 4 PDF | Verified | `ls` |
| 30 | Freeze integrity | All Module 4 frozen hashes match except `src/reasoning/multi_cohort_loader.py` (modified after freeze) | Verified | re-hash |

---

## L. Contradictions and Open Questions (prioritised)

**Critical — affects what the report may claim**

1. **CKP / Experimental path is not implemented in the repository.** Documents (Review-2, `project_report.md`, superseded context) describe it as built or in progress. The evidence is pilot checkpoints and deleted tests. **Resolved (user decision, 2026-10-19):** the first-half report presents CKP only as planned future work. It is not described as implemented, and the 500-sample pilot checkpoint is not reported as a result.
2. **Grounding gate and all grounding metrics are unimplemented.** Only the specification exists. Demo scripts (`scripts/pipeline_flow_terminal_runner.py:452–458`, `demo_video_presentation.py`, `generate_*_visual*.py`) print **hard-coded** values (e.g. "86.7 % MI … 100 % SUPPORTED") that are illustrative and must never be cited.
3. **Test results are AUTHORIZATION-UNVERIFIED** (H.5), and the run was relaunched after a crash. *Resolve:* agree on wording, e.g. "descriptive held-out results; pre-registration gate not satisfied".
4. **Record and patient counts.** 36,957 (processed) vs 36,888 (modelled). The superseded context's patient counts (23,802 / 3,388 / 6,837; 34,027) do not match the manifest (23,818 / 3,394 / 6,746; 33,958).
5. **Multi-seed figures are validation-pooled**, not test. They were presented without that qualifier.
6. **Old split scheme** (PTB-XL folds 1–8/9/10; 17,418/2,183/2,198 or 17,441/2,183/2,175) vs the current Module 1 v2 70/10/20 scheme. Chapman/MIMIC are described as "external validation" in old documents but are pooled in Module 4.
7. **RFB and AURDC definitions** in the superseded context and `project_report.md` (min{l : GQ ≥ τ}; no 1/11 normalisation) contradict the frozen protocol (first c_k with GQ < τ; normalised AURDC).
8. **Representation-drift numbers** exist in four incompatible versions (H.3). Only the Module 3 ECG-FM values are valid for the report.

**High — technical validity**

9. **QRS extractor capped at about 110 ms.** BBB criteria cannot be observed; observation tokens for QRS are biased; the B-2 baseline is invalid.
10. **Cross-attention "evidence" ≠ clinical claims.** The pipeline from trace to claim to gate is unspecified, and attention faithfulness is not established.
11. **ECG-FM YAML vs checkpoint** (24 vs 12 layers). Also unclear whether lead-masking pretraining (`mask_leads_prob 0.5`) applies to this checkpoint. It would explain the high 10/8-lead similarity.
12. **Possible sampling-rate mismatch:** ECG-FM receives 100 Hz, 1,000-sample input. Check the ECG-FM paper/model card for the pretraining rate and length. If it was pretrained at 500 Hz, the representations are out of distribution and this is a major limitation.
13. **Loss function:** masked BCE (code) vs "focal loss γ = 2" (`research_claims_correction.md` §3.3).
14. **MIMIC label provenance:** "machine report statements" (GE Marquette 12SL per MODULE3) vs "Philips PageWriter" (observation validation). Labels and the extractor reference are both machine-derived.
15. **`multi_cohort_loader.py` modified after the Module 4 freeze**, and the Module 3 checksum-manifest hash does not match its frozen value. Cause undetermined. It may be the documented test-quarantine remediation (`tests/test_blocker_remediation.py`).
16. **Training hardware unknown.** An RTX 4050 is present, but installed torch is CPU-only, and no device is logged for Module 4.

**Medium — documentation**

17. Chapman source count 10,247 / 10,249 / 10,158; local-file statement outdated.
18. Test-count snapshots 47 vs 108 (vs 120 historical, including deleted CKP tests).
19. PTB-XL prevalence variants (9,514 / 12,738 / tri-state v2). Only v2 matches the results.
20. "Cabrera sequence" naming is incorrect for the implemented standard order.
21. Thesis structure: the sample has 6 numbered chapters + unnumbered References; the superseded context claimed a "mandatory" 7-chapter scheme. *User to confirm.*
22. B-1 baseline script is outside the repository (IDE scratch path); B-1 per-record predictions are missing.
23. `PROJECT_PHASES_SPECIFICATION.md` (root, hash-frozen) differs from the copy in `FYP Material/` (e.g. sigmoid/BCE vs softmax). Use the root copy.
24. `report/first_half_implementation/01_INTRODUCTION.md` was written from the superseded context and may repeat items 1, 4–8, 20. Review it before reuse. (It was not modified in this audit.)
25. Device mappings (AliveCor K191406, Zio XT, IntelliVue) are undocumented assertions.
26. Root PNGs `01_dashboard_main.png` … `06_redoc_documentation.png` belong to an unrelated "customer dashboard" project. Exclude them.

**Open questions for the user**

- ~~Should CKP appear only as future work?~~ Resolved: yes, future work only (see item 1).
- Wording for the unverified test provenance?
- Confirm the chapter scheme (sample vs a department mandate).
- Was Module 4 trained on GPU or CPU?
- Is there any CKP/grounding-gate code outside `E:\ECG`?

---

## M. First-Half Scope Boundary

**Completed and verified implementation**
- Preprocessing P1–P5 and QC (36,957 processed → 36,888 eligible).
- Tri-state 8-label harmonisation and patient-disjoint 70/10/20 splits (zero leakage).
- Seven-level zero-imputation masking with lead provenance.
- Frozen ECG-FM feature extraction for all 7 × 36,888 inputs.
- Control Reasoning Transformer trained with 3 seeds, plus validation calibration.
- Trace generation (25,816 validation; 700 sample test).
- Test suite (107 passed, 1 skipped).

**Implemented but incompletely validated**
- Observation extractor: HR checked against machine measurements; PR/QRS heuristic; QRS capped.
- Trace "evidence" semantics (attention only).
- Representation-drift analysis (summary verified; per-file count documented only).
- B-1 linear-probe baseline (no per-record outputs).

**Preliminary experimental findings (state with caveats)**
- Diagnostic degradation curve on the test split: near-flat from 12 to 2 leads (AUROC 0.954 → 0.949; F1 0.738 → 0.710), with a drop at 1 lead (0.922 / 0.583). The 4-lead > 6-lead non-monotonicity appears in both performance and drift. Caveat: authorization-unverified, single pass, relaunched.
- Class-specific 1-lead collapse for RBBB/LBBB; rate classes robust.
- DFB = 1-lead (from the pre-registered rule; my arithmetic).
- Descriptive B-3 > B-1 margins.

**Planned work**
- Claim formalisation from traces; grounding gate; EO/GQ/TGR/UR/GSR; RFB; AURDC/AUTDC; Δ.
- CKP implementation and Control-vs-CKP comparison.
- Fixing the QRS window; ablations of observation and availability tokens.
- Paired-uncertainty baselines; any authorised confirmatory test run (`runs/module4_v4/` per protocol §8).

**Must not appear as established results**
- Any RFB value, GQ/TGR/EO number, or "% SUPPORTED/UNAVAILABLE" figure.
- Any CKP benefit or CKP implementation claim.
- That the model "hallucinates" or "reasons from absent leads" as a demonstrated finding (only the descriptive attention statistic G.4 exists).
- Causal attribution of robustness to V1, observation tokens or availability tokens.
- Statistical superiority over B-1; anything about the B-2 rule baseline.
- "51,625 patients".
- Clinical validation or deployment readiness.
- Cosine similarity presented as accuracy.
- Hard-coded demo numbers.

---

## N. Recommended Chapter-Writing Plan

The plan follows the sample (J). Chapter-file names are suggestions for `report/first_half_implementation/`.

1. **`01_INTRODUCTION.md`:** intro paragraphs, then 1.1–1.4. Pull from A, F.1 (the ladder as motivation), H.6 (RFB definition stated as the problem, not a result). Say clearly that the first half delivers the infrastructure and the diagnostic curve, and the grounding analysis is pending. Re-check the existing draft against L.
2. **`02_LITERATURE_SURVEY.md`:** 2.1.x one subsection per *verified* source cluster (I.2 must be done first); 2.2 gaps; 2.3 objectives table with first-half / second-half columns. No unverified "Check" references.
3. **`03_PROPOSED_METHODOLOGY.md`:** 3.1 architecture figure marking implemented and planned blocks; 3.2 Control transformer (E.2, with token table and equations); 3.3.1–3.3.8 per J; grounding gate and metrics in 3.3.8 written as the pre-registered specification (H.6), with pseudocode from protocol §3; CKP as a planned extension.
4. **`04_EXPERIMENTAL_RESULTS_AND_PERFORMANCE_ANALYSIS.md`:** 4.1 (D tables); 4.2 (E.3 hardware/software with the "unverified device" note); 4.3 experiments (QC, masking integrity, drift, extractor agreement, training, test evaluation, traces, tests); 4.4 analysis (H.1 table and curve, per-class, per-cohort, calibration, B-1 descriptive, drift-vs-performance, G.4 with its caveat, provenance statement, complete tabulation). Every test number carries the H.5 caveat.
5. **`05_SOCIAL_IMPACT_AND_SUSTAINABILITY.md`:** 5.1 over-trust in reduced-lead AI and wearables, equity of access, and the research-prototype disclaimer (CLM-04); 5.2 frozen backbone and a small trainable head (1.83 M vs 90.9 M parameters). No energy numbers unless measured.
6. **`06_CONCLUSIONS_AND_FUTURE_WORK.md`:** short summary of completed work, limitations (L items 9–14), roadmap (M "Planned").
7. **References:** unnumbered; numeric style; only verified entries.

---

## O. Evidence Index

**Protocol and contracts (authoritative, hash-frozen):** `reports/module4_evaluation_protocol.md`; `reports/evidence_observability_config.json`; `PROJECT_PHASES_SPECIFICATION.md` (root copy); `configs/lead_configurations.json`; `configs/benchmark_candidates.json`; `configs/control_path_training_config.json`; `runs/module1_v2/MODULE1_FROZEN.json`; `runs/module2_v1/MODULE2_FROZEN.json`; `runs/module3_v1/MODULE3_FROZEN.json`; `runs/module4_v3/MODULE4_FROZEN.json`; `reports/module4_unresolved_decisions.md`.

**Data and labels:** `runs/module4_v3/split_manifest.csv`; `runs/module1_v2/labels/label_mapping_rules.json`; `runs/module1_v2/labels/label_observation_mask.csv`; `runs/module1_v2/configs/module1_config.json`; `data/processed_100hz/*/preprocessing_summary.json`.

**Results:** `runs/module4_v3/metrics/test_evaluation_summary.json`; `runs/module4_v3/metrics/test_predictions_seed42.csv`; `runs/module4_v3/metrics/control_training_history_seed{42,1337,2024}.json`; `runs/module4_v3/metrics/multi_seed_variance_summary.json`; `runs/module4_v3/metrics/comprehensive_baseline_comparison.json`; `runs/module3_v1/reports/module3_cosine_similarity_summary.csv`; `runs/module3_v1/reports/module3_disease_stratified_drift.csv`; `reports/clinical_observation_validation.json`.

**Governance and corrections (read before writing any results text):** `runs/module4_v3/reports/evaluation_provenance_statement.md`; `runs/module4_v3/reports/research_claims_correction.md`; `runs/module4_v3/reports/post_evaluation_baseline_reconciliation_audit.md`; `runs/module4_v3/reports/validation_trace_documentation_correction.md`; `runs/module4_v3/reports/validation_trace_corrected_findings.csv`; `runs/module4_v3/reports/stage7_test_evaluation_report.md` (contains the corrected AFIB/10-lead errata).

**Traces:** `runs/module4_v3/validation_traces/traces_validation_*.json`; `runs/module4_v3/reports/sample_test_reasoning_traces.json`.

**Code:** `src/preprocessing/pipeline.py`, `filtering.py`; `src/degradation/lead_masking.py`; `src/models/ecg_fm_primary.py`; `src/observations/clinical_extractor.py` (QRS window lines 279–285); `src/reasoning/context_builder.py`, `reasoning_transformer.py`, `trace_generator.py`, `clinical_taxonomy.py`; `scripts/16_train_reasoning_transformer.py`; `scripts/17_generate_reasoning_traces.py`; `scripts/18_generate_validation_reasoning_traces.py`.

**Checkpoints:** `checkpoints/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt` (+ `.yaml`, which disagrees); `runs/module4_v3/checkpoints/control_reasoning_best_seed42.pt`; `checkpoints/reasoning_transformer_experimental.pt` (CKP pilot).

**Commands to reproduce this audit's checks:**
- `python -m pytest -q -p no:cacheprovider` (needs the Python 3.11 install with torch)
- Recompute test metrics from the CSV with `sklearn.metrics.roc_auc_score` / `f1_score`, using `observed_<class> == 1` rows, `calibrated_prob_<class>` and `pred_<class>`.

**Sample thesis:** `C:\Users\VEDANTH\Downloads\UCS2717-ProjectReport-LaTeX\UCS2717-ProjectReport-LaTeX\ug-project-report.tex`, `chapter1.tex`–`chapter6.tex`, `ug-project-report.pdf`.

**Do not cite:** demo/visual scripts' printed numbers; `presentation/*`; `project_report.md` numerical tables (outdated); `reports/representation_sensitivity_report.json` (legacy model); root dashboard PNGs.

# Scripts Catalog & Execution Guide

This directory contains executable scripts organized by pipeline stage and functional role.

---

## 📋 Categorized Script Overview

### 1. Sequential Pipeline Stages (Numbered Execution Order)

| Script | Purpose | Input | Key Output |
| :--- | :--- | :--- | :--- |
| [`01_run_dataset_inspection.py`](file:///e:/ECG/scripts/01_run_dataset_inspection.py) | Initial data format & sampling rate inspection | Raw records | `reports/dataset_inventory.md` |
| [`02_run_label_harmonization.py`](file:///e:/ECG/scripts/02_run_label_harmonization.py) | Harmonizes diagnostic statements across cohorts | Raw labels | `reports/harmonization_table.md` |
| [`03_generate_multihot_labels.py`](file:///e:/ECG/scripts/03_generate_multihot_labels.py) | Builds multi-hot label matrices for 8 diseases | Harmonized labels | `data/processed_100hz/*/labels.npy` |
| [`04_assign_cohort_splits.py`](file:///e:/ECG/scripts/04_assign_cohort_splits.py) | Disjoint patient-level train/val/test splits | Labels & metadata | `reports/cohort_splits_summary.md` |
| [`05_verify_lead_masking.py`](file:///e:/ECG/scripts/05_verify_lead_masking.py) | Verifies zero-leakage masking & Cabrera DAG | Preprocessed signals | `reports/lead_masking_verification.md` |
| [`06_generate_frozen_test_benchmarks.py`](file:///e:/ECG/scripts/06_generate_frozen_test_benchmarks.py) | Generates frozen test arrays across 7 levels | Test set signals | `data/benchmark_test/ptbxl/*.npy` |
| [`07_generate_full_random_masked_dataset.py`](file:///e:/ECG/scripts/07_generate_full_random_masked_dataset.py) | Generates legacy random-masked ablation dataset | All cohorts | `data/seeded_random_masked/` |
| [`08_plot_disease_distribution.py`](file:///e:/ECG/scripts/08_plot_disease_distribution.py) | Plots disease prevalence and co-occurrence | Multi-hot labels | `visualizations/disease_distribution_*.png` |
| [`09_train_baseline_adapter.py`](file:///e:/ECG/scripts/09_train_baseline_adapter.py) | Trains baseline 1D-ResNet adapter (ablation only)| Train signals | `checkpoints/baseline_resnet1d_adapter.pt` |
| [`10_extract_latent_embeddings.py`](file:///e:/ECG/scripts/10_extract_latent_embeddings.py) | Extracts test latents for baseline ResNet | Benchmark signals | `data/latents/ptbxl/z_*_lead.npy` |
| [`11_extract_structured_observations.py`](file:///e:/ECG/scripts/11_extract_structured_observations.py) | Extracts clinical observations $O_l$ across 7 levels| Benchmark signals | `data/observations/ptbxl/obs_*_lead.json` |
| [`12_probe_observations_from_latent.py`](file:///e:/ECG/scripts/12_probe_observations_from_latent.py) | Linear probing of clinical traits from latents | Latents & observations | `reports/latent_observation_probing_results.json` |
| [`13_precompute_train_val_representations.py`](file:///e:/ECG/scripts/13_precompute_train_val_representations.py) | Precomputes train/val representations | Train/val signals | `data/latents/ptbxl/`, `data/observations/ptbxl/` |
| [`14_prepare_control_path.py`](file:///e:/ECG/scripts/14_prepare_control_path.py) | Prepares and audits Phase 4 Control Path (ECG-Only) | Upstream artifacts & contracts | `reports/control_path_preparation_report.json` |
| [`15_train_observation_heads.py`](file:///e:/ECG/scripts/15_train_observation_heads.py) | Trains & validates linear perception heads (Part 1)| Multi-cohort partitions & Z_l | `reports/observation_extractor_audit.json` |

---

### 2. Preprocessing & Acquisition Runners

| Script | Purpose | Output |
| :--- | :--- | :--- |
| [`download_ecg_fm.py`](file:///e:/ECG/scripts/download_ecg_fm.py) | Downloads the official pre-trained ECG-FM checkpoint from PhysioNet. | `checkpoints/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt` |
| [`download_validation_data.py`](file:///e:/ECG/scripts/download_validation_data.py) | Downloads external validation cohorts (Chapman & MIMIC-IV-ECG). | `data/external_validation/` |
| [`run_full_preprocessing.py`](file:///e:/ECG/scripts/run_full_preprocessing.py) | Runs 500 Hz -> 100 Hz calibration, filtering, and resampling on PTB-XL. | `data/processed_100hz/signals.npy` |
| [`run_external_preprocessing.py`](file:///e:/ECG/scripts/run_external_preprocessing.py) | Runs standard preprocessing on Chapman and MIMIC-IV-ECG cohorts. | `data/processed_100hz/chapman/`, `mimic/` |

---

### 3. Formal Audit & Verification Suites

| Script | Purpose | Audit Artifact |
| :--- | :--- | :--- |
| [`audit_preprocessing_pipeline.py`](file:///e:/ECG/scripts/audit_preprocessing_pipeline.py) | Audits filtering order, notch frequencies, Nyquist compliance, and mV calibration. | `reports/preprocessing_audit.json` |
| [`audit_clinical_observations.py`](file:///e:/ECG/scripts/audit_clinical_observations.py) | Benchmarks rule-based extractor against MIMIC Philips PageWriter cart measurements. | `reports/clinical_observation_validation.json` |
| [`audit_ecg_fm_primary.py`](file:///e:/ECG/scripts/audit_ecg_fm_primary.py) | Audits frozen ECG-FM parameters (90.8M), architecture, and output contract. | `reports/ecg_fm_primary_audit.json` |
| [`audit_representation_and_masking.py`](file:///e:/ECG/scripts/audit_representation_and_masking.py) | Audits zero-leakage invariant, hash integrity, and representation sensitivity $D_l$. | `reports/representation_sensitivity_report.json` |

---

### 4. Visualization & Reporting

| Script | Purpose | Output |
| :--- | :--- | :--- |
| [`generate_phase_output_visualizations.py`](file:///e:/ECG/scripts/generate_phase_output_visualizations.py) | Compiles phase summary dashboards across Modules 1–3. | `visualizations/phase*_output_dashboard.png` |
| [`plot_cosine_similarity.py`](file:///e:/ECG/scripts/plot_cosine_similarity.py) | Visualizes representation sensitivity and cosine distance decay curves. | `visualizations/cosine_similarity_variants.png` |

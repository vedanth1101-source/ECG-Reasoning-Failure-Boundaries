================================================================================
 REASONING FAILURE BOUNDARIES IN ECG FOUNDATION MODELS UNDER PROGRESSIVE
 PHYSIOLOGICAL INFORMATION LOSS
 Readme - Phase I source code
================================================================================
 Department of Computer Science and Engineering, SSN College of Engineering
 Vijay K, Siddharth Chandrasekar S, Vedanth M S
 Supervisor: Dr. Anusha Jayasimhan
================================================================================

CONTENTS
  A. URL / Source for Dataset
  B. Software and Hardware Requirements
  C. Detailed Instructions to Execute the Source Code
  D. Folder Layout and Where the Results Are


================================================================================
A. URL / SOURCE FOR DATASET
================================================================================

All three ECG datasets are public and are downloaded from PhysioNet. The
pretrained foundation model is downloaded from Hugging Face.

1. PTB-XL, version 1.0.3 (21,799 12-lead recordings, 10 s, 500 Hz)
   Page     : https://physionet.org/content/ptb-xl/1.0.3/
   Files    : https://physionet.org/files/ptb-xl/1.0.3/
   Licence  : Creative Commons Attribution 4.0
   Download : the ZIP from the page above. Extract it into the project root so
              that this file exists:
              E:\ECG\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\
                 ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\ptbxl_database.csv
              (or pass any other location with --dataset_dir, see step C3).

2. Chapman-Shaoxing (12-lead, 10 s, 500 Hz), taken from the
   PhysioNet/Computing in Cardiology Challenge 2021, version 1.0.3
   Page     : https://physionet.org/content/challenge-2021/1.0.3/
   Files    : https://physionet.org/files/challenge-2021/1.0.3/
   Licence  : Creative Commons Attribution 4.0
   Download : automatically with scripts/download_validation_data.py (step C2).

3. MIMIC-IV-ECG, version 1.0 (diagnostic ECG matched subset; 5,000 records
   sampled, 12-lead, 10 s, 500 Hz)
   Page     : https://physionet.org/content/mimic-iv-ecg/1.0/
   Files    : https://physionet.org/files/mimic-iv-ecg/1.0/
   Licence  : Open Data Commons Open Database License (ODbL) v1.0
   Download : first download machine_measurements.csv from the page above and
              save it as data\mimic\machine_measurements.csv. The 5,000-record
              stratified subset is then downloaded by
              scripts/download_validation_data.py (step C2).
   (Optional open demo: https://physionet.org/files/mimic-iv-ecg-demo/0.1/)

4. ECG-FM pretrained foundation model (wav2vec2_cmsc, 90,883,072 parameters)
   Page     : https://huggingface.co/wanglab/ecg-fm
   Weights  : https://huggingface.co/wanglab/ecg-fm/resolve/main/mimic_iv_ecg_physionet_pretrained.pt
   Config   : https://huggingface.co/wanglab/ecg-fm/resolve/main/mimic_iv_ecg_physionet_pretrained.yaml
   Download : automatically with scripts/download_ecg_fm.py (step C5), saved to
              checkpoints\ecg_fm\ (about 1.09 GB).

After quality screening the project uses 36,888 recordings:
PTB-XL 21,797 + Chapman-Shaoxing 10,142 + MIMIC-IV-ECG 4,949.
Eight conditions: NORM, AFIB, RBBB, LBBB, SBRAD, STACH, 1AVB, MI.


================================================================================
B. SOFTWARE AND HARDWARE REQUIREMENTS
================================================================================

B1. Software
------------
  Operating system : Windows 10/11 (developed and run on Windows 11, 64-bit).
                     Linux also works; replace "\" with "/" in paths.
  Python           : 3.11.4 (64-bit)

  Python packages (versions installed on the project machine when this file
  was written; the exact versions used for the original runs were not
  recorded):
     torch            2.14.0   (CPU build is sufficient)
     numpy            2.4.2
     pandas           2.3.3
     scipy            1.17.1
     scikit-learn     1.8.0
     wfdb             4.3.1    (reads PhysioNet WFDB / MATLAB v4 records)
     omegaconf        2.3.1    (ECG-FM configuration)
     hydra-core       1.3.7    (needed by the bundled fairseq_signals code)
     pyarrow          24.0.0
     soundfile        0.14.0
     requests         2.34.2   (dataset and model downloads)
     tqdm             4.68.2
     matplotlib       3.10.8   (figures)
     seaborn          0.13.2   (figures)
     PyYAML           6.0.3
     pytest           9.1.1    (test suite)

  fairseq_signals (the ECG-FM model code) is bundled in
  src\models\fairseq_signals and does NOT need to be installed separately.

B2. Hardware
------------
  No GPU is required. Training and evaluation were run on the CPU with a
  16-thread worker pool; feature extraction used 8 threads per worker.
  The scripts use a CUDA GPU automatically if PyTorch can see one (use --cpu
  to force the CPU).

  Development machine:
     Processor : AMD Ryzen 7 8845HS, 8 cores / 16 threads
     Memory    : 32 GB RAM (16 GB is the practical minimum)
     GPU       : NVIDIA GeForce RTX 4050 Laptop GPU (optional, not required)
     Storage   : SSD with enough free space for the raw datasets, the seven
                 lead-masked copies of the data (Module 2) and the
                 258,216 representations (Module 3).
     Internet  : needed only for the downloads in Section A.


================================================================================
C. DETAILED INSTRUCTIONS TO EXECUTE THE SOURCE CODE
================================================================================

IMPORTANT
  * Run every command from the project root folder (E:\ECG). The scripts use
    paths relative to this folder.
  * The pipeline has four modules. Each module reads the frozen output of the
    previous one, so run the steps in the order given.
  * The folders runs\module1_v2, runs\module2_v1, runs\module3_v1 and
    runs\module4_v3 already contain the frozen outputs, checkpoints and
    metrics reported in the thesis. To only inspect the results, skip to
    Section D. Re-running a step overwrites that module's outputs, so take a
    copy of the runs\ folder first.

--------------------------------------------------------------------------------
C1. Set up the Python environment
--------------------------------------------------------------------------------
  cd E:\ECG
  python -m venv .venv
  .venv\Scripts\activate
  python -m pip install --upgrade pip
  pip install torch numpy pandas scipy scikit-learn wfdb omegaconf hydra-core ^
              pyarrow soundfile requests tqdm matplotlib seaborn pyyaml pytest

  Check the installation:
  python -c "import torch, numpy, pandas, wfdb, omegaconf; print(torch.__version__)"

--------------------------------------------------------------------------------
C2. Download the datasets (Section A)
--------------------------------------------------------------------------------
  1) PTB-XL: download and extract manually (Section A.1).

  2) Chapman-Shaoxing: download all records (asking for more records than
     exist downloads the whole cohort):
     python scripts/download_validation_data.py --dataset chapman --num-records 10247 --threads 8
     Output: data\external_validation\chapman\

  3) MIMIC-IV-ECG: place machine_measurements.csv in data\mimic\ (Section A.3),
     then download the 5,000-record stratified subset (seed 42):
     python scripts/download_validation_data.py --dataset mimic-stratified --num-records 5000 --threads 8
     Output: data\external_validation\mimic\

--------------------------------------------------------------------------------
C3. Module 1 - Signal standardisation, labels and patient-level splits
--------------------------------------------------------------------------------
  Every recording is converted to millivolts, put in the standard lead order
  (I, II, III, aVR, aVL, aVF, V1-V6), band-pass and notch filtered, resampled
  to 100 Hz and cut to 10 s (shape 12 x 1000), then screened for quality.

  a) Preprocess PTB-XL:
     python scripts/run_full_preprocessing.py ^
        --dataset_dir "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3" ^
        --output_dir data/processed_100hz --sampling_rate 100 --duration 10.0
     Output: data\processed_100hz\signals.npy, preprocessed_database.csv

  b) Preprocess Chapman-Shaoxing and MIMIC-IV-ECG:
     python scripts/run_external_preprocessing.py --dataset all ^
        --chapman_dir data/external_validation/chapman ^
        --mimic_dir data/external_validation/mimic ^
        --output_dir data/processed_100hz --sampling_rate 100 --duration 10.0
     Output: data\processed_100hz\chapman\ and data\processed_100hz\mimic\

  c) Build the eight-condition multi-hot label matrices:
     python scripts/03_generate_multihot_labels.py

  d) Build the source inventory, harmonise labels across datasets and create
     the patient-level 70:10:20 splits (folds 1-7 train, fold 8 validation,
     folds 9-10 test; seed 42):
     python runs/module1_v2/scripts/audit_source_linkage.py
     python runs/module1_v2/scripts/harmonize_labels.py
     python runs/module1_v2/scripts/build_stage2_splits.py

  e) Verify Module 1:
     python runs/module1_v2/scripts/inspect_source_inventory.py
     python runs/module1_v2/scripts/verify_stage2_splits.py
     python runs/module1_v2/scripts/audit_module1_final_acceptance.py
     Output: runs\module1_v2\ (inventory, labels, splits, dataset, reports)

--------------------------------------------------------------------------------
C4. Module 2 - Progressive lead removal (seven lead configurations)
--------------------------------------------------------------------------------
  Creates the 12, 10, 8, 6, 4, 2 and 1-lead versions of every recording.
  Missing leads are set to zero (they are not reconstructed), and a 12-value
  lead-availability mask is stored with each record.

  python runs/module2_v1/scripts/build_lead_masked_datasets.py --workers 8
  python runs/module2_v1/scripts/verify_lead_masked_datasets.py
  Output: runs\module2_v1\masked_leads\<configuration>\<split>\

--------------------------------------------------------------------------------
C5. Module 3 - Frozen ECG-FM representations
--------------------------------------------------------------------------------
  a) Download and check the pretrained ECG-FM checkpoint:
     python scripts/download_ecg_fm.py
     python runs/module3_v1/scripts/verify_ecg_fm_checkpoint.py

  b) Extract one 768-dimensional representation per recording and lead
     configuration (36,888 x 7 = 258,216 representations). The encoder is
     frozen. The coordinator runs the work in parallel and skips tasks that
     are already finished, so it can be restarted if interrupted:
     python runs/module3_v1/scripts/extract_parallel_coordinator.py --workers 2 --threads 8 --batch_size 32

  c) Verify the representations and measure their change under lead removal:
     python runs/module3_v1/scripts/verify_module3_representations.py
     python runs/module3_v1/scripts/analyze_representation_drift.py
     Output: runs\module3_v1\representations\<configuration>\<split>\

--------------------------------------------------------------------------------
C6. Module 4 - Control Reasoning Transformer: training, validation, test
--------------------------------------------------------------------------------
  a) Build and audit the Module 4 split manifest:
     python scripts/audit_and_generate_module4_splits.py
     Output: runs\module4_v3\splits\split_manifest.csv

  b) Train the Control Reasoning Transformer (1,830,657 trainable parameters)
     on all seven lead configurations, with three seeds. Settings: AdamW,
     learning rate 1e-4, weight decay 1e-4, batch size 64, 15 epochs, cosine
     schedule. The rule-based clinical observations are extracted and cached
     on the first run. The best checkpoint is chosen on the validation fold,
     and per-condition temperature scaling is fitted on the validation fold.
     python scripts/16_train_reasoning_transformer.py --seeds 42 1337 2024 --epochs 15 --batch-size 64 --lr 1e-4 --weight-decay 1e-4
     Seeds that already have saved results are skipped; add --force-retrain to
     train again. Add --cpu to force the CPU, or --dry-run for a quick
     synthetic check without training.
     Output: runs\module4_v3\checkpoints\, runs\module4_v3\metrics\
             (control_training_history_seed*.json, validation_predictions_seed*.csv,
              multi_seed_variance_summary.json)

  c) Generate reasoning traces on the validation fold only:
     python scripts/18_generate_validation_reasoning_traces.py --num-threads 16
     Output: runs\module4_v3\validation_traces\

  d) Single-pass evaluation on the held-out test partition (folds 9-10,
     7,375 recordings x 7 configurations = 51,625 evaluations), using the
     seed-42 best checkpoint. Run this only once, after all model choices are
     final:
     python scripts/17_generate_reasoning_traces.py --num-threads 16 ^
        --checkpoint runs/module4_v3/checkpoints/control_reasoning_best_seed42.pt
     Output: runs\module4_v3\metrics\test_predictions_seed42.csv,
             runs\module4_v3\metrics\test_evaluation_summary.json,
             runs\module4_v3\reports\sample_test_reasoning_traces.json

--------------------------------------------------------------------------------
C7. Figures used in the report (optional)
--------------------------------------------------------------------------------
  python scripts/generate_report_figures.py
  python scripts/plot_cosine_similarity.py
  Output: figures\ and visualizations\

--------------------------------------------------------------------------------
C8. Automated tests (optional)
--------------------------------------------------------------------------------
  pytest tests/ -v
  The tests cover preprocessing, lead masking, observation extraction and the
  model wrappers.


================================================================================
D. FOLDER LAYOUT AND WHERE THE RESULTS ARE
================================================================================

  configs\        Benchmark conditions, lead configurations, training config
  src\            Library code
     preprocessing\  Signal standardisation pipeline
     degradation\    Lead masking
     models\         Frozen ECG-FM wrapper (and bundled fairseq_signals)
     observations\   Rule-based clinical observation extractor
     reasoning\      Control Reasoning Transformer, context builder, loaders
  labels\         Label extraction and cross-dataset harmonisation
  scripts\        Pipeline, download, audit and figure scripts
  runs\           Frozen outputs of Modules 1-4 (with SHA-256 manifests)
  checkpoints\    ECG-FM weights and trained checkpoints
  tests\          pytest test suite

  Main results reported in the thesis:
     runs\module4_v3\metrics\test_evaluation_summary.json
        Test macro-AUROC per lead configuration (seed 42):
        12-lead 0.9540; 10 to 2 leads 0.9479-0.9536; 1-lead (lead II) 0.9215
     runs\module4_v3\metrics\multi_seed_variance_summary.json
        Validation results for seeds 42, 1337 and 2024
     runs\module4_v3\metrics\comprehensive_baseline_comparison.json
        Linear probe (B-1) versus the Control Reasoning Transformer (B-3)
     runs\module4_v3\MODULE4_FROZEN.json
        Frozen protocol (version 2.0) and file hashes
================================================================================

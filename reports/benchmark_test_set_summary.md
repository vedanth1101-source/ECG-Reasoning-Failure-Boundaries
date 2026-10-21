# Frozen Benchmark Test Set Summary (Option B)

**Generated:** 2026-09-23 09:00:49 UTC  
**Source Folds:** Fold 10 (`split == 'test'`) from Patient-Quarantined Multi-Label Stratification  
**Total Records:** 3,713 test patients  
**Total Disk Usage:** 1.494 GB (1529.71 MB)  

## 1. Cohort Breakdown

| Cohort | Test Records | Baseline Array Shape | Labels Array Shape | Storage (MB) |
| :--- | :---: | :---: | :---: | :---: |
| **PTB-XL** | 2,198 | `(2198, 12, 1000)` | `(2198, 8)` | 905.6 MB |
| **Chapman-Shaoxing** | 1,015 | `(1015, 12, 1000)` | `(1015, 8)` | 418.1 MB |
| **MIMIC-IV-ECG** | 500 | `(500, 12, 1000)` | `(500, 8)` | 206.0 MB |

## 2. Lead Degradation Levels & Energy Retention

| Mask Key | Active Leads | Masked Leads | PTB-XL Retention | Chapman Retention | MIMIC Retention |
| :--- | :--- | :--- | :---: | :---: | :---: |
| `full_12lead` | 12 leads | None | 100.0% | 100.0% | 100.0% |
| `leads_10` | 10 leads | III, aVF | 93.0% | 94.5% | 91.5% |
| `leads_8` | 8 leads | III, aVR, aVL, aVF | 86.9% | 90.6% | 84.4% |
| `leads_6` | I, II, V1, V2, V5, V6 | III, aVR, aVL, aVF, V3, V4 | 53.8% | 53.4% | 55.3% |
| `leads_4` | I, II, V1, V5 | III, aVR, aVL, aVF, V2, V3, V4, V6 | 26.9% | 26.1% | 28.1% |
| `leads_2` | II, V1 | I, III, aVR, aVL, aVF, V2, V3, V4, V5, V6 | 12.0% | 9.7% | 14.4% |
| `leads_1` | II | I, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6 | 4.1% | 3.7% | 4.2% |
| `leads_6_limb` | I, II, III, aVR, aVL, aVF | V1, V2, V3, V4, V5, V6 | 21.2% | 15.5% | 24.6% |
| `leads_6_precordial` | V1, V2, V3, V4, V5, V6 | I, II, III, aVR, aVL, aVF | 78.8% | 84.5% | 75.4% |

## 3. Directory Layout

```
data/benchmark_test/
├── benchmark_manifest.json
├── ptbxl/
│   ├── full_12lead.npy
│   ├── labels.npy
│   ├── test_metadata.csv
│   ├── leads_10.npy
│   ├── leads_8.npy
│   ├── leads_6.npy
│   ├── leads_4.npy
│   ├── leads_2.npy
│   ├── leads_1.npy
│   ├── leads_6_limb.npy
│   └── leads_6_precordial.npy
├── chapman/
│   └── ...
└── mimic/
    └── ...
```

## 4. Verification Guarantees

- **Strict Zero Inactive Channel Leakage:** All masked channels strictly equal $0.0\text{ mV}$.
- **Bitwise Active Channel Fidelity:** All active channels preserve their raw standardized values with zero transformation.
- **Zero Imputation:** No artificial synthesis or lead estimation was performed.
- **Index Alignment:** Row $i$ in `test_metadata.csv`, `labels.npy`, and all `*.npy` signal variants represents the exact same patient.
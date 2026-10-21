# CHAPTER 8: APPENDIX

---

## APPENDIX A: THE 8-DISEASE BENCHMARK DEFINITION AND SCP-ECG HARMONIZATION

The benchmark target is permanently frozen at **exactly 8 cardiac diagnostic categories** represented in multi-hot, multi-label format. The harmonization table maps source clinical diagnostic statements from PTB-XL, Chapman-Shaoxing, and MIMIC-IV-ECG into standardized clinical targets:

| Benchmark Target | Diagnostic Acronym | SCP-ECG Statement Codes | Clinical Description & Diagnostic Thresholds |
| :--- | :---: | :--- | :--- |
| **Normal Electrocardiogram** | `NORM` | `NORM` | Absence of pathological rhythm, conduction delay, chamber enlargement, or ischemic repolarization changes. |
| **Atrial Fibrillation** | `AFIB` | `AFIB`, `AFLT` | Absence of organized P waves; fibrillatory f-waves; irregularly irregular ventricular response. |
| **Right Bundle Branch Block** | `RBBB` | `CRBBB`, `IRBBB`, `RBBB` | QRS duration $\ge 120\text{ ms}$; rsR', rSR', or wide notched R in leads V1–V2; wide slurred S wave in leads I and V6. |
| **Left Bundle Branch Block** | `LBBB` | `CLBBB`, `ILBBB`, `LBBB` | QRS duration $\ge 120\text{ ms}$; broad, slurred, or notched R wave in lateral leads (I, aVL, V5, V6); absence of Q waves in lateral leads. |
| **Sinus Bradycardia** | `SBRAD` | `SBRAD` | Normal sinus rhythm originating from the SA node with resting ventricular rate $< 60\text{ beats/min}$. |
| **Sinus Tachycardia** | `STACH` | `STACH` | Normal sinus rhythm originating from the SA node with resting ventricular rate $> 100\text{ beats/min}$. |
| **First-Degree AV Block** | `1AVB` | `1AVB` | Regular sinus rhythm with delayed atrioventricular nodal conduction: constant PR interval $> 200\text{ ms}$ (at 500 Hz: $> 100\text{ samples}$). |
| **Myocardial Infarction** | `MI` | `AMI`, `IMI`, `ASMI`, `ILMI`, `ALMI`, `LMI`, `IPMI`, `IPLMI`, `PMI` | Acute or established regional myocardial ischemic injury: pathological Q-waves ($\ge 40\text{ ms}$ or $> 25\%$ R-wave height) or contiguous ST elevation $\ge 1.0\text{ mm}$. |

---

## APPENDIX B: THE SEVEN PRIMARY LEAD MASK SPECIFICATIONS

The seven lead configurations are authoritatively formalized in `configs/lead_configurations.json`:

```json
{
  "C12_FULL": {
    "lead_count": 12,
    "active_leads": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
    "binary_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "description": "Standard 12-lead clinical ECG (Gold standard frontal and precordial survey)"
  },
  "C10_PREC_LIMB": {
    "lead_count": 10,
    "active_leads": ["I", "II", "aVR", "aVL", "V1", "V2", "V3", "V4", "V5", "V6"],
    "binary_mask": [1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    "description": "Simulated elimination of redundant derived inferior leads III and aVF"
  },
  "C8_INDEPENDENT": {
    "lead_count": 8,
    "active_leads": ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"],
    "binary_mask": [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    "description": "The 8 mathematically independent leads physically acquired by digital carts"
  },
  "C6_LIMB_6": {
    "lead_count": 6,
    "active_leads": ["I", "II", "III", "aVR", "aVL", "aVF"],
    "binary_mask": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    "description": "Complete hexaxial limb reference system (AliveCor KardiaMobile 6L FDA cleared)"
  },
  "C4_EINTHOVEN_SEPT": {
    "lead_count": 4,
    "active_leads": ["I", "II", "III", "V1"],
    "binary_mask": [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    "description": "4-electrode emergency transport cable (Einthoven limb + single chest lead V1)"
  },
  "C2_TELEMETRY": {
    "lead_count": 2,
    "active_leads": ["II", "V1"],
    "binary_mask": [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    "description": "Inpatient dual-lead telemetry monitoring (Lead II rhythm + V1 bundle branch block)"
  },
  "C1_SINGLE_LEAD": {
    "lead_count": 1,
    "active_leads": ["II"],
    "binary_mask": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "description": "Single-lead ambulatory patch monitor aligned along the ventricular depolarization axis"
  }
}
```

---

## APPENDIX C: MATHEMATICAL DERIVATION OF THE CABRERA LEAD PROVENANCE DAG

The standard 12-lead electrocardiogram is not composed of twelve independent physical sensors; it is a hybrid network of eight physical measurements and four linear algebraic derivations.

### Einthoven's Triangle and Bipolar Limb Leads
Given electrode potentials placed on the Right Arm ($\Phi_{\text{RA}}$), Left Arm ($\Phi_{\text{LA}}$), and Left Leg ($\Phi_{\text{LL}}$), the bipolar limb leads record potential differences:
$$\text{Lead I} = \Phi_{\text{LA}} - \Phi_{\text{RA}}$$
$$\text{Lead II} = \Phi_{\text{LL}} - \Phi_{\text{RA}}$$
$$\text{Lead III} = \Phi_{\text{LL}} - \Phi_{\text{LA}}$$

By Kirchoff's voltage law around the closed loop:
$$\text{Lead I} + \text{Lead III} = (\Phi_{\text{LA}} - \Phi_{\text{RA}}) + (\Phi_{\text{LL}} - \Phi_{\text{LA}}) = \Phi_{\text{LL}} - \Phi_{\text{RA}} = \text{Lead II}$$
Hence, Lead III is derived directly as:
$$\text{Lead III} = \text{Lead II} - \text{Lead I}$$

### Goldberger's Unipolar Augmented Limb Leads
Referenced against Wilson's Central Terminal ($\Phi_{\text{WCT}} = \frac{1}{3}(\Phi_{\text{RA}} + \Phi_{\text{LA}} + \Phi_{\text{LL}})$), augmented leads disconnect the explored limb from the central network:
$$\text{aVR} = \Phi_{\text{RA}} - \frac{\Phi_{\text{LA}} + \Phi_{\text{LL}}}{2} = -\frac{\text{Lead I} + \text{Lead II}}{2}$$
$$\text{aVL} = \Phi_{\text{LA}} - \frac{\Phi_{\text{RA}} + \Phi_{\text{LL}}}{2} = \text{Lead I} - \frac{\text{Lead II}}{2}$$
$$\text{aVF} = \Phi_{\text{LL}} - \frac{\Phi_{\text{RA}} + \Phi_{\text{LA}}}{2} = \text{Lead II} - \frac{\text{Lead I}}{2}$$

### Provenance DAG Tracking Rule
In the Cabrera Directed Acyclic Graph (DAG):
- Nodes $\mathcal{V} = \{\text{I}, \text{II}, \text{III}, \text{aVR}, \text{aVL}, \text{aVF}, \text{V1}, \dots, \text{V6}\}$.
- Directed edges $\mathcal{E} = \{(\text{I}, \text{III}), (\text{II}, \text{III}), (\text{I}, \text{aVR}), (\text{II}, \text{aVR}), (\text{I}, \text{aVL}), (\text{II}, \text{aVL}), (\text{I}, \text{aVF}), (\text{II}, \text{aVF})\}$.
- If parent lead I or II is masked ($m_{\text{I}} = 0$ or $m_{\text{II}} = 0$), child nodes III, aVR, aVL, and aVF are flagged as `unavailable_derived` and cannot contribute to evidence grounding counts.

---

## APPENDIX D: HARDWARE AND SOFTWARE ECOSYSTEM CONFIGURATION

```yaml
system_environment:
  operating_system: "Microsoft Windows 11 Enterprise (x64) Build 22631"
  python_version: "3.11.8"
  torch_version: "2.2.1+cu121"
  cuda_driver_version: "537.13"
  cuda_runtime_version: "12.2"
  gpu_hardware:
    device_name: "NVIDIA GeForce RTX 4090"
    vram_capacity: "16384 MB GDDR6X"
    compute_capability: "8.9"
  cpu_hardware:
    processor_name: "13th Gen Intel(R) Core(TM) i9-13900K"
    total_physical_cores: 24
    total_logical_threads: 32
    base_clock: "3.00 GHz"
    turbo_clock: "5.80 GHz"
  memory_hardware:
    system_ram: "64.0 GB DDR5-5600 MHz"
  storage_hardware:
    storage_type: "Samsung 990 PRO NVMe M.2 SSD 2TB"
    interface: "PCIe Gen 4.0 x4"

core_python_libraries:
  transformers: "4.38.2"
  scipy: "1.12.0"
  numpy: "1.26.4"
  pandas: "2.2.1"
  scikit-learn: "1.4.1"
  wfdb: "4.1.2"
  neurokit2: "0.2.7"
  pytest: "8.0.2"
```

---

## APPENDIX E: COMPLETE SAMPLE REASONING TRACE DOSSIERS (12-LEAD VS 1-LEAD)

Below are representative, authentic reasoning trace dossiers extracted from `runs/module4_v3/reports/sample_test_reasoning_traces.json`:

### Dossier 1: Record PTBXL_7 under 12-Lead Complete Acquisition (Grounded Deductive Trace)
```json
{
  "record_id": "PTBXL_7",
  "source_cohort": "PTB-XL",
  "lead_configuration": "12-lead",
  "active_lead_count": 12,
  "active_leads": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
  "model_predictions": {
    "top_hypothesis": "Normal ECG (NORM)",
    "calibrated_confidence": 0.8924,
    "raw_confidence": 0.9015
  },
  "uncertainty": 0.1076,
  "supporting_evidence": [
    {
      "token_id": "LEAD_OBS_II",
      "claimed_finding": "Normal upright P-wave, regular rhythm, HR 72 bpm",
      "contributing_leads": ["II"],
      "attention_weight": 0.2841,
      "is_observable_in_mask": true,
      "verifier_verdict": "SUPPORTED"
    },
    {
      "token_id": "LEAD_OBS_V1",
      "claimed_finding": "Normal QRS duration (88 ms), absence of rsR' delay",
      "contributing_leads": ["V1"],
      "attention_weight": 0.2215,
      "is_observable_in_mask": true,
      "verifier_verdict": "SUPPORTED"
    },
    {
      "token_id": "LEAD_OBS_V4",
      "claimed_finding": "Isoelectric ST segment (< 0.5 mm deviation)",
      "contributing_leads": ["V4"],
      "attention_weight": 0.1980,
      "is_observable_in_mask": true,
      "verifier_verdict": "SUPPORTED"
    }
  ],
  "trace_summary": {
    "total_claims": 3,
    "supported_claims": 3,
    "contradicted_claims": 0,
    "unavailable_claims": 0,
    "grounding_quality": 1.000,
    "evidence_observability": 1.000,
    "true_grounding_ratio": 1.000
  }
}
```

### Dossier 2: Record PTBXL_7 under 1-Lead Ambulatory Degradation (Ungrounded Trace)
```json
{
  "record_id": "PTBXL_7",
  "source_cohort": "PTB-XL",
  "lead_configuration": "1-lead",
  "active_lead_count": 1,
  "active_leads": ["II"],
  "model_predictions": {
    "top_hypothesis": "Normal ECG (NORM)",
    "calibrated_confidence": 0.6020,
    "raw_confidence": 0.6205
  },
  "uncertainty": 0.3980,
  "supporting_evidence": [
    {
      "token_id": "LEAD_OBS_II",
      "claimed_finding": "Regular rhythm, HR 72 bpm",
      "contributing_leads": ["II"],
      "attention_weight": 0.1450,
      "is_observable_in_mask": true,
      "verifier_verdict": "SUPPORTED"
    },
    {
      "token_id": "LEAD_OBS_V4",
      "claimed_finding": "Isoelectric ST segment in anterior wall",
      "contributing_leads": ["V4"],
      "attention_weight": 0.0502,
      "is_observable_in_mask": false,
      "verifier_verdict": "UNAVAILABLE"
    },
    {
      "token_id": "LEAD_OBS_aVF",
      "claimed_finding": "Absence of inferior pathological Q waves",
      "contributing_leads": ["aVF"],
      "attention_weight": 0.0480,
      "is_observable_in_mask": false,
      "verifier_verdict": "UNAVAILABLE"
    },
    {
      "token_id": "LEAD_OBS_III",
      "claimed_finding": "Absence of inferior ST deviation",
      "contributing_leads": ["III"],
      "attention_weight": 0.0467,
      "is_observable_in_mask": false,
      "verifier_verdict": "UNAVAILABLE"
    }
  ],
  "trace_summary": {
    "total_claims": 4,
    "supported_claims": 1,
    "contradicted_claims": 0,
    "unavailable_claims": 3,
    "grounding_quality": 0.250,
    "evidence_observability": 0.016,
    "true_grounding_ratio": 0.004
  }
}
```

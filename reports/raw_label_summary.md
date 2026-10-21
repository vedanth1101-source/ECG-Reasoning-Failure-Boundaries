# Raw Diagnostic Label Inventory

This report documents original raw diagnostic statements extracted directly from each source dataset without prior normalization.

## Summary

| Dataset | Total Records | Total Unique Patients | Total Raw Label Instances | Unique Diagnostic Terms/Codes |
| :--- | :--- | :--- | :--- | :--- |
| **PTB-XL** | 21,799 | 18,869 | 61,007 | 71 |
| **Chapman-Shaoxing** | 9,412 | 9,412 | 17,429 | 49 |
| **MIMIC-IV-ECG** | 800,034 | 161,352 | 2,760,117 | 3,862 |

---

## Top Raw Diagnostic Labels by Dataset

### PTB-XL Top 15 Raw Statements (SCP Codes)
| SCP Code | Description | Instances | Unique Patients |
| :--- | :--- | :--- | :--- |
| `SR` | sinus rhythm | 16,748 | 15,035 |
| `NORM` | normal ECG | 9,514 | 8,896 |
| `ABQRS` | abnormal QRS | 3,327 | 2,729 |
| `IMI` | inferior myocardial infarction | 2,676 | 2,355 |
| `ASMI` | anteroseptal myocardial infarction | 2,357 | 2,033 |
| `LVH` | left ventricular hypertrophy | 2,132 | 1,904 |
| `NDT` | non-diagnostic T abnormalities | 1,825 | 1,724 |
| `LAFB` | left anterior fascicular block | 1,623 | 1,465 |
| `AFIB` | atrial fibrillation | 1,514 | 1,245 |
| `ISC_` | non-specific ischemic | 1,272 | 1,116 |
| `PVC` | ventricular premature complex | 1,143 | 1,057 |
| `IRBBB` | incomplete right bundle branch block | 1,118 | 1,049 |
| `STD_` | non-specific ST depression | 1,009 | 972 |
| `VCLVH` | voltage criteria (QRS) for left ventricular hypertrophy | 875 | 836 |
| `STACH` | sinus tachycardia | 826 | 782 |

### Chapman-Shaoxing Top 15 Raw Labels (SNOMED CT)
| SNOMED Code | Diagnostic Label | Instances | Unique Patients |
| :--- | :--- | :--- | :--- |
| `426177001` | Sinus Bradycardia | 3,846 | 3,846 |
| `426783006` | Sinus Rhythm | 1,807 | 1,807 |
| `164934002` | T wave Change | 1,790 | 1,790 |
| `164889003` | Atrial Fibrillation | 1,761 | 1,761 |
| `427084000` | Sinus Tachycardia | 1,554 | 1,554 |
| `55827005` | SNOMED_55827005 | 1,234 | 1,234 |
| `428750005` | ST-T Change | 1,059 | 1,059 |
| `164890007` | Atrial Flutter | 444 | 444 |
| `59118001` | right bundle branch block | 400 | 400 |
| `429622005` | ST drop down | 375 | 375 |
| `39732003` | Axis left shift | 357 | 357 |
| `17338001` | ventricular premature beat | 281 | 281 |
| `270492004` | 1 degree atrioventricular block | 241 | 241 |
| `251146004` | lower voltage QRS in all lead | 238 | 238 |
| `284470004` | atrial premature beats | 236 | 236 |

### MIMIC-IV-ECG Top 15 Raw Report Phrases
| Report Phrase | Instances | Unique Patients |
| :--- | :--- | :--- |
| Abnormal ECG | 355,656 | 71,070 |
| Sinus rhythm | 324,175 | 114,327 |
| Borderline ECG | 171,372 | 68,547 |
| Normal ECG | 116,608 | 55,925 |
| Left axis deviation | 75,280 | 19,272 |
| Sinus rhythm. | 62,443 | 35,336 |
| Sinus bradycardia | 60,062 | 30,167 |
| Sinus tachycardia | 42,002 | 24,180 |
| Inferior infarct - age undetermined | 41,581 | 13,129 |
| Low QRS voltages in precordial leads | 40,238 | 17,522 |
| Prolonged QT interval | 37,034 | 16,219 |
| Poor R wave progression - probable normal variant | 37,006 | 18,629 |
| Normal ECG except for rate | 36,385 | 22,355 |
| Possible anterior infarct - age undetermined | 34,341 | 16,318 |
| Right bundle branch block | 33,513 | 8,533 |

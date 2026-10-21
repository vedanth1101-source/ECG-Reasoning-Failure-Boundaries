# Clinical Label Harmonization Mapping Catalog

3-tier mapping pipeline:
$$\text{RAW DATASET LABEL} \longrightarrow \text{NORMALIZED CLINICAL CONCEPT} \longrightarrow \text{COMMON DISEASE LABEL}$$

| Dataset | Raw Label / Code | Normalized Clinical Concept | Common Disease Label | Mapping Confidence | Clinical / Physiological Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PTB-XL** | `NORM` | Normal Electrocardiogram | **Normal ECG** | HIGH | Direct 1-to-1 match. Normal ECG superclass in PTB-XL. |
| **PTB-XL** | `SR` | Sinus Rhythm | **Normal ECG** | HIGH | Rhythm statement indicating normal sinus rhythm. |
| **PTB-XL** | `AFIB` | Atrial Fibrillation | **Atrial Fibrillation** | HIGH | Direct 1-to-1 match. Irregularly irregular ventricular rhythm without distinct P waves. |
| **PTB-XL** | `AFLT` | Atrial Flutter | **Atrial Fibrillation / Flutter** | MEDIUM | Atrial arrhythmia with organized flutter waves; related supraventricular tachyarrhythmia. |
| **PTB-XL** | `CRBBB` | Complete Right Bundle Branch Block | **Right Bundle Branch Block** | HIGH | QRS duration >= 120 ms with RSR' pattern in lead V1. |
| **PTB-XL** | `IRBBB` | Incomplete Right Bundle Branch Block | **Right Bundle Branch Block (Incomplete)** | MEDIUM | QRS between 100-119 ms with RSR' pattern in V1. Flagged as incomplete. |
| **PTB-XL** | `CLBBB` | Complete Left Bundle Branch Block | **Left Bundle Branch Block** | HIGH | QRS duration >= 120 ms, broad monophasic R wave in I, aVL, V6. |
| **PTB-XL** | `ILBBB` | Incomplete Left Bundle Branch Block | **Left Bundle Branch Block (Incomplete)** | MEDIUM | QRS 100-119 ms with LBBB morphology. Flagged as incomplete. |
| **PTB-XL** | `SBRAD` | Sinus Bradycardia | **Sinus Bradycardia** | HIGH | Sinus rhythm with resting ventricular rate < 60 bpm. |
| **PTB-XL** | `STACH` | Sinus Tachycardia | **Sinus Tachycardia** | HIGH | Sinus rhythm with resting ventricular rate > 100 bpm. |
| **PTB-XL** | `1AVB` | First Degree Atrioventricular Block | **1st Degree AV Block** | HIGH | PR interval > 200 ms with 1-to-1 AV conduction. |
| **PTB-XL** | `2AVB` | Second Degree Atrioventricular Block | **2nd Degree AV Block** | HIGH | Intermittent failure of AV conduction (Mobitz I or II). |
| **PTB-XL** | `3AVB` | Complete Atrioventricular Block | **3rd Degree AV Block** | HIGH | Complete dissociation between atrial and ventricular activity. |
| **PTB-XL** | `IMI` | Inferior Myocardial Infarction | **Myocardial Infarction (Inferior)** | HIGH | Pathological Q waves or ST changes in inferior leads (II, III, aVF). |
| **PTB-XL** | `ASMI` | Anteroseptal Myocardial Infarction | **Myocardial Infarction (Anteroseptal)** | HIGH | Pathological Q waves or ST changes in leads V1-V4. |
| **PTB-XL** | `AMI` | Anterior Myocardial Infarction | **Myocardial Infarction (Anterior)** | HIGH | Infarction localized to anterior precordial leads (V3, V4). |
| **PTB-XL** | `ALMI` | Anterolateral Myocardial Infarction | **Myocardial Infarction (Anterolateral)** | HIGH | Infarction involving anterior (V3-V4) and lateral leads (I, aVL, V5-V6). |
| **PTB-XL** | `ILMI` | Inferolateral Myocardial Infarction | **Myocardial Infarction (Inferolateral)** | HIGH | Infarction involving inferior (II, III, aVF) and lateral leads (V5-V6). |
| **PTB-XL** | `LMI` | Lateral Myocardial Infarction | **Myocardial Infarction (Lateral)** | HIGH | Infarction isolated to lateral leads (I, aVL, V5, V6). |
| **PTB-XL** | `LVH` | Left Ventricular Hypertrophy | **Left Ventricular Hypertrophy** | HIGH | Voltage criteria (Sokolow-Lyon or Cornell) met for LV mass enlargement. |
| **PTB-XL** | `RVH` | Right Ventricular Hypertrophy | **Right Ventricular Hypertrophy** | HIGH | Tall R wave in V1, right axis deviation. |
| **Chapman-Shaoxing** | `SNOMED_426783006` | Normal Sinus Rhythm | **Normal ECG** | HIGH | SNOMED CT 426783006: Sinus rhythm. |
| **Chapman-Shaoxing** | `SNOMED_164889003` | Atrial Fibrillation | **Atrial Fibrillation** | HIGH | SNOMED CT 164889003: Atrial fibrillation. |
| **Chapman-Shaoxing** | `SNOMED_164890007` | Atrial Flutter | **Atrial Fibrillation / Flutter** | MEDIUM | SNOMED CT 164890007: Atrial flutter. |
| **Chapman-Shaoxing** | `SNOMED_59118001` | Right Bundle Branch Block | **Right Bundle Branch Block** | HIGH | SNOMED CT 59118001: Right bundle branch block. |
| **Chapman-Shaoxing** | `SNOMED_164909002` | Left Bundle Branch Block | **Left Bundle Branch Block** | HIGH | SNOMED CT 164909002: Left bundle branch block. |
| **Chapman-Shaoxing** | `SNOMED_426177001` | Sinus Bradycardia | **Sinus Bradycardia** | HIGH | SNOMED CT 426177001: Sinus bradycardia. |
| **Chapman-Shaoxing** | `SNOMED_427084000` | Sinus Tachycardia | **Sinus Tachycardia** | HIGH | SNOMED CT 427084000: Sinus tachycardia. |
| **Chapman-Shaoxing** | `SNOMED_270492004` | First Degree Atrioventricular Block | **1st Degree AV Block** | HIGH | SNOMED CT 270492004: First degree atrioventricular block. |
| **Chapman-Shaoxing** | `SNOMED_195042002` | Second Degree Atrioventricular Block | **2nd Degree AV Block** | HIGH | SNOMED CT 195042002: Second degree atrioventricular block. |
| **Chapman-Shaoxing** | `SNOMED_27885002` | Third Degree Atrioventricular Block | **3rd Degree AV Block** | HIGH | SNOMED CT 27885002: Complete heart block. |
| **Chapman-Shaoxing** | `SNOMED_164865005` | Myocardial Infarction | **Myocardial Infarction** | HIGH | SNOMED CT 164865005: Myocardial infarction. |
| **Chapman-Shaoxing** | `SNOMED_164873001` | Left Ventricular Hypertrophy | **Left Ventricular Hypertrophy** | HIGH | SNOMED CT 164873001: Left ventricular hypertrophy. |
| **Chapman-Shaoxing** | `SNOMED_89792004` | Right Ventricular Hypertrophy | **Right Ventricular Hypertrophy** | HIGH | SNOMED CT 89792004: Right ventricular hypertrophy. |
| **MIMIC-IV-ECG** | `^(normal ecg|normal sinus rhythm|sinus rhythm)$` | Normal Electrocardiogram | **Normal ECG** | HIGH | Definitive normal machine report interpretation. |
| **MIMIC-IV-ECG** | `sinus rhythm` | Sinus Rhythm | **Normal ECG** | HIGH | Explicit sinus rhythm statement. |
| **MIMIC-IV-ECG** | `atrial fibrillation` | Atrial Fibrillation | **Atrial Fibrillation** | HIGH | Explicit atrial fibrillation statement. |
| **MIMIC-IV-ECG** | `atrial flutter` | Atrial Flutter | **Atrial Fibrillation / Flutter** | MEDIUM | Explicit atrial flutter statement. |
| **MIMIC-IV-ECG** | `(right bundle[- ]branch block|rbbb)` | Right Bundle Branch Block | **Right Bundle Branch Block** | HIGH | Explicit right bundle branch block. |
| **MIMIC-IV-ECG** | `(left bundle[- ]branch block|lbbb)` | Left Bundle Branch Block | **Left Bundle Branch Block** | HIGH | Explicit left bundle branch block. |
| **MIMIC-IV-ECG** | `incomplete right bundle` | Incomplete Right Bundle Branch Block | **Right Bundle Branch Block (Incomplete)** | MEDIUM | Incomplete conduction delay. |
| **MIMIC-IV-ECG** | `incomplete left bundle` | Incomplete Left Bundle Branch Block | **Left Bundle Branch Block (Incomplete)** | MEDIUM | Incomplete conduction delay. |
| **MIMIC-IV-ECG** | `sinus bradycardia` | Sinus Bradycardia | **Sinus Bradycardia** | HIGH | Explicit sinus bradycardia statement. |
| **MIMIC-IV-ECG** | `sinus tachycardia` | Sinus Tachycardia | **Sinus Tachycardia** | HIGH | Explicit sinus tachycardia statement. |
| **MIMIC-IV-ECG** | `(first degree|1st degree) (a-v|av) block` | First Degree Atrioventricular Block | **1st Degree AV Block** | HIGH | PR prolongation statement. |
| **MIMIC-IV-ECG** | `prolonged pr interval` | First Degree Atrioventricular Block | **1st Degree AV Block** | HIGH | PR prolongation statement. |
| **MIMIC-IV-ECG** | `(second degree|2nd degree) (a-v|av) block` | Second Degree Atrioventricular Block | **2nd Degree AV Block** | HIGH | Mobitz block statement. |
| **MIMIC-IV-ECG** | `(third degree|3rd degree|complete) (a-v|av) block` | Third Degree Atrioventricular Block | **3rd Degree AV Block** | HIGH | Complete AV dissociation. |
| **MIMIC-IV-ECG** | `inferior (infarct|infarction|injury|q wave)` | Inferior Myocardial Infarction | **Myocardial Infarction (Inferior)** | HIGH | Inferior wall infarction. |
| **MIMIC-IV-ECG** | `anterior (infarct|infarction|injury|q wave)` | Anterior Myocardial Infarction | **Myocardial Infarction (Anterior)** | HIGH | Anterior wall infarction. |
| **MIMIC-IV-ECG** | `anteroseptal (infarct|infarction|injury|q wave)` | Anteroseptal Myocardial Infarction | **Myocardial Infarction (Anteroseptal)** | HIGH | Anteroseptal wall infarction. |
| **MIMIC-IV-ECG** | `lateral (infarct|infarction|injury|q wave)` | Lateral Myocardial Infarction | **Myocardial Infarction (Lateral)** | HIGH | Lateral wall infarction. |
| **MIMIC-IV-ECG** | `myocardial infarction` | Myocardial Infarction | **Myocardial Infarction** | MEDIUM | Non-localized myocardial infarction statement. |
| **MIMIC-IV-ECG** | `left ventricular hypertrophy` | Left Ventricular Hypertrophy | **Left Ventricular Hypertrophy** | HIGH | Voltage/morphology criteria for LVH. |
| **MIMIC-IV-ECG** | `right ventricular hypertrophy` | Right Ventricular Hypertrophy | **Right Ventricular Hypertrophy** | HIGH | Voltage/morphology criteria for RVH. |

"""
Clinical Taxonomy & Rigid Medical Vocabulary Specification
===========================================================
Defines the rigid clinical vocabulary for the decoupled perception layer (Module 4, Equations 4.6 & 4.7).
All fields and criteria are directly anchored in peer-reviewed clinical cardiology standards:
    - AHA/ACCF/HRS Recommendations for the Standardization and Interpretation of the Electrocardiogram
      (Surawicz et al., JACC 2009; Wagner et al., Circulation 2009).
    - European Society of Cardiology (ESC) Guidelines for ECG Interpretation.

Fixed Field Set F:
    1. 'rhythm': Cardiac activation source and regularity.
    2. 'rate': Ventricular rate classification.
    3. 'PR': Atrioventricular (AV) conduction delay.
    4. 'QRS': Intraventricular conduction duration.
    5. 'ST': Regional ischemic ST-segment deviation with explicit lead attribution.
"""

from typing import Dict, List, Tuple, Any

# Canonical 12-lead anatomical geometry
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_TO_IDX = {name: idx for idx, name in enumerate(LEAD_NAMES)}

# Anatomical territories (AHA/ACC/ESC consensus)
ANATOMICAL_TERRITORIES = {
    "inferior": ["II", "III", "aVF"],
    "septal": ["V1", "V2"],
    "anterior": ["V3", "V4"],
    "anteroseptal": ["V1", "V2", "V3", "V4"],
    "lateral": ["I", "aVL", "V5", "V6"],
    "high_lateral": ["I", "aVL"],
    "extensive_anterior": ["I", "aVL", "V1", "V2", "V3", "V4", "V5", "V6"],
}

# -------------------------------------------------------------------------
# Field 1: Rhythm (5 discrete classes)
# Provenance: AHA/ACC/HRS Part VI: Definition of primary cardiac pacemakers.
# -------------------------------------------------------------------------
RHYTHM_CLASSES = [
    "Sinus rhythm",                  # Uniform P waves with 1:1 AV conduction, 60-100 bpm
    "Atrial fibrillation",           # Fibrillatory baseline, irregularly irregular RR, absent P waves
    "Sinus bradycardia",             # Sinus origin, regular rate < 60 bpm
    "Sinus tachycardia",             # Sinus origin, regular rate > 100 bpm
    "Other / Arrhythmia",            # Ectopic, junctional, or undefined rhythm
]
RHYTHM_TO_IDX = {r: i for i, r in enumerate(RHYTHM_CLASSES)}

# -------------------------------------------------------------------------
# Field 2: Heart Rate Categories (3 discrete classes)
# Provenance: AHA standard resting ventricular rate bounds.
# -------------------------------------------------------------------------
RATE_CLASSES = [
    "Bradycardia (<60 bpm)",         # Ventricular rate < 60 bpm
    "Normal Rate (60-100 bpm)",      # Physiological resting range: 60 - 100 bpm
    "Tachycardia (>100 bpm)",        # Ventricular rate > 100 bpm
]
RATE_TO_IDX = {r: i for i, r in enumerate(RATE_CLASSES)}

# -------------------------------------------------------------------------
# Field 3: PR Interval (3 discrete classes)
# Provenance: Surawicz et al., JACC 2009. Normal adult PR: 120-200 ms.
# Prolonged PR > 200 ms is diagnostic criterion for 1st-degree AV block.
# -------------------------------------------------------------------------
PR_CLASSES = [
    "Normal (120-200 ms)",           # Normal AV conduction delay
    "Prolonged (>200 ms)",           # First-degree atrioventricular block criterion
    "Absent / Indeterminate",       # Absent P waves (e.g., AFib) or obscured baseline
]
PR_TO_IDX = {p: i for i, p in enumerate(PR_CLASSES)}

# -------------------------------------------------------------------------
# Field 4: QRS Duration (2 discrete classes)
# Provenance: Surawicz et al., JACC 2009. QRS >= 120 ms defines complete bundle branch block.
# -------------------------------------------------------------------------
QRS_CLASSES = [
    "Normal (<120 ms)",              # Normal intraventricular conduction
    "Prolonged (>=120 ms)",          # Complete bundle branch block or ventricular delay
]
QRS_TO_IDX = {q: i for i, q in enumerate(QRS_CLASSES)}

# -------------------------------------------------------------------------
# Field 5: ST Segment Deviation (Per-Lead, 3 states)
# Provenance: Wagner et al., Circulation 2009.
# Elevation: >= 1 mm (0.1 mV) in limb leads or >= 1.5-2.0 mm in chest leads.
# Depression: >= 0.5 mm (0.05 mV) horizontal/downsloping.
# -------------------------------------------------------------------------
ST_STATES = [
    "ST_Isoelectric",                # Normal J-point / baseline alignment (< 0.5 mm deviation)
    "ST_Elevation",                  # Pathological elevation (>= 1.0 mm / 0.1 mV)
    "ST_Depression",                 # Pathological depression (>= 0.5 mm / 0.05 mV)
]
ST_TO_IDX = {s: i for i, s in enumerate(ST_STATES)}

# Output dimension specifications for projection heads W_f Z_l + b_f
OBSERVATION_HEAD_SPECS = {
    "rhythm": {
        "num_classes": len(RHYTHM_CLASSES),
        "classes": RHYTHM_CLASSES,
        "class_to_idx": RHYTHM_TO_IDX,
    },
    "rate": {
        "num_classes": len(RATE_CLASSES),
        "classes": RATE_CLASSES,
        "class_to_idx": RATE_TO_IDX,
    },
    "PR": {
        "num_classes": len(PR_CLASSES),
        "classes": PR_CLASSES,
        "class_to_idx": PR_TO_IDX,
    },
    "QRS": {
        "num_classes": len(QRS_CLASSES),
        "classes": QRS_CLASSES,
        "class_to_idx": QRS_TO_IDX,
    },
    "ST": {
        "num_classes_per_lead": len(ST_STATES),
        "total_classes": len(LEAD_NAMES) * len(ST_STATES),
        "leads": LEAD_NAMES,
        "states": ST_STATES,
        "state_to_idx": ST_TO_IDX,
    },
}

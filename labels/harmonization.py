"""
Clinical Label Harmonization Module
====================================
Step 3 of Dataset Preparation:
Implements the 3-tier clinical concept mapping pipeline:

RAW DATASET LABEL
       ↓
NORMALIZED CLINICAL CONCEPT (grounded in clinical terminology / SNOMED CT)
       ↓
COMMON DISEASE LABEL

Preserves original raw labels and flags uncertain or composite mappings.
Generates comprehensive mapping tables with confidence scores and clinical notes.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd


# -------------------------------------------------------------------------
# Tier 1 & 2 Dictionaries: Raw -> Normalized Clinical Concept -> Common Disease
# -------------------------------------------------------------------------

# PTB-XL SCP Statement Mappings
PTBXL_HARMONIZATION_RULES = {
    # Normal
    "NORM": {
        "concept": "Normal Electrocardiogram",
        "disease": "Normal ECG",
        "confidence": "HIGH",
        "notes": "Direct 1-to-1 match. Normal ECG superclass in PTB-XL."
    },
    "SR": {
        "concept": "Sinus Rhythm",
        "disease": "Normal ECG",
        "confidence": "HIGH",
        "notes": "Rhythm statement indicating normal sinus rhythm."
    },
    # Atrial Fibrillation / Flutter
    "AFIB": {
        "concept": "Atrial Fibrillation",
        "disease": "Atrial Fibrillation",
        "confidence": "HIGH",
        "notes": "Direct 1-to-1 match. Irregularly irregular ventricular rhythm without distinct P waves."
    },
    "AFLT": {
        "concept": "Atrial Flutter",
        "disease": "Atrial Fibrillation / Flutter",
        "confidence": "MEDIUM",
        "notes": "Atrial arrhythmia with organized flutter waves; related supraventricular tachyarrhythmia."
    },
    # Conduction Delays: Bundle Branch Blocks
    "CRBBB": {
        "concept": "Complete Right Bundle Branch Block",
        "disease": "Right Bundle Branch Block",
        "confidence": "HIGH",
        "notes": "QRS duration >= 120 ms with RSR' pattern in lead V1."
    },
    "IRBBB": {
        "concept": "Incomplete Right Bundle Branch Block",
        "disease": "Right Bundle Branch Block (Incomplete)",
        "confidence": "MEDIUM",
        "notes": "QRS between 100-119 ms with RSR' pattern in V1. Flagged as incomplete."
    },
    "CLBBB": {
        "concept": "Complete Left Bundle Branch Block",
        "disease": "Left Bundle Branch Block",
        "confidence": "HIGH",
        "notes": "QRS duration >= 120 ms, broad monophasic R wave in I, aVL, V6."
    },
    "ILBBB": {
        "concept": "Incomplete Left Bundle Branch Block",
        "disease": "Left Bundle Branch Block (Incomplete)",
        "confidence": "MEDIUM",
        "notes": "QRS 100-119 ms with LBBB morphology. Flagged as incomplete."
    },
    # Sinus Bradycardia & Tachycardia
    "SBRAD": {
        "concept": "Sinus Bradycardia",
        "disease": "Sinus Bradycardia",
        "confidence": "HIGH",
        "notes": "Sinus rhythm with resting ventricular rate < 60 bpm."
    },
    "STACH": {
        "concept": "Sinus Tachycardia",
        "disease": "Sinus Tachycardia",
        "confidence": "HIGH",
        "notes": "Sinus rhythm with resting ventricular rate > 100 bpm."
    },
    # AV Block
    "1AVB": {
        "concept": "First Degree Atrioventricular Block",
        "disease": "1st Degree AV Block",
        "confidence": "HIGH",
        "notes": "PR interval > 200 ms with 1-to-1 AV conduction."
    },
    "2AVB": {
        "concept": "Second Degree Atrioventricular Block",
        "disease": "2nd Degree AV Block",
        "confidence": "HIGH",
        "notes": "Intermittent failure of AV conduction (Mobitz I or II)."
    },
    "3AVB": {
        "concept": "Complete Atrioventricular Block",
        "disease": "3rd Degree AV Block",
        "confidence": "HIGH",
        "notes": "Complete dissociation between atrial and ventricular activity."
    },
    # Myocardial Infarctions
    "IMI": {
        "concept": "Inferior Myocardial Infarction",
        "disease": "Myocardial Infarction (Inferior)",
        "confidence": "HIGH",
        "notes": "Pathological Q waves or ST changes in inferior leads (II, III, aVF)."
    },
    "ASMI": {
        "concept": "Anteroseptal Myocardial Infarction",
        "disease": "Myocardial Infarction (Anteroseptal)",
        "confidence": "HIGH",
        "notes": "Pathological Q waves or ST changes in leads V1-V4."
    },
    "AMI": {
        "concept": "Anterior Myocardial Infarction",
        "disease": "Myocardial Infarction (Anterior)",
        "confidence": "HIGH",
        "notes": "Infarction localized to anterior precordial leads (V3, V4)."
    },
    "ALMI": {
        "concept": "Anterolateral Myocardial Infarction",
        "disease": "Myocardial Infarction (Anterolateral)",
        "confidence": "HIGH",
        "notes": "Infarction involving anterior (V3-V4) and lateral leads (I, aVL, V5-V6)."
    },
    "ILMI": {
        "concept": "Inferolateral Myocardial Infarction",
        "disease": "Myocardial Infarction (Inferolateral)",
        "confidence": "HIGH",
        "notes": "Infarction involving inferior (II, III, aVF) and lateral leads (V5-V6)."
    },
    "LMI": {
        "concept": "Lateral Myocardial Infarction",
        "disease": "Myocardial Infarction (Lateral)",
        "confidence": "HIGH",
        "notes": "Infarction isolated to lateral leads (I, aVL, V5, V6)."
    },
    # Hypertrophy
    "LVH": {
        "concept": "Left Ventricular Hypertrophy",
        "disease": "Left Ventricular Hypertrophy",
        "confidence": "HIGH",
        "notes": "Voltage criteria (Sokolow-Lyon or Cornell) met for LV mass enlargement."
    },
    "RVH": {
        "concept": "Right Ventricular Hypertrophy",
        "disease": "Right Ventricular Hypertrophy",
        "confidence": "HIGH",
        "notes": "Tall R wave in V1, right axis deviation."
    }
}


# Chapman-Shaoxing SNOMED-CT Code Mappings
CHAPMAN_HARMONIZATION_RULES = {
    # Normal
    "426783006": {
        "concept": "Normal Sinus Rhythm",
        "disease": "Normal ECG",
        "confidence": "HIGH",
        "notes": "SNOMED CT 426783006: Sinus rhythm."
    },
    # Atrial Fibrillation / Flutter
    "164889003": {
        "concept": "Atrial Fibrillation",
        "disease": "Atrial Fibrillation",
        "confidence": "HIGH",
        "notes": "SNOMED CT 164889003: Atrial fibrillation."
    },
    "164890007": {
        "concept": "Atrial Flutter",
        "disease": "Atrial Fibrillation / Flutter",
        "confidence": "MEDIUM",
        "notes": "SNOMED CT 164890007: Atrial flutter."
    },
    # Bundle Branch Blocks
    "59118001": {
        "concept": "Right Bundle Branch Block",
        "disease": "Right Bundle Branch Block",
        "confidence": "HIGH",
        "notes": "SNOMED CT 59118001: Right bundle branch block."
    },
    "164909002": {
        "concept": "Left Bundle Branch Block",
        "disease": "Left Bundle Branch Block",
        "confidence": "HIGH",
        "notes": "SNOMED CT 164909002: Left bundle branch block."
    },
    # Sinus Rates
    "426177001": {
        "concept": "Sinus Bradycardia",
        "disease": "Sinus Bradycardia",
        "confidence": "HIGH",
        "notes": "SNOMED CT 426177001: Sinus bradycardia."
    },
    "427084000": {
        "concept": "Sinus Tachycardia",
        "disease": "Sinus Tachycardia",
        "confidence": "HIGH",
        "notes": "SNOMED CT 427084000: Sinus tachycardia."
    },
    # AV Blocks
    "270492004": {
        "concept": "First Degree Atrioventricular Block",
        "disease": "1st Degree AV Block",
        "confidence": "HIGH",
        "notes": "SNOMED CT 270492004: First degree atrioventricular block."
    },
    "195042002": {
        "concept": "Second Degree Atrioventricular Block",
        "disease": "2nd Degree AV Block",
        "confidence": "HIGH",
        "notes": "SNOMED CT 195042002: Second degree atrioventricular block."
    },
    "27885002": {
        "concept": "Third Degree Atrioventricular Block",
        "disease": "3rd Degree AV Block",
        "confidence": "HIGH",
        "notes": "SNOMED CT 27885002: Complete heart block."
    },
    # Infarction & Hypertrophy
    "164865005": {
        "concept": "Myocardial Infarction",
        "disease": "Myocardial Infarction",
        "confidence": "HIGH",
        "notes": "SNOMED CT 164865005: Myocardial infarction."
    },
    "164873001": {
        "concept": "Left Ventricular Hypertrophy",
        "disease": "Left Ventricular Hypertrophy",
        "confidence": "HIGH",
        "notes": "SNOMED CT 164873001: Left ventricular hypertrophy."
    },
    "89792004": {
        "concept": "Right Ventricular Hypertrophy",
        "disease": "Right Ventricular Hypertrophy",
        "confidence": "HIGH",
        "notes": "SNOMED CT 89792004: Right ventricular hypertrophy."
    }
}


# MIMIC-IV-ECG Machine Report Regex Patterns
MIMIC_REPORT_PATTERNS = [
    # Normal
    (r"^(normal ecg|normal sinus rhythm|sinus rhythm)$", "Normal Electrocardiogram", "Normal ECG", "HIGH", "Definitive normal machine report interpretation."),
    (r"sinus rhythm", "Sinus Rhythm", "Normal ECG", "HIGH", "Explicit sinus rhythm statement."),
    
    # Atrial Fibrillation
    (r"atrial fibrillation", "Atrial Fibrillation", "Atrial Fibrillation", "HIGH", "Explicit atrial fibrillation statement."),
    (r"atrial flutter", "Atrial Flutter", "Atrial Fibrillation / Flutter", "MEDIUM", "Explicit atrial flutter statement."),
    
    # Bundle Branch Block
    (r"(right bundle[- ]branch block|rbbb)", "Right Bundle Branch Block", "Right Bundle Branch Block", "HIGH", "Explicit right bundle branch block."),
    (r"(left bundle[- ]branch block|lbbb)", "Left Bundle Branch Block", "Left Bundle Branch Block", "HIGH", "Explicit left bundle branch block."),
    (r"incomplete right bundle", "Incomplete Right Bundle Branch Block", "Right Bundle Branch Block (Incomplete)", "MEDIUM", "Incomplete conduction delay."),
    (r"incomplete left bundle", "Incomplete Left Bundle Branch Block", "Left Bundle Branch Block (Incomplete)", "MEDIUM", "Incomplete conduction delay."),
    
    # Sinus Rates
    (r"sinus bradycardia", "Sinus Bradycardia", "Sinus Bradycardia", "HIGH", "Explicit sinus bradycardia statement."),
    (r"sinus tachycardia", "Sinus Tachycardia", "Sinus Tachycardia", "HIGH", "Explicit sinus tachycardia statement."),
    
    # AV Blocks
    (r"(first degree|1st degree) (a-v|av) block", "First Degree Atrioventricular Block", "1st Degree AV Block", "HIGH", "PR prolongation statement."),
    (r"prolonged pr interval", "First Degree Atrioventricular Block", "1st Degree AV Block", "HIGH", "PR prolongation statement."),
    (r"(second degree|2nd degree) (a-v|av) block", "Second Degree Atrioventricular Block", "2nd Degree AV Block", "HIGH", "Mobitz block statement."),
    (r"(third degree|3rd degree|complete) (a-v|av) block", "Third Degree Atrioventricular Block", "3rd Degree AV Block", "HIGH", "Complete AV dissociation."),
    
    # Infarctions
    (r"inferior (infarct|infarction|injury|q wave)", "Inferior Myocardial Infarction", "Myocardial Infarction (Inferior)", "HIGH", "Inferior wall infarction."),
    (r"anterior (infarct|infarction|injury|q wave)", "Anterior Myocardial Infarction", "Myocardial Infarction (Anterior)", "HIGH", "Anterior wall infarction."),
    (r"anteroseptal (infarct|infarction|injury|q wave)", "Anteroseptal Myocardial Infarction", "Myocardial Infarction (Anteroseptal)", "HIGH", "Anteroseptal wall infarction."),
    (r"lateral (infarct|infarction|injury|q wave)", "Lateral Myocardial Infarction", "Myocardial Infarction (Lateral)", "HIGH", "Lateral wall infarction."),
    (r"myocardial infarction", "Myocardial Infarction", "Myocardial Infarction", "MEDIUM", "Non-localized myocardial infarction statement."),
    
    # Hypertrophy
    (r"left ventricular hypertrophy", "Left Ventricular Hypertrophy", "Left Ventricular Hypertrophy", "HIGH", "Voltage/morphology criteria for LVH."),
    (r"right ventricular hypertrophy", "Right Ventricular Hypertrophy", "Right Ventricular Hypertrophy", "HIGH", "Voltage/morphology criteria for RVH.")
]


def harmonize_ptbxl_label(raw_code: str, raw_label: str) -> Dict[str, str]:
    """Maps a PTB-XL SCP code to normalized concept and common disease."""
    clean_code = str(raw_code).strip()
    if clean_code in PTBXL_HARMONIZATION_RULES:
        rule = PTBXL_HARMONIZATION_RULES[clean_code]
        return {
            "normalized_concept": rule["concept"],
            "common_disease": rule["disease"],
            "mapping_confidence": rule["confidence"],
            "mapping_notes": rule["notes"]
        }
    
    # Unmatched / Non-target
    return {
        "normalized_concept": str(raw_label).strip() if raw_label else clean_code,
        "common_disease": "Other / Non-Target",
        "mapping_confidence": "LOW",
        "mapping_notes": f"SCP statement '{clean_code}' is outside primary candidate target set."
    }


def harmonize_chapman_label(raw_code: str, raw_label: str) -> Dict[str, str]:
    """Maps a Chapman SNOMED code to normalized concept and common disease."""
    clean_code = str(raw_code).strip()
    if clean_code in CHAPMAN_HARMONIZATION_RULES:
        rule = CHAPMAN_HARMONIZATION_RULES[clean_code]
        return {
            "normalized_concept": rule["concept"],
            "common_disease": rule["disease"],
            "mapping_confidence": rule["confidence"],
            "mapping_notes": rule["notes"]
        }
    
    return {
        "normalized_concept": str(raw_label).strip() if raw_label else f"SNOMED_{clean_code}",
        "common_disease": "Other / Non-Target",
        "mapping_confidence": "LOW",
        "mapping_notes": f"SNOMED CT '{clean_code}' is outside primary candidate target set."
    }


def harmonize_mimic_label(raw_label: str) -> Dict[str, str]:
    """Maps a MIMIC machine report phrase to normalized concept and common disease."""
    text = str(raw_label).strip().lower()
    for pattern, concept, disease, conf, notes in MIMIC_REPORT_PATTERNS:
        if re.search(pattern, text):
            return {
                "normalized_concept": concept,
                "common_disease": disease,
                "mapping_confidence": conf,
                "mapping_notes": notes
            }
            
    return {
        "normalized_concept": str(raw_label).strip(),
        "common_disease": "Other / Non-Target",
        "mapping_confidence": "LOW",
        "mapping_notes": "Machine interpretation phrase outside primary candidate target set."
    }


def harmonize_labels_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the 3-tier harmonization pipeline to a DataFrame of raw labels.
    Preserves all original columns and appends harmonization fields.
    Uses unique key caching and merge for vector speed across millions of rows.
    """
    if raw_df.empty:
        return raw_df.copy()

    # Find unique (dataset, raw_label_code, raw_label) triplets
    cols_to_group = ["dataset", "raw_label_code", "raw_label"]
    available_cols = [c for c in cols_to_group if c in raw_df.columns]
    unique_pairs = raw_df[available_cols].drop_duplicates().copy()

    mapping_results = []
    for _, row in unique_pairs.iterrows():
        ds = row.get("dataset", "")
        r_code = row.get("raw_label_code", "")
        r_label = row.get("raw_label", "")

        if ds == "PTB-XL":
            h = harmonize_ptbxl_label(r_code, r_label)
        elif ds == "Chapman-Shaoxing":
            h = harmonize_chapman_label(r_code, r_label)
        elif ds == "MIMIC-IV-ECG":
            h = harmonize_mimic_label(r_label)
        else:
            h = {
                "normalized_concept": str(r_label),
                "common_disease": "Unknown",
                "mapping_confidence": "UNCERTAIN",
                "mapping_notes": f"Unknown dataset {ds}"
            }
        mapping_results.append(h)

    mapping_df = pd.DataFrame(mapping_results)
    for col in mapping_df.columns:
        unique_pairs[col] = mapping_df[col].values

    # Merge back to raw_df
    harmonized_df = pd.merge(raw_df, unique_pairs, on=available_cols, how="left")
    return harmonized_df


def generate_harmonization_mapping_table() -> pd.DataFrame:
    """
    Generates a unified master mapping catalog across all three datasets:
    [dataset, raw_label, normalized_concept, common_disease, mapping_confidence, mapping_notes]
    """
    rows = []
    # PTB-XL
    for code, info in PTBXL_HARMONIZATION_RULES.items():
        rows.append({
            "dataset": "PTB-XL",
            "raw_label": code,
            "normalized_concept": info["concept"],
            "common_disease": info["disease"],
            "mapping_confidence": info["confidence"],
            "mapping_notes": info["notes"]
        })
    # Chapman
    for code, info in CHAPMAN_HARMONIZATION_RULES.items():
        rows.append({
            "dataset": "Chapman-Shaoxing",
            "raw_label": f"SNOMED_{code}",
            "normalized_concept": info["concept"],
            "common_disease": info["disease"],
            "mapping_confidence": info["confidence"],
            "mapping_notes": info["notes"]
        })
    # MIMIC
    for pat, concept, disease, conf, notes in MIMIC_REPORT_PATTERNS:
        rows.append({
            "dataset": "MIMIC-IV-ECG",
            "raw_label": pat,
            "normalized_concept": concept,
            "common_disease": disease,
            "mapping_confidence": conf,
            "mapping_notes": notes
        })
        
    return pd.DataFrame(rows)
